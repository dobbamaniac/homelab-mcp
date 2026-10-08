from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Annotated, Any

import httpx
from pydantic import Field

from homelab_mcp.http import get_json
from homelab_mcp.redact import redact_value
from homelab_mcp.utils import pick, pick_all, require_env


# ---------------------------------------------------------------------------
# Tool parameter types
# ---------------------------------------------------------------------------

DockerHostRef = Annotated[
    str,
    Field(description="Docker host name returned by list_docker_hosts."),
]

DockerContainerRef = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description=(
            "Docker container name or ID returned by list_docker_containers for the selected "
            "Docker host."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Internal API helpers
# ---------------------------------------------------------------------------

_clients: dict[str, httpx.AsyncClient] = {}
_api_versions: dict[str, str] = {}
_CONTAINER_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _configured_hosts() -> dict[str, str]:
    value = require_env("DOCKER_HOSTS")

    hosts: dict[str, str] = {}

    for entry in value.split(","):
        name, separator, url = entry.partition("=")
        name = name.strip()
        url = url.strip()

        if not separator or not name or not url:
            raise RuntimeError("DOCKER_HOSTS must use name=url entries separated by commas")

        if name in hosts:
            raise RuntimeError(f"Duplicate Docker host {name!r}")

        if not url.startswith(("http://", "https://")):
            raise RuntimeError(f"Docker host {name!r} must use HTTP or HTTPS")

        hosts[name] = url

    return hosts


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Open shared Docker Engine API clients for the server lifetime."""
    if _clients:
        raise RuntimeError("Docker API clients are already initialized")

    async with AsyncExitStack() as stack:
        for name, url in _configured_hosts().items():
            _clients[name] = await stack.enter_async_context(
                httpx.AsyncClient(
                    base_url=f"{url.rstrip('/')}/",
                    timeout=10.0,
                )
            )

        try:
            yield

        finally:
            _api_versions.clear()
            _clients.clear()


def _client(host: str) -> httpx.AsyncClient:
    client = _clients.get(host)

    if client is None:
        raise RuntimeError(f"Docker host {host!r} is not configured")

    return client


async def _api_version(host: str) -> str:
    version = _api_versions.get(host)

    if version is not None:
        return version

    data = await get_json(
        _client(host),
        "version",
        source=f"Docker host {host!r}",
    )

    version = str(data.get("ApiVersion", ""))

    if not version:
        raise RuntimeError(f"Docker host {host!r} did not report an API version")

    _api_versions[host] = version

    return version


async def _get(
    host: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    versioned: bool = True,
) -> Any:
    request_path = path.lstrip("/")

    if versioned:
        request_path = f"v{await _api_version(host)}/{request_path}"

    data = await get_json(
        _client(host),
        request_path,
        params=params,
        source=f"Docker host {host!r}",
    )

    return redact_value(data)


def _container_ref(value: str) -> str:
    if not _CONTAINER_REF.fullmatch(value):
        raise RuntimeError(f"Invalid Docker container name or ID {value!r}")

    return value


# ---------------------------------------------------------------------------
# Host and Engine inventory
# ---------------------------------------------------------------------------

async def list_docker_hosts() -> dict[str, Any]:
    """
    List Docker host names configured on the MCP server.

    Source: server-side DOCKER_HOSTS configuration.

    No Docker Engine API request is made.
    """
    return {"hosts": sorted(_clients)}


async def get_docker_version(
    host: DockerHostRef,
) -> dict[str, Any]:
    """
    Get Docker Engine version information for one host.

    Upstream: Docker Engine API
    GET /version
    """
    data = await _get(
        host,
        "version",
        versioned=False,
    )

    api_version = str(data.get("ApiVersion", ""))

    if api_version:
        _api_versions[host] = api_version

    return pick(
        data,
        "Version", "ApiVersion", "MinAPIVersion", "GitCommit", "GoVersion", "Os", "Arch", 
        "KernelVersion", "BuildTime", "Experimental", "Platform"
    )


async def get_docker_info(
    host: DockerHostRef,
) -> dict[str, Any]:
    """
    Get Docker daemon and host inventory information.

    Upstream: Docker Engine API
    GET /info

    High-frequency resource-utilization metrics are not exposed.
    """
    data = await _get(host, "info")

    return pick(
        data,
        "Name", "ServerVersion", "OperatingSystem", "OSType", "Architecture", "KernelVersion", 
        "NCPU", "MemTotal", "Containers", "ContainersRunning", "ContainersPaused", 
        "ContainersStopped", "Images", "Driver", "CgroupDriver", "CgroupVersion", "LoggingDriver", 
        "DockerRootDir", "LiveRestoreEnabled"
    )


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

async def list_docker_containers(
    host: DockerHostRef,
) -> dict[str, Any]:
    """
    List Docker containers on one host, including stopped containers.

    Upstream: Docker Engine API
    GET /containers/json?all=1

    Environment variables and raw labels are omitted. Docker Compose project and service labels are
    retained for topology/discovery.
    """
    data = await _get(
        host,
        "containers/json",
        params={"all": "1"},
    )

    containers: list[dict[str, Any]] = []

    for item in data:
        labels = item.get("Labels") or {}
        host_config = item.get("HostConfig") or {}
        networks = ((item.get("NetworkSettings") or {}).get("Networks") or {})

        containers.append({
            **pick(
                item,
                "Id", "Image", "Created", "State", "Status"
            ),
            "Names": [
                str(name).removeprefix("/")
                for name in (item.get("Names") or [])
            ],
            "Ports": pick_all(
                item.get("Ports") or [],
                "IP", "PrivatePort", "PublicPort", "Type"
            ),
            "Labels": pick(
                labels,
                "com.docker.compose.project",
                "com.docker.compose.service"
            ),
            "HostConfig": pick(
                host_config,
                "NetworkMode"
            ),
            "NetworkSettings": {
                "Networks": sorted(str(name) for name in networks)
            },
        })

    return {"containers": containers}


async def get_docker_container(
    host: DockerHostRef,
    container: DockerContainerRef,
) -> dict[str, Any]:
    """
    Inspect one Docker container.

    Upstream: Docker Engine API
    GET /containers/{container}/json

    Environment variables, raw labels, healthcheck command/output, and low-value generated 
    identifiers are omitted. Docker Compose identity, lifecycle state, resource limits, security 
    settings, mounts, devices, and network configuration are retained.
    """
    data = await _get(
        host,
        f"containers/{_container_ref(container)}/json",
    )

    config = data.get("Config") or {}
    labels = config.get("Labels") or {}
    healthcheck = config.get("Healthcheck") or {}
    state = data.get("State") or {}
    health = state.get("Health") or {}
    host_config = data.get("HostConfig") or {}
    network_settings = data.get("NetworkSettings") or {}

    mounts: list[dict[str, Any]] = []

    for mount in data.get("Mounts") or []:
        filtered = pick(
            mount,
            "Type", "Name", "Destination", "Driver", "Mode", "RW", "Propagation"
        )

        if mount.get("Type") == "bind" and "Source" in mount:
            filtered["Source"] = mount["Source"]

        mounts.append(filtered)

    networks = {
        name: pick(
            network,
            "IPAddress", "IPPrefixLen", "Gateway", "GlobalIPv6Address", "GlobalIPv6PrefixLen", 
            "IPv6Gateway", "MacAddress", "Aliases"
        )
        for name, network in sorted(
            (network_settings.get("Networks") or {}).items()
        )
    }

    result = {
        **pick(
            data,
            "Name", "Created", "RestartCount", "Platform", "AppArmorProfile"
        ),
        "Config": {
            **pick(
                config,
                "Image", "Hostname", "Domainname", "User", "WorkingDir", "StopSignal", 
                "ExposedPorts"
            ),
            "Labels": pick(
                labels,
                "com.docker.compose.project",
                "com.docker.compose.service"
            ),
        },
        "State": {
            **pick(
                state,
                "Status", "Running", "Paused", "Restarting", "OOMKilled", "Dead", "Pid", 
                "ExitCode", "Error", "StartedAt", "FinishedAt"
            ),
            "Health": (
                pick(
                    health,
                    "Status", "FailingStreak"
                )
                if health
                else None
            ),
        },
        "HostConfig": {
            **pick(
                host_config,
                "Privileged", "ReadonlyRootfs", "NetworkMode", "Memory", "MemorySwap", "NanoCpus", 
                "CpuShares", "CpuPeriod", "CpuQuota", "CpusetCpus", "PidsLimit", "ShmSize", 
                "CapAdd", "CapDrop", "SecurityOpt", "PidMode", "IpcMode", "UTSMode", "UsernsMode"
            ),
            "RestartPolicy": pick(
                host_config.get("RestartPolicy") or {},
                "Name", "MaximumRetryCount"
            ),
            "Devices": pick_all(
                host_config.get("Devices") or [],
                "PathOnHost", "PathInContainer", "CgroupPermissions"
            ),
            "DeviceRequests": pick_all(
                host_config.get("DeviceRequests") or [],
                "Driver", "Count", "DeviceIDs", "Capabilities"
            ),
        },
        "Mounts": mounts,
        "NetworkSettings": {
            "Ports": network_settings.get("Ports") or {},
            "Networks": networks,
        },
    }

    if healthcheck:
        result["Config"]["Healthcheck"] = pick(
            healthcheck,
            "Interval", "Timeout", "Retries", "StartPeriod", "StartInterval"
        )

    return result


TOOLS = (
    list_docker_hosts,
    get_docker_version,
    get_docker_info,
    list_docker_containers,
    get_docker_container,
)
