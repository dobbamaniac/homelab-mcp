from __future__ import annotations

import re
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import httpx
from pydantic import Field

from homelab_mcp.http import (
    get as http_get,
    json_data,
)
from homelab_mcp.redact import redact_mapping, redact_text
from homelab_mcp.utils import pick, pick_all, require_env


# ------------------------------------------------------------------------------------------------
# Structured output
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DockerHost:
    name: str
    engine_version: str
    api_version: str
    operating_system: str
    kernel_version: str
    architecture: str
    cpu_count: int
    memory_total_bytes: int
    containers: int
    containers_running: int
    containers_paused: int
    containers_stopped: int
    images: int


@dataclass(frozen=True)
class DockerHosts:
    hosts: list[DockerHost]


@dataclass(frozen=True)
class DockerContainer:
    host: str
    name: str
    image: str
    state: str
    status: str
    compose_project: str | None
    compose_service: str | None
    network_mode: str
    network_peer: str | None
    ports: list[dict[str, Any]]
    networks: list[str]


@dataclass(frozen=True)
class DockerContainers:
    containers: list[DockerContainer]


@dataclass(frozen=True)
class DockerContainerConfig:
    host: str
    name: str
    image: str
    created: str
    state: dict[str, Any]
    health: dict[str, Any] | None
    compose_project: str | None
    compose_service: str | None
    user: str
    privileged: bool
    read_only_rootfs: bool
    restart_policy: dict[str, Any]
    network_mode: str
    network_peer: str | None
    mounts: list[dict[str, Any]]
    networks: dict[str, dict[str, Any]]
    resources: dict[str, Any]
    security: dict[str, Any]
    devices: list[dict[str, Any]]
    device_requests: list[dict[str, Any]]


@dataclass(frozen=True)
class DockerComposeProject:
    host: str
    name: str
    services: list[str]
    containers: list[str]
    states: dict[str, int]


@dataclass(frozen=True)
class DockerComposeProjects:
    projects: list[DockerComposeProject]


@dataclass(frozen=True)
class DockerHealthLogEntry:
    start: str
    end: str
    exit_code: int
    output: str


@dataclass(frozen=True)
class DockerContainerHealth:
    host: str
    name: str
    status: str | None
    failing_streak: int
    healthcheck: dict[str, Any] | None
    history: list[DockerHealthLogEntry]


# -------------------------------------------------------------------------------------------------
# Internal API helpers
# -------------------------------------------------------------------------------------------------


_clients: dict[str, httpx.AsyncClient] = {}
_versions: dict[str, dict[str, Any]] = {}
_CONTAINER_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


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
    """Open shared Docker Engine API clients for the server lifetime"""
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
            _versions.clear()
            _clients.clear()


async def _version(host: str) -> dict[str, Any]:
    if host not in _versions:
        _versions[host] = await _get(host, "version", versioned=False)

    return _versions[host]


async def _get_response(
    host: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    versioned: bool = True,
) -> httpx.Response:
    client = _clients.get(host)

    if client is None:
        raise RuntimeError(f"Docker host {host!r} is not configured")

    if versioned:
        version = await _version(host)
        path = f"v{version['ApiVersion']}/{path.lstrip('/')}"

    return await http_get(
        client,
        path,
        params=params,
        source=f"Docker host {host!r}",
    )


async def _get(
    host: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    versioned: bool = True,
) -> Any:
    response = await _get_response(
        host,
        path,
        params=params,
        versioned=versioned,
    )

    return json_data(
        response,
        source=f"Docker host {host!r}",
        path=path,
    )


def _container_ref(value: str) -> str:
    if not _CONTAINER_REF.fullmatch(value):
        raise RuntimeError(f"Invalid Docker container name or ID {value!r}")
    
    return value


async def _inspect_container(
    host: str,
    container: str,
) -> tuple[str, dict[str, Any]]:
    container = _container_ref(container)

    return container, await _get(
        host,
        f"containers/{container}/json",
    )


def _memory_used(stats: dict[str, Any]) -> int:
    usage = int(stats.get("usage", 0))
    values = stats.get("stats") or {}

    for key in ("total_inactive_file", "inactive_file"):
        cache = int(values.get(key, 0))

        if 0 < cache < usage:
            return usage - cache

    return usage


def _cpu_usage_ratio(stats: dict[str, Any]) -> float:
    current = stats.get("cpu_stats") or {}
    previous = stats.get("precpu_stats") or {}
    current_usage = current.get("cpu_usage") or {}
    previous_usage = previous.get("cpu_usage") or {}

    cpu_delta = (
        int(current_usage.get("total_usage", 0))
        - int(previous_usage.get("total_usage", 0))
    )

    system_delta = (
        int(current.get("system_cpu_usage", 0))
        - int(previous.get("system_cpu_usage", 0))
    )

    cpu_count = (
        int(current.get("online_cpus", 0))
        or len(current_usage.get("percpu_usage") or [])
        or 1
    )

    if cpu_delta <= 0 or system_delta <= 0:
        return 0.0

    return cpu_delta / system_delta * cpu_count


# -------------------------------------------------------------------------------------------------
# MCP tools
# -------------------------------------------------------------------------------------------------


async def list_docker_hosts() -> DockerHosts:
    """List configured Docker hosts and compact Engine/inventory information."""
    hosts: list[DockerHost] = []

    for name in sorted(_clients):
        version = await _version(name)
        info = await _get(name, "info")

        hosts.append(
            DockerHost(
                name=name,
                engine_version=str(version.get("Version", "")),
                api_version=str(version.get("ApiVersion", "")),
                operating_system=str(info.get("OperatingSystem", "")),
                kernel_version=str(info.get("KernelVersion", "")),
                architecture=str(info.get("Architecture", "")),
                cpu_count=int(info.get("NCPU", 0)),
                memory_total_bytes=int(info.get("MemTotal", 0)),
                containers=int(info.get("Containers", 0)),
                containers_running=int(info.get("ContainersRunning", 0)),
                containers_paused=int(info.get("ContainersPaused", 0)),
                containers_stopped=int(info.get("ContainersStopped", 0)),
                images=int(info.get("Images", 0)),
            )
        )

    return DockerHosts(hosts=hosts)


async def list_docker_containers(
    host: str | None = None,
) -> DockerContainers:
    """
    List Docker containers across all configured hosts or one named host.

    Includes stopped containers and compact Compose/network/port metadata.
    """
    host_names = [host] if host is not None else sorted(_clients)
    containers: list[DockerContainer] = []

    for host_name in host_names:
        data = await _get(
            host_name,
            "containers/json",
            params={"all": "1"},
        )

        names_by_id = {
            str(item.get("Id", "")): (str((item.get("Names") or [""])[0]).removeprefix("/"))
            for item in data
        }

        for item in data:
            labels = item.get("Labels") or {}
            names = item.get("Names") or []
            networks = ((item.get("NetworkSettings") or {}).get("Networks") or {})
            host_config = item.get("HostConfig") or {}
            network_mode = str(host_config.get("NetworkMode", ""))
            network_peer = None

            if network_mode.startswith("container:"):
                peer = network_mode.removeprefix("container:")
                network_mode = "container"
                network_peer = names_by_id.get(peer)

            containers.append(
                DockerContainer(
                    host=host_name,
                    name=str(names[0]).removeprefix("/") if names else "",
                    image=str(item.get("Image", "")),
                    state=str(item.get("State", "")),
                    status=str(item.get("Status", "")),
                    compose_project=labels.get("com.docker.compose.project"),
                    compose_service=labels.get("com.docker.compose.service"),
                    network_mode=network_mode,
                    network_peer=network_peer,
                    ports=sorted(
                        pick_all(
                            item.get("Ports") or [],
                            "IP", "PrivatePort", "PublicPort", "Type",
                        ),
                        key=lambda port: (
                            port.get("PrivatePort", 0),
                            port.get("PublicPort", 0),
                            port.get("Type", ""),
                            port.get("IP", ""),
                        )
                    ),
                    networks=sorted(str(name) for name in networks),
                )
            )

    containers.sort(key=lambda container: (container.host, container.name))

    return DockerContainers(containers=containers)


async def get_docker_container(
    host: str,
    container: str,
) -> DockerContainerConfig:
    """
    Get compact configuration, security, storage, and networking details for a Docker container.

    Environment variables, raw labels, logs, IDs, and healthcheck output are omitted.
    """
    container, data = await _inspect_container(host, container)

    config = data.get("Config") or {}
    host_config = data.get("HostConfig") or {}
    state = data.get("State") or {}
    labels = config.get("Labels") or {}

    network_mode = str(host_config.get("NetworkMode", ""))
    network_peer = None

    if network_mode.startswith("container:"):
        peer = network_mode.removeprefix("container:")
        peer_data = await _get(host, f"containers/{_container_ref(peer)}/json")
        network_mode = "container"
        network_peer = str(peer_data.get("Name", "")).removeprefix("/") or None

    health_data = state.get("Health")
    health = (
        pick(health_data, "Status", "FailingStreak")
        if health_data
        else None
    )

    mounts: list[dict[str, Any]] = []

    for mount in data.get("Mounts") or []:
        item = pick(mount, "Type", "Name", "Destination", "Driver", "Mode", "RW", "Propagation")

        if mount.get("Type") == "bind" and "Source" in mount:
            item["Source"] = mount["Source"]

        mounts.append(item)

    mounts.sort(key=lambda mount: str(mount.get("Destination", "")))

    networks = {
        name: pick(
            network, "IPAddress", "IPPrefixLen", "Gateway", "GlobalIPv6Address", 
            "GlobalIPv6PrefixLen", "IPv6Gateway", "MacAddress", "Aliases"
        )
        for name, network in sorted(
            ((data.get("NetworkSettings") or {}).get("Networks") or {}).items()
        )
    }

    security = pick(
        host_config, "CapAdd", "CapDrop", "SecurityOpt", "PidMode", "IpcMode", "UTSMode", 
        "UsernsMode"
    )

    if data.get("AppArmorProfile"):
        security["AppArmorProfile"] = data["AppArmorProfile"]

    return DockerContainerConfig(
        host=host,
        name=str(data.get("Name", container)).removeprefix("/"),
        image=str(config.get("Image", "")),
        created=str(data.get("Created", "")),
        state=redact_mapping(pick(
            state, "Status", "Running", "Paused", "Restarting", "OOMKilled", "Dead", "ExitCode", 
            "Error", "StartedAt", "FinishedAt"
        )),
        health=health,
        compose_project=labels.get("com.docker.compose.project"),
        compose_service=labels.get("com.docker.compose.service"),
        user=str(config.get("User", "")),
        privileged=bool(host_config.get("Privileged", False)),
        read_only_rootfs=bool(host_config.get("ReadonlyRootfs", False)),
        restart_policy=pick(
            host_config.get("RestartPolicy") or {},
            "Name",
            "MaximumRetryCount",
        ),
        network_mode=network_mode,
        network_peer=network_peer,
        mounts=mounts,
        networks=networks,
        resources=pick(
            host_config, "Memory", "MemorySwap", "NanoCpus", "CpuShares", "CpuPeriod", "CpuQuota", 
            "CpusetCpus", "PidsLimit", "ShmSize"
        ),
        security=security,
        devices=pick_all(
            host_config.get("Devices") or [],
            "PathOnHost", "PathInContainer", "CgroupPermissions",
        ),
        device_requests=pick_all(
            host_config.get("DeviceRequests") or [],
            "Driver", "Count", "DeviceIDs", "Capabilities",
        ),
    )


async def list_docker_compose_projects(host: str | None = None) -> DockerComposeProjects:
    """List Compose projects derived from Docker's canonical Compose container labels."""
    inventory = await list_docker_containers(host)
    grouped: dict[tuple[str, str], list[DockerContainer]] = {}

    for container in inventory.containers:
        if container.compose_project:
            grouped.setdefault(
                (container.host, container.compose_project),
                [],
            ).append(container)

    projects: list[DockerComposeProject] = []

    for (host_name, project_name), containers in sorted(grouped.items()):
        projects.append(
            DockerComposeProject(
                host=host_name,
                name=project_name,
                services=sorted(
                    {
                        container.compose_service
                        for container in containers
                        if container.compose_service
                    }
                ),
                containers=sorted(container.name for container in containers),
                states=dict(sorted(Counter(
                    container.state
                    for container in containers
                ).items())),
            )
        )

    return DockerComposeProjects(projects=projects)


async def get_docker_container_health(
    host: str,
    container: str,
    history_limit: Annotated[int, Field(ge=1, le=20)] = 10,
) -> DockerContainerHealth:
    """
    Get Docker healthcheck configuration, status, and recent results.

    Includes healthcheck command output that is intentionally omitted from
    get_docker_container().
    """
    container, data = await _inspect_container(host, container)

    config = data.get("Config") or {}
    state = data.get("State") or {}
    health = state.get("Health") or {}
    healthcheck = config.get("Healthcheck")

    history: list[DockerHealthLogEntry] = []

    for entry in (health.get("Log") or [])[-history_limit:]:
        output = redact_text(str(entry.get("Output", "")))

        if len(output) > 8192:
            output = output[:8192] + "...[truncated]"

        history.append(
            DockerHealthLogEntry(
                start=str(entry.get("Start", "")),
                end=str(entry.get("End", "")),
                exit_code=int(entry.get("ExitCode", 0)),
                output=output,
            )
        )

    return DockerContainerHealth(
        host=host,
        name=str(data.get("Name", container)).removeprefix("/"),
        status=(
            str(health["Status"])
            if health.get("Status") is not None
            else None
        ),
        failing_streak=int(health.get("FailingStreak", 0)),
        healthcheck=(
            redact_mapping(pick(
                healthcheck, "Test", "Interval", "Timeout", "Retries", "StartPeriod", 
                "StartInterval"
            ))
            if healthcheck
            else None
        ),
        history=history,
    )


TOOLS = (
    list_docker_hosts,
    list_docker_containers,
    get_docker_container,
    list_docker_compose_projects,
    get_docker_container_health,
)
