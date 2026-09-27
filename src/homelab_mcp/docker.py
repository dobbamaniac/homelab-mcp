from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from homelab_mcp.http import (
    RequestError as DockerRequestError,
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
class DockerContainerStats:
    host: str
    name: str
    cpu_usage_ratio: float
    memory_used_bytes: int
    memory_limit_bytes: int | None
    memory_usage_ratio: float | None
    network_rx_bytes: int
    network_tx_bytes: int
    block_read_bytes: int
    block_write_bytes: int
    pids: int


@dataclass(frozen=True)
class DockerLogChunk:
    stream: str
    output: str


@dataclass(frozen=True)
class DockerContainerLogs:
    host: str
    name: str
    tty: bool
    chunks: list[DockerLogChunk]


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
_MAX_LOG_BYTES = 512 * 1024


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


def _docker_log_chunks(
    data: bytes,
    *,
    tty: bool,
) -> list[DockerLogChunk]:
    if len(data) > _MAX_LOG_BYTES:
        raise DockerRequestError("Docker container log response exceeded maximum size")

    if tty:
        return [
            DockerLogChunk(
                stream="combined",
                output=redact_text(data.decode(errors="replace")),
            )
        ] if data else []

    chunks: list[DockerLogChunk] = []
    offset = 0

    stream_names = {
        0: "stdin",
        1: "stdout",
        2: "stderr",
    }

    while offset < len(data):
        if len(data) - offset < 8:
            raise DockerRequestError(
                "Docker returned a truncated log stream header"
            )

        stream_type = data[offset]
        size = int.from_bytes(
            data[offset + 4:offset + 8],
            byteorder="big",
        )
        offset += 8

        end = offset + size

        if end > len(data):
            raise DockerRequestError("Docker returned a truncated log stream payload")

        payload = data[offset:end]
        offset = end

        chunks.append(
            DockerLogChunk(
                stream=stream_names.get(stream_type, "unknown"),
                output=redact_text(payload.decode(errors="replace")),
            )
        )

    return chunks

# -------------------------------------------------------------------------------------------------
# MCP tools
# -------------------------------------------------------------------------------------------------


async def list_docker_hosts() -> DockerHosts:
    """List configured Docker hosts and their current Engine/resource status."""
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
        item = pick(
            mount,
            "Type",
            "Name",
            "Destination",
            "Driver",
            "Mode",
            "RW",
            "Propagation",
        )

        if mount.get("Type") == "bind" and "Source" in mount:
            item["Source"] = mount["Source"]

        mounts.append(item)

    mounts.sort(key=lambda mount: str(mount.get("Destination", "")))

    networks = {
        name: pick(
            network,
            "IPAddress",
            "IPPrefixLen",
            "Gateway",
            "GlobalIPv6Address",
            "GlobalIPv6PrefixLen",
            "IPv6Gateway",
            "MacAddress",
            "Aliases",
        )
        for name, network in sorted(
            ((data.get("NetworkSettings") or {}).get("Networks") or {}).items()
        )
    }

    security = pick(
        host_config,
        "CapAdd",
        "CapDrop",
        "SecurityOpt",
        "PidMode",
        "IpcMode",
        "UTSMode",
        "UsernsMode",
    )

    if data.get("AppArmorProfile"):
        security["AppArmorProfile"] = data["AppArmorProfile"]

    return DockerContainerConfig(
        host=host,
        name=str(data.get("Name", container)).removeprefix("/"),
        image=str(config.get("Image", "")),
        created=str(data.get("Created", "")),
        state=redact_mapping(pick(
            state,
            "Status",
            "Running",
            "Paused",
            "Restarting",
            "OOMKilled",
            "Dead",
            "ExitCode",
            "Error",
            "StartedAt",
            "FinishedAt",
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
            host_config,
            "Memory",
            "MemorySwap",
            "NanoCpus",
            "CpuShares",
            "CpuPeriod",
            "CpuQuota",
            "CpusetCpus",
            "PidsLimit",
            "ShmSize",
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


async def get_docker_container_stats(
    host: str,
    container: str,
) -> DockerContainerStats:
    """Get one non-streaming runtime resource snapshot for a running Docker container."""
    container, inspect = await _inspect_container(host, container)
    state = inspect.get("State") or {}

    if not state.get("Running"):
        raise RuntimeError(f"Docker container {container!r} on host {host!r} is not running")

    data = await _get(
        host,
        f"containers/{container}/stats",
        params={"stream": "false"},
    )

    memory = data.get("memory_stats") or {}
    memory_used = _memory_used(memory)
    memory_limit = int(memory.get("limit", 0))
    networks = data.get("networks") or {}

    block_read = 0
    block_write = 0

    for item in (
        (data.get("blkio_stats") or {}).get("io_service_bytes_recursive")
        or []
    ):
        operation = str(item.get("op", "")).lower()
        value = int(item.get("value", 0))

        if operation.startswith("r"):
            block_read += value

        elif operation.startswith("w"):
            block_write += value

    return DockerContainerStats(
        host=host,
        name=str(inspect.get("Name", container)).removeprefix("/"),
        cpu_usage_ratio=_cpu_usage_ratio(data),
        memory_used_bytes=memory_used,
        memory_limit_bytes=memory_limit,
        memory_usage_ratio=(
            memory_used / memory_limit
            if memory_limit
            else 0.0
        ),
        network_rx_bytes=sum(
            int(network.get("rx_bytes", 0))
            for network in networks.values()
        ),
        network_tx_bytes=sum(
            int(network.get("tx_bytes", 0))
            for network in networks.values()
        ),
        block_read_bytes=block_read,
        block_write_bytes=block_write,
        pids=int(
            (data.get("pids_stats") or {}).get("current", 0)
        ),
    )


async def get_docker_container_logs(
    host: str,
    container: str,
    tail: int = 200,
    since_seconds: int | None = None,
) -> DockerContainerLogs:
    """
    Get recent stdout/stderr logs for a Docker container.

    `tail` is limited to 1-1000 lines. `since_seconds`, when provided,
    restricts output to approximately that many seconds of recent logs.
    """
    if not 1 <= tail <= 1000:
        raise RuntimeError("tail must be between 1 and 1000")

    if since_seconds is not None and not 1 <= since_seconds <= 604800:
        raise RuntimeError(
            "since_seconds must be between 1 and 604800"
        )

    container, inspect = await _inspect_container(host, container)

    config = inspect.get("Config") or {}
    tty = bool(config.get("Tty", False))

    params = {
        "stdout": "true",
        "stderr": "true",
        "timestamps": "true",
        "tail": str(tail),
    }

    if since_seconds is not None:
        params["since"] = str(max(int(time.time()) - since_seconds, 0))

    response = await _get_response(
        host,
        f"containers/{container}/logs",
        params=params,
    )

    return DockerContainerLogs(
        host=host,
        name=str(inspect.get("Name", container)).removeprefix("/"),
        tty=tty,
        chunks=_docker_log_chunks(
            response.content,
            tty=tty,
        ),
    )


async def get_docker_container_health(
    host: str,
    container: str,
    history_limit: int = 10,
) -> DockerContainerHealth:
    """
    Get Docker healthcheck configuration, status, and recent results.

    Includes healthcheck command output that is intentionally omitted from
    get_docker_container().
    """
    if not 1 <= history_limit <= 20:
        raise RuntimeError(
            "history_limit must be between 1 and 20"
        )

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
                healthcheck,
                "Test",
                "Interval",
                "Timeout",
                "Retries",
                "StartPeriod",
                "StartInterval",
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
    get_docker_container_stats,
    get_docker_container_logs,
    get_docker_container_health,
)
