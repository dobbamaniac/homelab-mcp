from __future__ import annotations

import asyncio
import json
import re
import shlex
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from homelab_mcp.utils import require_env


# -------------------------------------------------------------------------------------------------
# Structured output
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LinuxHost:
    name: str
    hostname: str
    operating_system: str
    kernel_version: str
    architecture: str
    virtualization: str | None
    uptime_seconds: int


@dataclass(frozen=True)
class LinuxHosts:
    hosts: list[LinuxHost]


@dataclass(frozen=True)
class LinuxStatus:
    host: str
    cpu_count: int
    load_average: list[float]
    memory_total_bytes: int
    memory_available_bytes: int
    swap_total_bytes: int
    swap_used_bytes: int
    uptime_seconds: int


@dataclass(frozen=True)
class LinuxBlockDevice:
    name: str
    path: str
    type: str
    size_bytes: int
    filesystem: str | None
    mountpoints: list[str]


@dataclass(frozen=True)
class LinuxFilesystem:
    source: str
    target: str
    filesystem: str
    size_bytes: int
    used_bytes: int
    available_bytes: int
    usage_ratio: float
    read_only: bool


@dataclass(frozen=True)
class LinuxStorage:
    host: str
    block_devices: list[LinuxBlockDevice]
    filesystems: list[LinuxFilesystem]


@dataclass(frozen=True)
class LinuxInterface:
    name: str
    state: str
    mtu: int
    mac_address: str | None
    addresses: list[str]


@dataclass(frozen=True)
class LinuxRoute:
    destination: str
    gateway: str | None
    device: str | None
    source: str | None
    metric: int | None


@dataclass(frozen=True)
class LinuxListener:
    protocol: str
    address: str
    port: int


@dataclass(frozen=True)
class LinuxNetwork:
    host: str
    interfaces: list[LinuxInterface]
    routes: list[LinuxRoute]
    dns_servers: list[str]
    listeners: list[LinuxListener]


@dataclass(frozen=True)
class LinuxService:
    name: str
    description: str
    load_state: str
    active_state: str
    sub_state: str


@dataclass(frozen=True)
class LinuxServices:
    host: str
    services: list[LinuxService]

# -------------------------------------------------------------------------------------------------
# Internal SSH helpers
# -------------------------------------------------------------------------------------------------


class LinuxRequestError(RuntimeError):
    pass


_hosts: dict[str, str] = {}

_HOST_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_HOST_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
_SSH_TIMEOUT = 15.0
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_PSEUDO_FILESYSTEMS = {
    "devtmpfs",
    "tmpfs",
    "proc",
    "sysfs",
    "cgroup",
    "cgroup2",
    "devpts",
    "securityfs",
    "debugfs",
    "tracefs",
    "configfs",
    "pstore",
    "mqueue",
}
_DOCKER_BRIDGE = re.compile(r"br-[0-9a-f]{12}")


def _configured_hosts() -> dict[str, str]:
    value = require_env("LINUX_HOSTS")
    hosts: dict[str, str] = {}

    for entry in value.split(","):
        name, separator, target = entry.partition("=")
        name = name.strip()
        target = target.strip()

        if not separator or not name or not target:
            raise RuntimeError("LINUX_HOSTS must use name=host entries separated by commas")

        if not _HOST_REF.fullmatch(name):
            raise RuntimeError(f"Invalid Linux host name {name!r}")

        if not _HOST_TARGET.fullmatch(target):
            raise RuntimeError(f"Invalid Linux SSH target {target!r}")

        if name in hosts:
            raise RuntimeError(f"Duplicate Linux host {name!r}")

        hosts[name] = target

    return hosts


def _host(name: str) -> str:
    target = _hosts.get(name)

    if target is None:
        raise RuntimeError(f"Linux host {name!r} is not configured")

    return target


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Validate Linux SSH configuration for the server lifetime."""
    if _hosts:
        raise RuntimeError("Linux hosts are already initialized")

    if shutil.which("ssh") is None:
        raise RuntimeError("OpenSSH client is not installed")

    key = Path(require_env("LINUX_SSH_KEY"))
    known_hosts = Path(require_env("LINUX_KNOWN_HOSTS"))

    if not key.is_file():
        raise RuntimeError(f"Linux SSH key does not exist: {key}")

    if not known_hosts.is_file():
        raise RuntimeError(f"Linux known_hosts file does not exist: {known_hosts}")

    _hosts.update(_configured_hosts())

    try:
        yield

    finally:
        _hosts.clear()


async def _run(
    host: str,
    *command: str,
    check: bool = True,
) -> str:
    if not command:
        raise RuntimeError("Linux command cannot be empty")

    target = _host(host)
    user = require_env("LINUX_SSH_USER")
    key = require_env("LINUX_SSH_KEY")
    known_hosts = require_env("LINUX_KNOWN_HOSTS")

    remote_command = shlex.join((
        "env",
        "LC_ALL=C",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        *command,
    ))

    process = await asyncio.create_subprocess_exec(
        "ssh",
        "-F", "/dev/null",
        "-T",
        "-i", key,
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "GlobalKnownHostsFile=/dev/null",
        "-o", "UpdateHostKeys=no",
        "-o", f"HostKeyAlias={host}",
        "-o", "ClearAllForwardings=yes",
        "-o", "ConnectTimeout=5",
        "-o", "ConnectionAttempts=1",
        "-o", "LogLevel=ERROR",
        f"{user}@{target}",
        remote_command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=_SSH_TIMEOUT,
        )

    except TimeoutError as exc:
        process.kill()
        await process.communicate()

        raise LinuxRequestError(
            f"Linux host {host!r} timed out running {command[0]!r}"
        ) from exc

    if check and process.returncode:
        message = stderr.decode(errors="replace").strip()

        if len(message) > 500:
            message = message[:500] + "..."

        raise LinuxRequestError(
            f"Linux host {host!r} returned exit status "
            f"{process.returncode} for {command[0]!r}"
            + (f": {message}" if message else "")
        )

    if len(stdout) > _MAX_OUTPUT_BYTES:
        raise LinuxRequestError(
            f"Linux host {host!r} returned too much output "
            f"for {command[0]!r}"
        )

    try:
        return stdout.decode()

    except UnicodeDecodeError as exc:
        raise LinuxRequestError(
            f"Linux host {host!r} returned invalid text "
            f"for {command[0]!r}"
        ) from exc


async def _run_json(
    host: str,
    *command: str,
) -> Any:
    output = await _run(host, *command)

    try:
        return json.loads(output)

    except json.JSONDecodeError as exc:
        raise LinuxRequestError(
            f"Linux host {host!r} returned invalid JSON "
            f"for {command[0]!r}"
        ) from exc


def _flatten(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []

    for item in items:
        flattened.append(item)
        flattened.extend(_flatten(item.get("children") or []))

    return flattened


def _meminfo(value: str) -> dict[str, int]:
    memory: dict[str, int] = {}

    for line in value.splitlines():
        if ":" not in line:
            continue

        key, raw = line.split(":", 1)
        fields = raw.split()

        if fields:
            memory[key] = int(fields[0]) * 1024

    return memory


def _dns_servers(value: str) -> list[str]:
    servers = []

    for line in value.splitlines():
        fields = line.split()

        if len(fields) >= 2 and fields[0] == "nameserver":
            servers.append(fields[1])

    return sorted(set(servers))


def _endpoint(value: str) -> tuple[str, int]:
    address, separator, port = value.rpartition(":")

    if not separator or not port.isdigit():
        raise ValueError(f"Unable to parse socket endpoint {value!r}")

    return (
        address.replace("[", "").replace("]", ""),
        int(port),
    )


def _virtual_interface(name: str) -> bool:
    return (
        name == "docker0"
        or name.startswith("veth")
        or bool(_DOCKER_BRIDGE.fullmatch(name))
    )


# -------------------------------------------------------------------------------------------------
# MCP tools
# -------------------------------------------------------------------------------------------------


async def _get_linux_host(name: str) -> LinuxHost:
    info, architecture, virtualization, uptime = await asyncio.gather(
        _run_json(
            name,
            "hostnamectl",
            "--json=short",
            "status",
        ),
        _run(
            name,
            "uname",
            "-m",
        ),
        _run(
            name,
            "systemd-detect-virt",
            check=False,
        ),
        _run(
            name,
            "cat",
            "/proc/uptime",
        ),
    )

    if not isinstance(info, dict):
        raise LinuxRequestError(
            f"Linux host {name!r} returned invalid hostname information"
        )

    virtualization = virtualization.strip()

    return LinuxHost(
        name=name,
        hostname=str(info.get("Hostname", "")),
        operating_system=str(
            info.get("OperatingSystemPrettyName", "")
        ),
        kernel_version=str(info.get("KernelRelease", "")),
        architecture=architecture.strip(),
        virtualization=(
            virtualization
            if virtualization and virtualization != "none"
            else None
        ),
        uptime_seconds=int(float(uptime.split()[0])),
    )

async def list_linux_hosts() -> LinuxHosts:
    """List configured Linux hosts and compact operating-system information."""
    hosts = await asyncio.gather(*(
        _get_linux_host(name)
        for name in sorted(_hosts)
    ))

    return LinuxHosts(hosts=list(hosts))


async def get_linux_status(host: str) -> LinuxStatus:
    """Get current CPU, memory, swap, load, and uptime status for a Linux host."""
    cpu, load, raw_memory, uptime = await asyncio.gather(
        _run(
            host,
            "getconf",
            "_NPROCESSORS_ONLN",
        ),
        _run(
            host,
            "cat",
            "/proc/loadavg",
        ),
        _run(
            host,
            "cat",
            "/proc/meminfo",
        ),
        _run(
            host,
            "cat",
            "/proc/uptime",
        ),
    )

    memory = _meminfo(raw_memory)
    swap_total = memory.get("SwapTotal", 0)
    swap_free = memory.get("SwapFree", 0)

    return LinuxStatus(
        host=host,
        cpu_count=int(cpu.strip()),
        load_average=[
            float(value)
            for value in load.split()[:3]
        ],
        memory_total_bytes=memory.get("MemTotal", 0),
        memory_available_bytes=memory.get("MemAvailable", 0),
        swap_total_bytes=swap_total,
        swap_used_bytes=max(swap_total - swap_free, 0),
        uptime_seconds=int(float(uptime.split()[0])),
    )


async def get_linux_storage(host: str) -> LinuxStorage:
    """Get block-device and mounted-filesystem capacity for a Linux host."""
    lsblk, findmnt = await asyncio.gather(
        _run_json(
            host,
            "lsblk",
            "--json",
            "--bytes",
            "--output",
            "NAME,PATH,TYPE,SIZE,FSTYPE,MOUNTPOINTS",
        ),
        _run_json(
            host,
            "findmnt",
            "--json",
            "--bytes",
            "--df",
            "--output",
            "SOURCE,TARGET,FSTYPE,SIZE,USED,AVAIL,USE%,OPTIONS",
        ),
    )

    if not isinstance(lsblk, dict) or not isinstance(findmnt, dict):
        raise LinuxRequestError(f"Linux host {host!r} returned invalid storage information")

    block_devices = [
        LinuxBlockDevice(
            name=str(item.get("name", "")),
            path=str(item.get("path", "")),
            type=str(item.get("type", "")),
            size_bytes=int(item.get("size") or 0),
            filesystem=(
                str(item["fstype"])
                if item.get("fstype")
                else None
            ),
            mountpoints=[
                str(value)
                for value in (item.get("mountpoints") or [])
                if value
            ],
        )
        for item in _flatten(lsblk.get("blockdevices") or [])
        if str(item.get("type", "")) != "loop"
    ]

    filesystems: list[LinuxFilesystem] = []

    for item in _flatten(findmnt.get("filesystems") or []):
        if str(item.get("fstype", "")) in _PSEUDO_FILESYSTEMS:
            continue

        usage = str(item.get("use%") or "0").removesuffix("%")

        filesystems.append(
            LinuxFilesystem(
                source=str(item.get("source", "")),
                target=str(item.get("target", "")),
                filesystem=str(item.get("fstype", "")),
                size_bytes=int(item.get("size") or 0),
                used_bytes=int(item.get("used") or 0),
                available_bytes=int(item.get("avail") or 0),
                usage_ratio=float(usage or 0) / 100,
                read_only=(
                    "ro"
                    in str(item.get("options", "")).split(",")
                ),
            )
        )

    block_devices.sort(key=lambda item: item.path)
    filesystems.sort(key=lambda item: item.target)

    return LinuxStorage(
        host=host,
        block_devices=block_devices,
        filesystems=filesystems,
    )


async def get_linux_network(host: str) -> LinuxNetwork:
    """Get interfaces, routes, DNS servers, and listening sockets for a Linux host."""
    addresses, route_data, sockets, resolv_conf = await asyncio.gather(
        _run_json(
            host,
            "ip",
            "-json",
            "address",
            "show",
        ),
        _run_json(
            host,
            "ip",
            "-json",
            "route",
            "show",
            "table",
            "main",
        ),
        _run(
            host,
            "ss",
            "-H",
            "-lntu",
        ),
        _run(
            host,
            "cat",
            "/etc/resolv.conf",
            check=False,
        ),
    )

    if not isinstance(addresses, list) or not isinstance(route_data, list):
        raise LinuxRequestError(f"Linux host {host!r} returned invalid network information")

    interfaces = [
        LinuxInterface(
            name=str(item.get("ifname", "")),
            state=str(item.get("operstate", "")),
            mtu=int(item.get("mtu") or 0),
            mac_address=(
                str(item["address"])
                if item.get("address")
                else None
            ),
            addresses=[
                f"{address['local']}/{address['prefixlen']}"
                for address in item.get("addr_info", [])
                if address.get("local") is not None
                and address.get("prefixlen") is not None
            ],
        )
        for item in addresses
        if not _virtual_interface(str(item.get("ifname", "")))
    ]

    routes = [
        LinuxRoute(
            destination=str(item.get("dst", "default")),
            gateway=(
                str(item["gateway"])
                if item.get("gateway")
                else None
            ),
            device=(
                str(item["dev"])
                if item.get("dev")
                else None
            ),
            source=(
                str(item["prefsrc"])
                if item.get("prefsrc")
                else None
            ),
            metric=(
                int(item["metric"])
                if item.get("metric") is not None
                else None
            ),
        )
        for item in route_data
        if not _virtual_interface(str(item.get("dev", "")))
    ]

    listeners: list[LinuxListener] = []

    for line in sockets.splitlines():
        fields = line.split()

        if len(fields) < 5:
            continue

        try:
            address, port = _endpoint(fields[4])

        except ValueError:
            continue

        listeners.append(
            LinuxListener(
                protocol=fields[0],
                address=address,
                port=port,
            )
        )

    interfaces.sort(key=lambda item: item.name)
    routes.sort(key=lambda item: (
        item.destination,
        item.device or "",
        item.gateway or "",
    ))
    listeners.sort(key=lambda item: (
        item.port,
        item.protocol,
        item.address,
    ))

    return LinuxNetwork(
        host=host,
        interfaces=interfaces,
        routes=routes,
        dns_servers=_dns_servers(resolv_conf),
        listeners=listeners,
    )


async def get_linux_services(
    host: str,
    state: Literal["active", "inactive", "failed", "all"] = "active", 
) -> LinuxServices:
    """
    List loaded systemd services on a Linux host.

    Defaults to active services. Set state to "inactive", "failed",
    or "all" to select another view.
    """
    output = await _run(
        host,
        "systemctl",
        "--plain",
        "--no-legend",
        "--no-pager",
        "--full",
        "list-units",
        "--type=service",
        "--all",
    )

    services: list[LinuxService] = []

    for line in output.splitlines():
        line = line.lstrip()

        if line[:1] in {"●", "○", "×"}:
            line = line[1:].lstrip()

        fields = line.split(None, 4)

        if len(fields) < 4:
            continue

        services.append(
            LinuxService(
                name=fields[0],
                load_state=fields[1],
                active_state=fields[2],
                sub_state=fields[3],
                description=(
                    fields[4]
                    if len(fields) == 5
                    else ""
                ),
            )
        )

    services = [
        service
        for service in services
        if service.load_state == "loaded"
    ]

    if state != "all":
        services = [
            service
            for service in services
            if service.active_state == state
        ]

    services.sort(key=lambda service: service.name)

    return LinuxServices(
        host=host,
        services=services,
    )


TOOLS = (
    list_linux_hosts,
    get_linux_status,
    get_linux_storage,
    get_linux_network,
    get_linux_services,
)
