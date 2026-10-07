from __future__ import annotations

import asyncio
import posixpath
import re
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Field

from homelab_mcp.redact import redact_text
from homelab_mcp.utils import require_env
from homelab_mcp.ssh import (
    RequestError as LinuxRequestError,
    host_names as _ssh_hosts,
    run as _run,
    run_json as _run_json
)


# -------------------------------------------------------------------------------------------------
# Structured output
# -------------------------------------------------------------------------------------------------


UnitType = Literal[
    "service", "socket", "timer", "target", "device", "mount", "automount", "swap", "path", 
    "slice", "scope"
]

UnitState = Literal["active", "inactive", "failed", "all"]

UnitRef = Annotated[
    str,
    Field(
        pattern=(
            r"^[A-Za-z0-9][A-Za-z0-9_.@:-]*"
            r"\.(?:service|socket|timer|target|device|mount|automount|"
            r"swap|path|slice|scope)$"
        )
    ),
]

SysctlPrefix = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$"),
]


@dataclass(frozen=True)
class LinuxHost:
    name: str
    hostname: str
    operating_system: str
    kernel_version: str
    architecture: str
    virtualization: str | None


@dataclass(frozen=True)
class LinuxHosts:
    hosts: list[LinuxHost]


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
class LinuxPackage:
    name: str
    version: str
    architecture: str
    automatic: bool
    held: bool
    synopsis: str


@dataclass(frozen=True)
class LinuxPackages:
    host: str
    packages: list[LinuxPackage]


@dataclass(frozen=True)
class LinuxUnit:
    name: str
    description: str
    load_state: str
    active_state: str
    sub_state: str


@dataclass(frozen=True)
class LinuxUnits:
    host: str
    units: list[LinuxUnit]


@dataclass(frozen=True)
class LinuxUnitDetails:
    host: str
    name: str
    description: str
    load_state: str
    active_state: str
    sub_state: str
    unit_file_state: str | None
    unit_file_preset: str | None
    fragment_path: str | None
    source_path: str | None
    drop_in_paths: list[str]
    need_daemon_reload: bool


@dataclass(frozen=True)
class LinuxUnitFile:
    name: str
    state: str
    preset: str | None


@dataclass(frozen=True)
class LinuxUnitFiles:
    host: str
    unit_files: list[LinuxUnitFile]


@dataclass(frozen=True)
class LinuxSysctl:
    key: str
    value: str


@dataclass(frozen=True)
class LinuxSysctls:
    host: str
    prefix: str
    values: list[LinuxSysctl]


@dataclass(frozen=True)
class LinuxFile:
    host: str
    path: str
    file_type: str
    size_bytes: int
    mode: str
    uid: int
    gid: int
    owner: str
    group: str
    modified_at_epoch: int
    content: str | None
    redacted: bool
    read_error: str | None


# -------------------------------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------------------------------


_files: set[str] = set()

_PSEUDO_FILESYSTEMS = {
    "devtmpfs", "tmpfs", "proc", "sysfs", "cgroup", "cgroup2", "devpts", "securityfs", "debugfs", 
    "tracefs", "configfs", "pstore", "mqueue",
}

_DOCKER_BRIDGE = re.compile(r"br-[0-9a-f]{12}")

_MAX_FILE_BYTES = 128 * 1024


def _configured_files() -> set[str]:
    files: set[str] = set()

    for value in require_env("LINUX_FILES").split(","):
        path = value.strip()

        if not path:
            raise RuntimeError("LINUX_FILES contains an empty entry")

        if not path.startswith("/") or posixpath.normpath(path) != path:
            raise RuntimeError(
                f"LINUX_FILES must contain normalized absolute paths, got {path!r}"
            )

        if path in files:
            raise RuntimeError(f"Duplicate Linux configuration path {path!r}")

        files.add(path)

    return files


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Validate Linux SSH configuration for the server lifetime."""
    if _files:
        raise RuntimeError("Linux configuration is already initialized")

    _files.update(_configured_files())

    try:
        yield

    finally:
        _files.clear()


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


def _flatten(
    items: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for item in items:
        yield item
        yield from _flatten(item.get("children") or [])


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
    info, architecture, virtualization = await asyncio.gather(
        _run_json(name, "hostnamectl", "--json=short", "status"),
        _run(name, "uname", "-m"),
        _run(name, "systemd-detect-virt", check=False),
    )

    if not isinstance(info, dict):
        raise LinuxRequestError(f"Linux host {name!r} returned invalid hostname information")

    virtualization = virtualization.strip()

    return LinuxHost(
        name=name,
        hostname=str(info.get("Hostname", "")),
        operating_system=str(info.get("OperatingSystemPrettyName", "")),
        kernel_version=str(info.get("KernelRelease", "")),
        architecture=architecture.strip(),
        virtualization=(
            virtualization
            if virtualization and virtualization != "none"
            else None
        ),
    )


async def list_linux_hosts() -> LinuxHosts:
    """List configured Linux hosts and compact operating-system information."""
    hosts = await asyncio.gather(*(
        _get_linux_host(name)
        for name in _ssh_hosts()
    ))

    return LinuxHosts(hosts=list(hosts))


async def get_linux_storage(host: str) -> LinuxStorage:
    """Get block-device and mounted-filesystem capacity for a Linux host."""
    lsblk, findmnt = await asyncio.gather(
        _run_json(
            host, "lsblk", "--json", "--bytes", 
            "--output", "NAME,PATH,TYPE,SIZE,FSTYPE,MOUNTPOINTS"
        ),
        _run_json(
            host, "findmnt", "--json", "--bytes", "--df", 
            "--output", "SOURCE,TARGET,FSTYPE,SIZE,USED,AVAIL,USE%,OPTIONS"
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
                read_only=("ro" in str(item.get("options", "")).split(",")),
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
        _run_json(host, "ip", "-json", "address", "show"),
        _run_json(host, "ip", "-json", "route", "show", "table", "main"),
        _run(host, "ss", "-H", "-lntu"),
        _run(host, "cat", "/etc/resolv.conf", check=False),
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


async def list_linux_packages(
    host: str,
    manual_only: bool = False,
) -> LinuxPackages:
    """
    List installed Debian packages on a Linux host.

    Includes automatic/manual installation state and package holds.
    Set manual_only to omit automatically installed dependencies.
    """
    package_output, automatic_output, held_output = await asyncio.gather(
        _run(
            host,
            "dpkg-query",
            "--show",
            "--showformat=${binary:Package}\\t${Version}\\t${Architecture}\\t"
            "${db:Status-Status}\\t${binary:Synopsis}\\n",
        ),
        _run(host, "apt-mark", "showauto"),
        _run(host, "apt-mark", "showhold"),
    )

    automatic = set(automatic_output.splitlines())
    held = set(held_output.splitlines())

    def marked(name: str, values: set[str]) -> bool:
        base_name = name.partition(":")[0]
        return name in values or base_name in values

    packages: list[LinuxPackage] = []

    for line in package_output.splitlines():
        fields = line.split("\t", 4)

        if len(fields) != 5:
            continue

        name, version, architecture, status, synopsis = fields

        if status != "installed":
            continue

        is_automatic = marked(name, automatic)

        if manual_only and is_automatic:
            continue

        packages.append(
            LinuxPackage(
                name=name,
                version=version,
                architecture=architecture,
                automatic=is_automatic,
                held=marked(name, held),
                synopsis=redact_text(synopsis),
            )
        )

    packages.sort(key=lambda package: package.name)

    return LinuxPackages(
        host=host,
        packages=packages,
    )


async def list_linux_units(
    host: str,
    type: UnitType = "service",
    state: UnitState = "active",
) -> LinuxUnits:
    """
    List loaded systemd units on a Linux host.

    Defaults to active services. Select another unit type or state as needed.
    """
    output = await _run(
        host, "systemctl", "--plain", "--no-legend", "--no-pager", "--full", "list-units", 
        f"--type={type}", "--all"
    )

    units: list[LinuxUnit] = []

    for line in output.splitlines():
        line = line.lstrip()

        if line[:1] in {"●", "○", "×"}:
            line = line[1:].lstrip()

        fields = line.split(None, 4)

        if len(fields) < 4:
            continue

        units.append(
            LinuxUnit(
                name=fields[0],
                load_state=fields[1],
                active_state=fields[2],
                sub_state=fields[3],
                description=(
                    redact_text(fields[4])
                    if len(fields) == 5
                    else ""
                ),
            )
        )

    units = [
        unit for unit in units
        if unit.load_state == "loaded"
        and (state == "all" or unit.active_state == state)
    ]

    units.sort(key=lambda unit: unit.name)

    return LinuxUnits(
        host=host,
        units=units,
    )


async def get_linux_unit(
    host: str,
    unit: UnitRef,
) -> LinuxUnitDetails:
    """Get state and configuration provenance for one systemd unit."""
    output = await _run(
        host, "systemctl", "--no-pager", 
        "--property=" "Id,Description,LoadState,ActiveState,SubState," 
        "UnitFileState,UnitFilePreset,FragmentPath,SourcePath," "DropInPaths,NeedDaemonReload", 
        "show", unit,
    )

    properties = dict(
        line.split("=", 1)
        for line in output.splitlines()
        if "=" in line
    )

    return LinuxUnitDetails(
        host=host,
        name=properties.get("Id", unit),
        description=redact_text(properties.get("Description", "")),
        load_state=properties.get("LoadState", ""),
        active_state=properties.get("ActiveState", ""),
        sub_state=properties.get("SubState", ""),
        unit_file_state=properties.get("UnitFileState") or None,
        unit_file_preset=properties.get("UnitFilePreset") or None,
        fragment_path=properties.get("FragmentPath") or None,
        source_path=properties.get("SourcePath") or None,
        drop_in_paths=properties.get("DropInPaths", "").split(),
        need_daemon_reload=(properties.get("NeedDaemonReload") == "yes"),
    )


async def get_linux_unit_files(
    host: str,
    type: UnitType = "service"
) -> LinuxUnitFiles:
    """List installed systemd unit files and their enablement state."""

    output = await _run(
        host, "systemctl", "--no-legend", "--no-pager", "--full", "list-unit-files", 
        f"--type={type}"
    )

    unit_files: list[LinuxUnitFile] = []

    for line in output.splitlines():
        fields = line.split()

        if len(fields) < 2:
            continue

        unit_files.append(
            LinuxUnitFile(
                name=fields[0],
                state=fields[1],
                preset=(
                    fields[2]
                    if len(fields) >= 3 and fields[2] != "-"
                    else None
                ),
            )
        )

    unit_files.sort(key=lambda item: item.name)

    return LinuxUnitFiles(
        host=host,
        unit_files=unit_files,
    )


async def get_linux_sysctls(
    host: str,
    prefix: SysctlPrefix,
) -> LinuxSysctls:
    """
    Get current kernel sysctl values below a prefix.

    The prefix must use dotted sysctl notation, such as "net.ipv4" or "vm".
    """
    pattern = "^" + prefix.replace(".", r"\.") + r"(\.|$)"

    output = await _run(host, "sysctl", "--all", "--pattern", pattern)

    values: list[LinuxSysctl] = []

    for line in output.splitlines():
        key, separator, value = line.partition("=")

        if not separator:
            continue

        values.append(
            LinuxSysctl(
                key=key.strip(),
                value=redact_text(value.strip()),
            )
        )

    values.sort(key=lambda item: item.key)

    return LinuxSysctls(
        host=host,
        prefix=prefix,
        values=values,
    )


async def read_linux_file(
    host: str,
    path: str,
) -> LinuxFile:
    """
    Attempt to read one explicitly approved configuration file.

    The path must exactly match an entry in LINUX_FILES. Returns file metadata even when the 
    configured SSH account cannot read the content. Text content is bounded and redacted before
    being returned.
    """
    if path not in _files:
        raise RuntimeError(f"File path {path!r} is not approved")

    metadata = await _run(
        host, "stat", "--dereference", r"--printf=%F\n%s\n%a\n%u\n%g\n%U\n%G\n%Y\n", "--", path
    )

    fields = metadata.splitlines()

    if len(fields) != 8:
        raise LinuxRequestError(
            f"Linux host {host!r} returned invalid stat information for {path!r}"
        )

    file_type, raw_size, raw_mode, raw_uid, raw_gid, owner, group, raw_modified = fields

    try:
        size_bytes = int(raw_size)
        mode = f"{int(raw_mode, 8):04o}"
        uid = int(raw_uid)
        gid = int(raw_gid)
        modified_at_epoch = int(raw_modified)

    except ValueError as exc:
        raise LinuxRequestError(
            f"Linux host {host!r} returned invalid stat information for {path!r}"
        ) from exc

    content: str | None = None
    read_error: str | None = None

    if file_type != "regular file":
        read_error = "Path does not resolve to a regular file"

    elif size_bytes > _MAX_FILE_BYTES:
        read_error = f"File exceeds the {_MAX_FILE_BYTES}-byte limit"

    else:
        try:
            raw_content = await _run(host, "head", f"--bytes={_MAX_FILE_BYTES + 1}", "--", path)

            if len(raw_content.encode()) > _MAX_FILE_BYTES:
                read_error = "File exceeded the size limit while being read"

            elif "\0" in raw_content:
                read_error = "File is not text"

            else:
                content = redact_text(raw_content)

        except LinuxRequestError as exc:
            read_error = str(exc)

    return LinuxFile(
        host=host,
        path=path,
        file_type=file_type,
        size_bytes=size_bytes,
        mode=mode,
        uid=uid,
        gid=gid,
        owner=owner,
        group=group,
        modified_at_epoch=modified_at_epoch,
        content=content,
        redacted=(content is not None and content != raw_content),
        read_error=read_error
    )


TOOLS = (
    list_linux_hosts,
    get_linux_storage,
    get_linux_network,
    list_linux_packages,
    list_linux_units,
    get_linux_unit,
    get_linux_unit_files,
    get_linux_sysctls,
    read_linux_file,
)
