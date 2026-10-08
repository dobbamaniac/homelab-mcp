from __future__ import annotations

import posixpath
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from pydantic import Field

from homelab_mcp.redact import redact_text, redact_value
from homelab_mcp.ssh import (
    RequestError as LinuxRequestError,
    host_names as _ssh_hosts,
    run as _run,
    run_json as _run_json,
)
from homelab_mcp.utils import pick, require_env


# ---------------------------------------------------------------------------
# Tool parameter types
# ---------------------------------------------------------------------------

HostRef = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="Linux SSH host name returned by list_linux_hosts.",
    ),
]

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
        ),
        description="systemd unit name returned by list_linux_units or " "list_linux_unit_files.",
    ),
]

SysctlPrefix = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$",
        description='Dotted sysctl prefix such as "net.ipv4" or "vm".',
    ),
]

LinuxFileRef = Annotated[
    str,
    Field(
        description="Approved absolute path returned by list_linux_files.",
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_files: set[str] = set()

_PSEUDO_FILESYSTEMS = {
    "devtmpfs", "tmpfs", "proc", "sysfs", "cgroup", "cgroup2", "devpts", "securityfs", "debugfs", 
    "tracefs", "configfs", "pstore", "mqueue",
}

_MAX_FILE_BYTES = 128 * 1024


def _configured_files() -> set[str]:
    files: set[str] = set()

    for value in require_env("LINUX_FILES").split(","):
        path = value.strip()

        if not path:
            raise RuntimeError("LINUX_FILES contains an empty entry")

        if not path.startswith("/") or posixpath.normpath(path) != path:
            raise RuntimeError(f"LINUX_FILES must contain normalized absolute paths, got {path!r}")

        if path in files:
            raise RuntimeError(f"Duplicate Linux configuration path {path!r}")

        files.add(path)

    return files


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Validate Linux configuration for the server lifetime."""
    if _files:
        raise RuntimeError("Linux configuration is already initialized")

    _files.update(_configured_files())

    try:
        yield

    finally:
        _files.clear()


def _flatten(
    items: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for item in items:
        yield item
        yield from _flatten(item.get("children") or [])


def _endpoint(value: str) -> tuple[str, int]:
    address, separator, port = value.rpartition(":")

    if not separator or not port.isdigit():
        raise ValueError(f"Unable to parse socket endpoint {value!r}")

    return (
        address.replace("[", "").replace("]", ""),
        int(port),
    )


# ---------------------------------------------------------------------------
# Host inventory
# ---------------------------------------------------------------------------

async def list_linux_hosts() -> dict[str, Any]:
    """
    List Linux host names configured for SSH access.

    Source: server-side SSH_HOSTS configuration.

    No SSH command is run.
    """
    return {"hosts": list(_ssh_hosts())}


async def get_linux_host(
    host: HostRef,
) -> dict[str, Any]:
    """
    Get operating-system and hardware identity for one Linux host.

    Upstream command:
    hostnamectl --json=short status
    """
    data = await _run_json(
        host,
        "hostnamectl", "--json=short", "status",
    )

    if not isinstance(data, dict):
        raise LinuxRequestError(
            f"Linux host {host!r} returned invalid hostname information"
        )

    return redact_value(pick(
        data,
        "Hostname", "OperatingSystemPrettyName", "KernelName", "KernelRelease", "Architecture", 
        "Virtualization", "HardwareVendor", "HardwareModel", "FirmwareVersion"
    ))


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

async def list_linux_block_devices(
    host: HostRef,
) -> dict[str, Any]:
    """
    List block devices on one Linux host.

    Upstream command:
    lsblk --json --bytes --output NAME,PATH,PKNAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL
    """
    data = await _run_json(
        host,
        "lsblk", "--json", "--bytes", "--output",
        "NAME,PATH,PKNAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL"
    )

    if not isinstance(data, dict):
        raise LinuxRequestError(f"Linux host {host!r} returned invalid block-device information")

    devices = []

    for item in _flatten(data.get("blockdevices") or []):
        if str(item.get("type", "")) == "loop":
            continue

        devices.append(pick(
            item,
            "name", "path", "pkname", "type", "size", "fstype", "mountpoints", "model", "serial"
        ))

    devices.sort(key=lambda item: str(item.get("path", "")))

    return {"block_devices": devices}


async def list_linux_filesystems(
    host: HostRef,
) -> dict[str, Any]:
    """
    List mounted filesystems on one Linux host.

    Upstream command:
    findmnt --json --bytes --df --output SOURCE,TARGET,FSTYPE,SIZE,USED,AVAIL,USE%,OPTIONS

    Pseudo-filesystems such as proc, sysfs, tmpfs, and cgroup are omitted.
    """
    data = await _run_json(
        host,
        "findmnt",
        "--json", "--bytes", "--df", "--output",
        "SOURCE,TARGET,FSTYPE,SIZE,USED,AVAIL,USE%,OPTIONS",
    )

    if not isinstance(data, dict):
        raise LinuxRequestError(f"Linux host {host!r} returned invalid filesystem information")

    filesystems = [
        pick(
            item,
            "source", "target", "fstype", "size", "used", "avail", "use%", "options"
        )
        for item in _flatten(data.get("filesystems") or [])
        if str(item.get("fstype", "")) not in _PSEUDO_FILESYSTEMS
    ]

    filesystems.sort(key=lambda item: str(item.get("target", "")))

    return {"filesystems": filesystems}


# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

async def list_linux_interfaces(
    host: HostRef,
) -> dict[str, Any]:
    """
    List network interfaces and assigned addresses on one Linux host.

    Upstream command:
    ip -json address show
    """
    data = await _run_json(
        host,
        "ip", "-json", "address", "show",
    )

    if not isinstance(data, list):
        raise LinuxRequestError(
            f"Linux host {host!r} returned invalid interface information"
        )

    interfaces = []

    for item in data:
        interfaces.append({
            **pick(
                item,
                "ifindex", "ifname", "flags", "mtu", "qdisc", "operstate", "group", "link_type", 
                "address", "broadcast", "master",
            ),
            "addr_info": [
                pick(
                    address,
                    "family", "local", "prefixlen", "broadcast", "scope", "label", "dynamic", 
                    "deprecated", "valid_life_time", "preferred_life_time"
                )
                for address in (item.get("addr_info") or [])
            ],
        })

    interfaces.sort(key=lambda item: str(item.get("ifname", "")))

    return {"interfaces": interfaces}


async def list_linux_routes(
    host: HostRef,
) -> dict[str, Any]:
    """
    List routes from the main routing table on one Linux host.

    Upstream command:
    ip -json route show table main
    """
    data = await _run_json(
        host,
        "ip", "-json", "route", "show", "table", "main",
    )

    if not isinstance(data, list):
        raise LinuxRequestError(
            f"Linux host {host!r} returned invalid route information"
        )

    routes = [
        pick(
            item,
            "dst", "gateway", "dev", "prefsrc", "metric", "protocol", "scope", "type"
        )
        for item in data
    ]

    routes.sort(key=lambda item: (
        str(item.get("dst", "default")),
        str(item.get("dev", "")),
        str(item.get("gateway", "")),
    ))

    return {"routes": routes}


async def list_linux_listeners(
    host: HostRef,
) -> dict[str, Any]:
    """
    List listening TCP and UDP sockets on one Linux host.

    Upstream command:
    ss -H -lntu
    """
    output = await _run(
        host,
        "ss", "-H", "-lntu",
    )

    listeners = []

    for line in output.splitlines():
        fields = line.split()

        if len(fields) < 5:
            continue

        try:
            address, port = _endpoint(fields[4])

        except ValueError:
            continue

        listeners.append({
            "protocol": fields[0],
            "state": fields[1],
            "address": address,
            "port": port,
        })

    listeners.sort(key=lambda item: (
        item["port"],
        item["protocol"],
        item["address"],
    ))

    return {"listeners": listeners}


async def get_linux_resolver(
    host: HostRef,
) -> dict[str, Any]:
    """
    Get resolver configuration from one Linux host.

    Upstream command:
    cat /etc/resolv.conf
    """
    output = await _run(
        host,
        "cat", "/etc/resolv.conf",
        check=False,
    )

    nameservers: list[str] = []
    search: list[str] = []
    domain: str | None = None
    options: list[str] = []

    for line in output.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split()

        if not fields:
            continue

        if fields[0] == "nameserver" and len(fields) >= 2:
            nameservers.append(fields[1])

        elif fields[0] == "search":
            search.extend(fields[1:])

        elif fields[0] == "domain" and len(fields) >= 2:
            domain = fields[1]

        elif fields[0] == "options":
            options.extend(fields[1:])

    return {
        "nameservers": nameservers,
        "search": search,
        "domain": domain,
        "options": options,
    }


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------

async def list_linux_packages(
    host: HostRef,
) -> dict[str, Any]:
    """
    List installed Debian packages on one Linux host.

    Upstream command:
    dpkg-query --show --showformat=...

    Automatic/manual state and package holds are available through the separate
    apt-mark tools.
    """
    output = await _run(
        host,
        "dpkg-query", "--show", "--showformat=${binary:Package}\\t${Version}\\t${Architecture}\\t"
        "${db:Status-Status}\\t${binary:Synopsis}\\n",
    )

    packages = []

    for line in output.splitlines():
        fields = line.split("\t", 4)

        if len(fields) != 5:
            continue

        name, version, architecture, status, synopsis = fields

        if status != "installed":
            continue

        packages.append({
            "name": name,
            "version": version,
            "architecture": architecture,
            "synopsis": redact_text(synopsis),
        })

    packages.sort(key=lambda package: package["name"])

    return {"packages": packages}


async def list_linux_automatic_packages(
    host: HostRef,
) -> dict[str, Any]:
    """
    List packages marked as automatically installed.

    Upstream command:
    apt-mark showauto
    """
    output = await _run(
        host,
        "apt-mark", "showauto",
    )

    return {
        "packages": sorted(
            line.strip()
            for line in output.splitlines()
            if line.strip()
        )
    }


async def list_linux_held_packages(
    host: HostRef,
) -> dict[str, Any]:
    """
    List packages held at their current version.

    Upstream command:
    apt-mark showhold
    """
    output = await _run(
        host,
        "apt-mark", "showhold",
    )

    return {
        "packages": sorted(
            line.strip()
            for line in output.splitlines()
            if line.strip()
        )
    }


# ---------------------------------------------------------------------------
# systemd
# ---------------------------------------------------------------------------

async def list_linux_units(
    host: HostRef,
    type: UnitType = "service",
    state: UnitState = "active",
) -> dict[str, Any]:
    """
    List loaded systemd units on one Linux host.

    Upstream command:
    systemctl --plain --no-legend --no-pager --full list-units --type={type} --all

    Defaults to active services. `state` filters the returned unit state without
    making another command request.
    """
    output = await _run(
        host,
        "systemctl", "--plain", "--no-legend", "--no-pager", "--full", "list-units", 
        f"--type={type}", "--all",
    )

    units = []

    for line in output.splitlines():
        line = line.lstrip()

        if line[:1] in {"●", "○", "×"}:
            line = line[1:].lstrip()

        fields = line.split(None, 4)

        if len(fields) < 4:
            continue

        item = {
            "name": fields[0],
            "load_state": fields[1],
            "active_state": fields[2],
            "sub_state": fields[3],
            "description": (
                redact_text(fields[4])
                if len(fields) == 5
                else ""
            ),
        }

        if item["load_state"] != "loaded":
            continue

        if state != "all" and item["active_state"] != state:
            continue

        units.append(item)

    units.sort(key=lambda unit: unit["name"])

    return {"units": units}


async def get_linux_unit(
    host: HostRef,
    unit: UnitRef,
) -> dict[str, Any]:
    """
    Get state and configuration provenance for one systemd unit.

    Upstream command:
    systemctl --no-pager --property=... show {unit}
    """
    output = await _run(
        host,
        "systemctl", "--no-pager", "--property=Id,Description,LoadState,ActiveState,SubState,"
        "UnitFileState,UnitFilePreset,FragmentPath,SourcePath,DropInPaths,NeedDaemonReload",
        "show", unit
    )

    properties = dict(
        line.split("=", 1)
        for line in output.splitlines()
        if "=" in line
    )

    result = pick(
        properties,
        "Id", "Description", "LoadState", "ActiveState", "SubState", "UnitFileState", 
        "UnitFilePreset", "FragmentPath", "SourcePath", "DropInPaths", "NeedDaemonReload"
    )

    return redact_value(result)


async def list_linux_unit_files(
    host: HostRef,
    type: UnitType = "service",
) -> dict[str, Any]:
    """
    List installed systemd unit files and their enablement state.

    Upstream command:
    systemctl --no-legend --no-pager --full list-unit-files --type={type}
    """
    output = await _run(
        host,
        "systemctl", "--no-legend", "--no-pager", "--full", "list-unit-files", f"--type={type}"
    )

    unit_files = []

    for line in output.splitlines():
        fields = line.split()

        if len(fields) < 2:
            continue

        unit_files.append({
            "name": fields[0],
            "state": fields[1],
            "preset": (
                fields[2]
                if len(fields) >= 3 and fields[2] != "-"
                else None
            ),
        })

    unit_files.sort(key=lambda item: item["name"])

    return {"unit_files": unit_files}


# ---------------------------------------------------------------------------
# Kernel configuration
# ---------------------------------------------------------------------------

async def get_linux_sysctls(
    host: HostRef,
    prefix: SysctlPrefix,
) -> dict[str, Any]:
    """
    Get current kernel sysctl values below a prefix.

    Upstream command:
    sysctl --all --pattern {prefix}
    """
    pattern = "^" + prefix.replace(".", r"\.") + r"(\.|$)"

    output = await _run(
        host,
        "sysctl", "--all", "--pattern", pattern,
    )

    values = []

    for line in output.splitlines():
        key, separator, value = line.partition("=")

        if not separator:
            continue

        values.append({
            "key": key.strip(),
            "value": redact_text(value.strip()),
        })

    values.sort(key=lambda item: item["key"])

    return {"values": values}


# ---------------------------------------------------------------------------
# Approved configuration files
# ---------------------------------------------------------------------------

async def list_linux_files() -> dict[str, Any]:
    """
    List configuration-file paths approved for read_linux_file.

    Source: server-side LINUX_FILES configuration.

    No SSH command is run.
    """
    return {
        "paths": sorted(_files),
        "max_file_bytes": _MAX_FILE_BYTES,
    }


async def read_linux_file(
    host: HostRef,
    path: LinuxFileRef,
) -> dict[str, Any]:
    """
    Read one explicitly approved Linux configuration file.

    Safety exception: this tool intentionally uses two commands.

    Upstream commands:
    stat --dereference --printf=... -- {path}
    head --bytes=131073 -- {path}

    The path must exactly match an entry returned by list_linux_files. File type
    and size are checked before reading. Content is bounded and redacted.
    """
    if path not in _files:
        raise RuntimeError(f"File path {path!r} is not approved")

    metadata = await _run(
        host,
        "stat", "--dereference", r"--printf=%F\n%s\n%a\n%u\n%g\n%U\n%G\n%Y\n", "--", path,
    )

    fields = metadata.splitlines()

    if len(fields) != 8:
        raise LinuxRequestError(
            f"Linux host {host!r} returned invalid stat information for {path!r}"
        )

    (
        file_type,
        raw_size,
        raw_mode,
        raw_uid,
        raw_gid,
        owner,
        group,
        raw_modified,
    ) = fields

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

    raw_content: str | None = None
    content: str | None = None
    read_error: str | None = None

    if file_type != "regular file":
        read_error = "Path does not resolve to a regular file"

    elif size_bytes > _MAX_FILE_BYTES:
        read_error = f"File exceeds the {_MAX_FILE_BYTES}-byte limit"

    else:
        try:
            raw_content = await _run(
                host,
                "head", f"--bytes={_MAX_FILE_BYTES + 1}", "--", path
            )

            if len(raw_content.encode()) > _MAX_FILE_BYTES:
                read_error = "File exceeded the size limit while being read"

            elif "\0" in raw_content:
                read_error = "File is not text"

            else:
                content = redact_text(raw_content)

        except LinuxRequestError as exc:
            read_error = str(exc)

    return {
        "path": path,
        "file_type": file_type,
        "size_bytes": size_bytes,
        "mode": mode,
        "uid": uid,
        "gid": gid,
        "owner": owner,
        "group": group,
        "modified_at_epoch": modified_at_epoch,
        "content": content,
        "redacted": (
            raw_content is not None
            and content is not None
            and content != raw_content
        ),
        "read_error": read_error,
    }


TOOLS = (
    list_linux_hosts,
    get_linux_host,
    list_linux_block_devices,
    list_linux_filesystems,
    list_linux_interfaces,
    list_linux_routes,
    list_linux_listeners,
    get_linux_resolver,
    list_linux_packages,
    list_linux_automatic_packages,
    list_linux_held_packages,
    list_linux_units,
    get_linux_unit,
    list_linux_unit_files,
    get_linux_sysctls,
    list_linux_files,
    read_linux_file,
)

