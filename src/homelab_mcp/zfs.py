from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from pydantic import Field

from homelab_mcp.redact import redact_text
from homelab_mcp.ssh import host_names as _ssh_hosts, run as _run
from homelab_mcp.utils import require_env


# ---------------------------------------------------------------------------
# Tool parameter types
# ---------------------------------------------------------------------------

DatasetRef = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
        description="ZFS dataset name returned by list_zfs_datasets.",
    ),
]

ZFSPropertyRef = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@-]*$",
        description=(
            "ZFS property name, for example 'compression', "
            "'recordsize', or a user property such as 'org.example:foo'."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ZFS_PROPERTIES = (
    "mountpoint",
    "canmount",
    "compression",
    "compressratio",
    "recordsize",
    "atime",
    "relatime",
    "xattr",
    "acltype",
    "aclinherit",
    "sync",
    "dedup",
    "copies",
    "quota",
    "reservation",
    "encryption",
    "keyformat",
    "keylocation",
    "keystatus",
)


def _host() -> str:
    return require_env("ZFS_HOSTS")


def _optional_int(value: str) -> int | None:
    return None if value == "-" else int(value)


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Validate the configured ZFS host for the server lifetime."""
    host = _host()

    if host not in _ssh_hosts():
        raise RuntimeError(f"ZFS host {host!r} is not configured for SSH")

    yield


# ---------------------------------------------------------------------------
# ZFS
# ---------------------------------------------------------------------------

async def list_zfs_pools() -> dict[str, Any]:
    """
    List ZFS pools with current capacity and health.

    Upstream command:
    zpool list -Hp -o name,size,alloc,free,frag,cap,health
    """
    host = _host()

    output = await _run(
        host,
        "zpool", "list", "-Hp", "-o", "name,size,alloc,free,frag,cap,health",
    )

    pools = []

    for line in output.splitlines():
        fields = line.split("\t")

        if len(fields) != 7:
            continue

        (
            name,
            size,
            allocated,
            free,
            fragmentation,
            capacity,
            health,
        ) = fields

        pools.append({
            "name": name,
            "size_bytes": int(size),
            "allocated_bytes": int(allocated),
            "free_bytes": int(free),
            "fragmentation_percent": _optional_int(fragmentation),
            "capacity_percent": _optional_int(capacity),
            "health": health,
        })

    pools.sort(key=lambda pool: pool["name"])

    return {
        "host": host,
        "pools": pools,
    }


async def list_zfs_datasets() -> dict[str, Any]:
    """
    List ZFS filesystems and volumes with capacity and mount information.

    Upstream command:
    zfs list -Hp -t filesystem,volume -o name,type,used,available,referenced,mountpoint
    """
    host = _host()

    output = await _run(
        host,
        "zfs", "list", "-Hp", "-t", "filesystem,volume", 
        "-o", "name,type,used,available,referenced,mountpoint",
    )

    datasets = []

    for line in output.splitlines():
        fields = line.split("\t")

        if len(fields) != 6:
            continue

        (
            name,
            type_,
            used,
            available,
            referenced,
            mountpoint,
        ) = fields

        datasets.append({
            "name": name,
            "type": type_,
            "used_bytes": int(used),
            "available_bytes": int(available),
            "referenced_bytes": int(referenced),
            "mountpoint": (
                None
                if mountpoint == "-"
                else mountpoint
            ),
        })

    datasets.sort(key=lambda dataset: dataset["name"])

    return {
        "host": host,
        "datasets": datasets,
    }


async def get_zfs_properties(
    dataset: DatasetRef,
    properties: Annotated[
        list[ZFSPropertyRef] | None,
        Field(max_length=32),
    ] = None,
) -> dict[str, Any]:
    """
    Get properties for one ZFS dataset.

    Upstream command:
    zfs get -Hp -o property,value,source <properties> {dataset}

    Defaults to a curated set of configuration-relevant properties.
    Up to 32 explicit property names may be requested.
    """
    host = _host()

    selected = properties or list(_ZFS_PROPERTIES)

    if not selected:
        raise RuntimeError("At least one ZFS property is required")

    if "all" in selected:
        raise RuntimeError(
            "Request explicit ZFS properties instead of 'all'"
        )

    output = await _run(
        host,
        "zfs", "get", "-Hp", "-o", "property,value,source", ",".join(selected), dataset
    )

    properties = []

    for line in output.splitlines():
        fields = line.split("\t", 2)

        if len(fields) != 3:
            continue

        name, value, source = fields

        properties.append({
            "name": name,
            "value": redact_text(value),
            "source": source,
        })

    return {
        "host": host,
        "dataset": dataset,
        "properties": properties,
    }


async def list_zfs_snapshots(
    dataset: DatasetRef,
    recursive: bool = True,
    limit: Annotated[int, Field(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """
    List ZFS snapshots, newest first.

    Upstream command:
    zfs list -Hp -t snapshot -o name,creation,used,referenced -S creation -r {dataset}

    Results are bounded. Use a specific dataset to limit
    the upstream output.
    """
    host = _host()

    output = await _run(
        host,
        "zfs", "list",
        "-Hp",
        "-t", "snapshot",
        "-o", "name,creation,used,referenced",
        "-S", "creation",
        "-r", dataset,
    )

    snapshots = []
    has_more = False

    for line in output.splitlines():
        fields = line.split("\t")

        if len(fields) != 4:
            continue

        name, creation, used, referenced = fields

        parent = name.split("@", 1)[0]

        if not recursive and parent != dataset:
            continue

        if len(snapshots) >= limit:
            has_more = True
            break

        snapshots.append({
            "name": name,
            "dataset": parent,
            "created_at_epoch": int(creation),
            "used_bytes": int(used),
            "referenced_bytes": int(referenced),
        })

    return {
        "host": host,
        "dataset": dataset,
        "recursive": recursive,
        "limit": limit,
        "has_more": has_more,
        "snapshots": snapshots,
    }


TOOLS = (
    list_zfs_pools,
    list_zfs_datasets,
    get_zfs_properties,
    list_zfs_snapshots,
)
