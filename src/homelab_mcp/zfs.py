from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

from homelab_mcp.redact import redact_text
from homelab_mcp.utils import require_env
from homelab_mcp.ssh import host_names as _ssh_hosts, run as _run


# -------------------------------------------------------------------------------------------------
# Structured output
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ZFSPool:
    name: str
    size_bytes: int
    allocated_bytes: int
    free_bytes: int
    fragmentation_percent: int | None
    capacity_percent: int | None
    health: str


@dataclass(frozen=True)
class ZFSPools:
    host: str
    pools: list[ZFSPool]


@dataclass(frozen=True)
class ZFSDataset:
    name: str
    type: str
    used_bytes: int
    available_bytes: int
    referenced_bytes: int
    mountpoint: str | None


@dataclass(frozen=True)
class ZFSDatasets:
    host: str
    datasets: list[ZFSDataset]


@dataclass(frozen=True)
class ZFSProperty:
    name: str
    value: str
    source: str


@dataclass(frozen=True)
class ZFSProperties:
    host: str
    dataset: str
    properties: list[ZFSProperty]


# -------------------------------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------------------------------

DatasetRef = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$"),
]

_ZFS_PROPERTIES = (
    "mountpoint", "canmount", "compression", "compressratio", "recordsize", "atime", "relatime", 
    "xattr", "acltype", "aclinherit", "sync", "dedup", "copies", "quota", "reservation", 
    "encryption", "keyformat", "keylocation", "keystatus"
)


def _host() -> str:
    return require_env("ZFS_HOSTS")


def _optional_int(value: str) -> int | None:
    return None if value == "-" else int(value)


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """ZFS uses the configured Linux SSH transport."""
    host = _host()

    if host not in _ssh_hosts():
        raise RuntimeError(f"ZFS host {host!r} is not configured for SSH")

    yield


# -------------------------------------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------------------------------------


async def list_zfs_pools() -> ZFSPools:
    """List ZFS pools and their current capacity and health."""
    host = _host()

    output = await _run(host, "zpool", "list", "-Hp", "-o", "name,size,alloc,free,frag,cap,health")

    pools: list[ZFSPool] = []

    for line in output.splitlines():
        fields = line.split("\t")

        if len(fields) != 7:
            continue

        name, size, allocated, free, fragmentation, capacity, health = fields

        pools.append(
            ZFSPool(
                name=name,
                size_bytes=int(size),
                allocated_bytes=int(allocated),
                free_bytes=int(free),
                fragmentation_percent=_optional_int(fragmentation),
                capacity_percent=_optional_int(capacity),
                health=health,
            )
        )

    pools.sort(key=lambda pool: pool.name)

    return ZFSPools(
        host=host,
        pools=pools,
    )


async def list_zfs_datasets() -> ZFSDatasets:
    """List ZFS filesystems and volumes with capacity and mount information."""
    host = _host()

    output = await _run(
        host, "zfs", "list", "-Hp", "-t", "filesystem,volume", 
        "-o", "name,type,used,available,referenced,mountpoint"
    )

    datasets: list[ZFSDataset] = []

    for line in output.splitlines():
        fields = line.split("\t")

        if len(fields) != 6:
            continue

        name, type_, used, available, referenced, mountpoint = fields

        datasets.append(
            ZFSDataset(
                name=name,
                type=type_,
                used_bytes=int(used),
                available_bytes=int(available),
                referenced_bytes=int(referenced),
                mountpoint=None if mountpoint == "-" else mountpoint,
            )
        )

    datasets.sort(key=lambda dataset: dataset.name)

    return ZFSDatasets(
        host=host,
        datasets=datasets,
    )


async def get_zfs_properties(
    dataset: DatasetRef,
) -> ZFSProperties:
    """Get configuration-relevant properties for one ZFS dataset."""
    host = _host()

    output = await _run(
        host, "zfs", "get", "-Hp", "-o", "property,value,source", 
        ",".join(_ZFS_PROPERTIES), dataset
    )

    properties: list[ZFSProperty] = []

    for line in output.splitlines():
        fields = line.split("\t", 2)

        if len(fields) != 3:
            continue

        name, value, source = fields

        properties.append(
            ZFSProperty(
                name=name,
                value=redact_text(value),
                source=source,
            )
        )

    return ZFSProperties(
        host=host,
        dataset=dataset,
        properties=properties,
    )


TOOLS = (
    list_zfs_pools,
    list_zfs_datasets,
    get_zfs_properties,
)
