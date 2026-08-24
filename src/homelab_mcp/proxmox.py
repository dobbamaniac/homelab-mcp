from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx


GuestKind = Literal["qemu", "lxc"]


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProxmoxNode:
    name: str
    status: str
    uptime_seconds: int
    cpu_usage_ratio: float
    cpu_count: int
    memory_used_bytes: int
    memory_total_bytes: int


@dataclass(frozen=True)
class ProxmoxNodes:
    nodes: list[ProxmoxNode]


@dataclass(frozen=True)
class ProxmoxNodeStatus:
    node: str
    status: dict[str, Any]


@dataclass(frozen=True)
class ProxmoxNodeNetwork:
    node: str
    interfaces: list[dict[str, Any]]


@dataclass(frozen=True)
class ProxmoxGuest:
    vmid: int
    name: str
    kind: GuestKind
    node: str
    status: str
    uptime_seconds: int
    cpu_usage_ratio: float
    cpu_count: int
    memory_used_bytes: int
    memory_total_bytes: int
    template: bool


@dataclass(frozen=True)
class ProxmoxGuests:
    guests: list[ProxmoxGuest]


@dataclass(frozen=True)
class ProxmoxGuestConfig:
    vmid: int
    name: str
    kind: GuestKind
    node: str
    config: dict[str, Any]


@dataclass(frozen=True)
class ProxmoxStorage:
    storage_id: str
    storage_type: str
    config: dict[str, Any]
    status_by_node: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ProxmoxStorages:
    storages: list[ProxmoxStorage]


@dataclass(frozen=True)
class ProxmoxStorageHealth:
    node: str
    disks: list[dict[str, Any]]
    zpools: list[dict[str, Any]]
    zpool_details: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ProxmoxBackupStatus:
    jobs: list[dict[str, Any]]
    not_backed_up: list[dict[str, Any]]
    recent_tasks_by_node: dict[str, list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Internal API helpers
# ---------------------------------------------------------------------------


def _base_url() -> str:
    value = os.environ.get("PROXMOX_URL")

    if not value:
        raise RuntimeError("PROXMOX_URL is not configured")

    if not value.startswith("https://"):
        raise RuntimeError("PROXMOX_URL must use HTTPS")

    return value.rstrip("/")


def _api_token() -> str:
    value = os.environ.get("PROXMOX_API_TOKEN")

    if not value:
        raise RuntimeError("PROXMOX_API_TOKEN is not configured")

    return value


def _tls_context() -> ssl.SSLContext:
    ca_cert = os.environ.get("PROXMOX_CA_CERT")

    if ca_cert:
        return ssl.create_default_context(cafile=ca_cert)

    return ssl.create_default_context()


async def _get_data(
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> Any:
    async with httpx.AsyncClient(
        base_url=f"{_base_url()}/api2/json/",
        headers={"Authorization": f"PVEAPIToken={_api_token()}"},
        verify=_tls_context(),
        timeout=10.0,
    ) as client:
        try:
            response = await client.get(path, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Proxmox API returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Unable to reach Proxmox API: {exc}") from exc

    payload = response.json()

    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Proxmox API returned an unexpected response")

    return payload["data"]


async def _get_list(
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    data = await _get_data(path, params=params)

    if not isinstance(data, list):
        raise RuntimeError("Proxmox API returned an unexpected list response")

    return data


async def _get_object(
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = await _get_data(path, params=params)

    if not isinstance(data, dict):
        raise RuntimeError("Proxmox API returned an unexpected object response")

    return data


async def _find_guest(vmid: int) -> ProxmoxGuest:
    guests = await list_proxmox_guests()

    for guest in guests.guests:
        if guest.vmid == vmid:
            return guest

    raise RuntimeError(f"Proxmox guest {vmid} was not found")


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


async def list_proxmox_nodes() -> ProxmoxNodes:
    """
    List Proxmox VE nodes and their current resource utilization.

    Returns structured, read-only status information from the configured
    Proxmox VE API.
    """
    data = await _get_list("nodes")

    nodes = [
        ProxmoxNode(
            name=str(item["node"]),
            status=str(item.get("status", "unknown")),
            uptime_seconds=int(item.get("uptime", 0)),
            cpu_usage_ratio=float(item.get("cpu", 0.0)),
            cpu_count=int(item.get("maxcpu", 0)),
            memory_used_bytes=int(item.get("mem", 0)),
            memory_total_bytes=int(item.get("maxmem", 0)),
        )
        for item in data
    ]

    nodes.sort(key=lambda node: node.name)

    return ProxmoxNodes(nodes=nodes)


async def list_proxmox_guests() -> ProxmoxGuests:
    """
    List Proxmox VE virtual machines and LXC containers.

    Returns structured, read-only status and resource utilization from
    the configured Proxmox VE API.
    """
    data = await _get_list("cluster/resources", params={"type": "vm"})

    guests: list[ProxmoxGuest] = []

    for item in data:
        raw_kind = str(item.get("type", ""))

        if raw_kind not in ("qemu", "lxc"):
            continue

        guests.append(
            ProxmoxGuest(
                vmid=int(item["vmid"]),
                name=str(item.get("name", "")),
                kind=cast(GuestKind, raw_kind),
                node=str(item.get("node", "")),
                status=str(item.get("status", "unknown")),
                uptime_seconds=int(item.get("uptime", 0)),
                cpu_usage_ratio=float(item.get("cpu", 0.0)),
                cpu_count=int(item.get("maxcpu", 0)),
                memory_used_bytes=int(item.get("mem", 0)),
                memory_total_bytes=int(item.get("maxmem", 0)),
                template=bool(item.get("template", 0)),
            )
        )

    guests.sort(key=lambda guest: guest.vmid)

    return ProxmoxGuests(guests=guests)


async def get_proxmox_guest_config(vmid: int) -> ProxmoxGuestConfig:
    """
    Get the full Proxmox VE configuration for one VM or LXC container.

    The guest's node and type are resolved automatically from cluster
    resources before its read-only configuration endpoint is queried.
    """
    guest = await _find_guest(vmid)

    config = await _get_object(
        f"nodes/{guest.node}/{guest.kind}/{guest.vmid}/config"
    )

    return ProxmoxGuestConfig(
        vmid=guest.vmid,
        name=guest.name,
        kind=guest.kind,
        node=guest.node,
        config=config,
    )


async def list_proxmox_storage() -> ProxmoxStorages:
    """
    List configured Proxmox VE storage and its current per-node status.

    Combines cluster storage configuration with runtime storage status
    from each online Proxmox VE node.
    """
    config_data = await _get_list("storage")
    node_data = await _get_list("nodes")

    status_by_storage: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for node_item in node_data:
        if str(node_item.get("status", "")) != "online":
            continue

        node = str(node_item["node"])
        status_data = await _get_list(f"nodes/{node}/storage")

        for status in status_data:
            storage_id = str(status.get("storage", ""))

            if not storage_id:
                continue

            status_by_storage.setdefault(
                storage_id,
                {},
            )[node] = status

    storages = [
        ProxmoxStorage(
            storage_id=str(item["storage"]),
            storage_type=str(item.get("type", "")),
            config=item,
            status_by_node=status_by_storage.get(str(item["storage"]),{}),
        )
        for item in config_data
    ]

    storages.sort(key=lambda storage: storage.storage_id)

    return ProxmoxStorages(storages=storages)


async def get_proxmox_node_network(node: str) -> ProxmoxNodeNetwork:
    """
    Get the Proxmox VE network configuration for one node.

    Returns physical interfaces, bridges, bonds, VLAN interfaces, and
    other interfaces visible to the configured API token.
    """
    interfaces = await _get_list(f"nodes/{node}/network")

    interfaces.sort(key=lambda interface: str(interface.get("iface", "")))

    return ProxmoxNodeNetwork(
        node=node,
        interfaces=interfaces,
    )


async def get_proxmox_node_status(node: str) -> ProxmoxNodeStatus:
    """
    Get detailed runtime and system status for one Proxmox VE node.

    Includes the status information exposed by Proxmox for CPU, memory,
    swap, load, uptime, kernel, PVE version, and related host metrics.
    """
    status = await _get_object(f"nodes/{node}/status")

    return ProxmoxNodeStatus(
        node=node,
        status=status,
    )


async def get_proxmox_storage_health(node: str) -> ProxmoxStorageHealth:
    """
    Get physical disk and ZFS health information for one Proxmox VE node.

    Combines local disk inventory with ZFS pool summaries and detailed
    status for each detected ZFS pool.
    """
    disks = await _get_list(
        f"nodes/{node}/disks/list",
        params={"include-partitions": "0"},
    )

    zpools = await _get_list(
        f"nodes/{node}/disks/zfs",
    )

    zpool_details: dict[str, dict[str, Any]] = {}

    for pool in zpools:
        name = str(pool.get("name", ""))

        if not name:
            continue

        zpool_details[name] = await _get_object(
            f"nodes/{node}/disks/zfs/{name}"
        )

    disks.sort(
        key=lambda disk: str(
            disk.get(
                "devpath",
                disk.get("name", ""),
            )
        )
    )

    zpools.sort(
        key=lambda pool: str(pool.get("name", ""))
    )

    return ProxmoxStorageHealth(
        node=node,
        disks=disks,
        zpools=zpools,
        zpool_details=zpool_details,
    )


async def get_proxmox_backup_status() -> ProxmoxBackupStatus:
    """
    Get configured Proxmox VE backup jobs, guest backup coverage, and
    recent vzdump task history from online nodes.

    This is a read-only overview intended for backup auditing.
    """
    jobs = await _get_list("cluster/backup")

    not_backed_up = await _get_list(
        "cluster/backup-info/not-backed-up"
    )

    node_data = await _get_list("nodes")

    recent_tasks_by_node: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for node_item in node_data:
        if str(node_item.get("status", "")) != "online":
            continue

        node = str(node_item["node"])

        tasks = await _get_list(
            f"nodes/{node}/tasks",
            params={
                "typefilter": "vzdump",
                "limit": "50",
            },
        )

        recent_tasks_by_node[node] = tasks

    jobs.sort(
        key=lambda job: str(job.get("id", ""))
    )

    not_backed_up.sort(
        key=lambda guest: int(guest.get("vmid", 0))
    )

    return ProxmoxBackupStatus(
        jobs=jobs,
        not_backed_up=not_backed_up,
        recent_tasks_by_node=recent_tasks_by_node,
    )
