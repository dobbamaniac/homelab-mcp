from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

import httpx
from pydantic import Field

from homelab_mcp.http import get_json
from homelab_mcp.redact import redact_value
from homelab_mcp.utils import pick, pick_all, require_env


# ---------------------------------------------------------------------------
# Tool parameter types
# ---------------------------------------------------------------------------

GuestKind = Annotated[
    Literal["qemu", "lxc"],
    Field(description=("Guest type from the 'type' field returned by list_proxmox_guests.")),
]

MappingKind = Literal["pci", "usb", "dir"]

NodeRef = Annotated[
    str,
    Field(description="Proxmox VE node name returned by list_proxmox_nodes."),
]

VMIDRef = Annotated[
    int,
    Field(ge=0, description="VM or LXC VMID returned by list_proxmox_guests."),
]

TaskUPID = Annotated[
    str,
    Field(description="Exact Proxmox task UPID returned by list_proxmox_tasks."),
]

ZPoolRef = Annotated[
    str,
    Field(description="ZFS pool name returned by list_proxmox_zfs_pools."),
]

SecurityGroupRef = Annotated[
    str,
    Field(description="Security-group name returned by list_proxmox_security_groups."),
]


# ---------------------------------------------------------------------------
# Internal API helpers
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Open the shared Proxmox API client for the server lifetime."""
    global _client

    if _client is not None:
        raise RuntimeError("Proxmox API client is already initialized")

    base_url = require_env("PROXMOX_URL")

    if not base_url.startswith("https://"):
        raise RuntimeError("PROXMOX_URL must use HTTPS")

    try:
        async with httpx.AsyncClient(
            base_url=f"{base_url.rstrip('/')}/api2/json",
            headers={"Authorization": (f"PVEAPIToken={require_env('PROXMOX_API_TOKEN')}")},
            timeout=10.0,
        ) as _client:
            yield

    finally:
        _client = None


async def _get(
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> Any:
    if _client is None:
        raise RuntimeError("Proxmox API client is not initialized")

    data = await get_json(
        _client,
        path,
        params=params,
        source="Proxmox API",
        key="data",
    )

    return redact_value(data)


# ---------------------------------------------------------------------------
# Inventory and guest configuration
# ---------------------------------------------------------------------------

async def list_proxmox_nodes() -> dict[str, Any]:
    """
    List Proxmox VE nodes for inventory/discovery.

    Upstream: Proxmox VE API
    GET /api2/json/nodes

    Resource-utilization fields are omitted; use Grafana for metrics.
    """
    data = await _get("nodes")

    return {"nodes": pick_all(
        data, 
        "node", "status", "maxcpu", "maxmem"
    )}


async def list_proxmox_guests() -> dict[str, Any]:
    """
    List Proxmox VE virtual machines and LXC containers.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/resources?type=vm

    Resource-utilization fields are omitted; use Grafana for metrics.
    """
    data = await _get(
        "cluster/resources",
        params={"type": "vm"},
    )

    return {"guests": pick_all(
        data, 
        "vmid", "name", "type", "node", "status", "maxcpu", "maxmem", "template"
    )}


async def get_proxmox_guest_config(
    node: NodeRef,
    kind: GuestKind,
    vmid: VMIDRef,
) -> dict[str, Any]:
    """
    Get configuration for one VM or LXC container.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/{kind}/{vmid}/config

    The guest configuration schema varies substantially between QEMU and LXC,
    so the redacted upstream object is returned without a parallel local schema.
    """
    return await _get(f"nodes/{node}/{kind}/{vmid}/config")


# ---------------------------------------------------------------------------
# Storage and host topology
# ---------------------------------------------------------------------------

async def list_proxmox_storage() -> dict[str, Any]:
    """
    List cluster storage configuration.

    Upstream: Proxmox VE API
    GET /api2/json/storage
    """
    data = await _get("storage")

    return {"storage": pick_all(
        data, 
        "storage", "type", "content", "path", "pool", "sparse", "shared", "disable", "nodes", 
        "server", "export", "username", "namespace", "mountpoint", "options", "prune-backups"
    )}


async def list_proxmox_node_storage(
    node: NodeRef,
) -> dict[str, Any]:
    """
    List storage status visible from one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/storage
    """
    data = await _get(f"nodes/{node}/storage")

    return {"storage": pick_all(
        data,
        "storage", "type", "content", "active", "enabled", "shared", "total", "used", "avail", 
        "used_fraction"
    )}


async def get_proxmox_node_network(
    node: NodeRef,
) -> dict[str, Any]:
    """
    Get network-interface configuration for one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/network
    """
    data = await _get(f"nodes/{node}/network")

    return {"interfaces": pick_all(
        data,
        "iface", "type", "active", "exists", "autostart", "method", "method6", "address", 
        "netmask", "cidr", "gateway", "mtu", "bridge_ports", "bridge_stp", "bridge_fd", 
        "bond-slaves", "bond_mode", "vlan-id", "vlan-raw-device", "families", "altnames", 
        "priority", "comments"
    )}


async def list_proxmox_disks(
    node: NodeRef,
) -> dict[str, Any]:
    """
    List physical disks on one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/disks/list?include-partitions=0
    """
    data = await _get(
        f"nodes/{node}/disks/list",
        params={"include-partitions": "0"},
    )

    return {"disks": pick_all(
        data, 
        "devpath", "by_id_link", "type", "model", "serial", "size", "health", "wearout", "used", 
        "rpm"
    )}


async def list_proxmox_zfs_pools(
    node: NodeRef,
) -> dict[str, Any]:
    """
    List ZFS pools on one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/disks/zfs
    """
    data = await _get(f"nodes/{node}/disks/zfs")

    return {"pools": pick_all(
        data, 
        "name", "health", "size", "alloc", "free", "frag"
    )}


async def get_proxmox_zfs_pool(
    node: NodeRef,
    pool: ZPoolRef,
) -> dict[str, Any]:
    """
    Get detailed status for one Proxmox ZFS pool.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/disks/zfs/{pool}

    The recursive vdev tree is returned in the upstream shape after redaction.
    """
    return await _get(f"nodes/{node}/disks/zfs/{pool}")


# ---------------------------------------------------------------------------
# Backup configuration
# ---------------------------------------------------------------------------

async def list_proxmox_backup_jobs() -> dict[str, Any]:
    """
    List configured Proxmox backup jobs.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/backup
    """
    data = await _get("cluster/backup")

    return {"jobs": pick_all(
        data, 
        "id", "type", "enabled", "mode", "storage", "vmid", "schedule", "compress", 
        "notes-template", "notification-mode", "next-run", "prune-backups", "fleecing"
    )}


async def list_proxmox_unbacked_guests() -> dict[str, Any]:
    """
    List guests not covered by a configured backup job.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/backup-info/not-backed-up
    """
    data = await _get("cluster/backup-info/not-backed-up")

    return {"guests": pick_all(data, "vmid", "name", "type")}


# ---------------------------------------------------------------------------
# Firewall
# ---------------------------------------------------------------------------

async def get_proxmox_cluster_firewall_options() -> dict[str, Any]:
    """
    Get datacenter firewall options.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/firewall/options
    """
    return await _get("cluster/firewall/options")


async def list_proxmox_cluster_firewall_rules() -> dict[str, Any]:
    """
    List datacenter firewall rules.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/firewall/rules
    """
    data = await _get("cluster/firewall/rules")

    return {"rules": pick_all(
        data, 
        "pos", "type", "action", "enable", "log", "comment", "ipversion", "proto", "source", 
        "sport", "dest", "dport", "iface", "macro"
    )}


async def list_proxmox_cluster_firewall_aliases() -> dict[str, Any]:
    """
    List datacenter firewall aliases.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/firewall/aliases
    """
    data = await _get("cluster/firewall/aliases")

    return {"aliases": pick_all(
        data,
        "name", "cidr", "comment", "ipversion"
    )}


async def list_proxmox_security_groups() -> dict[str, Any]:
    """
    List Proxmox firewall security groups.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/firewall/groups
    """
    data = await _get("cluster/firewall/groups")

    return {"groups": pick_all(data, "group", "comment")}


async def list_proxmox_security_group_rules(
    group: SecurityGroupRef,
) -> dict[str, Any]:
    """
    List rules in one Proxmox firewall security group.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/firewall/groups/{group}
    """
    data = await _get(
        f"cluster/firewall/groups/{group}"
    )

    return {"rules": pick_all(
        data,
        "pos", "type", "action", "enable", "log", "comment", "ipversion", "proto", "source", 
        "sport", "dest", "dport", "iface", "macro"
    )}


async def get_proxmox_node_firewall_options(
    node: NodeRef,
) -> dict[str, Any]:
    """
    Get firewall options for one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/firewall/options
    """
    return await _get(f"nodes/{node}/firewall/options")


async def list_proxmox_node_firewall_rules(
    node: NodeRef,
) -> dict[str, Any]:
    """
    List firewall rules for one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/firewall/rules
    """
    data = await _get(f"nodes/{node}/firewall/rules")

    return {"rules": pick_all(
        data,
        "pos", "type", "action", "enable", "log", "comment", "ipversion", "proto", "source", 
        "sport", "dest", "dport", "iface", "macro"
    )}


async def get_proxmox_guest_firewall_options(
    node: NodeRef,
    kind: GuestKind,
    vmid: VMIDRef,
) -> dict[str, Any]:
    """
    Get firewall options for one VM or LXC container.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/{kind}/{vmid}/firewall/options
    """
    return await _get(f"nodes/{node}/{kind}/{vmid}/firewall/options")


async def list_proxmox_guest_firewall_rules(
    node: NodeRef,
    kind: GuestKind,
    vmid: VMIDRef,
) -> dict[str, Any]:
    """
    List firewall rules for one VM or LXC container.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/{kind}/{vmid}/firewall/rules
    """
    data = await _get(
        f"nodes/{node}/{kind}/{vmid}/firewall/rules"
    )

    return {"rules": pick_all(
        data,
        "pos", "type", "action", "enable", "log", "comment", "ipversion", "proto", "source", 
        "sport", "dest", "dport", "iface", "macro"
    )}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

async def list_proxmox_tasks(
    node: NodeRef,
    typefilter: Annotated[
        str | None,
        Field(
            description=("Optional Proxmox task type filter, for example 'vzdump' or 'aptupdate'.")
        ),
    ] = None,
    limit: Annotated[int, Field(ge=1, le=200)] = 20,
) -> dict[str, Any]:
    """
    List recent tasks on one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/tasks
    """
    params = {"limit": str(limit)}

    if typefilter:
        params["typefilter"] = typefilter

    data = await _get(f"nodes/{node}/tasks", params=params,)

    return {"tasks": pick_all(
        data, 
        "upid", "type", "id", "user", "pid", "pstart", "starttime", "endtime", "status"
    )}


async def get_proxmox_task_status(
    node: NodeRef,
    upid: TaskUPID,
) -> dict[str, Any]:
    """
    Get status for one Proxmox task.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/tasks/{upid}/status
    """
    data = await _get(f"nodes/{node}/tasks/{upid}/status")

    return pick(
        data, 
        "upid", "type", "id", "user", "pid", "pstart", "starttime", "endtime", "status", 
        "exitstatus"
    )


async def get_proxmox_task_log(
    node: NodeRef,
    upid: TaskUPID,
    start: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[int, Field(ge=1, le=1000)] = 200,
) -> dict[str, Any]:
    """
    Get log lines for one Proxmox task.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/tasks/{upid}/log
    """
    data = await _get(
        f"nodes/{node}/tasks/{upid}/log",
        params={
            "start": str(start),
            "limit": str(limit),
        },
    )

    return {"log": pick_all(data, "n", "t")}


# ---------------------------------------------------------------------------
# Guest runtime/configuration state
# ---------------------------------------------------------------------------

async def get_proxmox_guest_status(
    node: NodeRef,
    kind: GuestKind,
    vmid: VMIDRef,
) -> dict[str, Any]:
    """
    Get current lifecycle state for one VM or LXC container.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/{kind}/{vmid}/status/current

    High-frequency utilization counters are omitted; use Grafana for metrics.
    """
    data = await _get(f"nodes/{node}/{kind}/{vmid}/status/current")

    return pick(
        data, 
        "status", "qmpstatus", "pid", "uptime", "cpus", "maxmem", "maxdisk", "ha", "lock"
    )


async def get_proxmox_guest_pending(
    node: NodeRef,
    kind: GuestKind,
    vmid: VMIDRef,
) -> dict[str, Any]:
    """
    Get pending configuration changes for one VM or LXC container.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/{kind}/{vmid}/pending
    """
    data = await _get(f"nodes/{node}/{kind}/{vmid}/pending")

    return { "changes": pick_all(
        data,
        "key", "value", "pending", "delete"
    )}


async def list_proxmox_guest_snapshots(
    node: NodeRef,
    kind: GuestKind,
    vmid: VMIDRef,
) -> dict[str, Any]:
    """
    List snapshots for one VM or LXC container.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/{kind}/{vmid}/snapshot
    """
    data = await _get(f"nodes/{node}/{kind}/{vmid}/snapshot")

    return {"snapshots": pick_all(
        data,
        "name", "description", "parent", "snaptime", "vmstate", "running"
    )}


# ---------------------------------------------------------------------------
# Node software
# ---------------------------------------------------------------------------

async def get_proxmox_repositories(
    node: NodeRef,
) -> dict[str, Any]:
    """
    Get configured APT repositories for one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/apt/repositories

    Only the repository definitions and reported errors are returned.
    """
    data = await _get(f"nodes/{node}/apt/repositories")

    return pick(
        data,
        "standard-repos", "files", "errors"
    )


async def list_proxmox_package_versions(
    node: NodeRef,
) -> dict[str, Any]:
    """
    List installed and available Proxmox package versions.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/apt/versions
    """
    data = await _get(
        f"nodes/{node}/apt/versions"
    )

    return {"packages": pick_all(
        data,
        "Package", "OldVersion", "Version", "Origin", "Arch", "ManagerVersion", "RunningKernel"
    )}


async def get_proxmox_subscription(
    node: NodeRef,
) -> dict[str, Any]:
    """
    Get subscription status for one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/subscription
    """
    data = await _get(f"nodes/{node}/subscription")

    return pick(
        data,
        "status", "level", "productname", "regdate", "nextduedate", "message", "sockets"
    )


async def list_proxmox_certificates(
    node: NodeRef,
) -> dict[str, Any]:
    """
    List certificate information for one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/certificates/info
    """
    data = await _get(f"nodes/{node}/certificates/info")

    return {"certificates": pick_all(
        data,
        "filename", "subject", "issuer", "san", "fingerprint", "notbefore", "notafter", 
        "public-key-type", "public-key-bits"
    )}


# ---------------------------------------------------------------------------
# Hardware and mappings
# ---------------------------------------------------------------------------

async def list_proxmox_pci_devices(
    node: NodeRef,
) -> dict[str, Any]:
    """
    List PCI devices on one Proxmox node.

    Upstream: Proxmox VE API
    GET /api2/json/nodes/{node}/hardware/pci
    """
    data = await _get(f"nodes/{node}/hardware/pci")

    return {"devices": pick_all(
        data,
        "id", "class", "vendor", "vendor_name", "device", "device_name", "subsystem_vendor", 
        "subsystem_vendor_name", "subsystem_device", "subsystem_device_name", "iommugroup"
    )}


async def list_proxmox_mappings(
    kind: MappingKind,
) -> dict[str, Any]:
    """
    List Proxmox cluster resource mappings of one kind.

    Upstream: Proxmox VE API
    GET /api2/json/cluster/mapping/{kind}

    `kind` is one of: pci, usb, dir.
    """
    return {"mappings": await _get(f"cluster/mapping/{kind}")}


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

async def get_proxmox_permissions() -> dict[str, Any]:
    """
    Get effective permissions visible to the configured API token.

    Upstream: Proxmox VE API
    GET /api2/json/access/permissions
    """
    return await _get("access/permissions")


async def list_proxmox_acls() -> dict[str, Any]:
    """
    List Proxmox access-control ACL entries.

    Upstream: Proxmox VE API
    GET /api2/json/access/acl
    """
    data = await _get("access/acl")

    return {"acls": pick_all(
        data,
        "path", "roleid", "ugid", "type", "propagate"
    )}


async def list_proxmox_roles() -> dict[str, Any]:
    """
    List Proxmox access-control roles.

    Upstream: Proxmox VE API
    GET /api2/json/access/roles
    """
    data = await _get("access/roles")

    return {"roles": pick_all(
        data, 
        "roleid", "privs", "special"
    )}


async def list_proxmox_groups() -> dict[str, Any]:
    """
    List Proxmox access-control groups.

    Upstream: Proxmox VE API
    GET /api2/json/access/groups
    """
    data = await _get("access/groups")

    return {"groups": pick_all(
        data, 
        "groupid", "users", "comment"
    )}


async def list_proxmox_realms() -> dict[str, Any]:
    """
    List Proxmox authentication realms.

    Upstream: Proxmox VE API
    GET /api2/json/access/domains
    """
    data = await _get("access/domains")

    return {"realms": pick_all(
        data, 
        "realm", "type", "comment"
    )}


async def list_proxmox_users() -> dict[str, Any]:
    """
    List Proxmox users visible to the configured API token.

    Upstream: Proxmox VE API
    GET /api2/json/access/users
    """
    data = await _get("access/users")

    return {"users": pick_all(
        data, 
        "userid", "enable", "firstname", "lastname", "email", "groups"
    )}


TOOLS = (
    list_proxmox_nodes,
    list_proxmox_guests,
    get_proxmox_guest_config,
    list_proxmox_storage,
    list_proxmox_node_storage,
    get_proxmox_node_network,
    list_proxmox_disks,
    list_proxmox_zfs_pools,
    get_proxmox_zfs_pool,
    list_proxmox_backup_jobs,
    list_proxmox_unbacked_guests,
    get_proxmox_cluster_firewall_options,
    list_proxmox_cluster_firewall_rules,
    list_proxmox_cluster_firewall_aliases,
    list_proxmox_security_groups,
    list_proxmox_security_group_rules,
    get_proxmox_node_firewall_options,
    list_proxmox_node_firewall_rules,
    get_proxmox_guest_firewall_options,
    list_proxmox_guest_firewall_rules,
    list_proxmox_tasks,
    get_proxmox_task_status,
    get_proxmox_task_log,
    get_proxmox_guest_status,
    get_proxmox_guest_pending,
    list_proxmox_guest_snapshots,
    get_proxmox_repositories,
    list_proxmox_package_versions,
    get_proxmox_subscription,
    list_proxmox_certificates,
    list_proxmox_pci_devices,
    list_proxmox_mappings,
    get_proxmox_permissions,
    list_proxmox_acls,
    list_proxmox_roles,
    list_proxmox_groups,
    list_proxmox_realms,
    list_proxmox_users,
)
