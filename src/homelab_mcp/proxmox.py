from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


GuestKind = Literal["qemu", "lxc"]


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
    version: dict[str, Any]


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


@dataclass(frozen=True)
class ProxmoxFirewall:
    datacenter: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    guests: list[dict[str, Any]]


@dataclass(frozen=True)
class ProxmoxTasks:
    tasks_by_node: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class ProxmoxTask:
    node: str
    upid: str
    status: dict[str, Any]
    log: list[dict[str, Any]]


@dataclass(frozen=True)
class ProxmoxGuestRuntime:
    vmid: int
    name: str
    kind: GuestKind
    node: str
    current: dict[str, Any]
    pending: list[dict[str, Any]]
    snapshots: list[dict[str, Any]]
    agent: dict[str, Any] | None


@dataclass(frozen=True)
class ProxmoxNodeSoftware:
    node: str
    running_kernel: str
    repositories: dict[str, Any]
    versions: list[dict[str, Any]]
    subscription: dict[str, Any]
    certificates: list[dict[str, Any]]


@dataclass(frozen=True)
class ProxmoxHardware:
    node: str
    pci_devices: list[dict[str, Any]]
    usb_devices: list[dict[str, Any]]
    usb_error: str | None
    mappings: dict[str, Any]


@dataclass(frozen=True)
class ProxmoxAccessControl:
    effective_permissions: dict[str, Any]
    acls: list[dict[str, Any]]
    roles: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    realms: list[dict[str, Any]]
    users: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Internal API helpers
# ---------------------------------------------------------------------------

class ProxmoxRequestError(RuntimeError):
    pass


_client: httpx.AsyncClient | None = None


def _create_client() -> httpx.AsyncClient:
    base_url = os.environ.get("PROXMOX_URL")
    api_token = os.environ.get("PROXMOX_API_TOKEN")

    if not base_url:
        raise RuntimeError("PROXMOX_URL is not configured")

    if not base_url.startswith("https://"):
        raise RuntimeError("PROXMOX_URL must use HTTPS")

    if not api_token:
        raise RuntimeError("PROXMOX_API_TOKEN is not configured")

    return httpx.AsyncClient(
        base_url=f"{base_url.rstrip('/')}/api2/json/",
        headers={"Authorization": f"PVEAPIToken={api_token}"},
        timeout=10.0,
    )


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Open the shared Proxmox API client for the server lifetime."""
    global _client

    if _client is not None:
        raise RuntimeError("Proxmox API client is already initialized")

    async with _create_client() as client:
        _client = client

        try:
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

    try:
        response = await _client.get(path, params=params)
        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        raise ProxmoxRequestError(
            f"Proxmox API returned HTTP {exc.response.status_code} for GET {path}"
        ) from exc

    except httpx.RequestError as exc:
        raise ProxmoxRequestError(
            f"Unable to reach Proxmox API for GET {path}: {exc}"
        ) from exc

    try:
        payload = response.json()

    except ValueError as exc:
        raise ProxmoxRequestError(
            f"Proxmox API returned invalid JSON for GET {path}"
        ) from exc

    if not isinstance(payload, dict) or "data" not in payload:
        raise ProxmoxRequestError(f"Proxmox API returned an unexpected response for GET {path}")

    return payload["data"]


async def _try_get(
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        data = await _get(path, params=params)

    except ProxmoxRequestError as exc:
        return {
            "ok": False,
            "error": str(exc),
        }

    return {
        "ok": True,
        "data": data,
    }


async def _list_online_nodes() -> list[str]:
    data = await _get("nodes")

    return sorted(
        str(item["node"])
        for item in data
        if str(item.get("status", "")) == "online"
    )


async def _find_guest(vmid: int) -> ProxmoxGuest:
    guests = await list_proxmox_guests()

    for guest in guests.guests:
        if guest.vmid == vmid:
            return guest

    raise RuntimeError(f"Proxmox guest {vmid} was not found")


async def _get_firewall_ipsets(base_path: str) -> dict[str, Any]:
    ipsets = await _get(f"{base_path}/ipset")

    entries_by_name: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for ipset in ipsets:
        name = str(ipset.get("name", ""))

        if not name:
            continue

        entries_by_name[name] = await _get(f"{base_path}/ipset/{name}")

    ipsets.sort(key=lambda ipset: str(ipset.get("name", "")))

    return {
        "ipsets": ipsets,
        "entries_by_name": entries_by_name,
    }


def _remove_key(value: Any, key: str) -> None:
    if isinstance(value, dict):
        value.pop(key, None)

        for child in value.values():
            _remove_key(child, key)

    elif isinstance(value, list):
        for child in value:
            _remove_key(child, key)


def _pick(
    item: dict[str, Any],
    *keys: str,
) -> dict[str, Any]:
    return {key: item[key] for key in keys if key in item}


def _drop(
    item: dict[str, Any],
    *keys: str,
) -> None:
    for key in keys:
        item.pop(key, None)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


async def list_proxmox_nodes() -> ProxmoxNodes:
    """
    List Proxmox VE nodes and their current resource utilization.

    Returns structured, read-only status information from the configured Proxmox VE API.
    """
    data = await _get("nodes")

    nodes = [
        ProxmoxNode(
            name=str(item["node"]),
            status=str(item.get("status", "unknown")),
            uptime_seconds=int(item.get("uptime", 0)),
            cpu_usage_ratio=float(item.get("cpu", 0.0)),
            cpu_count=int(item.get("maxcpu", 0)),
            memory_used_bytes=int(item.get("mem", 0)),
            memory_total_bytes=int(item.get("maxmem", 0)),
        ) for item in data
    ]

    nodes.sort(key=lambda node: node.name)

    return ProxmoxNodes(nodes=nodes)


async def list_proxmox_guests() -> ProxmoxGuests:
    """
    List Proxmox VE virtual machines and LXC containers.

    Returns structured, read-only status and resource utilization from the configured Proxmox VE 
    API.
    """
    data = await _get("cluster/resources", params={"type": "vm"})

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
    Get the relevant Proxmox VE configuration for one VM or LXC container.

    Generated identifiers, credentials, SSH keys, and other low-value fields are omitted.
    """
    guest = await _find_guest(vmid)
    config = await _get(f"nodes/{guest.node}/{guest.kind}/{guest.vmid}/config")

    _drop(config, "cipassword", "digest", "meta", "smbios1", "sshkeys", "vmgenid")

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

    Combines cluster storage configuration with runtime storage status from each online Proxmox VE 
    node.
    """
    config_data = await _get("storage")
    _remove_key(config_data, "digest")

    status_by_storage: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    for node in await _list_online_nodes():
        status_data = await _get(f"nodes/{node}/storage")

        for status in status_data:
            storage_id = str(status.get("storage", ""))

            if storage_id:
                status_by_storage.setdefault(storage_id, {})[node] = status

    storages = [
        ProxmoxStorage(
            storage_id=str(item["storage"]),
            storage_type=str(item.get("type", "")),
            config=item,
            status_by_node=status_by_storage.get(str(item["storage"]), {}),
        ) for item in config_data
    ]

    for storage in storages:
        _drop(storage.config, "storage", "type")

        for status in storage.status_by_node.values():
            _drop(status, "storage", "type")

    storages.sort(key=lambda storage: storage.storage_id)

    return ProxmoxStorages(storages=storages)


async def get_proxmox_node_network(node: str) -> ProxmoxNodeNetwork:
    """
    Get the Proxmox VE network configuration for one node.

    Returns physical interfaces, bridges, bonds, VLAN interfaces, and other interfaces visible to 
    the configured API token.
    """
    interfaces = await _get(f"nodes/{node}/network")

    _remove_key(interfaces, "digest")

    interfaces.sort(key=lambda interface: str(interface.get("iface", "")))

    return ProxmoxNodeNetwork(
        node=node,
        interfaces=interfaces,
    )


async def get_proxmox_node_status(node: str) -> ProxmoxNodeStatus:
    """
    Get detailed runtime and system status for one Proxmox VE node.

    Includes the status information exposed by Proxmox for CPU, memory, swap, load, uptime, kernel,
    PVE version, and related host metrics.
    """
    status = await _get(f"nodes/{node}/status")
    version = await _get(f"nodes/{node}/version")

    status.get("cpuinfo", {}).pop("flags", None)
    _drop(status, "kversion", "pveversion")
    version.pop("release", None)

    return ProxmoxNodeStatus(
        node=node,
        status=status,
        version=version,
    )


async def get_proxmox_storage_health(node: str) -> ProxmoxStorageHealth:
    """
    Get physical disk and ZFS health information for one Proxmox VE node.

    Combines local disk inventory with ZFS pool summaries and detailed status for each detected ZFS
    pool.
    """
    disks = await _get(
        f"nodes/{node}/disks/list",
        params={"include-partitions": "0"},
    )

    zpools = await _get(f"nodes/{node}/disks/zfs")

    zpool_details: dict[str, dict[str, Any]] = {}

    for pool in zpools:
        name = str(pool.get("name", ""))

        if not name:
            continue

        zpool_details[name] = await _get(f"nodes/{node}/disks/zfs/{name}")
        zpool_details[name].pop("name", None)

    disks.sort(key=lambda disk: str(disk.get("devpath", disk.get("name", ""))))
    zpools.sort(key=lambda pool: str(pool.get("name", "")))

    return ProxmoxStorageHealth(
        node=node,
        disks=disks,
        zpools=zpools,
        zpool_details=zpool_details,
    )


async def get_proxmox_backup_status() -> ProxmoxBackupStatus:
    """
    Get configured Proxmox VE backup jobs, guest backup coverage, and recent vzdump task history 
    from online nodes.

    This is a read-only overview intended for backup auditing.
    """
    jobs = await _get("cluster/backup")
    not_backed_up = await _get("cluster/backup-info/not-backed-up")

    _remove_key(jobs, "digest")
    _remove_key(not_backed_up, "digest")

    recent_tasks_by_node = (
        await list_proxmox_tasks(typefilter="vzdump", limit=10)
    ).tasks_by_node

    jobs.sort(key=lambda job: str(job.get("id", "")))
    not_backed_up.sort(key=lambda guest: int(guest.get("vmid", 0)))

    return ProxmoxBackupStatus(
        jobs=jobs,
        not_backed_up=not_backed_up,
        recent_tasks_by_node=recent_tasks_by_node,
    )


async def get_proxmox_firewall() -> ProxmoxFirewall:
    """
    Get Proxmox VE firewall configuration across datacenter, node, and VM/LXC scopes.

    Includes firewall enable state, options and rules, cluster security groups, aliases, IP sets,
    guest network devices, and guest-local firewall objects.
    """
    # Datacenter firewall
    datacenter_options = await _get("cluster/firewall/options")
    datacenter_rules = await _get("cluster/firewall/rules")
    datacenter_aliases = await _get("cluster/firewall/aliases")
    datacenter_ipsets = await _get_firewall_ipsets("cluster/firewall")
    security_groups = await _get("cluster/firewall/groups")

    security_group_rules: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for group in security_groups:
        name = str(group.get("group", ""))

        if not name:
            continue

        security_group_rules[name] = await _get(f"cluster/firewall/groups/{name}")

    datacenter_rules.sort(key=lambda rule: int(rule.get("pos", 0)))
    datacenter_aliases.sort(key=lambda alias: str(alias.get("name", "")))
    security_groups.sort(key=lambda group: str(group.get("group", "")))

    datacenter = {
        "enabled": bool(datacenter_options.pop("enable", 0)),
        "options": datacenter_options,
        "rules": datacenter_rules,
        "aliases": datacenter_aliases,
        "ipsets": datacenter_ipsets,
        "security_groups": security_groups,
        "security_group_rules": security_group_rules,
    }

    # Node firewalls
    nodes: dict[
        str,
        dict[str, Any],
    ] = {}

    for node in await _list_online_nodes():
        options = await _get(f"nodes/{node}/firewall/options")
        rules = await _get(f"nodes/{node}/firewall/rules")
        rules.sort(key=lambda rule: int(rule.get("pos", 0)))

        nodes[node] = {
            "enabled": bool(options.pop("enable", 0)),
            "options": options,
            "rules": rules,
        }

    # Guest firewalls
    guest_inventory = await list_proxmox_guests()

    guests: list[dict[str, Any]] = []

    for guest in guest_inventory.guests:
        base_path = f"nodes/{guest.node}/{guest.kind}/{guest.vmid}/firewall"

        config = await _get(f"nodes/{guest.node}/{guest.kind}/{guest.vmid}/config")

        network_devices = {
            key: value
            for key, value in config.items()
            if key.startswith("net") and key[3:].isdigit()
        }

        options = await _get(f"{base_path}/options")
        rules = await _get(f"{base_path}/rules")
        aliases = await _get(f"{base_path}/aliases")
        ipsets = await _get_firewall_ipsets(base_path)

        rules.sort(key=lambda rule: int(rule.get("pos", 0)))
        aliases.sort(key=lambda alias: str(alias.get("name", "")))

        guest_firewall = {
            "vmid": guest.vmid,
            "name": guest.name,
            "kind": guest.kind,
            "node": guest.node,
            "enabled": bool(options.pop("enable", 0)),
            "network_devices": network_devices,
            "options": options,
            "rules": rules,
        }

        if aliases:
            guest_firewall["aliases"] = aliases

        if ipsets["ipsets"]:
            guest_firewall["ipsets"] = ipsets

        guests.append(guest_firewall)

    guests.sort(key=lambda guest: int(guest["vmid"]))

    _remove_key(datacenter, "digest")
    _remove_key(nodes, "digest")
    _remove_key(guests, "digest")

    return ProxmoxFirewall(
        datacenter=datacenter,
        nodes=nodes,
        guests=guests,
    )


async def list_proxmox_tasks(
    node: str | None = None,
    typefilter: str | None = None,
    errors_only: bool = False,
    limit: int = 20,
) -> ProxmoxTasks:
    """
    List recent Proxmox VE tasks from one node or all online nodes.

    Supports optional task-type filtering and filtering to completed failures. Use 
    get_proxmox_task() with a returned UPID for detailed status and log.
    """
    if limit < 1 or limit > 200:
        raise ValueError("limit must be between 1 and 200")

    nodes = [node] if node else await _list_online_nodes()

    tasks_by_node: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for current_node in nodes:
        params = {"limit": str(limit)}

        if typefilter:
            params["typefilter"] = typefilter

        tasks = await _get(
            f"nodes/{current_node}/tasks",
            params=params,
        )

        if errors_only:
            tasks = [
                task
                for task in tasks
                if str(task.get("status", "")) not in ("", "OK")
            ]

        tasks.sort(key=lambda task: int(task.get("starttime", 0)), reverse=True)

        for task in tasks:
            task.pop("node", None)

        tasks_by_node[current_node] = tasks

    return ProxmoxTasks(tasks_by_node=tasks_by_node)


async def get_proxmox_task(
    node: str,
    upid: str,
    log_limit: int = 200,
    log_start: int = 0,
) -> ProxmoxTask:
    """
    Get status and log output for one Proxmox VE task.
    """
    if log_limit < 1 or log_limit > 1000:
        raise ValueError("log_limit must be between 1 and 1000")

    if log_start < 0:
        raise ValueError("log_start must be non-negative")

    status = await _get(f"nodes/{node}/tasks/{upid}/status")
    _drop(status, "node", "upid")

    log = await _get(
        f"nodes/{node}/tasks/{upid}/log",
        params={"start": str(log_start), "limit": str(log_limit)},
    )

    return ProxmoxTask(
        node=node,
        upid=upid,
        status=status,
        log=log,
    )


async def get_proxmox_guest_runtime(vmid: int) -> ProxmoxGuestRuntime:
    """
    Get runtime state, pending changes, snapshots, and QEMU guest-agent information for one VM or 
    LXC container.

    Guest-agent subqueries are best-effort so a stopped VM or missing agent does not make the whole
    runtime inspection fail.
    """
    guest = await _find_guest(vmid)

    base_path = f"nodes/{guest.node}/{guest.kind}/{guest.vmid}"

    current = await _get(f"{base_path}/status/current")
    pending = await _get(f"{base_path}/pending")
    snapshots = await _get(f"{base_path}/snapshot")

    _drop(current, "blockstat", "proxmox-support", "nics", "vmid", "name")

    pending = [
        item
        for item in pending
        if "pending" in item or "delete" in item
    ]

    _remove_key(snapshots, "digest")

    agent: dict[str, Any] | None = None

    if guest.kind == "qemu":
        agent_base = f"{base_path}/agent"

        info = await _try_get(f"{agent_base}/info")
        network_interfaces = await _try_get(f"{agent_base}/network-get-interfaces")
        filesystems = await _try_get(f"{agent_base}/get-fsinfo")
        os_info = await _try_get(f"{agent_base}/get-osinfo")

        if info["ok"]:
            info["data"].get("result", {}).pop("supported_commands", None)

        if network_interfaces["ok"]:
            network_interfaces["data"]["result"] = [
                {key: value for key, value in interface.items() if key != "statistics"}
                for interface in network_interfaces["data"].get("result", [])
                if not str(interface.get("name", "")).startswith("veth")
            ]

        if filesystems["ok"]:
            filesystems["data"]["result"] = [
                {
                    key: value
                    for key, value in filesystem.items()
                    if key not in ("disk", "total-bytes-privileged")
                }
                for filesystem in filesystems["data"].get("result", [])
            ]

        agent = {
            "info": info,
            "network_interfaces": network_interfaces,
            "filesystems": filesystems,
            "os": os_info,
        }

    return ProxmoxGuestRuntime(
        vmid=guest.vmid,
        name=guest.name,
        kind=guest.kind,
        node=guest.node,
        current=current,
        pending=pending,
        snapshots=snapshots,
        agent=agent,
    )


async def get_proxmox_node_software(node: str) -> ProxmoxNodeSoftware:
    """
    Get repository, package, subscription, certificate, and running kernel information for one
    Proxmox VE node.

    Deliberately excludes /apt/update because that endpoint requires Sys.Modify and may refresh the
    APT package cache.
    """
    repositories = await _get(f"nodes/{node}/apt/repositories")
    versions = await _get(f"nodes/{node}/apt/versions")
    subscription = await _get(f"nodes/{node}/subscription")
    certificates = await _get(f"nodes/{node}/certificates/info")
    running_kernel = next(
        (str(package["RunningKernel"]) for package in versions if package.get("RunningKernel")),
        "",
    )

    _remove_key(repositories, "digest")
    repositories.pop("infos", None)
    repositories["files"] = [
        file
        for file in repositories.get("files", [])
        if file.get("repositories")
    ]

    repositories["standard-repos"] = [
        repo
        for repo in repositories.get("standard-repos", [])
        if "status" in repo
    ]

    versions = [
        _pick(
            package,
            "Package", "OldVersion", "Version", "Origin", "Arch", "ManagerVersion",
        )
        for package in versions
    ]

    _drop(subscription, "serverid", "url")

    for certificate in certificates:
        certificate.pop("pem", None)

    versions.sort(key=lambda package: str(package.get("Package", "")))
    certificates.sort(key=lambda certificate: str(certificate.get("filename", "")))

    return ProxmoxNodeSoftware(
        node=node,
        running_kernel=running_kernel,
        repositories=repositories,
        versions=versions,
        subscription=subscription,
        certificates=certificates,
    )


async def get_proxmox_hardware(
    node: str,
    all_devices: bool = False,
) -> ProxmoxHardware:
    """
    Get PCI/USB inventory and cluster PCI/USB/directory mappings for one Proxmox VE node.

    Known repetitive CPU-internal devices are omitted by default. Set all_devices to include the
    complete PCI inventory. USB inventory and cluster mappings are best-effort.
    """
    pci_devices = await _get(f"nodes/{node}/hardware/pci")
    usb_result = await _try_get(f"nodes/{node}/hardware/usb")

    usb_devices = usb_result.get("data", []) if usb_result["ok"] else []
    usb_error = None if usb_result["ok"] else str(usb_result["error"])

    mappings: dict[str, Any] = {}

    for kind in ("pci", "usb", "dir"):
        result = await _try_get(f"cluster/mapping/{kind}")
        mappings[kind] = result["data"] if result["ok"] else {"error": result["error"]}

    _remove_key(mappings, "digest")

    if not all_devices:
        noisy_devices = (
            "Registers",
            "Integrated Memory Controller",
            "LM Channel",
            "LMS Channel",
            "LMDP Channel",
            "DECS Channel",
        )

        pci_devices = [
            device
            for device in pci_devices
            if not (
                device.get("vendor") == "0x8086"
                and str(device.get("device_name", "")).startswith("Sky Lake-E")
                and any(name in str(device.get("device_name", "")) for name in noisy_devices)
            )
        ]

    pci_devices.sort(key=lambda device: str(device.get("id", device.get("pciid", ""))))
    usb_devices.sort(key=lambda device: (
        str(device.get("busnum", "")),
        str(device.get("devnum", "")),
    ))

    return ProxmoxHardware(
        node=node,
        pci_devices=pci_devices,
        usb_devices=usb_devices,
        usb_error=usb_error,
        mappings=mappings,
    )


async def get_proxmox_access_control() -> ProxmoxAccessControl:
    """
    Get effective permissions and relevant Proxmox VE access-control configuration.

    Repeated permissions, unused built-in roles, and low-value user metadata are omitted.
    """    
    effective_permissions = await _get("access/permissions")
    acls = await _get("access/acl")
    roles = await _get("access/roles")
    groups = await _get("access/groups")
    realms = await _get("access/domains")
    users = await _get("access/users")

    effective_permissions = {
        path: sorted(privilege for privilege, enabled in permissions.items() if enabled)
        for path, permissions in effective_permissions.items()
    }

    if len({tuple(privileges) for privileges in effective_permissions.values()}) == 1:
        effective_permissions = {"all_paths": next(iter(effective_permissions.values()), [])}

    acls = [_pick(acl, "path", "roleid", "ugid", "type", "propagate") for acl in acls]
    groups = [_pick(group, "groupid", "users") for group in groups]
    realms = [_pick(realm, "realm", "type") for realm in realms]
    users = [_pick(user, "userid", "enable") for user in users]

    used_roles = {str(acl.get("roleid", "")) for acl in acls}

    roles = [
        _pick(role, "roleid", "privs", "special")
        for role in roles
        if not role.get("special") or str(role.get("roleid", "")) in used_roles
    ]

    acls.sort(key=lambda acl: (
        str(acl.get("path", "")), 
        str(acl.get("roleid", "")), 
        str(acl.get("ugid", ""))
    ))
    roles.sort(key=lambda role: str(role.get("roleid", "")))
    groups.sort(key=lambda group: str(group.get("groupid", "")))
    realms.sort(key=lambda realm: str(realm.get("realm", "")))
    users.sort(key=lambda user: str(user.get("userid", "")))

    return ProxmoxAccessControl(
        effective_permissions=effective_permissions,
        acls=acls,
        roles=roles,
        groups=groups,
        realms=realms,
        users=users,
    )


TOOLS = (
    list_proxmox_nodes,
    list_proxmox_guests,
    get_proxmox_guest_config,
    list_proxmox_storage,
    get_proxmox_node_network,
    get_proxmox_node_status,
    get_proxmox_storage_health,
    get_proxmox_backup_status,
    get_proxmox_firewall,
    list_proxmox_tasks,
    get_proxmox_task,
    get_proxmox_guest_runtime,
    get_proxmox_node_software,
    get_proxmox_hardware,
    get_proxmox_access_control,
)
