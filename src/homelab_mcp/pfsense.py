from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from pydantic import Field

from homelab_mcp.http import get_json
from homelab_mcp.redact import redact_value
from homelab_mcp.utils import pick, pick_all, require_env


# ---------------------------------------------------------------------------
# Tool parameter types
# ---------------------------------------------------------------------------

DHCPServerRef = Annotated[
    str,
    Field(description="DHCP server ID from the 'id' field returned by list_pfsense_dhcp_servers.")
]


# ---------------------------------------------------------------------------
# Internal API helpers
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def session() -> AsyncIterator[None]:
    """Open the shared pfSense REST API client for the server lifetime."""
    global _client

    if _client is not None:
        raise RuntimeError("pfSense API client is already initialized")

    base_url = require_env("PFSENSE_URL")

    if not base_url.startswith("https://"):
        raise RuntimeError("PFSENSE_URL must use HTTPS")

    try:
        async with httpx.AsyncClient(
            base_url=f"{base_url.rstrip('/')}/api/v2/",
            headers={"X-API-Key": require_env("PFSENSE_API_KEY")},
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
        raise RuntimeError("pfSense API client is not initialized")

    data = await get_json(
        _client,
        path,
        params=params,
        source="pfSense API",
        key="data",
    )

    return redact_value(data)


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

async def get_pfsense_system_status() -> dict[str, Any]:
    """
    Get pfSense platform and hardware status.

    Upstream: pfSense REST API v2
    GET /api/v2/status/system

    High-frequency utilization, temperature, uptime, memory, swap, and disk metrics are omitted; 
    use Grafana for metrics and historical health.
    """
    data = await _get("status/system")

    return pick(
        data,
        "platform", "bios_vendor", "bios_version", "bios_date", "kernel_pti", "mds_mitigation", 
        "cpu_model", "cpu_count"
    )


async def get_pfsense_version() -> dict[str, Any]:
    """
    Get the installed pfSense version.

    Upstream: pfSense REST API v2
    GET /api/v2/system/version
    """
    data = await _get("system/version")

    return pick(
        data,
        "version", "base", "patch", "buildtime"
    )


async def list_pfsense_services() -> dict[str, Any]:
    """
    List pfSense services and their current enabled/running state.

    Upstream: pfSense REST API v2
    GET /api/v2/status/services
    """
    data = await _get("status/services")

    return {"services": pick_all(
        data,
        "name", "description", "enabled", "status"
    )}


async def list_pfsense_packages() -> dict[str, Any]:
    """
    List installed pfSense packages and available versions.

    Upstream: pfSense REST API v2
    GET /api/v2/system/packages
    """
    data = await _get("system/packages")

    return {"packages": pick_all(
        data,
        "name", "shortname", "installed_version", "latest_version", "update_available"
    )}


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

async def list_pfsense_interfaces() -> dict[str, Any]:
    """
    List configured pfSense interfaces.

    Upstream: pfSense REST API v2
    GET /api/v2/interfaces
    """
    data = await _get("interfaces")

    return {"interfaces": pick_all(
        data,
        "id", "if", "enable", "descr", "mtu", "mss", "blockpriv", "blockbogons", "typev4", 
        "ipaddr", "subnet", "gateway", "typev6", "ipaddrv6", "subnetv6", "gatewayv6"
    )}


async def list_pfsense_interface_status() -> dict[str, Any]:
    """
    List current pfSense interface link and address state.

    Upstream: pfSense REST API v2
    GET /api/v2/status/interfaces

    Traffic counters are omitted.
    """
    data = await _get("status/interfaces")

    return {"interfaces": pick_all(
        data,
        "name", "descr", "hwif", "macaddr", "mtu", "status", "ipaddr", "subnet", "ipaddrv6", 
        "subnetv6", "media", "gateway", "gatewayv6"
    )}


async def list_pfsense_vlans() -> dict[str, Any]:
    """
    List configured pfSense VLANs.

    Upstream: pfSense REST API v2
    GET /api/v2/interface/vlans

    The redacted upstream VLAN objects are returned without a parallel local schema.
    """
    return {"vlans": await _get("interface/vlans")}


async def list_pfsense_interface_groups() -> dict[str, Any]:
    """
    List configured pfSense interface groups.

    Upstream: pfSense REST API v2
    GET /api/v2/interface/groups
    """
    data = await _get("interface/groups")

    return {"groups": pick_all(
        data,
        "id", "ifname", "members", "descr"
    )}


async def list_pfsense_laggs() -> dict[str, Any]:
    """
    List configured pfSense link aggregations.

    Upstream: pfSense REST API v2
    GET /api/v2/interface/laggs

    The redacted upstream LAGG objects are returned without a parallel local schema.
    """
    return {"laggs": await _get("interface/laggs")}


async def list_pfsense_bridges() -> dict[str, Any]:
    """
    List configured pfSense bridges.

    Upstream: pfSense REST API v2
    GET /api/v2/interface/bridges

    The redacted upstream bridge objects are returned without a parallel local schema.
    """
    return {"bridges": await _get("interface/bridges")}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

async def get_pfsense_default_gateway() -> dict[str, Any]:
    """
    Get the configured default IPv4 and IPv6 gateways.

    Upstream: pfSense REST API v2
    GET /api/v2/routing/gateway/default
    """
    data = await _get("routing/gateway/default")

    return pick(
        data,
        "defaultgw4", "defaultgw6"
    )


async def list_pfsense_gateways() -> dict[str, Any]:
    """
    List configured pfSense gateways.

    Upstream: pfSense REST API v2
    GET /api/v2/routing/gateways
    """
    data = await _get("routing/gateways")

    return {"gateways": pick_all(
        data,
        "name", "descr", "disabled", "ipprotocol", "interface", "gateway", "monitor_disable", 
        "monitor", "force_down", "weight", "latencylow", "latencyhigh", "losslow", "losshigh"
    )}


async def list_pfsense_gateway_status() -> dict[str, Any]:
    """
    List current pfSense gateway-monitor state.

    Upstream: pfSense REST API v2
    GET /api/v2/status/gateways

    Current reachability state is retained; historical latency/loss is ommitted.
    """
    data = await _get("status/gateways")

    return {"gateways": pick_all(
        data,
        "name", "srcip", "monitorip", "delay", "stddev", "loss", "status", "substatus"
    )}


async def list_pfsense_gateway_groups() -> dict[str, Any]:
    """
    List configured pfSense gateway groups.

    Upstream: pfSense REST API v2
    GET /api/v2/routing/gateway/groups

    The redacted upstream gateway-group objects are returned without a parallel local schema.
    """
    return {"groups": await _get("routing/gateway/groups")}


async def list_pfsense_static_routes() -> dict[str, Any]:
    """
    List configured pfSense static routes.

    Upstream: pfSense REST API v2
    GET /api/v2/routing/static_routes
    """
    data = await _get("routing/static_routes")

    return {"routes": pick_all(
        data,
        "network", "gateway", "descr", "disabled"
    )}


# ---------------------------------------------------------------------------
# Firewall and NAT
# ---------------------------------------------------------------------------

async def list_pfsense_firewall_rules() -> dict[str, Any]:
    """
    List pfSense firewall rules.

    Upstream: pfSense REST API v2
    GET /api/v2/firewall/rules
    """
    data = await _get("firewall/rules")

    return {"rules": pick_all(
        data,
        "type", "interface", "ipprotocol", "protocol", "icmptype", "source", "source_port", 
        "destination", "destination_port", "descr", "disabled", "log", "statetype", "floating", 
        "quick", "direction", "gateway", "associated_rule_id"
    )}


async def list_pfsense_firewall_aliases() -> dict[str, Any]:
    """
    List pfSense firewall aliases.

    Upstream: pfSense REST API v2
    GET /api/v2/firewall/aliases
    """
    data = await _get("firewall/aliases")

    return {"aliases": pick_all(
        data,
        "name", "type", "descr", "address"
    )}


async def list_pfsense_port_forwards() -> dict[str, Any]:
    """
    List pfSense NAT port-forward mappings.

    Upstream: pfSense REST API v2
    GET /api/v2/firewall/nat/port_forwards
    """
    data = await _get("firewall/nat/port_forwards")

    return {"port_forwards": pick_all(
        data,
        "interface", "ipprotocol", "protocol", "source", "source_port", "destination", 
        "destination_port", "target", "local_port", "disabled", "descr", "natreflection"
    )}


async def get_pfsense_outbound_nat_mode() -> dict[str, Any]:
    """
    Get the pfSense outbound NAT mode.

    Upstream: pfSense REST API v2
    GET /api/v2/firewall/nat/outbound/mode
    """
    data = await _get("firewall/nat/outbound/mode")

    return pick(
        data,
        "mode"
    )


async def list_pfsense_outbound_nat_mappings() -> dict[str, Any]:
    """
    List pfSense outbound NAT mappings.

    Upstream: pfSense REST API v2
    GET /api/v2/firewall/nat/outbound/mappings
    """
    data = await _get("firewall/nat/outbound/mappings")

    return {"mappings": pick_all(
        data,
        "interface", "protocol", "disabled", "nonat", "source", "source_port", "destination", 
        "destination_port", "target", "target_subnet", "nat_port", "static_nat_port", "descr"
    )}


async def list_pfsense_one_to_one_nat_mappings() -> dict[str, Any]:
    """
    List pfSense 1:1 NAT mappings.

    Upstream: pfSense REST API v2
    GET /api/v2/firewall/nat/one_to_one/mappings

    The redacted upstream 1:1 NAT objects are returned without a parallel local schema.
    """
    return {"mappings": await _get("firewall/nat/one_to_one/mappings")}


async def list_pfsense_virtual_ips() -> dict[str, Any]:
    """
    List configured pfSense virtual IPs.

    Upstream: pfSense REST API v2
    GET /api/v2/firewall/virtual_ips
    """
    data = await _get("firewall/virtual_ips")

    return {"virtual_ips": pick_all(
        data,
        "mode", "interface", "type", "subnet", "subnet_bits", "descr", "carp_status"
    )}


# ---------------------------------------------------------------------------
# Client discovery
# ---------------------------------------------------------------------------

async def list_pfsense_arp_entries() -> dict[str, Any]:
    """
    List current pfSense ARP table entries.

    Upstream: pfSense REST API v2
    GET /api/v2/diagnostics/arp_table
    """
    data = await _get("diagnostics/arp_table")

    return {"entries": pick_all(
        data,
        "ip_address", "mac_address", "interface", "permanent", "dnsresolve", "expires"
    )}


async def list_pfsense_dhcp_leases() -> dict[str, Any]:
    """
    List current DHCP server leases.

    Upstream: pfSense REST API v2
    GET /api/v2/status/dhcp_server/leases
    """
    data = await _get("status/dhcp_server/leases")

    return {"leases": pick_all(
        data,
        "ip", "mac", "hostname", "if", "starts", "ends", "active_status", "online_status", "descr"
    )}


# ---------------------------------------------------------------------------
# DHCP and DNS
# ---------------------------------------------------------------------------

async def list_pfsense_dhcp_servers() -> dict[str, Any]:
    """
    List configured pfSense DHCP servers.

    Upstream: pfSense REST API v2
    GET /api/v2/services/dhcp_servers

    Use each returned `id` as `parent_id` when querying static mappings or address pools.
    """
    data = await _get("services/dhcp_servers")

    return {"servers": pick_all(
        data,
        "id", "interface", "enable", "range_from", "range_to", "gateway", "domain", "dnsserver", 
        "ntpserver", "defaultleasetime", "maxleasetime", "denyunknown"
    )}


async def list_pfsense_dhcp_static_mappings(
    parent_id: DHCPServerRef,
) -> dict[str, Any]:
    """
    List static DHCP mappings for one pfSense DHCP server.

    Upstream: pfSense REST API v2
    GET /api/v2/services/dhcp_server/static_mappings?parent_id={parent_id}
    """
    data = await _get(
        "services/dhcp_server/static_mappings",
        params={"parent_id": parent_id},
    )

    return {"mappings": pick_all(
        data,
        "parent_id", "mac", "ipaddr", "hostname", "domain", "descr", "arp_table_static_entry"
    )}


async def list_pfsense_dhcp_address_pools(
    parent_id: DHCPServerRef,
) -> dict[str, Any]:
    """
    List additional DHCP address pools for one pfSense DHCP server.

    Upstream: pfSense REST API v2
    GET /api/v2/services/dhcp_server/address_pools?parent_id={parent_id}
    """
    data = await _get(
        "services/dhcp_server/address_pools",
        params={"parent_id": parent_id},
    )

    return {"pools": pick_all(
        data,
        "parent_id", "range_from", "range_to", "gateway", "domain", "dnsserver", "ntpserver", 
        "descr"
    )}


async def get_pfsense_dns_resolver() -> dict[str, Any]:
    """
    Get pfSense DNS Resolver settings.

    Upstream: pfSense REST API v2
    GET /api/v2/services/dns_resolver/settings
    """
    data = await _get("services/dns_resolver/settings")

    return pick(
        data,
        "enable", "port", "enablessl", "active_interface", "outgoing_interface", "strictout", 
        "system_domain_local_zone_type", "dnssec", "forwarding", "regdhcp", "regdhcpstatic",
        "regovpnclients", "custom_options"
    )


async def list_pfsense_dns_host_overrides() -> dict[str, Any]:
    """
    List pfSense DNS Resolver host overrides.

    Upstream: pfSense REST API v2
    GET /api/v2/services/dns_resolver/host_overrides
    """
    data = await _get("services/dns_resolver/host_overrides")

    return {"overrides": pick_all(
        data,
        "host", "domain", "ip", "descr", "aliases"
    )}


async def list_pfsense_dns_domain_overrides() -> dict[str, Any]:
    """
    List pfSense DNS Resolver domain overrides.

    Upstream: pfSense REST API v2
    GET /api/v2/services/dns_resolver/domain_overrides
    """
    data = await _get("services/dns_resolver/domain_overrides")

    return {"overrides": pick_all(
        data,
        "domain", "ip", "descr", "tls_queries", "tls_hostname"
    )}


async def get_pfsense_system_dns() -> dict[str, Any]:
    """
    Get pfSense system DNS settings.

    Upstream: pfSense REST API v2
    GET /api/v2/system/dns
    """
    data = await _get("system/dns")

    return pick(
        data,
        "dnsallowoverride", "dnslocalhost", "dnsserver"
    )


# ---------------------------------------------------------------------------
# VPN runtime state
# ---------------------------------------------------------------------------

async def list_pfsense_wireguard_tunnels() -> dict[str, Any]:
    """
    List current WireGuard tunnel status.

    Upstream: pfSense REST API v2
    GET /api/v2/status/wireguard/tunnels
    """
    data = await _get("status/wireguard/tunnels")

    return {"tunnels": pick_all(
        data,
        "name", "status", "listen_port", "mtu", "descr"
    )}


async def list_pfsense_wireguard_peers() -> dict[str, Any]:
    """
    List current WireGuard peer status.

    Upstream: pfSense REST API v2
    GET /api/v2/status/wireguard/peers
    """
    data = await _get("status/wireguard/peers")

    return {"peers": pick_all(
        data,
        "tunnel_device", "endpoint", "allowed_ips", "latest_handshake", "persistent_keepalive", 
        "descr"
    )}


async def list_pfsense_openvpn_servers() -> dict[str, Any]:
    """
    List current OpenVPN server status.

    Upstream: pfSense REST API v2
    GET /api/v2/status/openvpn/servers

    The redacted upstream status objects are returned without a parallel local schema.
    """
    return {"servers": await _get("status/openvpn/servers")}


async def list_pfsense_openvpn_clients() -> dict[str, Any]:
    """
    List current OpenVPN client status.

    Upstream: pfSense REST API v2
    GET /api/v2/status/openvpn/clients

    The redacted upstream status objects are returned without a parallel local schema.
    """
    return {"clients": await _get("status/openvpn/clients")}


async def list_pfsense_openvpn_connections() -> dict[str, Any]:
    """
    List current OpenVPN server connections.

    Upstream: pfSense REST API v2
    GET /api/v2/status/openvpn/server/connections

    The redacted upstream status objects are returned without a parallel local schema.
    """
    return {"connections": await _get("status/openvpn/server/connections")}


async def list_pfsense_ipsec_sas() -> dict[str, Any]:
    """
    List current IPsec security associations.

    Upstream: pfSense REST API v2
    GET /api/v2/status/ipsec/sas

    The redacted upstream status objects are returned without a parallel local schema.
    """
    return {"sas": await _get("status/ipsec/sas")}


async def list_pfsense_ipsec_child_sas() -> dict[str, Any]:
    """
    List current IPsec child security associations.

    Upstream: pfSense REST API v2
    GET /api/v2/status/ipsec/child_sas

    The redacted upstream status objects are returned without a parallel local schema.
    """
    return {"child_sas": await _get("status/ipsec/child_sas")}


TOOLS = (
    get_pfsense_system_status,
    get_pfsense_version,
    list_pfsense_services,
    list_pfsense_packages,
    list_pfsense_interfaces,
    list_pfsense_interface_status,
    list_pfsense_vlans,
    list_pfsense_interface_groups,
    list_pfsense_laggs,
    list_pfsense_bridges,
    get_pfsense_default_gateway,
    list_pfsense_gateways,
    list_pfsense_gateway_status,
    list_pfsense_gateway_groups,
    list_pfsense_static_routes,
    list_pfsense_firewall_rules,
    list_pfsense_firewall_aliases,
    list_pfsense_port_forwards,
    get_pfsense_outbound_nat_mode,
    list_pfsense_outbound_nat_mappings,
    list_pfsense_one_to_one_nat_mappings,
    list_pfsense_virtual_ips,
    list_pfsense_arp_entries,
    list_pfsense_dhcp_leases,
    list_pfsense_dhcp_servers,
    list_pfsense_dhcp_static_mappings,
    list_pfsense_dhcp_address_pools,
    get_pfsense_dns_resolver,
    list_pfsense_dns_host_overrides,
    list_pfsense_dns_domain_overrides,
    get_pfsense_system_dns,
    list_pfsense_wireguard_tunnels,
    list_pfsense_wireguard_peers,
    list_pfsense_openvpn_servers,
    list_pfsense_openvpn_clients,
    list_pfsense_openvpn_connections,
    list_pfsense_ipsec_sas,
    list_pfsense_ipsec_child_sas,
)
