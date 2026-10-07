from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from homelab_mcp.http import get_json
from homelab_mcp.utils import pick, pick_all, require_env


# -------------------------------------------------------------------------------------------------
# Structured output
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class pfSenseSystem:
    status: dict[str, Any]
    version: dict[str, Any]
    services: list[dict[str, Any]]
    packages: list[dict[str, Any]]


@dataclass(frozen=True)
class pfSenseInterfaces:
    interfaces: list[dict[str, Any]]
    status: list[dict[str, Any]]
    vlans: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    laggs: list[dict[str, Any]]
    bridges: list[dict[str, Any]]


@dataclass(frozen=True)
class pfSenseRouting:
    default_gateway: dict[str, Any]
    gateways: list[dict[str, Any]]
    status: list[dict[str, Any]]
    gateway_groups: list[dict[str, Any]]
    static_routes: list[dict[str, Any]]


@dataclass(frozen=True)
class pfSenseFirewall:
    rules: list[dict[str, Any]]
    aliases: list[dict[str, Any]]
    port_forwards: list[dict[str, Any]]
    outbound_nat_mode: dict[str, Any]
    outbound_nat: list[dict[str, Any]]
    one_to_one_nat: list[dict[str, Any]]
    virtual_ips: list[dict[str, Any]]
    state_table_size: dict[str, Any]


@dataclass(frozen=True)
class pfSenseClients:
    arp_table: list[dict[str, Any]]
    dhcp_leases: list[dict[str, Any]]


@dataclass(frozen=True)
class pfSenseNetworkServices:
    dhcp_servers: list[dict[str, Any]]
    dhcp_static_mappings: dict[str, list[dict[str, Any]]]
    dhcp_address_pools: dict[str, list[dict[str, Any]]]
    dns_resolver: dict[str, Any]
    dns_host_overrides: list[dict[str, Any]]
    dns_domain_overrides: list[dict[str, Any]]
    system_dns: dict[str, Any]


@dataclass(frozen=True)
class pfSenseVPNStatus:
    wireguard_tunnels: list[dict[str, Any]]
    wireguard_peers: list[dict[str, Any]]
    openvpn_servers: list[dict[str, Any]]
    openvpn_clients: list[dict[str, Any]]
    openvpn_connections: list[dict[str, Any]]
    ipsec_sas: list[dict[str, Any]]
    ipsec_child_sas: list[dict[str, Any]]


# -------------------------------------------------------------------------------------------------
# Internal API helpers
# -------------------------------------------------------------------------------------------------


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

    return await get_json(
        _client,
        path,
        params=params,
        source="pfSense API",
        key="data",
    )


# -------------------------------------------------------------------------------------------------
# MCP tools
# -------------------------------------------------------------------------------------------------


async def get_pfsense_system() -> pfSenseSystem:
    """Get pfSense platform, version, service, and installed-package information."""
    status, version, services, packages = await asyncio.gather(
        _get("status/system"),
        _get("system/version"),
        _get("status/services"),
        _get("system/packages"),
    )

    return pfSenseSystem(
        status=pick(
            status, "platform", "bios_vendor", "bios_version", "bios_date", "kernel_pti", 
            "mds_mitigation", "cpu_model", "cpu_count",  
        ),
        version=version,
        services=pick_all(services, "name", "description", "enabled", "status"),
        packages=pick_all(
            packages, "name", "shortname", "installed_version", "latest_version", 
            "update_available"
        ),
    )


async def get_pfsense_interfaces() -> pfSenseInterfaces:
    """Get pfSense interface configuration, topology, and runtime status."""
    interfaces, status, vlans, groups, laggs, bridges = await asyncio.gather(
        _get("interfaces"),
        _get("status/interfaces"),
        _get("interface/vlans"),
        _get("interface/groups"),
        _get("interface/laggs"),
        _get("interface/bridges"),
    )

    return pfSenseInterfaces(
        interfaces=pick_all(
            interfaces, "id", "if", "enable", "descr", "mtu", "mss", "blockpriv", "blockbogons", 
            "typev4", "ipaddr", "subnet", "gateway", "typev6", "ipaddrv6", "subnetv6", "gatewayv6"
        ),
        status=pick_all(
            status, "name", "descr", "hwif", "macaddr", "mtu", "status", "ipaddr", "subnet", 
            "ipaddrv6", "subnetv6", "media", "gateway", "gatewayv6"
        ),
        vlans=vlans,
        groups=groups,
        laggs=laggs,
        bridges=bridges,
    )


async def get_pfsense_routing() -> pfSenseRouting:
    """Get pfSense gateway, gateway-group, route, and gateway status."""
    default_gateway, gateways, status, gateway_groups, static_routes = await asyncio.gather(
        _get("routing/gateway/default"),
        _get("routing/gateways"),
        _get("status/gateways"),
        _get("routing/gateway/groups"),
        _get("routing/static_routes"),
    )

    return pfSenseRouting(
        default_gateway=default_gateway,
        gateways=pick_all(
            gateways, "name", "descr", "disabled", "ipprotocol", "interface", 
            "gateway", "monitor_disable", "monitor", "force_down", "weight", "latencylow", 
            "latencyhigh", "losslow", "losshigh"
        ),
        status=pick_all(
            status, "name", "srcip", "monitorip", "delay", "stddev", "loss", "status", "substatus"
        ),
        gateway_groups=gateway_groups,
        static_routes=pick_all(static_routes, "network", "gateway", "descr", "disabled"),
    )


async def get_pfsense_firewall() -> pfSenseFirewall:
    """Get pfSense firewall rules, aliases, NAT, virtual IPs, and state-table size."""
    (
        rules, aliases, port_forwards, outbound_nat_mode, outbound_nat, one_to_one_nat, 
        virtual_ips, state_table_size
    ) = await asyncio.gather(
        _get("firewall/rules"),
        _get("firewall/aliases"),
        _get("firewall/nat/port_forwards"),
        _get("firewall/nat/outbound/mode"),
        _get("firewall/nat/outbound/mappings"),
        _get("firewall/nat/one_to_one/mappings"),
        _get("firewall/virtual_ips"),
        _get("firewall/states/size"),
    )

    return pfSenseFirewall(
        rules=pick_all(
            rules, "type", "interface", "ipprotocol", "protocol", "icmptype", "source", 
            "source_port", "destination", "destination_port", "descr", "disabled", "log", 
            "statetype", "floating", "quick", "direction", "gateway", "associated_rule_id"
        ),
        aliases=pick_all(aliases, "name", "type", "descr", "address"),
        port_forwards=pick_all(
            port_forwards, "interface", "ipprotocol", "protocol", "source", "source_port", 
            "destination", "destination_port", "target", "local_port", "disabled", "descr", 
            "natreflection"
        ),
        outbound_nat_mode=outbound_nat_mode,
        outbound_nat=pick_all(
            outbound_nat, "interface", "protocol", "disabled", "nonat", "source", "source_port", 
            "destination", "destination_port", "target", "target_subnet", "nat_port", 
            "static_nat_port", "descr"
        ),
        one_to_one_nat=one_to_one_nat,
        virtual_ips=pick_all(
            virtual_ips, "mode", "interface", "type", "subnet", "subnet_bits", "descr", 
            "carp_status"
        ),
        state_table_size=state_table_size,
    )


async def get_pfsense_clients() -> pfSenseClients:
    """Get pfSense ARP entries and current DHCP leases."""
    arp_table, dhcp_leases = await asyncio.gather(
        _get("diagnostics/arp_table"),
        _get("status/dhcp_server/leases"),
    )

    return pfSenseClients(
        arp_table=pick_all(
            arp_table, "ip_address", "mac_address", "interface", "permanent", "dnsresolve", 
            "expires"
        ),
        dhcp_leases=pick_all(
            dhcp_leases, "ip", "mac", "hostname", "if", "starts", "ends", "active_status", 
            "online_status", "descr"
        ),
    )


async def get_pfsense_network_services() -> pfSenseNetworkServices:
    """Get pfSense DHCP and DNS configuration relevant to local network services."""
    (
        dhcp_servers, dns_resolver, dns_host_overrides, dns_domain_overrides, system_dns
    ) = await asyncio.gather(
        _get("services/dhcp_servers"),
        _get("services/dns_resolver/settings"),
        _get("services/dns_resolver/host_overrides"),
        _get("services/dns_resolver/domain_overrides"),
        _get("system/dns"),
    )

    static_mappings: dict[str, list[dict[str, Any]]] = {}
    address_pools: dict[str, list[dict[str, Any]]] = {}

    for server in dhcp_servers:
        interface = str(server["id"])
        params = {"parent_id": interface}

        mappings, pools = await asyncio.gather(
            _get(
                "services/dhcp_server/static_mappings",
                params=params,
            ),
            _get(
                "services/dhcp_server/address_pools",
                params=params,
            ),
        )

        static_mappings[interface] = pick_all(
            mappings, "parent_id", "mac", "ipaddr", "hostname", "domain", "descr", 
            "arp_table_static_entry"
        )

        address_pools[interface] = pick_all(
            pools, "parent_id", "range_from", "range_to", "gateway", "domain", "dnsserver", 
            "ntpserver", "descr"
        )

    return pfSenseNetworkServices(
        dhcp_servers=pick_all(
            dhcp_servers, "id", "interface", "enable", "range_from", "range_to", "gateway", 
            "domain", "dnsserver", "ntpserver", "defaultleasetime", "maxleasetime", "denyunknown"
        ),
        dhcp_static_mappings=static_mappings,
        dhcp_address_pools=address_pools,
        dns_resolver=pick(
            dns_resolver, "enable", "port", "enablessl", "active_interface", "outgoing_interface", 
            "strictout", "system_domain_local_zone_type", "dnssec", "forwarding", "regdhcp", 
            "regdhcpstatic", "regovpnclients", "custom_options"
        ),
        dns_host_overrides=pick_all(
            dns_host_overrides, "host", "domain", "ip", "descr", "aliases"
        ),
        dns_domain_overrides=pick_all(
            dns_domain_overrides, "domain", "ip", "descr", "tls_queries", "tls_hostname"
        ),
        system_dns=pick(system_dns, "dnsallowoverride", "dnslocalhost", "dnsserver"),
    )


async def get_pfsense_vpn_status() -> pfSenseVPNStatus:
    """Get current WireGuard, OpenVPN, and IPsec runtime status."""
    (
        wireguard_tunnels, wireguard_peers, openvpn_servers, openvpn_clients, openvpn_connections, 
        ipsec_sas, ipsec_child_sas,
    ) = await asyncio.gather(
        _get("status/wireguard/tunnels"),
        _get("status/wireguard/peers"),
        _get("status/openvpn/servers"),
        _get("status/openvpn/clients"),
        _get("status/openvpn/server/connections"),
        _get("status/ipsec/sas"),
        _get("status/ipsec/child_sas"),
    )

    return pfSenseVPNStatus(
        wireguard_tunnels=pick_all(
            wireguard_tunnels, "name", "status", "listen_port", "mtu", "descr"
        ),
        wireguard_peers=pick_all(
            wireguard_peers, "tunnel_device", "endpoint", "allowed_ips", "latest_handshake", 
            "persistent_keepalive", "descr"
        ),
        openvpn_servers=openvpn_servers,
        openvpn_clients=openvpn_clients,
        openvpn_connections=openvpn_connections,
        ipsec_sas=ipsec_sas,
        ipsec_child_sas=ipsec_child_sas,
    )


TOOLS = (
    get_pfsense_system,
    get_pfsense_interfaces,
    get_pfsense_routing,
    get_pfsense_firewall,
    get_pfsense_clients,
    get_pfsense_network_services,
    get_pfsense_vpn_status,
)
