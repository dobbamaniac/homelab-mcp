from __future__ import annotations

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from homelab_mcp.proxmox import (
    get_proxmox_backup_status,
    get_proxmox_guest_config,
    get_proxmox_node_network,
    get_proxmox_node_status,
    get_proxmox_storage_health,
    list_proxmox_guests,
    list_proxmox_nodes,
    list_proxmox_storage,
)

# These are hints to MCP clients, not security controls
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,          # Tool does not modify its environment
    openWorldHint=False,        # Tool operates on a bounded/controlled set of resources
)

def create_server() -> MCPServer:
    mcp = MCPServer(
        "Homelab",
        instructions="Read-only tools for inspecting homelab infrastructure.",
    )

    mcp.add_tool(
        list_proxmox_guests,
        annotations=READ_ONLY,
        structured_output=True,
    )

    mcp.add_tool(
        list_proxmox_nodes,
        annotations=READ_ONLY,
        structured_output=True,
    )

    mcp.add_tool(
        get_proxmox_guest_config,
        annotations=READ_ONLY,
        structured_output=True,
    )

    mcp.add_tool(
        list_proxmox_storage,
        annotations=READ_ONLY,
        structured_output=True,
    )

    mcp.add_tool(
        get_proxmox_node_network,
        annotations=READ_ONLY,
        structured_output=True,
    )

    mcp.add_tool(
        get_proxmox_node_status,
        annotations=READ_ONLY,
        structured_output=True,
    )

    mcp.add_tool(
        get_proxmox_storage_health,
        annotations=READ_ONLY,
        structured_output=True,
    )

    mcp.add_tool(
        get_proxmox_backup_status,
        annotations=READ_ONLY,
        structured_output=True,
    )

    return mcp

def main() -> None:
    mcp = create_server()

    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        stateless_http=True,
        json_response=True,
    )
