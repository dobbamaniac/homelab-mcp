from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from homelab_mcp import docker, linux, pfsense, proxmox, ssh, zfs


MODULES = (
    proxmox,
    pfsense,
    docker,
    linux,
    zfs,
)


TRANSPORT_SECURITY = TransportSecuritySettings(
    allowed_hosts=[
        "homelab-server",
        "homelab-server:*",
    ],
    allowed_origins=[],
)


# These are hints to MCP clients, not security controls
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,          # Tool does not modify its environment
    openWorldHint=False,        # Tool operates on a bounded/controlled set of resources
)


@asynccontextmanager
async def lifespan(_: MCPServer) -> AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(ssh.session())

        for module in MODULES:
            await stack.enter_async_context(module.session())

        yield


def create_server() -> MCPServer:
    mcp = MCPServer(
        "Homelab",
        instructions=(
            "Read-only tools for homelab inventory, configuration, topology, "
            "and direct state inspection. Use Grafana for metrics, resource "
            "utilization, historical health, alert state, and logs when those "
            "data are available there. Prefer inventory tools to discover "
            "identifiers before calling detail tools."
        ),
        lifespan=lifespan,
    )

    for module in MODULES:
        for tool in module.TOOLS:
            mcp.add_tool(
                tool,
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
        transport_security=TRANSPORT_SECURITY,
    )
