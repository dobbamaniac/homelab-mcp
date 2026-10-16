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
        "mcp.amotelab.org",
        "mcp.amotelab.org:*",
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
            "Read-only adapters for homelab management APIs and host configuration. "
            "Most tools map to one upstream GET endpoint or one read-only host command; "
            "the exact upstream endpoint or command is named in each tool description. "
            "Response field names stay close to the upstream API unless fields are "
            "redacted or omitted for safety. Compose multiple tools when a question "
            "requires multiple resources. Use Grafana for metrics, historical health, "
            "alerts, and logs."
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
