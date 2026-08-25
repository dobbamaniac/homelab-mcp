from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from homelab_mcp import proxmox


MODULES = (
    proxmox,
)


# These are hints to MCP clients, not security controls
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,          # Tool does not modify its environment
    openWorldHint=False,        # Tool operates on a bounded/controlled set of resources
)


@asynccontextmanager
async def lifespan(_: MCPServer) -> AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        for module in MODULES:
            await stack.enter_async_context(module.session())

        yield


def create_server() -> MCPServer:
    mcp = MCPServer(
        "Homelab",
        instructions="Read-only tools for inspecting homelab infrastructure.",
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
    )
