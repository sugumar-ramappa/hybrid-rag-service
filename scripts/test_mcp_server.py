"""
Test script to verify the MCP server tools work correctly.
Simulates what Claude does when it calls your tools.

Usage:
    python -m scripts.test_mcp_server
"""

import asyncio
import json

from src.mcp_server.server import mcp


async def main() -> None:
    print("=" * 50)
    print("MCP Server Test - Simulating AI tool calls")
    print("=" * 50)

    # 1. List tools (what AI sees when it connects)
    tools = await mcp.list_tools()
    print(f"\n✅ Server has {len(tools)} tools registered:")
    for t in tools:
        print(f"   - {t.name}")

    # 2. Call get_pipeline_stats
    print("\n--- Calling: get_pipeline_stats() ---")
    result = await mcp.call_tool("get_pipeline_stats", {})
    print(result)

    # 3. Call search_documents
    print("\n--- Calling: search_documents(query='What is Python?') ---")
    result = await mcp.call_tool("search_documents", {"query": "What is Python?"})
    print(result)

    print("\n✅ All MCP tools working!")


if __name__ == "__main__":
    asyncio.run(main())
