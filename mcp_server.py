#!/usr/bin/env python3
"""
mcp_server.py — Model Context Protocol server for the HR Policy Assistant.

Exposes two MCP tools:
  • ask_hr_policy(question)      → Query the RAG pipeline
  • verify_policy_page(page_number) → Verify if a page exists in the index

Uses FastMCP from the `mcp` package, with a clean fallback wrapper
if the package is not installed.
"""

import sys

# ── Try importing FastMCP or MCPServer; provide a clean fallback ────────
try:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    print(
        "WARNING: 'mcp' package not installed. Install with:\n"
        "  pip install mcp\n"
        "Providing a stub so the script can be imported without error."
    )

    class FastMCP:
        """Minimal stub when the real mcp package is unavailable."""

        def __init__(self, name: str, **kwargs):
            self.name = name
            self._tools = {}

        def tool(self, *args, **kwargs):
            def decorator(func):
                self._tools[func.__name__] = func
                return func
            return decorator

        def run(self, transport: str = "stdio"):
            print(f"[{self.name}] MCP stub — no real server started.")
            print(f"  Registered tools: {list(self._tools.keys())}")


from rag_engine import HRPolicyRAG

# ── MCP Server ──────────────────────────────────────────────────────────
mcp = FastMCP("HR Policy Assistant")

# Lazy-initialized RAG engine
_rag: HRPolicyRAG | None = None


def _get_rag() -> HRPolicyRAG:
    global _rag
    if _rag is None:
        try:
            _rag = HRPolicyRAG()
        except RuntimeError:
            _rag = HRPolicyRAG(mock_llm=True)
    return _rag


@mcp.tool()
def ask_hr_policy(question: str) -> str:
    """
    Ask a question about the Enso Group HR Policy Manual.
    Returns a formatted answer with citation or 'Not found'.
    """
    rag = _get_rag()
    result = rag.query(question, stateless=True)
    return result["formatted_response"]


@mcp.tool()
def verify_policy_page(page_number: int) -> str:
    """
    Verify whether a specific printed page number exists in the indexed
    HR Policy Manual. Returns the section title if found.
    """
    rag = _get_rag()
    if page_number in rag._valid_printed_pages:
        # Find section from any chunk with that page
        for doc in rag._all_docs:
            pp = doc.metadata.get("printed_page")
            if pp is not None and int(pp) == page_number:
                section = doc.metadata.get("section", "Unknown")
                return f"Page {page_number} exists in the index. Section: {section}"
        return f"Page {page_number} exists in the index."
    else:
        return f"Page {page_number} was NOT found in the indexed manual."


# ── Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting MCP server (stdio transport)...")
    mcp.run(transport="stdio")
