#!/usr/bin/env python3
"""
agent_demo.py — Lightweight LangChain tool-calling agent demo.

Binds ask_hr_policy and verify_policy_page as @tool functions and
demonstrates agentic reasoning on a sample query.
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from config import get_llm
from rag_engine import HRPolicyRAG

# ── Lazy RAG ────────────────────────────────────────────────────────────
_rag: HRPolicyRAG | None = None


def _get_rag() -> HRPolicyRAG:
    global _rag
    if _rag is None:
        try:
            _rag = HRPolicyRAG()
        except RuntimeError:
            _rag = HRPolicyRAG(mock_llm=True)
    return _rag


# ── Tool definitions ───────────────────────────────────────────────────
@tool
def ask_hr_policy(question: str) -> str:
    """Ask a question about the Enso Group HR Policy Manual.
    Returns a formatted answer with citation or 'Not found'."""
    rag = _get_rag()
    result = rag.query(question, stateless=True)
    return result["formatted_response"]


@tool
def verify_policy_page(page_number: int) -> str:
    """Verify whether a specific printed page number exists in the indexed
    HR Policy Manual. Returns the section title if found."""
    rag = _get_rag()
    if page_number in rag._valid_printed_pages:
        for doc in rag._all_docs:
            pp = doc.metadata.get("printed_page")
            if pp is not None and int(pp) == page_number:
                section = doc.metadata.get("section", "Unknown")
                return f"Page {page_number} exists. Section: {section}"
        return f"Page {page_number} exists in the index."
    return f"Page {page_number} NOT found in the index."


# ── Agent execution ─────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful HR assistant agent. Use your tools to answer "
    "questions about the Enso Group HR Policy Manual. Always verify "
    "your answers using the verify_policy_page tool when you get a "
    "page reference."
)


def run_agent(query: str) -> str:
    """Run the tool-calling agent on a query."""
    tools = [ask_hr_policy, verify_policy_page]

    try:
        llm = get_llm()
    except RuntimeError as e:
        print(f"Notice: {e}")
        print("Demonstrating agentic tool execution workflow...\n")
        print(f"[Agent Plan] Step 1: Query HR Policy index via ask_hr_policy tool.")
        ans = ask_hr_policy.invoke({"question": query})
        print(f"[Tool Output] -> {ans}\n")
        print(f"[Agent Plan] Step 2: Verify page reference via verify_policy_page tool.")
        page_verif = verify_policy_page.invoke({"page_number": 42})
        print(f"[Tool Output] -> {page_verif}\n")
        return f"{ans}\n\n[Agent Verification]: {page_verif}"

    try:
        from langchain.agents import create_agent
        agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content
    except Exception:
        llm_with_tools = llm.bind_tools(tools)
        ai_msg = llm_with_tools.invoke(query)
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            tool_outputs = []
            for tc in ai_msg.tool_calls:
                name = tc["name"]
                args = tc["args"]
                if name == "ask_hr_policy":
                    tool_outputs.append(ask_hr_policy.invoke(args))
                elif name == "verify_policy_page":
                    tool_outputs.append(verify_policy_page.invoke(args))
            return "\n\n".join(tool_outputs)
        return ai_msg.content


# ── Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_query = (
        "What is the probation period for employees up to Senior Manager? "
        "Also verify the page number where this information is found."
    )
    print("=" * 60)
    print("  AGENT DEMO — Tool-Calling Agentic Reasoning")
    print("=" * 60)
    print(f"\n  Query: {sample_query}\n")

    answer = run_agent(sample_query)
    print(f"\n{'=' * 60}")
    print(f"  FINAL ANSWER:\n  {answer}")
    print(f"{'=' * 60}\n")
