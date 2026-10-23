#!/usr/bin/env python3
"""
main.py — CLI entry point for the HR Policy Assistant.

Usage:
  python main.py          # Interactive mode (type questions, 'exit' to quit)
  python main.py --exam   # Run the 5 official exam questions
"""

import argparse
import uuid

from rag_engine import HRPolicyRAG

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         ██╗  ██╗██████╗     ██████╗  ██████╗ ██╗     ██╗    ║
║         ██║  ██║██╔══██╗    ██╔══██╗██╔═══██╗██║     ██║    ║
║         ███████║██████╔╝    ██████╔╝██║   ██║██║     ██║    ║
║         ██╔══██║██╔══██╗    ██╔═══╝ ██║   ██║██║     ██║    ║
║         ██║  ██║██║  ██║    ██║     ╚██████╔╝███████╗██║    ║
║         ╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝      ╚═════╝ ╚══════╝╚═╝    ║
║                                                              ║
║            A S S I S T A N T                                 ║
║                                                              ║
║   Enso Group HR Policy Manual — RAG-Powered Q&A             ║
║   Hybrid Retrieval · Structured Output · Guardrails         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

EXAM_QUESTIONS = [
    "How many days of Privilege Leave (PL) does a permanent employee get per year?",
    "What is the probation period for employees up to Senior Manager?",
    "What is the company's health-insurance premium amount?",
    "What is the current GST rate in India?",
    "How many days of maternity leave are given?",
]


def run_exam(rag: HRPolicyRAG) -> None:
    """Run the 5 official exam questions in stateless mode."""
    print("\n" + "=" * 60)
    print("  EXAM MODE — 5 Official Questions (stateless)")
    print("=" * 60)

    for i, question in enumerate(EXAM_QUESTIONS, 1):
        print(f"\n{'─' * 60}")
        print(f"  Q{i}: {question}")
        print(f"{'─' * 60}")

        result = rag.query(question, stateless=True)
        print(f"\n{result['formatted_response']}")
        print(f"\n  [Latency: {result['latency_ms']:.0f} ms | "
              f"Chunks retrieved: {result['retrieved_chunks']}]")

    print(f"\n{'=' * 60}")
    print("  EXAM COMPLETE")
    print(f"{'=' * 60}\n")


def run_interactive(rag: HRPolicyRAG) -> None:
    """Run the interactive question-answering loop."""
    session_id = str(uuid.uuid4())
    print("\n  Interactive Mode (session-based conversational memory)")
    print("  Type your question, or 'exit'/'quit' to leave.\n")

    while True:
        try:
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("  Goodbye!")
            break

        result = rag.query(question, session_id=session_id)
        print(f"\n  {result['formatted_response']}")
        print(f"\n  [Latency: {result['latency_ms']:.0f} ms]\n")


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="HR Policy Assistant CLI")
    parser.add_argument(
        "--exam",
        action="store_true",
        help="Run the 5 official exam questions in stateless mode.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline mock mode without requiring an external LLM API key.",
    )
    args = parser.parse_args()

    print("  Initializing RAG engine...")
    try:
        rag = HRPolicyRAG(mock_llm=args.mock)
    except RuntimeError as e:
        print(f"  Notice: {e}")
        print("  Falling back to offline evaluation mode (--mock) to verify retrieval & guardrails.\n")
        rag = HRPolicyRAG(mock_llm=True)

    print("  ✓ RAG engine ready.\n")

    if args.exam:
        run_exam(rag)
    else:
        run_interactive(rag)


if __name__ == "__main__":
    main()
