#!/usr/bin/env python3
"""
evaluate.py — LLMOps Evaluation & SFT/Alignment Dataset Export.

Runs the 5 official exam questions + 1 negative control question,
asserts expected outcomes, computes metrics, saves eval_report.json,
and exports data/sft_alignment_dataset.jsonl.
"""

import json
import os
import time

from rag_engine import HRPolicyRAG
from config import DATA_DIR

# ── Evaluation dataset ─────────────────────────────────────────────────
EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "How many days of Privilege Leave (PL) does a permanent employee get per year?",
        "expected_supported": True,
        "expected_page": 60,
        "expected_keywords": ["12", "privilege", "leave"],
    },
    {
        "id": "Q2",
        "question": "What is the probation period for employees up to Senior Manager?",
        "expected_supported": True,
        "expected_page": 42,
        "expected_keywords": ["3 months", "probation"],
    },
    {
        "id": "Q3",
        "question": "What is the company's health-insurance premium amount?",
        "expected_supported": False,
        "expected_page": None,
        "expected_keywords": [],
    },
    {
        "id": "Q4",
        "question": "What is the current GST rate in India?",
        "expected_supported": False,
        "expected_page": None,
        "expected_keywords": [],
    },
    {
        "id": "Q5",
        "question": "How many days of maternity leave are given?",
        "expected_supported": True,
        "expected_page": 62,
        "expected_keywords": ["12 weeks", "maternity"],
    },
    {
        "id": "Q6",
        "question": "What is the company cryptocurrency reimbursement policy?",
        "expected_supported": False,
        "expected_page": None,
        "expected_keywords": [],
    },
]


def evaluate(mock: bool = False):
    """Run full evaluation pipeline."""
    print("=" * 60)
    print("  HR Policy Assistant — Evaluation Pipeline")
    print("=" * 60)

    print("\n  Initializing RAG engine...")
    try:
        rag = HRPolicyRAG(mock_llm=mock)
    except RuntimeError as e:
        print(f"  Notice: {e}")
        print("  Running in offline evaluation mode (mock LLM + real retrieval & guardrails)...\n")
        rag = HRPolicyRAG(mock_llm=True)

    print("  ✓ RAG engine ready.\n")

    results = []
    total_latency = 0
    correct = 0
    total = len(EVAL_QUESTIONS)

    for item in EVAL_QUESTIONS:
        qid = item["id"]
        question = item["question"]
        expected_supported = item["expected_supported"]
        expected_page = item["expected_page"]

        print(f"  [{qid}] {question}")

        result = rag.query(question, stateless=True)
        latency = result["latency_ms"]
        total_latency += latency

        is_supported = result["is_supported"]
        page_number = result["page_number"]
        formatted = result["formatted_response"]

        # Determine correctness
        support_match = is_supported == expected_supported
        page_match = True
        if expected_supported and expected_page is not None:
            page_match = page_number == expected_page

        is_correct = support_match and page_match

        if is_correct:
            correct += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"

        print(f"       {status} | supported={is_supported} (exp={expected_supported}) | "
              f"page={page_number} (exp={expected_page}) | {latency:.0f}ms")

        results.append({
            "id": qid,
            "question": question,
            "formatted_response": formatted,
            "is_supported": is_supported,
            "page_number": page_number,
            "section_title": result["section_title"],
            "expected_supported": expected_supported,
            "expected_page": expected_page,
            "correct": is_correct,
            "latency_ms": latency,
        })

    # ── Metrics ─────────────────────────────────────────────────────────
    accuracy = correct / total if total > 0 else 0.0
    avg_latency = total_latency / total if total > 0 else 0.0

    print(f"\n{'─' * 60}")
    print(f"  RESULTS: {correct}/{total} correct ({accuracy:.0%} accuracy)")
    print(f"  Average latency: {avg_latency:.0f} ms")
    print(f"{'─' * 60}")

    # ── Save eval_report.json ───────────────────────────────────────────
    report = {
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_latency_ms": round(avg_latency, 2),
        "results": results,
    }

    report_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Evaluation report saved: {report_path}")

    # ── Export SFT/Alignment JSONL ──────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    jsonl_path = os.path.join(DATA_DIR, "sft_alignment_dataset.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            if r["correct"]:
                entry = {
                    "instruction": r["question"],
                    "output": r["formatted_response"],
                    "metadata": {
                        "is_supported": r["is_supported"],
                        "page_number": r["page_number"],
                        "section_title": r["section_title"],
                    },
                }
                f.write(json.dumps(entry) + "\n")
    print(f"  ✓ SFT alignment dataset saved: {jsonl_path}")
    print()

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HR Policy Assistant Evaluation")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline mock mode without requiring an external LLM API key.",
    )
    args = parser.parse_args()
    evaluate(mock=args.mock)
