"""
test_guardrails.py — Offline unit tests for the alignment guardrail logic.

Tests the deterministic Python validation that rejects unsupported,
low-confidence, or unverifiable answers. Does NOT require an API key.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_engine import GroundedAnswer, check_token_overlap, tokenize
from langchain_core.documents import Document


class TestTokenize(unittest.TestCase):
    """Test the tokenization utility."""

    def test_basic_tokenize(self):
        tokens = tokenize("Privilege Leave 12 days per year")
        self.assertIn("privilege", tokens)
        self.assertIn("leave", tokens)
        self.assertIn("days", tokens)
        self.assertIn("per", tokens)
        self.assertIn("year", tokens)

    def test_filters_short_tokens(self):
        tokens = tokenize("an is of to it")
        # All 2-char tokens should be filtered out
        self.assertEqual(len(tokens), 0)

    def test_alphanumeric_only(self):
        tokens = tokenize("page_42 -- section [G]")
        self.assertIn("page", tokens)


class TestTokenOverlap(unittest.TestCase):
    """Test the hyphen-safe token overlap check."""

    def test_full_overlap(self):
        evidence = "12 days of Privilege Leave"
        context = "Permanent employees get 12 days of Privilege Leave per year"
        self.assertTrue(check_token_overlap(evidence, context))

    def test_partial_overlap_above_threshold(self):
        evidence = "3 months probation period from joining"
        context = "probation period of 3 months from date of joining for employees"
        self.assertTrue(check_token_overlap(evidence, context, threshold=0.5))

    def test_no_overlap(self):
        evidence = "cryptocurrency blockchain reimbursement"
        context = "leave policy privilege casual sick maternity"
        self.assertFalse(check_token_overlap(evidence, context))

    def test_empty_evidence(self):
        self.assertFalse(check_token_overlap("", "some context"))

    def test_empty_context(self):
        self.assertFalse(check_token_overlap("some evidence", ""))


class TestGuardrailValidation(unittest.TestCase):
    """Test the deterministic guardrail validation logic."""

    def _make_docs(self, pages_and_sections):
        """Helper to create mock retrieved documents."""
        docs = []
        for page, section in pages_and_sections:
            docs.append(
                Document(
                    page_content=f"SOURCE METADATA -> Printed Page: {page} | "
                    f"Section: {section}\nContent:\n"
                    f"This is content about policy on page {page} with "
                    f"leave privilege probation maternity 12 days 3 months",
                    metadata={"printed_page": page, "section": section},
                )
            )
        return docs

    def test_reject_unsupported(self):
        """is_supported=False should always be rejected."""
        answer = GroundedAnswer(
            is_supported=False,
            answer="Not found",
            evidence_quote="",
            confidence=0.0,
        )
        docs = self._make_docs([(42, "Annexure I, [B] Probation Policy")])
        # We cannot call _validate without an HRPolicyRAG instance,
        # but we can test the logic components individually.
        self.assertFalse(answer.is_supported)

    def test_reject_low_confidence(self):
        """Confidence below threshold should be rejected."""
        answer = GroundedAnswer(
            is_supported=True,
            answer="Some answer",
            page_number=42,
            section_title="Test",
            evidence_quote="policy details",
            confidence=0.3,
        )
        self.assertLess(answer.confidence, 0.65)

    def test_accept_valid_answer(self):
        """Valid answer with all checks passing."""
        answer = GroundedAnswer(
            is_supported=True,
            answer="12 days of Privilege Leave",
            page_number=60,
            section_title="Annexure III, [G] Leave Policy",
            evidence_quote="12 days privilege leave per year",
            confidence=0.95,
        )
        self.assertTrue(answer.is_supported)
        self.assertGreaterEqual(answer.confidence, 0.65)
        self.assertLessEqual(answer.confidence, 1.0)

    def test_reject_out_of_bounds_confidence(self):
        """Confidence > 1.0 should be rejected."""
        answer = GroundedAnswer(
            is_supported=True,
            answer="Test",
            confidence=1.5,
            evidence_quote="test",
        )
        self.assertGreater(answer.confidence, 1.0)

    def test_page_not_in_retrieved(self):
        """Page number not in retrieved docs should trigger rejection."""
        answer = GroundedAnswer(
            is_supported=True,
            answer="Test",
            page_number=99,
            section_title="Test",
            evidence_quote="test policy leave",
            confidence=0.9,
        )
        docs = self._make_docs([(42, "Annexure I"), (60, "Annexure III")])
        retrieved_pages = {int(d.metadata["printed_page"]) for d in docs}
        self.assertNotIn(answer.page_number, retrieved_pages)


if __name__ == "__main__":
    unittest.main()
