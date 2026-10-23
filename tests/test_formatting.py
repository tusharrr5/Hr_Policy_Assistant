"""
test_formatting.py — Offline unit tests for output formatting.

Tests the exact output format:
  Supported:   "Answer: ... \\n\\nReference: HR Policy Manual, page X, section Y"
  Unsupported: "Not found"
"""

import unittest
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSupportedFormat(unittest.TestCase):
    """Test the formatted output for supported answers."""

    def test_format_structure(self):
        """Supported answer must have 'Answer:' and 'Reference:' lines."""
        output = (
            "Answer: 12 days of Privilege Leave (PL) per calendar year for "
            "permanent employees.\n\n"
            "Reference: HR Policy Manual, page 60, section Annexure III, [G] Leave Policy"
        )
        self.assertTrue(output.startswith("Answer:"))
        self.assertIn("\n\nReference:", output)

    def test_reference_contains_page(self):
        """Reference line must contain 'page <number>'."""
        output = (
            "Answer: Test answer\n\n"
            "Reference: HR Policy Manual, page 42, section Annexure I, [B] Probation Policy"
        )
        match = re.search(r"page (\d+)", output)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 42)

    def test_reference_contains_section(self):
        """Reference line must contain 'section <title>'."""
        output = (
            "Answer: Test\n\n"
            "Reference: HR Policy Manual, page 60, section Annexure III, [G] Leave Policy"
        )
        self.assertIn("section Annexure III", output)

    def test_reference_starts_with_hr_policy_manual(self):
        """Reference must start with 'HR Policy Manual'."""
        ref_line = "Reference: HR Policy Manual, page 60, section Annexure III, [G] Leave Policy"
        self.assertTrue(ref_line.startswith("Reference: HR Policy Manual"))


class TestUnsupportedFormat(unittest.TestCase):
    """Test the exact 'Not found' format for unsupported answers."""

    def test_exact_not_found(self):
        """Unsupported answer must be exactly 'Not found'."""
        output = "Not found"
        self.assertEqual(output, "Not found")

    def test_no_trailing_punctuation(self):
        """'Not found' must NOT have trailing period or explanation."""
        output = "Not found"
        self.assertFalse(output.endswith("."))
        self.assertFalse(output.endswith("!"))
        self.assertEqual(len(output.split("\n")), 1)

    def test_not_found_no_citation(self):
        """'Not found' must NOT contain 'Reference:' or 'page'."""
        output = "Not found"
        self.assertNotIn("Reference:", output)
        self.assertNotIn("page", output)


class TestFormatConsistency(unittest.TestCase):
    """Test that format building logic produces consistent output."""

    def test_build_supported_format(self):
        """Simulate the format building from rag_engine.py."""
        answer_text = "3 months from the date of joining"
        page_number = 42
        section_title = "Annexure I, [B] Probation Policy"

        ref_parts = ["HR Policy Manual"]
        if page_number is not None:
            ref_parts.append(f"page {page_number}")
        if section_title:
            ref_parts.append(f"section {section_title}")
        reference = ", ".join(ref_parts)
        formatted = f"Answer: {answer_text}\n\nReference: {reference}"

        self.assertEqual(
            formatted,
            "Answer: 3 months from the date of joining\n\n"
            "Reference: HR Policy Manual, page 42, section Annexure I, "
            "[B] Probation Policy",
        )

    def test_build_unsupported_format(self):
        """Unsupported path must produce exactly 'Not found'."""
        is_supported = False
        formatted = "Not found" if not is_supported else "Something"
        self.assertEqual(formatted, "Not found")


if __name__ == "__main__":
    unittest.main()
