"""
test_metadata.py — Offline unit tests for page/section metadata extraction.

Tests that the ingestion pipeline correctly extracts printed page numbers
and section titles for key pages (42, 60, 62) without needing an API key.
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import (
    clean_hyphens,
    extract_printed_page,
    update_section_tracking,
    build_section_label,
)


class TestHyphenCleaning(unittest.TestCase):
    """Test broken-hyphen cleaning in PDF table text."""

    def test_single_hyphen_linebreak(self):
        text = "em-\nployee"
        self.assertEqual(clean_hyphens(text), "employee")

    def test_double_hyphen_linebreak(self):
        text = "al--\nlowed"
        self.assertEqual(clean_hyphens(text), "allowed")

    def test_hyphen_with_spaces(self):
        text = "to-  \n  tal"
        self.assertEqual(clean_hyphens(text), "total")

    def test_preserves_normal_hyphens(self):
        text = "well-known company"
        self.assertEqual(clean_hyphens(text), "well-known company")

    def test_preserves_unbroken_text(self):
        text = "This is normal text without any breaks."
        self.assertEqual(clean_hyphens(text), "This is normal text without any breaks.")


class TestPrintedPageExtraction(unittest.TestCase):
    """Test printed page number extraction from footer stamps."""

    def test_standard_footer(self):
        text = "Some content here\nHR Manual - 42\n"
        self.assertEqual(extract_printed_page(text, 50), 42)

    def test_footer_with_dash_variant(self):
        text = "Content\nHR Manual – 60\n"
        self.assertEqual(extract_printed_page(text, 70), 60)

    def test_fallback_to_loader_page(self):
        text = "Cover page with no footer"
        self.assertEqual(extract_printed_page(text, 0), 1)  # 0-based → 1-based

    def test_page_62(self):
        text = "Leave policy details\nHR Manual - 62"
        self.assertEqual(extract_printed_page(text, 75), 62)


class TestSectionTracking(unittest.TestCase):
    """Test stateful section heading detection."""

    def test_annexure_detection(self):
        text = "Annexure I\nSome content here"
        annexure, section = update_section_tracking(text, "", "")
        self.assertEqual(annexure, "Annexure I")

    def test_annexure_iii_detection(self):
        text = "Annexure III\nCompensation details"
        annexure, section = update_section_tracking(text, "", "")
        self.assertEqual(annexure, "Annexure III")

    def test_bracket_section_detection(self):
        text = "[B] Probation Policy\nDetails about probation"
        annexure, section = update_section_tracking(text, "Annexure I", "")
        self.assertEqual(section, "[B] Probation Policy")

    def test_leave_policy_detection(self):
        text = "[G] Leave Policy\nLeave details"
        annexure, section = update_section_tracking(text, "Annexure III", "")
        self.assertEqual(section, "[G] Leave Policy")

    def test_section_inheritance(self):
        """Continuation pages should inherit the current section."""
        annexure, section = update_section_tracking(
            "More leave details, no new heading",
            "Annexure III",
            "[G] Leave Policy",
        )
        self.assertEqual(annexure, "Annexure III")
        self.assertEqual(section, "[G] Leave Policy")


class TestSectionLabel(unittest.TestCase):
    """Test combined section label building."""

    def test_full_label(self):
        label = build_section_label("Annexure III", "[G] Leave Policy")
        self.assertEqual(label, "Annexure III, [G] Leave Policy")

    def test_annexure_only(self):
        label = build_section_label("Annexure I", "")
        self.assertEqual(label, "Annexure I")

    def test_no_metadata(self):
        label = build_section_label("", "")
        self.assertEqual(label, "General")


class TestPageMetadataIntegration(unittest.TestCase):
    """
    Integration test: verify that running ingestion on the real PDF
    produces correct metadata for pages 42, 60, and 62.
    Requires hr_manual.pdf to be present.
    """

    @classmethod
    def setUpClass(cls):
        """Check if PDF exists, skip if not."""
        from config import PDF_PATH
        if not os.path.isfile(PDF_PATH):
            raise unittest.SkipTest("hr_manual.pdf not found — skipping integration test")

        # Run ingestion to a temp directory
        import tempfile
        cls.temp_dir = tempfile.mkdtemp()
        from ingest import ingest
        cls.result = ingest(pdf_path=PDF_PATH, chroma_dir=cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp ChromaDB."""
        import shutil
        if hasattr(cls, "temp_dir") and os.path.isdir(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_page_42_metadata(self):
        meta = self.result["page_metadata_map"]
        self.assertIn(42, meta)
        self.assertIn("Annexure I", meta[42])
        self.assertIn("[B] Probation Policy", meta[42])

    def test_page_60_metadata(self):
        meta = self.result["page_metadata_map"]
        self.assertIn(60, meta)
        self.assertIn("Annexure III", meta[60])
        self.assertIn("[G] Leave Policy", meta[60])

    def test_page_62_metadata(self):
        meta = self.result["page_metadata_map"]
        self.assertIn(62, meta)
        self.assertIn("Annexure III", meta[62])
        self.assertIn("[G] Leave Policy", meta[62])


if __name__ == "__main__":
    unittest.main()
