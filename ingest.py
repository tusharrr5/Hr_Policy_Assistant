"""
ingest.py — PDF ingestion pipeline for the HR Policy Assistant.

1. Loads hr_manual.pdf via PyPDFLoader.
2. Cleans broken PDF hyphens (words split across lines in table cells).
3. Extracts printed page numbers from footer stamps.
4. Tracks Annexure + section headings stateily across pages.
5. Prepends source metadata header to each page.
6. Chunks with RecursiveCharacterTextSplitter and persists to ChromaDB.
"""

import os
import re
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import (
    PDF_PATH,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    get_embeddings,
)

# ── Regex patterns ──────────────────────────────────────────────────────
# Printed page number footer: "HR Manual - 42"
PRINTED_PAGE_RE = re.compile(r"HR\s*Manual\s*[-–—]\s*(\d+)", re.IGNORECASE)

# Annexure heading: "Annexure I", "ANNEXURE - I", "Annexure - III", etc.
# Anchored to start-of-line to skip TOC entries that have leading dots/text.
ANNEXURE_RE = re.compile(r"^((?:ANNEXURE|Annexure)\s*[-–—]?\s*[IVXLCDM]+)", re.IGNORECASE | re.MULTILINE)

# Section heading: "[A] Something Policy", "[B] Probation Policy", etc.
SECTION_BRACKET_RE = re.compile(
    r"\[([A-Z])\]\s*(.+?)(?:\n|$)"
)

# Section heading variant: "Section 2.2 - Controlling Absence Policy" etc.
SECTION_NUMBERED_RE = re.compile(
    r"(Section\s+\d+(?:\.\d+)*\s*[-–—]\s*.+?)(?:\n|$)", re.IGNORECASE
)

# ── Hyphen cleaning ────────────────────────────────────────────────────
def clean_hyphens(text: str) -> str:
    """
    Fix words broken across lines in PDF table cells.
    E.g. 'al-\\nowed' → 'alowed' (sic), 'em-\\nployee' → 'employee'.
    Also handles double hyphens: 'al--\\nlowed'.
    """
    # Pattern: word-fragment + one-or-more hyphens + optional whitespace/newline + word-fragment
    cleaned = re.sub(r"(\w+)-+\s*\n\s*(\w+)", r"\1\2", text)
    return cleaned


# ── Metadata extraction ────────────────────────────────────────────────
def extract_printed_page(text: str, loader_page_index: int) -> int:
    """Return the printed page number from the footer, or fallback to 1-based index."""
    match = PRINTED_PAGE_RE.search(text)
    if match:
        return int(match.group(1))
    return loader_page_index + 1  # 1-based fallback


def update_section_tracking(
    text: str, current_annexure: str, current_section: str
) -> tuple[str, str]:
    """
    Parse the page text for Annexure headers and section titles.
    Returns (updated_annexure, updated_section).
    Skips detection on Table of Contents pages to avoid false state tracking.
    """
    # Skip TOC pages (they list Annexure/section names with dotted leaders)
    first_lines = text[:300].lower()
    if "contents" in first_lines and "....." in text:
        return current_annexure, current_section

    # Check for new Annexure header
    annexure_match = ANNEXURE_RE.search(text)
    if annexure_match:
        raw = annexure_match.group(1).strip()
        # Extract just the Roman numeral part and normalize
        roman_match = re.search(r"[IVXLCDM]+$", raw, re.IGNORECASE)
        if roman_match:
            numeral = roman_match.group(0).upper()
            current_annexure = f"Annexure {numeral}"
            # Reset section when entering a new Annexure
            current_section = ""

    # Check for bracketed section heading like [B] Probation Policy
    bracket_matches = SECTION_BRACKET_RE.findall(text)
    if bracket_matches:
        # Take the last match on the page (in case multiple appear)
        letter, title = bracket_matches[-1]
        current_section = f"[{letter}] {title.strip()}"

    # Check for numbered section heading
    numbered_matches = SECTION_NUMBERED_RE.findall(text)
    if numbered_matches:
        current_section = numbered_matches[-1].strip()

    return current_annexure, current_section


def build_section_label(annexure: str, section: str) -> str:
    """Combine annexure and section into a single label."""
    parts = [p for p in (annexure, section) if p]
    return ", ".join(parts) if parts else "General"


# ── Main ingestion ──────────────────────────────────────────────────────
def ingest(pdf_path: str = PDF_PATH, chroma_dir: str = CHROMA_DIR) -> dict:
    """
    Run the full ingestion pipeline. Returns a summary dict with statistics.
    """
    # Validate PDF
    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)

    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    if not raw_docs:
        print("ERROR: PDF loaded but contains no pages.")
        sys.exit(1)

    print(f"  → Loaded {len(raw_docs)} pages from PDF.")

    # ── Pass 1: Clean text + extract metadata ───────────────────────────
    current_annexure = ""
    current_section = ""
    processed_docs = []
    page_metadata_map = {}  # printed_page → section (for verification)

    for doc in raw_docs:
        loader_page_index = doc.metadata.get("page", 0)
        raw_text = doc.page_content

        # Clean broken hyphens
        cleaned_text = clean_hyphens(raw_text)

        # Extract printed page number
        printed_page = extract_printed_page(raw_text, loader_page_index)

        # Update stateful section tracking
        current_annexure, current_section = update_section_tracking(
            cleaned_text, current_annexure, current_section
        )

        section_label = build_section_label(current_annexure, current_section)

        # Prepend source metadata header
        header = (
            f"SOURCE METADATA -> Printed Page: {printed_page} | "
            f"Section: {section_label}\nContent:\n"
        )
        doc.page_content = header + cleaned_text

        # Store metadata
        doc.metadata["printed_page"] = printed_page
        doc.metadata["section"] = section_label
        doc.metadata["source_file"] = os.path.basename(pdf_path)

        processed_docs.append(doc)
        page_metadata_map[printed_page] = section_label

    print(f"  → Processed metadata for {len(processed_docs)} pages.")

    # ── Verification prints ─────────────────────────────────────────────
    for check_page in [42, 60, 62]:
        if check_page in page_metadata_map:
            print(
                f"  ✓ Page {check_page}: section = \"{page_metadata_map[check_page]}\""
            )
        else:
            print(f"  ✗ Page {check_page}: NOT FOUND in extracted metadata")

    # ── Pass 2: Chunk ───────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(processed_docs)
    print(f"  → Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # ── Pass 3: Persist to ChromaDB ─────────────────────────────────────
    # Remove old DB if it exists
    if os.path.isdir(chroma_dir):
        import shutil
        shutil.rmtree(chroma_dir)
        print(f"  → Removed old ChromaDB at {chroma_dir}")

    embeddings = get_embeddings()
    print("  → Embedding model loaded. Persisting to ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=chroma_dir,
    )

    total = vectorstore._collection.count()
    print(f"  ✓ ChromaDB persisted: {total} vectors in collection '{COLLECTION_NAME}'")
    print(f"  ✓ ChromaDB directory: {chroma_dir}")

    return {
        "total_pages": len(raw_docs),
        "total_chunks": len(chunks),
        "total_vectors": total,
        "page_metadata_map": page_metadata_map,
    }


# ── CLI entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    result = ingest()
    print("\n=== Ingestion Complete ===")
    print(f"  Pages: {result['total_pages']}")
    print(f"  Chunks: {result['total_chunks']}")
    print(f"  Vectors: {result['total_vectors']}")
