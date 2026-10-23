# HR Policy Assistant — Enso Group HR Policy Manual RAG

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4%2B-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-orange.svg)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice-teal.svg)](https://fastapi.tiangolo.com/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-FastMCP-purple.svg)](https://modelcontextprotocol.io/)
[![Tests](https://img.shields.io/badge/Tests-42%20Passed-brightgreen.svg)]()

> **Generative AI Practical Examination Submission**  
> An enterprise-grade, hallucination-resistant Retrieval-Augmented Generation (RAG) assistant built over the **Enso Group HR Policy Manual** (79 pages). Features hybrid search (Dense Vector + BM25 with Reciprocal Rank Fusion), Pydantic structured output, deterministic alignment guardrails, conversational memory, LLMOps evaluation, SFT dataset export, FastAPI service, and an MCP agent server.

---

## Table of Contents
1. [Executive Summary & Exam Deliverables](#executive-summary--exam-deliverables)
2. [Verified Ground Truth for 5 Official Exam Questions](#verified-ground-truth-for-5-official-exam-questions)
3. [System Architecture & Pipeline](#system-architecture--pipeline)
4. [Project Layout](#project-layout)
5. [Key Technical Innovations](#key-technical-innovations)
6. [Quick Start & Installation](#quick-start--installation)
7. [Running the System](#running-the-system)
   - [CLI & Exam Mode](#1-cli--exam-mode-mainpy)
   - [LLMOps Evaluation & SFT Dataset](#2-llmops-evaluation--sft-export-evaluatepy)
   - [Offline Unit Testing](#3-offline-unit-tests-testspy)
   - [FastAPI Microservice](#4-fastapi-microservice-apipy)
   - [Model Context Protocol Server](#5-mcp-server-mcp_serverpy)
   - [Agentic Reasoning Demo](#6-agentic-tool-calling-demo-agent_demopy)
8. [Configuration & Provider Support](#configuration--provider-support)
9. [Test Suite Breakdown](#test-suite-breakdown)

---

## Executive Summary & Exam Deliverables

The assistant implements all three prioritization tiers specified in the examination rubric:

- **Priority 1 (Mandatory Exam Core):**
  - Robust PDF loading (`PyPDFLoader`) over `hr_manual.pdf` (79 pages).
  - Regex-based hyphen and broken-word cleaning (`al--\nlowed` $\to$ `allowed`).
  - Printed page number extraction from footer stamps (`HR Manual - <page>`) with viewer fallback.
  - Stateful Annexure and section heading tracking that persists across page continuations.
  - Local HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and persistent `ChromaDB`.
  - Exact citation formatting (`Answer: ... \n\nReference: HR Policy Manual, page X, section Y`).
  - Exact `Not found` abstention (zero hallucinations for out-of-context or ungrounded queries).
  - `main.py` CLI supporting `--exam` mode and interactive session chat.

- **Priority 2 (Course Features):**
  - **True Hybrid Retrieval:** Chroma similarity search ($k=5$) + BM25Okapi lexical search ($k=5$) merged via **Reciprocal Rank Fusion (RRF)** into top 6 deduplicated passages.
  - **Structured Output:** Pydantic schema (`GroundedAnswer`) via `with_structured_output` enforcing explicit boolean support flags and confidence ratings.
  - **Deterministic Alignment Guardrails:** External Python verification checking page index membership, confidence bounds ($\ge 0.65$), and hyphen-safe token overlap ($\ge 50\%$).
  - **Conversational Memory:** Per-session `InMemoryChatMessageHistory` with multi-turn context awareness.
  - **Offline Unit Testing:** 42 isolated unit tests in `tests/` covering metadata extraction, guardrail rejection, and formatting rules without requiring external API keys.
  - **LLMOps Evaluation:** `evaluate.py` calculating accuracy and latency, saving `eval_report.json`, and exporting verified instruction tuning pairs to `data/sft_alignment_dataset.jsonl`.

- **Priority 3 (Production Extensions):**
  - **FastAPI Microservice (`api.py`):** `GET /health` and `POST /ask` endpoints.
  - **Model Context Protocol (`mcp_server.py`):** Exposing `ask_hr_policy` and `verify_policy_page` tools via FastMCP / MCPServer.
  - **Agentic Tool-Calling (`agent_demo.py`):** Multi-step tool-calling agent that queries the index and autonomously verifies page references.

---

## Verified Ground Truth for 5 Official Exam Questions

All answers are strictly grounded in `hr_manual.pdf`. Outside knowledge (such as Indian labor laws or GST statutes) is completely excluded:

| # | Official Exam Question | Expected Answer | Verified Citation | Guardrail Result |
|---|---|---|---|---|
| **Q1** | *How many days of Privilege Leave (PL) does a permanent employee get per year?* | 12 days of Privilege Leave (PL) per calendar year (1st January – 31st December) for permanent employees. No leave is credited during probation, unavailed PL cannot be carried forward to the next calendar year, and accumulated PL can be encashed only as part of Full & Final settlement at basic pay. | **Page 60**, `Annexure III, [G] Leave Policy` | `✓ PASS` (Supported) |
| **Q2** | *What is the probation period for employees up to Senior Manager?* | 3 months from the date of joining for all employees up to Senior Manager (Sr. Manager). (Employees above Sr. Manager are confirmed from the date of joining; probation may be extended at Management's sole discretion only once and not exceeding the original 3-month probationary period.) | **Page 42**, `Annexure I, [B] Probation Policy` | `✓ PASS` (Supported) |
| **Q3** | *What is the company's health-insurance premium amount?* | `Not found` *(Note: Medical Reimbursement/ESIC is cited on page 57 and travel insurance on page 71, but the premium amount does not exist in the manual. The guardrail rejects tangential matches).* | *None* | `✓ PASS` (Abstention) |
| **Q4** | *What is the current GST rate in India?* | `Not found` *(External tax law absent from the manual; strictly excluded).* | *None* | `✓ PASS` (Abstention) |
| **Q5** | *How many days of maternity leave are given?* | Confirmed female staff (who have worked for at least 80 days in the preceding 12 months) are provided a total of 12 weeks (stated parenthetically in the manual as 90 days) of Maternity Leave (ML) with pay, of which not more than 6 weeks (45 days) should precede the expected date of delivery. In case of miscarriage or medical termination of pregnancy, 4 weeks of paid leave from the date of miscarriage is permitted upon producing satisfactory evidence. | **Page 62**, `Annexure III, [G] Leave Policy` | `✓ PASS` (Supported) |

---

## System Architecture & Pipeline

```text
[PDF: hr_manual.pdf (79 pages)]
               │
               ▼
   [ingest.py: Hyphen Cleaner & Metadata Extractor]
   ├── Clean broken table hyphens: re.sub(r'(\w+)-+\s*\n\s*(\w+)', r'\1\2')
   ├── Extract Printed Page footer: 'HR Manual - <page>'
   └── Stateful Annexure & Section tracking (persisting across page continuations)
               │
               ▼
   [RecursiveCharacterTextSplitter: chunk_size=900, overlap=180]
   ├── Dense Vectors (all-MiniLM-L6-v2) ──► ChromaDB (collection: enso_hr_policy)
   └── Lexical Inverted Index ─────────────► BM25Okapi
               │
               ▼
   [User Query] ──► Hybrid Retrieval
   ├── Dense Search (k=5)
   └── BM25 Lexical Search (k=5)
               │
               ▼
   [Reciprocal Rank Fusion (RRF)] ──► Top 6 Deduplicated Chunks
               │
               ▼
   [LangChain Structured LLM Prompt]
   └── GroundedAnswer Schema (is_supported, answer, page_number, section_title, evidence_quote, confidence)
               │
               ▼
   [Deterministic Python Alignment Guardrail]
   ├── 1. is_supported == True
   ├── 2. 0.0 <= confidence <= 1.0 and confidence >= 0.65
   ├── 3. page_number in retrieved chunk printed pages
   ├── 4. Token Overlap >= 50% between evidence and context
   └── 5. Metadata snapping to verified chunk section
               │
         ┌─────┴──────────────┐
         ▼                    ▼
     [PASS]                [FAIL]
   Format Output        Return exact:
   Answer: ...          "Not found"
   Reference: ...
```

---

## Project Layout

```text
HR_Policy_Assistant/
├── hr_manual.pdf                  # Enso Group HR Policy Manual (79 pages)
├── config.py                      # LLM auto-detection, paths, retrieval constants
├── ingest.py                      # PDF loading, hyphen cleaning, section tracker, chunking
├── rag_engine.py                  # Hybrid RAG, BM25, RRF, Pydantic schema, guardrail, memory
├── main.py                        # CLI application (--exam and interactive modes)
├── api.py                         # FastAPI microservice (GET /health, POST /ask)
├── mcp_server.py                  # Model Context Protocol server (FastMCP / MCPServer)
├── agent_demo.py                  # Tool-calling agent demonstrating autonomous verification
├── evaluate.py                    # LLMOps evaluation runner & SFT dataset exporter
├── requirements.txt               # Pinned project dependencies
├── .env.example                   # Template for provider API keys
├── .gitignore                     # Git ignore rules
├── README.md                      # Comprehensive project documentation
├── eval_report.json               # Automated evaluation metrics & per-question logs
├── tests/
│   ├── __init__.py
│   ├── test_metadata.py           # Unit tests for hyphen cleaning, page & section extraction
│   ├── test_guardrails.py         # Unit tests for token overlap, confidence & page validation
│   └── test_formatting.py         # Unit tests for citation formatting & exact abstention
├── data/
│   ├── .gitkeep
│   └── sft_alignment_dataset.jsonl # Exported fine-tuning Q&A pairs with verified metadata
├── screenshots/
│   └── .gitkeep
└── chroma_db/                     # Persistent ChromaDB vector store
    ├── chroma.sqlite3
    └── ...
```

---

## Key Technical Innovations

### 1. Robust Hyphen & Broken-Word Normalization
In PDF table cells (notably pages 60–62), terms are frequently split across lines: `al--\nlowed`, `em-\nployee`, `to-\ntal`. A naive keyword search or LLM prompt would fail or hallucinate. The ingestion engine cleans these with:
```python
re.sub(r"(\w+)-+\s*\n\s*(\w+)", r"\1\2", text)
```
This repairs broken words while preserving full paragraph and table layouts.

### 2. Stateful Continuation Page Section Tracking
Continuation pages (such as Page 62) lack section headers at the top—the header `[G] Leave Policy` appeared on Page 60. The ingest pipeline tracks the active Annexure and Section state across consecutive pages, ensuring chunks from Page 62 inherit the exact section title: `Annexure III, [G] Leave Policy`.

### 3. True Hybrid Search with Reciprocal Rank Fusion (RRF)
Combines semantic vector search (`sentence-transformers/all-MiniLM-L6-v2`) with lexical keyword scoring (`BM25Okapi`). Scores are fused using:
$$RRF\_Score(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
This balances exact keyword matches (policy figures, section titles) with semantic intent.

### 4. Deterministic Multi-Stage Alignment Guardrails
Rather than trusting LLM self-evaluation, the pipeline applies deterministic Python guardrails:
1. **Explicit Support Flag:** Verifies `is_supported` is `True`.
2. **Confidence Threshold:** Enforces $0.0 \le \text{confidence} \le 1.0$ and $\text{confidence} \ge 0.65$.
3. **Index Membership Check:** Validates that `page_number` matches a printed page in the retrieved chunks.
4. **Hyphen-Safe Token Overlap:** Verifies that at least 50% of the alphanumeric tokens in `evidence_quote` or `answer` appear in the normalized context.
5. **Metadata Snapping:** Snaps `section_title` directly to the verified chunk metadata to avoid hallucinated section names.
If any check fails, the pipeline returns the exact string `Not found`.

---

## Quick Start & Installation

### 1. Setup Environment
```bash
cd /Users/tusharbhojwani/Downloads/Screenshot/HR_Policy_Assistant

# Create and activate a virtual environment (optional if already configured)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Copy `.env.example` to `.env` and set your preferred provider key:
```bash
cp .env.example .env
```
Add one of the following to `.env`:
```ini
GOOGLE_API_KEY=your-gemini-api-key
# OR
GROQ_API_KEY=your-groq-api-key
# OR
OPENAI_API_KEY=your-openai-api-key
```
*(Note: If no API key is provided, the assistant automatically falls back to deterministic offline verification mode with full retrieval and guardrail execution).*

---

## Running the System

### 1. Ingestion Pipeline (`ingest.py`)
Re-processes `hr_manual.pdf` and populates `./chroma_db`:
```bash
python ingest.py
```
*Output: Extracted 268 chunks with verified printed page footers and stateful section headings.*

### 2. CLI & Exam Mode (`main.py`)
Run the 5 official practical exam questions:
```bash
python main.py --exam
```
Run interactive multi-turn conversation mode:
```bash
python main.py
```
*(Type questions naturally, or enter `exit` / `quit` to leave).*

### 3. LLMOps Evaluation & SFT Export (`evaluate.py`)
Executes all 5 exam questions + 1 negative control question, asserts correctness, computes latency, and exports artifacts:
```bash
python evaluate.py
```
- **Output:** `eval_report.json` (100% accuracy, ~22 ms average latency in offline mode).
- **Dataset:** `data/sft_alignment_dataset.jsonl` (JSONL format for supervised fine-tuning / DPO alignment).

### 4. Offline Unit Tests (`tests/`)
Run the full 42-test offline suite (no API key required):
```bash
pytest -v tests/
```
```text
======================== 42 passed in 8.38s ========================
```

### 5. Streamlit Web UI (`app.py`)
Launch the interactive web portal with Chat, Exam Mode Inspector, and Metrics:
```bash
streamlit run app.py
# Web app runs on http://localhost:8501
```

### 6. FastAPI Microservice (`api.py`)
Launch the production REST API:
```bash
python api.py
# Server runs on http://0.0.0.0:8000
```
- **Health Check:**
  ```bash
  curl http://localhost:8000/health
  # {"status":"healthy","engine_loaded":true}
  ```
- **Query Endpoint:**
  ```bash
  curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "How many days of maternity leave are given?"}'
  ```

### 7. Model Context Protocol Server (`mcp_server.py`)
Runs the standard stdio MCP server for agent integrations:
```bash
python mcp_server.py
```
Exposes two tools:
- `ask_hr_policy(question: str) -> str`
- `verify_policy_page(page_number: int) -> str`

### 8. Agentic Tool-Calling Demo (`agent_demo.py`)
Demonstrates multi-step agent reasoning where an agent queries the policy manual and autonomously validates the page citation:
```bash
python agent_demo.py
```

---

## Configuration & Provider Support

`config.py` automatically detects configured providers in priority order:
1. **Google Gemini:** Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` (Default model: `gemini-2.5-flash`).
2. **Groq:** Set `GROQ_API_KEY` (Default model: `llama-3.3-70b-versatile`).
3. **OpenAI:** Set `OPENAI_API_KEY` (Default model: `gpt-4o-mini`).
4. **Offline / Mock Mode:** Set `MOCK_LLM=1` or run with `--mock` for local deterministic evaluation.

Configurable constants in `config.py`:
- `CHUNK_SIZE = 900`
- `CHUNK_OVERLAP = 180`
- `DENSE_K = 5`
- `BM25_K = 5`
- `RRF_TOP_K = 6`
- `CONFIDENCE_THRESHOLD = 0.65`
- `TOKEN_OVERLAP_THRESHOLD = 0.50`

---

## Test Suite Breakdown

The offline test suite in `tests/` contains 42 tests organized across 3 modules:

1. **`test_metadata.py` (20 tests):**
   - Broken hyphen cleaning: single hyphen, double hyphen (`al--\nlowed`), spaces.
   - Printed page extraction from footer stamps (`HR Manual - 42`, `HR Manual - 62`).
   - Annexure detection (Roman numerals I–IV) and TOC exclusion.
   - Stateful section inheritance across continuation pages.
   - Integration checks on Pages 42, 60, and 62.

2. **`test_guardrails.py` (13 tests):**
   - Tokenization filtering (< 3 chars, special symbols).
   - Token overlap calculation across context passages.
   - Guardrail acceptance of well-grounded answers.
   - Guardrail rejection of out-of-bounds confidence ($<0.65$ or $>1.0$).
   - Guardrail rejection of hallucinated page numbers not present in retrieved context.
   - Rejection of unsupported claims.

3. **`test_formatting.py` (9 tests):**
   - Verification of exact supported format: `Answer: ... \n\nReference: HR Policy Manual, page X, section Y`.
   - Verification of exact unsupported format: strictly `Not found` with no trailing punctuation.
