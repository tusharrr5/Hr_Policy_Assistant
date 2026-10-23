"""
config.py — Central configuration for the HR Policy Assistant.
Auto-detects the LLM provider from environment variables and exposes
shared constants (paths, embedding model, ChromaDB collection name).
"""

import os
from dotenv import load_dotenv

# ── Load .env (if present) ──────────────────────────────────────────────
load_dotenv()

# ── Paths & Constants ───────────────────────────────────────────────────
PDF_PATH = os.path.join(os.path.dirname(__file__), "hr_manual.pdf")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "enso_hr_policy"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180

# ── Retrieval settings ──────────────────────────────────────────────────
DENSE_K = 5
BM25_K = 5
RRF_TOP_K = 6
CONFIDENCE_THRESHOLD = 0.65
TOKEN_OVERLAP_THRESHOLD = 0.50

# ── LLM provider detection ─────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()


def get_llm():
    """Return a LangChain LLM instance based on the configured provider."""
    provider = LLM_PROVIDER

    # Google (Gemini)
    if provider in ("google", "gemini") or (
        provider == "auto"
        and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    ):
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        model = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model, temperature=0, google_api_key=api_key
        )

    # Groq
    if provider == "groq" or (provider == "auto" and os.getenv("GROQ_API_KEY")):
        from langchain_groq import ChatGroq

        model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        return ChatGroq(model=model, temperature=0)

    # OpenAI
    if provider == "openai" or (provider == "auto" and os.getenv("OPENAI_API_KEY")):
        from langchain_openai import ChatOpenAI

        model = os.getenv("MODEL_NAME", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    raise RuntimeError(
        "No LLM API key found. Set GOOGLE_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY "
        "in your environment or .env file."
    )


def get_embeddings():
    """Return the HuggingFace embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
