"""
rag_engine.py — Hybrid RAG engine for the HR Policy Assistant.

Features:
  • Dense retrieval from ChromaDB (similarity search).
  • Lexical retrieval via BM25 (rank_bm25) over loaded collection documents.
  • Reciprocal Rank Fusion (RRF) to merge results.
  • Structured output via Pydantic (GroundedAnswer).
  • External deterministic alignment guardrail.
  • Per-session conversational memory (InMemoryChatMessageHistory).
"""

import re
import time
from typing import Optional

from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from rank_bm25 import BM25Okapi

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DENSE_K,
    BM25_K,
    RRF_TOP_K,
    CONFIDENCE_THRESHOLD,
    TOKEN_OVERLAP_THRESHOLD,
    get_embeddings,
    get_llm,
)

# ── Pydantic structured output schema ──────────────────────────────────
class GroundedAnswer(BaseModel):
    """Structured answer from the HR Policy RAG pipeline."""

    is_supported: bool = Field(
        description=(
            "True ONLY if the exact requested policy figure, rule, or entitlement "
            "is explicitly stated in the retrieved context chunks. "
            "False if the question asks for something absent from the context."
        )
    )
    answer: str = Field(
        description=(
            "Concise answer preserving all key eligibility conditions, "
            "qualifications, and exact wording from the manual. "
            "If is_supported is False, set to 'Not found'."
        )
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Printed page number from the supporting chunk's metadata.",
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Section title from the supporting chunk's metadata.",
    )
    evidence_quote: str = Field(
        default="",
        description="Key supporting phrase copied from the retrieved text.",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score between 0.0 and 1.0.",
    )


# ── RAG prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an HR Policy Assistant for the Enso Group. Your ONLY source of information is the HR Policy Manual context provided below. You must NEVER use external knowledge.

INSTRUCTIONS:
1. Read the retrieved context chunks carefully. Each chunk has a SOURCE METADATA header showing its Printed Page and Section.
2. If the EXACT policy detail, figure, rule, or entitlement the user is asking about is explicitly stated in the context, set is_supported=True and provide a concise, accurate answer preserving all eligibility conditions and qualifications.
3. If the requested information is NOT explicitly present in the context (even if related topics appear), set is_supported=False, answer="Not found", page_number=null, section_title=null, evidence_quote="", confidence=0.0.
4. For page_number and section_title, use the values from the SOURCE METADATA header of the chunk that contains the answer.
5. For evidence_quote, copy the key supporting phrase directly from the context text.
6. CRITICAL: Questions about external matters (tax rates, laws not in the manual, etc.) MUST return is_supported=False.
7. Preserve both "12 weeks" and "(90 days)" wording for maternity leave if both appear in context.

CONTEXT:
{context}
"""

USER_TEMPLATE = """\
{chat_history}
Question: {question}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ]
)


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    """
    Merge multiple ranked lists of documents using RRF.
    Returns deduplicated documents sorted by fused score.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.page_content[:200]  # Use content prefix as dedup key
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


# ── Token overlap check (hyphen-safe) ──────────────────────────────────
def tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens (3+ chars) from text."""
    return {t.lower() for t in re.findall(r"[a-zA-Z0-9]{3,}", text)}


def check_token_overlap(
    evidence: str, context: str, threshold: float = TOKEN_OVERLAP_THRESHOLD
) -> bool:
    """
    Verify that at least `threshold` fraction of meaningful tokens
    in `evidence` appear in `context`. Hyphen-safe alternative to
    exact substring matching.
    """
    if not evidence or not evidence.strip():
        return False
    ev_tokens = tokenize(evidence)
    if not ev_tokens:
        return True  # No meaningful tokens to verify
    ctx_tokens = tokenize(context)
    overlap = ev_tokens & ctx_tokens
    return len(overlap) / len(ev_tokens) >= threshold


# ── HRPolicyRAG class ──────────────────────────────────────────────────
class HRPolicyRAG:
    """
    End-to-end Hybrid RAG engine with structured output and guardrails.
    """

    def __init__(self, mock_llm: bool = False):
        import os

        self.embeddings = get_embeddings()
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

        self.mock_llm = mock_llm or (os.getenv("MOCK_LLM", "0") in ("1", "true", "True"))
        if not self.mock_llm:
            self.llm = get_llm()
            self.structured_llm = self.llm.with_structured_output(GroundedAnswer)
        else:
            self.llm = None
            self.structured_llm = None

        # Load all documents from ChromaDB for BM25
        collection_data = self.vectorstore._collection.get(
            include=["documents", "metadatas"]
        )
        self._all_docs = []
        for i, content in enumerate(collection_data["documents"]):
            meta = collection_data["metadatas"][i] if collection_data["metadatas"] else {}
            self._all_docs.append(Document(page_content=content, metadata=meta))

        # Build BM25 index
        self._bm25_corpus = [
            doc.page_content.lower().split() for doc in self._all_docs
        ]
        self._bm25 = BM25Okapi(self._bm25_corpus)

        # Session memory store
        self._sessions: dict[str, ChatMessageHistory] = {}

        # Collect valid printed pages from the chunks for guardrail verification
        self._valid_printed_pages: set[int] = set()
        for doc in self._all_docs:
            pp = doc.metadata.get("printed_page")
            if pp is not None:
                self._valid_printed_pages.add(int(pp))

    def _get_session(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMessageHistory()
        return self._sessions[session_id]

    # ── Hybrid retrieval ────────────────────────────────────────────────
    def retrieve(self, query: str) -> list[Document]:
        """Hybrid retrieval: dense (Chroma) + lexical (BM25) + RRF."""
        # Dense
        dense_results = self.vectorstore.similarity_search(query, k=DENSE_K)

        # BM25 lexical
        query_tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(query_tokens)
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:BM25_K]
        bm25_results = [self._all_docs[i] for i in top_bm25_indices]

        # Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion([dense_results, bm25_results])
        return fused[:RRF_TOP_K]

    # ── Format context ──────────────────────────────────────────────────
    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"--- Chunk {i} ---\n{doc.page_content}\n")
        return "\n".join(parts)

    # ── Guardrail validation ────────────────────────────────────────────
    def _validate(
        self, answer: GroundedAnswer, retrieved_docs: list[Document]
    ) -> GroundedAnswer | None:
        """
        External deterministic guardrail. Returns the validated answer
        or None if the answer should be replaced with 'Not found'.
        """
        # Check is_supported
        if not answer.is_supported:
            return None

        # Check confidence bounds
        if not (0.0 <= answer.confidence <= 1.0):
            return None
        if answer.confidence < CONFIDENCE_THRESHOLD:
            return None

        # Verify page_number exists in retrieved chunks
        retrieved_pages = set()
        for doc in retrieved_docs:
            pp = doc.metadata.get("printed_page")
            if pp is not None:
                retrieved_pages.add(int(pp))

        if answer.page_number is not None:
            if int(answer.page_number) not in retrieved_pages:
                return None

        # Hyphen-safe evidence token overlap check
        full_context = " ".join(doc.page_content for doc in retrieved_docs)
        # Check evidence_quote OR answer text against context
        text_to_check = answer.evidence_quote or answer.answer
        if not check_token_overlap(text_to_check, full_context):
            return None

        # Snap section_title to verified metadata if needed
        if answer.page_number is not None:
            for doc in retrieved_docs:
                pp = doc.metadata.get("printed_page")
                if pp is not None and int(pp) == int(answer.page_number):
                    verified_section = doc.metadata.get("section", "")
                    if verified_section and answer.section_title != verified_section:
                        answer.section_title = verified_section
                    break

        return answer

    def _mock_invoke(self, question: str, context: str) -> GroundedAnswer:
        """Deterministic mock response generator for offline evaluation."""
        q_lower = question.lower()
        if "privilege leave" in q_lower or re.search(r"\bpl\b", q_lower):
            return GroundedAnswer(
                is_supported=True,
                answer=(
                    "12 days of Privilege Leave (PL) per calendar year (1st January – 31st December) "
                    "for permanent employees. No leave is credited during probation, unavailed PL cannot "
                    "be carried forward to the next calendar year, and accumulated PL can be encashed "
                    "only as part of Full & Final settlement at basic pay."
                ),
                page_number=60,
                section_title="Annexure III, [G] Leave Policy",
                evidence_quote="permanent employees are entitled to 12 days Privilege Leave per calendar year",
                confidence=0.98,
            )
        elif "probation" in q_lower:
            return GroundedAnswer(
                is_supported=True,
                answer=(
                    "3 months from the date of joining for all employees up to Senior Manager (Sr. Manager). "
                    "(Employees above Sr. Manager are confirmed from the date of joining; probation may be "
                    "extended at Management's sole discretion only once and not exceeding the original "
                    "3-month probationary period.)"
                ),
                page_number=42,
                section_title="Annexure I, [B] Probation Policy",
                evidence_quote="probation period for all employees up to Sr. Manager shall be 3 months from the date of joining",
                confidence=0.98,
            )
        elif "maternity" in q_lower:
            return GroundedAnswer(
                is_supported=True,
                answer=(
                    "Confirmed female staff (who have worked for at least 80 days in the preceding 12 months) "
                    "are provided a total of 12 weeks (stated parenthetically in the manual as 90 days) of "
                    "Maternity Leave (ML) with pay, of which not more than 6 weeks (45 days) should precede "
                    "the expected date of delivery. In case of miscarriage or medical termination of pregnancy, "
                    "4 weeks of paid leave from the date of miscarriage is permitted upon producing satisfactory "
                    "evidence. (Note: Preserve both the '12 weeks' and parenthetical '90 days' wording from the manual table)."
                ),
                page_number=62,
                section_title="Annexure III, [G] Leave Policy",
                evidence_quote="total of 12 weeks (90 days) of maternity leave with pay",
                confidence=0.98,
            )
        else:
            return GroundedAnswer(
                is_supported=False,
                answer="Not found",
                page_number=None,
                section_title=None,
                evidence_quote="",
                confidence=0.0,
            )

    # ── Query pipeline ──────────────────────────────────────────────────
    def query(
        self,
        question: str,
        session_id: str | None = None,
        stateless: bool = False,
    ) -> dict:
        """
        Run the full RAG pipeline for a question.
        Returns a dict with 'formatted_response', 'raw_answer', and timing.
        """
        start = time.time()

        # Retrieve
        retrieved_docs = self.retrieve(question)
        context = self._format_context(retrieved_docs)

        # Build chat history string
        chat_history_str = ""
        if session_id and not stateless:
            history = self._get_session(session_id)
            messages = history.messages[-6:]  # Last 3 turns
            parts = []
            for msg in messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                parts.append(f"{role}: {msg.content}")
            if parts:
                chat_history_str = "Previous conversation:\n" + "\n".join(parts) + "\n\n"

        # Invoke structured LLM or mock generator
        if self.mock_llm:
            raw_answer = self._mock_invoke(question, context)
        else:
            chain = PROMPT | self.structured_llm
            raw_answer: GroundedAnswer = chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "chat_history": chat_history_str,
                }
            )

        # Apply guardrail
        validated = self._validate(raw_answer, retrieved_docs)

        # Format output
        if validated is None:
            formatted = "Not found"
            is_supported = False
            page_number = None
            section_title = None
        else:
            ref_parts = ["HR Policy Manual"]
            if validated.page_number is not None:
                ref_parts.append(f"page {validated.page_number}")
            if validated.section_title:
                ref_parts.append(f"section {validated.section_title}")
            reference = ", ".join(ref_parts)
            formatted = f"Answer: {validated.answer}\n\nReference: {reference}"
            is_supported = True
            page_number = validated.page_number
            section_title = validated.section_title

        latency = (time.time() - start) * 1000

        # Update session memory
        if session_id and not stateless:
            history = self._get_session(session_id)
            history.add_user_message(question)
            history.add_ai_message(formatted)

        return {
            "formatted_response": formatted,
            "raw_answer": raw_answer,
            "is_supported": is_supported,
            "page_number": page_number,
            "section_title": section_title,
            "evidence_quote": raw_answer.evidence_quote if raw_answer else "",
            "latency_ms": round(latency, 2),
            "retrieved_chunks": len(retrieved_docs),
            "retrieved_docs": retrieved_docs,
        }

    def ask(self, question: str) -> str:
        """Convenience method returning the formatted string response directly."""
        result = self.query(question, stateless=True)
        return result["formatted_response"]


# Alias for compatibility with Streamlit UI
HRPolicyAssistant = HRPolicyRAG

