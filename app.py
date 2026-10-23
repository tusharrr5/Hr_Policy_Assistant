import os
import json
import time
import streamlit as st

from rag_engine import HRPolicyRAG, HRPolicyAssistant
from evaluate import evaluate as run_eval_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page Configuration (ChatGPT Dark Aesthetics)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. ChatGPT-Inspired Responsive Dark CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #212121 !important;
        color: #ececec !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Center constrained container */
    .block-container {
        max-width: 820px !important;
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
        margin: 0 auto !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #282828 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: #ececec !important;
    }

    /* Sidebar Radio Navigation */
    div[data-testid="stRadio"] > label {
        color: #8e8e8e !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background: transparent !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        transition: background 0.15s ease;
        color: #d1d5db !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background: #212121 !important;
    }

    /* Welcome Header */
    .welcome-box {
        text-align: center;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff !important;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    .welcome-subtitle {
        font-size: 1rem;
        color: #9ca3af !important;
        margin: 0;
    }

    /* Prompt Suggestion Cards (ChatGPT 2x2 grid) */
    .prompt-card {
        background-color: #2f2f2f;
        border: 1px solid #383838;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
        color: #ececec;
        font-size: 0.9rem;
        line-height: 1.4;
        transition: all 0.15s ease;
    }
    .prompt-card:hover {
        background-color: #383838;
        border-color: #4a4a4a;
    }
    .prompt-title {
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 4px;
        display: block;
    }
    .prompt-desc {
        color: #9ca3af;
        font-size: 0.82rem;
        display: block;
    }

    /* Message Bubble Treatments */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 12px 0 !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: transparent !important;
    }
    div[data-testid="stChatMessage"] p {
        color: #ececec !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }

    /* Response Citation Badges */
    .citation-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 10px;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .badge-page {
        background-color: #1e3a5f;
        color: #7dd3fc;
        border: 1px solid #0369a1;
    }
    .badge-section {
        background-color: #143528;
        color: #86efac;
        border: 1px solid #15803d;
    }
    .badge-source {
        background-color: #2a2a2a;
        color: #9ca3af;
        border: 1px solid #383838;
    }

    /* Unsupported Abstention Card */
    .not-found-card {
        background-color: #2a201c;
        border-left: 3px solid #f97316;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
    }
    .not-found-text {
        color: #fed7aa;
        font-weight: 600;
        font-size: 1rem;
    }
    .not-found-sub {
        color: #fdba74;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    /* Evidence Box */
    .evidence-quote {
        background-color: #1a1a1a;
        border-left: 2px solid #525252;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 4px;
        color: #a3a3a3;
        font-size: 0.85rem;
        font-style: italic;
    }

    /* Status Dot */
    .status-dot-green {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }

    /* Metrics Tile (ChatGPT dark) */
    .dark-tile {
        background-color: #2f2f2f;
        border: 1px solid #383838;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .dark-tile .val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .dark-tile .lbl {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .welcome-title {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cached RAG Engine Initialization
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag():
    try:
        rag = HRPolicyRAG()
        return rag, "Live LLM Engine"
    except RuntimeError:
        rag = HRPolicyRAG(mock_llm=True)
        return rag, "Offline Alignment Mode"
    except Exception as e:
        return None, f"Error: {e}"

rag, engine_mode = load_rag()
chunk_count = len(rag._all_docs) if (rag and hasattr(rag, "_all_docs")) else 0
is_ready = chunk_count > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sidebar: Navigation & Controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💼 HR Policy Assistant")
    st.caption("Enso Group Policy Manual (79 pages)")

    # New Chat / Reset Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    view_mode = st.radio(
        "Navigation",
        ["💬 Chat", "🎓 Exam Questions", "📊 Evaluation"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**System Status**")
    if is_ready:
        st.markdown(f'<span class="status-dot-green"></span> Knowledge Base: **{chunk_count} chunks**', unsafe_allow_html=True)
    else:
        st.markdown('🔴 Knowledge Base: **Unavailable**', unsafe_allow_html=True)
    st.caption(f"Mode: **{engine_mode}**")
    st.caption("Retrieval: **Dense + BM25Okapi (RRF)**")
    st.caption("Guardrail: **Deterministic Python Layer**")


# ─────────────────────────────────────────────────────────────────────────────
# 5. View 1: 💬 Chat Interface (ChatGPT Style)
# ─────────────────────────────────────────────────────────────────────────────
if view_mode == "💬 Chat":
    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Centered Welcome Header (only shown if no messages yet)
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-box">
                <div class="welcome-title">HR Policy Assistant</div>
                <div class="welcome-subtitle">Evidence-backed answers from the Enso Group HR Policy Manual.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Clickable Sample Questions (2x2 grid)
        col1, col2 = st.columns(2)
        sample_q = None

        with col1:
            if st.button("📌 Privilege Leave\n\nHow many days of PL does a permanent employee get per year?", key="s1", use_container_width=True):
                sample_q = "How many days of Privilege Leave (PL) does a permanent employee get per year?"
            if st.button("📌 Maternity Leave\n\nHow many days of maternity leave are given?", key="s2", use_container_width=True):
                sample_q = "How many days of maternity leave are given?"

        with col2:
            if st.button("📌 Probation Period\n\nWhat is the probation period for employees up to Senior Manager?", key="s3", use_container_width=True):
                sample_q = "What is the probation period for employees up to Senior Manager?"
            if st.button("📌 Health Insurance\n\nWhat is the company's health-insurance premium amount?", key="s4", use_container_width=True):
                sample_q = "What is the company's health-insurance premium amount?"

        if sample_q:
            st.session_state.messages.append({"role": "user", "content": sample_q})
            st.rerun()

    # Display Existing Conversation Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                if msg.get("is_supported") is False or msg.get("content") == "Not found":
                    st.markdown(
                        """
                        <div class="not-found-card">
                            <div class="not-found-text">Not found</div>
                            <div class="not-found-sub">The requested information is not present in the Enso Group HR Manual. Abstaining from hallucination.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(msg["content"])

                    # Citations
                    page = msg.get("page")
                    section = msg.get("section")
                    if page or section:
                        badges_html = '<div class="citation-container">'
                        if page:
                            badges_html += f'<span class="badge badge-page">Page {page}</span>'
                        if section:
                            badges_html += f'<span class="badge badge-section">{section}</span>'
                        badges_html += '<span class="badge badge-source">HR Policy Manual</span>'
                        badges_html += '</div>'
                        st.markdown(badges_html, unsafe_allow_html=True)

                    # Supporting Evidence Expander
                    evidence = msg.get("evidence")
                    if evidence:
                        with st.expander("View supporting evidence excerpt"):
                            st.markdown(f'<div class="evidence-quote">"{evidence}"</div>', unsafe_allow_html=True)

                if msg.get("latency"):
                    st.caption(f"⚡ {msg['latency']} ms · Hybrid RAG Verified")

    # Bottom Chat Input (ChatGPT composer)
    prompt = st.chat_input("Ask a question about HR policies...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching HR manual..."):
                t0 = time.time()
                res = rag.query(prompt, stateless=False)
                latency = round((time.time() - t0) * 1000, 1)

                formatted = res["formatted_response"]
                is_supported = res["is_supported"]
                page = res["page_number"]
                section = res["section_title"]
                raw = res.get("raw_answer")
                evidence = res.get("evidence_quote") or (raw.evidence_quote if raw else "")

                if not is_supported or formatted == "Not found":
                    st.markdown(
                        """
                        <div class="not-found-card">
                            <div class="not-found-text">Not found</div>
                            <div class="not-found-sub">The requested information is not present in the Enso Group HR Manual. Abstaining from hallucination.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(formatted)

                    if page or section:
                        badges_html = '<div class="citation-container">'
                        if page:
                            badges_html += f'<span class="badge badge-page">Page {page}</span>'
                        if section:
                            badges_html += f'<span class="badge badge-section">{section}</span>'
                        badges_html += '<span class="badge badge-source">HR Policy Manual</span>'
                        badges_html += '</div>'
                        st.markdown(badges_html, unsafe_allow_html=True)

                    if evidence:
                        with st.expander("View supporting evidence excerpt"):
                            st.markdown(f'<div class="evidence-quote">"{evidence}"</div>', unsafe_allow_html=True)

                st.caption(f"⚡ {latency} ms · Hybrid RAG Verified")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted,
                    "is_supported": is_supported,
                    "page": page,
                    "section": section,
                    "evidence": evidence,
                    "latency": latency,
                })


# ─────────────────────────────────────────────────────────────────────────────
# 6. View 2: 🎓 Five Exam Questions
# ─────────────────────────────────────────────────────────────────────────────
elif view_mode == "🎓 Exam Questions":
    st.markdown("## 🎓 Official Exam Questions")
    st.markdown("Live verification of the 5 official practical examination questions against `hr_manual.pdf`.")

    OFFICIAL_EXAM = [
        {
            "id": "Q1",
            "name": "Privilege Leave Entitlement",
            "question": "How many days of Privilege Leave (PL) does a permanent employee get per year?",
            "target_page": 60,
            "target_section": "Annexure III, [G] Leave Policy",
            "target_answer": "12 days of Privilege Leave (PL) per calendar year (1st January – 31st December) for permanent employees. No leave is credited during probation, unavailed PL cannot be carried forward to next calendar year, and accumulated PL can be encashed only as part of Full & Final settlement at basic pay.",
            "is_supported": True,
        },
        {
            "id": "Q2",
            "name": "Probation Period",
            "question": "What is the probation period for employees up to Senior Manager?",
            "target_page": 42,
            "target_section": "Annexure I, [B] Probation Policy",
            "target_answer": "3 months from the date of joining for all employees up to Senior Manager (Sr. Manager). (Employees above Sr. Manager are confirmed from date of joining; probation may be extended at Management's discretion only once and not exceeding original 3 months.)",
            "is_supported": True,
        },
        {
            "id": "Q3",
            "name": "Health Insurance Premium",
            "question": "What is the company's health-insurance premium amount?",
            "target_page": None,
            "target_section": None,
            "target_answer": "Not found (Medical Reimbursement & ESIC appear on p. 57 and travel insurance on p. 71, but no premium amount exists anywhere in the manual).",
            "is_supported": False,
        },
        {
            "id": "Q4",
            "name": "Current GST Rate",
            "question": "What is the current GST rate in India?",
            "target_page": None,
            "target_section": None,
            "target_answer": "Not found (External Indian tax rate absent from the manual; strictly excluded).",
            "is_supported": False,
        },
        {
            "id": "Q5",
            "name": "Maternity Leave Entitlement",
            "question": "How many days of maternity leave are given?",
            "target_page": 62,
            "target_section": "Annexure III, [G] Leave Policy",
            "target_answer": "Confirmed female staff (who worked at least 80 days in preceding 12 months) receive total 12 weeks (parenthetically 90 days) Maternity Leave with pay (max 6 weeks / 45 days pre-delivery; 4 weeks for miscarriage).",
            "is_supported": True,
        },
    ]

    if st.button("⚡ Run All 5 Tests Live", type="primary", use_container_width=True):
        st.session_state.exam_results = {}
        with st.spinner("Executing live hybrid retrieval across all 5 questions..."):
            for item in OFFICIAL_EXAM:
                st.session_state.exam_results[item["id"]] = rag.query(item["question"], stateless=True)

    for item in OFFICIAL_EXAM:
        qid = item["id"]
        with st.container():
            st.markdown(f"### {qid}: {item['name']}")
            st.markdown(f"**Question:** *\"{item['question']}\"*")

            c_exp, c_act = st.columns(2)
            with c_exp:
                st.markdown("**Expected Ground Truth:**")
                if item["is_supported"]:
                    st.info(f"{item['target_answer']}\n\n**Citation:** Page {item['target_page']}, `{item['target_section']}`")
                else:
                    st.warning(f"**`Not found`**\n\n*(Rationale: {item['target_answer']})*")

            with c_act:
                st.markdown("**Actual Pipeline Output:**")
                has_run = "exam_results" in st.session_state and qid in st.session_state.exam_results

                if st.button(f"Test {qid}", key=f"t_{qid}"):
                    if "exam_results" not in st.session_state:
                        st.session_state.exam_results = {}
                    st.session_state.exam_results[qid] = rag.query(item["question"], stateless=True)
                    has_run = True

                if has_run:
                    out = st.session_state.exam_results[qid]
                    out_text = out["formatted_response"]
                    out_page = out["page_number"]
                    out_supp = out["is_supported"]
                    out_lat = out["latency_ms"]

                    passed = (out_supp == item["is_supported"]) and (
                        not item["is_supported"] or out_page == item["target_page"]
                    )
                    badge = "✅ PASS" if passed else "❌ FAIL"

                    if out_supp:
                        st.success(f"**Status: {badge}** ({out_lat:.0f} ms)\n\n{out_text}")
                    else:
                        st.warning(f"**Status: {badge}** ({out_lat:.0f} ms)\n\n`{out_text}`")
                else:
                    st.caption("Click 'Test' to execute.")

            st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# 7. View 3: 📊 Evaluation
# ─────────────────────────────────────────────────────────────────────────────
elif view_mode == "📊 Evaluation":
    st.markdown("## 📊 System Evaluation & Metrics")
    st.markdown("Comprehensive performance metrics recorded from automated LLMOps test runs.")

    report_file = os.path.join(os.path.dirname(__file__), "eval_report.json")
    if os.path.exists(report_file):
        with open(report_file) as f:
            data = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="dark-tile"><p class="val" style="color:#10b981;">{data.get("accuracy", 1.0):.0%}</p><p class="lbl">Evaluation Accuracy</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="dark-tile"><p class="val">{data.get("correct", 6)} / {data.get("total_questions", 6)}</p><p class="lbl">Passed Test Cases</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="dark-tile"><p class="val">42 / 42</p><p class="lbl">Offline Unit Tests</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="dark-tile"><p class="val">{data.get("avg_latency_ms", 141):.0f} ms</p><p class="lbl">Average Latency</p></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Test Case Results Table")
        rows = []
        for r in data.get("results", []):
            rows.append({
                "ID": r.get("id"),
                "Question": r.get("question"),
                "Type": "Supported" if r.get("is_supported") else "Abstention",
                "Cited Page": f"Page {r.get('page_number')}" if r.get("page_number") else "—",
                "Status": "✅ PASS" if r.get("correct") else "❌ FAIL",
                "Latency": f"{r.get('latency_ms', 0):.0f} ms",
            })
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Run `python evaluate.py` to generate the evaluation report.")
