"""
AutoDoc Insight — AUTOSAR HLD Architecture Intelligence Platform
Developed for Tata Technologies Recruitment Drive — Case Study CS1 Capstone

Main Streamlit Application — Automotive Engineering Intelligence Dashboard.
Features:
- Page-aware PDF ingestion & sliding-window semantic chunking
- Dense semantic vector search via FAISS & local BGE embeddings
- Multi-provider LLM integration (Local Ollama Llama 3.1, Cloud OpenAI, Mock mode)
- Automated architecture element extraction (Components, Interfaces, Ports, Signals)
- Interactive visual architecture dependency diagrams (Mermaid)
- Potential inconsistency & conflict detection with page-level citations
- Comprehensive, audit-ready full analysis report export (.md / .json)
- Transparent evaluation benchmark suite with real metrics
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Path fix: run from project root ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
import json
import pandas as pd
import streamlit as st

from config.settings import DATA_INPUT_DIR
from app.ingestion.pdf_extractor import extract_pages, get_document_stats
from app.ingestion.chunker import chunk_pages
from app.rag.vector_store import VectorStore
from app.rag.pipeline import ask
from app.rag.llm_client import is_mock_mode, get_llm_label, get_llm_setup_message
from app.extraction.arch_extractor import (
    extract_architecture, detect_inconsistencies, generate_summary,
)
from app.evaluation.evaluator import run_evaluation, SYNTHETIC_QA
from app.utils.helpers import project_id_for_files, format_chunk_as_evidence
from app.utils.synthetic_hld import create_synthetic_hld
from app.reporting.report_generator import (
    generate_full_analysis_report, generate_mermaid_diagram,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoDoc Insight | AUTOSAR Architecture Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Advanced Cyber-Automotive Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Design Tokens ────────────────────────────────────────────────────────── */
:root {
  --bg-primary:     #080c14;
  --bg-secondary:   #0f172a;
  --bg-tertiary:    #1e293b;
  --accent-cyan:    #00d2ff;
  --accent-blue:    #3b82f6;
  --accent-indigo:  #6366f1;
  --accent-emerald: #10b981;
  --accent-amber:   #f59e0b;
  --accent-rose:    #f43f5e;
  --text-main:      #f8fafc;
  --text-muted:     #94a3b8;
  --border-subtle:  rgba(255, 255, 255, 0.08);
  --border-glow:    rgba(0, 210, 255, 0.3);
  --card-shadow:    0 10px 30px -10px rgba(0, 0, 0, 0.6);
}

/* ── Base ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-main);
}
.stApp {
  background: radial-gradient(circle at 15% 15%, #0f1c30 0%, #080c14 60%, #05080e 100%);
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
  border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--accent-cyan);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border-subtle);
  margin-top: 14px;
  margin-bottom: 10px;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: linear-gradient(145deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.5));
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 16px 20px;
  backdrop-filter: blur(12px);
  transition: all 0.25s ease-in-out;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}
[data-testid="metric-container"]:hover {
  transform: translateY(-3px);
  border-color: var(--border-glow);
  box-shadow: 0 8px 28px rgba(0, 210, 255, 0.15);
}
[data-testid="stMetricValue"] {
  color: var(--accent-cyan) !important;
  font-weight: 800;
  font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stMetricLabel"] {
  color: var(--text-muted) !important;
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #2563eb 0%, #00b4d8 100%);
  color: #ffffff;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 9px;
  font-weight: 600;
  font-size: 0.86rem;
  letter-spacing: 0.02em;
  padding: 10px 22px;
  transition: all 0.25s ease;
  box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
}
.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 22px rgba(0, 210, 255, 0.55);
  border-color: rgba(255, 255, 255, 0.3);
}

/* ── Chat Messages ────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  border-radius: 12px;
  margin-bottom: 12px;
  border: 1px solid var(--border-subtle);
  background: rgba(15, 23, 42, 0.7);
  backdrop-filter: blur(8px);
}

/* ── Custom Content Card ──────────────────────────────────────────────────── */
.custom-card {
  background: linear-gradient(145deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.6));
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 22px 26px;
  margin-bottom: 18px;
  box-shadow: var(--card-shadow);
}
.custom-card-header {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── Badges ───────────────────────────────────────────────────────────────── */
.badge {
  display: inline-flex;
  align-items: center;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.badge-high    { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; }
.badge-medium  { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; }
.badge-low     { background: rgba(244, 63, 94, 0.15);  color: #fb7185; border: 1px solid #f43f5e; }
.badge-mock    { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid var(--border-subtle); }
.badge-tag     { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }

/* ── Evidence block ───────────────────────────────────────────────────────── */
.evidence-box {
  background: rgba(30, 41, 59, 0.6);
  border-left: 3px solid var(--accent-cyan);
  border-radius: 0 8px 8px 0;
  padding: 12px 16px;
  margin: 10px 0;
  font-size: 0.82rem;
  font-family: 'JetBrains Mono', monospace;
  color: #e2e8f0;
}

/* ── Banners ──────────────────────────────────────────────────────────────── */
.status-banner {
  background: linear-gradient(90deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.03));
  border: 1px solid rgba(245, 158, 11, 0.4);
  border-radius: 10px;
  padding: 12px 18px;
  font-size: 0.84rem;
  color: #fbbf24;
  margin-bottom: 14px;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(15, 23, 42, 0.9);
  border-radius: 10px;
  padding: 4px;
  border: 1px solid var(--border-subtle);
  gap: 6px;
}
.stTabs [data-baseweb="tab"] {
  font-weight: 600;
  font-size: 0.84rem;
  color: var(--text-muted);
  border-radius: 8px;
  padding: 8px 18px;
  transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(0, 210, 255, 0.15)) !important;
  color: var(--accent-cyan) !important;
  border-bottom: 2px solid var(--accent-cyan) !important;
}

/* ── Dataframes & Tables ──────────────────────────────────────────────────── */
[data-testid="stTable"], [data-testid="stDataFrame"] {
  background: var(--bg-secondary);
  border-radius: 10px;
  border: 1px solid var(--border-subtle);
}

/* ── Scrollbars ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "store": None,
        "doc_stats": [],
        "chunk_count": 0,
        "doc_names": [],
        "chat_history": [],
        "arch_result": None,
        "inconsistencies": None,
        "summary_text": "",
        "eval_result": None,
        "project_id": None,
        "processed": False,
        "full_report_md": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:14px 0 10px 0;">
            <div style="font-size:2.4rem; filter: drop-shadow(0 0 12px rgba(0,210,255,0.4));">⚡</div>
            <div style="font-size:1.15rem; font-weight:800; letter-spacing:0.02em; background: linear-gradient(135deg, #ffffff 30%, #00d2ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                AutoDoc Insight
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">
                AUTOSAR Architecture Intelligence
            </div>
            <div style="margin-top:8px;">
                <span class="badge badge-tag">Tata TechPulse · Case Study CS1</span>
            </div>
        </div>
        <hr style="margin: 12px 0; border-color: rgba(255,255,255,0.08);"/>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📄 HLD Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload AUTOSAR HLD PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple AUTOSAR High-Level Design PDFs. PyMuPDF handles page-aware text and heading extraction.",
        key="uploader",
    )

    use_demo = st.checkbox(
        "Use Synthetic Demo HLD (AUTOSAR PCU)",
        value=not bool(uploaded_files),
        help="Loads an EV Powertrain Control Unit demo HLD with full components, interfaces, signals, and functional flows.",
    )

    st.markdown("### ⚙️ Engine Parameters")
    col_a, col_b = st.columns(2)
    with col_a:
        chunk_size = st.number_input("Chunk Size", value=600, min_value=200, max_value=2000, step=100, help="Sliding window chunk length in characters")
    with col_b:
        top_k = st.number_input("Top-K Retrieval", value=5, min_value=1, max_value=15, step=1, help="Number of nearest vector chunks to retrieve")

    process_btn = st.button("⚡ Process & Index Document", use_container_width=True, type="primary")

    st.markdown("### 📊 System Telemetry")

    if st.session_state.processed:
        store: VectorStore = st.session_state.store
        stats = store.get_stats()
        st.markdown(
            f"""
<div class="custom-card" style="padding:14px; margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
    <span style="color:#94a3b8;font-size:0.75rem;">Engine State</span>
    <span style="color:#10b981;font-size:0.75rem;font-weight:700;">● ONLINE / READY</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span style="color:#94a3b8;font-size:0.75rem;">Indexed Chunks</span>
    <span style="color:#f8fafc;font-size:0.75rem;font-weight:700;">{stats['n_chunks']}</span>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span style="color:#94a3b8;font-size:0.75rem;">Documents Ingested</span>
    <span style="color:#f8fafc;font-size:0.75rem;font-weight:700;">{len(stats['documents'])}</span>
  </div>
  <div style="display:flex;justify-content:space-between;">
    <span style="color:#94a3b8;font-size:0.75rem;">Vector Store</span>
    <span style="color:#00d2ff;font-size:0.75rem;font-weight:600;">FAISS IndexFlatIP</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.session_state.doc_names:
            st.markdown("<span style='font-size:0.75rem; color:#94a3b8; font-weight:600;'>ACTIVE DOCUMENTS:</span>", unsafe_allow_html=True)
            for dn in st.session_state.doc_names:
                st.markdown(f"- 📄 `{dn}`")
    else:
        st.markdown(
            '<div style="color:#94a3b8;font-size:0.8rem;padding:8px 0;">No HLD documents processed yet. Click the button above to begin.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    llm_label = get_llm_label()
    setup_msg = get_llm_setup_message()

    if is_mock_mode():
        st.markdown(
            '<div class="status-banner">'
            '<b>⚡ Standalone Mode (Demo / No LLM)</b><br/>'
            'Vector retrieval and page citations are 100% operational.<br/><br/>'
            '<b>To connect a local private LLM:</b><br/>'
            'Set <code>LLM_PROVIDER=ollama</code> in <code>.env</code><br/>'
            'and run <code>ollama pull llama3.1 && ollama serve</code>'
            '</div>',
            unsafe_allow_html=True,
        )
        if setup_msg:
            with st.expander("Setup Details"):
                st.code(setup_msg)
    else:
        st.markdown(
            f'<div style="color:#10b981;font-size:0.8rem;font-weight:600;">'
            f'🤖 Active LLM: {llm_label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="color:#64748b;font-size:0.7rem;margin-top:10px;">'
        'Embeddings: BAAI/bge-small-en-v1.5 (384-dim)<br/>'
        'Isolation: Deterministic SHA-256 Project Hash'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if process_btn:
    pdf_paths: list[Path] = []

    if use_demo or not uploaded_files:
        with st.spinner("Compiling synthetic AUTOSAR HLD specification..."):
            p = create_synthetic_hld(DATA_INPUT_DIR)
            pdf_paths = [p]
            st.sidebar.success(f"Generated Demo HLD: {p.name}")

    if uploaded_files and not use_demo:
        for uf in uploaded_files:
            save_path = DATA_INPUT_DIR / uf.name
            save_path.write_bytes(uf.read())
            pdf_paths.append(save_path)

    if not pdf_paths:
        st.sidebar.error("No valid PDF documents to process.")
    else:
        project_id = project_id_for_files(pdf_paths)
        store = VectorStore(project_id=project_id)

        if store.is_ready:
            store.reset()

        all_chunks = []
        doc_stats_list = []

        progress = st.sidebar.progress(0, text="Extracting page-aware text...")

        for i, pdf_path in enumerate(pdf_paths):
            if pdf_path.suffix == ".txt":
                text = pdf_path.read_text(encoding="utf-8")
                pages = [{"page": 1, "text": text, "heading": "Overview", "char_count": len(text), "is_scanned": False}]
            else:
                pages = extract_pages(pdf_path)

            stats = get_document_stats(pages)
            stats["doc_name"] = pdf_path.name
            doc_stats_list.append(stats)

            chunks = chunk_pages(pages, doc_name=pdf_path.name, chunk_size=int(chunk_size))
            all_chunks.extend(chunks)
            progress.progress((i + 1) / len(pdf_paths), text=f"Processed: {pdf_path.name}")

        if all_chunks:
            progress.progress(0.8, text="Computing dense embeddings & building FAISS index...")
            store.add_chunks(all_chunks, show_progress=False)
            progress.progress(1.0, text="Ready!")

            st.session_state.store           = store
            st.session_state.doc_stats       = doc_stats_list
            st.session_state.chunk_count     = store.chunk_count
            st.session_state.doc_names       = store.document_names
            st.session_state.project_id      = project_id
            st.session_state.processed       = True
            st.session_state.arch_result     = None
            st.session_state.inconsistencies = None
            st.session_state.summary_text    = ""
            st.session_state.eval_result     = None
            st.session_state.full_report_md  = ""

            st.sidebar.success(f"✅ Ingested {store.chunk_count} chunks from {len(pdf_paths)} document(s).")
        else:
            st.sidebar.error("No text could be extracted. Please ensure document has selectable text (not scanned images).")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:8px;padding-top:4px;">
  <div style="display:flex;align-items:center;gap:16px;">
    <span style="font-size:2.8rem; filter: drop-shadow(0 0 16px rgba(0,210,255,0.4));">🚗</span>
    <div>
      <h1 style="margin:0;font-size:2.1rem;font-weight:800;letter-spacing:-0.02em;background:linear-gradient(135deg, #ffffff 40%, #00d2ff 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        AutoDoc Insight
      </h1>
      <p style="margin:2px 0 0 0;color:#94a3b8;font-size:0.88rem;">
        Intelligent AUTOSAR High-Level Design (HLD) Architecture Verification & Analysis Platform
      </p>
    </div>
  </div>
  <div style="text-align:right;margin-top:6px;">
    <span class="badge badge-tag" style="font-size:0.75rem;">Tata Technologies TechPulse FY-26</span><br/>
    <span style="font-size:0.72rem;color:#64748b;">Case Study CS1 Applied AI/ML Capstone</span>
  </div>
</div>
<hr style="margin: 12px 0 20px 0; border-color: rgba(255,255,255,0.08);"/>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW (document statistics)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.processed:
    store: VectorStore = st.session_state.store
    ds_list = st.session_state.doc_stats

    total_pages = sum(ds.get("total_pages", 0) for ds in ds_list)
    total_chars = sum(ds.get("total_chars", 0) for ds in ds_list)
    total_headings = sum(len(ds.get("headings_found", [])) for ds in ds_list)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📄 Ingested Documents", len(st.session_state.doc_names))
    k2.metric("📑 Total Pages",        total_pages)
    k3.metric("🔢 Vector Chunks",      st.session_state.chunk_count)
    k4.metric("📌 Section Headings",   total_headings)
    k5.metric("🔤 Characters Extracted", f"{total_chars:,}")
    st.markdown("<br/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_arch, tab_incon, tab_summary, tab_report, tab_eval = st.tabs([
    "💬 Architecture Q&A",
    "🔍 Architecture & Dependency Graph",
    "⚠️ Inconsistency & Risk Audit",
    "📋 Architecture Summary",
    "📥 Full Analysis Report Export",
    "📊 Evaluation & Verification Benchmark",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ARCHITECTURE Q&A CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not st.session_state.processed:
        st.info("👆 Upload or load an AUTOSAR HLD document using the sidebar to enable conversational architecture analysis.", icon="ℹ️")
    else:
        store: VectorStore = st.session_state.store

        st.markdown("#### 💬 Natural Language Architecture Investigation")
        st.caption(
            "Query component allocations, interface specifications, signal routing, or lifecycle states. "
            "Every statement is backed by verifiable document citations and similarity scores."
        )

        # Categorized Guided Example Prompts
        with st.expander("💡 Guided Prompt Library (Click to ask)"):
            c_cat1, c_cat2 = st.columns(2)
            with c_cat1:
                st.markdown("**🧩 Components & Architecture Layers**")
                if st.button("What are the main software components described in the HLD?", key="q1", use_container_width=True):
                    st.session_state["prefill_question"] = "What are the main software components described in the HLD?"
                if st.button("What AUTOSAR layer does the CANDriver belong to?", key="q2", use_container_width=True):
                    st.session_state["prefill_question"] = "What AUTOSAR layer does the CANDriver belong to?"
                if st.button("What is the role of the DiagnosticManager?", key="q3", use_container_width=True):
                    st.session_state["prefill_question"] = "What is the role of the DiagnosticManager?"

                st.markdown("**🔌 Interfaces & Signal Routing**")
                if st.button("Which interfaces does the CANDriver expose?", key="q4", use_container_width=True):
                    st.session_state["prefill_question"] = "Which interfaces does the CANDriver expose?"
                if st.button("What signals are used between DiagnosticManager and CANDriver?", key="q5", use_container_width=True):
                    st.session_state["prefill_question"] = "What signals are used between DiagnosticManager and CANDriver?"

            with c_cat2:
                st.markdown("**🔗 Interdependencies & Functional Flows**")
                if st.button("What is the dependency between FlashManager and NvMManager?", key="q6", use_container_width=True):
                    st.session_state["prefill_question"] = "What is the dependency between FlashManager and NvMManager?"
                if st.button("Describe the functional flow for a diagnostic request.", key="q7", use_container_width=True):
                    st.session_state["prefill_question"] = "Describe the functional flow for a diagnostic request."
                if st.button("Describe the flash reprogramming functional flow.", key="q8", use_container_width=True):
                    st.session_state["prefill_question"] = "Describe the flash reprogramming functional flow."

                st.markdown("**⚙️ Lifecycle & System Management**")
                if st.button("What ports does the ECUStateManager provide?", key="q9", use_container_width=True):
                    st.session_state["prefill_question"] = "What ports does the ECUStateManager provide?"
                if st.button("What is the initialization sequence for the ECU?", key="q10", use_container_width=True):
                    st.session_state["prefill_question"] = "What is the initialization sequence for the ECU?"

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "⚡"):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and "meta" in msg:
                        meta = msg["meta"]
                        conf = meta.get("confidence", "N/A")
                        badge_cls = f"badge-{conf.lower()}" if conf in ("High", "Medium", "Low") else "badge-tag"
                        
                        m_col1, m_col2 = st.columns([1, 4])
                        with m_col1:
                            st.markdown(
                                f'<span class="badge {badge_cls}">Confidence: {conf}</span>',
                                unsafe_allow_html=True,
                            )
                        if meta.get("mock_mode"):
                            with m_col2:
                                st.markdown(
                                    '<span class="badge badge-mock">⚙️ Standalone Mock Mode</span>',
                                    unsafe_allow_html=True,
                                )

                        if meta.get("citations"):
                            with st.expander(f"📌 Traceable Page Citations ({len(meta['citations'])})"):
                                for c in meta["citations"]:
                                    st.markdown(f"- 📑 {c}")

                        if meta.get("chunks"):
                            with st.expander(f"🔍 Retrieved Evidence Passages ({len(meta['chunks'])} chunks)"):
                                for i, chunk in enumerate(meta["chunks"], 1):
                                    st.markdown(
                                        format_chunk_as_evidence(chunk, i),
                                        unsafe_allow_html=False,
                                    )
                                    if i < len(meta["chunks"]):
                                        st.divider()

        # Input
        prefill = st.session_state.pop("prefill_question", "")
        user_input = st.chat_input(
            "Ask an automotive architecture question about your HLD...",
            key="chat_input",
        )
        question = user_input or prefill

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Retrieving nearest chunks & formulating grounded architectural answer..."):
                result = ask(question, store, top_k=int(top_k))

            answer = result["answer"]
            st.session_state.chat_history.append(
                {
                    "role":    "assistant",
                    "content": answer,
                    "meta":    result,
                }
            )
            st.rerun()

        if st.session_state.chat_history:
            c_btn1, c_btn2 = st.columns([1, 4])
            with c_btn1:
                if st.button("🗑️ Clear Chat History", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            with c_btn2:
                # Chat export
                chat_txt = "\n\n".join(
                    f"**{m['role'].capitalize()}**: {m['content']}"
                    for m in st.session_state.chat_history
                )
                st.download_button(
                    "💾 Export Chat Transcript (.md)",
                    data=chat_txt,
                    file_name="autodoc_chat_transcript.md",
                    mime="text/markdown",
                    key="dl_chat",
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ARCHITECTURE ANALYSIS & DEPENDENCY GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab_arch:
    if not st.session_state.processed:
        st.info("Ingest a document first to run Architecture Extraction & Dependency Mapping.", icon="ℹ️")
    else:
        store: VectorStore = st.session_state.store
        st.markdown("#### 🔍 Automated Architecture Extraction & Component Mapping")
        st.caption("Parses structured software entities, communication boundaries, signals, and dependencies strictly from retrieved HLD evidence.")

        col_run, _ = st.columns([2, 5])
        with col_run:
            if st.button("🚀 Extract Architecture Elements", type="primary", key="run_arch", use_container_width=True):
                with st.spinner("Extracting software components, interfaces, ports, and signals..."):
                    st.session_state.arch_result = extract_architecture(store)

        arch = st.session_state.arch_result
        if arch:
            if arch.get("error"):
                st.error(arch["error"])
            else:
                # Interactive Mermaid Diagram
                st.markdown("##### 🌐 Architectural Component Interaction Graph")
                mermaid_code = generate_mermaid_diagram(
                    arch.get("dependencies", []), arch.get("components", [])
                )
                st.markdown(mermaid_code)

                # Search / Filter Bar
                st.markdown("##### 📋 Extracted Architecture Entities")
                search_query = st.text_input(
                    "🔍 Filter extracted entities by keyword (e.g. 'CAN', 'Flash', 'NvM', 'Diag'):",
                    "",
                    key="arch_filter",
                ).strip().lower()

                def filter_items(items: list) -> list:
                    if not search_query:
                        return items
                    return [it for it in items if search_query in str(it).lower()]

                def show_entity_table(title: str, items: list, col_name: str = "Item"):
                    filtered = filter_items(items)
                    st.markdown(f"**{title}** ({len(filtered)}/{len(items)})")
                    if filtered:
                        df = pd.DataFrame({col_name: filtered})
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown(
                            f'<div style="color:#94a3b8;font-size:0.8rem;">{"No matching items found." if search_query else "Not found in document."}</div>',
                            unsafe_allow_html=True,
                        )

                c1, c2 = st.columns(2)
                with c1:
                    show_entity_table("🧩 Software Components", arch.get("components", []), "Component Name")
                    show_entity_table("🔌 Interfaces", arch.get("interfaces", []), "Interface Specification")
                    show_entity_table("🚪 Ports", arch.get("ports", []), "Port Definition")
                with c2:
                    show_entity_table("📡 Signals", arch.get("signals", []), "Signal Name")
                    show_entity_table("🔗 Dependencies", arch.get("dependencies", []), "Component Dependency Flow")
                    show_entity_table("🔄 Functional Flows", arch.get("functional_flows", []), "End-to-End Sequence")

                if arch.get("citations"):
                    with st.expander(f"📌 Grounding Citations ({len(arch['citations'])})"):
                        for c in arch["citations"]:
                            st.markdown(f"- 📑 {c}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INCONSISTENCY & RISK AUDIT
# ══════════════════════════════════════════════════════════════════════════════
with tab_incon:
    if not st.session_state.processed:
        st.info("Ingest an HLD document first to run Inconsistency Auditing.", icon="ℹ️")
    else:
        store: VectorStore = st.session_state.store
        st.markdown("#### ⚠️ Automotive Design Inconsistency & Risk Audit")
        st.caption(
            "Identifies potential architectural discrepancies, dangling port dependencies, conflicting interface definitions, "
            "or terminology mismatches across document sections. Findings include page citations for engineer verification."
        )

        col_incon, _ = st.columns([2, 5])
        with col_incon:
            if st.button("⚠️ Run Inconsistency Audit", type="primary", key="run_incon", use_container_width=True):
                with st.spinner("Cross-referencing component definitions, interfaces, and dependencies..."):
                    st.session_state.inconsistencies = detect_inconsistencies(store)

        incons = st.session_state.inconsistencies
        if incons is not None:
            if not incons:
                st.success("✅ No conflicting interface definitions, dangling ports, or terminology contradictions detected in the analyzed sections.")
            else:
                st.warning(f"⚠️ {len(incons)} potential architectural inconsistency/inconsistencies detected.")
                for i, item in enumerate(incons, 1):
                    sev = item.get("severity", "Low")
                    sev_class = "badge-low" if sev == "High" else ("badge-medium" if sev == "Medium" else "badge-high")
                    
                    with st.expander(f"Finding #{i} — {item.get('finding', 'Finding')[:80]}"):
                        st.markdown(
                            f'**Severity Rating:** <span class="badge {sev_class}">{sev.upper()}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Observed Issue:** {item.get('finding', '—')}")
                        evid = item.get("evidence", [])
                        if evid:
                            st.markdown("**Evidence & Section Citations:**")
                            for e in evid:
                                st.markdown(f"- 📍 {e}")
                        
                        st.markdown("**Recommended Verification Action:**")
                        st.markdown(f"> *Verify section against formal ARXML software component description and interface contracts before freeze.*")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ARCHITECTURE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    if not st.session_state.processed:
        st.info("Ingest a document first to generate Architecture Summary.", icon="ℹ️")
    else:
        store: VectorStore = st.session_state.store
        st.markdown("#### 📋 High-Level Architecture Summary Report")
        st.caption("Produces an executive technical summary of the AUTOSAR system architecture with grounded citations.")

        col_sum, _ = st.columns([2, 5])
        with col_sum:
            if st.button("📋 Generate Architecture Summary", type="primary", key="run_summary", use_container_width=True):
                arch = st.session_state.arch_result
                if arch is None:
                    with st.spinner("Extracting architecture elements first..."):
                        arch = extract_architecture(store)
                        st.session_state.arch_result = arch
                with st.spinner("Synthesizing structured executive summary..."):
                    st.session_state.summary_text = generate_summary(store, arch)

        if st.session_state.summary_text:
            st.markdown(st.session_state.summary_text)
            st.download_button(
                "⬇️ Download Summary Only (.md)",
                data=st.session_state.summary_text,
                file_name="autodoc_architecture_summary.md",
                mime="text/markdown",
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FULL ANALYSIS REPORT EXPORT (NEW COMPREHENSIVE FEATURE)
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("#### 📥 Full Analysis Report Export")
    st.caption(
        "Consolidates architecture metadata, extracted components, interfaces, signal flows, "
        "architectural Mermaid diagrams, inconsistency audits, benchmark metrics, and Q&A transcripts into a single comprehensive report."
    )

    if not st.session_state.processed:
        st.info("Ingest and analyze a document to generate the full engineering audit report.", icon="ℹ️")
    else:
        col_gen_rep, _ = st.columns([2, 5])
        with col_gen_rep:
            if st.button("⚡ Generate Full Analysis Report", type="primary", key="btn_gen_report", use_container_width=True):
                # Ensure architecture is extracted
                if st.session_state.arch_result is None:
                    with st.spinner("Extracting architecture elements..."):
                        st.session_state.arch_result = extract_architecture(st.session_state.store)
                # Ensure inconsistencies checked
                if st.session_state.inconsistencies is None:
                    with st.spinner("Auditing for potential inconsistencies..."):
                        st.session_state.inconsistencies = detect_inconsistencies(st.session_state.store)
                # Ensure summary generated
                if not st.session_state.summary_text:
                    with st.spinner("Generating architecture narrative..."):
                        st.session_state.summary_text = generate_summary(st.session_state.store, st.session_state.arch_result)

                # Generate full report
                report_md = generate_full_analysis_report(
                    doc_stats=st.session_state.doc_stats,
                    arch_result=st.session_state.arch_result,
                    inconsistencies=st.session_state.inconsistencies,
                    summary_text=st.session_state.summary_text,
                    eval_result=st.session_state.eval_result,
                    chat_history=st.session_state.chat_history,
                    project_id=st.session_state.project_id,
                )
                st.session_state.full_report_md = report_md
                st.success("✅ Full Analysis Report generated successfully!")

        if st.session_state.full_report_md:
            st.markdown("---")
            d_col1, d_col2 = st.columns([2, 3])
            with d_col1:
                st.download_button(
                    "💾 Download Full Report (Markdown .md)",
                    data=st.session_state.full_report_md,
                    file_name="AutoDoc_Insight_Full_Analysis_Report.md",
                    mime="text/markdown",
                    type="primary",
                    use_container_width=True,
                )

            with st.expander("👁️ Live Preview: Full Engineering Report", expanded=True):
                st.markdown(st.session_state.full_report_md)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EVALUATION & VERIFICATION BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("#### 📊 RAG Pipeline Evaluation & Quality Benchmarks")
    st.caption(
        "Automated evaluation against a curated benchmark set of 15 architecture test questions. "
        "Quantifies retrieval accuracy, citation grounding, and strict factual compliance."
    )

    st.markdown(f"**Curated Benchmark Set:** {len(SYNTHETIC_QA)} domain-specific questions covering components, interfaces, signals, dependencies, and lifecycle flows.")

    if not st.session_state.processed:
        st.info("Ingest an HLD document to execute the evaluation benchmark.", icon="ℹ️")
    else:
        store: VectorStore = st.session_state.store

        col_ev_btn, _ = st.columns([2, 5])
        with col_ev_btn:
            if st.button("▶️ Execute Full Evaluation Suite", type="primary", key="run_eval", use_container_width=True):
                with st.spinner(f"Evaluating RAG pipeline against {len(SYNTHETIC_QA)} automotive benchmark questions..."):
                    st.session_state.eval_result = run_evaluation(store)

        ev = st.session_state.eval_result
        if ev:
            metrics = ev["metrics"]
            results = ev["results"]

            st.markdown("##### 📈 Benchmark Performance Scorecard")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrieval Success",  f"{metrics['retrieval_success']:.0%}", help="Percentage of queries where relevant evidence was located")
            m2.metric("Citation Coverage",  f"{metrics['citation_coverage']:.0%}", help="Percentage of responses containing page citations")
            m3.metric("Factual Groundedness", f"{metrics['groundedness_rate']:.0%}", help="Percentage of answers directly supported by retrieved evidence")
            m4.metric("Answer Pass Rate",   f"{metrics['answer_pass_rate']:.0%}", help="Strict correctness against synthetic ground truth")

            st.markdown("##### 🎯 Answer Correctness Distribution")
            pass_c    = int(metrics["answer_pass_rate"]    * len(results))
            partial_c = int(metrics["answer_partial_rate"] * len(results))
            fail_c    = int(metrics["answer_fail_rate"]    * len(results))
            unv_c     = int(metrics["unverifiable_rate"]   * len(results))

            col_p, col_pa, col_f, col_u = st.columns(4)
            col_p.metric( "✅ Pass (Accurate)",    pass_c)
            col_pa.metric("🟡 Partial Coverage",   partial_c)
            col_f.metric( "❌ Inaccurate / Missed", fail_c)
            col_u.metric( "🔘 Unverifiable",       unv_c)

            st.divider()
            st.markdown("##### 📋 Per-Question Test Matrix")

            rows = []
            for r in results:
                rows.append({
                    "ID":          r["id"],
                    "Benchmark Question": r["question"][:65] + ("..." if len(r["question"]) > 65 else ""),
                    "Confidence":  r["confidence"],
                    "Retrieval":   "✅ Pass" if r["retrieval_success"] else "❌ Fail",
                    "Citations":   "✅ Pass" if r["citation_coverage"] else "❌ Fail",
                    "Grounded":    "✅ Pass" if r["groundedness"] else "❌ Fail",
                    "Verdict":     r["answer_correctness"],
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            with st.expander("🔎 Detailed Question-by-Question Audit"):
                for r in results:
                    st.markdown(f"**Test Case {r['id']}: {r['question']}**")
                    st.markdown(f"*Expected Ground Truth:* {r['ground_truth']}")
                    st.markdown(f"*Generated Answer:* {r['generated_answer'][:320]}...")
                    st.markdown(f"*Citations:* {', '.join(r['citations']) if r['citations'] else 'None'}")
                    st.divider()

            st.download_button(
                "⬇️ Download Evaluation Results (.json)",
                data=json.dumps(ev, indent=2),
                file_name="autodoc_evaluation_results.json",
                mime="application/json",
            )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<hr style="margin: 30px 0 16px 0; border-color: rgba(255,255,255,0.08);"/>
<div style="text-align:center;color:#64748b;font-size:0.75rem;padding:8px 0 20px 0;line-height:1.6;">
  <strong style="color:#94a3b8;">AutoDoc Insight</strong> · Tata Technologies TechPulse FY-26 Applied AI/ML Capstone · Case Study CS1<br/>
  Engineered with Streamlit · FAISS Vector Store · BAAI BGE Embeddings · PyMuPDF Extraction Engine<br/>
  <span style="color:#475569;">Privacy-First Engineering: All embeddings, vector index searches, and PDF extractions execute strictly on the local machine.</span>
</div>
""",
    unsafe_allow_html=True,
)
