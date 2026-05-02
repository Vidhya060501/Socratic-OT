"""
pages/4_Retrieval_Debug.py
==========================
Backend panel showing the hybrid retrieval reranking pipeline.
Intended for evaluation/demo — shows how chunks are retrieved and reranked
for each student query.

Pipeline:
  Dense (ChromaDB cosine) + Sparse (BM25) → RRF Fusion → CrossEncoder Reranker
"""

import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="Retrieval Debug — Socratic-OT",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sid = st.session_state.get("student_id")
    if sid:
        st.markdown(f"👤 **{sid}**")
        if st.button("Logout", key="debug_logout_btn", use_container_width=True):
            for key in ["student_id", "active_session_id", "engines"]:
                st.session_state.pop(key, None)
            st.switch_page("streamlit_app.py")

st.markdown("""
<style>footer { visibility: hidden; }</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔍 Retrieval Debug — Backend Panel")
st.markdown(
    "Shows how chunks are retrieved and reranked for the **last student query**. "
    "Go to the Chat page, ask a question, then come back here to inspect the pipeline."
)
st.divider()

# ── Pipeline diagram ──────────────────────────────────────────────────────────
st.markdown("### Pipeline Architecture")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
**Stage 1a: Dense Retrieval**
- Model: `all-MiniLM-L6-v2`
- Store: ChromaDB (cosine)
- Pool: top-20 candidates
- Score: cosine similarity
    """)
with col2:
    st.markdown("""
**Stage 1b: Sparse Retrieval**
- Algorithm: BM25 Okapi
- Index: all 997 chunks
- Pool: top-20 candidates
- Score: TF-IDF weighting
    """)
with col3:
    st.markdown("""
**Stage 2: RRF Fusion**
- Method: Reciprocal Rank Fusion (k=60)
- Merges dense + sparse ranked lists
- Pool: top-30 fused candidates
- No score normalization needed
    """)
with col4:
    st.markdown("""
**Stage 3: CrossEncoder Reranker**
- Model: `ms-marco-MiniLM-L-6-v2`
- Scores each (query, chunk) pair
- Final score: sigmoid(CE score)
- Returns top-3 after section merge
    """)

st.divider()

# ── Last query debug info ─────────────────────────────────────────────────────
from src.knowledge_base import get_retriever as _get_retriever_fn
_debug = getattr(_get_retriever_fn, "_last_debug", None)

if not _debug or not _debug.get("rows"):
    st.info(
        "No retrieval data yet. Go to **Chat**, ask a question, then come back here.",
        icon="💡"
    )
    st.stop()

st.markdown(f"### Last Query")
st.code(_debug["query"], language=None)

st.markdown("### Reranking Results")
st.caption(
    f"Showing top {len(_debug['rows'])} candidates after RRF fusion, "
    "sorted by CrossEncoder score (highest = most relevant to query)."
)

import pandas as pd

df = pd.DataFrame(_debug["rows"])
df = df.rename(columns={
    "rank":      "Rank",
    "chunk_id":  "Chunk ID",
    "section":   "Section",
    "dense":     "Dense Score\n(cosine)",
    "rrf":       "RRF Score\n(fusion)",
    "cross_enc": "CrossEncoder Score\n(raw logit)",
    "final":     "Final Score\n(sigmoid)",
})

# Color-code: highlight top 3 rows (those passed to LLM)
def _highlight_top3(row):
    if row["Rank"] <= 3:
        return ["background-color: #1a3a1a"] * len(row)
    return [""] * len(row)

styled = df.style.apply(_highlight_top3, axis=1).format({
    "Dense Score\n(cosine)":        "{:.3f}",
    "RRF Score\n(fusion)":          "{:.4f}",
    "CrossEncoder Score\n(raw logit)": "{:.2f}",
    "Final Score\n(sigmoid)":       "{:.4f}",
})

st.dataframe(styled, use_container_width=True, hide_index=True)
st.caption("🟢 Green rows = chunks passed to LLM as context")

st.divider()

# ── Score explanation ─────────────────────────────────────────────────────────
st.markdown("### Score Interpretation")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Dense Score", "0.0 – 1.0", help="Cosine similarity between query and chunk embedding. Higher = more semantically similar.")
with col_b:
    st.metric("RRF Score", "~0.01 – 0.05", help="Reciprocal Rank Fusion combines dense + sparse ranks. Rewards chunks appearing high in both lists.")
with col_c:
    st.metric("CrossEncoder", "raw logit", help="Cross-encoder relevance score. High positive = very relevant. Near 0 or negative = not relevant.")
with col_d:
    st.metric("Final Score", "0.0 – 1.0", help="sigmoid(CrossEncoder). Used for final ranking. > 0.95 = highly relevant chunk.")
