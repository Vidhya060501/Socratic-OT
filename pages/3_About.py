"""
pages/3_About.py
================
About page — explains how Socratic-OT works. Good for the demo context.
"""

import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="About — Socratic-OT",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    sid = st.session_state.get("student_id")
    if sid:
        st.markdown(f"👤 **{sid}**")
        if st.button("Logout", key="about_logout_btn", use_container_width=True):
            for key in ["student_id", "active_session_id", "engines"]:
                st.session_state.pop(key, None)
            st.switch_page("streamlit_app.py")

st.markdown("""
<style>footer { visibility: hidden; }</style>
""", unsafe_allow_html=True)

# ── Content ───────────────────────────────────────────────────────────────────
st.markdown("## ℹ️ About Socratic-OT")
st.markdown(
    "**Socratic-OT** is a grounded multimodal AI tutor designed for occupational therapy "
    "students studying anatomy and neuroscience. It enforces a *tutor-not-teller* policy: "
    "the system never gives away the answer directly — it guides you through Socratic clues "
    "until you arrive at the answer yourself."
)

st.divider()

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("### How it works")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Text tutoring")
    st.markdown("""
1. You ask an anatomy or neuroscience question
2. The system retrieves evidence from **OpenStax A&P 2e** (28 chapters)
3. It masks the answer and starts a Socratic dialogue:
   - **Clue 1** — spatial or structural hint
   - **Clue 2** — functional hint (if wrong)
   - **Clue 3** — comparison or clinical hint (if still wrong)
   - **Reveal** — answer shown with full explanation after 3 clues or on give-up
4. After each topic you can take a mastery quiz or start a new topic
    """)

with col2:
    st.markdown("#### Image tutoring")
    st.markdown("""
1. Upload any anatomy diagram using the 📎 button
2. The VLM identifies the structure internally — **without telling you**
3. It matches the structure to the knowledge base and starts Socratic clues
4. If the image is not anatomy, it politely declines
5. All image sessions follow the same CLUE → REVEAL flow as text
    """)

st.divider()

# ── Tech stack ────────────────────────────────────────────────────────────────
st.markdown("### Tech stack")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
**Language models**
- Groq Llama 3.1 8B Instruct (tutor)
- GPT-4o-mini (fallback)
- Groq llama-4-scout-17b (vision)
    """)
with col_b:
    st.markdown("""
**Retrieval & memory**
- ChromaDB + BM25 hybrid retrieval
- Cross-encoder reranking (RRF fusion)
- all-MiniLM-L6-v2 embeddings
- 997 chunks · 28 chapters
- SQLite (session + progress)
    """)
with col_c:
    st.markdown("""
**Orchestration**
- LangGraph FSM
- RAPPORT → CLUE(1-3) → REVEAL
- → POST_REVEAL → QUIZ → DONE
- Purity guard (zero answer leakage)
    """)

st.divider()

# ── Evaluation results ────────────────────────────────────────────────────────
st.markdown("### Evaluation results")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Socratic Purity", "100%", delta="✅ 8/8 transcripts")
    st.metric("VLM Structure ID", "100%", delta="✅ 6/6 images")
with col2:
    st.metric("RAGAS Faithfulness", "0.91", delta="✅ ≥ 0.90 target")
    st.metric("RAGAS Answer Relevancy", "0.97", delta="✅ ≥ 0.85 target")
with col3:
    st.metric("Context Precision", "0.82", delta="✅ ≥ 0.80 target")
    st.metric("Context Recall", "0.81", delta="✅ ≥ 0.80 target")

st.divider()

# ── Team ──────────────────────────────────────────────────────────────────────
st.markdown("### Team")
st.markdown("""
**Vidhyadhari Bandaru** · bandaru7@buffalo.edu
**Richie M Ilavarapu** · richiemo@buffalo.edu

Department of Computer Science and Engineering, University at Buffalo
NLP Course Project · Spring 2025
""")
