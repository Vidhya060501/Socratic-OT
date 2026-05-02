"""
streamlit_app.py
================
Entry point for Socratic-OT Streamlit app.
Handles login / register gate. All pages are under pages/.

Run locally:
    GROQ_API_KEY=... .venv/bin/streamlit run streamlit_app.py

Deploy to Streamlit Community Cloud:
    Push repo to GitHub → connect at share.streamlit.io → add GROQ_API_KEY secret.
"""

import os
import sys
import streamlit as st

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import Database
from src.knowledge_base import build_knowledge_base, get_retriever
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.vlm import VLMModule

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Socratic-OT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar nav polish */
    [data-testid="stSidebarNav"] { padding-top: 0.5rem; }

    /* Chat bubbles */
    [data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 4px; }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }

    /* Login card */
    .login-card {
        max-width: 420px;
        margin: 80px auto 0 auto;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        background: #f8f9fb;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ── Init DB (once per server process) ────────────────────────────────────────
@st.cache_resource
def get_db() -> Database:
    return Database()

db = get_db()

# ── Init KB + LLM + VLM (once per server process, shared across all pages) ───
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_components():
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", None)

    if not groq_key:
        st.error("GROQ_API_KEY not set. Add it to your environment or Streamlit secrets.")
        st.stop()

    collection, embedder, img_meta, img_by_topic, img_by_struct = build_knowledge_base(
        PROJECT_ROOT, force_rebuild=False
    )
    retrieve = get_retriever(collection, embedder)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        max_tokens=512,
        api_key=groq_key,
    )

    vlm = VLMModule(
        retrieve, llm,
        openai_api_key=openai_key,
        groq_api_key=groq_key,
        img_by_structure=img_by_struct,
    )

    session_memory_dir = os.path.join(PROJECT_ROOT, "data", "session_memory")
    os.makedirs(session_memory_dir, exist_ok=True)

    return {
        "retrieve":           retrieve,
        "llm":                llm,
        "vlm":                vlm,
        "groq_key":           groq_key,
        "openai_key":         openai_key,
        "session_memory_dir": session_memory_dir,
    }

# NOTE: load_components() is NOT called here — it's called only by pages/1_Chat.py.
# @st.cache_resource guarantees it runs only once across all page navigations.

# ── Auth state helpers ────────────────────────────────────────────────────────
def is_logged_in() -> bool:
    return st.session_state.get("student_id") is not None

def logout():
    for key in ["student_id", "active_session_id", "engine", "memory",
                "vlm_result", "_recorded_topics", "topics_done"]:
        st.session_state.pop(key, None)
    st.rerun()

# ── Sidebar: user info + logout ───────────────────────────────────────────────
def render_sidebar_auth():
    pass  # Each page manages its own sidebar

# ── Login / Register UI ───────────────────────────────────────────────────────
def render_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🧠 Socratic-OT")
        st.markdown("*AI Anatomy Tutor for Occupational Therapy Students*")
        st.markdown("---")

        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            st.markdown("#### Welcome back")
            with st.form("login_form"):
                lid = st.text_input("Student ID", placeholder="e.g. bandaru7")
                lpw = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
            if submitted:
                if not lid or not lpw:
                    st.error("Please enter both Student ID and password.")
                else:
                    ok, msg = db.login(lid, lpw)
                    if ok:
                        st.session_state["student_id"] = lid.strip().lower()
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_register:
            st.markdown("#### Create account")
            with st.form("register_form"):
                rid = st.text_input("Student ID", placeholder="e.g. bandaru7", key="reg_id")
                rpw = st.text_input("Password", type="password", key="reg_pw")
                rpw2 = st.text_input("Confirm Password", type="password", key="reg_pw2")
                submitted_r = st.form_submit_button("Register", use_container_width=True, type="primary")
            if submitted_r:
                if rpw != rpw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = db.register(rid, rpw)
                    if ok:
                        st.success(msg + " Please switch to the Login tab.")
                    else:
                        st.error(msg)

        st.markdown("---")
        st.caption("OpenStax A&P 2e · 28 chapters · LangGraph + Groq Llama 3.1")

# ── Main ──────────────────────────────────────────────────────────────────────
render_sidebar_auth()

if not is_logged_in():
    render_login()
else:
    # Logged in — show welcome and let Streamlit multipage nav take over
    student_id = st.session_state["student_id"]
    st.markdown(f"## 🧠 Socratic-OT — Welcome, **{student_id}**!")
    st.markdown(
        "Use the **sidebar** to navigate:\n"
        "- 💬 **Chat** — Start or continue a tutoring session\n"
        "- 📊 **My Progress** — View your weak topics and mastery\n"
        "- ℹ️ **About** — How the system works\n"
        "- 🏥 **Clinical Challenge** — Apply knowledge to open-ended clinical scenarios\n"
    )
    st.info("Click **Chat** in the sidebar to begin.", icon="💬")
