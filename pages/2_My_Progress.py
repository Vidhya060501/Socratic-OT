"""
pages/2_My_Progress.py
======================
Weak-topics dashboard — persists across logins via SQLite.
Shows mastered topics, weak topics, priority review, and confused terms
aggregated across all past sessions for the logged-in student.
Includes a radar/spider chart for mastery by anatomical domain (B2).
"""

import os
import sys
import streamlit as st
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import Database

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="My Progress — Socratic-OT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.session_state.get("student_id"):
    st.warning("Please log in first.")
    st.page_link("streamlit_app.py", label="Go to Login", icon="🔑")
    st.stop()

student_id = st.session_state["student_id"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"👤 **{student_id}**")
    if st.button("Logout", key="progress_logout_btn", use_container_width=True):
        for key in ["student_id", "active_session_id", "engines"]:
            st.session_state.pop(key, None)
        st.switch_page("streamlit_app.py")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    footer { visibility: hidden; }
    .metric-card {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .weak-chip {
        display: inline-block;
        background: #ffe0e0;
        color: #c00;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .mastered-chip {
        display: inline-block;
        background: #d4edda;
        color: #155724;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }

</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_db() -> Database:
    return Database()

db = get_db()
dash = db.get_dashboard(student_id)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## 📊 My Progress")
st.caption(f"Student: **{student_id}** · Data updates after each session")

if dash.get("sessions", 0) == 0:
    st.info(
        "No session history yet. Complete a tutoring session in the **Chat** tab "
        "and your progress will appear here.",
        icon="💡"
    )
    st.stop()

# ── Top metrics row ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Sessions completed", dash["sessions"])
with col2:
    st.metric("Mastered topics", len(set(t.strip().lower() for t in dash.get("mastered_topics", []) if t.strip())))
with col3:
    st.metric("Weak topics", len(set(t.strip().lower() for t in dash.get("weak_topics", []) if t.strip())))
with col4:
    st.metric("Last session", dash.get("last_session", "—"))

st.divider()

# ── Radar chart: mastery by anatomical domain ─────────────────────────────────
# Map known topics to broad OT-relevant anatomical domains.
# Any topic not in the map is bucketed into "General Anatomy".
_DOMAIN_MAP = {
    # Neuroscience
    "neuron": "Neuroscience", "nervous tissue": "Neuroscience",
    "brain lobes": "Neuroscience", "nervous system basics": "Neuroscience",
    "nervous system overview": "Neuroscience", "action potential": "Neuroscience",
    "myelin": "Neuroscience", "cerebellum": "Neuroscience",
    "basal ganglia": "Neuroscience", "spinal cord": "Neuroscience",
    "dorsal column": "Neuroscience", "upper motor neuron": "Neuroscience",
    "lower motor neuron": "Neuroscience", "cerebral cortex": "Neuroscience",
    "dermatome": "Neuroscience", "autonomic nervous system": "Neuroscience",

    # Peripheral Nerves
    "brachial plexus": "Peripheral Nerves", "peripheral nerves": "Peripheral Nerves",
    "peripheral nervous system": "Peripheral Nerves", "nerve plexus": "Peripheral Nerves",
    "lumbar plexus": "Peripheral Nerves", "cervical plexus": "Peripheral Nerves",

    # Musculoskeletal
    "muscle structure": "Musculoskeletal", "skeletal muscle fiber": "Musculoskeletal",
    "muscle contraction": "Musculoskeletal", "sliding filament": "Musculoskeletal",
    "sarcomere": "Musculoskeletal", "rotator cuff": "Musculoskeletal",
    "shoulder muscles": "Musculoskeletal", "biceps brachii": "Musculoskeletal",
    "glenohumeral joint": "Musculoskeletal", "synovial joint": "Musculoskeletal",
    "tendon": "Musculoskeletal", "ligament": "Musculoskeletal",
    "slow-twitch": "Musculoskeletal", "fast-twitch": "Musculoskeletal",

    # Cardiorespiratory
    "cardiac cycle": "Cardiorespiratory", "gas exchange": "Cardiorespiratory",
    "alveoli": "Cardiorespiratory", "heart": "Cardiorespiratory",
    "respiratory": "Cardiorespiratory", "pulmonary": "Cardiorespiratory",
}

_DOMAINS = [
    "Neuroscience", "Peripheral Nerves", "Musculoskeletal",
    "Cardiorespiratory", "General Anatomy"
]


def _topic_to_domain(topic: str) -> str:
    t = topic.lower().strip()
    for key, domain in _DOMAIN_MAP.items():
        if key in t or t in key:
            return domain
    return "General Anatomy"


def _build_radar_data(mastered: list, weak: list) -> dict:
    """Return {domain: score 0-100} where 100 = all mastered, 0 = all weak."""
    domain_mastered = {d: 0 for d in _DOMAINS}
    domain_weak     = {d: 0 for d in _DOMAINS}
    for t in mastered:
        domain_mastered[_topic_to_domain(t)] += 1
    for t in weak:
        domain_weak[_topic_to_domain(t)] += 1
    scores = {}
    for d in _DOMAINS:
        total = domain_mastered[d] + domain_weak[d]
        if total == 0:
            scores[d] = 0
        else:
            scores[d] = round(100 * domain_mastered[d] / total)
    return scores


mastered_topics = dash.get("mastered_topics", [])
weak_topics_all = dash.get("weak_topics", [])

if mastered_topics or weak_topics_all:
    st.markdown("### 🕸️ Mastery Radar")
    st.caption("Mastery % by anatomical domain — 100 = all topics in that domain answered correctly.")

    scores = _build_radar_data(mastered_topics, weak_topics_all)
    # Close the radar polygon
    theta  = _DOMAINS + [_DOMAINS[0]]
    values = [scores[d] for d in _DOMAINS] + [scores[_DOMAINS[0]]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=theta,
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.25)",
        line=dict(color="rgba(99, 110, 250, 0.9)", width=2),
        name="Mastery %",
        hovertemplate="%{theta}: %{r}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=10),
                gridcolor="rgba(200,200,200,0.4)",
            ),
            angularaxis=dict(tickfont=dict(size=12)),
        ),
        showlegend=False,
        margin=dict(t=30, b=30, l=40, r=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

def _dedup(topics: list) -> list:
    """Deduplicate case-insensitively, preserving first occurrence."""
    seen = set()
    out = []
    for t in topics:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip().title())
    return out

# ── Weak topics ───────────────────────────────────────────────────────────────
weak = _dedup(dash.get("weak_topics", []))
if weak:
    st.markdown("### 🟡 Weak Topics")
    st.caption("Topics where you needed hints or the reveal — worth more practice.")
    chips = " ".join(f'<span class="weak-chip">{t}</span>' for t in weak)
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("")

# ── Mastered topics ───────────────────────────────────────────────────────────
mastered = _dedup(dash.get("mastered_topics", []))
if mastered:
    st.markdown("### 🟢 Mastered Topics")
    st.caption("Topics you answered correctly before the reveal.")
    chips = " ".join(f'<span class="mastered-chip">✓ {t}</span>' for t in mastered)
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("")

# ── Confused terms ────────────────────────────────────────────────────────────
top_confused = dash.get("top_confused", [])
if top_confused:
    st.markdown("### 🔁 Most Confused Terms")
    st.caption("Terms you guessed incorrectly most often across sessions.")
    for term, count in top_confused:
        st.markdown(f"- **{term}** — missed {count}×")

st.divider()

# ── Session timeline ──────────────────────────────────────────────────────────
st.markdown("### 📅 Session History")
all_sessions = db.get_sessions(student_id)
if all_sessions:
    for s in all_sessions:
        ended = s.get("ended_at")
        status = "✅ Completed" if ended else "🟡 In progress"
        st.markdown(
            f"**{s['title']}** &nbsp;·&nbsp; {s['started_at'][:16].replace('T', ' ')} "
            f"&nbsp;·&nbsp; {status}"
        )
else:
    st.caption("No sessions yet.")
