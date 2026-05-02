"""
pages/4_Clinical_Challenge.py
==============================
Clinical Scenario Synthesis — Task 3 implementation.

Flow:
  1. Student selects a topic (weak topics shown first, then all topics)
  2. System generates an open-ended clinical scenario question grounded in KB
  3. Student types a free-text answer
  4. LLM judge compares answer against KB gold-standard context
  5. Structured feedback: correct elements, missing elements, gold-standard, score
  6. Outcome saved to session memory (updates weak/mastered topics)
"""

import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import Database
from streamlit_app import load_components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Challenge — Socratic-OT",
    page_icon="🏥",
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
    if st.button("Logout", key="cc_logout_btn", use_container_width=True):
        for key in ["student_id", "active_session_id", "engines"]:
            st.session_state.pop(key, None)
        st.switch_page("streamlit_app.py")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    footer { visibility: hidden; }
    .correct-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .missing-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .gold-box {
        background: #e8f4f8;
        border-left: 4px solid #17a2b8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .score-box {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Shared resources ──────────────────────────────────────────────────────────
@st.cache_resource
def get_db() -> Database:
    return Database()

db = get_db()
components = load_components()
retrieve   = components["retrieve"]
groq_key   = components["groq_key"]

# ── LLM setup ─────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=800,
    api_key=groq_key,
)

# ── All topics available in KB ─────────────────────────────────────────────────
_ALL_TOPICS = [
    "Neuron structure", "Action potential", "Myelin sheath", "Brain lobes",
    "Cerebral cortex", "Cerebellum", "Basal ganglia", "Spinal cord",
    "Dorsal column pathway", "Upper motor neuron", "Lower motor neuron",
    "Brachial plexus", "Peripheral nervous system", "Autonomic nervous system",
    "Dermatomes", "Muscle structure", "Skeletal muscle fiber", "Sarcomere",
    "Muscle contraction", "Sliding filament theory", "Rotator cuff",
    "Glenohumeral joint", "Synovial joint", "Tendon", "Ligament",
    "Cardiac cycle", "Gas exchange", "Alveoli",
]

# ── Helper: generate clinical scenario ────────────────────────────────────────
_SCENARIO_PROMPT = """\
You are a clinical educator creating an exam question for an occupational therapy student.

Topic: {topic}
Textbook context (gold standard):
\"\"\"
{context}
\"\"\"

Write ONE open-ended clinical scenario question that:
- Presents a realistic OT patient case (injury, deficit, or clinical finding)
- Requires the student to apply their knowledge of {topic}
- Asks for an explanation of the underlying anatomy/physiology and its clinical implication
- Does NOT include the answer

Output ONLY the question. No preamble, no answer, no hints."""

def _generate_scenario(topic: str) -> tuple[str, str]:
    """Returns (scenario_question, kb_context)."""
    chunks = retrieve(topic, top_k=5)
    context = "\n\n".join(c["text"] for c in chunks)[:2500]
    prompt = _SCENARIO_PROMPT.format(topic=topic, context=context)
    question = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return question, context


# ── Helper: evaluate student answer ───────────────────────────────────────────
_EVAL_PROMPT = """\
You are an expert OT anatomy educator evaluating a student's free-text answer.

Clinical question: {question}

Gold-standard textbook context:
\"\"\"
{context}
\"\"\"

Student's answer:
\"\"\"
{answer}
\"\"\"

Evaluate the student's answer against the gold-standard context.
Respond in exactly this format (use these exact section headers):

SCORE: [integer 0-100]

CORRECT:
[Bullet list of concepts the student got right, with brief explanation of why each is correct. If nothing is correct, write "None."]

MISSING:
[Bullet list of key concepts from the gold standard that the student missed or got wrong. If nothing is missing, write "None."]

GOLD_STANDARD:
[A concise 3-5 sentence gold-standard explanation of the answer, drawn strictly from the textbook context above.]

VERDICT:
[One sentence: either "Mastered" (score >= 70) or "Needs more practice" (score < 70), with brief reason.]"""


def _evaluate_answer(question: str, context: str, answer: str) -> dict:
    """Returns dict with score, correct, missing, gold_standard, verdict."""
    prompt = _EVAL_PROMPT.format(question=question, context=context, answer=answer)
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    result = {
        "score": 0,
        "correct": "",
        "missing": "",
        "gold_standard": "",
        "verdict": "",
        "raw": raw,
    }

    # Parse sections
    import re
    score_match = re.search(r"SCORE:\s*(\d+)", raw)
    if score_match:
        result["score"] = min(100, max(0, int(score_match.group(1))))

    for section in ["CORRECT", "MISSING", "GOLD_STANDARD", "VERDICT"]:
        pattern = rf"{section}:\s*(.*?)(?=\n(?:CORRECT|MISSING|GOLD_STANDARD|VERDICT|$))"
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            result[section.lower()] = m.group(1).strip()

    return result


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("## 🏥 Clinical Challenge")
st.markdown(
    "Apply your anatomy and neuroscience knowledge to real OT clinical scenarios. "
    "Type a free-text answer — the system will evaluate it against the textbook gold standard."
)
st.divider()

# ── Topic selection ───────────────────────────────────────────────────────────
dash = db.get_dashboard(student_id)
weak_topics = [t.strip().title() for t in dash.get("weak_topics", []) if t.strip()]

# Build dropdown: weak topics first (marked), then remaining all topics
weak_set = {t.lower() for t in weak_topics}
weak_options  = [f"⚠️ {t} (weak)" for t in weak_topics]
other_options = [t for t in _ALL_TOPICS if t.lower() not in weak_set]
all_options   = weak_options + other_options

if not all_options:
    all_options = _ALL_TOPICS

col_topic, col_btn = st.columns([4, 1])
with col_topic:
    selected_option = st.selectbox(
        "Select a topic",
        options=all_options,
        help="Your weak topics appear first with a ⚠️ marker.",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("Generate Scenario", type="primary", use_container_width=True)

# Clean topic name (strip ⚠️ prefix and "(weak)" suffix)
selected_topic = selected_option.replace("⚠️ ", "").replace(" (weak)", "").strip()

# ── Session state for current challenge ───────────────────────────────────────
if "cc_topic"    not in st.session_state: st.session_state["cc_topic"]    = None
if "cc_question" not in st.session_state: st.session_state["cc_question"] = None
if "cc_context"  not in st.session_state: st.session_state["cc_context"]  = None
if "cc_result"   not in st.session_state: st.session_state["cc_result"]   = None
if "cc_answer"   not in st.session_state: st.session_state["cc_answer"]   = None

# Generate new scenario
if generate_btn:
    st.session_state["cc_result"]   = None
    st.session_state["cc_answer"]   = None
    st.session_state["cc_question"] = None
    st.session_state["cc_topic"]    = selected_topic
    with st.spinner(f"Generating clinical scenario for **{selected_topic}**..."):
        q, ctx = _generate_scenario(selected_topic)
        st.session_state["cc_question"] = q
        st.session_state["cc_context"]  = ctx

# ── Show scenario + answer form ───────────────────────────────────────────────
if st.session_state.get("cc_question"):
    topic    = st.session_state["cc_topic"]
    question = st.session_state["cc_question"]
    context  = st.session_state["cc_context"]

    st.markdown(f"### Topic: {topic}")
    st.markdown("**Clinical Scenario:**")
    st.info(question, icon="🏥")

    st.markdown("**Your Answer:**")
    st.caption("Explain the underlying anatomy/physiology and its clinical implication. Write as much as you know.")

    with st.form("answer_form"):
        answer_text = st.text_area(
            label="Type your answer here",
            height=180,
            placeholder="e.g. The nerve responsible for this deficit is... because... which leads to...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Submit Answer", type="primary", use_container_width=True)

    if submitted:
        if not answer_text.strip():
            st.warning("Please write an answer before submitting.")
        else:
            st.session_state["cc_answer"] = answer_text.strip()
            with st.spinner("Evaluating your answer against the textbook..."):
                result = _evaluate_answer(question, context, answer_text.strip())
                st.session_state["cc_result"] = result

                # Save outcome to DB
                score = result["score"]
                outcome = "mastered" if score >= 70 else "weak"
                session_id = db.create_session(student_id, f"Clinical Challenge: {topic}")
                if outcome == "mastered":
                    db.save_session_memory(session_id, [], [topic], {}, {topic: score})
                else:
                    db.save_session_memory(session_id, [topic], [], {topic: 1}, {topic: score})
                db.end_session(session_id)

# ── Show evaluation results ───────────────────────────────────────────────────
if st.session_state.get("cc_result"):
    result = st.session_state["cc_result"]
    score  = result["score"]

    st.divider()
    st.markdown("### Evaluation Results")

    # Score display
    if score >= 70:
        score_color = "#d4edda"
        score_label = "Mastered"
        score_icon  = "🏆"
    elif score >= 40:
        score_color = "#fff3cd"
        score_label = "Partial"
        score_icon  = "📈"
    else:
        score_color = "#f8d7da"
        score_label = "Needs Practice"
        score_icon  = "📚"

    col_score, col_verdict = st.columns([1, 3])
    with col_score:
        st.markdown(
            f'<div class="score-box" style="background:{score_color}">'
            f'{score_icon}<br>{score}/100<br>'
            f'<span style="font-size:1rem;font-weight:500">{score_label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col_verdict:
        st.markdown(f"**Verdict:** {result.get('verdict', '')}")
        st.markdown(f"**Topic:** {st.session_state['cc_topic']}")
        if score >= 70:
            st.success("This topic has been marked as **Mastered** in your progress.", icon="✅")
        else:
            st.warning("This topic has been marked as **Weak** in your progress.", icon="⚠️")

    st.markdown("")

    # What you got right
    correct = result.get("correct", "")
    if correct:
        st.markdown('<div class="correct-box"><strong>✅ What you got right</strong><br><br>' +
                    correct.replace("\n", "<br>") + '</div>', unsafe_allow_html=True)

    # What you missed
    missing = result.get("missing", "")
    if missing:
        st.markdown('<div class="missing-box"><strong>⚠️ What you missed</strong><br><br>' +
                    missing.replace("\n", "<br>") + '</div>', unsafe_allow_html=True)

    # Gold standard
    gold = result.get("gold_standard", "")
    if gold:
        st.markdown('<div class="gold-box"><strong>📖 Gold-Standard Explanation (from OpenStax A&P 2e)</strong><br><br>' +
                    gold.replace("\n", "<br>") + '</div>', unsafe_allow_html=True)

    st.divider()

    col_retry, col_new = st.columns(2)
    with col_retry:
        if st.button("Try another answer", use_container_width=True):
            st.session_state["cc_result"] = None
            st.session_state["cc_answer"] = None
            st.rerun()
    with col_new:
        if st.button("New scenario", type="primary", use_container_width=True):
            st.session_state["cc_result"]   = None
            st.session_state["cc_answer"]   = None
            st.session_state["cc_question"] = None
            st.session_state["cc_topic"]    = None
            st.rerun()
