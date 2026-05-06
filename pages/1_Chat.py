"""
pages/1_Chat.py
===============
Multi-chat tutoring page — ChatGPT-style sidebar with conversation history.
Each conversation is one TutoringEngine session persisted to SQLite.
"""

import io
import os
import sys
from PIL import Image

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import Database
from src.tutor import TutoringEngine
from src.memory import SessionMemory
from langchain_core.messages import HumanMessage
from streamlit_app import load_components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat — Socratic-OT",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.session_state.get("student_id"):
    st.warning("Please log in first.")
    st.page_link("streamlit_app.py", label="Go to Login", icon="🔑")
    st.stop()

student_id = st.session_state["student_id"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    footer { visibility: hidden; }
    [data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 2px; }

</style>
""", unsafe_allow_html=True)

# ── Shared resources (cached process-wide) ────────────────────────────────────
@st.cache_resource
def get_db() -> Database:
    return Database()

components = load_components()
db         = get_db()

# ── Engine store in session_state (survives reruns) ───────────────────────────
if "engines" not in st.session_state:
    st.session_state["engines"] = {}

if "active_session_id" not in st.session_state:
    st.session_state["active_session_id"] = None


def _new_engine_session(session_id: str):
    engine = TutoringEngine(
        retrieve_fn=components["retrieve"],
        groq_api_key=components["groq_key"],
    )
    memory = SessionMemory(
        student_id=student_id,
        save_dir=components["session_memory_dir"],
    )
    st.session_state["engines"][session_id] = {
        "engine":           engine,
        "memory":           memory,
        "vlm_result":       None,
        "_recorded_topics": [],
        "topics_done":      0,
    }


def _get_sess(session_id: str) -> dict:
    if session_id not in st.session_state["engines"]:
        _new_engine_session(session_id)
    return st.session_state["engines"][session_id]


def _sync_memory(sess, active_sid, engine, memory):
    """Save weak/mastered topics to DB whenever covered_topics changes."""
    covered_now = set(engine.state.get("covered_topics", []))
    recorded    = set(sess.get("_recorded_topics", []))
    new_covered = covered_now - recorded

    engine_weak      = set(engine.state.get("weak_topics", []))
    current_topic    = engine.state.get("topic_label", "")
    for topic in new_covered:
        # A topic needed a reveal if the topic_label is in engine's weak_topics
        # (set by _reveal()) OR the masked answer itself is there.
        _needed = (topic in engine_weak) or (current_topic in engine_weak)
        memory.record_outcome(
            topic=topic,
            turns_to_correct=engine.state.get("total_turns", 3),
            needed_reveal=_needed,
        )
    if new_covered:
        sess["_recorded_topics"] = list(recorded | new_covered)
        sess["topics_done"] += len(new_covered)
        db.save_session_memory(
            active_sid,
            weak_topics=memory.weak_topics,
            mastered_topics=memory.mastered_topics,
            confused_terms=memory.confused_terms,
            topic_scores=memory.topic_scores,
        )
    return new_covered


# ── Speaker button helper ─────────────────────────────────────────────────────
def _speaker_btn(text: str):
    """Render a 🔊 button using browser TTS — appears immediately, no rerun needed."""
    _safe = text.replace("`", "").replace("'", "\\'").replace("\n", " ")
    st.components.v1.html(
        f"""<button onclick="window.speechSynthesis.cancel();
        var u=new SpeechSynthesisUtterance('{_safe}');
        u.rate=0.95;window.speechSynthesis.speak(u);"
        style="background:none;border:none;cursor:pointer;font-size:20px;
        padding:2px 4px;border-radius:6px;color:#555;"
        title="Read aloud">🔊</button>""",
        height=36,
    )

# ── TTS helper ───────────────────────────────────────────────────────────────
def _tts_bytes(text: str):
    """Convert text to MP3 bytes via gTTS. Returns None on any failure."""
    try:
        from gtts import gTTS
        import re
        # Strip markdown so TTS reads clean text
        clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
        clean = re.sub(r"#{1,6}\s*", "", clean)
        clean = re.sub(r"`[^`]*`", "", clean)
        clean = clean.strip()
        if not clean:
            return None
        buf = io.BytesIO()
        gTTS(text=clean[:2000], lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"👤 **{student_id}**")
    if st.button("Logout", key="chat_logout_btn", use_container_width=True):
        for key in ["student_id", "active_session_id", "engines"]:
            st.session_state.pop(key, None)
        st.switch_page("streamlit_app.py")

    st.divider()

    tts_on = st.toggle("🔊 Read responses aloud", value=False, key="tts_toggle")

    st.divider()

    # ── Voice input in sidebar ────────────────────────────────────────────────
    st.markdown("**🎙 Voice Input**")
    _mic_key = st.session_state.get("active_session_id", "default")
    _audio_file = st.audio_input("Record your answer", key=f"mic_{_mic_key}")
    if _audio_file is not None:
        _audio_bytes = _audio_file.read()
        _cur_hash = str(hash(_audio_bytes))
        if _cur_hash != st.session_state.get("_last_audio_hash", ""):
            st.session_state["_last_audio_hash"] = _cur_hash
            with st.spinner("Transcribing..."):
                try:
                    from groq import Groq as _Groq
                    _gc = _Groq(api_key=components["groq_key"])
                    _result = _gc.audio.transcriptions.create(
                        model="whisper-large-v3-turbo",
                        file=("audio.wav", _audio_bytes, "audio/wav"),
                        response_format="text",
                    )
                    st.session_state["stt_draft"] = _result.strip()
                except Exception as _e:
                    st.warning(f"Transcription failed: {_e}")
    _stt_draft = st.session_state.get("stt_draft", "")
    if _stt_draft:
        st.info(f"🎙 **Heard:** {_stt_draft}")
        if st.button("Send ↑ to chat", key="voice_send_btn", use_container_width=True):
            st.session_state["stt_pending"] = _stt_draft
            st.session_state.pop("stt_draft", None)
            st.rerun()

    st.divider()

    if st.button("➕  New Chat", key="new_chat_btn", use_container_width=True, type="primary"):
        new_sid = db.create_session(student_id, "New Chat")
        st.session_state["active_session_id"] = new_sid
        _new_engine_session(new_sid)
        # Clear stale voice state so old transcriptions don't leak into new session
        st.session_state.pop("stt_text", None)
        st.session_state.pop("stt_pending", None)
        st.session_state.pop("stt_draft", None)
        st.session_state.pop("_last_audio_hash", None)
        st.rerun()

    st.markdown("#### Conversations")
    all_sessions = db.get_sessions(student_id)

    if not all_sessions:
        st.caption("No conversations yet. Click **New Chat** to start.")
    else:
        for s in all_sessions:
            sid       = s["session_id"]
            title     = s["title"]
            date      = s["started_at"][:10]
            is_active = sid == st.session_state.get("active_session_id")
            label     = f"{'▶ ' if is_active else ''}{title}"
            if st.button(label, key=f"sess_{sid}", use_container_width=True,
                         help=f"Started {date}"):
                st.session_state["active_session_id"] = sid
                # Clear stale voice state when switching sessions
                st.session_state.pop("stt_text", None)
                st.session_state.pop("stt_pending", None)
                st.session_state.pop("_last_audio_hash", None)
                st.rerun()


# ── No active session ─────────────────────────────────────────────────────────
active_sid = st.session_state.get("active_session_id")
if active_sid is None:
    st.markdown("## 💬 Socratic-OT Chat")
    st.info("Click **➕ New Chat** in the sidebar to start a tutoring session.", icon="💬")
    st.stop()

# ── Load session state ────────────────────────────────────────────────────────
sess   = _get_sess(active_sid)
engine = sess["engine"]
memory = sess["memory"]

session_info = next(
    (s for s in db.get_sessions(student_id) if s["session_id"] == active_sid), {}
)

# ── Header ────────────────────────────────────────────────────────────────────
_display_title = session_info.get('title', 'Chat')
st.markdown(f"### 💬 {'New Chat' if _display_title == 'New Chat' else 'Chat'}")
st.caption(f"Session started: {session_info.get('started_at', '')[:16].replace('T', ' ')}")
st.divider()

# ── Render existing chat history from DB ──────────────────────────────────────
_all_msgs = db.get_messages(active_sid)

if not _all_msgs:
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(
            "👋 Welcome! I'm your Socratic tutor.\n\n"
            "What topic would you like to explore today? "
            "Are you studying for a particular exam, or is there a concept you'd like to work through?"
        )

for _i, msg in enumerate(_all_msgs):
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _safe = msg["content"].replace("`", "").replace("'", "\\'").replace("\n", " ")
            st.components.v1.html(
                f"""<button onclick="window.speechSynthesis.cancel();
                var u=new SpeechSynthesisUtterance('{_safe}');
                u.rate=0.95;window.speechSynthesis.speak(u);"
                style="background:none;border:none;cursor:pointer;font-size:20px;
                padding:2px 4px;border-radius:6px;color:#555;"
                title="Read aloud">🔊</button>""",
                height=36,
            )


# ── Chat input with native file attach (📎 paperclip in the input bar) ───────
chat_submission = st.chat_input(
    "Ask any anatomy or neuroscience question...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"],
    key=f"chat_input_{active_sid}_{sess.get('img_upload_count', 0)}",
)

# ── Only process when user submits (typed OR voice) ──────────────────────────
_stt_pending = st.session_state.pop("stt_pending", None)

if not chat_submission and not _stt_pending:
    st.stop()


# Resolve user_text and image from whichever input fired
image = None
if _stt_pending:
    user_text = _stt_pending
else:
    user_text = chat_submission.text if chat_submission.text else ""
    if chat_submission["files"]:
        image = Image.open(chat_submission["files"][0]).convert("RGB")

if not user_text and not image:
    st.stop()

# ── Show user message immediately ────────────────────────────────────────────
display_text = user_text or ""
if image:
    with st.chat_message("user", avatar="👤"):
        if display_text:
            st.markdown(display_text)
        st.image(image, width=280, caption="Uploaded diagram")
    db.append_message(active_sid, "user", f"[Image uploaded] {display_text}".strip())
else:
    with st.chat_message("user", avatar="👤"):
        st.markdown(display_text)
    db.append_message(active_sid, "user", display_text)

# ── IMAGE BRANCH step 1: new image — always restart image session ─────────────
# If student uploads a new image mid-session (e.g. after POST_REVEAL_WAIT),
# treat it as a fresh image topic regardless of current vlm_result state.
if image is not None:
    sess["vlm_result"] = None  # always reset for a new image upload

if image is not None and sess.get("vlm_result") is None:
    with st.spinner("Analyzing diagram..."):
        result = components["vlm"].analyze(image, student_text=user_text)

    # ── Out-of-anatomy: reject immediately, do NOT wire engine state ──────────
    if result.get("out_of_anatomy"):
        rejection = result["socratic_question"]
        sess["img_upload_count"] = sess.get("img_upload_count", 0) + 1
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(rejection)
            _speaker_btn(rejection)
            if tts_on:
                audio = _tts_bytes(rejection)
                if audio:
                    st.audio(audio, format="audio/mp3", autoplay=True)
        db.append_message(active_sid, "assistant", rejection)
        st.stop()

    sess["vlm_result"] = result
    sess["img_upload_count"] = sess.get("img_upload_count", 0) + 1

    masked      = result["structure"]
    topic       = result.get("topic", "anatomy")

    retrieval_q = f"{masked} {topic} anatomy {user_text or ''}".strip()
    chunks      = components["retrieve"](retrieval_q, top_k=5)

    target_lower  = masked.lower()
    target_tokens = [t for t in target_lower.split() if len(t) > 3]  # meaningful words
    chunks_scored = sorted(
        chunks,
        key=lambda c: (
            (2 if target_lower in c["text"].lower() else 0) +
            # partial credit: all target tokens present in chunk (handles "brain lobes" etc.)
            (1 if target_tokens and all(t in c["text"].lower() for t in target_tokens) else 0) +
            (1 if c.get("topic", "").lower() == topic.lower() else 0)
        ),
        reverse=True,
    )
    ctx_str     = "\n\n".join(c["text"] for c in chunks_scored)
    forbidden   = engine._build_forbidden_set(masked, ctx_str)
    target_type = engine._infer_target_type(masked, ctx_str)

    engine.state.update({
        "masked_answer":        masked,
        "topic_label":          topic,
        "retrieved_chunks":     chunks_scored,
        "forbidden_terms":      forbidden,
        "target_type":          target_type,
        "clue_dimensions_used": [],
        "last_attempt_class":   "NONE",
        "partial_elements":     [],
        "last_tutor_response":  "",
        "stuck_count":          0,
        "wrong_guesses":        [],
        "correct_count":        0,
        "quiz_questions":       [],
        "quiz_index":           0,
        "out_of_scope":         False,
        "total_turns":          0,
        "phase":                "RAPPORT",
    })

    with st.spinner("Thinking..."):
        if user_text:
            response = engine.chat(user_text)
        else:
            llm = components["llm"]
            response = llm.invoke([HumanMessage(content=(
                f"You are a warm Socratic anatomy tutor for OT students.\n"
                f"A student uploaded an anatomy diagram (topic: '{topic}', type: '{target_type}').\n"
                "Write 1-2 warm sentences: acknowledge the diagram without naming the structure, "
                "then invite the student to share what they want to understand. End with a question."
            ))]).content.strip()

    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(response)
        _speaker_btn(response)
        if tts_on:
            audio = _tts_bytes(response)
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=True)
    db.append_message(active_sid, "assistant", response)

    if session_info.get("title") == "New Chat":
        db.update_session_title(active_sid, topic.title())

# ── IMAGE BRANCH step 2: student replying inside image Socratic loop ──────────
elif sess.get("vlm_result") and not sess["vlm_result"].get("out_of_anatomy") and engine.state.get("phase") in (
    "RAPPORT", "CLUE", "REVEAL", "POST_REVEAL_WAIT", "TOPIC_QUIZ", "POST_TOPIC_QUIZ"
):
    with st.spinner("Thinking..."):
        response = engine.chat(user_text)

    new_covered = _sync_memory(sess, active_sid, engine, memory)
    if new_covered or engine.get_phase() == "DONE":
        sess["vlm_result"] = None

    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(response)
        _speaker_btn(response)
        if tts_on:
            audio = _tts_bytes(response)
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=True)
    db.append_message(active_sid, "assistant", response)

# ── TEXT BRANCH ───────────────────────────────────────────────────────────────
else:
    # Proactive opener if weak topics from earlier
    if engine.get_phase() in ("RAPPORT", "HINT") and not engine.state.get("messages"):
        opener = memory.proactive_opener()
        if opener:
            user_text = opener + user_text

    with st.spinner("Thinking..."):
        response = engine.chat(user_text)

    phase  = engine.get_phase()
    masked = engine.get_masked_answer()

    _sync_memory(sess, active_sid, engine, memory)

    if phase == "DONE":
        db.end_session(active_sid)
        db.save_session_memory(
            active_sid,
            weak_topics=memory.weak_topics,
            mastered_topics=memory.mastered_topics,
            confused_terms=memory.confused_terms,
            topic_scores=memory.topic_scores,
        )

    # Auto-title on first response (no rerun — title updates on next natural rerun)
    if session_info.get("title") == "New Chat" and masked:
        db.update_session_title(active_sid, masked.title())


    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(response)
        _speaker_btn(response)
        if tts_on:
            audio = _tts_bytes(response)
            if audio:
                st.audio(audio, format="audio/mp3", autoplay=True)
    db.append_message(active_sid, "assistant", response)
