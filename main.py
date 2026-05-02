"""
main.py
=======
Socratic-OT Multimodal AI Tutor
Team: Vidhyadhari Bandaru, Richie Ilavarapu

Run this file to start everything:
  - Builds (or loads) the ChromaDB knowledge base
  - Launches the Gradio chat UI with image upload
  - Accessible via a public Gradio link in Colab

Usage:
  python main.py [--rebuild] [--eval] [--port 7860]

Flags:
  --rebuild   : Force rebuild ChromaDB from CSV (skips if already built)
  --eval      : Run RAGAS evaluation after startup
  --port N    : Gradio port (default 7860)
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.knowledge_base import build_knowledge_base, get_retriever
from src.tutor import TutoringEngine
from src.memory import SessionMemory
from src.vlm import VLMModule
from src.evaluation import run_ragas, audit_purity, run_vlm_blind_test, print_full_report


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — reads API keys from environment
# ─────────────────────────────────────────────────────────────────────────────

def get_config():
    groq_key   = os.environ.get("GROQ_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", None)   # optional

    if not groq_key or groq_key == "gsk_YOUR_KEY_HERE":
        print("ERROR: GROQ_API_KEY not set.")
        print("       Set it in Colab Secrets (🔑 icon) or: os.environ['GROQ_API_KEY'] = 'your_key'")
        sys.exit(1)

    return groq_key, openai_key


# ─────────────────────────────────────────────────────────────────────────────
# Global components (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

_components = {}

def initialize(force_rebuild: bool = False):
    """Load all components. Called once at startup."""
    print("\n" + "=" * 55)
    print("  Socratic-OT — Initializing ...")
    print("=" * 55)

    groq_key, openai_key = get_config()

    # 1. Knowledge base + retriever
    collection, embedder, img_meta, img_by_topic, img_by_struct = build_knowledge_base(
        PROJECT_ROOT, force_rebuild=force_rebuild
    )
    retrieve = get_retriever(collection, embedder)

    # 2. LLM
    print("[Init] Connecting to Groq (Llama 3.1 8B) ...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4,
                   max_tokens=512, api_key=groq_key)
    # Quick test
    test = llm.invoke([HumanMessage(content="Reply with OK only.")])
    print(f"[Init] ✅ Groq connected: {test.content.strip()}")

    # 3. VLM — pass img_by_struct so the VLM can ground image labels to KB metadata
    vlm = VLMModule(retrieve, llm,
                    openai_api_key=openai_key,
                    img_by_structure=img_by_struct)

    # 4. Paths
    session_memory_dir = os.path.join(PROJECT_ROOT, "Data", "session_memory")
    transcripts_dir    = os.path.join(PROJECT_ROOT, "Evaluation", "transcripts")
    eval_dir           = os.path.join(PROJECT_ROOT, "Evaluation")
    os.makedirs(session_memory_dir, exist_ok=True)
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    _components.update({
        "retrieve":           retrieve,
        "llm":                llm,
        "vlm":                vlm,
        "groq_key":           groq_key,
        "openai_key":         openai_key,
        "img_by_struct":      img_by_struct,   # kept for direct lookup if needed
        "session_memory_dir": session_memory_dir,
        "transcripts_dir":    transcripts_dir,
        "eval_dir":           eval_dir,
    })

    print("\n" + "=" * 55)
    print("  ✅ All components ready. Launching Gradio UI ...")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Per-session state (one TutoringEngine + Memory per user session)
# ─────────────────────────────────────────────────────────────────────────────

def _new_session(student_id: str = "student") -> dict:
    engine = TutoringEngine(
        retrieve_fn=_components["retrieve"],
        groq_api_key=_components["groq_key"],
    )
    memory = SessionMemory(
        student_id=student_id,
        save_dir=_components["session_memory_dir"]
    )
    return {
        "engine":            engine,
        "memory":            memory,
        "vlm_result":        None,   # last image analysis result
        "topics_done":       0,
        "_recorded_topics":  [],     # topics already saved to memory this session
    }


def _generate_session_summary(memory: SessionMemory) -> str:
    llm = _components["llm"]
    prompt = (
        f"Generate a short 5-sentence mastery summary for an OT student.\n"
        f"Mastered topics: {memory.mastered_topics}\n"
        f"Weak topics: {memory.weak_topics}\n"
        f"Confused terms: {list(memory.confused_terms.items())[:5]}\n"
        f"Include: overall performance, strengths, topics to revisit, NBCOT study recommendation."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr
    import uuid as _uuid

    _sessions: dict = {}

    # ── Core chat logic ───────────────────────────────────────────────────────

    def _process(message: str, image, history: list,
                 session_key: str, student_id: str):
        """Shared logic for text + optional image input."""

        if not session_key or session_key not in _sessions:
            session_key = str(_uuid.uuid4())
            _sessions[session_key] = _new_session(student_id or "student")
        sess = _sessions[session_key]

        engine: TutoringEngine = sess["engine"]
        memory: SessionMemory  = sess["memory"]

        # ── IMAGE BRANCH ─────────────────────────────────────────────────────────
        # Step 1: New image → VLM pipeline (identify → KB-metadata match →
        #   topic-grounded retrieval → first orientation clue).
        # VLMModule.analyze() now returns a fully grounded result: the teaching
        # target comes from image_metadata.json when possible, falling back to
        # LLM refinement only when no metadata record matches.
        # After this turn the engine is in CLUE phase so all subsequent student
        # answers run through _clue (staged clues → reveal on CORRECT or stuck≥2).
        # ── G1 fix: second image uploaded mid-session ────────────────────────────
        # If a student uploads a new image while the vlm_result sentinel is still
        # set (i.e. they are mid-way through a previous image topic), we gracefully
        # abandon the old topic and start fresh with the new image.
        # Without this, the second image is silently ignored because the condition
        # `vlm_result is None` fails and the image falls into the text path.
        if image is not None and sess.get("vlm_result") is not None:
            print("[IMG] New image uploaded mid-session — clearing previous image topic.")
            sess["vlm_result"] = None
            engine._reset_for_new_topic()

        if image is not None and sess.get("vlm_result") is None:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")

            # Pass the student's typed text (if any) so retrieval is enriched
            # by both the image structure and the student's specific question.
            result = _components["vlm"].analyze(image, student_text=message)

            # ── Out-of-anatomy early exit ─────────────────────────────────────
            # If the VLM guard flagged the image as non-anatomical, surface the
            # rejection message directly without entering the tutoring engine.
            # The sentinel is NOT set so the session stays clean for the next upload.
            if result.get("out_of_anatomy"):
                display = "📷 *(uploaded image)*"
                return history + [(display, result["socratic_question"])], session_key

            sess["vlm_result"] = result          # image-branch sentinel

            # ── Logging: full diagnostic trace ───────────────────────────────
            print("[IMG] ═══════════════════════════════════════════")
            print(f"[IMG] KB matched      : {result.get('kb_matched', False)}  "
                  f"(pass {result.get('kb_match_pass', 0)})")
            print(f"[IMG] Teaching target : '{result['structure']}'")
            print(f"[IMG] KB topic        : '{result.get('topic', 'anatomy')}'")
            print(f"[IMG] Student text    : '{(message or '')[:60]}'")
            print(f"[IMG] Chunks from KB  : {len(result.get('context', '').split(chr(10)+chr(10)))}"
                  " chunk(s) in context")
            print("[IMG] ═══════════════════════════════════════════")
            # ─────────────────────────────────────────────────────────────────

            # Re-run retrieval here to get list-of-dicts form for engine state.
            # Query includes the student's text when present for better alignment.
            #
            # Sub-structure uplift (KB match pass 4):
            #   When the VLM identified a sub-component (e.g. "frontal lobe") that
            #   uplifted to a parent KB entry (e.g. "brain lobes"), we keep the
            #   VLM's specific label as the masked answer — the student should learn
            #   to name WHAT THEY SEE, not the parent category.  But we use the
            #   parent's topic for retrieval so ChromaDB returns grounded content.
            #   img_rec metadata (kb_function, common_misidentifications) also comes
            #   from the parent since the sub-structure has no dedicated entry.
            masked      = result["structure"]   # kb_structure from match (or VLM label)
            topic       = result.get("topic", "anatomy")

            # Pull concept-proxy and error-library fields from KB metadata.
            # Try exact match first; if pass-4 uplifted, the parent key is in result.
            _img_rec      = _components["img_by_struct"].get(masked.lower(), {})
            _kb_function  = _img_rec.get("function", "")
            _common_misid = _img_rec.get("common_misidentifications", [])

            # Build retrieval query — include both the specific label and the topic
            # so ChromaDB finds chunks even when the sub-structure name isn't indexed.
            retrieval_q = f"{masked} {topic} anatomy"
            if message:
                retrieval_q = f"{retrieval_q} {message}"
            chunks      = _components["retrieve"](retrieval_q, top_k=5)

            # Re-score: boost chunks whose text contains the target or matches topic
            target_lower = masked.lower()
            chunks_scored = sorted(
                chunks,
                key=lambda c: (
                    (2 if target_lower in c["text"].lower() else 0) +
                    (1 if c.get("topic", "").lower() == topic.lower() else 0)
                ),
                reverse=True,
            )
            ctx_str = "\n\n".join(c["text"] for c in chunks_scored)

            # Build forbidden-term shield and target-type (same as text _router)
            forbidden   = engine._build_forbidden_set(masked, ctx_str)
            target_type = engine._infer_target_type(masked, ctx_str)

            first_question = result.get("socratic_question", "")

            # ── Both modes now use the same RAPPORT-based flow ───────────────
            #
            # The image-grounded state (masked_answer, topic_label, retrieved_chunks,
            # forbidden_terms, target_type) is pre-wired below at RAPPORT phase.
            #
            # Mode A — image only (no typed text):
            #   Generate a warm, natural invitation asking what the student wants
            #   to understand or explore about this diagram.  The LLM uses topic_label
            #   and target_type to make it contextually relevant without naming the
            #   answer.  The student's reply then drives _rapport → first clue.
            #
            # Mode B — image + typed text:
            #   The student already expressed intent.  Call engine.chat(message)
            #   immediately so _rapport generates a response that acknowledges
            #   their specific question and delivers the first Socratic clue.
            #
            # In both modes: phase = RAPPORT, sentinel = set.
            # Step 2 guard includes RAPPORT so the student's reply stays in the
            # image branch and routes through engine.chat().
            # ─────────────────────────────────────────────────────────────────

            engine.state.update({
                "masked_answer":        masked,
                "topic_label":          topic,
                "retrieved_chunks":     chunks_scored,
                "forbidden_terms":      forbidden,
                "target_type":          target_type,
                "clue_dimensions_used": [],
                "last_attempt_class":   "NONE",
                "partial_elements":     [],
                # Pass the VLM-generated structural-class clue to _rapport so it
                # can wrap it in a warm opener without regenerating it from scratch.
                # _rapport reads this via state["last_tutor_response"].
                "last_tutor_response":  result.get("socratic_question", ""),
                "stuck_count":          0,
                "wrong_guesses":        [],
                "correct_count":        0,
                "quiz_questions":       [],
                "quiz_index":           0,
                "out_of_scope":         False,
                "total_turns":          0,   # _rapport will increment to 1
                "phase":                "RAPPORT",
                # Signal to _rapport and _clue that this is an image-upload session.
                # Controls dimension sequence and prompt template selection.
                "is_image_path":              True,
                # Concept proxy: the structure's function from image_metadata.json.
                # Injected into CLUE prompts so the LLM can describe the role
                # of the hidden structure without naming it.
                "kb_function":                _kb_function,
                # Error library: known wrong guesses for this specific image.
                # Triggers a targeted comparison hint when the student names one.
                "common_misidentifications":  _common_misid,
            })

            if message:
                # Mode B: student typed a question — feed it straight to _rapport
                response = engine.chat(message)
                return history + [(message, response)], session_key
            else:
                # Mode A: image only — generate a warm invitation using the LLM
                # so it is contextually grounded to the topic/type, not canned.
                llm = _components["llm"]
                invitation = llm.invoke([HumanMessage(content=(
                    "You are a warm Socratic anatomy tutor for OT students.\n"
                    "A student just uploaded an anatomy diagram without any text.\n"
                    f"The diagram is related to the topic: '{topic}' "
                    f"(structure type: '{target_type}').\n\n"
                    "Write a short, natural, warm 1-2 sentence response that:\n"
                    "  1. Acknowledges the diagram without naming the structure.\n"
                    "  2. Asks the student to predict, name, or explain something SPECIFIC\n"
                    "     about what they see — push them to think actively, not just\n"
                    "     say what they want to 'explore'. Examples of active questions:\n"
                    "     'What do you think this structure does?' or\n"
                    "     'Can you tell me which system you think this belongs to?' or\n"
                    "     'Before I give you any hints — what is your first guess?'\n\n"
                    "RULES:\n"
                    "❌ Do NOT name the specific structure (it is the hidden answer).\n"
                    "❌ Do NOT say 'I can see an anatomy diagram' or any robotic opener.\n"
                    "❌ Do NOT ask 'What would you like to learn today?' or 'What would\n"
                    "   you like to explore?' — these are passive and generic.\n"
                    "✅ Reference the topic area naturally "
                    "   (e.g. 'nervous system', 'muscle structure', 'peripheral nerves').\n"
                    "✅ Ask the student to commit to a guess or prediction before hints.\n"
                    "✅ Sound like a real tutor, not a chatbot.\n"
                    "✅ End with a question."
                ))]).content.strip()
                display = "📷 *(anatomy diagram)*"
                return history + [(display, invitation)], session_key
        # ── END IMAGE BRANCH step 1 ───────────────────────────────────────────

        if not message or not message.strip():
            return history, session_key

        # ── IMAGE BRANCH step 2: student answering inside the image Socratic loop.
        # Active whenever vlm_result sentinel is set AND the engine is in any
        # phase that belongs to the active tutoring cycle for this image topic:
        #   CLUE           — staged clue progression (attempt classify → clue/reveal)
        #   REVEAL         — transitional: _clue set this, _reveal runs next chat()
        #   POST_REVEAL_WAIT — student choosing quiz/new/done after reveal
        #   TOPIC_QUIZ     — optional quiz on the image structure
        #   POST_TOPIC_QUIZ — choosing next step after quiz
        # All of these stay inside the image branch. Sentinel is cleared only
        # AFTER the image topic is confirmed covered (covered_topics grew), which
        # prevents premature escape when the student says "I don't know" at any
        # stage — including after a reveal menu is shown.
        if sess.get("vlm_result") and engine.state.get("phase") in (
            "RAPPORT", "CLUE", "REVEAL", "POST_REVEAL_WAIT", "TOPIC_QUIZ", "POST_TOPIC_QUIZ"
        ):
            covered_before = set(engine.state.get("covered_topics", []))

            # Delegate entirely to the normal engine loop — identical to text path.
            response = engine.chat(message)
            phase    = engine.get_phase()

            covered_now = set(engine.state.get("covered_topics", []))
            recorded    = sess.get("_recorded_topics", [])
            new_covered = [t for t in covered_now - covered_before
                           if t not in recorded]

            # Record newly covered topics to session memory
            for topic in new_covered:
                memory.record_outcome(
                    topic=topic,
                    turns_to_correct=engine.state.get("total_turns", 3),
                    needed_reveal=engine.state.get("stuck_count", 0) > 0,
                )
            if new_covered:
                sess["_recorded_topics"] = recorded + new_covered
                sess["topics_done"]      += len(new_covered)
                memory.save()
                engine.save_transcript(
                    transcripts_dir=_components["transcripts_dir"],
                    student_id=student_id or "student",
                )

            # Clear the image sentinel ONLY once the image topic is confirmed
            # covered — i.e. covered_topics grew this turn.  This prevents the
            # sentinel from disappearing while the student is still inside
            # POST_REVEAL_WAIT (e.g. after typing "I don't know" to the reveal
            # menu), which would cause the next message to fall into the text path
            # before the image topic cycle is fully complete.
            # Exception: always clear on DONE (session end).
            if new_covered or phase == "DONE":
                sess["vlm_result"] = None

            return history + [(message, response)], session_key
        # ── END IMAGE BRANCH step 2 ───────────────────────────────────────────

        # ── Normal Socratic text tutoring ─────────────────────────────────────
        if engine.get_phase() in ("RAPPORT", "HINT") and not engine.state.get("messages"):
            opener = memory.proactive_opener()
            if opener:
                message = opener + message

        response = engine.chat(message)
        phase    = engine.get_phase()
        masked   = engine.get_masked_answer()

        # Record newly covered topics to session memory
        covered_now = engine.state.get("covered_topics", [])
        recorded    = sess.get("_recorded_topics", [])
        new_covered = [t for t in covered_now if t not in recorded]
        for topic in new_covered:
            memory.record_outcome(
                topic=topic,
                turns_to_correct=engine.state.get("total_turns", 3),
                needed_reveal=engine.state.get("stuck_count", 0) > 0,
            )
        if new_covered:
            sess["_recorded_topics"] = recorded + new_covered
            sess["topics_done"]      += len(new_covered)
            memory.save()
            # Save a transcript each time a new topic is completed (for purity audit)
            engine.save_transcript(
                transcripts_dir=_components["transcripts_dir"],
                student_id=student_id or "student",
            )

        if phase == "DONE":
            memory.save()
            engine.save_transcript(
                transcripts_dir=_components["transcripts_dir"],
                student_id=student_id or "student",
            )

        return history + [(message, response)], session_key

    def chat_wrapper(msg_data, history, session_key, student_id):
        """Handle MultimodalTextbox input: {text, files}."""
        if isinstance(msg_data, dict):
            message = (msg_data.get("text") or "").strip()
            files   = msg_data.get("files") or []
        else:
            message = str(msg_data or "").strip()
            files   = []

        # Load image if attached
        image = None
        if files:
            try:
                path  = files[0] if isinstance(files[0], str) else getattr(files[0], "path", str(files[0]))
                image = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"[UI] Image load error: {e}")

        new_history, new_key = _process(message, image, history, session_key, student_id)
        return None, new_history, new_key   # None clears the MultimodalTextbox

    def reset_session(session_key, student_id):
        if session_key and session_key in _sessions:
            del _sessions[session_key]
        new_key = str(_uuid.uuid4())
        _sessions[new_key] = _new_session(student_id or "student")
        return [], new_key

    # ── Layout — ChatGPT-style ────────────────────────────────────────────────
    css = """
    .gradio-container { max-width: 860px !important; margin: 0 auto !important; }
    footer { display: none !important; }
    """

    def load_dashboard(student_id: str) -> str:
        """Load and format the weak-spots dashboard for a given student."""
        from src.memory import SessionMemory
        sid = (student_id or "student").strip()
        dash = SessionMemory.get_dashboard(sid, _components["session_memory_dir"])
        if dash.get("sessions", 0) == 0:
            return f"No session history found for student **{sid}** yet. Complete a tutoring session first."
        lines = [
            f"### Dashboard for: {sid}",
            f"- **Sessions completed:** {dash['sessions']}",
            f"- **Last session:** {dash.get('last_session', 'N/A')[:10]}",
            "",
            f"**Mastered topics ({len(dash.get('mastered_topics', []))}):**",
        ]
        for t in dash.get("mastered_topics", []):
            lines.append(f"  - {t}")
        lines += [
            "",
            f"**Weak topics to revisit ({len(dash.get('weak_topics', []))}):**",
        ]
        for t in dash.get("weak_topics", []):
            lines.append(f"  - ⚠️ {t}")
        lines += [
            "",
            f"**Priority review (most frequently missed):**",
        ]
        for t in dash.get("priority_review", []):
            lines.append(f"  - 🔴 {t}")
        if dash.get("top_confused"):
            lines += ["", "**Most confused terms:**"]
            for term, cnt in dash.get("top_confused", []):
                lines.append(f"  - `{term}` — missed {cnt}x")
        return "\n".join(lines)

    with gr.Blocks(
        title="Socratic-OT Tutor",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=css,
    ) as demo:

        gr.Markdown(
            "# Socratic-OT: AI Anatomy Tutor\n"
            "**OpenStax A&P 2e · 28 chapters · Tutor-Not-Teller · LangGraph + Groq Llama 3.1**"
        )

        with gr.Tabs():

            # ── Tab 1: Chat ───────────────────────────────────────────────────
            with gr.Tab("Chat"):

                chatbot = gr.Chatbot(
                    label="",
                    height=520,
                    bubble_full_width=False,
                    show_copy_button=True,
                    placeholder=(
                        "**Welcome!** Ask any anatomy or neuroscience question, "
                        "or attach a diagram using the 📎 button below.\n\n"
                        "I'll *guide* you to the answer — not just give it."
                    ),
                )

                msg_box = gr.MultimodalTextbox(
                    placeholder="Ask about any anatomy topic, or attach a diagram...",
                    file_types=["image"],
                    label="",
                    show_label=False,
                    scale=1,
                    lines=1,
                )

                with gr.Row():
                    student_id_box = gr.Textbox(
                        label="Student ID",
                        value="student_001",
                        placeholder="student_001",
                        scale=3,
                    )
                    reset_btn = gr.Button("New Session", variant="secondary", scale=1)

                with gr.Accordion("How it works", open=False):
                    gr.Markdown(
                        "**Text:** Ask any anatomy/neuroscience question — "
                        "I give a hint → clue → reveal across 3 turns before explaining.\n\n"
                        "**Image:** Attach a diagram using the clip icon — "
                        "I'll ask you about the structure *before* naming it.\n\n"
                        "**After each topic:** Choose to quiz yourself, start a new topic, "
                        "or finish with a 3-question mastery quiz.\n\n"
                        "Type **done** to end the session and get your mastery summary."
                    )

                session_key = gr.State(value="")

                msg_box.submit(
                    fn=chat_wrapper,
                    inputs=[msg_box, chatbot, session_key, student_id_box],
                    outputs=[msg_box, chatbot, session_key],
                )
                reset_btn.click(
                    fn=reset_session,
                    inputs=[session_key, student_id_box],
                    outputs=[chatbot, session_key],
                )

            # ── Tab 2: Weak-Spots Dashboard ───────────────────────────────────
            with gr.Tab("My Progress"):
                gr.Markdown(
                    "### Student Weak-Spots Dashboard\n"
                    "Enter your Student ID below and click **Load Dashboard** "
                    "to see your mastered topics, weak spots, and priority review areas "
                    "aggregated across all your sessions."
                )
                with gr.Row():
                    dash_student_id = gr.Textbox(
                        label="Student ID",
                        value="student_001",
                        placeholder="student_001",
                        scale=3,
                    )
                    dash_load_btn = gr.Button("Load Dashboard", variant="primary", scale=1)

                dash_output = gr.Markdown(value="_No data loaded yet._")

                dash_load_btn.click(
                    fn=load_dashboard,
                    inputs=[dash_student_id],
                    outputs=[dash_output],
                )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Socratic-OT Tutor")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild ChromaDB")
    parser.add_argument("--eval",    action="store_true", help="Run RAGAS evaluation")
    parser.add_argument("--port",    type=int, default=7860, help="Gradio port")
    args = parser.parse_args()

    # Initialize all components
    initialize(force_rebuild=args.rebuild)

    # Optional evaluation run
    if args.eval:
        print("\n[Main] Running evaluation ...")
        ragas_res  = run_ragas(
            retrieve_fn=_components["retrieve"],
            llm=_components["llm"],
            eval_dir=_components["eval_dir"],
            openai_api_key=_components["openai_key"],
            groq_api_key=_components["groq_key"],
        )
        purity_res = audit_purity(
            transcripts_dir=_components["transcripts_dir"],
            eval_dir=_components["eval_dir"],
        )
        vlm_res = run_vlm_blind_test(
            vlm_module=_components["vlm"],
            images_dir=os.path.join(PROJECT_ROOT, "Data", "images"),
            eval_dir=_components["eval_dir"],
        )
        print_full_report(ragas_res, purity_res, vlm_res)

    # Launch Gradio
    demo = build_ui()
    demo.launch(
        share=True,        # creates a public URL — required for Colab
        server_port=args.port,
        debug=False,
        quiet=False,
    )


if __name__ == "__main__":
    main()
