"""
generate_image_transcripts.py
==============================
Generates 2 image-path Socratic tutoring transcripts and saves them to
Evaluation/transcripts/ alongside the text transcripts.

Scenario 1 (eval_img1): Student uploads an anatomy diagram IN the KB
  → IMG001_neuron_structure.PNG (no labels assumed — VLM identifies it)
  → Expected: VLM identifies → KB match → Socratic CLUE flow → REVEAL

Scenario 2 (eval_img2): Student uploads a RANDOM non-anatomy image
  → A synthetic noise image (simulates laptop photo / random upload)
  → Expected: IS_ANATOMY=no guard fires → polite rejection, no tutoring

Replicates the main.py image branch (the Gradio _process() function) directly
so the full VLM → KB-match → engine.state.update → engine.chat() pipeline
is exercised without needing the Gradio UI.

Run:
    python generate_image_transcripts.py

Requires GROQ_API_KEY in environment.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.knowledge_base import build_knowledge_base, get_retriever
from src.tutor import TutoringEngine
from src.vlm import VLMModule
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

TRANSCRIPTS_DIR = os.path.join(PROJECT_ROOT, "Evaluation", "transcripts")
IMAGES_DIR      = os.path.join(PROJECT_ROOT, "Data", "images")


# ─────────────────────────────────────────────────────────────────────────────
# Image conversations
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_CONVERSATIONS = [
    # ── Scenario IMG1: In-KB anatomy image, student guesses wrong then gives up ─
    {
        "student_id":  "eval_img1",
        "image_file":  "IMG001_neuron_structure.PNG",   # real KB image, no labels
        "description": "In-KB anatomy image (neuron) — VLM identifies → Socratic flow",
        "turns": [
            "I uploaded this diagram — can you help me identify it?",
            "Is it a glial cell?",
            "Maybe an astrocyte?",
            "I give up, just tell me.",
        ],
    },

    # ── Scenario IMG2: Non-anatomy image — should be rejected by IS_ANATOMY guard ─
    {
        "student_id":  "eval_img2",
        "image_file":  None,           # synthetic noise image — not anatomy
        "description": "Non-anatomy random image — IS_ANATOMY guard should fire",
        "turns": [
            "What is this diagram showing?",
        ],
    },

    # ── Scenario IMG3: Unlabeled 3D-rendered neuron — NOT an exact KB image ────
    # Tests the sub-structure / fuzzy KB match path:
    # VLM sees a 3D neuron render (no labels, different style from IMG001).
    # Should still match KB via token overlap or description, not SHA-256 hash.
    # Student makes a partial guess then gets it right.
    {
        "student_id":  "eval_img3",
        "image_file":  "IMG_unlabeled_neuron_3d.png",  # synthetic 3D-style, no labels
        "description": "Unlabeled 3D neuron render (not exact KB image) — fuzzy KB match → Socratic flow → correct guess",
        "turns": [
            "I see some kind of cell with branches — what structure is this?",
            "Is it a Schwann cell?",        # common misidentification
            "Maybe a neuron?",              # correct guess on turn 3
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic non-anatomy image generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_non_anatomy_image() -> Image.Image:
    """
    Create a synthetic RGB image that looks nothing like anatomy:
    random colored blocks arranged like a simple flowchart / architecture diagram.
    This exercises the IS_ANATOMY=no guard in vlm.py without needing an actual
    laptop photo on disk.
    """
    img_array = np.zeros((300, 400, 3), dtype=np.uint8)

    # Blue background
    img_array[:, :] = [30, 30, 80]

    # Draw some colored rectangles (simulate architecture diagram boxes)
    # Box 1 — green
    img_array[40:90, 50:150]   = [50, 200, 50]
    # Box 2 — red
    img_array[40:90, 250:350]  = [200, 50, 50]
    # Box 3 — yellow (bottom center)
    img_array[180:230, 150:250] = [220, 220, 50]
    # Connector lines — white
    img_array[88:92, 100:300]  = [255, 255, 255]   # horizontal line
    img_array[88:182, 198:202] = [255, 255, 255]   # vertical line

    return Image.fromarray(img_array, mode="RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Image branch logic (mirrors main.py _process() image branch)
# ─────────────────────────────────────────────────────────────────────────────

def run_image_conversation(engine: TutoringEngine,
                           vlm: VLMModule,
                           retrieve_fn,
                           img_by_struct: dict,
                           image: Image.Image,
                           turns: list,
                           student_id: str,
                           description: str):

    print(f"\n{'='*60}")
    print(f"  Image Conversation: {student_id}")
    print(f"  {description}")
    print(f"{'='*60}")

    transcript_log = []
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Step 1: VLM analysis ──────────────────────────────────────────────────
    print("\n[VLM] Analyzing image ...")
    result = vlm.analyze(image, student_text=turns[0])

    print(f"[VLM] IS_ANATOMY  : {result.get('is_anatomy', 'N/A')}")
    print(f"[VLM] Structure   : {result.get('structure', 'N/A')}")
    print(f"[VLM] Confidence  : {result.get('confidence', 'N/A')}")
    print(f"[VLM] KB matched  : {result.get('kb_matched', False)} "
          f"(pass {result.get('kb_match_pass', 0)})")
    print(f"[VLM] Backend     : {result.get('backend_used', 'N/A')}")

    # ── Out-of-anatomy guard ──────────────────────────────────────────────────
    if result.get("out_of_anatomy"):
        rejection = result["socratic_question"]
        print(f"\n[GUARD] Non-anatomy image detected — rejection fired.")
        print(f"[GUARD] Response: {rejection[:120]}...")

        transcript_log.append({
            "role": "student", "phase": "RAPPORT",
            "text": turns[0], "timestamp": datetime.now().isoformat()
        })
        transcript_log.append({
            "role": "tutor", "phase": "OUT_OF_ANATOMY",
            "text": rejection, "timestamp": datetime.now().isoformat()
        })

        _save_transcript(TRANSCRIPTS_DIR, student_id, session_id,
                         masked_answer="", transcript_log=transcript_log,
                         covered_topics=[], weak_topics=[])
        return

    # ── In-scope anatomy image: wire engine state (mirrors main.py) ───────────
    masked      = result["structure"]
    topic       = result.get("topic", "anatomy")
    first_clue  = result.get("socratic_question", "")

    _img_rec      = img_by_struct.get(masked.lower(), {})
    _kb_function  = _img_rec.get("function", "")
    _common_misid = _img_rec.get("common_misidentifications", [])

    retrieval_q   = f"{masked} {topic} anatomy"
    chunks        = retrieve_fn(retrieval_q, top_k=5)
    target_lower  = masked.lower()
    chunks_scored = sorted(
        chunks,
        key=lambda c: (
            (2 if target_lower in c["text"].lower() else 0) +
            (1 if c.get("topic", "").lower() == topic.lower() else 0)
        ),
        reverse=True,
    )

    forbidden   = engine._build_forbidden_set(masked, "\n\n".join(c["text"] for c in chunks_scored))
    target_type = engine._infer_target_type(masked, "\n\n".join(c["text"] for c in chunks_scored))

    engine.state.update({
        "masked_answer":             masked,
        "topic_label":               topic,
        "retrieved_chunks":          chunks_scored,
        "forbidden_terms":           forbidden,
        "target_type":               target_type,
        "clue_dimensions_used":      [],
        "last_attempt_class":        "NONE",
        "partial_elements":          [],
        "last_tutor_response":       first_clue,
        "stuck_count":               0,
        "wrong_guesses":             [],
        "correct_count":             0,
        "quiz_questions":            [],
        "quiz_index":                0,
        "out_of_scope":              False,
        "total_turns":               0,
        "phase":                     "RAPPORT",
        "is_image_path":             True,
        "kb_function":               _kb_function,
        "common_misidentifications": _common_misid,
    })

    # ── Step 2: Run scripted student turns through engine ─────────────────────
    for i, user_turn in enumerate(turns, 1):
        print(f"\n[Turn {i}] Student: {user_turn}")
        prior_phase = engine.get_phase()

        response = engine.chat(user_turn)
        phase    = engine.get_phase()

        print(f"         Tutor [{phase}]: {response[:120]}{'...' if len(response) > 120 else ''}")

        transcript_log.append({
            "role": "student", "phase": prior_phase,
            "text": user_turn, "timestamp": datetime.now().isoformat()
        })
        transcript_log.append({
            "role": "tutor", "phase": phase,
            "text": response, "timestamp": datetime.now().isoformat()
        })

        if phase in ("DONE", "POST_REVEAL_WAIT", "TOPIC_QUIZ"):
            print(f"         → Phase reached {phase}, stopping early.")
            break

    _save_transcript(TRANSCRIPTS_DIR, student_id, session_id,
                     masked_answer=masked,
                     transcript_log=transcript_log,
                     covered_topics=engine.state.get("covered_topics", []),
                     weak_topics=engine.state.get("weak_topics", []))


def _save_transcript(transcripts_dir, student_id, session_id,
                     masked_answer, transcript_log, covered_topics, weak_topics):
    os.makedirs(transcripts_dir, exist_ok=True)
    data = {
        "student_id":     student_id,
        "session_id":     session_id,
        "masked_answer":  masked_answer,
        "covered_topics": covered_topics,
        "weak_topics":    weak_topics,
        "turns":          transcript_log,
    }
    safe_masked = masked_answer.replace(" ", "_")[:20] if masked_answer else "out_of_scope"
    fname = f"transcript_{student_id}_{session_id}_{safe_masked}.json"
    path  = os.path.join(transcripts_dir, fname)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  ✅ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("ERROR: Set GROQ_API_KEY in your environment first.")
        sys.exit(1)

    print("\n[Setup] Building / loading knowledge base ...")
    collection, embedder, img_meta, img_by_topic, img_by_struct = build_knowledge_base(
        PROJECT_ROOT, force_rebuild=False
    )
    retrieve = get_retriever(collection, embedder)
    print("[Setup] ✅ Knowledge base ready.")

    print("[Setup] Connecting to Groq ...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4,
                   max_tokens=512, api_key=groq_key)
    llm.invoke([HumanMessage(content="Reply OK only.")])   # warm-up
    print("[Setup] ✅ Groq connected.")

    vlm = VLMModule(retrieve, llm, openai_api_key=None, img_by_structure=img_by_struct)
    print("[Setup] ✅ VLM module ready.")

    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    print(f"[Setup] Transcripts will be saved to: {TRANSCRIPTS_DIR}")

    for conv in IMAGE_CONVERSATIONS:
        # Fresh engine per conversation
        engine = TutoringEngine(retrieve_fn=retrieve, groq_api_key=groq_key)

        # Load the image
        if conv["image_file"] is None:
            image = _make_non_anatomy_image()
            print(f"\n[IMG] Using synthetic non-anatomy image for {conv['student_id']}")
        else:
            img_path = os.path.join(IMAGES_DIR, conv["image_file"])
            image = Image.open(img_path).convert("RGB")
            print(f"\n[IMG] Loaded: {conv['image_file']}")

        run_image_conversation(
            engine=engine,
            vlm=vlm,
            retrieve_fn=retrieve,
            img_by_struct=img_by_struct,
            image=image,
            turns=conv["turns"],
            student_id=conv["student_id"],
            description=conv["description"],
        )

    print(f"\n{'='*60}")
    print(f"  Done. {len(IMAGE_CONVERSATIONS)} image transcripts saved.")
    print(f"{'='*60}")
    print("\nNext: run purity audit — python main.py --eval")


if __name__ == "__main__":
    main()
