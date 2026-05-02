"""
generate_transcripts.py
=======================
Generates 5 scripted Socratic tutoring transcripts and saves them to
Evaluation/transcripts/ so that audit_purity() has data to scan.

Each conversation covers a different topic from the knowledge base:
  1. nervous tissue    → neuron
  2. nervous system    → brain lobes
  3. nervous system    → spinal cord
  4. peripheral nerves → brachial plexus
  5. muscle structure  → skeletal muscle fiber

The student turns are scripted to exercise the full tutoring flow:
  RAPPORT → CLUE (wrong/partial) → CLUE (partial) → REVEAL (budget exhausted)

Run:
    python generate_transcripts.py

Requires GROQ_API_KEY in environment.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.knowledge_base import build_knowledge_base, get_retriever
from src.tutor import TutoringEngine

TRANSCRIPTS_DIR = os.path.join(PROJECT_ROOT, "Evaluation", "transcripts")

# ---------------------------------------------------------------------------
# 5 scripted conversations
#
# Each entry is:
#   student_id : used in the transcript filename
#   turns      : list of student utterances to feed in sequence
#
# Turn design rationale:
#   - Turn 1 (first student message) enters RAPPORT phase → tutor gives first clue
#   - Turn 2: wrong / vague answer  → tutor gives clue 2
#   - Turn 3: partial / close answer → tutor gives clue 3 or reveals
#   - Turn 4: explicit "tell me" → forces reveal if not already revealed
#
# We do NOT try to guess the masked answer; the goal is to exercise the
# purity-relevant phases (RAPPORT, CLUE) and reach REVEAL.
# ---------------------------------------------------------------------------

CONVERSATIONS = [
    # ── Scenario 1: Normal flow — student exhausts all clues, then gives up ─────
    # Tests: RAPPORT → CLUE x3 → REVEAL path with all wrong guesses
    {
        "student_id": "eval_s1",
        "turns": [
            "What is the basic structural and functional unit of the nervous system?",
            "Is it some kind of cell?",
            "Maybe a glial cell?",
            "I give up, just tell me.",
        ],
    },

    # ── Scenario 2: Student guesses correctly after the 2nd clue ─────────────
    # Tests: REVEAL_CORRECT triggers before budget is exhausted
    {
        "student_id": "eval_s2",
        "turns": [
            "What lobe of the cerebral cortex is primarily responsible for visual processing?",
            "Is it the frontal lobe?",
            "The occipital lobe?",   # correct guess after clue 2 → should trigger REVEAL_CORRECT
        ],
    },

    # ── Scenario 3: Student says I don't know / stuck repeatedly (3 times) ───
    # Tests: stuck_count increment → forced REVEAL after stuck_count ≥ 2
    {
        "student_id": "eval_s3",
        "turns": [
            "What nerve network originates from the cervical and upper thoracic spinal roots and supplies the upper limb?",
            "I don't know.",
            "I have no idea, I'm stuck.",
            "I really don't know, I give up.",
        ],
    },

    # ── Scenario 4: Out-of-KB topic — student asks about something clearly not in KB ──
    # Tests: out_of_scope flag fires → graceful "I don't cover that" response, no hallucination
    {
        "student_id": "eval_s4",
        "turns": [
            "Can you explain how mRNA vaccines work and why they are effective against COVID-19?",
            "I don't know.",
            "I give up, just tell me.",
        ],
    },

    # ── Scenario 5: Student makes a common wrong guess, then recovers ─────────
    # Tests: WRONG_NAMED classification → targeted comparison clue → partial correct
    {
        "student_id": "eval_s5",
        "turns": [
            "What is the contractile unit that makes up skeletal muscle?",
            "Is it the myosin filament?",   # common wrong guess → comparison clue
            "Is it the actin filament?",    # another wrong guess
            "The sarcomere?",               # correct after comparison clues
        ],
    },
]


def run_conversation(engine: TutoringEngine, turns: list, student_id: str):
    print(f"\n{'='*60}")
    print(f"  Conversation: {student_id}")
    print(f"{'='*60}")

    for i, user_turn in enumerate(turns, 1):
        print(f"\n[Turn {i}] Student: {user_turn}")
        response = engine.chat(user_turn)
        phase = engine.get_phase()
        print(f"         Tutor [{phase}]: {response[:120]}{'...' if len(response) > 120 else ''}")

        # Stop early if we reach DONE or POST_REVEAL_WAIT
        if phase in ("DONE", "POST_REVEAL_WAIT", "TOPIC_QUIZ"):
            print(f"         → Phase reached {phase}, stopping early.")
            break

    path = engine.save_transcript(TRANSCRIPTS_DIR, student_id=student_id)
    print(f"\n  ✅ Saved: {path}")


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

    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    print(f"[Setup] Transcripts will be saved to: {TRANSCRIPTS_DIR}")

    for conv in CONVERSATIONS:
        # Fresh engine per conversation (fresh state)
        engine = TutoringEngine(retrieve_fn=retrieve, groq_api_key=groq_key)
        run_conversation(engine, conv["turns"], conv["student_id"])

    print(f"\n{'='*60}")
    print(f"  Done. {len(CONVERSATIONS)} transcripts saved to Evaluation/transcripts/")
    print(f"{'='*60}")
    print("\nNext step: run `audit_purity()` from src/evaluation.py")
    print("  or:  python main.py --eval")


if __name__ == "__main__":
    main()
