"""
tutor.py — Socratic-OT LangGraph tutoring engine
=================================================

CONTROLLER POLICY (research-level design):
─────────────────────────────────────────
Turn 1 (HINT):
  • Clue drawn from a SAFE dimension: consequence / symptom / test / population
  • Forbidden set = {masked_answer} ∪ near-answer lexical shield
  • Purity guard: exact + token-level check against entire forbidden set

Turn 2 (CLUE):
  • Attempt is CLASSIFIED before prompting:
      CORRECT       → immediate REVEAL_CORRECT
      PARTIAL       → acknowledge the partial, tighten the clue (new dimension)
      WRONG_NAMED   → "not quite — here's another angle" (different dimension)
      DONT_KNOW     → strong clue, stuck_count += 1
  • Clue dimension is selected to NEVER repeat the previous dimension
  • Forbidden set re-injected; purity guard re-run

Turn 3+ (CLUE, stuck_count ≥ 1):
  • Same classify → respond loop
  • If stuck_count ≥ 2 → REVEAL unconditionally

PHASE FLOW:
  RAPPORT → HINT → CLUE (n times, max 3) → REVEAL → CLINICAL_SCENARIO → POST_REVEAL_WAIT
                                              ↑ (if correct at any CLUE turn → REVEAL_CORRECT)
  POST_REVEAL_WAIT → TOPIC_QUIZ → POST_TOPIC_QUIZ → (new topic | SESSION_QUIZ)
                   → SESSION_QUIZ → SESSION_QUIZ_FEEDBACK → DONE

CLINICAL_SCENARIO (Task 3):
  After REVEAL, the agent poses one open-ended OT clinical scenario question.
  Student types free-text reasoning. LLM judges against KB gold-standard.
  Provides structured feedback: correct elements, missing elements, gold-standard, score.
  Outcome updates weak/mastered topics before moving to POST_REVEAL_WAIT.

FORBIDDEN-TERM SHIELD (stops near-answer leakage):
  • masked_answer tokens (each word independently)
  • known synonyms/aliases from metadata
  • explicit near-answer list built per masked_answer by LLM at topic-start

CLUE DIMENSIONS (ordered; never reuse same dimension consecutively):
  consequence, symptom, innervation_territory, motor_test, population,
  anatomical_neighbor, comparison, mechanism
"""

from typing import TypedDict, Annotated, List, Dict, Optional
import operator
import re
import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END


# ─────────────────────────────────────────────────────────────────────────────
# Dialogue State  (extended for target-aware control)
# ─────────────────────────────────────────────────────────────────────────────

class DialogueState(TypedDict):
    messages          : Annotated[list, operator.add]
    current_input     : str

    # Phase control
    phase             : str
    total_turns       : int

    # Retrieval
    retrieved_chunks  : list
    masked_answer     : str
    topic_label       : str
    out_of_scope      : bool

    # ── target-aware clue tracking ───────────────────────────────────────────
    forbidden_terms      : list   # exact answer + aliases only (NOT generic type words)
    clue_dimensions_used : list   # ordered list of dimension names already used
    last_attempt_class   : str    # CORRECT | PARTIAL | WRONG_NAMED | DONT_KNOW | NONE
    partial_elements     : list   # sub-concepts the student got right so far
    last_tutor_response  : str    # text of the previous tutor clue (anti-repetition)
    target_type          : str    # safe ontology label: "nerve" | "muscle" | "vessel" |
                                  # "bone" | "tract" | "process" | "structure" | ...
    # ─────────────────────────────────────────────────────────────────────────

    # Stuck tracking (per topic)
    stuck_count       : int
    wrong_guesses     : list   # [{text, classification, turn}]
    correct_count     : int

    # Session-level tracking
    weak_topics       : list
    covered_topics    : list

    # Quiz state
    quiz_questions    : list
    quiz_index        : int

    # Output
    tutor_response    : str

    # Image-path flag — set True by main.py when an image was uploaded.
    # Controls dimension sequencing and rapport style in _rapport/_clue.
    # Never set by the text path — defaults to False in _reset_state.
    is_image_path     : bool

    # Image-path metadata extras (populated from image_metadata.json by main.py).
    # kb_function: the "function" field — injected as a concept proxy in CLUE
    #   prompts so the LLM can talk around the answer naturally.
    # common_misidentifications: list of known wrong guesses for this image —
    #   if the student names one, _clue triggers a targeted comparison hint.
    kb_function               : str
    common_misidentifications : list

    # Set True after out-of-scope rejection so _rapport knows to invite a
    # new question instead of generating a clue with no masked answer.
    waiting_for_question      : bool


# ─────────────────────────────────────────────────────────────────────────────
# Clue dimension catalogue
# ─────────────────────────────────────────────────────────────────────────────

# Full dimension catalogue (ordered for cycling in turns 2+).
CLUE_DIMENSIONS = [
    "consequence",        # "What happens to grip strength when this structure is damaged?"
    "symptom",            # "Patients with compression here often report tingling in specific fingers"
    "motor_test",         # "There is a clinical test where the examiner taps the wrist…"
    "innervation_territory",  # "This structure supplies sensation to the thumb, index, and middle finger"
    "population",         # "It is most commonly compressed in repetitive wrist-flexion occupations"
    "anatomical_neighbor",# "It passes through a narrow tunnel formed by the carpal bones"
    "comparison",         # "Unlike the ulnar nerve, this one does not supply the ring or little finger"
    "mechanism",          # "Compression reduces axonal conduction velocity distal to the wrist"
]

# Turn-1 safe dimensions — broad enough that multiple candidates remain plausible.
# Excludes: consequence, symptom, innervation_territory (all give away the functional map).
# Excludes: comparison (requires knowing what the wrong guess was).
# Order: anatomical_neighbor and mechanism first — these generalize across all concept types
# (brain regions, nerves, muscles, etc.). population is last because it can bias the LLM
# toward inventing clinical/occupational-risk framing for basic anatomy concepts.
_TURN1_DIMENSIONS = ["anatomical_neighbor", "mechanism", "population"]

_DIM_INSTRUCTION = {
    "consequence": (
        "Clue dimension: CONSEQUENCE.\n"
        "Describe what FAILS clinically or functionally when this structure is damaged "
        "or compressed — muscle weakness, loss of sensation, loss of dexterity. "
        "Do NOT name the structure. Do NOT name any nerve by name."
    ),
    "symptom": (
        "Clue dimension: SYMPTOM.\n"
        "Describe the sensory or motor symptom pattern a patient would report — "
        "which fingers tingle, which movements are weak — without naming the nerve. "
        "Do NOT name the structure. Do NOT name any nerve by name."
    ),
    "motor_test": (
        "Clue dimension: CLINICAL TEST.\n"
        "Describe ONE clinical test used to assess this structure (e.g. Phalen, Tinel, "
        "Froment, Finkelstein) by its action and finding, NOT by its eponymous name if "
        "that would give away the answer. Do NOT name the nerve."
    ),
    "innervation_territory": (
        "Clue dimension: INNERVATION TERRITORY.\n"
        "Describe exactly which skin region or which muscles are innervated by this structure, "
        "using finger numbers or anatomical landmarks — without naming the nerve itself."
    ),
    "population": (
        "Clue dimension: CONTEXT / SETTING.\n"
        "Describe which body system, functional domain, or anatomical division this "
        "structure belongs to — only as supported by the CONTEXT provided. "
        "Do NOT invent a clinical-injury or occupational-risk framing unless the CONTEXT "
        "explicitly describes one. Do NOT name the structure itself."
    ),
    "anatomical_neighbor": (
        "Clue dimension: LOCATION / ANATOMICAL SETTING.\n"
        "Describe where this structure sits in the body — the general region, division, "
        "or the surrounding anatomical landmarks near it — without naming the structure "
        "itself. This applies equally to brain regions, nerves, muscles, bones, or any "
        "other anatomical concept."
    ),
    "comparison": (
        "Clue dimension: COMPARISON.\n"
        "The student just named a specific WRONG structure. Your job is to contrast it "
        "with the hidden target — explain in 1-2 sentences what makes the student's guess "
        "different from the target, focusing on ROLE or FUNCTION, not internal composition. "
        "You MUST name the structure the student guessed (it is already known to them). "
        "Do NOT name the correct answer. Do NOT describe what the target is made of or "
        "enumerate its components — only describe what it DOES or what role it plays "
        "that distinguishes it from the wrong guess. End with a question asking them to try again."
    ),
    "mechanism": (
        "Clue dimension: FUNCTION / MECHANISM.\n"
        "Describe what this structure does — its primary physiological role, the process "
        "it drives, or the system it serves — in one broad sentence grounded in the "
        "CONTEXT provided. Do NOT name the structure. Do NOT invent a clinical injury "
        "scenario unless the CONTEXT explicitly describes one."
    ),

    # ── Image-path specific dimensions ────────────────────────────────────────
    # These are used ONLY when is_image_path=True. They never appear in the
    # text-path dimension sequences (_TURN1_DIMENSIONS, _CLUE_DIMS).
    "image_function": (
        "Clue dimension: FUNCTION (image path).\n"
        "Looking at the diagram the student uploaded, describe what this structure DOES — "
        "its primary role or action in the body. Focus on what it enables, controls, or "
        "produces. Do NOT name the structure. You MAY use the safe type label. "
        "Frame it so the student can observe what they see in the diagram (e.g. arrangement, "
        "connections, shape) and map it to the function."
    ),
    "image_insertion_clinical": (
        "Clue dimension: INSERTION / ORIGIN / CLINICAL (image path).\n"
        "Give a clue about WHERE this structure attaches, originates, or connects to, "
        "OR describe what happens clinically when this structure is damaged or absent — "
        "choose whichever is more visually apparent in an anatomy diagram. "
        "For muscles: mention the attachment points without naming the muscle. "
        "For nerves: describe the deficit territory without naming the nerve. "
        "For brain structures: describe the pathway or projection target. "
        "Do NOT name the structure itself."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

_BASE = (
    "You are Socratic-OT, a warm and knowledgeable anatomy tutor for OT students. "
    "Speak naturally — conversational, encouraging, never robotic. "
    "All facts must come from the provided context. NEVER fabricate anatomy. "
    "NEVER say 'textbook', 'context', 'chapter', 'the text says', or 'according to'. "
    "Speak as if the knowledge is yours."
)

PROMPTS = {

    # ── RAPPORT_HINT: single first-turn response ──────────────────────────────
    # Replaces the old two-turn RAPPORT → HINT split.
    # Delivers: (1) a short contextual engagement line, (2) the first clue.
    # The clue dimension and type label are injected by the controller.
    "RAPPORT_HINT": _BASE + """

A student just asked an anatomy question. You know the answer and will guide them toward it
Socratically — do NOT reveal it yet.

THE ANSWER: {masked}

Respond in exactly 3 sentences:
1. A warm, encouraging opener — e.g. "Great question, let's think through this together!"
   Do NOT state any facts here. Just invite them to engage.
2. One clue from the CONTEXT that hints at the answer — its location, function, or consequence.
   Do not name the answer directly.
3. A natural question that invites the student to reason toward the answer.

FORBIDDEN — never write these terms: {forbidden_block}
Never fabricate facts not in the CONTEXT. Output only the 3 sentences, no labels, end with "?"

CONTEXT: {context}""",

    # ── IMAGE_RAPPORT_HINT: first response when student uploads a diagram ─────
    # Used ONLY for is_image_path=True. The VLM already identified the structure
    # and built a first clue (socratic_question). This prompt wraps that clue in
    # a warm, natural opener so it reads as a tutor response, not a robot output.
    #
    # Policy:
    #   Sentence 1: Warm acknowledgement of the diagram (no structure name, no anatomy facts).
    #   Sentence 2: The VLM-generated structural-class clue (injected verbatim or reworded).
    #   Sentence 3: A direct question asking the student to name the structure.
    "IMAGE_RAPPORT_HINT": _BASE + """

TASK: Write exactly 3 sentences as the opening response to a student who uploaded an anatomy diagram.
Do NOT output any labels, headers, or markers — output only the 3 sentences.

Topic area: {topic_area}
Target type: {target_type}

FORBIDDEN TERMS — never write any of these, not even partially:
{forbidden_block}

OUTPUT FORMAT (3 sentences, no labels):
Sentence 1: A warm, natural acknowledgement that the student uploaded a diagram. Do NOT
  name the structure. Do NOT state any anatomy facts. Sound like a real tutor, not a chatbot.
  Example openers: "Oh nice, let's dig into this diagram together!" or "Great — let's see
  what we can work out from this." Vary it each time.
Sentence 2: Use this structural-class clue EXACTLY as provided — do not add to it, do not
  name the hidden structure: "{vlm_clue}"
Sentence 3: A direct question asking the student to name the {target_type}.
  Example: "Can you tell me which {target_type} this is?"

ABSOLUTE RULES:
- Output only 3 sentences. No headers, no labels, no bullet points.
- Never write any forbidden term.
- Never add anatomy facts beyond what is in Sentence 2.
- End with "?"

CONTEXT (for reference only — do not inject extra facts into Sentence 2): {context}""",

    # ── HINT: kept for internal use when RAPPORT was already handled ──────────
    # Used only if the engine is mid-topic and needs a fresh clue turn.
    "HINT": _BASE + """

PHASE: HINT — first clue turn.
Target concept type (safe to use): {target_type}

{dim_instruction}

ABSOLUTE FORBIDDEN TERMS (never write ANY of these, not even partially):
{forbidden_block}

RULES:
❌ Do NOT write any forbidden term, even embedded in a longer word.
❌ Do NOT answer the question. Do NOT define anything.
❌ Do NOT write more than 2 sentences.
✅ You MAY use the safe type label "{target_type}" (e.g. "which {target_type}").
✅ Write exactly ONE leading question using ONLY the allowed clue dimension above.
✅ Phrase it so a student who knows the answer would recognize the hint, but it
   cannot be deduced by someone who doesn't already know.
✅ End with "?"

CONTEXT (for facts only — do not quote or recite it): {context}""",

    # ── CLUE: Turn 2+ — single unified prompt ────────────────────────────────
    # One prompt handles all student reply types (partial, wrong, stuck, vague).
    # attempt_class is now injected by the controller so the LLM knows exactly
    # how to acknowledge — no more "you're on the right track" for DONT_KNOW.
    "CLUE": _BASE + """

You are in a Socratic tutoring conversation. The student asked a question and you are
guiding them toward the answer through natural conversation — NOT by telling them directly.

THE ANSWER: {masked}
Do NOT say this answer. Guide the student toward it with a clue.

YOUR LAST CLUE: "{prev_clue}"
THE STUDENT JUST SAID: "{attempt}"

Respond naturally like a good tutor would:
- If the student said they don't know → acknowledge warmly and give a fresh clue from a different angle
- If the student guessed something wrong → gently say it's not quite right, briefly explain why, then give a new clue
- If the student is partially right → encourage them and build on what they got right
- If the student said something vague → give a clearer, more direct clue

Your clue must:
- Come from the CONTEXT below — do not invent facts
- Be a completely different angle from your last clue
- Not reveal the answer directly
- End with a natural question that invites the student to think

FORBIDDEN — never write these terms: {forbidden_block}

Keep it conversational, 2-3 sentences, end with "?"

CONTEXT: {context}""",

    "REVEAL": (
        "You are Socratic-OT, a warm anatomy tutor for OT students. "
        "NEVER say 'textbook', 'context', 'chapter', or 'according to'.\n\n"
        "PHASE: REVEAL — the answer is now being given in full.\n"
        "THE ANSWER IS: {masked}\n"
        "All prior forbidden-term rules from earlier turns are NOW LIFTED. "
        "You MUST say '{masked}' clearly. Do not avoid or substitute it.\n\n"
        "Format:\n"
        "1. Warmly acknowledge their effort ('No worries!' or 'Good effort!' or similar)\n"
        "2. State clearly: 'The answer is {masked}.'\n"
        "3. Explain in 2-3 natural sentences what {masked} is and why it matters\n"
        "4. One OT CLINICAL SCENARIO sentence: describe a real patient case where "
        "   damage to or dysfunction of {masked} would affect a patient's daily life or ADLs\n"
        "5. End with: 'Does that make sense? Here is what you can do next:\n"
        "   1  Yes, I got it — quiz me on this!\n"
        "   2  I would like help with another topic\n"
        "   3  I am done for now\n"
        "   4  Give me a clinical scenario to apply this'\n\n"
        "CONTEXT: {context}\nANSWER: {masked}"
    ),

    "REVEAL_CORRECT": _BASE + """

PHASE: REVEAL_CORRECT — the student just got the answer right.
THE ANSWER IS: {masked}
You MUST say "{masked}" in your response. This is a confirmation, not a clue.
All prior forbidden-term rules from earlier turns are now LIFTED for this message.

Student said: "{attempt}"

Write in this order:
1. Celebrate warmly (1 sentence) — name the answer: "Yes, exactly — {masked}!"
2. Explain in 2 sentences what {masked} is and why it matters clinically.
3. One OT clinical scenario: how a patient with a deficit involving {masked}
   would present in terms of ADLs or hand function.
4. End with exactly:
   "Great work! What would you like to do next?
   1  Quiz me on this topic
   2  Move to a new topic
   3  I am done for now
   4  Give me a clinical scenario to apply this"

CONTEXT: {context}""",

    "CLINICAL_SCENARIO_PROMPT": (
        "You are Socratic-OT, a clinical OT educator. "
        "NEVER say 'textbook', 'context', 'chapter', or 'according to'.\n\n"
        "The student just learned about: {masked}\n\n"
        "Textbook context:\n\"\"\"\n{context}\n\"\"\"\n\n"
        "Write ONE open-ended OT clinical scenario question that:\n"
        "- Presents a realistic patient case involving {masked}\n"
        "- Asks the student to explain the underlying anatomy/physiology AND its impact on daily life or ADLs\n"
        "- Does NOT reveal or hint at the answer\n\n"
        "Format your response as:\n"
        "\"Now let's apply what you've learned! Here's a clinical scenario:\n\n"
        "[Your scenario question here]\n\n"
        "Take your time and explain your reasoning — there's no single right answer format.\""
    ),

    "CLINICAL_SCENARIO_EVAL": (
        "You are an expert OT anatomy educator evaluating a student's clinical reasoning.\n\n"
        "Topic: {masked}\n"
        "Clinical question: {question}\n\n"
        "Gold-standard textbook context:\n\"\"\"\n{context}\n\"\"\"\n\n"
        "Student's answer:\n\"\"\"\n{answer}\n\"\"\"\n\n"
        "Evaluate the student's free-text answer against the gold standard.\n"
        "Respond in this EXACT format:\n\n"
        "SCORE: [integer 0-100]\n\n"
        "CORRECT:\n[Bullet list of concepts the student got right. If none, write 'None.']\n\n"
        "MISSING:\n[Bullet list of key concepts the student missed. If none, write 'None.']\n\n"
        "EXPECTED ANSWER:\n[3-5 sentence explanation of the expected answer drawn strictly from the textbook context.]\n\n"
        "VERDICT:\n[One sentence: 'Mastered' if score >= 70, else 'Needs more practice', with brief reason.]"
    ),

    "POST_REVEAL_WAIT": _BASE + """

PHASE: POST_REVEAL_WAIT
The student just responded to the options. Detect their intent and respond briefly.
If unclear, ask them to clarify which option they meant (1, 2, or 3).""",

    "TOPIC_QUIZ": _BASE + """

PHASE: TOPIC QUIZ
Ask quiz question {qnum} of {total}: "{question}"
Be encouraging. Keep it short — just ask the question warmly.""",

    "TOPIC_QUIZ_FEEDBACK": _BASE + """

PHASE: QUIZ FEEDBACK
Topic: "{masked}"
Question: "{question}"
Student answer: "{answer}"

Rules:
- Do NOT quote or repeat the student's answer back to them.
- If correct: say "Correct!" and confirm in 1 sentence.
- If wrong or "I don't know": say "No worries!" then give the correct answer in 2 sentences drawn from the KB context below.
- Keep total response to 3 sentences max before the next instruction.
{next_or_done}""",

    "SESSION_QUIZ": _BASE + """

PHASE: SESSION SUMMARY — this is the final message of the session.
Topics covered: {covered_topics}

Write ONLY:
1. Warm closing (1 sentence)
2. Brief mastery summary — what they did well and what to revisit (3-4 sentences)
3. One NBCOT study tip relevant to today's topics — focus on CLINICAL REASONING, not memorization
4. End with: "Great work today! See you next time. 👋"

Do NOT ask any questions. Do NOT say "before you go". This is the final message — end the session warmly.""",

    "OUT_OF_SCOPE": _BASE + """

PHASE: OUT OF SCOPE
The student asked about something not covered in the knowledge base.
Politely say you do not have information on that specific topic,
mention that you focus on anatomy and neuroscience from OpenStax A&P,
and invite them to ask about a different anatomy/neuroscience topic.
Keep it warm and brief (2-3 sentences).""",
}


# ─────────────────────────────────────────────────────────────────────────────
# TutoringEngine
# ─────────────────────────────────────────────────────────────────────────────

class TutoringEngine:

    def __init__(self, retrieve_fn, groq_api_key: str,
                 model: str = "llama-3.1-8b-instant",
                 subject_domain: str = "human anatomy, physiology, or neuroscience"):
        self.retrieve = retrieve_fn
        self.llm = ChatGroq(
            model=model, temperature=0.4,
            max_tokens=600, api_key=groq_api_key
        )
        self.subject_domain = subject_domain
        self._graph = self._build_graph()
        self._reset_state()
        self._transcript_log: list = []
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── State management ──────────────────────────────────────────────────────

    def _reset_state(self):
        self.state: DialogueState = {
            "messages":              [],
            "current_input":         "",
            "phase":                 "RAPPORT",
            "total_turns":           0,
            "retrieved_chunks":      [],
            "masked_answer":         "",
            "topic_label":           "",
            "out_of_scope":          False,
            # target-aware fields
            "forbidden_terms":       [],
            "clue_dimensions_used":  [],
            "last_attempt_class":    "NONE",
            "partial_elements":      [],
            "last_tutor_response":   "",
            "target_type":           "structure",
            # stuck tracking
            "stuck_count":           0,
            "wrong_guesses":         [],
            "correct_count":         0,
            # session-level
            "weak_topics":           [],
            "covered_topics":        [],
            # quiz
            "quiz_questions":        [],
            "quiz_index":            0,
            "tutor_response":        "",
            # image path
            "is_image_path":              False,
            "kb_function":                "",
            "common_misidentifications":  [],
            "waiting_for_question":       False,
            # clinical scenario (Task 3)
            "clinical_scenario_question": "",
            "clinical_scenario_score":    0,
        }

    def _reset_for_new_topic(self):
        """Keep session-level memory; reset per-topic fields."""
        self.state.update({
            "current_input":        "",
            "phase":                "RAPPORT",
            "retrieved_chunks":     [],
            "masked_answer":        "",
            "topic_label":          "",
            "out_of_scope":         False,
            "forbidden_terms":      [],
            "clue_dimensions_used": [],
            "last_attempt_class":   "NONE",
            "partial_elements":     [],
            "last_tutor_response":  "",
            "target_type":          "structure",
            "stuck_count":          0,
            "wrong_guesses":        [],
            "quiz_questions":             [],
            "quiz_index":                 0,
            "tutor_response":             "",
            "is_image_path":              False,
            "kb_function":                "",
            "common_misidentifications":  [],
            "waiting_for_question":       False,
            # clinical scenario (Task 3)
            "clinical_scenario_question": "",
            "clinical_scenario_score":    0,
        })

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        prior_phase = self.state.get("phase", "RAPPORT")
        self.state["current_input"] = user_input
        result = self._graph.invoke(self.state)
        self.state = result
        response = result["tutor_response"]
        self._transcript_log.append({
            "role": "student", "phase": prior_phase,
            "text": user_input, "timestamp": datetime.now().isoformat(),
        })
        self._transcript_log.append({
            "role": "tutor", "phase": result.get("phase", prior_phase),
            "text": response, "timestamp": datetime.now().isoformat(),
        })
        return response

    def save_transcript(self, transcripts_dir: str, student_id: str = "student") -> str:
        os.makedirs(transcripts_dir, exist_ok=True)
        masked = self.state.get("masked_answer", "")
        data = {
            "student_id":     student_id,
            "session_id":     self._session_id,
            "masked_answer":  masked,
            "covered_topics": self.state.get("covered_topics", []),
            "weak_topics":    self.state.get("weak_topics", []),
            "turns":          self._transcript_log,
        }
        fname = (f"transcript_{student_id}_{self._session_id}"
                 f"_{masked.replace(' ', '_')[:20]}.json")
        path = os.path.join(transcripts_dir, fname)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Transcript] Saved: {path}")
        return path

    def get_phase(self) -> str:
        return self.state["phase"]

    def get_masked_answer(self) -> str:
        return self.state["masked_answer"]

    def is_done(self) -> bool:
        return self.state["phase"] == "DONE"

    # ─────────────────────────────────────────────────────────────────────────
    # Core LLM caller
    # ─────────────────────────────────────────────────────────────────────────

    def _call(self, system_prompt: str, user_msg: str = "",
              history: list = None) -> str:
        import time
        msgs = [SystemMessage(content=system_prompt)]
        if history:
            msgs.extend(history[-6:])
        if user_msg:
            msgs.append(HumanMessage(content=user_msg))
        for attempt in range(3):
            try:
                return self.llm.invoke(msgs).content.strip()
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep(2 ** attempt)
                else:
                    raise
        return self.llm.invoke(msgs).content.strip()

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: Forbidden-term shield
    # ─────────────────────────────────────────────────────────────────────────

    # ── safe ontology-type words: never added to the forbidden block ─────────
    _SAFE_TYPE_WORDS = {
        "nerve", "nerves", "muscle", "muscles", "vessel", "vessels",
        "artery", "arteries", "vein", "veins", "bone", "bones",
        "tract", "tracts", "lobe", "lobes", "process", "processes",
        "joint", "joints", "tendon", "tendons", "ligament", "ligaments",
        "region", "regions", "layer", "layers", "nucleus", "nuclei",
        "ganglion", "ganglia", "plexus", "organ", "organs",
        "structure", "structures", "cell", "cells", "fiber", "fibers",
        "pathway", "pathways", "area", "zone",
    }

    def _infer_target_type(self, masked: str, context: str) -> str:
        """
        Return a safe one-word ontology label for the masked answer
        (e.g. "nerve", "muscle", "vessel", "bone", "process", "structure").

        Strategy:
          1. Fast lexical check — if any known type word appears in masked, use it.
          2. If not found, ask the LLM once (cheap: single-word answer).

        This label is NEVER forbidden — it is injected into prompts so the tutor
        can say "which nerve" instead of the over-suppressed "which structure."
        """
        masked_lower = masked.lower()
        # Fast path: exact type word in the masked string itself
        for word in ("nerve", "muscle", "vessel", "artery", "vein", "bone",
                     "tract", "lobe", "joint", "tendon", "ligament",
                     "ganglion", "plexus", "nucleus", "process"):
            if word in masked_lower:
                return word

        # LLM path: ask for the concept category
        p = (
            f"Classify the anatomical concept '{masked}' into exactly ONE of these categories:\n"
            f"nerve, muscle, vessel, bone, joint, tract, lobe, ganglion, plexus,\n"
            f"tendon, ligament, process, organ, cell, fiber, structure\n"
            f"Context: {context[:400]}\n"
            f"Reply with ONE word only — the category name."
        )
        try:
            raw = self.llm.invoke([HumanMessage(content=p)]).content.strip().lower()
            # Accept only a word from our safe list; fallback to "structure"
            candidate = re.sub(r"[^a-z]", "", raw.split()[0]) if raw.split() else ""
            if candidate in self._SAFE_TYPE_WORDS:
                return candidate
        except Exception:
            pass
        return "structure"

    def _build_forbidden_set(self, masked: str, context: str) -> list:
        """
        Build the forbidden-term set for prompt injection.

        POLICY (fixes ontology-word suppression):
          1. Exact masked_answer string  — always forbidden
          2. Content tokens of masked_answer that are NOT safe type words — forbidden
             e.g. "median nerve" → "median" forbidden, "nerve" NOT forbidden
          3. LLM-generated synonyms/aliases — forbidden

        Safe type words (nerve, muscle, vessel, …) are deliberately excluded so
        the tutor can still say "which nerve" without leaking "median nerve."
        """
        forbidden = set()
        # 1. Exact term — always forbidden
        forbidden.add(masked.lower())

        # 2. Content tokens — skip stop words AND safe type words normally.
        #    G7 fix: if ALL content tokens are safe type words (e.g. masked =
        #    "nerve fiber", "muscle fiber", "bone marrow"), the loop below would
        #    produce zero individual tokens — leaving only the exact phrase in the
        #    set. The purity guard then only blocks the exact phrase, allowing the
        #    LLM to say individual words like "nerve" + "fiber" adjacently without
        #    triggering the guard. Fix: when all tokens are safe-type words, add
        #    them to forbidden anyway so those specific combinations are blocked.
        tokens = [tok for tok in re.split(r"[\s\-]+", masked.lower()) if len(tok) > 3]
        content_tokens = [t for t in tokens if t not in self._SAFE_TYPE_WORDS]
        if content_tokens:
            # Normal case: only add non-type-word tokens
            for tok in content_tokens:
                forbidden.add(tok)
        else:
            # All tokens are safe type words — add them all individually
            # so the specific combination cannot be reconstructed by the LLM.
            for tok in tokens:
                forbidden.add(tok)

        # 3. LLM-generated near-answer aliases
        p = (
            f"You are building a leakage shield for a Socratic tutor.\n"
            f"The hidden answer is: '{masked}'.\n"
            f"List up to 6 OTHER terms that would give away this answer if spoken — "
            f"synonyms, aliases, abbreviations, or common names.\n"
            f"Do NOT include generic category words like 'nerve', 'muscle', 'bone'.\n"
            f"Context snippet: {context[:600]}\n"
            f"Return ONLY a comma-separated list of terms, no explanation."
        )
        try:
            raw = self.llm.invoke([HumanMessage(content=p)]).content.strip()
            for t in re.split(r"[,;]+", raw):
                t = t.strip().lower()
                if t and len(t) > 2 and t not in self._SAFE_TYPE_WORDS:
                    forbidden.add(t)
        except Exception:
            pass
        return sorted(forbidden)

    def _forbidden_block(self, forbidden: list) -> str:
        """Format the forbidden list for prompt injection."""
        return "\n".join(f"  - {t}" for t in forbidden)

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: Attempt classifier
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_attempt(self, attempt: str, masked: str,
                          partial_elements: list) -> dict:
        """
        Classify the student's attempt into one of:
          CORRECT      — names the masked answer (exactly or with articles stripped)
          PARTIAL      — names a sub-concept that is part of the answer path
          WRONG_NAMED  — names a specific wrong structure/nerve/term
          DONT_KNOW    — expresses ignorance/gives up
          OTHER        — vague, off-topic, or uninterpretable

        Returns {classification, partial_elements (updated)}
        """
        # Fast exact-match check first (no LLM call needed)
        def _strip(s):
            return re.sub(r"\b(the|a|an)\b\s*", "", s.lower()).strip()

        if _strip(masked) in _strip(attempt):
            return {"classification": "CORRECT", "partial_elements": partial_elements}

        # LLM-based classification
        partial_str = ", ".join(partial_elements) if partial_elements else "none yet"
        p = (
            f"You are classifying a student's attempt in a Socratic tutoring session.\n"
            f"Hidden answer (DO NOT reveal): '{masked}'\n"
            f"Student said: \"{attempt}\"\n"
            f"Previously identified partial elements: {partial_str}\n\n"
            f"Classify the attempt as exactly ONE of:\n"
            f"  CORRECT     — the student named '{masked}' (even with different phrasing)\n"
            f"  PARTIAL     — the student named something that is literally a PART OF or\n"
            f"                directly INSIDE the target structure: e.g. a physical component,\n"
            f"                a sub-region, or a step within the target's own process.\n"
            f"                PARTIAL is ONLY for things that are structurally inside the target.\n"
            f"                It is NEVER for a different named cell/structure that exists\n"
            f"                alongside, wraps, or interacts with the target.\n"
            f"                NOTE: naming only a generic TYPE word (e.g. 'nerve', 'cell') → OTHER.\n"
            f"  WRONG_NAMED — the student named a specific WRONG structure, nerve, cell, or term.\n"
            f"                This includes ALL cases where the student names a distinct named\n"
            f"                entity that is NOT the answer — regardless of biological relationship:\n"
            f"                  • 'Schwann cell' when answer is 'neuron' → WRONG_NAMED\n"
            f"                  • 'glial cell' when answer is 'neuron' → WRONG_NAMED\n"
            f"                  • 'astrocyte' when answer is 'neuron' → WRONG_NAMED\n"
            f"                  • 'frontal lobe' when answer is 'occipital lobe' → WRONG_NAMED\n"
            f"                Biological proximity, wrapping, or interaction does NOT make it PARTIAL.\n"
            f"  DONT_KNOW   — the student expressed ignorance or gave up\n"
            f"  OTHER       — vague, question, or off-target\n\n"
            f"CRITICAL — apply this check FIRST before classifying:\n"
            f"  Did the student explicitly name a specific anatomical term, structure, cell, or nerve?\n"
            f"  If NO specific term was named (e.g. 'I don't know', 'not sure', 'give up', 'no idea',\n"
            f"  blank, vague phrases, or just punctuation) → MUST be DONT_KNOW or OTHER.\n"
            f"  WRONG_NAMED is ONLY valid when the student has explicitly named a specific wrong term.\n\n"
            f"Decision rule:\n"
            f"  - Any specific named structure/cell that is NOT the answer → WRONG_NAMED\n"
            f"  - Only a generic type word (cell, nerve, muscle) → OTHER\n"
            f"  - Something physically inside or a step within the target itself → PARTIAL\n\n"
            f"If PARTIAL, list only the SPECIFIC correct sub-concepts (not type words).\n"
            f"Reply in this exact format (2 lines):\n"
            f"CLASSIFICATION: <one of the five labels>\n"
            f"PARTIAL_ELEMENTS: <comma list of specific sub-concepts, or 'none'>"
        )
        try:
            raw = self.llm.invoke([HumanMessage(content=p)]).content.strip()
            cls_match = re.search(r"CLASSIFICATION:\s*(\w+)", raw)
            pel_match = re.search(r"PARTIAL_ELEMENTS:\s*(.+)", raw)
            cls = cls_match.group(1).upper() if cls_match else "OTHER"
            if cls not in ("CORRECT", "PARTIAL", "WRONG_NAMED", "DONT_KNOW", "OTHER"):
                cls = "OTHER"
            new_partials = list(partial_elements)
            if cls == "PARTIAL" and pel_match:
                for el in re.split(r"[,;]+", pel_match.group(1)):
                    el = el.strip()
                    if el and el.lower() != "none":
                        if el not in new_partials:
                            new_partials.append(el)
            return {"classification": cls, "partial_elements": new_partials}
        except Exception:
            return {"classification": "OTHER", "partial_elements": partial_elements}

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: Clue dimension selector
    # ─────────────────────────────────────────────────────────────────────────

    def _select_clue_dimension(self, used: list,
                                attempt_class: str,
                                wrong_guesses: list,
                                partial_elements: list = None) -> str:
        """
        Choose the next clue dimension.

        General policy:
          0. TURN 1 (attempt_class == "NONE", used == []):
             Use only _TURN1_DIMENSIONS (population, anatomical_neighbor, mechanism).
             These are broad — they give context/location without revealing the
             functional deficit map that directly implies the answer.
             Consequence, symptom, innervation_territory are EXCLUDED from turn 1
             because they enumerate the exact deficit pattern (which is near-answer).
          1. NEVER repeat the immediately preceding dimension.
          2. WRONG_NAMED → 'comparison' (contrasts target with wrong guess).
          3. PARTIAL → zoom-in cascade: innervation_territory → motor_test →
             mechanism → anatomical_neighbor → consequence.
          4. Default: cycle through full CLUE_DIMENSIONS, never reuse last.

        Domain-agnostic: "innervation_territory" works for nerves;
        "mechanism" works for physiology; "anatomical_neighbor" for structure
        questions; "motor_test" for muscle/function questions.
        """
        last = used[-1] if used else None
        partial_elements = partial_elements or []

        # ── TURN 1: broad dimension only ──────────────────────────────────────
        # attempt_class "NONE" + empty used = first turn (rapport+hint combined).
        if attempt_class == "NONE" and not used:
            for dim in _TURN1_DIMENSIONS:
                return dim          # always returns first available broad dim
            return _TURN1_DIMENSIONS[0]

        # ── WRONG_NAMED: comparison is universally the sharpest pivot ─────────
        if attempt_class == "WRONG_NAMED" and last != "comparison":
            return "comparison"

        # ── PARTIAL: zoom-in cascade ───────────────────────────────────────────
        if attempt_class == "PARTIAL":
            # Ordered zoom-in: mechanism → consequence → anatomical_neighbor → symptom
            # Excludes innervation_territory and motor_test — these are nerve/muscle-specific
            # and produce nonsense output for brain regions, cells, and physiology concepts.
            # The four retained dimensions generalize across all anatomical concept types.
            for dim in ("mechanism", "consequence", "anatomical_neighbor", "symptom"):
                if dim != last and dim not in used:
                    return dim
            # All zoom-in dims exhausted → any not-last unused dim
            for dim in CLUE_DIMENSIONS:
                if dim != last and dim not in used:
                    return dim

        # ── Default: cycle full catalogue, never repeat last ───────────────────
        for dim in CLUE_DIMENSIONS:
            if dim not in used:
                return dim

        # All exhausted — cycle again, skip the immediately preceding one
        for dim in CLUE_DIMENSIONS:
            if dim != last:
                return dim

        return CLUE_DIMENSIONS[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Hardened purity guard
    # ─────────────────────────────────────────────────────────────────────────

    def _purity_guard(self, response: str, forbidden: list,
                      sys_p: str, user_msg: str, history: list,
                      target_type: str = "structure") -> str:
        """
        Check every forbidden term against the response (token-level, not just
        exact substring). If any leak is found, regenerate with an explicit
        rewrite instruction listing every leaked term.

        Key rule: terms that appear in the student's own attempt (user_msg) are
        excluded from the leak check. The tutor is allowed — even expected — to
        reference what the student just said when giving a comparison clue.
        Blocking those words makes it impossible to say "not quite, a glial cell
        is different because..." when "glial" is in the forbidden set as an alias.
        Only the masked answer and its true synonyms need purity protection.
        """
        # Words the student just said — tutor may reference these freely
        attempt_tokens = set(re.sub(r"[^a-z ]", "", user_msg.lower()).split())

        leaked = [t for t in forbidden
                  if t in response.lower()
                  and not any(tok in attempt_tokens for tok in t.split())]
        if not leaked:
            return response
        leaked_str = ", ".join(f"'{t}'" for t in leaked)
        stricter = (
            sys_p +
            f"\n\nCRITICAL REWRITE: Your previous response contained forbidden term(s): {leaked_str}. "
            f"Rewrite your response completely WITHOUT any of these terms: {leaked_str}. "
            f"Use ONLY the allowed clue dimension. Do not name any nerve, structure, or synonym."
        )
        response = self._call(stricter, user_msg, history)
        # One more pass — if still leaking, use a safe functional clue fallback
        # that avoids naming the answer but still gives a meaningful hint.
        still_leaked = [t for t in forbidden if t in response.lower()]
        if still_leaked:
            response = (
                f"Let me give you a functional clue: this {target_type} plays a key role "
                f"in how the body coordinates and produces movement — without it, muscle "
                f"contraction as we know it would not be possible. "
                f"What {target_type} do you think this could be?"
            )
        return response

    # ─────────────────────────────────────────────────────────────────────────
    # Existing helpers (preserved)
    # ─────────────────────────────────────────────────────────────────────────

    def _student_doesnt_know(self, text: str) -> bool:
        prompt = (
            "You are judging a student's response in a tutoring session.\n"
            "Does this response mean the student does NOT know the answer?\n"
            "Examples of 'doesn't know': 'I don't know', 'no idea', 'idk', "
            "'not sure', 'can't remember', 'I give up', 'help me', "
            "'just tell me', 'confused', 'no clue', 'skip', shrug, "
            "or any expression of uncertainty or surrender.\n"
            f"Student said: \"{text}\"\n"
            "Reply with only YES or NO."
        )
        result = self.llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
        return result.startswith("Y")

    def _detect_post_reveal_choice(self, text: str) -> str:
        prompt = (
            "A student just finished learning a topic and was given four options:\n"
            "1 = quiz me on this topic\n"
            "2 = I want help with another topic\n"
            "3 = I'm done for now\n"
            "4 = give me a clinical scenario to apply this\n\n"
            "Examples of what maps to each option:\n"
            "  new_topic : 'move to a new topic', 'different topic', 'another topic',\n"
            "              'something else', 'explore another', 'new topic', 'switch topics',\n"
            "              'I would like help with another topic', 'let's try something different'\n"
            "  quiz      : 'quiz me', 'yes quiz', 'test me', 'yes I got it', 'quiz',\n"
            "              'yes quiz me on this'\n"
            "  done      : 'done', 'I am done', 'finished', 'that's all', 'goodbye',\n"
            "              'I'm done for now', 'end session'\n"
            "  clinical  : 'clinical scenario', 'apply this', 'give me a scenario',\n"
            "              'clinical challenge', '4', 'scenario', 'apply my knowledge'\n\n"
            f"Student said: \"{text}\"\n\n"
            "Which option did they choose?\n"
            "Reply with exactly one word: quiz, new_topic, done, clinical, or unclear.\n"
            "Use 'unclear' ONLY if the student's response genuinely does not map to any "
            "of the four options — e.g. random words, unrelated questions, or gibberish."
        )
        raw = self.llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        if "clinical" in raw or "scenario" in raw:
            return "clinical"
        if "new" in raw or "topic" in raw or "another" in raw or "move" in raw or "different" in raw or "switch" in raw:
            return "new_topic"
        if "done" in raw or "finish" in raw or "end" in raw:
            return "done"
        if "quiz" in raw:
            return "quiz"
        return "unclear"

    def _extract_masked(self, query: str, context: str) -> str:
        p = (
            "A student asked an anatomy question. Using the context, identify the specific "
            "answer the student is trying to learn.\n"
            "Rules:\n"
            "- Return the KEY ANSWER — not a term they already named in their question\n"
            "- Be specific and accurate, using terms from the context\n"
            "- Examples:\n"
            "    'What happens when radial nerve is damaged?' → 'wrist drop'\n"
            "    'What is the function of the cerebellum?' → 'motor coordination and balance'\n"
            "    'Which lobe handles vision?' → 'occipital lobe'\n"
            "    'Which structure connects muscle to bone?' → 'tendon'\n"
            "- Return ONLY the answer term or short phrase (1-5 words), no explanation\n"
            f"Question: {query}\nContext snippet: {context[:500]}"
        )
        return self.llm.invoke([HumanMessage(content=p)]).content.strip()

    def _is_in_scope(self, query: str, chunks: list) -> bool:
        """
        Determine whether the student's query is within the anatomy/neuroscience KB scope.

        Two-stage check:
        1. Domain guard (primary): ask the LLM whether the student's question is
           about anatomy, physiology, or neuroscience at all. This is intentionally
           broad — if the topic is ANY human anatomy concept, it passes.
           Only truly foreign topics (diabetes mechanisms, cooking, programming, etc.)
           are rejected here.
        2. Keyword overlap (secondary): if LLM call fails, fall back to checking
           that at least one key query word appears in the retrieved chunks.

        Returns False → router sets out_of_scope=True → OUT_OF_SCOPE response.
        """
        if not chunks:
            return False

        # Primary: domain guard — is this question within the configured subject domain?
        domain = getattr(self, "subject_domain", "human anatomy, physiology, or neuroscience")
        prompt = (
            f"Is this student question about {domain}?\n"
            f"Question: \"{query}\"\n\n"
            f"Answer YES if the question is clearly about {domain}.\n"
            f"Answer NO only if the question is clearly outside this subject area.\n"
            f"Reply with only YES or NO."
        )
        try:
            result = self.llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
            return result.startswith("Y")
        except Exception:
            pass

        # Fallback: keyword overlap
        query_words = set(re.sub(r"[^a-z ]", "", query.lower()).split())
        stop = {"what", "is", "the", "a", "an", "tell", "me", "about",
                "explain", "how", "does", "do", "can", "you", "i", "my"}
        key_words = query_words - stop
        if not key_words:
            return True
        top_text = " ".join(c["text"][:200].lower() for c in chunks[:3])
        overlap = sum(1 for w in key_words if w in top_text)
        return overlap >= max(1, len(key_words) // 3)

    def _get_context(self, state: DialogueState) -> str:
        return "\n\n".join(c["text"] for c in state["retrieved_chunks"])

    def _get_focused_context(self, masked: str, topic: str) -> str:
        query = f"{masked} {topic} anatomy physiology function"
        chunks = self.retrieve(query, top_k=5)
        masked_lower = masked.lower()
        scored = sorted(
            chunks,
            key=lambda c: (2 if masked_lower in c["text"].lower() else 0) +
                          (1 if c.get("topic", "") == topic else 0),
            reverse=True
        )
        return "\n\n".join(c["text"] for c in scored)

    def _generate_quiz_questions(self, masked: str, context: str, n: int = 2) -> list:
        p = (
            f"Generate exactly {n} SHORT quiz questions to test an OT student's "
            f"understanding of '{masked}'.\n\n"
            f"STRICT RULES:\n"
            f"- Each question must be ONE sentence only — no multi-part questions\n"
            f"- No 'describe', 'explain', 'discuss' — use 'what', 'which', 'how does', 'name'\n"
            f"- Keep each question under 20 words\n\n"
            f"Question 1: A simple factual question about what '{masked}' is or its main function.\n"
            f"  Example format: 'What is the main function of [structure]?'\n"
            f"Question 2: A one-sentence clinical question — name ONE specific deficit and ask "
            f"ONE thing about how it affects a patient.\n"
            f"  Example format: 'If [structure] is damaged, which daily activity would be most affected?'\n\n"
            f"Format: return exactly {n} numbered lines. One question per line.\n"
            f"Context: {context[:600]}"
        )
        raw = self.llm.invoke([HumanMessage(content=p)]).content.strip()
        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                q = re.sub(r"^[\d\.\-\s]+", "", line).strip()
                if q:
                    questions.append(q)
        while len(questions) < n:
            questions.append(
                f"A patient has damage to the {masked}. "
                f"How would this affect their daily activities as an OT patient?"
            )
        return questions[:n]

    def _generate_session_quiz_questions(self, covered_topics: list) -> list:
        topics_str = ", ".join(covered_topics) if covered_topics else "anatomy"
        p = (
            f"Generate 3 short-answer review questions for an OT student who studied: {topics_str}.\n"
            f"Question 1: A factual question about one of the topics (structure or function).\n"
            f"Question 2: A CLINICAL SCENARIO question — describe a patient with a specific deficit "
            f"related to one of the topics and ask the student to explain the impact on ADLs "
            f"or what clinical signs an OT would observe.\n"
            f"Question 3: An APPLICATION question — ask the student to compare two of the topics "
            f"OR describe how understanding one topic would inform an OT treatment plan.\n"
            f"Format: 3 numbered lines. Keep each question concise (1-2 sentences)."
        )
        raw = self.llm.invoke([HumanMessage(content=p)]).content.strip()
        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                q = re.sub(r"^[\d\.\-\s]+", "", line).strip()
                if q:
                    questions.append(q)
        while len(questions) < 3:
            questions.append(
                f"A patient has a deficit involving {topics_str}. "
                f"Explain how an OT would assess and address this in treatment."
            )
        return questions[:3]

    # ─────────────────────────────────────────────────────────────────────────
    # Graph nodes
    # ─────────────────────────────────────────────────────────────────────────

    def _router(self, state: DialogueState) -> DialogueState:
        user_input = state["current_input"]
        phase = state.get("phase", "RAPPORT")

        if phase in ("POST_REVEAL_WAIT", "CLINICAL_SCENARIO", "TOPIC_QUIZ",
                     "POST_TOPIC_QUIZ", "SESSION_QUIZ", "SESSION_QUIZ_FEEDBACK", "DONE"):
            return state

        if phase in ("HINT", "CLUE", "REVEAL"):
            return state

        # RAPPORT: fresh retrieval + scope check + forbidden-set construction.
        # Exception 1: if masked_answer is already set (image branch pre-wired the
        # state with a KB-grounded target), skip re-retrieval so the image
        # grounding is preserved.  _rapport will use the pre-loaded chunks.
        if state.get("masked_answer", ""):
            return state

        # Exception 2: if waiting_for_question is set (after out-of-scope or
        # similar), the student's input is an acknowledgment ("yes", "sure", etc.)
        # not a real anatomy question. Skip retrieval and let _rapport invite them.
        if state.get("waiting_for_question", False):
            return state

        # Exception 3: pure greeting — don't run retrieval or scope check,
        # let _rapport handle it as a warm welcome and invite an anatomy question.
        _GREETINGS = re.compile(
            r"^\s*(hi|hello|hey|howdy|good\s*(morning|afternoon|evening)|"
            r"what'?s up|sup|greetings|yo)\W*$", re.I
        )
        if _GREETINGS.match(user_input):
            return {**state, "waiting_for_question": True}

        chunks = self.retrieve(user_input, top_k=5)
        masked = self._extract_masked(user_input,
                                      "\n\n".join(c["text"] for c in chunks))

        in_scope = self._is_in_scope(user_input, chunks)

        topic = chunks[0]["topic"] if chunks else "anatomy"

        # Build forbidden set and infer target type (once per topic)
        ctx_str = "\n\n".join(c["text"] for c in chunks)
        forbidden    = self._build_forbidden_set(masked, ctx_str)
        target_type  = self._infer_target_type(masked, ctx_str)

        return {
            **state,
            "retrieved_chunks":      chunks,
            "masked_answer":         masked,
            "topic_label":           topic,
            "out_of_scope":          not in_scope,
            "forbidden_terms":       forbidden,
            "target_type":           target_type,
            "clue_dimensions_used":  [],
            "last_attempt_class":    "NONE",
            "partial_elements":      [],
        }

    def _out_of_scope(self, state: DialogueState) -> DialogueState:
        resp = self._call(PROMPTS["OUT_OF_SCOPE"], state["current_input"], state["messages"])
        return {
            **state,
            "messages": state["messages"] + [
                HumanMessage(content=state["current_input"]),
                AIMessage(content=resp)
            ],
            "tutor_response":        resp,
            "phase":                 "RAPPORT",
            "out_of_scope":          False,
            # Clear all stale retrieval state so _router does a fresh retrieval
            # on the student's actual next question instead of reusing this query's chunks.
            "masked_answer":         "",
            "retrieved_chunks":      [],
            "forbidden_terms":       [],
            "topic_label":           "",
            "target_type":           "structure",
            "clue_dimensions_used":  [],
            "last_attempt_class":    "NONE",
            "partial_elements":      [],
            "last_tutor_response":   "",
            "stuck_count":           0,
            "wrong_guesses":         [],
            "total_turns":           0,
            "waiting_for_question":  True,   # signal to _rapport to invite a question
        }

    def _rapport(self, state: DialogueState) -> DialogueState:
        """
        Combined rapport + first clue in a single LLM call.

        Text path  (is_image_path=False): RAPPORT_HINT — warm opener + broad text clue
        Image path (is_image_path=True):  IMAGE_RAPPORT_HINT — wraps the VLM-generated
          structural-class clue (stored in state["last_tutor_response"] by main.py)
          in a warm, natural opener sentence. The VLM clue is Sentence 2 verbatim so
          no new anatomy facts are injected — purity is preserved.

        In both paths the engine transitions to CLUE after this turn.
        """
        masked       = state.get("masked_answer", "this topic")
        attempt      = state["current_input"]
        forbidden    = state.get("forbidden_terms", [masked.lower()])
        target_type  = state.get("target_type", "structure")
        topic_area   = state.get("topic_label", "anatomy")
        is_image     = state.get("is_image_path", False)
        ctx          = self._get_context(state)

        # After out-of-scope rejection, student sent an acknowledgment ("yes",
        # "sure", etc.) rather than a real question. Invite them to ask one
        # instead of generating a clue with no masked answer.
        if state.get("waiting_for_question", False):
            resp = "Great! What anatomy or neuroscience topic would you like to explore?"
            return {
                **state,
                "messages": state["messages"] + [
                    HumanMessage(content=attempt),
                    AIMessage(content=resp),
                ],
                "tutor_response":       resp,
                "phase":                "RAPPORT",
                "waiting_for_question": False,
            }

        if is_image:
            # Use the VLM-generated first-clue stored in state by main.py.
            # Falls back to a safe generic clue if nothing was stored.
            vlm_clue = state.get("last_tutor_response", "") or (
                f"This is a {target_type} structure visible in the diagram."
            )
            sys_p = PROMPTS["IMAGE_RAPPORT_HINT"].format(
                target_type=target_type,
                topic_area=topic_area,
                vlm_clue=vlm_clue,
                forbidden_block=self._forbidden_block(forbidden),
                context=ctx[:2000],
            )
        else:
            # Text path — unchanged
            dim       = self._select_clue_dimension([], "NONE", [], partial_elements=[])
            dim_instr = _DIM_INSTRUCTION[dim]
            sys_p = PROMPTS["RAPPORT_HINT"].format(
                target_type=target_type,
                topic_area=topic_area,
                dim_instruction=dim_instr,
                masked=masked,
                forbidden_block=self._forbidden_block(forbidden),
                context=ctx[:2000],
            )

        resp = self._call(sys_p, attempt, state["messages"])
        resp = self._purity_guard(resp, forbidden, sys_p, attempt, state["messages"],
                                  target_type=target_type)

        return {
            **state,
            "messages": state["messages"] + [
                HumanMessage(content=attempt),
                AIMessage(content=resp)
            ],
            "tutor_response":      resp,
            "phase":               "CLUE",
            "last_tutor_response": resp,
            "total_turns":         state.get("total_turns", 0) + 1,
        }

    def _hint(self, state: DialogueState) -> DialogueState:
        """
        Fallback first-clue turn — used only if the engine enters HINT phase
        directly (e.g. from VLM flow) without going through _rapport.
        """
        masked      = state["masked_answer"]
        attempt     = state["current_input"]
        forbidden   = state.get("forbidden_terms", [masked.lower()])
        target_type = state.get("target_type", "structure")

        dim       = self._select_clue_dimension([], "NONE", [], partial_elements=[])
        dim_instr = _DIM_INSTRUCTION[dim]
        ctx       = self._get_context(state)

        sys_p = PROMPTS["HINT"].format(
            target_type=target_type,
            dim_instruction=dim_instr,
            forbidden_block=self._forbidden_block(forbidden),
            context=ctx[:2000],
        )
        resp = self._call(sys_p, attempt, state["messages"])
        resp = self._purity_guard(resp, forbidden, sys_p, attempt, state["messages"],
                                  target_type=state.get("target_type", "structure"))

        return {
            **state,
            "messages": state["messages"] + [
                HumanMessage(content=attempt),
                AIMessage(content=resp)
            ],
            "tutor_response":      resp,
            "phase":               "CLUE",
            "last_tutor_response": resp,
            "total_turns":         state.get("total_turns", 0) + 1,
        }

    def _clue(self, state: DialogueState) -> DialogueState:
        """
        Turn 2+ — simplified fixed-budget Socratic controller.

        Contract:
          - Fast-path: exact correct answer → REVEAL_CORRECT immediately.
          - Fast-path: explicit reveal request ("tell me", "just tell me") → _reveal.
          - Clue budget: total_turns 1, 2, 3 → progressively narrower clues.
          - Budget exhausted (turn >= 3) and student still wrong/stuck → _reveal.
          - One unified CLUE prompt — no pre-classification, no dimension dispatch.
          - Context comes from already-retrieved state chunks (no re-retrieval).
        """
        masked      = state["masked_answer"]
        attempt     = state["current_input"]
        forbidden   = state.get("forbidden_terms", [masked.lower()])
        target_type = state.get("target_type", "structure")
        topic_area  = state.get("topic_label", "anatomy")
        prev_clue   = state.get("last_tutor_response", "")
        # total_turns was incremented by _rapport to 1 after the first clue.
        # So here: total_turns=1 means this is clue 2, total_turns=2 → clue 3.
        total_turns = state.get("total_turns", 1)
        clue_number = total_turns + 1   # clue we are about to give (2 or 3+)

        # ── Fast path 1: exact correct answer ────────────────────────────────
        def _strip(s):
            return re.sub(r"\b(the|a|an)\b\s*", "", s.lower()).strip()

        if _strip(masked) in _strip(attempt):
            ctx   = self._get_context(state)
            sys_p = PROMPTS["REVEAL_CORRECT"].format(
                context=ctx[:2000], masked=masked, attempt=attempt
            )
            resp = self._call(sys_p, attempt, history=[])
            return {
                **state,
                "messages": state["messages"] + [
                    HumanMessage(content=attempt), AIMessage(content=resp)
                ],
                "tutor_response":     resp,
                "phase":              "POST_REVEAL_WAIT",
                "correct_count":      state.get("correct_count", 0) + 1,
                "total_turns":        total_turns + 1,
                "covered_topics":     state.get("covered_topics", []) + [masked],
                "last_attempt_class": "CORRECT",
                "last_tutor_response": resp,
            }

        # ── Fast path 2: explicit reveal request ─────────────────────────────
        _REVEAL_TRIGGERS = re.compile(
            r"\b(tell me|just tell|give me the answer|what is it|what'?s the answer"
            r"|i give up|reveal|just say it|answer please)\b",
            re.IGNORECASE
        )
        if _REVEAL_TRIGGERS.search(attempt):
            return self._reveal({**state, "last_attempt_class": "DONT_KNOW"})

        # ── Clue budget exhausted → reveal ───────────────────────────────────
        # After 3 clues (total_turns >= 3) and student is still not correct,
        # reveal regardless of what they said.
        if total_turns >= 3:
            return self._reveal({**state, "last_attempt_class": "OTHER"})

        # ── Give the next clue ────────────────────────────────────────────────
        # Dimension sequence differs by path so clues are pedagogically appropriate.
        #
        # TEXT PATH (is_image_path=False):
        #   Clue 1 (rapport): anatomical_neighbor — location / anatomical setting
        #   Clue 2:           mechanism           — function / what it does
        #   Clue 3:           comparison          — contrast with wrong guess
        #
        # IMAGE PATH (is_image_path=True):
        #   Clue 1 (rapport): structural_class    — VLM first clue (wrapped by _rapport)
        #   Clue 2:           image_function      — what this structure does / its role
        #   Clue 3:           image_insertion_clinical — insertion/origin OR clinical deficit
        #
        # The image dimensions are tuned for diagrams: students can see the structure
        # visually so "anatomical_neighbor" (location) is redundant; function and
        # insertion/clinical are the pedagogically high-value follow-ups.
        is_image = state.get("is_image_path", False)

        # ── Image path: check common misidentifications ───────────────────────
        # If the student named a known wrong answer for this image, override the
        # normal dimension and use "image_insertion_clinical" — which is tuned to
        # contrast the target against related-but-wrong structures.
        common_misids = state.get("common_misidentifications", [])
        attempt_lower = attempt.lower()
        is_common_misid = is_image and any(
            m.lower() in attempt_lower for m in common_misids
        )

        if is_image:
            if is_common_misid:
                # Known wrong guess — pivot immediately to contrasting clue
                clue_dim = "image_insertion_clinical"
            else:
                _IMG_CLUE_DIMS = {
                    2: "image_function",
                    3: "image_insertion_clinical",
                }
                clue_dim = _IMG_CLUE_DIMS.get(clue_number, "image_function")
        else:
            # Clue 2 → mechanism (function/what it does)
            # Clue 3 → comparison ONLY if student named something wrong (WRONG_NAMED)
            #           otherwise consequence — comparison needs a wrong guess to contrast against
            last_attempt_class = state.get("last_attempt_class", "NONE")
            if clue_number == 3 and last_attempt_class != "WRONG_NAMED":
                clue_dim = "consequence"
            elif clue_number == 3:
                clue_dim = "comparison"
            else:
                clue_dim = "mechanism"
        dim_instr = _DIM_INSTRUCTION[clue_dim]

        # ── Concept proxy block (image path only) ─────────────────────────────
        # Inject kb_function from image_metadata.json as allowed proxy phrases.
        # This gives the LLM safe vocabulary to describe the structure's role
        # without naming it — replacing the need for blanket term suppression.
        kb_function = state.get("kb_function", "")
        if is_image and kb_function:
            concept_proxy_block = (
                f"CONCEPT PROXIES — you MAY use these phrases to describe the "
                f"hidden structure's role without naming it:\n"
                f"  \"{kb_function}\"\n\n"
            )
        else:
            concept_proxy_block = ""

        # ── Classify the student's attempt before building the prompt ─────────
        # Fast-path override: if the student named a known misidentification from
        # image_metadata.json, force WRONG_NAMED without calling the LLM classifier.
        # This bypasses LLM ambiguity for biologically-proximate cells/structures
        # (e.g. "astrocyte" or "glial cell" when answer is "neuron") that the LLM
        # often mis-classifies as PARTIAL due to biological association.
        attempt_lower_cls = attempt.lower()
        if is_image and any(m.lower() in attempt_lower_cls for m in common_misids):
            attempt_class    = "WRONG_NAMED"
            partial_elements = state.get("partial_elements", [])
        else:
            classification   = self._classify_attempt(
                attempt, masked, state.get("partial_elements", [])
            )
            attempt_class    = classification["classification"]
            partial_elements = classification["partial_elements"]

        # Build a short partial hint for PARTIAL class (safe — sub-concepts only)
        partial_hint = ""
        if attempt_class == "PARTIAL" and partial_elements:
            partial_hint = f"you identified {', '.join(partial_elements)} correctly."

        # Build wrong guesses block so comparison clues reference ALL prior wrong
        # guesses, not just the current one — prevents identical clues each turn.
        wrong_guesses = state.get("wrong_guesses", [])
        if wrong_guesses and attempt_class == "WRONG_NAMED":
            all_wrong = [g["text"] if isinstance(g, dict) else str(g) for g in wrong_guesses]
            # Include current attempt if not already in list
            if attempt not in all_wrong:
                all_wrong.append(attempt)
            wrong_guesses_block = (
                f"PREVIOUS WRONG GUESSES BY STUDENT: {', '.join(all_wrong)}\n"
                f"When giving a comparison clue, contrast the hidden answer against "
                f"ALL of these wrong guesses — do not repeat the same comparison as before.\n\n"
            )
        else:
            wrong_guesses_block = ""

        ctx   = self._get_context(state)
        sys_p = PROMPTS["CLUE"].format(
            topic_area=topic_area,
            target_type=target_type,
            masked=masked,
            clue_number=clue_number,
            dim_instruction=dim_instr,
            concept_proxy_block=concept_proxy_block,
            wrong_guesses_block=wrong_guesses_block,
            prev_clue=prev_clue or "(none yet)",
            attempt=attempt,
            attempt_class=attempt_class,
            partial_hint=partial_hint,
            forbidden_block=self._forbidden_block(forbidden),
            context=ctx[:2000],
        )
        resp = self._call(sys_p, attempt, state["messages"])
        resp = self._purity_guard(resp, forbidden, sys_p, attempt, state["messages"],
                                  target_type=target_type)

        # Track wrong guesses so comparison clues can reference all of them
        updated_wrong_guesses = list(state.get("wrong_guesses", []))
        if attempt_class == "WRONG_NAMED":
            updated_wrong_guesses.append({
                "text":           attempt,
                "classification": attempt_class,
                "turn":           total_turns + 1,
            })

        return {
            **state,
            "messages": state["messages"] + [
                HumanMessage(content=attempt), AIMessage(content=resp)
            ],
            "tutor_response":      resp,
            "phase":               "CLUE",
            "last_tutor_response": resp,
            "total_turns":         total_turns + 1,
            "last_attempt_class":  attempt_class,
            "partial_elements":    partial_elements,
            "wrong_guesses":       updated_wrong_guesses,
        }

    def _reveal(self, state: DialogueState) -> DialogueState:
        masked   = state["masked_answer"]
        ctx      = self._get_focused_context(masked, state.get("topic_label", ""))
        sys_p    = PROMPTS["REVEAL"].format(context=ctx[:2000], masked=masked)
        user_msg = state["current_input"] or "I don't know — please explain."
        # Pass empty history: prior forbidden-term injections must not suppress the reveal.
        resp     = self._call(sys_p, user_msg, history=[])

        weak    = state.get("weak_topics", [])
        topic   = state.get("topic_label", masked)
        if topic and topic not in weak:
            weak = weak + [topic]
        covered = state.get("covered_topics", [])
        if masked and masked not in covered:
            covered = covered + [masked]

        return {
            **state,
            "messages": state["messages"] + [
                HumanMessage(content=user_msg), AIMessage(content=resp)
            ],
            "tutor_response": resp,
            "phase":          "POST_REVEAL_WAIT",
            "weak_topics":    weak,
            "covered_topics": covered,
            "total_turns":    state.get("total_turns", 0) + 1,
        }

    def _post_reveal_wait(self, state: DialogueState) -> DialogueState:
        choice = self._detect_post_reveal_choice(state["current_input"])

        if choice == "new_topic":
            resp = "Great! What topic would you like to explore next?"
            new_state = {**state}
            new_state.update({
                "phase":                 "RAPPORT",
                "masked_answer":         "",
                "topic_label":           "",
                "retrieved_chunks":      [],
                "stuck_count":           0,
                "wrong_guesses":         [],
                "correct_count":         0,
                "quiz_questions":        [],
                "quiz_index":            0,
                "out_of_scope":          False,
                "forbidden_terms":       [],
                "clue_dimensions_used":  [],
                "last_attempt_class":    "NONE",
                "partial_elements":      [],
                "last_tutor_response":   "",
                "target_type":           "structure",
                "total_turns":           0,
            })
            # Reset LLM message history so previous topic context doesn't bleed
            # into the new topic. The chat UI still shows full history via DB.
            new_state["messages"] = [AIMessage(content=resp)]
            new_state["tutor_response"] = resp
            return new_state

        if choice == "done":
            return self._session_quiz({
                **state,
                "messages": state["messages"] + [
                    HumanMessage(content=state["current_input"])
                ],
            })

        if choice == "clinical":
            return self._clinical_scenario_ask(state)

        # G5 fix: unclear response → ask for clarification, stay in POST_REVEAL_WAIT.
        # Previously the fallback was to start a quiz, which was wrong for random
        # input like "banana" or unrelated questions.
        if choice == "unclear":
            resp = (
                "Sorry, I didn't quite catch that! Please choose one of the options:\n"
                "  1  Yes, quiz me on this topic\n"
                "  2  I'd like help with another topic\n"
                "  3  I'm done for now\n"
                "  4  Give me a clinical scenario to apply this"
            )
            return {
                **state,
                "messages": state["messages"] + [
                    HumanMessage(content=state["current_input"]),
                    AIMessage(content=resp)
                ],
                "tutor_response": resp,
                "phase":          "POST_REVEAL_WAIT",   # stay here until clear choice
            }

        masked = state["masked_answer"]
        ctx    = self._get_focused_context(masked, state.get("topic_label", ""))
        qs     = self._generate_quiz_questions(masked, ctx, n=2)
        sys_p  = PROMPTS["TOPIC_QUIZ"].format(question=qs[0], qnum=1, total=len(qs))
        # Use clean history to avoid contamination from clinical eval SCORE/CORRECT blocks
        resp   = self._call(sys_p, "", history=[])
        clean_messages = [AIMessage(content=resp)]
        return {
            **state,
            "messages":       clean_messages,
            "tutor_response": resp,
            "phase":          "TOPIC_QUIZ",
            "quiz_questions": qs,
            "quiz_index":     0,
        }

    def _topic_quiz(self, state: DialogueState) -> DialogueState:
        qs       = state.get("quiz_questions", [])
        idx      = state.get("quiz_index", 0)
        masked   = state["masked_answer"]
        answer   = state["current_input"]
        question = qs[idx] if idx < len(qs) else ""
        next_idx = idx + 1
        is_last  = next_idx >= len(qs)

        if is_last:
            next_part = (
                "2. Since that was the last question, say:\n"
                "   \"Nice work on the quiz! What would you like to do next?\n"
                "   1  Explore another topic\n"
                "   2  I'm done for now — give me my session summary\""
            )
        else:
            next_q = qs[next_idx]
            next_part = f"2. Ask the next question: \"{next_q}\""

        ctx   = self._get_focused_context(masked, state.get("topic_label", ""))
        sys_p = PROMPTS["TOPIC_QUIZ_FEEDBACK"].format(
            answer=answer, masked=masked, question=question, next_or_done=next_part
        )
        sys_p += f"\n\nKB CONTEXT (use this to reveal the answer if needed):\n{ctx[:1500]}"
        resp = self._call(sys_p, "", history=[])

        # Always append the options explicitly after the last question so
        # the LLM cannot accidentally omit them.
        if is_last:
            resp = resp.rstrip()
            if "explore another" not in resp.lower() and "done for now" not in resp.lower():
                resp += (
                    "\n\nWhat would you like to do next?\n"
                    "  1  Explore another topic\n"
                    "  2  I'm done for now — give me my session summary"
                )

        clean_messages = [AIMessage(content=resp)]
        return {
            **state,
            "messages":       clean_messages,
            "tutor_response": resp,
            "phase":          "POST_TOPIC_QUIZ" if is_last else "TOPIC_QUIZ",
            "quiz_index":     next_idx,
        }

    def _post_topic_quiz(self, state: DialogueState) -> DialogueState:
        choice = self._detect_post_reveal_choice(state["current_input"])
        if choice == "done":
            return self._session_quiz({
                **state,
                "messages": state["messages"] + [
                    HumanMessage(content=state["current_input"])
                ],
            })
        if choice == "unclear":
            resp = (
                "Sorry, I didn't catch that! Please choose:\n"
                "  1  Explore another topic\n"
                "  2  I'm done for now — give me my session summary"
            )
            return {
                **state,
                "messages": state["messages"] + [
                    HumanMessage(content=state["current_input"]),
                    AIMessage(content=resp),
                ],
                "tutor_response": resp,
                "phase":          "POST_TOPIC_QUIZ",
            }
        resp = "Awesome! What topic would you like to explore next?"
        new_state = {**state}
        new_state.update({
            "phase":                 "RAPPORT",
            "masked_answer":         "",
            "topic_label":           "",
            "retrieved_chunks":      [],
            "stuck_count":           0,
            "wrong_guesses":         [],
            "correct_count":         0,
            "quiz_questions":        [],
            "quiz_index":            0,
            "out_of_scope":          False,
            "forbidden_terms":       [],
            "clue_dimensions_used":  [],
            "last_attempt_class":    "NONE",
            "partial_elements":      [],
            "last_tutor_response":   "",
            "target_type":           "structure",
            "total_turns":           0,
        })
        # Reset LLM message history so previous topic context doesn't bleed in
        new_state["messages"] = [AIMessage(content=resp)]
        new_state["tutor_response"] = resp
        return new_state

    def _session_quiz(self, state: DialogueState) -> DialogueState:
        covered = state.get("covered_topics", [])
        sys_p   = PROMPTS["SESSION_QUIZ"].format(
            covered_topics=", ".join(covered) if covered else "anatomy topics",
        )
        resp = self._call(sys_p, "", history=[])
        return {
            **state,
            "messages": [AIMessage(content=resp)],
            "tutor_response": resp,
            "phase":          "DONE",
        }

    def _session_quiz_feedback(self, state: DialogueState) -> DialogueState:
        qs       = state.get("quiz_questions", [])
        idx      = state.get("quiz_index", 0)
        answer   = state["current_input"]
        question = qs[idx] if idx < len(qs) else ""
        covered  = state.get("covered_topics", [])
        next_idx = idx + 1
        is_last  = next_idx >= len(qs)

        if is_last:
            next_part = (
                "2. Since that was the final question, write a warm 2-sentence closing message "
                "and encourage them to keep studying!"
            )
            next_phase = "DONE"
        else:
            next_q = qs[next_idx]
            next_part = f"2. Ask the next review question: \"{next_q}\""
            next_phase = "SESSION_QUIZ_FEEDBACK"

        sys_p = PROMPTS["SESSION_QUIZ_FEEDBACK"].format(
            answer=answer, question=question,
            qnum=idx + 1, total=len(qs),
            covered_topics=", ".join(covered) if covered else "anatomy",
            next_or_done=next_part
        )
        resp = self._call(sys_p, "", state["messages"])
        return {
            **state,
            "messages": state["messages"] + [
                HumanMessage(content=answer), AIMessage(content=resp)
            ],
            "tutor_response": resp,
            "phase":          next_phase,
            "quiz_index":     next_idx,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Task 3: Clinical Scenario Synthesis
    # ─────────────────────────────────────────────────────────────────────────

    def _clinical_scenario_ask(self, state: DialogueState) -> DialogueState:
        """Generate an open-ended OT clinical scenario question grounded in KB."""
        masked  = state["masked_answer"]
        ctx     = self._get_focused_context(masked, state.get("topic_label", ""))
        sys_p   = PROMPTS["CLINICAL_SCENARIO_PROMPT"].format(
            masked=masked, context=ctx[:2000]
        )
        resp = self._call(sys_p, "", history=[])
        return {
            **state,
            "messages": state["messages"] + [
                HumanMessage(content=state["current_input"]),
                AIMessage(content=resp),
            ],
            "tutor_response":             resp,
            "phase":                      "CLINICAL_SCENARIO",
            "clinical_scenario_question": resp,
            "clinical_scenario_score":    0,
        }

    def _clinical_scenario_eval(self, state: DialogueState) -> DialogueState:
        """Evaluate student's free-text answer against KB gold standard, then return to POST_REVEAL_WAIT."""
        masked   = state["masked_answer"]
        question = state.get("clinical_scenario_question", "")
        answer   = state["current_input"]
        ctx      = self._get_focused_context(masked, state.get("topic_label", ""))

        sys_p = PROMPTS["CLINICAL_SCENARIO_EVAL"].format(
            masked=masked, question=question,
            context=ctx[:2500], answer=answer,
        )
        raw = self._call(sys_p, "", history=[])

        # Parse score
        score = 0
        score_m = re.search(r"SCORE:\s*(\d+)", raw)
        if score_m:
            score = min(100, max(0, int(score_m.group(1))))

        # Update weak/mastered based on score
        topic = state.get("topic_label", masked)
        weak     = list(state.get("weak_topics", []))
        covered  = list(state.get("covered_topics", []))
        if score >= 70:
            # mastered — remove from weak if present
            weak = [t for t in weak if t.lower() != topic.lower()]
        else:
            if topic and topic not in weak:
                weak = weak + [topic]

        # Build feedback response shown to student
        feedback = (
            f"{raw}\n\n"
            "---\n"
            "What would you like to do next?\n"
            "  1  Quiz me on this topic\n"
            "  2  I'd like help with another topic\n"
            "  3  I'm done for now"
        )

        # Reset messages so the structured SCORE/CORRECT/MISSING text doesn't
        # contaminate the next LLM call — same pattern as the new_topic reset.
        clean_messages = [AIMessage(content=feedback)]

        return {
            **state,
            "messages":                clean_messages,
            "tutor_response":          feedback,
            "phase":                   "POST_REVEAL_WAIT",
            "clinical_scenario_score": score,
            "weak_topics":             weak,
            "covered_topics":          covered,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Graph construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_graph(self):
        def route(state: DialogueState) -> str:
            if state.get("out_of_scope"):
                return "out_of_scope_node"
            return {
                "RAPPORT":               "rapport",
                "HINT":                  "hint",
                "CLUE":                  "clue",
                "REVEAL":                "reveal",
                "POST_REVEAL_WAIT":      "post_reveal_wait",
                "CLINICAL_SCENARIO":     "clinical_scenario_eval",
                "TOPIC_QUIZ":            "topic_quiz",
                "POST_TOPIC_QUIZ":       "post_topic_quiz",
                "SESSION_QUIZ":          "session_quiz",
                "SESSION_QUIZ_FEEDBACK": "session_quiz_feedback",
                "DONE":                  END,
            }.get(state.get("phase", "RAPPORT"), END)

        builder = StateGraph(DialogueState)
        nodes = {
            "router":                self._router,
            "out_of_scope_node":     self._out_of_scope,
            "rapport":               self._rapport,
            "hint":                  self._hint,
            "clue":                  self._clue,
            "reveal":                self._reveal,
            "post_reveal_wait":      self._post_reveal_wait,
            "clinical_scenario_eval": self._clinical_scenario_eval,
            "topic_quiz":            self._topic_quiz,
            "post_topic_quiz":       self._post_topic_quiz,
            "session_quiz":          self._session_quiz,
            "session_quiz_feedback": self._session_quiz_feedback,
        }
        for name, fn in nodes.items():
            builder.add_node(name, fn)
        builder.set_entry_point("router")
        builder.add_conditional_edges("router", route)
        for name in nodes:
            if name != "router":
                builder.add_edge(name, END)
        return builder.compile()
