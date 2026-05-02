# Socratic-OT: Project Context File
# Last Updated: April 17, 2026
# Purpose: Single source of truth for all project state, metrics, architecture, and plans.
#          Read this at the start of every session — no need to re-derive context.

---

## 1. Project Identity

- **Title:** Socratic-OT: A Grounded Multimodal Socratic Tutor for Anatomy and Neuroscience Learning in Rehabilitation Science
- **Team:** Vidhyadhari Bandaru (bandaru7@buffalo.edu) · Richie M Ilavarapu (richiemo@buffalo.edu)
- **Course:** NLP / MSOT coursework, University at Buffalo, Spring 2025
- **Repo root:** `/Users/vidhyadharibandaru/Documents/Socratic_OT/`
- **Hugging Face Space:** Gradio app (`app.py` / `main.py`)

---

## 2. Core Design Principle

> **Tutor-not-teller**: The system NEVER reveals the answer directly. It masks the target answer and guides the student through Socratic clues (RAPPORT → CLUE 1 → CLUE 2 → CLUE 3 → REVEAL). The answer is only revealed after clues are exhausted, the student gives up explicitly, or the student guesses correctly.

---

## 3. Knowledge Base (Final State)

| Asset | Count |
|---|---|
| Source | OpenStax Anatomy & Physiology 2e (all 28 chapters) |
| Text chunks | **997 chunks** (~341 words each) |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent, local) |
| Labeled anatomy images | **6 images** |
| Image metadata records | **6 records** |

**Images in KB:**
- IMG001_neuron_structure.PNG — nervous tissue
- IMG002_brain_lobes.PNG — nervous system basics
- IMG003_spinal_cord_section.PNG — nervous system basics
- IMG004_nervous_system_overview.jpg — nervous system basics
- IMG005_brachial_plexus.png — peripheral nerves
- IMG006_skeletal_muscle_fiber.png — muscle structure

---

## 4. System Architecture (5 Layers)

```
Student Input (text or image)
        ↓
1. ROUTING LAYER — classifies input as text / image / follow-up
        ↓
2. RETRIEVAL LAYER — semantic search in ChromaDB (top-5 chunks)
                   — metadata-aware reranking by topic label
        ↓
3. TUTORING POLICY LAYER (LangGraph FSM)
   Phases: RAPPORT → CLUE(1-3) → REVEAL → POST_REVEAL_WAIT → TOPIC_QUIZ → DONE
   - _build_forbidden_set(): masks answer + LLM-generated aliases
   - _classify_attempt(): LLM judge → CORRECT / PARTIAL / WRONG_NAMED / DONT_KNOW / OTHER
   - _purity_guard(): checks response for forbidden term leakage, regenerates if leaked
   - _is_in_scope(): LLM domain guard — rejects non-anatomy questions
   - Clue dimensions: comparison (WRONG_NAMED), image_function, image_insertion_clinical
   - common_misidentifications fast-path: bypasses LLM classifier for known wrong guesses
        ↓
4. GENERATION LAYER
   - Text: Groq / LLaMA 3.1 8B Instruct (primary) → GPT-4o-mini (fallback)
   - Vision (VLM): Groq llama-4-scout-17b (primary) → LLaVA-NeXT (GPU, if available) → GPT-4o
        ↓
5. SESSION MEMORY & ASSESSMENT
   - Tracks weak topics, wrong guesses, stuck count
   - Post-topic mastery quiz (3 questions)
   - Session memory resets per conversation
```

**Orchestration:** LangGraph state machine (`src/tutor.py`)
**Key files:**
- `src/tutor.py` — TutoringEngine, FSM, all policy logic
- `src/vlm.py` — VLMModule, image analysis pipeline
- `src/knowledge_base.py` — KB build, ChromaDB, retriever
- `src/evaluation.py` — audit_purity(), run_ragas(), run_vlm_blind_test()
- `main.py` / `app.py` — Gradio UI entry point
- `generate_transcripts.py` — generates 5 text-path transcripts
- `generate_image_transcripts.py` — generates 3 image-path transcripts
- `Evaluation/transcripts/` — 8 saved transcript JSON files
- `Evaluation/ragas_results.json` — RAGAS scores
- `Evaluation/purity_audit.json` — purity scores
- `Evaluation/vlm_blind_test.json` — VLM scores

---

## 5. Baseline Evaluation Results (Milestone 2 — April 17, 2026)

### 5.1 Socratic Purity Audit
- **Score: 8/8 = 100%** (target: ≥ 5/5)
- Method: String scan — checks if masked answer appears in any CLUE-phase tutor turn
- Transcripts tested: 5 text-path + 3 image-path
- All 8 passed — zero leakage across all scenarios

**Scenarios tested:**
| ID | Scenario | Masked Answer | Result |
|---|---|---|---|
| eval_s1 | Normal flow, all wrong → REVEAL | neuron | ✅ PURE |
| eval_s2 | Correct after 2nd clue | occipital lobe | ✅ PURE |
| eval_s3 | Stuck 3× (DONT_KNOW) → REVEAL | brachial plexus | ✅ PURE |
| eval_s4 | Out-of-KB topic (mRNA vaccines) → rejected | — | ✅ PURE |
| eval_s5 | Wrong named guess → comparison clue → correct | sarcomere | ✅ PURE |
| eval_img1 | In-KB anatomy image (neuron) → Socratic flow | neuron | ✅ PURE |
| eval_img2 | Non-anatomy image (noise) → IS_ANATOMY guard fires | — | ✅ PURE |
| eval_img3 | Synthetic 3D neuron (unlabeled) → fuzzy KB match | neuron | ✅ PURE |

### 5.2 RAGAS Groundedness (Judge: qwen/qwen3-32b, 20 Q&A pairs)
| Metric | Score | Target | Status |
|---|---|---|---|
| Faithfulness | **0.9135** | ≥ 0.90 | ✅ PASS |
| Answer Relevancy | **0.9694** | ≥ 0.85 | ✅ PASS |
| Context Recall | **0.6471** | ≥ 0.80 | ❌ BELOW |
| Context Precision | **0.7792** | ≥ 0.80 | ❌ BELOW |

**Observations:**
- Faithfulness (0.91) and Answer Relevancy (0.97) are strong — the tutor answers are grounded in retrieved context and on-topic
- Context Recall (0.65) is the weakest metric — the retriever fails to surface all gold-standard content for broad questions that span multiple KB topics
- Context Precision (0.78) is close to target — most retrieved chunks are relevant but some noise chunks slip through
- Root cause for recall/precision gap: retriever uses pure semantic similarity; questions spanning multiple chapters (e.g., "describe the dorsal column pathway") pull chunks from different sections with limited overlap

### 5.3 VLM Blind Test (6 unlabeled anatomy images, Groq llama-4-scout)
| Image | Predicted | Expected | Result |
|---|---|---|---|
| IMG001_neuron_structure.PNG | neuron | neuron | ✅ |
| IMG002_brain_lobes.PNG | brain lobes | brain lobes | ✅ |
| IMG003_spinal_cord_section.PNG | skeletal muscle fiber | spinal cord | ❌ |
| IMG004_nervous_system_overview.jpg | nervous system overview | nervous system overview | ✅ |
| IMG005_brachial_plexus.png | brachial plexus | brachial plexus | ✅ |
| IMG006_skeletal_muscle_fiber.png | brachial plexus | skeletal muscle fiber | ❌ |

- **Score: 4/6 = 66.7%** (target: ≥ 80%)
- **Failures:** Both failures involve histological/cross-sectional images (spinal cord cross-section, skeletal muscle fiber microscopy) — the VLM (llama-4-scout) lacks fine-grained anatomical grounding for these image styles
- **Successes:** Overview/diagram-style images (neuron, brain lobes, nervous system, brachial plexus) identified correctly

---

## 6. Sample Transcript Excerpts (for paper/demo)

### Text Path — eval_s2 (Student guesses correctly after 2nd clue)
```
STUDENT: What lobe of the cerebral cortex is primarily responsible for visual processing?
TUTOR [CLUE 1]: Great question! This lobe is located on the posterior aspect of the 
  cerebrum, near the midline, above the tentorium cerebelli. Can you name this lobe?
STUDENT: Is it the frontal lobe?
TUTOR [CLUE 2]: Not quite — that's a different lobe. This lobe processes and integrates 
  visual sensory information. Try again?
STUDENT: The occipital lobe?
TUTOR [REVEAL]: Yes, exactly — Occipital lobe! The occipital lobe is the region primarily 
  responsible for processing visual information and interpreting visual stimuli...
```

### Image Path — eval_img2 (Non-anatomy image rejection)
```
STUDENT: [uploads synthetic noise/flowchart image] What is this diagram showing?
TUTOR [OUT_OF_ANATOMY]: Hmm, that doesn't look like an anatomy diagram to me! I'm 
  designed to help with anatomy and neuroscience images — things like muscles, nerves, 
  brain diagrams, or skeletal structures. Try uploading an anatomy diagram and I'll be 
  happy to work through it with you.
```

---

## 7. Milestone History

### Milestone 1 (Submitted ~March 2026)
- Planned architecture, no working system yet
- KB had 70 sections, 168 chunks (early stage)
- No evaluation results — all targets stated as planned
- Feedback received:
  - ✅ Good: text + image data clear, architecture flow easy to follow, sample conversations appreciated
  - ❌ Improve: add more detail on component interaction and reasoning behind design choices

### Milestone 2 (Due April 17, 2026 — 11:59pm)
- Full working prototype
- KB expanded to 997 chunks, 28 chapters
- 8 evaluation transcripts generated and audited
- All 6 baseline metrics computed
- Report: ACL format, ≤5 pages excl. references

---

## 8. Known Weaknesses & Root Causes

| Weakness | Root Cause | Severity |
|---|---|---|
| Context Recall = 0.65 | Retriever uses pure semantic similarity; broad multi-chapter questions pull incomplete chunks | Medium |
| Context Precision = 0.78 | Some off-topic chunks retrieved when query is ambiguous | Low |
| VLM accuracy = 66.7% | llama-4-scout fails on cross-sectional/histological images; no fine-grained anatomical grounding | Medium |
| LLM classifier ambiguity | Biologically-proximate cells (astrocyte, glial cell) sometimes classified as PARTIAL instead of WRONG_NAMED | Low (mitigated by fast-path) |

---

## 9. Planned Improvements (Future Work / Milestone 3)

1. **Retrieval reranking** — add topic-label filtering + BM25 hybrid retrieval to improve context recall toward 0.80 target
2. **VLM improvement** — add GPT-4o as primary vision backend (not just fallback); enrich image metadata with more visual cues for histological images
3. **Classifier hardening** — expand `common_misidentifications` library in image metadata for more anatomy structures
4. **Domain transfer test** — replace KB with a different subject (e.g., pharmacology) while keeping architecture fixed — validate generalizability
5. **Human evaluation** — Follow-up Relevance rubric (1–5) from actual OT students; pedagogical quality scoring
6. **Persistent memory** — Currently session-only; add cross-session weak-topic tracking

---

## 10. Tech Stack Summary

| Component | Technology |
|---|---|
| LLM (tutor) | Groq · LLaMA 3.1 8B Instruct |
| LLM (fallback) | GPT-4o-mini |
| Vision (VLM) | Groq llama-4-scout-17b → LLaVA-NeXT (GPU) → GPT-4o |
| Dialogue FSM | LangGraph |
| Knowledge Base | ChromaDB + all-MiniLM-L6-v2 |
| Evaluation | RAGAS (qwen3-32b judge) + custom purity audit |
| UI | Gradio 5.x |
| Python | 3.9 (main venv) · 3.11 (RAGAS venv: .venv_ragas) |
| API | Groq (free tier — 500K TPD llama-3.1-8b, 100K TPD llama-3.3-70b) |

---

## 11. Environment Notes

- Main venv: `.venv/` (Python 3.9) — used for all tutoring, transcripts, VLM
- RAGAS venv: `.venv_ragas/` (Python 3.11) — used only for RAGAS evaluation (RAGAS 0.4.3 requires Python 3.10+)
- GROQ_API_KEY: set in environment before running any script
- llama-3.1-8b-instant: exhausted daily quota (500K) on April 17 during RAGAS runs — resets daily
- llama-3.3-70b-versatile: exhausted daily quota (100K) on April 17 — resets daily
- qwen/qwen3-32b: successfully used as RAGAS judge on April 17 — separate quota

---

## 12. Key Commands

```bash
# Generate text transcripts
GROQ_API_KEY=... .venv/bin/python3 generate_transcripts.py

# Generate image transcripts  
GROQ_API_KEY=... .venv/bin/python3 generate_image_transcripts.py

# Run purity audit
.venv/bin/python3 -c "from src.evaluation import audit_purity; audit_purity('Evaluation/transcripts', 'Evaluation')"

# Run RAGAS (requires .venv_ragas + Python 3.11)
GROQ_API_KEY=... .venv_ragas/bin/python3 -c "..."  # see evaluation.py run_ragas()

# Run VLM blind test
GROQ_API_KEY=... .venv/bin/python3 -c "from src.evaluation import run_vlm_blind_test; ..."

# Launch Gradio UI
GROQ_API_KEY=... .venv/bin/python3 main.py
```
