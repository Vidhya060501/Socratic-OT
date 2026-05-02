---
title: Socratic-OT AI Anatomy Tutor
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.1
app_file: app.py
pinned: true
license: mit
short_description: Socratic AI anatomy tutor grounded in A&P 2e
---

# 🩺 Socratic-OT: Multimodal AI Anatomy Tutor

**Team:** Vidhyadhari Bandaru · Richie Ilavarapu

A Socratic AI tutor for Occupational Therapy (OT) students, grounded in the full **OpenStax Anatomy & Physiology 2e** textbook (28 chapters, 997 chunks).

---

## What It Does

Socratic-OT guides students to answers through questions — it never just *gives* the answer.

- Ask any anatomy or neuroscience question
- Upload an anatomy diagram — the tutor asks you Socratically *before* naming the structure
- After each topic: choose to extend, move on, or finish with a **mastery quiz**
- Session memory tracks weak topics and revisits them proactively

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq · Llama 3.1 8B Instruct |
| Dialogue | LangGraph state machine |
| Knowledge Base | ChromaDB · all-MiniLM-L6-v2 |
| Vision | LLaVA-NeXT (GPU) / GPT-4o (fallback) |
| Evaluation | RAGAS (Faithfulness ≥0.90, Relevance ≥0.85) |
| UI | Gradio |

---

## Phase Flow

```
RAPPORT → HINT → CLUE → REVEAL → ASSESS → TRANSITION
                                               ↓
                                   Student chooses:
                                   ├─ extend topic  → HINT (new angle)
                                   ├─ new topic     → RAPPORT
                                   └─ done          → QUIZ (mastery summary + 3 Qs)
                                                        ↓
                                                      DONE (memory updated)
```

---

## Setup

### Required Secret
- `GROQ_API_KEY` — free at [console.groq.com](https://console.groq.com)

### Optional Secret
- `OPENAI_API_KEY` — enables GPT-4o vision fallback for anatomy diagrams

Set these in **Space Settings → Variables and secrets**.

---

## Data

Knowledge base: OpenStax A&P 2e, all 28 chapters, 997 text chunks (~341 words each), embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB.

The vector store is built automatically on first launch (~3 minutes). Subsequent launches load from cache instantly.

---

## Running Locally / on Colab

See `run_in_colab.ipynb` for step-by-step Colab instructions.

```bash
# Local
export GROQ_API_KEY=your_key
python main.py

# Force rebuild ChromaDB
python main.py --rebuild

# Run RAGAS evaluation
python main.py --eval
```

---

*Built for MSOT coursework · Spring 2025*

Mile stone 2: Instructuions

Milestone 2 will be due on April 17, 11:59pm.

For Milestone 2, you are required to submit a report in ACL format focusing on your baseline results and experiments. No PPT or code submission is needed.

Baseline results refer to the initial performance of your model before applying any major optimizations.

We would like to see how your initial working prototype is shaping up. It does not need to be perfect, but it should demonstrate the direction you are progressing in.

Your report should be written like a research paper. It does not need to be organized by phases (e.g., Phase 1, Phase 2). Instead, it should be a cohesive document that includes:

Problem Statement – Clearly define the problem you are addressing.

Data – Describe the dataset used. Show sample transcripts and excerpts of the dataset or scraped data.

Solution Architecture – Explain your approach and model setup. Ensure your architecture is comprehensive - demonstrate how the initial user query passes through the entire system to give you the deeded output. Doing this will also give you clarity on what happens to the input at each stage.

Experiments & Results – Present your baseline performance and experiments done with appropriate evaluation metrics.

Also, be sure to include diagrams and tables where appropriate, and avoid heavy verbiage in the report, as visuals often communicate ideas more effectively than dense text. The final report will be a comprehensive document, so it is important to structure it as a unified research paper rather than a phase-wise breakdown. We hope you incorporate the feedback from milestone 1 to better your projects.

Note: Please also keep in mind that the final report must be within 5 pages (excluding references). All phases will be compiled into the same document, so ensure that each phase is written concisely and thoughtfully. We noticed that some Milestone 1 reports already exceeded this overall limit-please be mindful of this going forward.

Additionally, for Milestone 1, some submissions used arbitrary paper formats despite clear instructions. For Milestone 2, please strictly follow the ACL format, as a significant portion of points will be deducted for any format violations.
