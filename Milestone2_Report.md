# Socratic-OT: A Grounded Multimodal Socratic Tutor for Anatomy and Neuroscience Learning in Rehabilitation Science

**Vidhyadhari Bandaru** · **Richie M Ilavarapu**  
Department of Computer Science and Engineering, University at Buffalo  
{bandaru7, richiemo}@buffalo.edu

---

## Abstract

We present **Socratic-OT**, a grounded multimodal tutoring system for occupational therapy (OT) students studying anatomy and neuroscience. Unlike standard QA assistants that answer questions directly, Socratic-OT enforces a *tutor-not-teller* policy: it masks the target answer, retrieves supporting evidence from an approved textbook corpus, and guides students through Socratic clues before revealing the answer. The system supports both text and image inputs through a five-layer pipeline — routing, retrieval, policy control, generation, and assessment — orchestrated by a LangGraph finite-state machine. Baseline evaluation on 20 anatomy QA pairs and 8 scripted tutoring transcripts yields RAGAS Faithfulness of **0.91** and Answer Relevancy of **0.97**, confirming strong grounding. Socratic purity (no answer leakage in clue turns) achieves **100%** across all 8 transcripts. Multimodal structure identification reaches **66.7%** on six blind-test anatomy images, with failures concentrated on histological and cross-sectional views. We identify retrieval breadth as the primary remaining bottleneck and outline concrete improvements.

---

## 1. Introduction

Large language models have made conversational tutoring widely accessible, but general-purpose assistants present a core pedagogical problem: they deliver answers directly, weakening the recall and clinical reasoning skills that occupational therapy education demands. Anatomy and neuroscience are bottleneck courses for OT programs, with high content density and deep dependence on 3D spatial relationships — domains where guided discovery outperforms passive answer receipt.

Socratic-OT addresses this by enforcing a strict *tutor-not-teller* constraint at the system level. The answer is never revealed in hint or clue turns; only after clues are exhausted or the student guesses correctly does the system reveal and consolidate. The system retrieves exclusively from faculty-approved content (OpenStax A&P 2e), preventing hallucination. It also supports diagram-based tutoring: a student may upload an unlabeled anatomy image, and the system identifies the structure internally and guides the student Socratically before naming it.

In this milestone, we present the first working prototype and its baseline evaluation results, characterize where the system succeeds and falls short, and chart the path to final-milestone targets.

---

## 2. Related Work

Our design is informed by three lines of work. **Grounded tutoring** systems such as SocraticLM (Liu et al., 2024) frame tutoring as guided reasoning rather than answer delivery, motivating our Socratic phase structure. **Retrieval-augmented generation** (RAG) with RAGAS evaluation (Es et al., 2024) provides the framework for measuring faithfulness and relevance of retrieved-context answers. **Multimodal instruction tuning** work including LLaVA-NeXT (Liu et al., 2023a,b) informs our vision pipeline design, demonstrating that vision-language models can support image-grounded dialogue. We distinguish our work from Khanmigo (Khan Academy, 2025) and BoodleBox, which promote Socratic interaction but do not enforce retrieval-grounded, no-leak constraints at the system level.

---

## 3. Data

### 3.1 Text Knowledge Base

The text corpus is derived from OpenStax Anatomy & Physiology 2e, the standard approved reference for OT programs. Our processing pipeline: (1) extracts full text from all 28 chapters; (2) detects section boundaries; (3) cleans page markers and formatting artifacts; (4) assigns topic labels (e.g., *muscle*, *nervous system*); (5) chunks each section into paragraphs of ~341 words with 40-word overlap; (6) stores each chunk with source identifier, chapter, section title, topic label, and keyword summary.

This yields **997 retrieval-ready text chunks** embedded with `all-MiniLM-L6-v2` and indexed in ChromaDB. Table 1 summarizes the knowledge base assets.

| Asset | Count |
|---|---|
| Source chapters | 28 |
| Text chunks | 997 |
| Embedding model | all-MiniLM-L6-v2 |
| Labeled anatomy images | 6 |
| Image metadata records | 6 |

*Table 1: Knowledge base assets at Milestone 2.*

### 3.2 Image Knowledge Base

The visual knowledge base contains 6 labeled anatomy diagrams covering: neuron structure, brain lobes, spinal cord cross-section, nervous system overview, brachial plexus, and skeletal muscle fiber. Each image is paired with a compact metadata record containing: image ID, file name, topic, structure name, aliases, body region, function, and optional clinical cue. These metadata records serve as the bridge between VLM identification and KB-grounded tutoring.

**Sample metadata record (IMG001):**

| Field | Value |
|---|---|
| Image ID | IMG001 |
| File name | IMG001_neuron_structure.PNG |
| Topic | nervous tissue |
| Structure name | Neuron |
| Aliases | nerve cell; neuron structure |
| Function | basic neural signaling structure |

### 3.3 Evaluation Transcripts

Eight scripted tutoring transcripts were generated and saved to `Evaluation/transcripts/`, covering five text-path scenarios and three image-path scenarios. Text scenarios exercise: normal CLUE→REVEAL flow, early correct guess, repeated DONT_KNOW, out-of-KB topic rejection, and WRONG_NAMED comparison clue. Image scenarios exercise: in-KB anatomy image (neuron), non-anatomy image rejection (IS_ANATOMY guard), and unlabeled 3D render with fuzzy KB match.

**Sample transcript excerpt — eval_s2 (student guesses correctly after 2nd clue):**
```
STUDENT: What lobe of the cerebral cortex is primarily responsible for visual processing?
TUTOR  : This lobe is located on the posterior aspect of the cerebrum, near the midline,
         above the tentorium cerebelli. Can you name this lobe?
STUDENT: Is it the frontal lobe?
TUTOR  : Not quite — that's a different lobe. This lobe processes visual sensory
         information. Try again?
STUDENT: The occipital lobe?
TUTOR  : Yes, exactly — Occipital lobe! The occipital lobe is primarily responsible
         for processing visual information and interpreting visual stimuli...
```

**Sample transcript excerpt — eval_img2 (non-anatomy image rejected):**
```
STUDENT: [uploads synthetic flowchart image] What is this diagram showing?
TUTOR  : Hmm, that doesn't look like an anatomy diagram to me! I'm designed to help
         with anatomy and neuroscience images. Try uploading an anatomy diagram and
         I'll be happy to work through it with you.
```

---

## 4. Solution Architecture

The runtime system is organized into five layers that process every student input end-to-end (Figure 1).

```
Student Input (text or image)
        │
        ▼
┌─────────────────────────────┐
│  1. ROUTING LAYER           │  Classifies input: text / image / follow-up
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. RETRIEVAL LAYER         │  Semantic search (ChromaDB, top-5 chunks)
│                             │  + topic-label metadata reranking
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  3. TUTORING POLICY LAYER   │  LangGraph FSM:
│                             │  RAPPORT → CLUE(1-3) → REVEAL
│  • _build_forbidden_set()   │       → POST_REVEAL_WAIT → QUIZ → DONE
│  • _classify_attempt()      │
│  • _purity_guard()          │
│  • _is_in_scope()           │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  4. GENERATION LAYER        │  Text: Groq/LLaMA 3.1 8B → GPT-4o-mini
│                             │  Vision: llama-4-scout → LLaVA-NeXT → GPT-4o
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  5. SESSION MEMORY          │  Tracks weak topics, wrong guesses,
│     & ASSESSMENT            │  stuck count; mastery quiz post-topic
└─────────────────────────────┘
```

*Figure 1: Socratic-OT data flow from student input to grounded response.*

### 4.1 Routing and Retrieval

A student query enters the routing layer, which classifies it as text, image, or follow-up. For text inputs, the retriever computes an `all-MiniLM-L6-v2` embedding and retrieves the top-5 chunks from ChromaDB by cosine similarity. A metadata-aware reranking step then prioritizes chunks whose topic label matches the inferred query intent. Adjacent chunks from the same section may be merged before final prompting to improve coherence.

For image inputs, the VLM module runs a cascade: Groq `llama-4-scout-17b` (primary) → LLaVA-NeXT on GPU (if available) → GPT-4o (fallback). The VLM identifies the structure and returns an IS_ANATOMY flag, structure name, topic, and confidence. If IS_ANATOMY is false, a polite rejection is returned immediately. Otherwise, the identified structure is matched against KB metadata (exact name or alias), and the corresponding image metadata is used to enrich the tutoring context.

### 4.2 Tutoring Policy Controller

The tutoring policy layer is the core of Socratic-OT. It enforces four invariants:

1. **Answer masking** — `_build_forbidden_set()` constructs a set of forbidden terms from the masked answer plus LLM-generated aliases. The purity guard (`_purity_guard()`) scans every generated response and regenerates if any forbidden term appears.

2. **Attempt classification** — `_classify_attempt()` uses an LLM judge to classify the student's response as CORRECT, PARTIAL, WRONG_NAMED, DONT_KNOW, or OTHER. For image-path tutoring, a fast-path override uses the `common_misidentifications` list from image metadata to deterministically classify known wrong guesses, bypassing the LLM to avoid ambiguity with biologically-proximate structures.

3. **Clue dimension selection** — The tutor selects among clue dimensions (comparison, image_function, image_insertion_clinical) based on the attempt class. A WRONG_NAMED response triggers a comparison clue that names the student's wrong guess and contrasts it with the target without revealing the answer.

4. **Domain guard** — `_is_in_scope()` uses an LLM to determine if the question is about human anatomy or physiology. Out-of-domain questions (e.g., mRNA vaccines, programming) receive a polite rejection rather than a hallucinated answer.

### 4.3 Design Rationale

The separation of retrieval, policy, and generation layers is deliberate. The retrieval layer is domain-agnostic: replacing the ChromaDB collection with a different subject's content would port the tutor to a new domain with no architecture changes. The policy layer enforces pedagogical constraints independently of the generation layer — the LLM is used only for response generation, not for deciding *whether* to reveal the answer.

---

## 5. Experiments and Results

### 5.1 Groundedness — RAGAS Evaluation

We evaluate retrieval-augmented generation quality using RAGAS (Es et al., 2024) on 20 anatomy QA pairs spanning all 28 KB chapters. Answers are generated using Groq `qwen/qwen3-32b` conditioned strictly on retrieved context. The same model serves as RAGAS judge. We report four metrics with batch_size=4 to respect API rate limits.

| Metric | Score | Target | Status |
|---|---|---|---|
| Faithfulness | **0.9135** | ≥ 0.90 | ✅ Pass |
| Answer Relevancy | **0.9694** | ≥ 0.85 | ✅ Pass |
| Context Recall | **0.6471** | ≥ 0.80 | ❌ Below target |
| Context Precision | **0.7792** | ≥ 0.80 | ❌ Below target |

*Table 2: RAGAS baseline results. Judge: qwen/qwen3-32b. 20 QA pairs.*

**Faithfulness (0.91)** confirms that generated answers stay within the retrieved context — the system does not hallucinate facts. **Answer Relevancy (0.97)** confirms that responses are on-topic and address the question asked.

**Context Recall (0.65)** is the weakest metric. Root cause: the retriever uses pure semantic similarity; broad questions spanning multiple KB topics (e.g., "describe the dorsal column–medial lemniscal pathway") fail to surface all gold-standard content, as relevant chunks are distributed across multiple chapters. **Context Precision (0.78)** falls just below target — a small proportion of retrieved chunks are off-topic due to lexical overlap with unrelated sections.

### 5.2 Socratic Purity

We define *Socratic purity* as the absence of the masked answer in any CLUE-phase tutor turn. We scan all 8 evaluation transcripts and check whether the masked answer string (case-insensitive) appears in any tutor turn with phase=CLUE.

**Result: 8/8 = 100% purity** across all scenarios, including adversarial cases:

| Scenario | Masked Answer | Purity |
|---|---|---|
| Normal CLUE→REVEAL flow | neuron | ✅ |
| Early correct guess | occipital lobe | ✅ |
| Repeated DONT_KNOW (stuck) | brachial plexus | ✅ |
| Out-of-KB topic | — | ✅ |
| WRONG_NAMED → comparison clue | sarcomere | ✅ |
| In-KB anatomy image (neuron) | neuron | ✅ |
| Non-anatomy image (guard fires) | — | ✅ |
| Unlabeled 3D neuron render | neuron | ✅ |

*Table 3: Purity audit results across 8 evaluation transcripts.*

The purity guard regenerates responses when forbidden terms appear, and the student's own attempt tokens are excluded from the leak check to allow comparison clues (e.g., "you said *astrocyte*, but the target is different" — the student's wrong guess must be named to contrast it).

### 5.3 Multimodal VLM Evaluation

We run a blind test on all 6 KB anatomy images. Each image is passed to the VLM without labels; the system must identify the structure name. A prediction is correct if the expected structure name or any known alias appears in the predicted string.

| Image | Predicted | Expected | Correct |
|---|---|---|---|
| Neuron structure (PNG) | neuron | neuron | ✅ |
| Brain lobes (PNG) | brain lobes | brain lobes | ✅ |
| Spinal cord cross-section (PNG) | skeletal muscle fiber | spinal cord | ❌ |
| Nervous system overview (JPG) | nervous system overview | nervous system overview | ✅ |
| Brachial plexus (PNG) | brachial plexus | brachial plexus | ✅ |
| Skeletal muscle fiber (PNG) | brachial plexus | skeletal muscle fiber | ❌ |

*Table 4: VLM blind test results. Backend: Groq llama-4-scout-17b.*

**Result: 4/6 = 66.7%** (target ≥ 80%). Both failures involve histological and cross-sectional views. The spinal cord cross-section and skeletal muscle fiber microscopy images share visual features (elongated fiber structures, staining patterns) that confuse the vision model. Overview and diagram-style images — which have clearer spatial structure — are all identified correctly.

---

## 6. Discussion

The baseline results reveal a clear pattern: **generation quality is strong, retrieval breadth is the bottleneck**.

Faithfulness (0.91) and Answer Relevancy (0.97) indicate the generation layer behaves well — it stays grounded and on-topic. The 100% purity score confirms the masking policy works reliably, including in adversarial cases like biologically-proximate wrong guesses (e.g., "glial cell" or "astrocyte" when the answer is "neuron").

The weak link is Context Recall (0.65). The top-5 semantic retrieval strategy works well for focused single-topic questions but misses relevant chunks for multi-chapter queries. This is a known limitation of pure embedding-based retrieval. Hybrid retrieval (BM25 + semantic) with topic-label filtering would address this.

The VLM accuracy of 66.7% is acceptable for a baseline but needs improvement before clinical use. The failures are interpretable — cross-sectional views lack the spatial landmarks that diagram-style images provide — and point to a concrete fix: enriching image metadata with histological cues and routing low-confidence predictions to GPT-4o rather than accepting the llama-4-scout result.

---

## 7. Conclusion

Socratic-OT demonstrates that a pedagogically controlled, retrieval-grounded tutoring system is feasible with open-source components and free-tier APIs. The Socratic purity constraint holds reliably at 100%, and the generation layer produces faithful, relevant responses. The primary gaps — retrieval recall and VLM accuracy on histological images — are well-understood and addressable. Future work will pursue hybrid retrieval, GPT-4o vision integration, cross-session memory, and a domain-transfer test to validate generalizability across subjects.

---

## References

- Es, Shahul, et al. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *EACL System Demonstrations*, 150–158.
- Khan Academy. (2025). Khanmigo for learners. https://www.khanmigo.ai/learners
- LangChain. (2025). LangGraph overview. https://docs.langchain.com/langgraph
- Liu, Haotian, et al. (2023a). Visual instruction tuning. *arXiv:2304.08485*.
- Liu, Haotian, et al. (2023b). Improved baselines with visual instruction tuning. *arXiv:2310.03744*.
- Liu, Jiayu, et al. (2024). SocraticLM: Exploring socratic personalized learning with LLMs. *NeurIPS*.
- OpenAI. (2025a). GPT-4o. https://developers.openai.com/api/models/gpt-4o
- OpenAI. (2025b). GPT-4o mini. https://developers.openai.com/api/models/gpt-4o-mini
- OpenStax. (2022). Anatomy and Physiology 2e. https://openstax.org/details/books/anatomy-and-physiology-2e
- Chroma. (2025). Chroma documentation. https://docs.trychroma.com
- Sentence-Transformers. (2025). all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Groq. (2025). Groq documentation. https://console.groq.com/docs/overview
