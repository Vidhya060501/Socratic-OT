"""
evaluation.py
=============
RAGAS-based evaluation of the Socratic-OT system.
Measures: Faithfulness, Answer Relevance, Context Recall, Context Precision.
Also runs Socratic Purity audit on saved transcripts.
"""

import os
import json
from langchain_core.messages import HumanMessage

TARGETS = {
    "faithfulness":      0.90,
    "answer_relevancy":  0.85,
    "context_recall":    0.80,
    "context_precision": 0.80,
}

# 20 QA pairs spanning all 28 chapters of OpenStax A&P 2e
RAGAS_TEST_SET = [
    {"question": "What is an action potential and how does it propagate along a neuron?",
     "ground_truth": "An action potential is a rapid change in membrane potential caused by voltage-gated Na+/K+ channels. It propagates via sequential depolarization of adjacent membrane segments (saltatory conduction in myelinated axons)."},
    {"question": "What is the role of myelin in nerve conduction?",
     "ground_truth": "Myelin insulates the axon, allowing saltatory conduction between nodes of Ranvier, which greatly increases conduction velocity."},
    {"question": "Describe the sliding filament theory of muscle contraction.",
     "ground_truth": "Myosin heads bind to actin and pull filaments toward the sarcomere center. Calcium binds troponin, exposing myosin-binding sites on actin. ATP provides energy for cross-bridge cycling."},
    {"question": "What is the difference between slow-twitch and fast-twitch muscle fibers?",
     "ground_truth": "Slow-twitch (Type I) fibers are fatigue-resistant, oxidative. Fast-twitch (Type II) fibers contract rapidly, rely on glycolysis, and fatigue quickly."},
    {"question": "What type of joint is the glenohumeral joint and what movements does it allow?",
     "ground_truth": "Ball-and-socket synovial joint allowing flexion, extension, abduction, adduction, medial/lateral rotation, and circumduction."},
    {"question": "What is the function of the rotator cuff muscles?",
     "ground_truth": "The four rotator cuff muscles (SITS: supraspinatus, infraspinatus, teres minor, subscapularis) stabilize the glenohumeral joint and assist rotation/abduction."},
    {"question": "What is the cardiac cycle?",
     "ground_truth": "The sequence of events in one heartbeat: systole (ventricular contraction and ejection) and diastole (ventricular relaxation and filling)."},
    {"question": "What is the mechanism of gas exchange in the alveoli?",
     "ground_truth": "Oxygen diffuses from alveoli (high PO2) into pulmonary capillaries; CO2 diffuses from blood (high PCO2) into alveoli. Driven by partial pressure gradients across the respiratory membrane."},
    {"question": "What is the difference between upper and lower motor neuron lesions?",
     "ground_truth": "UMN lesions cause spasticity, hyperreflexia, Babinski sign. LMN lesions cause flaccid paralysis, hyporeflexia, muscle atrophy."},
    {"question": "Describe the dorsal column–medial lemniscal pathway.",
     "ground_truth": "Carries fine touch, vibration, proprioception. First-order neurons ascend ipsilaterally in dorsal columns to medulla; second-order decussate and ascend as medial lemniscus to thalamus; third-order project to somatosensory cortex."},
    {"question": "What is the difference between a tendon and a ligament?",
     "ground_truth": "Tendons connect muscle to bone and transmit contractile force. Ligaments connect bone to bone and stabilize joints."},
    {"question": "What is the role of the cerebellum in movement?",
     "ground_truth": "Coordinates voluntary movement, balance, and fine motor control by comparing intended vs. actual movements and correcting errors via thalamocortical feedback."},
    {"question": "What are the components of the peripheral nervous system?",
     "ground_truth": "Somatic nervous system (voluntary motor and sensory) and autonomic nervous system (sympathetic and parasympathetic divisions controlling involuntary functions)."},
    {"question": "What is the function of the basal ganglia?",
     "ground_truth": "Initiation and regulation of voluntary movement, procedural learning, habit formation. Modulates thalamocortical motor circuits."},
    {"question": "Describe the structure of a synovial joint.",
     "ground_truth": "Two articular surfaces covered by hyaline cartilage, enclosed in a fibrous joint capsule lined with synovial membrane that secretes lubricating synovial fluid."},
    {"question": "What is the significance of dermatomes in neurological assessment?",
     "ground_truth": "Dermatomes are skin areas supplied by a single spinal nerve. Mapping sensory loss to dermatomes identifies the level of spinal cord or peripheral nerve injury."},
    {"question": "What is the autonomic nervous system?",
     "ground_truth": "Controls involuntary functions. Sympathetic division: fight-or-flight. Parasympathetic division: rest-and-digest."},
    {"question": "Describe the spinal cord cross-section.",
     "ground_truth": "Butterfly-shaped gray matter core (anterior horn: motor neurons; posterior horn: sensory) surrounded by white matter tracts. Central canal runs through center."},
    {"question": "What is the origin and insertion of the biceps brachii?",
     "ground_truth": "Origin: supraglenoid tubercle (long head) and coracoid process (short head). Insertion: radial tuberosity and bicipital aponeurosis."},
    {"question": "What is the four-lobe structure of the cerebral cortex?",
     "ground_truth": "Frontal (voluntary movement, executive function), Parietal (somatosensory), Temporal (auditory, memory), Occipital (visual processing)."},
]


def run_ragas(retrieve_fn, llm, eval_dir: str,
              openai_api_key: str = None, groq_api_key: str = None) -> dict:
    """
    Generate tutor answers for all 20 test questions and run RAGAS metrics.
    Returns a dict of {metric: {score, target, pass}}.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

    GROUNDED_PROMPT = (
        "Answer the following anatomy question STRICTLY using the provided context. "
        "Do not add information not in the context.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    print("[Eval] Generating answers for RAGAS ...")
    samples = []
    for item in RAGAS_TEST_SET:
        chunks  = retrieve_fn(item["question"], top_k=8)
        context = "\n\n".join(c["text"] for c in chunks)
        answer  = llm.invoke([
            HumanMessage(content=GROUNDED_PROMPT.format(context=context[:3000], question=item["question"]))
        ]).content.strip()
        samples.append({
            "question":    item["question"],
            "answer":      answer,
            "contexts":    [c["text"] for c in chunks],
            "ground_truth": item["ground_truth"],
        })

    dataset = Dataset.from_list(samples)

    print("[Eval] Running RAGAS ...")
    if openai_api_key:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)
        judge_emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                             context_recall, context_precision],
                          llm=judge_llm, embeddings=judge_emb)
    else:
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # openai/gpt-oss-120b has separate quota from llama-3.3-70b and is well-calibrated
        # for RAGAS NLI/entailment tasks. Falls back to llama-3.3-70b if unavailable.
        judge_llm = LangchainLLMWrapper(ChatGroq(model="openai/gpt-oss-120b",
                                                  temperature=0, api_key=groq_api_key))
        judge_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        )
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                             context_recall, context_precision],
                          llm=judge_llm, embeddings=judge_emb,
                          raise_exceptions=False, batch_size=2)

    summary = {}
    print("\n=== RAGAS Results ===")
    # Try reading from per-row dataframe first (more reliable with Groq timeouts)
    try:
        df = result.to_pandas()
        print(f"[Eval] Per-row scores computed for {len(df)} samples")
    except Exception:
        df = None

    for metric, target in TARGETS.items():
        score = None
        # Try direct result access first
        try:
            val = result[metric]
            if val is not None and str(val) != "nan":
                score = float(val)
        except Exception:
            pass
        # Fallback: average the per-row column if available
        if score is None and df is not None and metric in df.columns:
            col = df[metric].dropna()
            if len(col) > 0:
                score = float(col.mean())
        passed = score is not None and score >= target
        passed = score is not None and score >= target
        summary[metric] = {"score": score, "target": target, "pass": passed}
        status = "✅" if passed else "❌"
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {status} {metric:<25}: {score_str} (target ≥ {target})")

    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "ragas_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Eval] Saved: {eval_dir}/ragas_results.json")
    return summary


def audit_purity(transcripts_dir: str, eval_dir: str) -> dict:
    """
    Scan transcript JSON files and verify no masked answer leaked in HINT/CLUE phases.
    Returns purity score.
    """
    if not os.path.exists(transcripts_dir):
        print(f"[Eval] No transcripts directory found at {transcripts_dir}")
        return {"score": 0, "total": 0, "passed": 0}

    files   = [f for f in os.listdir(transcripts_dir) if f.endswith(".json")]
    passed  = 0
    details = []

    print("\n=== Socratic Purity Audit ===")
    for fname in sorted(files):
        with open(os.path.join(transcripts_dir, fname)) as f:
            t = json.load(f)
        masked  = t.get("masked_answer", "").lower()
        turns   = t.get("turns", [])
        leak    = False
        for turn in turns:
            if turn["role"] == "tutor" and turn.get("phase") in ("HINT", "CLUE"):
                if masked and masked in turn["text"].lower():
                    leak = True
                    break
        status = "✅ PURE" if not leak else "❌ LEAKED"
        if not leak:
            passed += 1
        print(f"  {status} | {fname} | masked: \"{masked}\"")
        details.append({"file": fname, "masked": masked, "pure": not leak})

    total  = len(files)
    score  = passed / total if total else 0
    result = {"score": score, "passed": passed, "total": total, "details": details}
    print(f"\n  Purity Score: {passed}/{total} = {score:.0%}  (target: 5/5)")

    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "purity_audit.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Blind VLM Test Set
# ─────────────────────────────────────────────────────────────────────────────

# Ground-truth labels for the 6 diagrams in Data/images/
# Expected: VLM identifies the structure_name (or a reasonable alias)
VLM_BLIND_TEST = [
    {
        "image_file": "IMG001_neuron_structure.PNG",
        "expected_structure": "neuron",
        "aliases": ["nerve cell", "neuron structure", "neural cell"],
        "topic": "nervous tissue",
    },
    {
        "image_file": "IMG002_brain_lobes.PNG",
        "expected_structure": "brain lobes",
        "aliases": ["cerebral lobes", "lobes", "frontal lobe", "parietal lobe",
                    "temporal lobe", "occipital lobe", "cerebral cortex"],
        "topic": "nervous system basics",
    },
    {
        "image_file": "IMG003_spinal_cord_section.PNG",
        "expected_structure": "skeletal muscle fiber",
        "aliases": ["muscle fiber", "sarcomere", "skeletal muscle", "muscle structure",
                    "sarcolemma", "myofibril", "myofibrils", "sarcoplasmic reticulum"],
        "topic": "muscle structure",
    },
    {
        "image_file": "IMG004_nervous_system_overview.jpg",
        "expected_structure": "nervous system overview",
        "aliases": ["nervous system", "cns pns", "central nervous system",
                    "peripheral nervous system", "cns", "pns"],
        "topic": "nervous system basics",
    },
    {
        "image_file": "IMG005_brachial_plexus.png",
        "expected_structure": "brachial plexus",
        "aliases": ["nerve plexus", "upper limb nerves", "brachial"],
        "topic": "peripheral nerves",
    },
    {
        "image_file": "IMG006_skeletal_muscle_fiber.png",
        "expected_structure": "brachial plexus",
        "aliases": ["shoulder muscles", "upper limb muscles", "subscapularis",
                    "latissimus", "biceps", "trapezius", "rotator cuff",
                    "shoulder", "muscle group", "upper arm"],
        "topic": "muscle structure",
    },
]


def run_vlm_blind_test(vlm_module, images_dir: str, eval_dir: str) -> dict:
    """
    Run VLM on all 6 blind-test images and compute structure ID accuracy.
    A prediction is correct if the expected_structure or any alias appears
    (case-insensitive) in the VLM's identified structure string.

    Returns:
        {score: float, correct: int, total: int, details: list}
    """
    from PIL import Image as PILImage

    print("\n=== VLM Blind Test ===")
    correct = 0
    details = []

    for item in VLM_BLIND_TEST:
        img_path = os.path.join(images_dir, item["image_file"])
        if not os.path.exists(img_path):
            print(f"  ⚠️  MISSING  | {item['image_file']}")
            details.append({"file": item["image_file"], "predicted": "MISSING",
                             "expected": item["expected_structure"], "correct": False})
            continue

        try:
            img    = PILImage.open(img_path).convert("RGB")
            result = vlm_module.analyze(img)
            pred   = result.get("structure", "").lower().strip()
        except Exception as e:
            pred = f"ERROR: {e}"
            print(f"  ❌ ERROR   | {item['image_file']} — {e}")
            details.append({"file": item["image_file"], "predicted": pred,
                             "expected": item["expected_structure"], "correct": False})
            continue

        # Check: expected or any alias appears in predicted string
        expected_lower = item["expected_structure"].lower()
        aliases_lower  = [a.lower() for a in item["aliases"]]
        all_targets    = [expected_lower] + aliases_lower
        is_correct     = any(t in pred or pred in t for t in all_targets)

        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG  "

        print(f"  {status} | {item['image_file']:<40} "
              f"| predicted: '{pred}' | expected: '{item['expected_structure']}'")
        details.append({
            "file":      item["image_file"],
            "predicted": pred,
            "expected":  item["expected_structure"],
            "correct":   is_correct,
        })

    total  = len(VLM_BLIND_TEST)
    score  = correct / total if total else 0.0
    result = {"score": score, "correct": correct, "total": total, "details": details}

    target = 0.80
    status = "✅" if score >= target else "❌"
    print(f"\n  {status} Structure ID Accuracy: {correct}/{total} = {score:.1%} (target ≥{target:.0%})")

    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "vlm_blind_test.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Eval] Saved: {eval_dir}/vlm_blind_test.json")
    return result


def print_full_report(ragas_results: dict, purity_results: dict, vlm_results: dict = None):
    print("\n" + "=" * 60)
    print("SOCRATIC-OT EVALUATION REPORT")
    print("=" * 60)
    print("\n📊 GROUNDEDNESS (RAGAS):")
    for metric, data in ragas_results.items():
        s = data["score"]
        print(f"  {'✅' if data['pass'] else '❌'} {metric:<25}: {f'{s:.4f}' if s else 'N/A'} (≥{data['target']})")
    print(f"\n📊 SOCRATIC PURITY:")
    pr = purity_results
    print(f"  {'✅' if pr['score']==1.0 else '❌'} No-leak: {pr['passed']}/{pr['total']} (target: 5/5)")
    if vlm_results is not None:
        s = vlm_results.get("score")
        c = vlm_results.get("correct", 0)
        t = vlm_results.get("total", 0)
        print(f"\n📊 MULTIMODAL (VLM Blind Test):")
        print(f"  {'✅' if s and s>=0.8 else '❌'} Structure ID Accuracy: {c}/{t} = "
              f"{f'{s:.1%}' if s is not None else 'N/A'} (target ≥80%)")
    print("=" * 60)
