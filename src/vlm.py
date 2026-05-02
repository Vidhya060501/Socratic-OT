"""
vlm.py
======
Vision-Language Module for anatomy diagram tutoring.

Backend priority (tried in order):
  1. GPT-4o Vision      — if openai_api_key set and quota available
  2. LLaVA-NeXT local   — if CUDA GPU available (Colab T4)
  3. Groq vision        — llama-4-scout-17b (free, CPU-safe, uses existing GROQ_API_KEY)
  4. Metadata mock      — deterministic lookup for known baseline images (testing)

Every backend returns the same normalized dict:
  {structure, topic, confidence, description,
   backend_used, fallback_used, error_message}

The downstream tutoring controller consumes only structure/topic/description
and never needs to know which backend answered.
"""

import io
import os
import base64
import hashlib
import re
import torch
from typing import Optional
from PIL import Image
from langchain_core.messages import HumanMessage


# ─────────────────────────────────────────────────────────────────────────────
# Known-image mock registry (for baseline testing when all VLMs are unavailable)
# Keys are SHA-256 hex prefixes (first 16 chars) of the raw image bytes.
# Values are pre-filled normalized results.
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_REGISTRY: dict = {}   # populated lazily from Data/images/ on first use

_FILENAME_MOCK = {
    "IMG001_neuron_structure": {
        "structure": "neuron structure", "topic": "nervous tissue",
        "confidence": "High",
        "description": "Diagram of a neuron showing soma, dendrites, axon, and myelin sheath.",
    },
    "IMG002_brain_lobes": {
        "structure": "brain lobes", "topic": "nervous system basics",
        "confidence": "High",
        "description": "Lateral view of the cerebral cortex showing the four lobes.",
    },
    "IMG003_spinal_cord_section": {
        "structure": "skeletal muscle fiber", "topic": "muscle structure",
        "confidence": "High",
        "description": "Diagram of a skeletal muscle fiber showing sarcolemma, myofibrils, and T-tubules.",
    },
    "IMG004_nervous_system_overview": {
        "structure": "nervous system overview", "topic": "nervous system basics",
        "confidence": "High",
        "description": "Overview diagram of the central and peripheral nervous systems.",
    },
    "IMG005_brachial_plexus": {
        "structure": "brachial plexus", "topic": "peripheral nerves",
        "confidence": "High",
        "description": "Diagram of the brachial plexus nerve network of the upper limb.",
    },
    "IMG006_skeletal_muscle_fiber": {
        "structure": "shoulder muscles", "topic": "muscle structure",
        "confidence": "High",
        "description": "Diagram of shoulder muscles including subscapularis, latissimus dorsi, biceps, and trapezius.",
    },
}


def _img_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def _build_mock_registry(images_dir: Optional[str] = None):
    """Index known baseline images by SHA-256 prefix so we can match by content."""
    if _MOCK_REGISTRY:
        return  # already built

    # Try to find Data/images/ relative to this file
    if images_dir is None:
        here = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(os.path.dirname(here), "Data", "images")

    if not os.path.isdir(images_dir):
        return

    for fname in os.listdir(images_dir):
        stem = os.path.splitext(fname)[0]
        meta = _FILENAME_MOCK.get(stem)
        if meta is None:
            continue
        try:
            img  = Image.open(os.path.join(images_dir, fname)).convert("RGB")
            h    = _img_hash(img)
            _MOCK_REGISTRY[h] = meta
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fields(raw: str, fields: list) -> dict:
    result = {"raw": raw}
    for field in fields:
        match = re.search(rf"{field}:\s*(.+?)(?=\n[A-Z]|$)", raw, re.DOTALL)
        result[field.lower().replace(" ", "_")] = match.group(1).strip() if match else "unknown"
    return result


def _normalize(raw: dict, backend: str, fallback: bool,
               error_message: str = "") -> dict:
    """Produce a guaranteed-consistent output dict regardless of backend."""
    return {
        "structure":     raw.get("structure", "unknown structure"),
        "topic":         raw.get("topic", "anatomy"),
        "confidence":    raw.get("confidence", "Low"),
        "description":   raw.get("description", ""),
        "is_anatomy":    raw.get("is_anatomy", ""),   # "yes" / "no" / "" (missing)
        "backend_used":  backend,
        "fallback_used": fallback,
        "error_message": error_message,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1: GPT-4o Vision
# ─────────────────────────────────────────────────────────────────────────────

_QUOTA_ERRORS = ("insufficient_quota", "rate_limit_exceeded",
                 "billing_hard_limit", "429")

def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in _QUOTA_ERRORS) or getattr(exc, "status_code", 0) == 429


def _identify_gpt4o(image: Image.Image, openai_api_key: str) -> tuple[dict, str]:
    """
    Returns (raw_result, error_message).
    error_message is "" on success.
    Raises nothing — all errors are caught and returned as error_message.
    """
    try:
        import openai
        client = openai.OpenAI(api_key=openai_api_key)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                    {"type": "text",
                     "text": (
                         "Look at this image carefully.\n"
                         "Reply in EXACTLY this format (no extra text):\n"
                         "IS_ANATOMY: [yes / no — is this an anatomy or neuroscience diagram?]\n"
                         "STRUCTURE: [anatomical structure name, or 'not anatomy' if IS_ANATOMY is no]\n"
                         "TOPIC: [muscle / nerve / brain / skeleton / organ / nervous tissue / not anatomy]\n"
                         "CONFIDENCE: [High / Medium / Low]\n"
                         "DESCRIPTION: [one sentence describing what is shown]"
                     )}
                ]
            }],
            max_tokens=200,
        )
        raw = _parse_fields(resp.choices[0].message.content,
                            ["IS_ANATOMY", "STRUCTURE", "TOPIC", "CONFIDENCE", "DESCRIPTION"])
        return raw, ""
    except Exception as e:
        return {}, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2: LLaVA-NeXT local (GPU only)
# ─────────────────────────────────────────────────────────────────────────────

_llava_processor = None
_llava_model     = None


def _load_llava() -> bool:
    global _llava_processor, _llava_model
    if _llava_model is not None:
        return True
    if not torch.cuda.is_available():
        print("[VLM] No GPU — LLaVA-NeXT local not loaded.")
        return False
    try:
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            BitsAndBytesConfig,
        )
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        print(f"[VLM] Loading LLaVA-NeXT ({model_id}) with 4-bit quantization ...")
        _llava_processor = LlavaNextProcessor.from_pretrained(model_id)
        _llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto"
        )
        print("[VLM] ✅ LLaVA-NeXT loaded")
        return True
    except Exception as e:
        print(f"[VLM] LLaVA-NeXT load failed: {e}")
        return False


def _run_llava(image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}]
    }]
    formatted = _llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs    = _llava_processor(images=image, text=formatted, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = _llava_model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None
        )
    full = _llava_processor.decode(out[0], skip_special_tokens=True)
    marker = "[/INST]"
    return full.split(marker)[-1].strip() if marker in full else full.strip()


def _identify_llava_local(image: Image.Image) -> tuple[dict, str]:
    try:
        prompt = (
            "Look at this image carefully.\n"
            "Reply in EXACTLY this format (no extra text):\n"
            "IS_ANATOMY: [yes / no — is this an anatomy or neuroscience diagram?]\n"
            "STRUCTURE: [anatomical structure name, or 'not anatomy' if IS_ANATOMY is no]\n"
            "TOPIC: [muscle / nerve / brain / skeleton / organ / nervous tissue / not anatomy]\n"
            "CONFIDENCE: [High / Medium / Low]\n"
            "DESCRIPTION: [one sentence describing what is shown]"
        )
        raw_text = _run_llava(image, prompt)
        raw = _parse_fields(raw_text, ["IS_ANATOMY", "STRUCTURE", "TOPIC", "CONFIDENCE", "DESCRIPTION"])
        return raw, ""
    except Exception as e:
        return {}, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Backend 3: Groq vision (llama-4-scout — free, CPU-safe)
# ─────────────────────────────────────────────────────────────────────────────

# Primary vision model. Fallback candidates tried in order if primary fails
# with a "model not found" / "deprecated" error.
_GROQ_VISION_PRIMARY   = "meta-llama/llama-4-scout-17b-16e-instruct"
_GROQ_VISION_FALLBACKS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # next-gen scout variant
    "llava-v1.5-7b-4096-preview",                      # older Groq LLaVA preview
]

_groq_vision_model_verified: Optional[str] = None   # cached after first successful call


def _get_groq_vision_model(groq_api_key: str) -> str:
    """
    Return the best available Groq vision model ID.

    On the first call: verifies _GROQ_VISION_PRIMARY with a tiny text-only request
    (no image — just checking model availability). If it fails with a not-found /
    deprecated error, walks through _GROQ_VISION_FALLBACKS. The result is cached in
    _groq_vision_model_verified so subsequent calls never repeat the check.

    This prevents silent fallback to the mock backend when Groq renames a model.
    """
    global _groq_vision_model_verified
    if _groq_vision_model_verified is not None:
        return _groq_vision_model_verified

    _NOT_FOUND_HINTS = ("not found", "deprecated", "does not exist",
                        "model_not_found", "invalid model")

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        for candidate in [_GROQ_VISION_PRIMARY] + _GROQ_VISION_FALLBACKS:
            try:
                # Minimal text-only probe — cheapest possible API call.
                client.chat.completions.create(
                    model=candidate,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
                _groq_vision_model_verified = candidate
                if candidate != _GROQ_VISION_PRIMARY:
                    print(f"[VLM] ⚠️  Primary Groq vision model unavailable; "
                          f"using fallback: {candidate}")
                else:
                    print(f"[VLM] ✅ Groq vision model verified: {candidate}")
                return candidate
            except Exception as e:
                msg = str(e).lower()
                if any(h in msg for h in _NOT_FOUND_HINTS):
                    print(f"[VLM] ⚠️  Groq model '{candidate}' not available ({e}); "
                          f"trying next fallback ...")
                    continue
                # Non-model-availability error (auth, rate limit) — stop probing,
                # use primary anyway (actual vision call will surface the real error).
                print(f"[VLM] ⚠️  Groq model probe failed with non-availability "
                      f"error: {e}. Will attempt with primary model.")
                _groq_vision_model_verified = _GROQ_VISION_PRIMARY
                return _GROQ_VISION_PRIMARY
    except Exception as e:
        print(f"[VLM] ⚠️  Groq client init failed during model probe: {e}")

    # All candidates exhausted or groq import failed
    print(f"[VLM] ❌ No working Groq vision model found — will attempt primary anyway.")
    _groq_vision_model_verified = _GROQ_VISION_PRIMARY
    return _GROQ_VISION_PRIMARY


def _identify_groq_vision(image: Image.Image, groq_api_key: str) -> tuple[dict, str]:
    """
    Uses the best available Groq vision model (verified at first call).
    Requires only the GROQ_API_KEY already used for text — no extra dependency.
    """
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        model  = _get_groq_vision_model(groq_api_key)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text",
                     "text": (
                         "Look at this image carefully.\n"
                         "Reply in EXACTLY this format (no extra text):\n"
                         "IS_ANATOMY: [yes / no — is this an anatomy or neuroscience diagram?]\n"
                         "STRUCTURE: [anatomical structure name, or 'not anatomy' if IS_ANATOMY is no]\n"
                         "TOPIC: [muscle / nerve / brain / skeleton / organ / nervous tissue / not anatomy]\n"
                         "CONFIDENCE: [High / Medium / Low]\n"
                         "DESCRIPTION: [one sentence describing what is shown]"
                     )}
                ],
            }],
            max_tokens=200,
        )
        raw = _parse_fields(resp.choices[0].message.content,
                            ["IS_ANATOMY", "STRUCTURE", "TOPIC", "CONFIDENCE", "DESCRIPTION"])
        return raw, ""
    except Exception as e:
        return {}, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Backend 4: Metadata mock (known baseline images — testing only)
# ─────────────────────────────────────────────────────────────────────────────

def _identify_mock(image: Image.Image) -> tuple[dict, str]:
    """Match image by SHA-256 content hash against known baseline registry."""
    _build_mock_registry()
    h = _img_hash(image)
    meta = _MOCK_REGISTRY.get(h)
    if meta:
        return dict(meta), ""
    return {}, "Image not found in mock registry"


# ─────────────────────────────────────────────────────────────────────────────
# VLMModule — public API
# ─────────────────────────────────────────────────────────────────────────────

class VLMModule:
    """
    Unified Vision-Language Module.
    Call analyze(image) to get a normalized identification + Socratic question.

    Backend priority:
      1. GPT-4o  (if openai_api_key set)
      2. LLaVA-NeXT local  (if CUDA GPU)
      3. Groq vision  (if groq_api_key set — always available on HF Spaces)
      4. Metadata mock  (known baseline images)
      5. Graceful degradation  (ask student to describe)
    """

    def __init__(self, retrieve_fn, llm,
                 openai_api_key:   str  = None,
                 groq_api_key:     str  = None,
                 img_by_structure: dict = None):
        self.retrieve          = retrieve_fn
        self.llm               = llm
        self.openai_api_key    = openai_api_key
        self.groq_api_key      = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        # img_by_structure: canonical lookup from image_metadata.json
        # keys = structure_name.lower() and alias.lower()
        # values = full metadata record {structure_name, topic, related_text_topic, …}
        self._img_by_structure = img_by_structure or {}
        self._llava_ready      = _load_llava()
        _build_mock_registry()   # pre-index known images at startup

    # ── Out-of-anatomy guard ─────────────────────────────────────────────────

    # Known non-anatomy topic words that signal a clearly off-topic upload.
    # Checked against raw VLM topic field before running the full pipeline.
    _NON_ANATOMY_TOPICS = {
        "electronics", "technology", "device", "computer", "laptop", "phone",
        "food", "animal", "plant", "vehicle", "furniture", "clothing", "sport",
        "architecture", "landscape", "person", "face", "text", "document",
        "chart", "graph", "screenshot", "unknown",
    }

    # Known anatomy topic words — if the VLM topic contains any of these,
    # the image is presumed anatomical even if structure identification is vague.
    _ANATOMY_TOPICS = {
        "muscle", "nerve", "brain", "skeleton", "organ", "bone", "vessel",
        "tissue", "anatomy", "nervous", "spinal", "cardiac", "respiratory",
        "digestive", "lymphatic", "endocrine", "integumentary", "reproductive",
        "urinary", "joint", "ligament", "tendon", "cell", "histology",
        "neuroscience", "physiology", "medical", "clinical",
    }

    def _is_anatomy_image(self, raw_structure: str, raw_topic: str,
                          is_anatomy_field: str = "",
                          raw_description: str = "") -> bool:
        """
        Guard: return False if the uploaded image is clearly not an anatomy diagram.

        Primary signal — IS_ANATOMY field from VLM prompt (added to all backends):
          The VLM is explicitly asked "is this an anatomy diagram? yes/no" before
          being forced into anatomy topic categories. This bypasses the circular
          problem where the constrained TOPIC field always returns anatomy words
          even for non-anatomy images (architecture diagrams, AI flowcharts, etc.).

        Fallback — keyword check on structure/topic strings:
          Used when IS_ANATOMY field is missing or unparseable (e.g. mock backend).
          Checks known non-anatomy words against known anatomy words.
        """
        # ── Primary: use IS_ANATOMY field if present ──────────────────────────
        flag = is_anatomy_field.strip().lower()
        if flag in ("yes", "no"):
            if flag == "no":
                print(f"[VLM] IS_ANATOMY=no — image is not an anatomy diagram.")
                return False
            # IS_ANATOMY=yes — still verify via keyword check as a safety net.
            # The vision model sometimes says "yes" for non-anatomy images
            # (laptops, flowcharts, screenshots) when it can't identify the content.
            # A non-anatomy topic word in the structure/topic/description is a strong counter-signal.
            combined = (raw_structure + " " + raw_topic + " " + raw_description).lower()
            for word in self._NON_ANATOMY_TOPICS:
                if word in combined:
                    print(f"[VLM] IS_ANATOMY=yes but non-anatomy keyword '{word}' found "
                          f"in structure/topic — rejecting.")
                    return False
            return True

        # ── Fallback: keyword heuristic (mock backend / unparsed field) ───────
        combined = (raw_structure + " " + raw_topic + " " + raw_description).lower()

        # Hard reject: explicit non-anatomy topic word
        for word in self._NON_ANATOMY_TOPICS:
            if word in combined:
                # Override if a strong anatomy word is also present
                for aword in self._ANATOMY_TOPICS:
                    if aword in combined:
                        return True
                return False

        # Soft pass: known anatomy word found
        for aword in self._ANATOMY_TOPICS:
            if aword in combined:
                return True

        # No signal — assume anatomy (safer than false rejection)
        return True

    # ── identification with fallback chain ───────────────────────────────────

    def _identify(self, image: Image.Image) -> dict:
        """
        Try each backend in priority order.
        Returns a normalized dict with backend_used / fallback_used / error_message.
        """
        tried = []

        # ── Backend 1: GPT-4o ────────────────────────────────────────────────
        if self.openai_api_key:
            print("[VLM] Attempting GPT-4o vision ...")
            raw, err = _identify_gpt4o(image, self.openai_api_key)
            if not err:
                print("[VLM] ✅ GPT-4o succeeded")
                return _normalize(raw, backend="gpt-4o", fallback=bool(tried))
            tried.append("gpt-4o")
            if _is_quota_error(Exception(err)):
                print(f"[VLM] ⚠️  GPT-4o quota/rate-limit error — falling back. ({err[:80]})")
            else:
                print(f"[VLM] ⚠️  GPT-4o error — falling back. ({err[:80]})")

        # ── Backend 2: LLaVA-NeXT local ──────────────────────────────────────
        if self._llava_ready:
            print("[VLM] Attempting LLaVA-NeXT local ...")
            raw, err = _identify_llava_local(image)
            if not err:
                print("[VLM] ✅ LLaVA-NeXT succeeded")
                return _normalize(raw, backend="llava-local", fallback=bool(tried))
            tried.append("llava-local")
            print(f"[VLM] ⚠️  LLaVA-NeXT error — falling back. ({err[:80]})")

        # ── Backend 3: Groq vision ────────────────────────────────────────────
        if self.groq_api_key:
            print("[VLM] Attempting Groq vision (llama-4-scout) ...")
            raw, err = _identify_groq_vision(image, self.groq_api_key)
            if not err:
                print("[VLM] ✅ Groq vision succeeded")
                return _normalize(raw, backend="groq-vision", fallback=bool(tried))
            tried.append("groq-vision")
            print(f"[VLM] ⚠️  Groq vision error — falling back. ({err[:80]})")

        # ── Backend 4: Metadata mock ──────────────────────────────────────────
        print("[VLM] Attempting metadata mock (known baseline images) ...")
        raw, err = _identify_mock(image)
        if not err:
            print("[VLM] ✅ Metadata mock matched")
            return _normalize(raw, backend="mock", fallback=True,
                              error_message=f"VLM unavailable ({', '.join(tried)}); mock used")
        tried.append("mock")
        print(f"[VLM] ⚠️  Mock: {err}")

        # ── Last resort: graceful degradation ────────────────────────────────
        print(f"[VLM] ❌ All backends failed: {tried}. Returning degraded result.")
        return _normalize(
            {"structure": "the structure in your diagram",
             "topic": "anatomy",
             "confidence": "N/A",
             "description": "Could not identify structure automatically."},
            backend="none",
            fallback=True,
            error_message=f"All backends failed: {tried}",
        )

    # ── KB metadata matching ─────────────────────────────────────────────────────

    def _match_kb_metadata(self, raw_structure: str) -> dict:
        """
        Map the raw VLM structure label to a canonical record in image_metadata.json.

        This is the authoritative grounding step.  The metadata file contains the
        exact structure_name, topic, and related_text_topic that align with the
        ChromaDB chunk metadata["topic"] field.  Using them for retrieval gives a
        topic-grounded query instead of a noisy free-text guess.

        Matching strategy (three passes, most precise first):
          Pass 1 — exact key match in img_by_structure (covers structure_name + aliases)
          Pass 2 — any metadata key is a substring of raw_structure (or vice-versa)
          Pass 3 — token overlap ≥ 50% between raw_structure tokens and each key

        Returns a dict with keys:
          matched          (bool)
          kb_structure     (str) — canonical structure_name from metadata
          kb_topic         (str) — topic field from metadata
          kb_related_topic (str) — related_text_topic field (maps to ChromaDB topic)
          match_pass       (int) — which pass found the match (1/2/3), or 0 if none
        """
        NOMATCH = {"matched": False, "kb_structure": "", "kb_topic": "",
                   "kb_related_topic": "", "match_pass": 0}

        if not self._img_by_structure:
            return NOMATCH

        raw_lower = raw_structure.lower().strip()

        # Pass 1: exact key lookup
        if raw_lower in self._img_by_structure:
            rec = self._img_by_structure[raw_lower]
            return {
                "matched":          True,
                "kb_structure":     rec.get("structure_name", raw_structure),
                "kb_topic":         rec.get("topic", ""),
                "kb_related_topic": rec.get("related_text_topic", rec.get("topic", "")),
                "match_pass":       1,
            }

        # Pass 2: substring containment (handles "Cervical and Brachial Plexuses"
        # matching key "brachial plexus" because "brachial plexus" ⊂ raw_lower)
        best_key = None
        best_len = 0
        for key in self._img_by_structure:
            if key in raw_lower or raw_lower in key:
                if len(key) > best_len:
                    best_len = len(key)
                    best_key = key
        if best_key:
            rec = self._img_by_structure[best_key]
            return {
                "matched":          True,
                "kb_structure":     rec.get("structure_name", best_key),
                "kb_topic":         rec.get("topic", ""),
                "kb_related_topic": rec.get("related_text_topic", rec.get("topic", "")),
                "match_pass":       2,
            }

        # Pass 3: token overlap ≥ 50%
        raw_tokens = set(re.sub(r"[^a-z ]", "", raw_lower).split())
        raw_tokens -= {"the", "a", "an", "of", "and", "or", "in", "its"}
        best_key   = None
        best_score = 0.0
        for key in self._img_by_structure:
            key_tokens = set(re.sub(r"[^a-z ]", "", key).split())
            key_tokens -= {"the", "a", "an", "of", "and", "or", "in", "its"}
            if not key_tokens:
                continue
            overlap = len(raw_tokens & key_tokens) / len(key_tokens)
            if overlap >= 0.5 and overlap > best_score:
                best_score = overlap
                best_key   = key
        if best_key:
            rec = self._img_by_structure[best_key]
            return {
                "matched":          True,
                "kb_structure":     rec.get("structure_name", best_key),
                "kb_topic":         rec.get("topic", ""),
                "kb_related_topic": rec.get("related_text_topic", rec.get("topic", "")),
                "match_pass":       3,
            }

        # Pass 4: sub-structure uplift ────────────────────────────────────────
        # Handles the case where the VLM identified a sub-component of a KB
        # structure (e.g. "frontal lobe" when KB only has "brain lobes").
        #
        # Strategy: for each KB key, check whether any token in raw_lower
        # appears as a token in the KB key — a single shared content word is
        # enough to suggest the sub-structure belongs to that parent entry.
        # We rank candidates by number of shared tokens and take the best.
        #
        # This is intentionally looser than Pass 3 (which requires ≥50% overlap
        # of KEY tokens) — here we only need ONE shared token because
        # "frontal" ∩ "brain lobes" = 0 but anatomical domain knowledge says
        # "frontal lobe" IS a component of "brain lobes".
        #
        # To prevent false uplifts (e.g. "nerve" matching everything), we require
        # the shared token to be a non-trivial content word (len > 3) and not a
        # generic type word.
        _GENERIC = {"nerve", "nerves", "muscle", "muscles", "bone", "bones",
                    "lobe", "lobes", "cord", "cell", "cells", "fiber", "fibers",
                    "tract", "tracts", "structure", "system", "tissue", "region"}

        # Build a reverse lookup: anatomy domain words → KB keys that contain them.
        # We use the parent structure's topic/description text if available,
        # but for now we also check whether raw_lower tokens appear as substrings
        # within the KB key itself OR within known aliases.
        best_key   = None
        best_score = 0
        raw_tokens_content = {
            t for t in raw_tokens
            if len(t) > 3 and t not in _GENERIC
        }

        for key, rec in self._img_by_structure.items():
            # Collect all text associated with this KB entry for matching
            aliases    = [a.lower() for a in rec.get("aliases", [])]
            all_text   = " ".join([key] + aliases + [
                rec.get("topic", ""), rec.get("related_text_topic", ""),
                rec.get("body_region", ""), rec.get("function", "")
            ]).lower()
            all_tokens = set(re.sub(r"[^a-z ]", "", all_text).split())

            shared = raw_tokens_content & all_tokens
            if shared and len(shared) > best_score:
                best_score = len(shared)
                best_key   = key

        if best_key:
            rec = self._img_by_structure[best_key]
            print(f"[VLM] KB match pass 4 (sub-structure uplift): "
                  f"'{raw_lower}' → '{rec.get('structure_name', best_key)}'")
            return {
                "matched":          True,
                "kb_structure":     rec.get("structure_name", best_key),
                "kb_topic":         rec.get("topic", ""),
                "kb_related_topic": rec.get("related_text_topic", rec.get("topic", "")),
                "match_pass":       4,
            }

        return NOMATCH

    # ── Image target refinement ──────────────────────────────────────────────────

    def _refine_target(self, raw_structure: str, context: str) -> str:
        """
        Post-VLM target refinement: given whatever the VLM returned (which may be
        a compound label, an over-broad category, or a multi-structure description),
        extract the single most instructionally specific anatomical structure to use
        as the teaching target.

        General rules enforced by the prompt:
          1. Return exactly ONE named structure — no conjunctions, no "and", no "overview"
          2. Prefer the most clinically dominant / pedagogically primary component when
             the VLM returned a compound label (e.g. "Cervical and Brachial Plexuses"
             → "brachial plexus"; "gray and white matter" → "spinal cord gray matter")
          3. Prefer the peripheral / specific structure over the general / central one
             when both are present (plexus > spinal cord; peripheral nerve > CNS)
          4. Must be a real named anatomical structure that appears in textbooks
          5. Three words maximum; no description, no adjectives beyond anatomical qualifiers

        This is fully general — it normalises any VLM output into the smallest correct
        teaching unit, without any hardcoding for specific image content.
        """
        # If already a clean single-structure name (≤3 words, no "and"), skip LLM call
        words = raw_structure.split()
        if (len(words) <= 3
                and " and " not in raw_structure.lower()
                and "overview" not in raw_structure.lower()
                and "system" not in raw_structure.lower()):
            return raw_structure

        p = (
            "You are helping a Socratic anatomy tutor choose a single teaching target.\n\n"
            f"A vision model identified this anatomy diagram as: \"{raw_structure}\"\n\n"
            "TASK: Extract the single most instructionally specific, anatomically grounded "
            "structure name that is the best teaching target for this image.\n\n"
            "RULES (must all be followed):\n"
            "1. Return EXACTLY ONE structure name — no conjunctions (no 'and', no 'or'), "
            "   no 'overview', no 'system'.\n"
            "2. If the label is compound (e.g. 'Cervical and Brachial Plexuses'), choose the "
            "   single most clinically prominent and pedagogically primary component.\n"
            "3. Prefer the more peripheral / more specific structure over the general / central "
            "   one (e.g. 'brachial plexus' over 'spinal cord'; 'median nerve' over "
            "   'peripheral nervous system').\n"
            "4. The result must be a real named anatomical structure that appears in anatomy "
            "   textbooks — not a description, not a category, not a process.\n"
            "5. Maximum 3 words. No adjectives beyond anatomical qualifiers.\n\n"
            f"Textbook context snippet (for grounding): {context[:600]}\n\n"
            "Reply with ONLY the structure name — no explanation, no punctuation."
        )
        try:
            refined = self.llm.invoke([HumanMessage(content=p)]).content.strip()
            # Accept only if it looks like a short anatomical name (no newlines, ≤6 words)
            refined = refined.splitlines()[0].strip(" \"'.")
            if refined and len(refined.split()) <= 6:
                print(f"[VLM] Target refined: '{raw_structure}' → '{refined}'")
                return refined
        except Exception as e:
            print(f"[VLM] Target refinement failed ({e}); using raw VLM output.")
        return raw_structure

    # ── First-clue generation ─────────────────────────────────────────────────

    def _build_first_clue(self, teaching_target: str, target_type: str) -> str:
        """
        Generate the first Socratic orientation clue for an image question.

        Design policy (mirrors text-path _TURN1_DIMENSIONS constraint):
          • Reveal ONLY the structural class — is this CNS or PNS? nerve or muscle?
            single structure or network? That is the single allowed Turn-1 dimension.
          • Do NOT give: body region, innervation territory, functional deficit map,
            spinal level origins, subdivision details, or clinical signs.
          • Broad enough that 3-5 candidate answers remain plausible after reading it.
          • The body-region and functional clues come in _clue turns 2+ when the
            engine's attempt-aware dimension selector drives them.
          • No KB context is injected — factual context enables precision we are
            deliberately suppressing on Turn 1.

        Uses target_type (safe ontology label: nerve/muscle/bone/plexus/…) so the
        clue can say "which [type]" without naming the specific structure. General
        across all anatomy diagram types.
        """
        p = (
            "You are a Socratic OT tutor. A student just uploaded an anatomy diagram.\n"
            f"The diagram shows: {teaching_target}  "
            f"(safe type label you MAY use: '{target_type}').\n\n"
            "Write EXACTLY ONE sentence that describes ONLY the structural class of this "
            f"{target_type} — without naming it.\n\n"
            "The single sentence must answer ONLY: "
            "Is this part of the central or peripheral nervous system? "
            "Or is it a muscle / bone / organ / vessel? "
            "Is it a single discrete structure or a network/complex of structures? "
            "Pick whichever single fact is most useful WITHOUT revealing more.\n\n"
            "STRICT RULES:\n"
            f"❌ Do NOT write '{teaching_target}' or any synonym, abbreviation, or alias.\n"
            "❌ Do NOT mention the body region, limb, or area it supplies.\n"
            "❌ Do NOT mention spinal levels, nerve roots, or origin points.\n"
            "❌ Do NOT describe function, clinical deficits, or innervation territory.\n"
            "❌ Do NOT write more than ONE sentence before the question.\n"
            f"✅ You MAY use the safe type label '{target_type}'.\n"
            f"✅ End with a short question: 'Which {target_type} is this?'\n\n"
            "Example format (do not copy verbatim — generate your own):\n"
            "  'This is a peripheral [type] structure — not a single cord, but a "
            f"  branching network. Which {target_type} is this?'"
        )
        try:
            return self.llm.invoke([HumanMessage(content=p)]).content.strip()
        except Exception:
            return (
                f"This diagram shows a {target_type} structure. "
                f"Can you name which {target_type} it is?"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, image: Image.Image,
                student_text: str = "") -> dict:
        """
        Full pipeline:
          identify → KB-metadata match → topic-grounded retrieval
          → (LLM refinement if no metadata hit) → first orientation clue

        student_text: optional text the student typed alongside the image.
          When provided, it is appended to the retrieval query so ChromaDB
          returns chunks relevant to both the image structure AND the student's
          specific question.  It does not affect VLM identification or KB
          metadata matching — only the retrieval step.

        The KB metadata match is the key grounding step.  It maps the raw VLM
        label to the canonical structure_name / related_text_topic in
        image_metadata.json, which in turn aligns exactly with ChromaDB's
        metadata["topic"] field.  This guarantees the retrieval query is
        grounded to the real KB topic — not a noisy freetext VLM guess.

        Returns dict with keys:
          structure, topic, confidence, description,
          context, socratic_question,
          kb_matched, kb_match_pass,
          backend_used, fallback_used, error_message
        """
        print("[VLM] ─────────────────────────────────────────")
        print("[VLM] Image received — starting identification pipeline")
        if student_text:
            print(f"[VLM] Student text    : '{student_text[:80]}'  (used in retrieval)")

        # ── Step 1: Identify via VLM backend cascade ─────────────────────────
        result        = self._identify(image)
        raw_structure = result["structure"]
        raw_topic     = result["topic"]
        print(f"[VLM] RAW VLM output  : structure='{raw_structure}'  "
              f"topic='{raw_topic}'  backend='{result['backend_used']}'  "
              f"fallback={result['fallback_used']}")

        _DEGRADED = ("the structure in your diagram", "unknown structure", "unknown")
        is_degraded = raw_structure.lower().strip() in _DEGRADED

        # ── Out-of-anatomy guard ──────────────────────────────────────────────
        # Pass the IS_ANATOMY field from the VLM response as the primary signal.
        # This is set by the updated prompt on all backends. The keyword fallback
        # inside _is_anatomy_image handles mocks and unparsed responses.
        is_anatomy_field = result.get("is_anatomy", "")
        raw_description  = result.get("description", "")
        if not is_degraded and not self._is_anatomy_image(
                raw_structure, raw_topic, is_anatomy_field, raw_description):
            print(f"[VLM] ❌ Out-of-anatomy image detected "
                  f"(structure='{raw_structure}', topic='{raw_topic}') — rejecting.")
            return {
                "structure":         "",
                "topic":             "",
                "confidence":        "N/A",
                "description":       "",
                "context":           "",
                "socratic_question": (
                    "Hmm, that doesn't look like an anatomy diagram to me! "
                    "I'm designed to help with anatomy and neuroscience images — "
                    "things like muscles, nerves, brain diagrams, or skeletal structures. "
                    "Try uploading an anatomy diagram and I'll be happy to work through it with you."
                ),
                "kb_matched":        False,
                "kb_match_pass":     0,
                "backend_used":      result["backend_used"],
                "fallback_used":     True,
                "error_message":     "non-anatomy image rejected",
                "out_of_anatomy":    True,
            }

        # ── Low-confidence gate ───────────────────────────────────────────────
        # When the VLM reports Low confidence AND no KB metadata will ground it,
        # treat it the same as degraded: ask the student to describe what they see
        # before guessing the structure. This avoids giving a confident-sounding
        # clue about a structure the VLM only guessed at.
        # High/Medium confidence: proceed normally.
        # Low confidence + kb_match will still proceed (metadata overrides VLM).
        if result.get("confidence", "").lower() == "low" and not is_degraded:
            # Peek at KB metadata — if we can ground it, allow it through
            _kb_peek = self._match_kb_metadata(raw_structure)
            if not _kb_peek["matched"]:
                print("[VLM] ⚠️  Low confidence + no KB match — routing to describe-first.")
                is_degraded = True   # treat as degraded so Step 6 uses the describe prompt

        # ── Step 2: Ground to KB metadata ────────────────────────────────────
        # Try to map raw VLM label → canonical record in image_metadata.json.
        # If matched, kb_structure and kb_related_topic are authoritative.
        if not is_degraded:
            kb = self._match_kb_metadata(raw_structure)
        else:
            kb = {"matched": False, "kb_structure": "", "kb_topic": "",
                  "kb_related_topic": "", "match_pass": 0}

        if kb["matched"]:
            teaching_target  = kb["kb_structure"]
            retrieval_topic  = kb["kb_related_topic"] or kb["kb_topic"]
            print(f"[VLM] KB match        : pass={kb['match_pass']}  "
                  f"kb_structure='{teaching_target}'  "
                  f"kb_related_topic='{retrieval_topic}'")
        else:
            # No metadata hit — fall back to LLM-based refinement
            print(f"[VLM] KB match        : NONE — will use LLM refinement")
            teaching_target = raw_structure   # placeholder; refined below
            retrieval_topic = raw_topic

        # ── Step 3: Topic-grounded retrieval ─────────────────────────────────
        # Build query using the verified KB topic so ChromaDB returns chunks
        # whose metadata["topic"] matches what we actually want to teach.
        # If the student typed a question alongside the image, append it to
        # the query so retrieval is also sensitive to what they are asking about
        # (e.g. "what is the function" pulls more functional/clinical chunks).
        retrieval_query = f"{teaching_target} {retrieval_topic} anatomy"
        if student_text:
            retrieval_query = f"{retrieval_query} {student_text}"
        chunks = self.retrieve(retrieval_query)

        # Re-score: boost chunks whose text contains the teaching target
        # and/or whose topic metadata exactly matches retrieval_topic.
        target_lower = teaching_target.lower()
        scored = []
        for c in chunks:
            score = 0
            if target_lower in c["text"].lower():
                score += 2
            if c.get("topic", "").lower() == retrieval_topic.lower():
                score += 1
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        context = "\n\n".join(c["text"] for _, c in scored)

        print(f"[VLM] Retrieval query : '{retrieval_query}'")
        print(f"[VLM] Chunks returned : {len(chunks)}  "
              f"(topics: {[c.get('topic','?') for c in chunks]})")

        # ── Step 4: LLM refinement (only if KB metadata did NOT match) ────────
        if not kb["matched"] and not is_degraded:
            teaching_target = self._refine_target(raw_structure, context)
            print(f"[VLM] LLM refinement  : '{raw_structure}' → '{teaching_target}'")
        else:
            print(f"[VLM] Teaching target : '{teaching_target}'  "
                  f"(source={'kb_metadata' if kb['matched'] else 'degraded'})")

        # ── Step 5: Infer target_type for the clue (fast lexical, no LLM needed) ──
        # We derive the safe type label from the teaching_target string itself.
        # This mirrors the fast-path in TutoringEngine._infer_target_type.
        # The label is used in the first-clue prompt so the tutor can say
        # "which plexus" instead of the over-suppressed "which structure".
        _TYPE_WORDS = (
            "nerve", "muscle", "vessel", "artery", "vein", "bone",
            "tract", "lobe", "joint", "tendon", "ligament",
            "ganglion", "plexus", "nucleus", "process", "fiber",
        )
        t_lower = teaching_target.lower()
        vlm_target_type = "structure"
        for tw in _TYPE_WORDS:
            if tw in t_lower:
                vlm_target_type = tw
                break

        # ── Step 6: Generate first orientation clue ───────────────────────────
        # Clue uses ONLY the structural-class dimension (one sentence + question).
        # No KB context is passed — that suppresses precision on Turn 1.
        if is_degraded or teaching_target.lower().strip() in _DEGRADED:
            question = (
                "I can see you've uploaded a diagram. "
                "Before I identify the structure, can you tell me what region of the body "
                "this shows — is it a muscle, nerve, or bone structure?"
            )
        else:
            question = self._build_first_clue(teaching_target, vlm_target_type)

        print(f"[VLM] Target type     : '{vlm_target_type}'")
        print(f"[VLM] Final target    : '{teaching_target}'")
        print(f"[VLM] First clue      : {question[:120]}...")
        print("[VLM] ─────────────────────────────────────────")

        return {
            "structure":         teaching_target,
            "topic":             retrieval_topic,
            "confidence":        result["confidence"],
            "description":       result["description"],
            "context":           context,
            "socratic_question": question,
            "kb_matched":        kb["matched"],
            "kb_match_pass":     kb["match_pass"],
            "backend_used":      result["backend_used"],
            "fallback_used":     result["fallback_used"],
            "error_message":     result["error_message"],
        }

    def explain(self, structure: str, context: str, student_answer: str) -> str:
        """After student answers, name the structure and give a grounded explanation."""
        prompt = (
            f"You are a Socratic OT tutor. The student answered: '{student_answer}'.\n"
            f"Now reveal and explain: Structure = {structure}\n"
            f"Write 3-4 sentences: confirm/correct the student, give a textbook explanation, "
            f"add an OT clinical connection.\n"
            f"Textbook context: {context[:1200]}"
        )
        return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
