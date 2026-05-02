"""
knowledge_base.py
=================
Builds and manages the ChromaDB vector store from the OpenStax A&P 2e text chunks.
Auto-skips rebuilding if the database already exists on disk.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch

COLLECTION_NAME = "socratic_ot_textbook"
BATCH_SIZE = 64


def _is_low_quality(text: str) -> bool:
    """Filter out stub chunks (key-terms lists, review-question pages, etc.)."""
    import re
    if len(text.split()) < 40:
        return True
    boilerplate = [
        r"^(Key Terms|Chapter Review|Review Questions|Interactive Link)",
        r"(Key Terms\s*Chapter Review\s*Interactive)",
    ]
    for pat in boilerplate:
        if re.search(pat, text[:200], re.IGNORECASE):
            body = re.sub(r".*(By the end of this section.*?:\s*)", "", text, flags=re.DOTALL)
            if len(body.split()) < 50:
                return True
    return False


def build_knowledge_base(project_root: str, force_rebuild: bool = False) -> tuple:
    """
    Build (or load) the ChromaDB vector store.

    Args:
        project_root  : Path to the Socratic_OT project root directory
        force_rebuild : If True, delete and recreate the collection

    Returns:
        (collection, embedder, image_metadata, image_by_topic, image_by_structure)
    """
    chroma_dir   = os.path.join(project_root, "Data", "chroma_db")
    chunks_csv   = os.path.join(project_root, "Data", "text_chunks", "text_chunks_full.csv")
    img_meta_json = os.path.join(project_root, "Data", "image_metadata", "image_metadata.json")
    lookup_pkl   = os.path.join(chroma_dir, "image_lookup.pkl")

    os.makedirs(chroma_dir, exist_ok=True)

    # ── Load embedding model ─────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[KB] Loading all-MiniLM-L6-v2 on {device} ...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # ── Connect to ChromaDB ───────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=chroma_dir)
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing and not force_rebuild:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"[KB] Loaded existing ChromaDB collection: {collection.count()} chunks")
    else:
        # Build from scratch
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            print("[KB] Deleted old collection, rebuilding ...")

        print(f"[KB] Loading chunks from: {chunks_csv}")
        df = pd.read_csv(chunks_csv, dtype=str).fillna("")

        # Filter low-quality
        df["_lq"] = df["chunk_text"].apply(_is_low_quality)
        df = df[~df["_lq"]].drop(columns=["_lq"]).reset_index(drop=True)
        print(f"[KB] {len(df)} chunks after quality filter")

        # Embed
        print(f"[KB] Embedding {len(df)} chunks ...")
        texts = df["chunk_text"].tolist()
        all_embs = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
            batch = texts[i: i + BATCH_SIZE]
            emb = embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embs.append(emb)
        embeddings = np.vstack(all_embs)

        # Create collection
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Ingest in batches
        print("[KB] Ingesting into ChromaDB ...")
        for start in tqdm(range(0, len(df), 500), desc="Ingesting"):
            end = min(start + 500, len(df))
            batch_df = df.iloc[start:end]
            collection.add(
                ids=batch_df["chunk_id"].tolist(),
                embeddings=embeddings[start:end].tolist(),
                documents=batch_df["chunk_text"].tolist(),
                metadatas=[
                    {
                        "source_id": row["source_id"],
                        "chapter":   row["chapter"],
                        "section":   row["section"],
                        "topic":     row["topic"],
                        "keywords":  row["keywords"],
                    }
                    for _, row in batch_df.iterrows()
                ]
            )
        print(f"[KB] ✅ ChromaDB ready: {collection.count()} chunks")

    # ── Load image metadata ───────────────────────────────────────────────────
    print(f"[KB] Loading image metadata from: {img_meta_json}")
    with open(img_meta_json) as f:
        image_metadata = json.load(f)

    # Build lookup dicts
    from collections import defaultdict
    image_by_topic     = defaultdict(list)
    image_by_structure = {}

    for rec in image_metadata:
        topic     = rec.get("topic", "").lower()
        structure = rec.get("structure_name", "").lower()
        image_by_topic[topic].append(rec)
        image_by_structure[structure] = rec
        for alias in rec.get("aliases", []):
            image_by_structure[alias.lower()] = rec

    # Save lookup for other modules
    with open(lookup_pkl, "wb") as f:
        pickle.dump({"image_by_topic": dict(image_by_topic),
                     "image_by_structure": image_by_structure}, f)

    print(f"[KB] ✅ Image metadata: {len(image_metadata)} records indexed")
    return collection, embedder, image_metadata, dict(image_by_topic), image_by_structure


def get_retriever(collection, embedder, chunks_df=None):
    """
    Returns a hybrid retrieval function:
      1. Dense retrieval  — ChromaDB cosine similarity (all-MiniLM-L6-v2)
      2. Sparse retrieval — BM25 over raw chunk texts
      3. Fusion          — Reciprocal Rank Fusion (RRF, k=60)
      4. Reranking       — CrossEncoder (ms-marco-MiniLM-L-6-v2) on fused candidates
      5. Merge           — same-section dedup + text concat
    Return format is identical to the original retrieve() — all callers unaffected.
    Logs retrieval debug info at INFO level for the UI debug expander.
    """
    import re
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder

    # ── Build BM25 index over all chunks ─────────────────────────────────────
    # Load from CSV if not passed in (build_knowledge_base already has this data)
    if chunks_df is None:
        # Derive path from collection metadata or fall back to known location
        import os, inspect
        _src_dir  = os.path.dirname(os.path.abspath(inspect.getfile(lambda: None)))
        _root     = os.path.dirname(_src_dir)
        _csv_path = os.path.join(_root, "Data", "text_chunks", "text_chunks_full.csv")
        chunks_df = pd.read_csv(_csv_path, dtype=str).fillna("")
        chunks_df = chunks_df[~chunks_df["chunk_text"].apply(_is_low_quality)].reset_index(drop=True)

    # Tokenize: lowercase, split on non-alphanumeric (preserves anatomy terms)
    _tokenize = lambda t: re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()
    _corpus_tokens = [_tokenize(t) for t in chunks_df["chunk_text"].tolist()]
    _bm25 = BM25Okapi(_corpus_tokens)
    _chunk_ids   = chunks_df["chunk_id"].tolist()
    _chunk_texts = chunks_df["chunk_text"].tolist()
    _chunk_metas = chunks_df[["chapter", "section", "topic", "keywords"]].to_dict("records")

    print(f"[KB] BM25 index built: {len(_chunk_ids)} chunks")

    # ── Load CrossEncoder (once — expensive, ~11s) ───────────────────────────
    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    print("[KB] CrossEncoder loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── RRF fusion helper ─────────────────────────────────────────────────────
    def _rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
        """Reciprocal Rank Fusion over multiple ranked lists of chunk_ids."""
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, cid in enumerate(ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        return scores

    def retrieve(query: str, top_k: int = 5) -> list:
        _DENSE_POOL  = 20   # candidates from dense retrieval
        _SPARSE_POOL = 20   # candidates from BM25
        _RERANK_POOL = 30   # max candidates fed to CrossEncoder

        # ── Stage 1a: Dense retrieval (ChromaDB) ─────────────────────────────
        qemb = embedder.encode(query, normalize_embeddings=True).tolist()
        res  = collection.query(
            query_embeddings=[qemb],
            n_results=min(_DENSE_POOL, collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        dense_hits: dict[str, dict] = {}
        dense_ranked: list[str] = []
        for doc, meta, dist, cid in zip(
            res["documents"][0], res["metadatas"][0],
            res["distances"][0], res["ids"][0]
        ):
            dense_hits[cid] = {
                "chunk_id": cid,
                "dense_score": round(1 - dist, 4),
                "chapter":  meta.get("chapter", ""),
                "section":  meta.get("section", ""),
                "topic":    meta.get("topic", ""),
                "keywords": meta.get("keywords", ""),
                "text":     doc,
            }
            dense_ranked.append(cid)

        # ── Stage 1b: Sparse retrieval (BM25) ────────────────────────────────
        qtokens      = _tokenize(query)
        bm25_scores  = _bm25.get_scores(qtokens)
        top_bm25_idx = sorted(range(len(bm25_scores)),
                               key=lambda i: bm25_scores[i], reverse=True)[:_SPARSE_POOL]
        sparse_ranked: list[str] = []
        sparse_hits:   dict[str, dict] = {}
        for idx in top_bm25_idx:
            cid = _chunk_ids[idx]
            sparse_ranked.append(cid)
            if cid not in dense_hits:
                sparse_hits[cid] = {
                    "chunk_id": cid,
                    "dense_score": 0.0,
                    "chapter":  _chunk_metas[idx].get("chapter", ""),
                    "section":  _chunk_metas[idx].get("section", ""),
                    "topic":    _chunk_metas[idx].get("topic", ""),
                    "keywords": _chunk_metas[idx].get("keywords", ""),
                    "text":     _chunk_texts[idx],
                }

        # ── Stage 2: RRF fusion ───────────────────────────────────────────────
        rrf_scores = _rrf([dense_ranked, sparse_ranked])
        all_hits   = {**dense_hits, **sparse_hits}

        # Keep top-_RERANK_POOL by RRF score
        top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:_RERANK_POOL]
        candidates = [all_hits[cid] for cid, _ in top_rrf if cid in all_hits]

        # ── Stage 3: CrossEncoder reranking ──────────────────────────────────
        if len(candidates) > 1:
            pairs   = [(query, c["text"]) for c in candidates]
            ce_scores = _cross_encoder.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, ce_scores):
                c["ce_score"] = float(s)
            candidates.sort(key=lambda c: c["ce_score"], reverse=True)
        else:
            for c in candidates:
                c["ce_score"] = 0.0

        # ── Stage 4: Assign final score + log debug info ─────────────────────
        # Final score = CrossEncoder score (normalized to 0-1 via sigmoid for readability)
        import math
        def _sigmoid(x): return 1 / (1 + math.exp(-x))

        debug_rows = []
        for rank, c in enumerate(candidates[:_RERANK_POOL], 1):
            c["score"] = round(_sigmoid(c.get("ce_score", 0.0)), 4)
            rrf_s = rrf_scores.get(c["chunk_id"], 0.0)
            debug_rows.append({
                "rank":       rank,
                "chunk_id":   c["chunk_id"],
                "section":    c["section"][:50],
                "dense":      round(c["dense_score"], 3),
                "rrf":        round(rrf_s, 4),
                "cross_enc":  round(c.get("ce_score", 0.0), 2),
                "final":      c["score"],
            })

        # Store debug info for UI expander (module-level, overwritten each call)
        get_retriever._last_debug = {
            "query":  query,
            "rows":   debug_rows,
        }

        # ── Stage 5: Take top-k, merge same-section chunks ───────────────────
        top_candidates = candidates[:max(top_k, 5)]

        merged: dict[str, dict] = {}
        for h in top_candidates:
            key = h["section"]
            if key not in merged:
                merged[key] = h.copy()
            else:
                if h["score"] > merged[key]["score"]:
                    merged[key]["score"] = h["score"]
                if h["text"] not in merged[key]["text"]:
                    merged[key]["text"] += "\n\n" + h["text"]

        result = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:3]

        # Log to terminal
        print(f"[Retrieval] Query: '{query[:60]}' → "
              f"dense={len(dense_ranked)} sparse={len(sparse_ranked)} "
              f"rrf_pool={len(candidates)} → top3 sections: "
              f"{[r['section'][:30] for r in result]}")

        return result

    # Attach debug slot
    get_retriever._last_debug = {"query": "", "rows": []}
    return retrieve
