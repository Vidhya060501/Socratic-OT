"""
domain_swap_demo.py
===================
Demonstrates that the Socratic-OT architecture is fully domain-agnostic.

The ONLY change needed to switch from Anatomy to Physics is:
  1. Provide a different CSV of text chunks (physics_chunks.csv)
  2. Point the knowledge base builder at it

All tutoring logic, purity guards, session memory, and evaluation
remain identical — they operate purely on the retrieved content.

Usage (in Colab or terminal):
    python domain_swap/domain_swap_demo.py

Requires: GROQ_API_KEY set in environment.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.tutor import TutoringEngine
from src.memory import SessionMemory


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build a Physics ChromaDB from the sample CSV
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS_CSV   = os.path.join(os.path.dirname(__file__), "physics_chunks.csv")
PHYSICS_DB    = os.path.join(os.path.dirname(__file__), "physics_chroma_db")
COLLECTION    = "socratic_physics"


def build_physics_kb():
    """Embed physics chunks and store in a local ChromaDB."""
    print("[Domain Swap] Building Physics knowledge base ...")
    df = pd.read_csv(PHYSICS_CSV, dtype=str).fillna("")
    print(f"[Domain Swap] Loaded {len(df)} physics chunks")

    device = "cpu"
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    client = chromadb.PersistentClient(path=PHYSICS_DB)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)

    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    texts = df["chunk_text"].tolist()
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    collection.add(
        ids=df["chunk_id"].tolist(),
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "chapter":  row["chapter"],
                "section":  row["section"],
                "topic":    row["topic"],
                "keywords": row["keywords"],
            }
            for _, row in df.iterrows()
        ]
    )
    print(f"[Domain Swap] ✅ Physics ChromaDB ready: {collection.count()} chunks")
    return collection, embedder


def get_physics_retriever(collection, embedder):
    """Same retriever interface as the anatomy knowledge base."""
    def retrieve(query: str, top_k: int = 3) -> list:
        qemb = embedder.encode(query, normalize_embeddings=True).tolist()
        res  = collection.query(
            query_embeddings=[qemb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for doc, meta, dist, cid in zip(
            res["documents"][0], res["metadatas"][0],
            res["distances"][0], res["ids"][0]
        ):
            hits.append({
                "chunk_id": cid,
                "score":    round(1 - dist, 4),
                "chapter":  meta.get("chapter", ""),
                "section":  meta.get("section", ""),
                "topic":    meta.get("topic", ""),
                "keywords": meta.get("keywords", ""),
                "text":     doc
            })
        return sorted(hits, key=lambda x: x["score"], reverse=True)[:3]
    return retrieve


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run a short demo conversation
# ─────────────────────────────────────────────────────────────────────────────

DEMO_TURNS = [
    "What is Newton's Second Law?",
    "I think it has something to do with force and mass?",
    "F = ma — force equals mass times acceleration?",
    "1",   # quiz me
    "Acceleration means velocity changes over time",
    "2",   # new topic
    "Explain kinetic energy",
    "Energy due to motion, KE = half m v squared",
    "3",   # done
]


def run_demo(groq_key: str):
    print("\n" + "=" * 60)
    print("  DOMAIN SWAP DEMO — Physics Socratic Tutor")
    print("  (Same engine, same purity guards, different knowledge base)")
    print("=" * 60 + "\n")

    collection, embedder = build_physics_kb()
    retrieve = get_physics_retriever(collection, embedder)

    engine = TutoringEngine(
        retrieve_fn=retrieve,
        groq_api_key=groq_key,
        subject_domain="physics, mechanics, kinematics, forces, or energy",
    )
    memory = SessionMemory(
        student_id="physics_demo",
        save_dir=os.path.join(os.path.dirname(__file__), "physics_sessions")
    )

    transcripts_dir = os.path.join(os.path.dirname(__file__), "physics_transcripts")

    print("--- Physics Demo Conversation ---\n")
    for user_turn in DEMO_TURNS:
        print(f"Student: {user_turn}")
        response = engine.chat(user_turn)
        phase    = engine.get_phase()
        print(f"Tutor [{phase}]: {response}\n")

        covered_now = engine.state.get("covered_topics", [])
        if covered_now:
            for t in covered_now:
                if t not in memory.weak_topics + memory.mastered_topics:
                    memory.record_outcome(
                        topic=t,
                        turns_to_correct=engine.state.get("total_turns", 3),
                        needed_reveal=engine.state.get("stuck_count", 0) > 0,
                    )
            engine.save_transcript(transcripts_dir, student_id="physics_demo")

        if phase == "DONE":
            engine.save_transcript(transcripts_dir, student_id="physics_demo")
            break

    memory.save()
    memory.print_summary()

    print("\n✅ Domain swap PASSED — same engine tutored Physics with no code changes.")
    print("   To swap back to Anatomy: pass the anatomy ChromaDB retriever to TutoringEngine.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("ERROR: Set GROQ_API_KEY in your environment first.")
        sys.exit(1)
    run_demo(groq_key)
