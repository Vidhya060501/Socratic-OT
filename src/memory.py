"""
memory.py
=========
Session-based memory store.
Tracks weak topics, confused terms, wrong guesses, and mastery state
across a 15-minute tutoring session. Persisted to JSON on disk.
"""

import os
import json
from datetime import datetime
from collections import Counter


class SessionMemory:
    """
    Tracks one student's performance within a session and across sessions.

    Per-session state:
        weak_topics      : topics where student needed full REVEAL
        confused_terms   : {term: count} — specific terms missed repeatedly
        wrong_guesses    : {topic: [guesses]} — wrong attempts by topic
        mastered_topics  : topics answered correctly by Turn 2
        timeline         : full timestamped log

    Cross-session state loaded from disk (previous JSON files).
    """

    def __init__(self, student_id: str, save_dir: str):
        self.student_id    = student_id
        self.save_dir      = save_dir
        self.session_start = datetime.now().isoformat()
        self.session_id    = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.weak_topics    : list  = []
        self.confused_terms : dict  = {}
        self.wrong_guesses  : dict  = {}
        self.mastered_topics: list  = []
        self.timeline       : list  = []
        self.topic_scores   : dict  = {}

        os.makedirs(save_dir, exist_ok=True)

    # ── Record events ─────────────────────────────────────────────────────────

    def record_attempt(self, topic: str, guess: str, is_correct: bool, turn: int):
        if topic not in self.wrong_guesses:
            self.wrong_guesses[topic] = []
        if not is_correct:
            self.wrong_guesses[topic].append(guess)
            self.confused_terms[guess] = self.confused_terms.get(guess, 0) + 1
        self.timeline.append({
            "timestamp": datetime.now().isoformat(),
            "topic": topic, "guess": guess,
            "correct": is_correct, "turn": turn
        })

    def record_outcome(self, topic: str, turns_to_correct: int, needed_reveal: bool):
        self.topic_scores[topic] = {
            "turns_to_correct": turns_to_correct,
            "needed_reveal":    needed_reveal
        }
        if needed_reveal and topic not in self.weak_topics:
            self.weak_topics.append(topic)
        elif not needed_reveal and topic not in self.mastered_topics:
            self.mastered_topics.append(topic)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_weak_topics(self, n: int = 2) -> list:
        """Top-n topics to proactively revisit."""
        return self.weak_topics[-n:]

    def get_confused_terms(self, n: int = 3) -> list:
        sorted_t = sorted(self.confused_terms.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_t[:n]]

    def proactive_opener(self) -> str:
        """Return a reminder message if weak topics exist from this session."""
        weak = self.get_weak_topics(1)
        if not weak:
            return ""
        return (
            f"Before we continue, I noticed you had some difficulty with "
            f'"{weak[0]}" earlier — let\'s make sure that\'s solid. '
        )

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> str:
        data = {
            "student_id":    self.student_id,
            "session_id":    self.session_id,
            "session_start": self.session_start,
            "weak_topics":   self.weak_topics,
            "confused_terms":self.confused_terms,
            "wrong_guesses": self.wrong_guesses,
            "mastered_topics": self.mastered_topics,
            "topic_scores":  self.topic_scores,
            "timeline":      self.timeline,
        }
        path = os.path.join(self.save_dir, f"{self.session_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Memory] Session saved: {path}")
        return path

    @staticmethod
    def load_history(student_id: str, save_dir: str) -> list:
        sessions = []
        if not os.path.exists(save_dir):
            return sessions
        for fname in os.listdir(save_dir):
            if fname.startswith(student_id) and fname.endswith(".json"):
                with open(os.path.join(save_dir, fname)) as f:
                    sessions.append(json.load(f))
        return sorted(sessions, key=lambda s: s["session_start"])

    @staticmethod
    def get_dashboard(student_id: str, save_dir: str) -> dict:
        """Aggregate all past sessions into a weak-spot dashboard."""
        history = SessionMemory.load_history(student_id, save_dir)
        if not history:
            return {"student_id": student_id, "sessions": 0, "message": "No history yet."}

        all_weak      = []
        all_mastered  = []
        all_confused  = Counter()
        for s in history:
            all_weak.extend(s.get("weak_topics", []))
            all_mastered.extend(s.get("mastered_topics", []))
            for term, cnt in s.get("confused_terms", {}).items():
                all_confused[term] += cnt

        weak_freq     = Counter(all_weak)
        mastered_freq = Counter(all_mastered)
        priority = [
            t for t, c in weak_freq.most_common(10)
            if c > mastered_freq.get(t, 0)
        ]

        return {
            "student_id":       student_id,
            "sessions":         len(history),
            "last_session":     history[-1]["session_start"],
            "mastered_topics":  list(set(all_mastered)),
            "weak_topics":      list(set(all_weak)),
            "priority_review":  priority[:5],
            "top_confused":     all_confused.most_common(5),
        }

    def print_summary(self):
        print("─" * 50)
        print(f"SESSION SUMMARY — {self.student_id}")
        print(f"  Mastered : {self.mastered_topics}")
        print(f"  Weak     : {self.weak_topics}")
        print(f"  Confused : {list(self.confused_terms.items())[:5]}")
        print(f"  Turns    : {len(self.timeline)}")
        print("─" * 50)
