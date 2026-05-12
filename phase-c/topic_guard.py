"""Task C.2 — Input Guardrail: Topic Scope Validator

Validates that queries are within the allowed topic scope using
embedding similarity (Option 1 — works without extra packages).

Allowed topics (matched to Day 18 corpus):
  - Vietnamese personal data protection law
  - Financial reports and accounting
  - Business compliance and regulations

Output: (printed results + stats)
        phase-c/topic_test_results.csv

Run: python phase-c/topic_guard.py
"""
import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import require_env, OPENAI_API_KEY

require_env("OPENAI_API_KEY")

import numpy as np
import pandas as pd
from langchain_openai import OpenAIEmbeddings

PHASE_C = Path(__file__).parent
OUT_CSV = PHASE_C / "topic_test_results.csv"

ALLOWED_TOPICS = [
    "Vietnamese personal data protection law and regulations",
    "Personal data privacy rights and obligations",
    "Data protection compliance and penalties",
    "Financial reports and accounting statements",
    "Business revenue profit and financial analysis",
    "Corporate financial data and balance sheets",
]

SIMILARITY_THRESHOLD = 0.773


class TopicGuard:
    def __init__(self, allowed_topics: list[str], threshold: float = SIMILARITY_THRESHOLD):
        print(f"[TopicGuard] Embedding {len(allowed_topics)} allowed topics…")
        self.embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.topics = allowed_topics
        self.threshold = threshold
        self.topic_vectors = [
            np.array(self.embedder.embed_query(t)) for t in allowed_topics
        ]
        print(f"[TopicGuard] Ready. Threshold = {threshold}")

    def check(self, text: str) -> tuple[bool, str]:
        if not text or not text.strip():
            return False, "Empty input"

        q_vec = np.array(self.embedder.embed_query(text))
        sims = [
            float(np.dot(q_vec, tv) / (np.linalg.norm(q_vec) * np.linalg.norm(tv) + 1e-8))
            for tv in self.topic_vectors
        ]
        max_sim = max(sims)
        best_topic = self.topics[sims.index(max_sim)]

        if max_sim >= self.threshold:
            return True, f"On topic: {best_topic} (sim={max_sim:.3f})"
        return False, (
            f"Off topic. Closest match: '{best_topic}' (sim={max_sim:.3f} < threshold {self.threshold}). "
            f"I can only answer questions about Vietnamese personal data protection law and financial reports."
        )

    async def check_async(self, text: str) -> tuple[bool, str]:
        return await asyncio.to_thread(self.check, text)


# ── Test cases ─────────────────────────────────────────────────────────────
ON_TOPIC = [
    "Dữ liệu cá nhân nhạy cảm theo Nghị định 13 bao gồm những gì?",
    "Quyền của chủ thể dữ liệu trong Nghị định 13/2023 là gì?",
    "Điều kiện để chuyển dữ liệu cá nhân ra nước ngoài?",
    "Mức phạt khi vi phạm quy định bảo vệ dữ liệu là bao nhiêu?",
    "Doanh thu của công ty ABC trong năm 2023 là bao nhiêu?",
    "Lợi nhuận sau thuế trong báo cáo tài chính là gì?",
    "Tài sản ngắn hạn trong bảng cân đối kế toán gồm những gì?",
    "Trách nhiệm của bên kiểm soát dữ liệu cá nhân?",
    "Thời hạn thông báo vi phạm dữ liệu cá nhân là bao lâu?",
    "Báo cáo tài chính hợp nhất năm 2023 của công ty?",
]

OFF_TOPIC = [
    "How do I make carbonara pasta?",
    "What is the best programming language for AI?",
    "Tell me a joke",
    "Who is the president of the United States?",
    "What is the weather like today?",
    "How to hack into a computer system?",
    "Write me a poem about flowers",
    "What is the latest iPhone model?",
    "Recommend a good movie to watch",
    "How do I lose weight quickly?",
]


def run_tests(guard: TopicGuard) -> list[dict]:
    rows = []
    correct = 0
    total = len(ON_TOPIC) + len(OFF_TOPIC)

    print(f"\n[C.2] Testing topic validator ({len(ON_TOPIC)} on-topic, {len(OFF_TOPIC)} off-topic)\n")

    for text, expected_on in [(t, True) for t in ON_TOPIC] + [(t, False) for t in OFF_TOPIC]:
        t0 = time.perf_counter()
        is_on, reason = guard.check(text)
        latency_ms = (time.perf_counter() - t0) * 1000

        match = is_on == expected_on
        if match:
            correct += 1
        status = "✓" if match else "✗"
        label = "ON " if expected_on else "OFF"
        print(f"  {status} [{label}] {text[:55]:<55} ({latency_ms:.0f}ms)")
        if not match:
            print(f"         → {reason}")

        rows.append({
            "text": text,
            "expected": "on-topic" if expected_on else "off-topic",
            "predicted": "on-topic" if is_on else "off-topic",
            "correct": match,
            "reason": reason,
            "latency_ms": latency_ms,
        })

    accuracy = correct / total * 100
    refuse_rate = sum(1 for r in rows if r["predicted"] == "off-topic") / total * 100

    print(f"\n[C.2] Accuracy: {correct}/{total} = {accuracy:.1f}%  (target ≥ 75%)")
    print(f"[C.2] Refuse rate: {refuse_rate:.1f}%")

    if accuracy >= 75:
        print("[C.2] ✓ Accuracy meets threshold")
    else:
        print("[C.2] ✗ Accuracy below 75% — try adjusting SIMILARITY_THRESHOLD")
        print(f"       Current threshold: {SIMILARITY_THRESHOLD}. Try 0.55 or 0.65.")

    return rows


def main():
    guard = TopicGuard(ALLOWED_TOPICS)
    rows = run_tests(guard)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n[C.2] Results → {OUT_CSV}")
    print("\nNext: python phase-c/adversarial_test.py")


_topic_guard_instance = None

def get_topic_guard() -> TopicGuard:
    global _topic_guard_instance
    if _topic_guard_instance is None:
        _topic_guard_instance = TopicGuard(ALLOWED_TOPICS)
    return _topic_guard_instance


if __name__ == "__main__":
    main()
