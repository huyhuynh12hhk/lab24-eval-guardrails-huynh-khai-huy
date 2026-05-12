"""Task B.1 + B.2 — Pairwise Judge Pipeline & Absolute Scoring

B.1: Runs LLM-as-Judge comparing Version A (full production RAG) vs
     Version B (single-context baseline RAG) on 30+ questions.
     Uses swap-and-average to mitigate position bias.

B.2: Runs absolute 4-dimension scoring (accuracy, relevance,
     conciseness, helpfulness) on the same 30 questions.

Output: phase-b/pairwise_results.csv
        phase-b/absolute_scores.csv

Run: python phase-b/run_judge.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import require_env, get_openai_client, check_qdrant, PHASE_A, PHASE_B
require_env("OPENAI_API_KEY")

import pandas as pd
from tqdm import tqdm

PHASE_B.mkdir(exist_ok=True)

PAIRWISE_CSV = PHASE_B / "pairwise_results.csv"
ABSOLUTE_CSV = PHASE_B / "absolute_scores.csv"

NUM_QUESTIONS = 30

JUDGE_PROMPT = """You are an impartial evaluator. Compare two answers to the same question.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Rate based on:
- Factual accuracy
- Relevance to the question
- Conciseness and clarity

Output JSON only (no markdown):
{{"winner": "A" or "B" or "tie", "reason": "one sentence explanation"}}"""

ABSOLUTE_PROMPT = """Score this answer on 4 dimensions (1–5 scale each):

1. Factual accuracy (1=many errors, 5=fully accurate)
2. Relevance (1=off-topic, 5=directly answers the question)
3. Conciseness (1=very verbose, 5=appropriately brief)
4. Helpfulness (1=confusing/useless, 5=clear and actionable)

Question: {question}
Answer: {answer}

Output JSON only (no markdown):
{{"accuracy": int, "relevance": int, "conciseness": int, "helpfulness": int}}"""


def parse_json(text: str) -> dict:
    text = text.strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def pairwise_judge_with_swap(client, question: str, ans_a: str, ans_b: str) -> dict:
    """Run judge twice (A vs B, then B vs A) and aggregate."""
    def _call(a, b):
        prompt = JUDGE_PROMPT.format(question=question, answer_a=a, answer_b=b)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        return parse_json(resp.choices[0].message.content)

    r1 = _call(ans_a, ans_b)
    r2 = _call(ans_b, ans_a)

    w1 = r1.get("winner", "tie")
    w2_raw = r2.get("winner", "tie")

    # Flip r2 winner because A and B were swapped
    if w2_raw == "A":
        w2 = "B"
    elif w2_raw == "B":
        w2 = "A"
    else:
        w2 = "tie"

    final = w1 if w1 == w2 else "tie"

    return {
        "run1_winner": w1,
        "run1_reason": r1.get("reason", ""),
        "run2_winner": w2,
        "run2_reason": r2.get("reason", ""),
        "winner_after_swap": final,
    }


def absolute_score(client, question: str, answer: str) -> dict:
    prompt = ABSOLUTE_PROMPT.format(question=question, answer=answer)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    )
    parsed = parse_json(resp.choices[0].message.content)
    dims = ["accuracy", "relevance", "conciseness", "helpfulness"]
    for d in dims:
        if d not in parsed:
            parsed[d] = 3
    parsed["overall"] = sum(parsed[d] for d in dims) / 4
    return parsed


def load_questions() -> list[str]:
    """Load questions from testset or use fallback list."""
    testset_csv = PHASE_A / "testset_v1.csv"
    if testset_csv.exists():
        df = pd.read_csv(testset_csv)
        qs = df["question"].dropna().tolist()
        print(f"[B.1] Loaded {len(qs)} questions from testset_v1.csv")
        return qs[:NUM_QUESTIONS]

    print("[B.1] testset_v1.csv not found, using fallback questions.")
    return [
        "Dữ liệu cá nhân nhạy cảm bao gồm những loại thông tin nào?",
        "Nghị định 13/2023 quy định về quyền của chủ thể dữ liệu như thế nào?",
        "Điều kiện để xử lý dữ liệu cá nhân là gì theo Nghị định 13?",
        "Trách nhiệm của bên kiểm soát dữ liệu cá nhân bao gồm những gì?",
        "Chuyển dữ liệu cá nhân ra nước ngoài cần đáp ứng điều kiện gì?",
        "Mức phạt vi phạm quy định về bảo vệ dữ liệu cá nhân là bao nhiêu?",
        "Thông báo vi phạm dữ liệu cá nhân phải được thực hiện trong bao lâu?",
        "BCTC công ty ABC năm 2023 có doanh thu bao nhiêu?",
        "Lợi nhuận sau thuế của công ty trong kỳ báo cáo là bao nhiêu?",
        "Các khoản mục chính trong bảng cân đối kế toán gồm những gì?",
    ] * 3


def main():
    if not check_qdrant():
        sys.exit(1)

    client = get_openai_client()
    questions = load_questions()

    from rag_bridge import query_rag, query_rag_v2

    pairwise_rows = []
    absolute_rows = []

    print(f"\n[B.1] Running pairwise judge + absolute scoring on {len(questions)} questions")
    print("      Version A = production RAG (top-3 reranked)")
    print("      Version B = single-context baseline\n")

    for i, q in enumerate(tqdm(questions, desc="Judging"), 1):
        try:
            ans_a, _ = query_rag(q)
            ans_b, _ = query_rag_v2(q)
        except Exception as e:
            print(f"\n  [SKIP] {q[:50]} — RAG error: {e}")
            continue

        try:
            pw = pairwise_judge_with_swap(client, q, ans_a, ans_b)
            pairwise_rows.append({
                "question": q,
                "answer_a": ans_a,
                "answer_b": ans_b,
                **pw,
            })
        except Exception as e:
            print(f"\n  [SKIP pairwise] {q[:50]} — {e}")

        try:
            abs_score = absolute_score(client, q, ans_a)
            absolute_rows.append({"question": q, "answer": ans_a, **abs_score})
        except Exception as e:
            print(f"\n  [SKIP absolute] {q[:50]} — {e}")

        time.sleep(0.3)  # Rate limit buffer

    pd.DataFrame(pairwise_rows).to_csv(PAIRWISE_CSV, index=False, encoding="utf-8")
    pd.DataFrame(absolute_rows).to_csv(ABSOLUTE_CSV, index=False, encoding="utf-8")

    print(f"\n[B.1] Pairwise results → {PAIRWISE_CSV}  ({len(pairwise_rows)} rows)")
    print(f"[B.2] Absolute scores  → {ABSOLUTE_CSV}  ({len(absolute_rows)} rows)")

    if pairwise_rows:
        df_pw = pd.DataFrame(pairwise_rows)
        print("\n── Pairwise Summary ────────────────")
        print(df_pw["winner_after_swap"].value_counts().to_string())
        print("────────────────────────────────────")

    if absolute_rows:
        df_abs = pd.DataFrame(absolute_rows)
        print("\n── Absolute Score Summary ──────────")
        for dim in ["accuracy", "relevance", "conciseness", "helpfulness", "overall"]:
            if dim in df_abs.columns:
                print(f"  {dim:<15} avg={df_abs[dim].mean():.2f}")
        print("────────────────────────────────────")

    print("\nNext: python phase-b/generate_to_label.py")


if __name__ == "__main__":
    main()
