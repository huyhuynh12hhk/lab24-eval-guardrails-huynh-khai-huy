"""Task A.2 — Run RAGAS 4 Metrics

Runs the Day 18 RAG pipeline on every question in testset_v1.csv,
then scores the results with 4 RAGAS metrics (ragas 0.4.x API).

Output: phase-a/ragas_results.csv   (per-question scores)
        phase-a/ragas_summary.json  (aggregate scores)

Run: python phase-a/run_eval.py
     (takes ~10–20 min for 50 questions; saves progress incrementally)
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import require_env, get_ragas_llm, get_ragas_embeddings, check_qdrant, PHASE_A
require_env("OPENAI_API_KEY")

import pandas as pd
from datasets import Dataset
import warnings
from ragas import evaluate
# ragas 0.4.x: collections metrics require InstructorLLM (not LangchainLLMWrapper).
# The ragas.metrics path returns pre-instantiated objects that accept LangchainLLMWrapper
# via the top-level llm/embeddings args in evaluate(). Suppress the deprecation warning.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

TESTSET_CSV = PHASE_A / "testset_v1.csv"
RESULTS_CSV = PHASE_A / "ragas_results.csv"
SUMMARY_JSON = PHASE_A / "ragas_summary.json"
CHECKPOINT_CSV = PHASE_A / "ragas_checkpoint.csv"


def load_testset() -> pd.DataFrame:
    if not TESTSET_CSV.exists():
        print(f"[ERROR] {TESTSET_CSV} not found.")
        print("  Run: python phase-a/generate_testset.py")
        sys.exit(1)
    df = pd.read_csv(TESTSET_CSV)
    print(f"[A.2] Loaded {len(df)} questions from testset_v1.csv")
    return df


def run_rag_on_testset(df: pd.DataFrame) -> list[dict]:
    """Run every question through the Day 18 RAG pipeline."""
    from rag_bridge import query_rag

    # Resume from checkpoint if exists
    done_ids = set()
    results = []
    if CHECKPOINT_CSV.exists():
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        done_ids = set(ckpt["question"].tolist())
        results = ckpt.to_dict("records")
        print(f"[A.2] Resuming from checkpoint — {len(done_ids)} already done")

    remaining = df[~df["question"].isin(done_ids)]
    total = len(remaining)
    print(f"[A.2] Running RAG on {total} remaining questions…")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        q = row["question"]
        gt = row.get("ground_truth", "")
        print(f"  [{i}/{total}] {q[:70]}…", end=" ", flush=True)
        t0 = time.time()
        try:
            answer, contexts = query_rag(q)
            elapsed = time.time() - t0
            print(f"({elapsed:.1f}s)")
            results.append({
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": gt,
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "question": q,
                "answer": f"ERROR: {e}",
                "contexts": [],
                "ground_truth": gt,
            })

        # Save checkpoint every 5 questions
        if i % 5 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)
            print(f"  [checkpoint saved at {i} questions]")

    return results


def build_ragas_dataset(records: list[dict]) -> Dataset:
    # ragas 0.4.x expects: user_input, response, retrieved_contexts, reference
    return Dataset.from_list([
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"] if isinstance(r["contexts"], list) else [r["contexts"]],
            "reference": r["ground_truth"],
        }
        for r in records
        if not str(r.get("answer", "")).startswith("ERROR")
    ])


def run_ragas_eval(dataset: Dataset) -> pd.DataFrame:
    print(f"\n[A.2] Running RAGAS evaluation on {len(dataset)} samples...")
    print("      Estimated cost: $0.50-$1.50")

    ragas_llm = get_ragas_llm()
    ragas_emb = get_ragas_embeddings()

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        show_progress=True,
        raise_exceptions=False,
    )
    return result.to_pandas()


def main():
    if not check_qdrant():
        sys.exit(1)

    df_test = load_testset()
    records = run_rag_on_testset(df_test)

    dataset = build_ragas_dataset(records)
    if len(dataset) == 0:
        print("[ERROR] No valid records to evaluate. Check RAG pipeline errors above.")
        sys.exit(1)

    scores_df = run_ragas_eval(dataset)

    # Normalize ragas 0.4.x column names back to legacy for downstream scripts
    scores_df = scores_df.rename(columns={
        "user_input": "question",
        "response": "answer",
        "retrieved_contexts": "contexts",
        "reference": "ground_truth",
    })

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col in metric_cols:
        if col not in scores_df.columns:
            scores_df[col] = None

    scores_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\n[A.2] Results → {RESULTS_CSV}")

    summary = {col: float(scores_df[col].mean()) for col in metric_cols if col in scores_df.columns}
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[A.2] Summary → {SUMMARY_JSON}")

    print("\n── RAGAS Summary ──────────────────")
    targets = {"faithfulness": 0.85, "answer_relevancy": 0.80,
               "context_precision": 0.70, "context_recall": 0.75}
    for k, v in summary.items():
        target = targets.get(k, 0.7)
        flag = "✓" if v >= target else "✗ (below target)"
        print(f"  {k:<25} {v:.4f}  (target ≥ {target})  {flag}")
    print("────────────────────────────────────")
    print("\nNext: python phase-a/failure_analysis_gen.py")

    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()


if __name__ == "__main__":
    main()
