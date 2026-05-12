"""Task A.4 — CI/CD Eval Gate Script

Used by .github/workflows/eval-gate.yml to block a PR if RAGAS scores
fall below thresholds. Exit code 0 = pass, exit code 1 = fail.

Usage:
    python phase-a/eval_gate.py --threshold faithfulness=0.85
    python phase-a/eval_gate.py  # uses defaults
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import PHASE_A

SUMMARY_JSON = PHASE_A / "ragas_summary.json"

DEFAULT_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.75,
}


def parse_threshold_arg(s: str) -> tuple[str, float]:
    k, v = s.split("=")
    return k.strip(), float(v.strip())


def main():
    parser = argparse.ArgumentParser(description="RAGAS eval gate")
    parser.add_argument("--threshold", action="append", default=[],
                        help="metric=value e.g. --threshold faithfulness=0.85")
    parser.add_argument("--summary", default=str(SUMMARY_JSON),
                        help="Path to ragas_summary.json")
    args = parser.parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    for t in args.threshold:
        k, v = parse_threshold_arg(t)
        thresholds[k] = v

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"[GATE] ERROR: {summary_path} not found. Run run_eval.py first.")
        sys.exit(1)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    print("[GATE] RAGAS Evaluation Results")
    print("=" * 50)

    failures = []
    for metric, threshold in thresholds.items():
        score = summary.get(metric)
        if score is None:
            print(f"  {metric:<25} MISSING  (threshold: {threshold})")
            failures.append(metric)
            continue
        status = "PASS ✓" if score >= threshold else "FAIL ✗"
        print(f"  {metric:<25} {score:.4f}  (threshold: ≥{threshold})  {status}")
        if score < threshold:
            failures.append(metric)

    print("=" * 50)

    if failures:
        print(f"\n[GATE] BLOCKED — {len(failures)} metric(s) below threshold: {', '.join(failures)}")
        print("       Fix the RAG pipeline before merging.")
        sys.exit(1)
    else:
        print("\n[GATE] PASSED — all metrics above threshold. Merge allowed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
