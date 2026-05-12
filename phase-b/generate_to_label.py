"""Task B.3 (Step 1) — Generate to_label.csv for human calibration

Samples 10 pairs from pairwise_results.csv that you will manually judge.
Open the output file, read each pair carefully, decide who wins, and
fill in phase-b/human_labels.csv.

Output: phase-b/to_label.csv         (read this to make your judgments)
        phase-b/human_labels.csv     (template — YOU fill this in)

Run: python phase-b/generate_to_label.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import PHASE_B

import pandas as pd

PAIRWISE_CSV = PHASE_B / "pairwise_results.csv"
TO_LABEL_CSV = PHASE_B / "to_label.csv"
HUMAN_LABELS_CSV = PHASE_B / "human_labels.csv"


def main():
    if not PAIRWISE_CSV.exists():
        print(f"[ERROR] {PAIRWISE_CSV} not found. Run run_judge.py first.")
        sys.exit(1)

    df = pd.read_csv(PAIRWISE_CSV)
    sample = df.sample(min(10, len(df)), random_state=42).reset_index(drop=True)
    sample.index = sample.index + 1  # 1-based IDs

    to_label = sample[["question", "answer_a", "answer_b"]].copy()
    to_label.index.name = "question_id"
    to_label.to_csv(TO_LABEL_CSV, encoding="utf-8")
    print(f"[B.3] to_label.csv → {TO_LABEL_CSV}")

    # Human labels template
    labels = pd.DataFrame({
        "question_id": range(1, len(sample) + 1),
        "human_winner": [""] * len(sample),
        "confidence": [""] * len(sample),
        "notes": [""] * len(sample),
    })
    labels.to_csv(HUMAN_LABELS_CSV, index=False, encoding="utf-8")
    print(f"[B.3] human_labels.csv template → {HUMAN_LABELS_CSV}")

    print()
    print("=" * 60)
    print("[ACTION REQUIRED — Manual Step]")
    print("=" * 60)
    print(f"1. Open: {TO_LABEL_CSV}")
    print("   Read each of the 10 question/answer pairs carefully.")
    print()
    print(f"2. Fill in: {HUMAN_LABELS_CSV}")
    print("   For each row:")
    print("     human_winner: A  or  B  or  tie")
    print("     confidence:   high / medium / low")
    print("     notes:        1 sentence on why you chose this winner")
    print()
    print("3. Then run: python phase-b/kappa_analysis.py")
    print("=" * 60)

    # Print a preview of the pairs
    print("\n── Preview of pairs to label ───────────────────────────")
    for qid, row in sample.iterrows():
        print(f"\n[{qid}] Q: {row['question'][:80]}")
        print(f"     A: {str(row['answer_a'])[:100]}…")
        print(f"     B: {str(row['answer_b'])[:100]}…")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
