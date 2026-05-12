"""Task B.3 (Step 2) — Compute Cohen's Kappa

Compares your manual labels (human_labels.csv) against the LLM judge
(pairwise_results.csv) and computes Cohen's kappa agreement score.

Run AFTER manually filling phase-b/human_labels.csv.

Output: prints kappa + interpretation
        phase-b/kappa_analysis.txt  (for submission)

Run: python phase-b/kappa_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import PHASE_B

import pandas as pd
from sklearn.metrics import cohen_kappa_score

PAIRWISE_CSV = PHASE_B / "pairwise_results.csv"
HUMAN_LABELS_CSV = PHASE_B / "human_labels.csv"
OUT_TXT = PHASE_B / "kappa_analysis.txt"


def interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "WORSE than chance — judge is systematically wrong. Re-check prompts and re-label."
    elif kappa < 0.2:
        return "Slight agreement — not reliable. Check: are labels normalized? (A vs answer_a?)"
    elif kappa < 0.4:
        return "Fair agreement — still weak. Identify bias in B.4 report."
    elif kappa < 0.6:
        return "Moderate agreement — borderline. Label 10 more pairs to confirm."
    elif kappa < 0.8:
        return "Substantial agreement ✓ — production-ready judge."
    else:
        return "Almost perfect agreement — excellent calibration."


def normalize_label(v: str) -> str:
    v = str(v).strip().upper()
    if v in ("A", "ANSWER_A"):
        return "A"
    if v in ("B", "ANSWER_B"):
        return "B"
    return "tie"


def main():
    if not HUMAN_LABELS_CSV.exists():
        print(f"[ERROR] {HUMAN_LABELS_CSV} not found.")
        print("  Run: python phase-b/generate_to_label.py")
        print("  Then fill in the CSV manually.")
        sys.exit(1)

    human_df = pd.read_csv(HUMAN_LABELS_CSV)
    if human_df["human_winner"].isna().any() or (human_df["human_winner"] == "").any():
        print("[ERROR] human_labels.csv has empty entries.")
        print("  Please fill in ALL human_winner values (A, B, or tie) before running.")
        sys.exit(1)

    if not PAIRWISE_CSV.exists():
        print(f"[ERROR] {PAIRWISE_CSV} not found. Run run_judge.py first.")
        sys.exit(1)

    judge_df = pd.read_csv(PAIRWISE_CSV)
    n = len(human_df)

    human_labels = [normalize_label(v) for v in human_df["human_winner"].tolist()]
    judge_labels = [normalize_label(v) for v in judge_df.head(n)["winner_after_swap"].tolist()]

    kappa = cohen_kappa_score(human_labels, judge_labels)
    interpretation = interpret_kappa(kappa)

    agreement_count = sum(h == j for h, j in zip(human_labels, judge_labels))
    agreement_pct = agreement_count / n * 100

    output = [
        "=" * 55,
        "Cohen's Kappa Analysis — Phase B Task B.3",
        "=" * 55,
        f"Samples compared:   {n}",
        f"Raw agreement:      {agreement_count}/{n} = {agreement_pct:.1f}%",
        f"Cohen's kappa:      {kappa:.4f}",
        f"Interpretation:     {interpretation}",
        "",
        "── Per-pair comparison ──────────────────────",
    ]
    for i, (h, j) in enumerate(zip(human_labels, judge_labels), 1):
        match = "✓" if h == j else "✗"
        output.append(f"  [{i:2}] human={h:<4}  judge={j:<4}  {match}")

    if kappa < 0.6:
        output += [
            "",
            "── Root Cause Analysis (kappa < 0.6) ───────",
            "Possible causes:",
            "  1. Position bias: judge prefers answer listed first",
            "  2. Length bias: judge prefers longer/shorter answers",
            "  3. Label inconsistency: check if 'A' means the same in both files",
            "  4. Prompt ambiguity: rubric criteria not clear enough",
            "Action: Review bias_report_gen.py output (Task B.4)",
        ]

    output.append("=" * 55)
    text = "\n".join(output)

    print(text)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(f"\n[B.3] Saved → {OUT_TXT}")
    print("\nNext: python phase-b/bias_report_gen.py")


if __name__ == "__main__":
    main()
