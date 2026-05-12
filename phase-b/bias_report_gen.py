"""Task B.4 — Bias Observations Report

Measures position bias and length bias in the LLM judge,
generates a quantified report with a chart.

Output: phase-b/judge_bias_report.md
        phase-b/bias_chart.png

Run: python phase-b/bias_report_gen.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import PHASE_B

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PAIRWISE_CSV = PHASE_B / "pairwise_results.csv"
OUT_MD = PHASE_B / "judge_bias_report.md"
OUT_CHART = PHASE_B / "bias_chart.png"


def analyze_position_bias(df: pd.DataFrame) -> dict:
    """Bias 1: Does the judge favour whichever answer is listed first (A)?"""
    total = len(df)
    run1_a_wins = (df["run1_winner"] == "A").sum()
    run1_b_wins = (df["run1_winner"] == "B").sum()
    run1_ties = (df["run1_winner"] == "tie").sum()
    pct_a_first = run1_a_wins / total * 100

    # After swap: if judge has NO position bias, win rate should be ~50/50
    final_a = (df["winner_after_swap"] == "A").sum()
    final_b = (df["winner_after_swap"] == "B").sum()
    final_tie = (df["winner_after_swap"] == "tie").sum()

    has_bias = pct_a_first > 55 or pct_a_first < 45
    return {
        "run1_a_wins": int(run1_a_wins),
        "run1_b_wins": int(run1_b_wins),
        "run1_ties": int(run1_ties),
        "pct_a_first": pct_a_first,
        "final_a": int(final_a),
        "final_b": int(final_b),
        "final_tie": int(final_tie),
        "bias_detected": has_bias,
        "severity": "HIGH" if abs(pct_a_first - 50) > 15 else "MODERATE" if abs(pct_a_first - 50) > 5 else "LOW",
    }


def analyze_length_bias(df: pd.DataFrame) -> dict:
    """Bias 2: Does the judge favour longer answers?"""
    df = df.copy()
    df["len_a"] = df["answer_a"].astype(str).str.len()
    df["len_b"] = df["answer_b"].astype(str).str.len()
    df["len_diff"] = df["len_b"] - df["len_a"]

    longer_b = df[df["len_diff"] > 50]
    longer_a = df[df["len_diff"] < -50]
    similar = df[df["len_diff"].abs() <= 50]

    def _win_rate(subset, winner):
        if len(subset) == 0:
            return 0.0
        return (subset["winner_after_swap"] == winner).sum() / len(subset) * 100

    b_wins_when_longer = _win_rate(longer_b, "B")
    a_wins_when_longer = _win_rate(longer_a, "A")
    avg_win_rate = (b_wins_when_longer + a_wins_when_longer) / 2

    has_bias = avg_win_rate > 60
    return {
        "longer_b_count": len(longer_b),
        "b_wins_when_longer_pct": b_wins_when_longer,
        "longer_a_count": len(longer_a),
        "a_wins_when_longer_pct": a_wins_when_longer,
        "similar_count": len(similar),
        "avg_longer_win_rate": avg_win_rate,
        "bias_detected": has_bias,
        "severity": "HIGH" if avg_win_rate > 70 else "MODERATE" if avg_win_rate > 55 else "LOW",
    }


def make_chart(pos: dict, length: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Position bias chart
    ax1 = axes[0]
    categories = ["A first\n(Run 1)", "B first\n(Run 2 flipped)"]
    a_rates = [pos["run1_a_wins"] / max(pos["run1_a_wins"] + pos["run1_b_wins"] + pos["run1_ties"], 1) * 100,
               pos["final_a"] / max(pos["final_a"] + pos["final_b"] + pos["final_tie"], 1) * 100]
    bars = ax1.bar(categories, a_rates, color=["#e74c3c", "#3498db"], alpha=0.8, edgecolor="black")
    ax1.axhline(50, color="gray", linestyle="--", label="50% (no bias)")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("A wins (%)")
    ax1.set_title(f"Position Bias\n(severity: {pos['severity']})")
    ax1.legend()
    for bar, val in zip(bars, a_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    # Length bias chart
    ax2 = axes[1]
    labels = [f"B longer\n(n={length['longer_b_count']})",
              f"A longer\n(n={length['longer_a_count']})"]
    rates = [length["b_wins_when_longer_pct"], length["a_wins_when_longer_pct"]]
    colors = ["#2ecc71" if r < 60 else "#e74c3c" for r in rates]
    bars2 = ax2.bar(labels, rates, color=colors, alpha=0.8, edgecolor="black")
    ax2.axhline(50, color="gray", linestyle="--", label="50% (no bias)")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Longer answer wins (%)")
    ax2.set_title(f"Length Bias\n(severity: {length['severity']})")
    ax2.legend()
    for bar, val in zip(bars2, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[B.4] Bias chart → {OUT_CHART}")


def generate_md(pos: dict, length: dict, n: int) -> str:
    lines = [
        "# Judge Bias Observations Report\n\n",
        f"Analyzed {n} pairwise comparisons from `pairwise_results.csv`.\n\n",
        "## Bias 1: Position Bias\n\n",
        "**Hypothesis:** Does the judge favor the answer listed first (Answer A)?\n\n",
        "| Condition | A wins | B wins | Ties | A win rate |\n",
        "|---|---|---|---|---|\n",
        f"| Run 1 (A listed first) | {pos['run1_a_wins']} | {pos['run1_b_wins']} | {pos['run1_ties']} | {pos['pct_a_first']:.1f}% |\n",
        f"| Final after swap-avg   | {pos['final_a']} | {pos['final_b']} | {pos['final_tie']} | {pos['final_a']/n*100:.1f}% |\n\n",
        f"**Severity:** {pos['severity']}\n\n",
    ]

    if pos["bias_detected"]:
        lines.append(f"**Finding:** Position bias DETECTED. Answer A wins {pos['pct_a_first']:.1f}% when listed first ")
        lines.append("(expected ~50% if unbiased). The swap-and-average mitigation reduces this.\n\n")
        lines.append("**Mitigation applied:** Swap-and-average (run judge twice with reversed order, ")
        lines.append("declare winner only when both runs agree).\n\n")
    else:
        lines.append(f"**Finding:** No significant position bias. A wins {pos['pct_a_first']:.1f}% when first (close to 50%).\n\n")

    lines += [
        "## Bias 2: Length Bias\n\n",
        "**Hypothesis:** Does the judge favor longer answers regardless of quality?\n\n",
        "| Condition | Count | Longer answer wins |\n",
        "|---|---|---|\n",
        f"| B is ≥50 chars longer | {length['longer_b_count']} | {length['b_wins_when_longer_pct']:.1f}% |\n",
        f"| A is ≥50 chars longer | {length['longer_a_count']} | {length['a_wins_when_longer_pct']:.1f}% |\n",
        f"| Similar length (±50)  | {length['similar_count']} | — |\n\n",
        f"**Average longer-answer win rate:** {length['avg_longer_win_rate']:.1f}%\n\n",
        f"**Severity:** {length['severity']}\n\n",
    ]

    if length["bias_detected"]:
        lines.append(f"**Finding:** Length bias DETECTED. Longer answers win {length['avg_longer_win_rate']:.1f}% of the time ")
        lines.append("(expected ~50%). The judge appears to equate length with quality.\n\n")
        lines.append("**Mitigation strategy:**\n")
        lines.append("- Add explicit rubric instruction: 'Do NOT favor longer answers'\n")
        lines.append("- Normalize answer length before judging (truncate to same max chars)\n")
        lines.append("- Add 'Conciseness' as an explicit scoring dimension (already in B.2)\n\n")
    else:
        lines.append(f"**Finding:** No significant length bias ({length['avg_longer_win_rate']:.1f}% longer-wins ≈ 50%).\n\n")

    lines += [
        "## Conclusion\n\n",
        "| Bias Type | Detected | Severity | Mitigation |\n",
        "|---|---|---|---|\n",
        f"| Position bias | {'Yes' if pos['bias_detected'] else 'No'} | {pos['severity']} | Swap-and-average (applied) |\n",
        f"| Length bias | {'Yes' if length['bias_detected'] else 'No'} | {length['severity']} | Add conciseness rubric instruction |\n\n",
        "See `bias_chart.png` for visual analysis.\n",
    ]

    return "".join(lines)


def main():
    if not PAIRWISE_CSV.exists():
        print(f"[ERROR] {PAIRWISE_CSV} not found. Run run_judge.py first.")
        sys.exit(1)

    df = pd.read_csv(PAIRWISE_CSV)
    if len(df) < 5:
        print(f"[ERROR] Only {len(df)} rows in pairwise_results.csv. Need at least 5.")
        sys.exit(1)

    pos = analyze_position_bias(df)
    length = analyze_length_bias(df)

    print(f"[B.4] Position bias: A wins first {pos['pct_a_first']:.1f}% — severity {pos['severity']}")
    print(f"[B.4] Length bias: longer wins {length['avg_longer_win_rate']:.1f}% — severity {length['severity']}")

    make_chart(pos, length)

    md = generate_md(pos, length, len(df))
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"[B.4] Bias report → {OUT_MD}")
    print("\nPhase B complete! Next: python phase-c/input_guard.py")


if __name__ == "__main__":
    main()
