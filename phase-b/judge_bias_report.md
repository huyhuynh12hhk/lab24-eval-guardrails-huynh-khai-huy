# Judge Bias Observations Report

Analyzed 30 pairwise comparisons from `pairwise_results.csv`.

## Bias 1: Position Bias

**Hypothesis:** Does the judge favor the answer listed first (Answer A)?

| Condition | A wins | B wins | Ties | A win rate |
|---|---|---|---|---|
| Run 1 (A listed first) | 14 | 10 | 6 | 46.7% |
| Final after swap-avg   | 13 | 9 | 8 | 43.3% |

**Severity:** LOW

**Finding:** No significant position bias. A wins 46.7% when first (close to 50%).

## Bias 2: Length Bias

**Hypothesis:** Does the judge favor longer answers regardless of quality?

| Condition | Count | Longer answer wins |
|---|---|---|
| B is ≥50 chars longer | 6 | 100.0% |
| A is ≥50 chars longer | 10 | 80.0% |
| Similar length (±50)  | 14 | — |

**Average longer-answer win rate:** 90.0%

**Severity:** HIGH

**Finding:** Length bias DETECTED. Longer answers win 90.0% of the time (expected ~50%). The judge appears to equate length with quality.

**Mitigation strategy:**
- Add explicit rubric instruction: 'Do NOT favor longer answers'
- Normalize answer length before judging (truncate to same max chars)
- Add 'Conciseness' as an explicit scoring dimension (already in B.2)

## Conclusion

| Bias Type | Detected | Severity | Mitigation |
|---|---|---|---|
| Position bias | No | LOW | Swap-and-average (applied) |
| Length bias | Yes | HIGH | Add conciseness rubric instruction |

See `bias_chart.png` for visual analysis.
