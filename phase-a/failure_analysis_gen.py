"""Task A.3 — Failure Cluster Analysis

Finds the bottom-10 questions by average RAGAS score,
groups them into clusters, and writes failure_analysis.md.

Output: phase-a/failure_analysis.md

Run: python phase-a/failure_analysis_gen.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import PHASE_A

import pandas as pd

RESULTS_CSV = PHASE_A / "ragas_results.csv"
OUT_MD = PHASE_A / "failure_analysis.md"

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        print(f"[ERROR] {RESULTS_CSV} not found. Run run_eval.py first.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_CSV)
    # ragas 0.4.x uses new column names — normalize to legacy names used below
    df = df.rename(columns={
        "user_input": "question",
        "response": "answer",
        "retrieved_contexts": "contexts",
        "reference": "ground_truth",
    })
    for col in METRIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df["avg_score"] = df[METRIC_COLS].mean(axis=1)
    return df


def infer_cluster(row: pd.Series) -> str:
    """Heuristic cluster assignment based on score patterns."""
    q = str(row.get("question", "")).lower()
    f = row.get("faithfulness", 1.0)
    cp = row.get("context_precision", 1.0)
    cr = row.get("context_recall", 1.0)
    ar = row.get("answer_relevancy", 1.0)

    multi_hop_keywords = ["compare", "difference", "relationship", "between", "versus",
                          "khác nhau", "so sánh", "mối quan hệ", "liên quan"]
    if any(kw in q for kw in multi_hop_keywords) or (cp < 0.5 and cr < 0.5):
        return "C1: Multi-hop reasoning failures"

    if f < 0.5:
        return "C2: Hallucination / Low faithfulness"

    if ar < 0.5:
        return "C3: Off-topic or irrelevant answers"

    if cp < 0.5:
        return "C4: Poor retrieval precision"

    return "C5: General low quality"


def generate_md(bottom10: pd.DataFrame, full_df: pd.DataFrame) -> str:
    cluster_counts = bottom10["cluster"].value_counts()
    clusters = bottom10["cluster"].unique().tolist()

    lines = ["# Failure Cluster Analysis\n\n"]

    # Summary stats
    lines.append("## Overall RAGAS Scores\n\n")
    lines.append("| Metric | Score | Target |\n|---|---|---|\n")
    targets = {"faithfulness": 0.85, "answer_relevancy": 0.80,
               "context_precision": 0.70, "context_recall": 0.75}
    for m in METRIC_COLS:
        avg = full_df[m].mean()
        tgt = targets.get(m, 0.7)
        flag = "✓" if avg >= tgt else "✗"
        lines.append(f"| {m} | {avg:.4f} | {tgt} {flag} |\n")
    lines.append("\n")

    # Bottom 10 table
    lines.append("## Bottom 10 Questions\n\n")
    lines.append("| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for i, (_, row) in enumerate(bottom10.iterrows(), 1):
        q = str(row.get("question", ""))[:60].replace("|", "\\|")
        etype = str(row.get("evolution_type", row.get("synthesizer_name", "?"))).split(".")[-1][:20]
        f = row.get("faithfulness", 0)
        ar = row.get("answer_relevancy", 0)
        cp = row.get("context_precision", 0)
        cr = row.get("context_recall", 0)
        avg = row.get("avg_score", 0)
        cl = row["cluster"].split(":")[0]
        lines.append(f"| {i} | \"{q}\" | {etype} | {f:.2f} | {ar:.2f} | {cp:.2f} | {cr:.2f} | {avg:.2f} | {cl} |\n")
    lines.append("\n")

    # Cluster analysis
    lines.append("## Clusters Identified\n\n")

    cluster_details = {
        "C1: Multi-hop reasoning failures": {
            "pattern": "Questions requiring facts from 2+ documents or multi-step inference.",
            "root_cause": "Retriever returns top-3 chunks. Multi-hop questions need ≥5 chunks spanning multiple sections.",
            "fixes": [
                "Increase `RERANK_TOP_K` from 3 → 5 in Day 18 config.py",
                "Add Cohere Rerank for better cross-document relevance ordering",
                "Switch to hybrid search with BM25 weight ↑ for exact-match multi-hop queries",
            ],
        },
        "C2: Hallucination / Low faithfulness": {
            "pattern": "Answer contains information not present in retrieved context.",
            "root_cause": "LLM (gpt-4o-mini) fills gaps with prior knowledge when context is insufficient.",
            "fixes": [
                "Strengthen system prompt: 'If the context does not contain the answer, say \"Không có thông tin\"'",
                "Add NLI-based faithfulness check before returning answer",
                "Increase context window: pass parent chunk instead of child chunk",
            ],
        },
        "C3: Off-topic or irrelevant answers": {
            "pattern": "Answer doesn't address the question despite retrieving relevant chunks.",
            "root_cause": "Semantic gap between Vietnamese query embedding and chunk embeddings.",
            "fixes": [
                "Increase `EMBEDDING_DIM` or switch to a multilingual model with better VN support",
                "Add query expansion: translate VN terms to legal/financial synonyms",
                "Use HyDE (Hypothetical Document Embedding) to bridge semantic gap",
            ],
        },
        "C4: Poor retrieval precision": {
            "pattern": "Retrieved chunks are weakly related to the question.",
            "root_cause": "BM25 keyword mismatch for complex Vietnamese terminology.",
            "fixes": [
                "Tune BM25 parameters: increase `k1`, adjust `b` for Vietnamese text length norms",
                "Add keyword aliases in chunk enrichment (M5) for legal terms",
                "Implement DPR (Dense Passage Retrieval) fine-tuned on Vietnamese legal corpus",
            ],
        },
        "C5: General low quality": {
            "pattern": "No single dominant failure mode; uniformly low scores.",
            "root_cause": "Questions may be ambiguous or require domain knowledge outside the corpus.",
            "fixes": [
                "Review and remove ambiguous questions from test set",
                "Expand corpus with more domain-specific documents",
                "Add document metadata filtering to scope retrieval",
            ],
        },
    }

    for cluster in clusters:
        if cluster not in cluster_details:
            continue
        detail = cluster_details[cluster]
        examples = bottom10[bottom10["cluster"] == cluster]["question"].head(3).tolist()
        count = cluster_counts.get(cluster, 0)

        lines.append(f"### {cluster} ({count} questions)\n\n")
        lines.append(f"**Pattern:** {detail['pattern']}\n\n")
        lines.append("**Examples:**\n")
        for ex in examples:
            lines.append(f'- "{ex[:80]}"\n')
        lines.append(f"\n**Root cause:** {detail['root_cause']}\n\n")
        lines.append("**Proposed fixes:**\n")
        for fix in detail["fixes"]:
            lines.append(f"- {fix}\n")
        lines.append("\n")

    return "".join(lines)


def main():
    df = load_results()
    bottom10 = df.nsmallest(10, "avg_score").copy()
    bottom10["cluster"] = bottom10.apply(infer_cluster, axis=1)

    print("[A.3] Bottom 10 questions by average RAGAS score:")
    for i, (_, row) in enumerate(bottom10.iterrows(), 1):
        print(f"  {i:2}. avg={row['avg_score']:.3f}  {row.get('question','')[:70]}")

    print(f"\n[A.3] Clusters identified:")
    for cl, count in bottom10["cluster"].value_counts().items():
        print(f"  {cl}: {count} questions")

    md = generate_md(bottom10, df)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"\n[A.3] Failure analysis → {OUT_MD}")
    print("\n[ACTION REQUIRED] Review the generated failure_analysis.md.")
    print("  Add your own observations and confirm the proposed fixes make sense.")
    print("\nNext: python phase-b/run_judge.py")


if __name__ == "__main__":
    main()
