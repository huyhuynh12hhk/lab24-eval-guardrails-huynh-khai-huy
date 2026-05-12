"""Task A.1 — Synthetic Test Set Generation (ragas 0.4.x API)

Generates 50 questions from the Day 18 document corpus:
  - 50% SingleHop (simple factual)
  - 25% MultiHop Abstract (reasoning)
  - 25% MultiHop Specific (multi-context / cross-document)

Output: phase-a/testset_v1.csv
        phase-a/testset_review_notes.md  (template for manual review)

Run: python phase-a/generate_testset.py
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import require_env, get_ragas_llm, get_ragas_embeddings, PHASE_A
require_env("OPENAI_API_KEY")

import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)

PHASE_A.mkdir(exist_ok=True)
OUT_CSV = PHASE_A / "testset_v1.csv"
OUT_NOTES = PHASE_A / "testset_review_notes.md"

DATA_DIR = str(Path(__file__).parent.parent / "data")
TEST_SIZE = 50


def load_documents():
    print(f"[A.1] Loading documents from {DATA_DIR}")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"}, show_progress=True)
    docs = loader.load()
    print(f"[A.1] Loaded {len(docs)} documents")
    for d in docs:
        print(f"      {d.metadata.get('source', '?')} — {len(d.page_content)} chars")
    return docs


def build_query_distribution(ragas_llm):
    return [
        (SingleHopSpecificQuerySynthesizer(llm=ragas_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=ragas_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=ragas_llm), 0.25),
    ]


def generate(docs, ragas_llm, ragas_emb):
    print(f"\n[A.1] Generating {TEST_SIZE} test questions with ragas 0.4.x…")
    print("      This uses LLM calls — estimated cost: $0.50–$1.50")

    generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_emb)
    query_dist = build_query_distribution(ragas_llm)

    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=TEST_SIZE,
        transforms_llm=ragas_llm,
        transforms_embedding_model=ragas_emb,
        query_distribution=query_dist,
        raise_exceptions=False,
    )
    return testset


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map ragas 0.4.x column names to the expected lab schema."""
    rename_map = {
        "user_input": "question",
        "reference": "ground_truth",
        "reference_contexts": "contexts",
        "synthesizer_name": "evolution_type",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required columns exist
    for col in ["question", "ground_truth", "contexts", "evolution_type"]:
        if col not in df.columns:
            df[col] = ""

    # Convert contexts list → string if needed
    if df["contexts"].dtype == object:
        def _ctx_to_str(v):
            if isinstance(v, list):
                return " ||| ".join(str(x) for x in v)
            return str(v) if v else ""
        df["contexts"] = df["contexts"].apply(_ctx_to_str)

    return df[["question", "ground_truth", "contexts", "evolution_type"]]


def write_review_template(df: pd.DataFrame):
    lines = ["# Testset Review Notes\n",
             "Review at least 10 questions. Edit any that look wrong or off-topic.\n",
             "Mark corrections with **[EDITED]** so graders can see you reviewed.\n\n"]
    for i, row in df.head(15).iterrows():
        lines.append(f"## Question {i+1} ({row.get('evolution_type','?')})\n")
        lines.append(f"**Q:** {row['question']}\n\n")
        lines.append(f"**GT:** {row['ground_truth']}\n\n")
        lines.append("**Review:** _(OK / EDITED / DELETED)_\n\n")
        lines.append("---\n\n")
    OUT_NOTES.write_text("".join(lines), encoding="utf-8")
    print(f"[A.1] Review template → {OUT_NOTES}")


def main():
    docs = load_documents()
    ragas_llm = get_ragas_llm()
    ragas_emb = get_ragas_embeddings()

    testset = generate(docs, ragas_llm, ragas_emb)

    df = testset.to_pandas()
    print(f"\n[A.1] Raw columns: {list(df.columns)}")
    df = normalize_columns(df)
    df = df.dropna(subset=["question"]).reset_index(drop=True)

    print(f"\n[A.1] Generated {len(df)} questions")
    print(df["evolution_type"].value_counts().to_string())

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[A.1] Saved → {OUT_CSV}")

    write_review_template(df)
    print("\n[ACTION REQUIRED] Open phase-a/testset_review_notes.md")
    print("  Read at least 10 questions, edit any that look wrong.")
    print("  Then run: python phase-a/run_eval.py")


if __name__ == "__main__":
    main()
