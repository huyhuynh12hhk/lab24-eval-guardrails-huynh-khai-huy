"""Run this FIRST before any lab scripts.

Verifies: Python version, API keys, Qdrant, required packages,
presidio spacy model, and the Day 18 RAG pipeline.

Usage:
    python scripts/setup_check.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

errors = []


def check(label: str, fn):
    try:
        result = fn()
        if result is False:
            print(f"{FAIL} {label}")
            errors.append(label)
        else:
            msg = f" ({result})" if isinstance(result, str) else ""
            print(f"{PASS} {label}{msg}")
    except Exception as e:
        print(f"{FAIL} {label} — {e}")
        errors.append(label)


# 1. Python version
check("Python >= 3.10", lambda: sys.version_info >= (3, 10) or False)

# 2. .env file
from common import OPENAI_API_KEY, HF_TOKEN
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
check("OPENAI_API_KEY set in .env", lambda: bool(OPENAI_API_KEY) or False)
check("HF_TOKEN set in .env (needed for Llama Guard)", lambda: bool(HF_TOKEN) or WARN)
check(f"data/ directory exists: {DATA_DIR}", lambda: os.path.isdir(DATA_DIR) or False)

# 3. Required packages
packages = [
    ("ragas", "ragas"),
    ("presidio_analyzer", "presidio-analyzer"),
    ("presidio_anonymizer", "presidio-anonymizer"),
    ("sklearn", "scikit-learn"),
    ("langchain_openai", "langchain-openai"),
    ("datasets", "datasets"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
]
for mod, pkg in packages:
    def _check(m=mod, p=pkg):
        __import__(m)
        return p
    check(f"Package: {pkg}", _check)

# 4. spacy English model
def _spacy_model():
    import spacy
    spacy.load("en_core_web_lg")
    return "en_core_web_lg"
check("spacy en_core_web_lg model", _spacy_model)

# 5. ragas version
def _ragas_ver():
    import ragas
    v = ragas.__version__
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) >= (0, 4):
        return f"v{v} (0.4.x API — new synthesizers)"
    return f"v{v}"
check("ragas version", _ragas_ver)

# 6. Qdrant
import requests as _req
def _qdrant():
    r = _req.get("http://localhost:6333/healthz", timeout=3)
    return r.status_code == 200 or False
check("Qdrant running at localhost:6333", _qdrant)

# 7. Model cache — verify bge-m3 and bge-reranker are downloaded
# NOTE: pipeline.py sets HF_HUB_OFFLINE="1" but huggingface_hub has already been
# imported above (by ragas/langchain), so that env-var lands too late.
# Pre-downloading here ensures the pipeline test never hits the network.
REQUIRED_MODELS = [
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
]

def _ensure_model_cached(model_id: str) -> str:
    from huggingface_hub import try_to_load_from_cache, snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    # Fast path: check if already in cache
    cached = try_to_load_from_cache(model_id, "config.json")
    if cached and cached != "<no_cached_version>":
        return "already cached"

    # Not cached — download now with visible progress
    print(f"  [Downloading {model_id} — first-time only, may take a few minutes…]")
    snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "rust_model.ot", "flax_model.msgpack"],
    )
    return "downloaded"

for model_id in REQUIRED_MODELS:
    def _model_check(mid=model_id):
        return _ensure_model_cached(mid)
    check(f"Model cache: {model_id}", _model_check)

# 8. Quick RAG smoke test (uses the self-contained Day 24 rag_bridge)
def _rag_test():
    import huggingface_hub.constants as _hf_const
    _hf_const.HF_HUB_OFFLINE = True
    from rag_bridge import query_rag
    answer, contexts = query_rag("Nghị định 13 quy định về điều gì?")
    assert isinstance(answer, str) and len(answer) > 0, "Empty answer"
    assert isinstance(contexts, list) and len(contexts) > 0, "Empty contexts"
    return f"answer: {len(answer)} chars, {len(contexts)} context(s)"
check("RAG pipeline smoke test", _rag_test)

# Summary
print()
if errors:
    print(f"[SETUP INCOMPLETE] Fix {len(errors)} issue(s) before running lab scripts:")
    for e in errors:
        print(f"  [X] {e}")
    sys.exit(1)
else:
    print("[SETUP COMPLETE] All checks passed. You are ready to start Lab 24!")
    print()
    print("Run order:")
    print("  1. python phase-a/generate_testset.py")
    print("  2. python phase-a/run_eval.py")
    print("  3. python phase-a/failure_analysis_gen.py")
    print("  4. python phase-b/run_judge.py")
    print("  5. python phase-b/generate_to_label.py")
    print("     --> MANUALLY fill phase-b/human_labels.csv")
    print("  6. python phase-b/kappa_analysis.py")
    print("  7. python phase-b/bias_report_gen.py")
    print("  8. python phase-c/input_guard.py")
    print("  9. python phase-c/topic_guard.py")
    print(" 10. python phase-c/adversarial_test.py")
    print(" 11. python phase-c/output_guard.py")
    print(" 12. python phase-c/full_pipeline.py")
