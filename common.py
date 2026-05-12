"""Shared utilities for all Lab 24 scripts."""
import os
import sys
from pathlib import Path

# Windows fixes — applied once here so all lab scripts inherit them automatically.
if sys.platform == "win32":
    import asyncio as _asyncio
    # ProactorEventLoop crashes when httpx closes TLS connections after asyncio.run().
    _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    # cp1252 terminal can't encode Vietnamese / box-drawing chars; switch to UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

PHASE_A = ROOT / "phase-a"
PHASE_B = ROOT / "phase-b"
PHASE_C = ROOT / "phase-c"
PHASE_D = ROOT / "phase-d"


def require_env(*keys: str) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        print(f"[ERROR] Missing .env variables: {', '.join(missing)}")
        print("  Copy .env.example -> .env and fill in the values.")
        sys.exit(1)


def check_qdrant() -> bool:
    import requests
    try:
        r = requests.get("http://localhost:6333/healthz", timeout=3)
        if r.status_code == 200:
            print("[OK] Qdrant is running at localhost:6333")
            return True
    except Exception:
        pass
    print("[ERROR] Qdrant is NOT running.")
    print("  Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    return False


def get_openai_client():
    from openai import OpenAI
    require_env("OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY)


def get_ragas_llm():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    require_env("OPENAI_API_KEY")
    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY))


def get_ragas_embeddings():
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    require_env("OPENAI_API_KEY")
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=OPENAI_API_KEY))
