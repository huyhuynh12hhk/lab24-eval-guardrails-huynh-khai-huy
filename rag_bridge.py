"""Self-contained RAG pipeline for Day 24.

Reads documents from the local data/ directory.
No dependency on the Day 18 project.

Usage:
    from rag_bridge import query_rag, query_rag_v2

    answer, contexts = query_rag("What is personal data?")
    answer_b, contexts_b = query_rag_v2("What is personal data?")
"""
import asyncio
import glob
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from common import OPENAI_API_KEY, get_openai_client, require_env

DATA_DIR = Path(__file__).parent / "data"

# Qdrant collection name — intentionally different from Day 18 ("lab18_production")
COLLECTION_NAME = "lab24_rag"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANK_TOP_K = 3
BM25_TOP_K = 20
DENSE_TOP_K = 20

# ── Data loading ───────────────────────────────────────────────────────────────

def _load_documents() -> list[dict]:
    docs = []
    for fp in sorted(glob.glob(str(DATA_DIR / "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ── Chunking (hierarchical parent-child) ──────────────────────────────────────

@dataclass
class _Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _split_into_children(parent_text: str, pid: str, metadata: dict, child_size: int) -> list[_Chunk]:
    result = []
    start = 0
    while start < len(parent_text):
        child_text = parent_text[start:start + child_size].strip()
        if child_text:
            result.append(_Chunk(child_text, {**metadata, "parent_id": pid, "child_index": len(result)}))
        start += child_size
    return result


def _flush_parent(current: str, p_idx: int, metadata: dict,
                  parents: list, children: list, child_size: int) -> str:
    pid = f"parent_{p_idx}"
    parents.append(_Chunk(current.strip(), {**metadata, "chunk_type": "parent", "parent_id": pid}))
    children.extend(_split_into_children(current, pid, metadata, child_size))
    return ""


def _chunk_hierarchical(text: str, metadata: dict,
                         parent_size: int = 2048, child_size: int = 256):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents, children = [], []
    current, p_idx = "", 0

    for para in paragraphs:
        if len(current) + len(para) > parent_size and current:
            current = _flush_parent(current, p_idx, metadata, parents, children, child_size)
            p_idx += 1
        current += para + "\n\n"

    if current.strip():
        _flush_parent(current, p_idx, metadata, parents, children, child_size)

    return parents, children


# ── Vietnamese segmentation ────────────────────────────────────────────────────

def _segment_vn(text: str) -> str:
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except Exception:
        return text


# ── Encoder singleton ──────────────────────────────────────────────────────────

_ENCODER = None

def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _ENCODER


# ── Pipeline build + cache ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_pipeline():
    print("[RAG-D24] Building standalone pipeline…")
    print("          First load ~30s, subsequent calls instant.")

    print("  [1/3] Loading and chunking documents…")
    docs = _load_documents()
    all_chunks = []
    for doc in docs:
        _, children = _chunk_hierarchical(doc["text"], doc["metadata"])
        all_chunks.extend({"text": c.text, "metadata": c.metadata} for c in children)
    print(f"         {len(all_chunks)} chunks from {len(docs)} documents")

    print("  [2/3] Indexing BM25 + Dense (Qdrant)…")
    bm25_index, bm25_docs = _build_bm25(all_chunks)
    _build_qdrant(all_chunks)
    print(f"         Indexed into collection '{COLLECTION_NAME}'")

    print("  [3/3] Loading CrossEncoder reranker…")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
    print("  [RAG-D24] Pipeline ready.")

    return bm25_index, bm25_docs, reranker


def _build_bm25(chunks: list[dict]):
    from rank_bm25 import BM25Okapi
    tokenized = [_segment_vn(c["text"]).split() for c in chunks]
    return BM25Okapi(tokenized), chunks


def _build_qdrant(chunks: list[dict]):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    encoder = _get_encoder()

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    texts = [c["text"] for c in chunks]
    vectors = encoder.encode(texts, batch_size=32, show_progress_bar=True, device="cpu")

    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": chunks[i]["text"], **chunks[i]["metadata"]})
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


# ── Query ──────────────────────────────────────────────────────────────────────

def _hybrid_search(query: str, bm25_index, bm25_docs: list[dict], top_k: int = DENSE_TOP_K) -> list[dict]:
    from qdrant_client import QdrantClient

    # BM25
    q_tokens = _segment_vn(query).split()
    bm25_scores = bm25_index.get_scores(q_tokens)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_TOP_K]

    # Dense
    encoder = _get_encoder()
    q_vec = encoder.encode([query], device="cpu")[0].tolist()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    dense_results = client.query_points(
        collection_name=COLLECTION_NAME, query=q_vec, limit=DENSE_TOP_K
    ).points

    # RRF fusion
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0) + 1.0 / (60 + rank + 1)
    for rank, pt in enumerate(dense_results):
        scores[pt.id] = scores.get(pt.id, 0) + 1.0 / (60 + rank + 1)

    top_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"text": bm25_docs[i]["text"], "score": scores[i], "metadata": bm25_docs[i]["metadata"]}
            for i in top_ids if i < len(bm25_docs)]


def query_rag(question: str) -> tuple[str, list[str]]:
    """Full pipeline: hybrid search → crossencoder rerank → GPT-4o-mini."""
    require_env("OPENAI_API_KEY")
    bm25_index, bm25_docs, reranker = _build_pipeline()

    results = _hybrid_search(question, bm25_index, bm25_docs)
    if not results:
        return "Không tìm thấy thông tin trong tài liệu.", []

    pairs = [(question, r["text"]) for r in results]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)[:RERANK_TOP_K]
    contexts = [r["text"] for _, r in reranked]

    context_str = "\n\n".join(contexts)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Trả lời CHỈ dựa trên context được cung cấp. "
                "Nếu không tìm thấy thông tin → nói 'Không tìm thấy thông tin trong tài liệu.' "
                "Trả lời bằng tiếng Việt, ngắn gọn và chính xác."
            )},
            {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {question}"},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return resp.choices[0].message.content, contexts


def query_rag_v2(question: str) -> tuple[str, list[str]]:
    """Weaker baseline: top-1 context, basic prompt. Used for Phase B pairwise."""
    require_env("OPENAI_API_KEY")
    _, contexts = query_rag(question)
    top1 = contexts[0] if contexts else ""

    client = get_openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context. Be concise."},
            {"role": "user", "content": f"Context:\n{top1}\n\nQuestion: {question}"},
        ],
        max_tokens=256,
        temperature=0.1,
    )
    return resp.choices[0].message.content, [top1]


async def query_rag_async(question: str) -> tuple[str, list[str]]:
    return await asyncio.to_thread(query_rag, question)
