# Failure Cluster Analysis

## Overall RAGAS Scores

| Metric | Score | Target |
|---|---|---|
| faithfulness | 0.6824 | 0.85 ✗ |
| answer_relevancy | 0.6297 | 0.8 ✗ |
| context_precision | 1.0000 | 0.7 ✓ |
| context_recall | 0.7340 | 0.75 ✗ |

## Bottom 10 Questions

| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---|---|---|---|---|---|
| 1 | "What are the obligations of the data processor in relation t" | ? | 0.00 | 0.00 | 1.00 | 0.00 | 0.25 | C2 |
| 2 | "What are the conditions outlined in Điều 15 for processing p" | ? | 0.00 | 0.00 | 1.00 | 0.00 | 0.25 | C2 |
| 3 | "What are the responsibilities of organizations and individua" | ? | 0.00 | 0.00 | 1.00 | 0.20 | 0.30 | C2 |
| 4 | "What are the key financial indicators for Tập đoàn Công nghệ" | ? | 0.00 | 0.00 | 1.00 | 0.30 | 0.32 | C2 |
| 5 | "What types of sensitive personal data are included in the re" | ? | 0.00 | 0.00 | 1.00 | 0.50 | 0.37 | C2 |
| 6 | "What are the legal regulations regarding the processing of p" | ? | 0.00 | 0.00 | 1.00 | 0.50 | 0.37 | C2 |
| 7 | "Dữ liệu cá nhân nhạy cảm và dữ liệu cá nhân trẻ em có những " | ? | 0.00 | 0.00 | 1.00 | 0.67 | 0.42 | C2 |
| 8 | "Quyền xóa dữ liệu có liên quan như thế nào đến nghĩa vụ thôn" | ? | 0.00 | 0.00 | 1.00 | 0.67 | 0.42 | C1 |
| 9 | "What are the responsibilities of organizations and individua" | ? | 0.00 | 0.00 | 1.00 | 0.75 | 0.44 | C2 |
| 10 | "Quy định pháp luật nào liên quan đến việc xử lý dữ liệu cá n" | ? | 0.00 | 0.00 | 1.00 | 1.00 | 0.50 | C1 |

## Clusters Identified

### C2: Hallucination / Low faithfulness (8 questions)

**Pattern:** Answer contains information not present in retrieved context.

**Examples:**
- "What are the obligations of the data processor in relation to the personal data "
- "What are the conditions outlined in Điều 15 for processing personal data, and ho"
- "What are the responsibilities of organizations and individuals regarding the pro"

**Root cause:** LLM (gpt-4o-mini) fills gaps with prior knowledge when context is insufficient.

**Proposed fixes:**
- Strengthen system prompt: 'If the context does not contain the answer, say "Không có thông tin"'
- Add NLI-based faithfulness check before returning answer
- Increase context window: pass parent chunk instead of child chunk

### C1: Multi-hop reasoning failures (2 questions)

**Pattern:** Questions requiring facts from 2+ documents or multi-step inference.

**Examples:**
- "Quyền xóa dữ liệu có liên quan như thế nào đến nghĩa vụ thông báo vi phạm dữ liệ"
- "Quy định pháp luật nào liên quan đến việc xử lý dữ liệu cá nhân và các nghĩa vụ "

**Root cause:** Retriever returns top-3 chunks. Multi-hop questions need ≥5 chunks spanning multiple sections.

**Proposed fixes:**
- Increase `RERANK_TOP_K` from 3 → 5 in Day 18 config.py
- Add Cohere Rerank for better cross-document relevance ordering
- Switch to hybrid search with BM25 weight ↑ for exact-match multi-hop queries

