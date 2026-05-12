# Lab 24 — Full Evaluation & Guardrail System

## Overview

Built a production-ready evaluation and guardrail stack for a Vietnamese legal/financial RAG pipeline (Day 18). The system measures RAG quality with RAGAS, uses LLM-as-Judge for flexible evaluation, and layers PII redaction + topic validation + Llama Guard 3 for defense-in-depth safety.

## Setup

```bash
# 1. Copy and fill in API keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY, HF_TOKEN (for Llama Guard), optionally GROQ_API_KEY

# 2. Install additional packages
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 3. Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 4. Verify everything is ready
python scripts/setup_check.py
```

## Run Order

```bash
python phase-a/generate_testset.py          # A.1: generate 50 test questions
# → review phase-a/testset_review_notes.md (manual step)
python phase-a/run_eval.py                  # A.2: run RAGAS 4 metrics
python phase-a/failure_analysis_gen.py      # A.3: cluster failures
# eval-gate.yml handles A.4 (CI/CD)

python phase-b/run_judge.py                 # B.1+B.2: pairwise + absolute scoring
python phase-b/generate_to_label.py         # B.3 setup
# → manually fill phase-b/human_labels.csv (manual step)
python phase-b/kappa_analysis.py            # B.3: Cohen's kappa
python phase-b/bias_report_gen.py           # B.4: bias analysis

python phase-c/input_guard.py               # C.1: PII redaction
python phase-c/topic_guard.py               # C.2: topic validator
python phase-c/adversarial_test.py          # C.3: adversarial testing
python phase-c/output_guard.py              # C.4: Llama Guard
python phase-c/full_pipeline.py             # C.5: full stack benchmark
# → fill in phase-d/blueprint.md with actual numbers
```

## Results Summary

### Phase A (RAGAS)

- Test set: 50 questions (50% simple, 25% reasoning, 25% multi-context)
- Faithfulness: **0.6824** (target ≥ 0.85 — below target; see failure analysis)
- Answer Relevancy: **0.6297** (target ≥ 0.80 — below target)
- Context Precision: **1.0000** (target ≥ 0.70 — ✓)
- Context Recall: **0.7340** (target ≥ 0.75 — near target)
- Failure clusters identified: C2 Hallucination (low faithfulness) and C3 Off-topic answers; see `phase-a/failure_analysis.md`

### Phase B (LLM-Judge)

- Cohen's kappa vs human: **0.6970** (Substantial agreement — production ready)
- Position bias: **46.7%** (LOW — acceptable)
- Length bias: **90%** (HIGH — judge prefers longer answers; apply length-balanced prompts in production)
- See `phase-b/judge_bias_report.md`

### Phase C (Guardrails)

- PII detection rate: **8/8** PII inputs detected (100%)
- Topic validator accuracy: **20/20 = 100%** (threshold = 0.773, on-top min sim = 0.774)
- Adversarial defense rate: **20/20 = 100%** (DAN, roleplay, split, encoding, indirect injection)
- Output safety detection: **8/10 = 80%** via OpenAI Moderation API (meets ≥ 80% target)
- False positive rate: **0%** on legitimate queries
- L1 P95 latency: see `phase-c/latency_benchmark.csv`
- L3 P95 latency: see `phase-c/latency_benchmark.csv`

### Phase D (Blueprint)

See `phase-d/blueprint.md`

## Bài Học Kinh Nghiệm

### 1. Đánh giá RAG không chỉ là một con số duy nhất

Trước khi làm Lab 24, tôi nghĩ "đánh giá RAG" nghĩa là đo một metric tổng hợp rồi kết luận pipeline tốt hay xấu. Sau khi chạy RAGAS với 4 metrics riêng biệt, tôi thấy rõ context_precision đạt 1.00 (tuyệt vời) trong khi faithfulness chỉ đạt 0.68 (kém). Điều này chỉ ra rằng retrieval đang lấy đúng chunks nhưng LLM lại bịa thêm thông tin không có trong context — hai vấn đề hoàn toàn khác nhau, cần hai cách sửa khác nhau. RAGAS cho phép "diagnostic tree" thực sự: nhìn vào pattern của 4 metrics để biết nên fix retrieval, hay chunking, hay system prompt.

### 2. LLM-as-Judge cần được hiệu chỉnh, không thể tin mù quáng

Kết quả Phase B cho thấy judge đạt Cohen's kappa κ = 0.697 so với human labels — mức "Substantial agreement" đủ dùng cho production, nhưng length bias lên tới 90% là vấn đề nghiêm trọng: judge gần như luôn chọn câu trả lời dài hơn bất kể chất lượng. Nếu không đo bias này, tôi sẽ không phát hiện ra rằng mọi so sánh A/B đều bị ảnh hưởng bởi độ dài câu trả lời. Bài học: trước khi deploy LLM judge vào production, phải chạy swap-position test và length-correlation test để đo bias — chi phí thấp nhưng giá trị cao.

### 3. Guardrail hiệu quả nhất là kết hợp nhiều lớp bổ sung cho nhau

Topic guard dùng embedding similarity ban đầu có accuracy 50% vì các câu tấn công kiểu roleplay chứa từ khóa pháp lý ("steal personal data") khiến similarity score vượt ngưỡng. Sau khi thêm lớp keyword-based injection detection — đơn giản chỉ là regex tìm patterns như "pretend you are", "bypass GDPR", "act as" — detection rate nhảy từ 65% lên 100% mà không tốn thêm API call nào. Đây là ví dụ điển hình của defense-in-depth: semantic similarity giỏi phân biệt topic, keyword matching giỏi phát hiện injection pattern — kết hợp hai cái mạnh hơn từng cái riêng lẻ rất nhiều.

