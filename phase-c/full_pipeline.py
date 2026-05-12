"""Task C.5 — Full Stack Integration & Latency Benchmark

Wires all 4 guardrail layers into a single async pipeline:
  [L1] Input (PII + Topic) — parallel
  [L2] RAG LLM call (Day 18 pipeline)
  [L3] Output safety check (Llama Guard 3) — async
  [L4] Audit log — fire-and-forget (not counted in latency budget)

Runs a benchmark over ≥100 queries and reports P50/P95/P99 latency.

Output: phase-c/latency_benchmark.csv

Run: python phase-c/full_pipeline.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from common import require_env, check_qdrant, PHASE_A

require_env("OPENAI_API_KEY")

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

PHASE_C = Path(__file__).parent
OUT_CSV = PHASE_C / "latency_benchmark.csv"
AUDIT_LOG = PHASE_C / "audit_log.jsonl"

BENCHMARK_N = 100


def refuse_response(reason: str = "") -> str:
    if "topic" in reason.lower():
        return ("Xin lỗi, tôi chỉ có thể trả lời câu hỏi về luật bảo vệ dữ liệu cá nhân "
                "và báo cáo tài chính. Vui lòng đặt câu hỏi liên quan đến các chủ đề này.")
    return "Xin lỗi, tôi không thể trả lời câu hỏi này vì lý do an toàn."


def audit_log_sync(user_input: str, answer: str, timings: dict, blocked: bool) -> None:
    """Synchronous audit write — called via asyncio.to_thread for non-blocking I/O."""
    entry = {
        "timestamp": time.time(),
        "input": user_input[:100],
        "blocked": blocked,
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
    }
    if not blocked:
        entry["answer_preview"] = answer[:80]
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def guarded_pipeline(user_input: str, input_guard, topic_guard, output_guard,
                            rag_query_fn) -> tuple[str, dict, bool]:
    """
    Full 4-layer async pipeline.
    Returns: (answer, timings, was_blocked)
    """
    timings = {}

    # ── L1: Input checks (parallel) ───────────────────────────────────────
    t0 = time.perf_counter()
    pii_task = asyncio.create_task(input_guard.sanitize_async(user_input))
    topic_task = asyncio.create_task(topic_guard.check_async(user_input))

    (sanitized, _, _), (topic_ok, topic_reason) = await asyncio.gather(pii_task, topic_task)
    timings["L1"] = (time.perf_counter() - t0) * 1000

    if not topic_ok:
        answer = refuse_response(topic_reason)
        _audit_task = asyncio.create_task(
            asyncio.to_thread(audit_log_sync, user_input, answer, timings, True)
        )
        return answer, timings, True

    # ── L2: RAG LLM call ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    answer, _ = await rag_query_fn(sanitized)
    timings["L2"] = (time.perf_counter() - t0) * 1000

    # ── L3: Output safety check ───────────────────────────────────────────
    t0 = time.perf_counter()
    is_safe, _, _ = await output_guard.check_async(sanitized, answer)
    timings["L3"] = (time.perf_counter() - t0) * 1000

    if not is_safe:
        safe_answer = refuse_response("unsafe")
        _audit_task = asyncio.create_task(
            asyncio.to_thread(audit_log_sync, user_input, safe_answer, timings, True)
        )
        return safe_answer, timings, True

    # ── L4: Async audit log (fire-and-forget) ─────────────────────────────
    _audit_task = asyncio.create_task(
        asyncio.to_thread(audit_log_sync, user_input, answer, timings, False)
    )

    return answer, timings, False


def load_benchmark_queries() -> list[str]:
    """Load queries from testset or generate synthetic ones."""
    testset_csv = PHASE_A / "testset_v1.csv"
    base_queries = []

    if testset_csv.exists():
        df = pd.read_csv(testset_csv)
        base_queries = df["question"].dropna().tolist()

    if not base_queries:
        base_queries = [
            "Dữ liệu cá nhân nhạy cảm bao gồm những loại thông tin nào?",
            "Quyền của chủ thể dữ liệu theo Nghị định 13/2023?",
            "Điều kiện xử lý dữ liệu cá nhân theo luật Việt Nam?",
            "Mức phạt vi phạm bảo vệ dữ liệu cá nhân?",
            "Doanh thu công ty ABC năm 2023 là bao nhiêu?",
            "Lợi nhuận sau thuế trong kỳ báo cáo?",
            "Trách nhiệm của bên kiểm soát dữ liệu?",
            "Chuyển dữ liệu cá nhân ra nước ngoài cần điều kiện gì?",
        ]

    # Repeat to reach BENCHMARK_N
    queries = base_queries * (BENCHMARK_N // len(base_queries) + 1)
    return queries[:BENCHMARK_N]


async def run_benchmark():
    from input_guard import get_input_guard
    from topic_guard import get_topic_guard
    # OutputGuard must be initialized BEFORE rag_bridge is imported:
    # Day 18 pipeline.py sets CUDA_VISIBLE_DEVICES="" at import time.
    # rag_bridge.py restores it afterward, but torch.cuda is already
    # initialized by then if OutputGuardGPU loaded first. Loading output
    # guard here (before rag_bridge import) ensures Llama Guard sees the GPU.
    from output_guard import get_cached_output_guard
    output_guard = get_cached_output_guard()

    from rag_bridge import query_rag_async

    print("[C.5] Initializing all guardrail components…")
    input_guard = get_input_guard()
    topic_guard = get_topic_guard()

    queries = load_benchmark_queries()
    print(f"[C.5] Benchmarking {len(queries)} queries…\n")

    all_timings = []
    rows = []

    # Warm-up run (not counted)
    print("[C.5] Warm-up run…")
    try:
        await guarded_pipeline(queries[0], input_guard, topic_guard, output_guard, query_rag_async)
    except Exception as e:
        print(f"[WARN] Warm-up failed: {e}")

    # Benchmark
    for i, q in enumerate(queries, 1):
        try:
            t_total_start = time.perf_counter()
            _, timings, blocked = await guarded_pipeline(
                q, input_guard, topic_guard, output_guard, query_rag_async
            )
            t_total = (time.perf_counter() - t_total_start) * 1000

            timings["total"] = t_total
            all_timings.append(timings)
            rows.append({"query": q[:60], "blocked": blocked, **{k: round(v, 1) for k, v in timings.items()}})

            if i % 10 == 0:
                print(f"  {i}/{len(queries)} done — last total: {t_total:.0f}ms")
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")

    return all_timings, rows


def print_latency_report(all_timings: list[dict]):
    print("\n" + "=" * 55)
    print("Latency Benchmark Report (Phase C.5)")
    print("=" * 55)
    print(f"{'Layer':<8} {'P50':>8} {'P95':>8} {'P99':>8}  Target")
    print("-" * 55)

    targets = {"L1": 50, "L2": None, "L3": 100, "total": 2500}

    for layer in ["L1", "L2", "L3", "total"]:
        vals = [t[layer] for t in all_timings if layer in t]
        if not vals:
            continue
        p50 = np.percentile(vals, 50)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        target = targets.get(layer)
        tgt_str = f"< {target}ms" if target else "—"
        flag = ""
        if target and p95 > target:
            flag = "  ✗ P95 exceeds target"
        elif target:
            flag = "  ✓"
        print(f"{layer:<8} {p50:>7.0f}ms {p95:>7.0f}ms {p99:>7.0f}ms  {tgt_str}{flag}")

    print("=" * 55)
    total_vals = [t.get("total", 0) for t in all_timings if "total" in t]
    if total_vals:
        overhead_pct = np.mean([
            (t.get("L1", 0) + t.get("L3", 0)) / max(t.get("total", 1), 1) * 100
            for t in all_timings
        ])
        print(f"\nGuardrail overhead (L1+L3 vs total): {overhead_pct:.1f}% of latency budget")


async def main():
    if not check_qdrant():
        sys.exit(1)

    all_timings, rows = await run_benchmark()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False, encoding="utf-8")
        print(f"\n[C.5] Benchmark results → {OUT_CSV}")

    print_latency_report(all_timings)
    print(f"\n[C.5] Audit log → {AUDIT_LOG}")
    print("\nPhase C complete! Next: fill in phase-d/blueprint.md with your actual numbers.")


if __name__ == "__main__":
    asyncio.run(main())
