"""Task C.1 — Input Guardrail: PII Redaction

Chains Presidio (multilingual NER) + Vietnamese-specific regex patterns
to detect and redact PII from user inputs. Tracks latency per call.

Output: phase-c/pii_test_results.csv

Run: python phase-c/input_guard.py
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import require_env  # noqa (imported for side-effect env check)

import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

PHASE_C = Path(__file__).parent
OUT_CSV = PHASE_C / "pii_test_results.csv"

# Vietnamese PII regex patterns
VN_PII = {
    "CCCD": r"\b\d{12}\b",                          # Citizen ID (12 digits)
    "PHONE_VN": r"(\+84|0)[35789]\d{8}\b",          # Vietnamese mobile
    "TAX_CODE": r"\b\d{10}(-\d{3})?\b",             # Tax code 10 or 13 digits
    "EMAIL": r"\b[\w.\-]+@[\w.\-]+\.\w{2,}\b",      # Email
    "CCCD_ALT": r"\b\d{9}\b",                        # Old 9-digit ID
}

# Prompt injection / jailbreak keyword patterns
_INJECTION_PATTERNS = re.compile(
    r"(\bDAN\b|jailbreak|pretend you are|roleplay|act as|imagine you|"
    r"no (restrictions|guidelines|rules|limits)|ignore (your|all|previous|prior) "
    r"(instructions?|rules?|guidelines?|constraints?)|"
    r"system override|developer mode|disable (all )?safety|"
    r"bypass (GDPR|data protection|privacy|security|rules?|laws?)|"
    r"steal (personal )?data|leak (data|user|personal)|"
    r"reveal (PII|all|user|confidential|training)|"
    r"extract personal data|access private (user )?records|"
    r"decode (this|the)|interpret:?\s*\d|ROT13:|Base64|"
    r"\[SYSTEM\]|\[OVERRIDE\]|AI INSTRUCTION)",
    re.IGNORECASE | re.DOTALL,
)


class InputGuard:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def scrub_vn(self, text: str) -> tuple[str, list[str]]:
        """Layer 1: Vietnamese-specific regex redaction."""
        found = []
        for name, pattern in VN_PII.items():
            matches = re.findall(pattern, text)
            if matches:
                found.extend([name] * len(matches))
                text = re.sub(pattern, f"[{name}]", text)
        return text, found

    def scrub_ner(self, text: str) -> tuple[str, list[str]]:
        """Layer 2: Presidio NER (English + international PII)."""
        results = self.analyzer.analyze(text=text, language="en")
        if not results:
            return text, []
        found = [r.entity_type for r in results]
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text, found

    def _detect_injection(self, text: str) -> list[str]:
        """Layer 0: keyword-based prompt injection / jailbreak detection."""
        return ["INJECTION"] if _INJECTION_PATTERNS.search(text) else []

    def sanitize(self, text: str) -> tuple[str, list[str], float]:
        """Full pipeline: injection check → VN regex → Presidio NER. Returns (sanitized, pii_types, latency_ms)."""
        if not text or not text.strip():
            return text, [], 0.0
        t0 = time.perf_counter()
        found0 = self._detect_injection(text)
        step1, found1 = self.scrub_vn(text)
        step2, found2 = self.scrub_ner(step1)
        latency_ms = (time.perf_counter() - t0) * 1000
        return step2, found0 + found1 + found2, latency_ms

    async def sanitize_async(self, text: str):
        import asyncio
        return await asyncio.to_thread(self.sanitize, text)


# ── Test cases ──────────────────────────────────────────────────────────────
TEST_INPUTS = [
    # English NER (Presidio catches these)
    ("EN_PERSON",    "Hi, I'm John Smith from Microsoft. Email: john.smith@ms.com"),
    ("EN_PHONE",     "Call me at +1-555-1234 or visit 123 Main Street, NYC"),
    ("VN_CCCD",      "Số CCCD của tôi là 012345678901"),
    ("VN_PHONE",     "Liên hệ qua 0987654321 hoặc 0912345678"),
    ("VN_TAX",       "Mã số thuế doanh nghiệp: 0123456789"),
    ("VN_MIXED",     "Khách hàng Nguyễn Văn A, CCCD 098765432101, phone 0912345678"),
    ("EMAIL",        "Gửi thông tin đến nguyen.van.a@example.com.vn"),
    ("EDGE_EMPTY",   ""),
    ("EDGE_CLEAN",   "What are the regulations for personal data protection?"),
    ("EDGE_LONG",    "A" * 2000 + " CCCD 012345678901 " + "B" * 2000),
]


def run_tests(guard: InputGuard) -> list[dict]:
    rows = []
    latencies = []
    detected_count = 0

    print("[C.1] Running PII detection tests…\n")
    print(f"{'Label':<15} {'PII Found':<40} {'Latency ms':>10}  {'Detected':>8}")
    print("-" * 80)

    for label, text in TEST_INPUTS:
        sanitized, pii_types, latency_ms = guard.sanitize(text)
        latencies.append(latency_ms)

        pii_str = ", ".join(pii_types) if pii_types else "—"
        detected = bool(pii_types)
        if detected:
            detected_count += 1

        has_pii_input = label not in ("EDGE_EMPTY", "EDGE_CLEAN")
        status = "✓" if (detected == has_pii_input) else "✗"

        print(f"{label:<15} {pii_str:<40} {latency_ms:>10.1f}  {status}")
        rows.append({
            "label": label,
            "input": text[:200],
            "output": sanitized[:200],
            "pii_found": pii_str,
            "pii_types_count": len(pii_types),
            "latency_ms": latency_ms,
        })

    print("-" * 80)

    pii_inputs = [l for l, _ in TEST_INPUTS if l not in ("EDGE_EMPTY", "EDGE_CLEAN")]
    detection_rate = detected_count / len(pii_inputs) * 100
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"\n[C.1] Detection rate: {detected_count}/{len(pii_inputs)} = {detection_rate:.1f}%  (target ≥ 80%)")
    print(f"[C.1] Latency P95:    {p95:.1f}ms  (target < 50ms)")

    if detection_rate >= 80:
        print("[C.1] ✓ Detection rate meets threshold")
    else:
        print("[C.1] ✗ Detection rate below 80% — check VN regex patterns")

    if p95 < 50:
        print("[C.1] ✓ Latency P95 meets threshold")
    else:
        print("[C.1] ✗ P95 latency exceeds 50ms — check presidio model loading")

    return rows


def main():
    print("[C.1] Initializing Presidio engines…")
    guard = InputGuard()
    print("[C.1] Ready.\n")

    rows = run_tests(guard)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n[C.1] Results → {OUT_CSV}")
    print("\nNext: python phase-c/topic_guard.py")


# Export guard instance for use in full_pipeline.py
_guard_instance = None

def get_input_guard() -> InputGuard:
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = InputGuard()
    return _guard_instance


if __name__ == "__main__":
    main()
