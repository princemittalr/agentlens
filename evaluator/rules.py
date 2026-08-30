import re
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuleResult:
    metric: str
    score: float        # 0.0 to 1.0
    passed: bool
    reason: str


def check_latency_sla(latency_ms: float, threshold_ms: float = 3000.0) -> RuleResult:
    passed = latency_ms <= threshold_ms
    return RuleResult(
        metric="latency_sla",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason=f"Latency {latency_ms}ms {'within' if passed else 'exceeded'} {threshold_ms}ms SLA"
    )


def check_output_not_empty(output: str) -> RuleResult:
    passed = bool(output and output.strip())
    return RuleResult(
        metric="output_not_empty",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason="Output is non-empty" if passed else "Output is empty"
    )


def check_no_error(error: Optional[str]) -> RuleResult:
    passed = error is None
    return RuleResult(
        metric="no_error",
        score=1.0 if passed else 0.0,
        passed=passed,
        reason="No error occurred" if passed else f"Error: {error}"
    )


def check_output_length(output: str, min_words: int = 5, max_words: int = 500) -> RuleResult:
    word_count = len(output.split())
    passed = min_words <= word_count <= max_words
    return RuleResult(
        metric="output_length",
        score=1.0 if passed else 0.5,
        passed=passed,
        reason=f"Output has {word_count} words (expected {min_words}-{max_words})"
    )


def check_no_refusal(output: str) -> RuleResult:
    refusal_phrases = [
        "i cannot", "i can't", "i am unable", "i'm unable",
        "i won't", "i will not", "as an ai", "i don't have the ability"
    ]
    lower_output = output.lower()
    refused = any(phrase in lower_output for phrase in refusal_phrases)
    return RuleResult(
        metric="no_refusal",
        score=0.0 if refused else 1.0,
        passed=not refused,
        reason="Agent refused to answer" if refused else "Agent answered without refusal"
    )


def check_json_format(output: str) -> RuleResult:
    try:
        json.loads(output)
        return RuleResult(metric="json_format", score=1.0, passed=True, reason="Valid JSON output")
    except Exception:
        return RuleResult(metric="json_format", score=0.0, passed=False, reason="Output is not valid JSON")


def run_default_rules(output: str, latency_ms: float, error: Optional[str]) -> list:
    return [
        check_no_error(error),
        check_output_not_empty(output),
        check_latency_sla(latency_ms),
        check_output_length(output),
        check_no_refusal(output),
    ]
