"""
Unit tests for evaluator/rules.py
Tests all rule-based metric checks deterministically.
"""
import sys
sys.path.insert(0, "/home/prince-mittal/agentlens")

import pytest
from evaluator.rules import (
    check_latency_sla,
    check_output_not_empty,
    check_no_error,
    check_output_length,
    check_no_refusal,
    check_json_format,
    run_default_rules,
)


# ─── check_latency_sla ───────────────────────────────────────────────────────

class TestLatencySLA:
    def test_passes_within_threshold(self):
        result = check_latency_sla(500, threshold_ms=3000)
        assert result.passed is True
        assert result.score == 1.0
        assert result.metric == "latency_sla"

    def test_fails_above_threshold(self):
        result = check_latency_sla(5000, threshold_ms=3000)
        assert result.passed is False
        assert result.score == 0.0

    def test_passes_exactly_at_threshold(self):
        result = check_latency_sla(3000, threshold_ms=3000)
        assert result.passed is True

    def test_custom_threshold(self):
        result = check_latency_sla(200, threshold_ms=100)
        assert result.passed is False

    def test_zero_latency(self):
        result = check_latency_sla(0, threshold_ms=3000)
        assert result.passed is True
        assert result.score == 1.0


# ─── check_output_not_empty ──────────────────────────────────────────────────

class TestOutputNotEmpty:
    def test_passes_with_content(self):
        result = check_output_not_empty("Hello world")
        assert result.passed is True
        assert result.score == 1.0

    def test_fails_with_empty_string(self):
        result = check_output_not_empty("")
        assert result.passed is False
        assert result.score == 0.0

    def test_fails_with_whitespace_only(self):
        result = check_output_not_empty("   ")
        assert result.passed is False

    def test_fails_with_none(self):
        result = check_output_not_empty(None)
        assert result.passed is False

    def test_passes_with_single_character(self):
        result = check_output_not_empty("x")
        assert result.passed is True


# ─── check_no_error ──────────────────────────────────────────────────────────

class TestNoError:
    def test_passes_with_no_error(self):
        result = check_no_error(None)
        assert result.passed is True
        assert result.score == 1.0

    def test_fails_with_error_string(self):
        result = check_no_error("Connection timeout")
        assert result.passed is False
        assert result.score == 0.0

    def test_fails_with_empty_error_string(self):
        # Empty string is still an error being set
        result = check_no_error("")
        assert result.passed is False

    def test_error_message_in_reason(self):
        result = check_no_error("API rate limit exceeded")
        assert "API rate limit exceeded" in result.reason


# ─── check_output_length ─────────────────────────────────────────────────────

class TestOutputLength:
    def test_passes_within_range(self):
        result = check_output_length("This is a valid output with enough words here", 5, 500)
        assert result.passed is True
        assert result.score == 1.0

    def test_fails_too_short(self):
        result = check_output_length("Hi", min_words=5, max_words=500)
        assert result.passed is False
        assert result.score == 0.5

    def test_fails_too_long(self):
        long_text = " ".join(["word"] * 600)
        result = check_output_length(long_text, min_words=5, max_words=500)
        assert result.passed is False

    def test_passes_at_minimum(self):
        result = check_output_length("one two three four five", min_words=5, max_words=500)
        assert result.passed is True

    def test_passes_single_word_at_min_1(self):
        result = check_output_length("Paris", min_words=1, max_words=500)
        assert result.passed is True

    def test_word_count_in_reason(self):
        result = check_output_length("hello world", min_words=5, max_words=500)
        assert "2" in result.reason


# ─── check_no_refusal ────────────────────────────────────────────────────────

class TestNoRefusal:
    def test_passes_normal_response(self):
        result = check_no_refusal("The capital of France is Paris.")
        assert result.passed is True
        assert result.score == 1.0

    def test_fails_i_cannot(self):
        result = check_no_refusal("I cannot answer that question.")
        assert result.passed is False
        assert result.score == 0.0

    def test_fails_i_cant(self):
        result = check_no_refusal("I can't help with that.")
        assert result.passed is False

    def test_fails_as_an_ai(self):
        result = check_no_refusal("As an AI, I don't have opinions.")
        assert result.passed is False

    def test_fails_i_am_unable(self):
        result = check_no_refusal("I am unable to provide that information.")
        assert result.passed is False

    def test_passes_can_in_positive_context(self):
        result = check_no_refusal("I can help you with that. Here is the answer.")
        assert result.passed is True

    def test_case_insensitive(self):
        result = check_no_refusal("I CANNOT do that.")
        assert result.passed is False


# ─── check_json_format ───────────────────────────────────────────────────────

class TestJsonFormat:
    def test_passes_valid_json_object(self):
        result = check_json_format('{"key": "value"}')
        assert result.passed is True
        assert result.score == 1.0

    def test_passes_valid_json_array(self):
        result = check_json_format('[1, 2, 3]')
        assert result.passed is True

    def test_fails_invalid_json(self):
        result = check_json_format("This is not JSON")
        assert result.passed is False
        assert result.score == 0.0

    def test_fails_partial_json(self):
        result = check_json_format('{"key": ')
        assert result.passed is False

    def test_passes_nested_json(self):
        result = check_json_format('{"a": {"b": [1, 2, 3]}}')
        assert result.passed is True


# ─── run_default_rules ───────────────────────────────────────────────────────

class TestRunDefaultRules:
    def test_returns_five_rules(self):
        results = run_default_rules("valid output here with words", 500, None)
        assert len(results) == 5

    def test_all_pass_on_good_output(self):
        results = run_default_rules("This is a valid output with enough content.", 500, None)
        assert all(r.passed for r in results)

    def test_error_fails_no_error_rule(self):
        results = run_default_rules("some output", 500, "Connection error")
        no_error_result = next(r for r in results if r.metric == "no_error")
        assert no_error_result.passed is False

    def test_high_latency_fails_sla(self):
        results = run_default_rules("some output here today", 5000, None)
        sla_result = next(r for r in results if r.metric == "latency_sla")
        assert sla_result.passed is False

    def test_metrics_present(self):
        results = run_default_rules("output text with several words here", 500, None)
        metrics = {r.metric for r in results}
        assert "no_error" in metrics
        assert "output_not_empty" in metrics
        assert "latency_sla" in metrics
        assert "output_length" in metrics
        assert "no_refusal" in metrics
