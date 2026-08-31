"""
Unit tests for tracer/regression.py
Tests metric summarization and regression detection logic.
"""
import sys
sys.path.insert(0, "/home/prince-mittal/agentlens")

import pytest
from tracer.regression import (
    summarize_metrics,
    detect_regressions,
    MetricSummary,
)


# ─── summarize_metrics ───────────────────────────────────────────────────────

class TestSummarizeMetrics:
    def test_basic_mean(self):
        scores = {"overall_score": [0.8, 0.9, 0.7]}
        result = summarize_metrics(scores)
        assert result["overall_score"].mean == pytest.approx(0.8, rel=1e-3)

    def test_pass_rate_above_threshold(self):
        scores = {"task_success": [1.0, 1.0, 0.0]}
        result = summarize_metrics(scores)
        # 2 out of 3 above 0.7
        assert result["task_success"].pass_rate == pytest.approx(0.6667, rel=1e-2)

    def test_perfect_pass_rate(self):
        scores = {"coherence": [1.0, 1.0, 1.0]}
        result = summarize_metrics(scores)
        assert result["coherence"].pass_rate == 1.0

    def test_zero_pass_rate(self):
        scores = {"hallucination": [0.0, 0.0, 0.0]}
        result = summarize_metrics(scores)
        assert result["hallucination"].pass_rate == 0.0

    def test_sample_count(self):
        scores = {"latency_ms": [100, 200, 300, 400]}
        result = summarize_metrics(scores)
        assert result["latency_ms"].sample_count == 4

    def test_system_metrics_have_full_pass_rate(self):
        scores = {"latency_ms": [5000, 6000, 7000]}
        result = summarize_metrics(scores)
        # System metrics always pass_rate=1.0
        assert result["latency_ms"].pass_rate == 1.0

    def test_empty_scores_skipped(self):
        scores = {"metric_a": [], "metric_b": [0.5]}
        result = summarize_metrics(scores)
        assert "metric_a" not in result
        assert "metric_b" in result

    def test_multiple_metrics(self):
        scores = {
            "overall_score": [0.9, 0.8],
            "coherence": [1.0, 0.5],
            "latency_ms": [300, 400],
        }
        result = summarize_metrics(scores)
        assert len(result) == 3


# ─── detect_regressions ──────────────────────────────────────────────────────

def make_summary(metric: str, mean: float, pass_rate: float = 1.0) -> MetricSummary:
    return MetricSummary(metric=metric, mean=mean, pass_rate=pass_rate, sample_count=4)


class TestDetectRegressions:
    def test_no_regression_when_stable(self):
        baseline = {"overall_score": make_summary("overall_score", 0.9)}
        candidate = {"overall_score": make_summary("overall_score", 0.9)}
        alerts, improvements = detect_regressions(baseline, candidate)
        assert len(alerts) == 0
        assert len(improvements) == 0

    def test_critical_regression_on_coherence(self):
        baseline = {"coherence": make_summary("coherence", 0.9)}
        candidate = {"coherence": make_summary("coherence", 0.0)}
        alerts, _ = detect_regressions(baseline, candidate)
        assert any(a.metric == "coherence" and a.severity == "critical" for a in alerts)

    def test_warning_regression(self):
        baseline = {"overall_score": make_summary("overall_score", 1.0)}
        candidate = {"overall_score": make_summary("overall_score", 0.93)}
        alerts, _ = detect_regressions(baseline, candidate)
        # -7% should be a warning
        assert any(a.metric == "overall_score" and a.severity == "warning" for a in alerts)

    def test_improvement_detected(self):
        baseline = {"task_success": make_summary("task_success", 0.7)}
        candidate = {"task_success": make_summary("task_success", 0.9)}
        _, improvements = detect_regressions(baseline, candidate)
        assert any(a.metric == "task_success" for a in improvements)

    def test_latency_regression_higher_is_worse(self):
        baseline = {"latency_ms": make_summary("latency_ms", 500)}
        candidate = {"latency_ms": make_summary("latency_ms", 1000)}
        alerts, _ = detect_regressions(baseline, candidate)
        # +100% latency should be critical
        assert any(a.metric == "latency_ms" and a.severity == "critical" for a in alerts)

    def test_latency_improvement(self):
        baseline = {"latency_ms": make_summary("latency_ms", 1000)}
        candidate = {"latency_ms": make_summary("latency_ms", 400)}
        _, improvements = detect_regressions(baseline, candidate)
        assert any(a.metric == "latency_ms" for a in improvements)

    def test_skips_zero_baseline(self):
        baseline = {"cost_usd": make_summary("cost_usd", 0.0)}
        candidate = {"cost_usd": make_summary("cost_usd", 0.001)}
        alerts, improvements = detect_regressions(baseline, candidate)
        # Should skip when baseline is 0 (can't compute percentage)
        assert len(alerts) == 0

    def test_metric_missing_in_candidate_skipped(self):
        baseline = {"hallucination": make_summary("hallucination", 1.0)}
        candidate = {}
        alerts, improvements = detect_regressions(baseline, candidate)
        assert len(alerts) == 0

    def test_delta_calculation(self):
        baseline = {"overall_score": make_summary("overall_score", 1.0)}
        candidate = {"overall_score": make_summary("overall_score", 0.8)}
        alerts, _ = detect_regressions(baseline, candidate)
        assert len(alerts) > 0
        assert alerts[0].delta == pytest.approx(-0.2, rel=1e-3)
        assert alerts[0].delta_pct == pytest.approx(-20.0, rel=1e-3)


# ─── Integration: full regression scenario ───────────────────────────────────

class TestRegressionScenario:
    def test_v1_vs_degraded_v2(self):
        """Simulates the real v1.0 → v2.0 regression we observed."""
        baseline = {
            "coherence": make_summary("coherence", 0.5),
            "hallucination": make_summary("hallucination", 1.0),
            "overall_score": make_summary("overall_score", 0.89),
            "latency_ms": make_summary("latency_ms", 1463),
        }
        candidate = {
            "coherence": make_summary("coherence", 0.0),
            "hallucination": make_summary("hallucination", 0.75),
            "overall_score": make_summary("overall_score", 0.77),
            "latency_ms": make_summary("latency_ms", 506),
        }
        alerts, improvements = detect_regressions(baseline, candidate)

        # Should detect coherence regression
        assert any(a.metric == "coherence" for a in alerts)
        # Should detect hallucination regression
        assert any(a.metric == "hallucination" for a in alerts)
        # Should detect latency improvement
        assert any(a.metric == "latency_ms" for a in improvements)
