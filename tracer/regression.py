import json
import sqlite3
import os
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "agentlens.db")


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class MetricSummary:
    metric: str
    mean: float
    pass_rate: float
    sample_count: int


@dataclass
class RegressionAlert:
    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    delta_pct: float
    severity: str        # "critical", "warning", "ok"
    message: str


@dataclass
class RegressionReport:
    agent_name: str
    baseline_version: str
    candidate_version: str
    baseline_run_count: int
    candidate_run_count: int
    alerts: List[RegressionAlert]
    improved: List[RegressionAlert]
    timestamp: str
    verdict: str         # "REGRESSION", "IMPROVEMENT", "STABLE"


# ─── Data Fetcher ─────────────────────────────────────────────────────────────

def fetch_evals_for_version(agent_name: str, version: str) -> List[dict]:
    """Fetch all evaluations for a specific agent + version."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.*, t.total_latency_ms, t.total_tokens, t.cost_usd
        FROM evaluations e
        JOIN traces t ON e.run_id = t.run_id
        WHERE e.agent_name = ? AND e.agent_version = ?
        ORDER BY e.timestamp DESC
    """, (agent_name, version))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def extract_metric_scores(evals: List[dict]) -> Dict[str, List[float]]:
    """Extract per-metric scores from eval records."""
    metric_scores = {}

    for ev in evals:
        # Rule-based metrics
        rules = json.loads(ev.get("rule_results", "[]"))
        for r in rules:
            m = r["metric"]
            metric_scores.setdefault(m, []).append(r["score"])

        # LLM judge metrics
        judges = json.loads(ev.get("judge_results", "[]"))
        for j in judges:
            m = j["metric"]
            metric_scores.setdefault(m, []).append(j["score"])

        # System metrics
        metric_scores.setdefault("latency_ms", []).append(ev.get("total_latency_ms", 0))
        metric_scores.setdefault("total_tokens", []).append(ev.get("total_tokens", 0))
        metric_scores.setdefault("cost_usd", []).append(ev.get("cost_usd", 0))
        metric_scores.setdefault("overall_score", []).append(ev.get("overall_score", 0))

    return metric_scores


def summarize_metrics(metric_scores: Dict[str, List[float]]) -> Dict[str, MetricSummary]:
    """Compute mean and pass rate for each metric."""
    summaries = {}
    for metric, scores in metric_scores.items():
        if not scores:
            continue
        mean = round(sum(scores) / len(scores), 4)
        # Pass rate = % of scores >= 0.7 (not applicable for latency/tokens/cost)
        system_metrics = {"latency_ms", "total_tokens", "cost_usd"}
        if metric in system_metrics:
            pass_rate = 1.0
        else:
            pass_rate = round(sum(1 for s in scores if s >= 0.7) / len(scores), 4)
        summaries[metric] = MetricSummary(
            metric=metric,
            mean=mean,
            pass_rate=pass_rate,
            sample_count=len(scores)
        )
    return summaries


# ─── Regression Detector ──────────────────────────────────────────────────────

# Thresholds for regression detection
REGRESSION_THRESHOLDS = {
    # Quality metrics — lower is worse
    "overall_score":   {"warn": -0.05, "critical": -0.10, "direction": "lower_is_worse"},
    "task_success":    {"warn": -0.05, "critical": -0.15, "direction": "lower_is_worse"},
    "hallucination":   {"warn": -0.05, "critical": -0.15, "direction": "lower_is_worse"},
    "groundedness":    {"warn": -0.05, "critical": -0.15, "direction": "lower_is_worse"},
    "coherence":       {"warn": -0.05, "critical": -0.10, "direction": "lower_is_worse"},
    "no_error":        {"warn": -0.05, "critical": -0.10, "direction": "lower_is_worse"},
    "no_refusal":      {"warn": -0.05, "critical": -0.10, "direction": "lower_is_worse"},
    "output_length":   {"warn": -0.10, "critical": -0.20, "direction": "lower_is_worse"},
    # System metrics — higher is worse
    "latency_ms":      {"warn": 0.20,  "critical": 0.50,  "direction": "higher_is_worse"},
    "total_tokens":    {"warn": 0.20,  "critical": 0.50,  "direction": "higher_is_worse"},
    "cost_usd":        {"warn": 0.20,  "critical": 0.50,  "direction": "higher_is_worse"},
}


def detect_regressions(
    baseline: Dict[str, MetricSummary],
    candidate: Dict[str, MetricSummary]
) -> tuple:
    """Compare baseline vs candidate metrics. Returns (alerts, improvements)."""
    alerts = []
    improvements = []

    all_metrics = set(baseline.keys()) | set(candidate.keys())

    for metric in all_metrics:
        if metric not in baseline or metric not in candidate:
            continue

        b_mean = baseline[metric].mean
        c_mean = candidate[metric].mean

        if b_mean == 0:
            continue

        delta = round(c_mean - b_mean, 4)
        delta_pct = round((delta / abs(b_mean)) * 100, 2)

        config = REGRESSION_THRESHOLDS.get(metric, {
            "warn": -0.05, "critical": -0.15, "direction": "lower_is_worse"
        })

        direction = config["direction"]
        warn_threshold = config["warn"]
        critical_threshold = config["critical"]

        if direction == "lower_is_worse":
            # Negative delta = regression
            if delta_pct <= critical_threshold * 100:
                severity = "critical"
            elif delta_pct <= warn_threshold * 100:
                severity = "warning"
            elif delta_pct > 5:
                severity = "improvement"
            else:
                severity = "ok"
        else:
            # higher_is_worse: positive delta = regression
            if delta_pct >= critical_threshold * 100:
                severity = "critical"
            elif delta_pct >= warn_threshold * 100:
                severity = "warning"
            elif delta_pct < -5:
                severity = "improvement"
            else:
                severity = "ok"

        alert = RegressionAlert(
            metric=metric,
            baseline_mean=b_mean,
            candidate_mean=c_mean,
            delta=delta,
            delta_pct=delta_pct,
            severity=severity,
            message=_build_message(metric, b_mean, c_mean, delta_pct, severity)
        )

        if severity in ("critical", "warning"):
            alerts.append(alert)
        elif severity == "improvement":
            improvements.append(alert)

    return alerts, improvements


def _build_message(metric, baseline, candidate, delta_pct, severity) -> str:
    direction = "↓ decreased" if candidate < baseline else "↑ increased"
    return f"{metric}: {direction} by {abs(delta_pct):.1f}% ({baseline:.4f} → {candidate:.4f}) [{severity.upper()}]"


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def compare_versions(
    agent_name: str,
    baseline_version: str,
    candidate_version: str,
    verbose: bool = True
) -> RegressionReport:
    """Full regression comparison between two versions of an agent."""

    baseline_evals = fetch_evals_for_version(agent_name, baseline_version)
    candidate_evals = fetch_evals_for_version(agent_name, candidate_version)

    if not baseline_evals:
        raise ValueError(f"No evaluations found for {agent_name} version {baseline_version}")
    if not candidate_evals:
        raise ValueError(f"No evaluations found for {agent_name} version {candidate_version}")

    baseline_scores = extract_metric_scores(baseline_evals)
    candidate_scores = extract_metric_scores(candidate_evals)

    baseline_summary = summarize_metrics(baseline_scores)
    candidate_summary = summarize_metrics(candidate_scores)

    alerts, improvements = detect_regressions(baseline_summary, candidate_summary)

    # Overall verdict
    has_critical = any(a.severity == "critical" for a in alerts)
    has_warning = any(a.severity == "warning" for a in alerts)
    if has_critical:
        verdict = "REGRESSION 🔴"
    elif has_warning:
        verdict = "WARNING 🟡"
    elif improvements:
        verdict = "IMPROVEMENT 🟢"
    else:
        verdict = "STABLE ✅"

    report = RegressionReport(
        agent_name=agent_name,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        baseline_run_count=len(baseline_evals),
        candidate_run_count=len(candidate_evals),
        alerts=alerts,
        improved=improvements,
        timestamp=datetime.utcnow().isoformat(),
        verdict=verdict,
    )

    if verbose:
        _print_report(report, baseline_summary, candidate_summary)

    return report


def _print_report(
    report: RegressionReport,
    baseline: Dict[str, MetricSummary],
    candidate: Dict[str, MetricSummary]
):
    print(f"\n{'='*65}")
    print(f"  AgentLens Regression Report")
    print(f"{'='*65}")
    print(f"  Agent     : {report.agent_name}")
    print(f"  Baseline  : {report.baseline_version} ({report.baseline_run_count} runs)")
    print(f"  Candidate : {report.candidate_version} ({report.candidate_run_count} runs)")
    print(f"  Verdict   : {report.verdict}")
    print(f"  Timestamp : {report.timestamp}")

    print(f"\n  {'Metric':<20} {'Baseline':>10} {'Candidate':>10} {'Delta%':>10} {'Status':>12}")
    print(f"  {'-'*64}")

    all_metrics = set(baseline.keys()) | set(candidate.keys())
    for metric in sorted(all_metrics):
        b = baseline.get(metric)
        c = candidate.get(metric)
        if not b or not c:
            continue
        delta_pct = round(((c.mean - b.mean) / abs(b.mean)) * 100, 1) if b.mean != 0 else 0

        alert_map = {a.metric: a.severity for a in report.alerts}
        imp_map = {a.metric: "improvement" for a in report.improved}

        status = alert_map.get(metric, imp_map.get(metric, "ok"))
        icon = {"critical": "🔴", "warning": "🟡", "improvement": "🟢", "ok": "✅"}.get(status, "")

        print(f"  {metric:<20} {b.mean:>10.4f} {c.mean:>10.4f} {delta_pct:>9.1f}% {icon:>8} {status}")

    if report.alerts:
        print(f"\n  ⚠️  Regressions detected ({len(report.alerts)}):")
        for a in report.alerts:
            print(f"    • {a.message}")

    if report.improved:
        print(f"\n  ✨ Improvements detected ({len(report.improved)}):")
        for a in report.improved:
            print(f"    • {a.message}")

    print(f"{'='*65}\n")
