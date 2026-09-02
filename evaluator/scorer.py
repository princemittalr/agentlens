import json
import sqlite3
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Optional
from .rules import run_default_rules, RuleResult
from .llm_judge import run_judges_parallel, JudgeResult

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "agentlens.db")


@dataclass
class EvalResult:
    run_id: str
    agent_name: str
    rule_results: List[RuleResult]
    judge_results: List[JudgeResult]
    overall_score: float
    passed: bool


def _ensure_eval_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            run_id TEXT PRIMARY KEY,
            agent_name TEXT,
            agent_version TEXT,
            rule_results TEXT,
            judge_results TEXT,
            overall_score REAL,
            passed INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def _save_eval(eval_result: EvalResult, agent_version: str):
    from datetime import datetime
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        eval_result.run_id,
        eval_result.agent_name,
        agent_version,
        json.dumps([asdict(r) for r in eval_result.rule_results]),
        json.dumps([asdict(r) for r in eval_result.judge_results]),
        eval_result.overall_score,
        int(eval_result.passed),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()


def evaluate_trace(
    run_id: str,
    agent_name: str,
    agent_version: str,
    input_query: str,
    output: str,
    latency_ms: float,
    error: Optional[str] = None,
    context: Optional[str] = None,
    known_facts: Optional[str] = None,
    rubric: Optional[str] = None,
    run_judge: bool = True,
) -> EvalResult:
    _ensure_eval_table()

    # Rule-based (instant)
    rule_results = run_default_rules(output, latency_ms, error)

    # LLM judges — now parallel
    judge_results = []
    if run_judge and output:
        print("  Running LLM judges (parallel)...")
        t0 = time.time()
        judge_results = run_judges_parallel(
            input_query=input_query,
            output=output,
            rubric=rubric or "",
            context=context,
            known_facts=known_facts,
        )
        elapsed = round(time.time() - t0, 2)
        print(f"  Judges completed in {elapsed}s ({len(judge_results)} metrics parallel)")

    # Overall score
    all_scores = [r.score for r in rule_results] + [r.score for r in judge_results]
    overall_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    passed = overall_score >= 0.7

    result = EvalResult(
        run_id=run_id,
        agent_name=agent_name,
        rule_results=rule_results,
        judge_results=judge_results,
        overall_score=overall_score,
        passed=passed
    )

    _save_eval(result, agent_version)
    _print_eval_summary(result)
    return result


def _print_eval_summary(result: EvalResult):
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    print(f"\n{'='*50}")
    print(f"AgentLens Eval [{status}] — Score: {result.overall_score}")
    print(f"  Run ID: {result.run_id}")
    print(f"\n  Rule-based metrics:")
    for r in result.rule_results:
        icon = "✅" if r.passed else "❌"
        print(f"    {icon} {r.metric}: {r.score} — {r.reason}")
    if result.judge_results:
        print(f"\n  LLM Judge metrics:")
        for r in result.judge_results:
            icon = "✅" if r.passed else "❌"
            print(f"    {icon} {r.metric}: {r.score} — {r.reasoning[:80]}...")
    print(f"{'='*50}\n")
