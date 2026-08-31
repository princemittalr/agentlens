"""
AgentLens CI Evaluation Runner
Runs eval suite and exits with code 1 if regressions detected.
Used by GitHub Actions on every push.
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from tracer import Tracer, Step
from evaluator import evaluate_trace
from tracer.database import get_all_traces

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tracer = Tracer(project="agentlens-ci")

MODEL = "qwen/qwen3.8-27b"

# ─── Minimal CI Agent ─────────────────────────────────────────────────────────
# Lightweight version of react agent — no tools, just LLM calls
# Fast and cheap for CI runs

SYSTEM_PROMPT = """You are a helpful assistant. Answer questions clearly and completely.
Always provide context and explanation, not just a bare answer."""

@tracer.trace(agent="ci-agent", version="v1.0", model=MODEL)
def ci_agent(query: str, _trace=None) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=200,
        temperature=0.1,
    )
    output = response.choices[0].message.content
    if _trace:
        _trace.prompt_tokens = response.usage.prompt_tokens
        _trace.completion_tokens = response.usage.completion_tokens
        _trace.total_tokens = response.usage.total_tokens
        _trace.steps.append(Step(
            step_index=0,
            type="llm_call",
            input=query,
            output=output,
            latency_ms=0,
            tokens_used=response.usage.total_tokens,
        ))
    return output


# ─── CI Test Suite ────────────────────────────────────────────────────────────

CI_TESTS = [
    {
        "query": "What is the capital of France?",
        "known_facts": "The capital of France is Paris.",
        "rubric": "Did the agent correctly identify Paris as the capital of France?",
        "min_score": 0.7,
    },
    {
        "query": "What is 15 multiplied by 8?",
        "known_facts": "15 * 8 = 120",
        "rubric": "Did the agent correctly calculate 120?",
        "min_score": 0.7,
    },
    {
        "query": "Explain what a REST API is in simple terms.",
        "known_facts": "A REST API allows communication between systems over HTTP using standard methods like GET, POST, PUT, DELETE.",
        "rubric": "Did the agent explain REST API correctly with HTTP methods mentioned?",
        "min_score": 0.7,
    },
    {
        "query": "What are the three primary colors?",
        "known_facts": "The three primary colors are red, blue, and yellow.",
        "rubric": "Did the agent correctly identify red, blue, and yellow as primary colors?",
        "min_score": 0.7,
    },
    {
        "query": "What does CPU stand for?",
        "known_facts": "CPU stands for Central Processing Unit.",
        "rubric": "Did the agent correctly state CPU stands for Central Processing Unit?",
        "min_score": 0.7,
    },
]


def run_ci_evals():
    print("\n" + "="*60)
    print("  AgentLens CI — Evaluation Suite")
    print("="*60)

    results = []
    failures = []

    for i, tc in enumerate(CI_TESTS):
        print(f"\n[{i+1}/{len(CI_TESTS)}] {tc['query'][:55]}...")

        try:
            output = ci_agent(tc["query"])
            latest = get_all_traces()[0]

            eval_result = evaluate_trace(
                run_id=latest["run_id"],
                agent_name=latest["agent_name"],
                agent_version=latest["agent_version"],
                input_query=tc["query"],
                output=output,
                latency_ms=latest["total_latency_ms"],
                error=latest["error"],
                known_facts=tc["known_facts"],
                rubric=tc["rubric"],
                run_judge=True,
            )

            results.append(eval_result)

            if eval_result.overall_score < tc["min_score"]:
                failures.append({
                    "query": tc["query"],
                    "score": eval_result.overall_score,
                    "min_score": tc["min_score"],
                    "run_id": latest["run_id"],
                })

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failures.append({
                "query": tc["query"],
                "score": 0.0,
                "min_score": tc["min_score"],
                "error": str(e),
            })

    # ─── Summary ──────────────────────────────────────────────────────────────
    passed_count = len(results) - len(failures)
    avg_score = round(
        sum(r.overall_score for r in results) / len(results), 3
    ) if results else 0

    print(f"\n{'='*60}")
    print(f"  CI RESULTS")
    print(f"{'='*60}")
    print(f"  Tests run    : {len(CI_TESTS)}")
    print(f"  Passed       : {passed_count}")
    print(f"  Failed       : {len(failures)}")
    print(f"  Avg Score    : {avg_score}")

    if failures:
        print(f"\n  ❌ FAILURES:")
        for f in failures:
            print(f"    • {f['query'][:50]}")
            print(f"      Score: {f['score']} (min required: {f['min_score']})")

    # Write CI report JSON for GitHub Actions summary
    report = {
        "total": len(CI_TESTS),
        "passed": passed_count,
        "failed": len(failures),
        "avg_score": avg_score,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }

    report_path = os.path.join(
        os.path.dirname(__file__), "ci_report.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved to tests/ci_report.json")

    if failures:
        print(f"\n  🔴 CI FAILED — {len(failures)} test(s) below minimum score")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"\n  ✅ CI PASSED — all tests above minimum score")
        print(f"{'='*60}\n")
        sys.exit(0)


if __name__ == "__main__":
    run_ci_evals()
