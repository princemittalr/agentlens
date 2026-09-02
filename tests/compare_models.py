"""
AgentLens — Multi-model Comparison
Runs the same eval suite against multiple models and compares results.
Usage: python3 tests/compare_models.py
"""
import sys
import os
import time
sys.path.insert(0, "/home/prince-mittal/agentlens")

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/prince-mittal/agentlens/.env")

from groq import Groq
from tracer import Tracer, Step
from evaluator import evaluate_trace
from tracer.database import get_all_traces
from tracer.regression import compare_versions

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Models to compare ───────────────────────────────────────────────────────

MODELS = [
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
]

# ─── Test cases ───────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "query": "What is the capital of France?",
        "known_facts": "The capital of France is Paris.",
        "rubric": "Did the agent correctly identify Paris as the capital of France?",
    },
    {
        "query": "Explain what machine learning is in 2 sentences.",
        "known_facts": "Machine learning is a subset of AI where systems learn from data to make predictions.",
        "rubric": "Did the agent correctly explain machine learning as learning from data?",
    },
    {
        "query": "What is 17 multiplied by 13?",
        "known_facts": "17 * 13 = 221",
        "rubric": "Did the agent correctly calculate 221?",
    },
    {
        "query": "Name the three states of matter.",
        "known_facts": "The three states of matter are solid, liquid, and gas.",
        "rubric": "Did the agent correctly name solid, liquid, and gas?",
    },
    {
        "query": "What does API stand for?",
        "known_facts": "API stands for Application Programming Interface.",
        "rubric": "Did the agent correctly expand API as Application Programming Interface?",
    },
]

# ─── Agent factory ────────────────────────────────────────────────────────────

def make_agent(model: str, tracer: Tracer):
    """Create a traced agent for a specific model."""
    # Sanitize model name for use as agent version
    version = model.replace("/", "-").replace(".", "-")

    @tracer.trace(agent="model-comparison", version=version, model=model)
    def agent(query: str, _trace=None) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer clearly and completely."},
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

    return agent, version


# ─── Run comparison ───────────────────────────────────────────────────────────

def run_model_comparison():
    print("\n" + "="*65)
    print("  AgentLens — Multi-model Comparison")
    print("="*65)
    print(f"  Models : {', '.join(MODELS)}")
    print(f"  Tests  : {len(TEST_CASES)} cases per model")
    print("="*65)

    tracer = Tracer(project="model-comparison")
    model_results = {}

    for model in MODELS:
        print(f"\n\n{'─'*65}")
        print(f"  Running: {model}")
        print(f"{'─'*65}")

        agent, version = make_agent(model, tracer)
        scores = []
        latencies = []
        tokens_list = []
        costs = []

        for i, tc in enumerate(TEST_CASES):
            print(f"\n  [{i+1}/{len(TEST_CASES)}] {tc['query'][:55]}...")

            try:
                output = agent(tc["query"])
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

                scores.append(eval_result.overall_score)
                latencies.append(latest["total_latency_ms"])
                tokens_list.append(latest["total_tokens"])
                costs.append(latest["cost_usd"])

            except Exception as e:
                print(f"  ❌ Error: {e}")
                scores.append(0.0)

        model_results[model] = {
            "version": version,
            "scores": scores,
            "avg_score": round(sum(scores) / len(scores), 3),
            "pass_rate": round(sum(1 for s in scores if s >= 0.7) / len(scores) * 100, 1),
            "avg_latency": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "avg_tokens": round(sum(tokens_list) / len(tokens_list), 1) if tokens_list else 0,
            "total_cost": round(sum(costs), 6),
        }

    # ─── Side-by-side comparison table ───────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  MULTI-MODEL COMPARISON RESULTS")
    print(f"{'='*65}")

    metrics = ["avg_score", "pass_rate", "avg_latency", "avg_tokens", "total_cost"]
    labels = {
        "avg_score": "Avg Score",
        "pass_rate": "Pass Rate %",
        "avg_latency": "Avg Latency (ms)",
        "avg_tokens": "Avg Tokens",
        "total_cost": "Total Cost ($)",
    }

    # Header
    header = f"  {'Metric':<20}"
    for model in MODELS:
        short = model.split("/")[-1][:15]
        header += f"  {short:>15}"
    print(header)
    print(f"  {'─'*60}")

    # Rows
    for metric in metrics:
        row = f"  {labels[metric]:<20}"
        values = [model_results[m][metric] for m in MODELS]

        for i, (model, val) in enumerate(zip(MODELS, values)):
            # Highlight best value
            if metric in ["avg_score", "pass_rate"]:
                is_best = val == max(values)
            else:
                is_best = val == min(values)

            marker = " ✓" if is_best else "  "
            row += f"  {str(val):>13}{marker}"
        print(row)

    # Winner summary
    print(f"\n  {'─'*60}")
    best_quality = max(MODELS, key=lambda m: model_results[m]["avg_score"])
    best_speed = min(MODELS, key=lambda m: model_results[m]["avg_latency"])
    best_cost = min(MODELS, key=lambda m: model_results[m]["total_cost"])

    print(f"  🏆 Best quality : {best_quality}")
    print(f"  ⚡ Fastest      : {best_speed}")
    print(f"  💰 Cheapest     : {best_cost}")

    # Per-test breakdown
    print(f"\n  Per-test scores:")
    print(f"  {'Query':<35}", end="")
    for model in MODELS:
        short = model.split("/")[-1][:12]
        print(f"  {short:>12}", end="")
    print()
    print(f"  {'─'*60}")

    for i, tc in enumerate(TEST_CASES):
        print(f"  {tc['query'][:33]:<35}", end="")
        for model in MODELS:
            score = model_results[model]["scores"][i]
            icon = "✅" if score >= 0.7 else "❌"
            print(f"  {icon} {score:>8.3f}", end="")
        print()

    print(f"\n{'='*65}\n")

    # Run regression detection between models
    if len(MODELS) == 2:
        print(f"\nRunning regression detection: {MODELS[0]} vs {MODELS[1]}...")
        try:
            v1 = model_results[MODELS[0]]["version"]
            v2 = model_results[MODELS[1]]["version"]
            compare_versions("model-comparison", v1, v2)
        except Exception as e:
            print(f"Regression comparison error: {e}")

    return model_results


if __name__ == "__main__":
    run_model_comparison()
