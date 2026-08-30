import sys
import os
sys.path.insert(0, "/home/prince-mittal/agentlens")

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/prince-mittal/agentlens/.env")

from groq import Groq
from tracer import Tracer, Step
from evaluator import evaluate_trace
from tracer.regression import compare_versions

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tracer = Tracer(project="agentlens")

# ─── Degraded v2 agent (bad system prompt — simulates a bad prompt change) ───

BAD_SYSTEM_PROMPT = """Answer questions. Be very brief. One word only."""

@tracer.trace(agent="react-agent", version="v2.0", model="qwen/qwen3.8-27b")
def react_agent_v2(query: str, _trace=None) -> str:
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": BAD_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=50,
        temperature=0.0,
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


TEST_CASES = [
    {
        "query": "What is the capital of France and how tall is the Eiffel Tower?",
        "known_facts": "Paris is the capital of France. The Eiffel Tower is 330 meters tall.",
        "rubric": "Did the agent correctly state Paris as capital and 330m as Eiffel Tower height?",
    },
    {
        "query": "What is 25 multiplied by 48?",
        "known_facts": "25 * 48 = 1200",
        "rubric": "Did the agent correctly calculate 1200?",
    },
    {
        "query": "Who founded Anthropic and when was it founded?",
        "known_facts": "Anthropic is an AI safety company founded in 2021.",
        "rubric": "Did the agent mention Anthropic was founded in 2021?",
    },
    {
        "query": "What is the population of Mars?",
        "known_facts": "Mars has no human population. It is uninhabited.",
        "rubric": "Did the agent correctly state Mars has no human population?",
    },
]


def run_v2_evals():
    print("\n" + "="*60)
    print("  Running degraded v2.0 agent evals...")
    print("="*60)

    for i, tc in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}] {tc['query'][:55]}...")
        output = react_agent_v2(tc["query"])

        from tracer.database import get_all_traces
        latest = get_all_traces()[0]

        evaluate_trace(
            run_id=latest["run_id"],
            agent_name=latest["agent_name"],
            agent_version=latest["agent_version"],
            input_query=tc["query"],
            output=output,
            latency_ms=latest["total_latency_ms"],
            error=latest["error"],
            known_facts=tc["known_facts"],
            rubric=tc["rubric"],
        )


if __name__ == "__main__":
    # Step 1 — Run degraded v2 evals
    run_v2_evals()

    # Step 2 — Compare v1.0 (baseline) vs v2.0 (candidate)
    print("\nRunning regression detection: v1.0 → v2.0")
    report = compare_versions(
        agent_name="react-agent",
        baseline_version="v1.0",
        candidate_version="v2.0",
    )
