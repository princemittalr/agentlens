import os
import sys
import time
import json
sys.path.insert(0, "/home/prince-mittal/agentlens")

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/prince-mittal/agentlens/.env")

from groq import Groq
from tracer import Tracer, Step
from evaluator import evaluate_trace

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tracer = Tracer(project="agentlens")

# ─── Tool Registry ────────────────────────────────────────────────────────────

def tool_summarize(text: str) -> str:
    """Summarize a block of text using LLM."""
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": "Summarize the following text in 2-3 sentences."},
            {"role": "user", "content": text},
        ],
        max_tokens=150,
        temperature=0.1,
    )
    return response.choices[0].message.content


def tool_extract_keywords(text: str) -> str:
    """Extract keywords from text."""
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": "Extract 5 key topics or keywords from this text. Return as comma-separated list only."},
            {"role": "user", "content": text},
        ],
        max_tokens=100,
        temperature=0.1,
    )
    return response.choices[0].message.content


def tool_classify_sentiment(text: str) -> str:
    """Classify sentiment of text."""
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": "Classify the sentiment of this text. Reply with only: POSITIVE, NEGATIVE, or NEUTRAL."},
            {"role": "user", "content": text},
        ],
        max_tokens=10,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def tool_translate(text: str) -> str:
    """Translate text to French."""
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": "Translate the following text to French. Return only the translation."},
            {"role": "user", "content": text},
        ],
        max_tokens=200,
        temperature=0.1,
    )
    return response.choices[0].message.content


def tool_broken_json_parser(text: str) -> str:
    """Intentionally broken tool — simulates a real pipeline failure."""
    # This tool always fails — to demonstrate error cascade detection
    raise ValueError(f"JSON parse error: unexpected token at position 0 in: '{text[:30]}...'")


TOOLS = {
    "summarize": tool_summarize,
    "extract_keywords": tool_extract_keywords,
    "classify_sentiment": tool_classify_sentiment,
    "translate": tool_translate,
    "parse_json": tool_broken_json_parser,  # intentionally broken
}

# ─── Planner ─────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a task planner. Given a user goal, break it into a sequence of steps.
Each step must use one of these tools: summarize, extract_keywords, classify_sentiment, translate, parse_json.

Respond ONLY with a JSON array of steps like:
[
  {"step": 1, "tool": "tool_name", "input": "what to pass to the tool"},
  {"step": 2, "tool": "tool_name", "input": "use previous result if needed"}
]

Return ONLY the JSON array, no explanation."""


def generate_plan(goal: str) -> list:
    """Ask LLM to create a step-by-step plan."""
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Goal: {goal}"},
        ],
        max_tokens=400,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    return json.loads(raw)


@tracer.trace(agent="planner-agent", version="v1.0", model="qwen/qwen3.8-27b")
def planner_agent(goal: str, input_text: str, _trace=None) -> str:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    step_results = {}
    final_output = ""
    errors = []

    # Step 0 — Generate plan
    plan_start = time.time()
    try:
        plan = generate_plan(goal)
        plan_latency = round((time.time() - plan_start) * 1000, 2)

        if _trace:
            _trace.steps.append(Step(
                step_index=0,
                type="llm_call",
                input=goal,
                output=json.dumps(plan),
                latency_ms=plan_latency,
                tokens_used=0,
                tool_name="planner",
            ))
    except Exception as e:
        if _trace:
            _trace.prompt_tokens = total_prompt_tokens
            _trace.completion_tokens = total_completion_tokens
            _trace.total_tokens = total_prompt_tokens + total_completion_tokens
        return f"Planning failed: {e}"

    # Execute each step
    for plan_step in plan:
        step_num = plan_step.get("step", 0)
        tool_name = plan_step.get("tool", "")
        tool_input = plan_step.get("input", input_text)

        # Replace placeholder with actual input text
        if "previous" in tool_input.lower() or "result" in tool_input.lower():
            tool_input = step_results.get(step_num - 1, input_text)

        step_start = time.time()
        tool_fn = TOOLS.get(tool_name)

        if not tool_fn:
            error_msg = f"Unknown tool: {tool_name}"
            errors.append(error_msg)
            step_results[step_num] = error_msg
            if _trace:
                _trace.steps.append(Step(
                    step_index=step_num,
                    type="tool_call",
                    input=tool_input,
                    output=error_msg,
                    latency_ms=0,
                    tokens_used=0,
                    tool_name=tool_name,
                    error=error_msg,
                ))
            continue

        try:
            result = tool_fn(tool_input)
            step_latency = round((time.time() - step_start) * 1000, 2)
            step_results[step_num] = result
            final_output = result  # last successful step = final output

            if _trace:
                _trace.steps.append(Step(
                    step_index=step_num,
                    type="tool_call",
                    input=tool_input[:200],
                    output=result[:200],
                    latency_ms=step_latency,
                    tokens_used=0,
                    tool_name=tool_name,
                ))

        except Exception as e:
            step_latency = round((time.time() - step_start) * 1000, 2)
            error_msg = f"Tool '{tool_name}' failed: {e}"
            errors.append(error_msg)
            step_results[step_num] = ""

            # ── Error cascade — downstream steps get empty input ──
            if _trace:
                _trace.steps.append(Step(
                    step_index=step_num,
                    type="tool_call",
                    input=tool_input[:200],
                    output="",
                    latency_ms=step_latency,
                    tokens_used=0,
                    tool_name=tool_name,
                    error=error_msg,
                ))

    if _trace:
        _trace.prompt_tokens = total_prompt_tokens
        _trace.completion_tokens = total_completion_tokens
        _trace.total_tokens = total_prompt_tokens + total_completion_tokens

    if errors and not final_output:
        return f"Pipeline failed. Errors: {'; '.join(errors)}"

    if errors:
        return f"{final_output}\n\n[WARNING: {len(errors)} step(s) failed: {'; '.join(errors)}]"

    return final_output


# ─── Test Suite ──────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
AgentLens is a production-grade observability platform designed for LLM agent pipelines.
It automatically instruments any agent, captures detailed execution traces, and runs both
rule-based and LLM-judge evaluations. The platform detects regressions across prompt versions
and visualizes failure clusters, making it invaluable for teams shipping AI products at scale.
Built with FastAPI, SQLite, and Groq, it represents a significant step forward in LLM ops tooling.
"""

TEST_CASES = [
    {
        "goal": "Summarize the text, then extract keywords from the summary",
        "input_text": SAMPLE_TEXT,
        "known_facts": "AgentLens is an observability platform for LLM agents with tracing, evaluation, and regression detection.",
        "rubric": "Did the agent produce a summary and keywords related to AgentLens observability platform?",
    },
    {
        "goal": "Classify the sentiment of the text, then translate it to French",
        "input_text": SAMPLE_TEXT,
        "known_facts": "The text is positive in sentiment. A French translation should be provided.",
        "rubric": "Did the agent classify sentiment as POSITIVE and provide a French translation?",
    },
    {
        "goal": "Parse the text as JSON, then summarize it",
        "input_text": SAMPLE_TEXT,
        "known_facts": "The parse_json tool is broken and will fail. The agent should handle this gracefully.",
        "rubric": "Did the agent report the JSON parse failure clearly without crashing entirely?",
    },
    {
        "goal": "Extract keywords from the text, classify their sentiment, then summarize",
        "input_text": SAMPLE_TEXT,
        "known_facts": "Keywords about AgentLens should be extracted. Sentiment should be classified. A final summary produced.",
        "rubric": "Did the agent complete all three steps: extract keywords, classify sentiment, and summarize?",
    },
]


def run_planner_eval():
    print("\n" + "="*60)
    print("  AgentLens — Planner Agent Evaluation Suite")
    print("="*60)

    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}] {tc['goal'][:60]}...")

        output = planner_agent(tc["goal"], tc["input_text"])

        from tracer.database import get_all_traces
        latest = get_all_traces()[0]

        eval_result = evaluate_trace(
            run_id=latest["run_id"],
            agent_name=latest["agent_name"],
            agent_version=latest["agent_version"],
            input_query=tc["goal"],
            output=output,
            latency_ms=latest["total_latency_ms"],
            error=latest["error"],
            known_facts=tc["known_facts"],
            rubric=tc["rubric"],
        )
        results.append(eval_result)

    # Summary
    passed = sum(1 for r in results if r.passed)
    avg_score = round(sum(r.overall_score for r in results) / len(results), 3)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: {passed}/{len(results)} tests passed")
    print(f"  Avg Score    : {avg_score}")

    print(f"\n  Per-metric failure analysis:")
    metric_failures = {}
    for r in results:
        for rule in r.rule_results:
            if not rule.passed:
                metric_failures[rule.metric] = metric_failures.get(rule.metric, 0) + 1
        for judge in r.judge_results:
            if not judge.passed:
                metric_failures[judge.metric] = metric_failures.get(judge.metric, 0) + 1

    if metric_failures:
        for metric, count in sorted(metric_failures.items(), key=lambda x: -x[1]):
            print(f"    ❌ {metric}: failed {count}/{len(TEST_CASES)} times")
    else:
        print("    ✅ No metric failures detected")

    print(f"\n  Step-level error analysis:")
    from tracer.database import get_all_traces
    all_traces = get_all_traces()
    planner_traces = [t for t in all_traces if t["agent_name"] == "planner-agent"][:len(TEST_CASES)]

    total_steps = 0
    failed_steps = 0
    for t in planner_traces:
        steps = json.loads(t["steps"])
        total_steps += len(steps)
        failed_steps += sum(1 for s in steps if s.get("error"))

    print(f"    Total steps executed : {total_steps}")
    print(f"    Failed steps         : {failed_steps}")
    print(f"    Step success rate    : {round((total_steps - failed_steps) / total_steps * 100, 1)}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_planner_eval()
