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
    if not text or not text.strip():
        raise ValueError("Cannot summarize empty input.")
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
    if not text or not text.strip():
        raise ValueError("Cannot extract keywords from empty input.")
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
    if not text or not text.strip():
        raise ValueError("Cannot classify sentiment of empty input.")
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
    if not text or not text.strip():
        raise ValueError("Cannot translate empty input.")
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": "Translate the following text to French. Return only the translation."},
            {"role": "user", "content": text},
        ],
        max_tokens=300,
        temperature=0.1,
    )
    return response.choices[0].message.content


def tool_broken_json_parser(text: str) -> str:
    raise ValueError(f"JSON parse error: unexpected token at position 0 in: '{text[:30]}...'")


TOOLS = {
    "summarize": tool_summarize,
    "extract_keywords": tool_extract_keywords,
    "classify_sentiment": tool_classify_sentiment,
    "translate": tool_translate,
    "parse_json": tool_broken_json_parser,
}

# ─── Planner ─────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are a task planner. Given a user goal and input text, break it into steps.
Each step must use one of these tools: summarize, extract_keywords, classify_sentiment, translate, parse_json.

Respond ONLY with a JSON array like:
[
  {"step": 1, "tool": "tool_name", "use_input": "original"},
  {"step": 2, "tool": "tool_name", "use_input": "previous"}
]

"use_input" must be either:
- "original" → use the original input text
- "previous" → use the output of the previous step

Return ONLY the JSON array, no explanation."""


def generate_plan(goal: str) -> list:
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
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


@tracer.trace(agent="planner-agent", version="v2.0", model="qwen/qwen3.8-27b")
def planner_agent(goal: str, input_text: str, _trace=None) -> str:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    errors = []

    # ── Step 0: Generate plan ──────────────────────────────────────
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
        return f"Planning failed: {e}"

    # ── Execute steps with proper input chaining ───────────────────
    # KEY FIX: maintain a results dict keyed by step number
    # Each step explicitly declares whether it uses "original" or "previous"
    step_results = {0: input_text}  # step 0 = original input
    final_output = ""

    for plan_step in plan:
        step_num = plan_step.get("step", 0)
        tool_name = plan_step.get("tool", "")
        use_input = plan_step.get("use_input", "original")

        # ── FIXED: explicit input routing ──────────────────────────
        if use_input == "previous":
            # Use the output of the immediately preceding step
            prev_step = step_num - 1
            tool_input = step_results.get(prev_step, "")
            if not tool_input:
                error_msg = f"Step {step_num}: previous step ({prev_step}) produced no output — skipping"
                errors.append(error_msg)
                step_results[step_num] = ""
                if _trace:
                    _trace.steps.append(Step(
                        step_index=step_num,
                        type="tool_call",
                        input="(empty — previous step failed)",
                        output="",
                        latency_ms=0,
                        tokens_used=0,
                        tool_name=tool_name,
                        error=error_msg,
                    ))
                continue
        else:
            # use_input == "original"
            tool_input = input_text

        # ── Execute tool ───────────────────────────────────────────
        step_start = time.time()
        tool_fn = TOOLS.get(tool_name)

        if not tool_fn:
            error_msg = f"Unknown tool: {tool_name}"
            errors.append(error_msg)
            step_results[step_num] = ""
            if _trace:
                _trace.steps.append(Step(
                    step_index=step_num,
                    type="tool_call",
                    input=tool_input[:200],
                    output="",
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
            final_output = result

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


# ─── Test Suite ───────────────────────────────────────────────────────────────

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
        "rubric": "Did the agent produce a meaningful summary and relevant keywords about AgentLens?",
    },
    {
        "goal": "Classify the sentiment of the text, then translate the sentiment result to French",
        "input_text": SAMPLE_TEXT,
        "known_facts": "The text is positive in sentiment. A French translation of the sentiment word should be provided.",
        "rubric": "Did the agent classify sentiment as POSITIVE and provide a French translation?",
    },
    {
        "goal": "Parse the text as JSON, then summarize it",
        "input_text": SAMPLE_TEXT,
        "known_facts": "The parse_json tool is broken and will fail. The agent should handle this gracefully.",
        "rubric": "Did the agent report the JSON parse failure clearly without crashing entirely?",
    },
    {
        "goal": "Extract keywords from the text, then summarize those keywords",
        "input_text": SAMPLE_TEXT,
        "known_facts": "Keywords about AgentLens should be extracted first. Then a summary of those keywords produced.",
        "rubric": "Did the agent correctly chain extract_keywords → summarize with real output passing between steps?",
    },
]


def run_planner_eval():
    print("\n" + "="*62)
    print("  AgentLens — Planner Agent v2.0 (Fixed) Evaluation Suite")
    print("="*62)

    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}] {tc['goal'][:58]}...")
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

    passed = sum(1 for r in results if r.passed)
    avg_score = round(sum(r.overall_score for r in results) / len(results), 3)

    print(f"\n{'='*62}")
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
    print(f"{'='*62}\n")


if __name__ == "__main__":
    run_planner_eval()
