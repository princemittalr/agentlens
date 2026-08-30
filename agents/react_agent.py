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

# ─── Fake Tools (simulating real search/calc) ────────────────────────────────

KNOWLEDGE_BASE = {
    "paris": "Paris is the capital of France. Population ~2.1 million.",
    "eiffel tower": "The Eiffel Tower is 330 meters tall, built in 1889.",
    "france": "France is a country in Western Europe. Capital: Paris.",
    "python": "Python is a programming language created by Guido van Rossum in 1991.",
    "anthropic": "Anthropic is an AI safety company founded in 2021. Creator of Claude.",
    "openai": "OpenAI was founded in 2015. Creator of GPT and ChatGPT.",
}

def search(query: str) -> str:
    """Simulated search tool."""
    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return value
    # Intentional gap — returns nothing for unknown queries
    return "No results found."

def calculator(expression: str) -> str:
    """Safe calculator tool."""
    try:
        allowed = set("0123456789+-*/()., ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return str(result)
        return "Invalid expression."
    except Exception as e:
        return f"Calculation error: {e}"

TOOLS = {
    "search": search,
    "calculator": calculator,
}

TOOL_DESCRIPTIONS = """
Available tools:
- search(query): Search for factual information
- calculator(expression): Evaluate a math expression

To use a tool, respond with:
Action: tool_name
Action Input: your input

When you have the final answer, respond with:
Final Answer: your answer
"""

# ─── ReAct Loop ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a helpful assistant that solves tasks step by step.
{TOOL_DESCRIPTIONS}
Always think before acting. Format:
Thought: your reasoning
Action: tool_name
Action Input: input to tool

Or if done:
Thought: I have enough information
Final Answer: your final answer"""


def parse_action(text: str):
    """Extract action and input from model response."""
    lines = text.strip().split("\n")
    action = None
    action_input = None
    final_answer = None

    for i, line in enumerate(lines):
        if line.startswith("Action:"):
            action = line.replace("Action:", "").strip()
        elif line.startswith("Action Input:"):
            action_input = line.replace("Action Input:", "").strip()
        elif line.startswith("Final Answer:"):
            final_answer = line.replace("Final Answer:", "").strip()

    return action, action_input, final_answer


@tracer.trace(agent="react-agent", version="v1.0", model="qwen/qwen3.8-27b")
def react_agent(query: str, _trace=None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query}
    ]

    max_steps = 5
    steps = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for step_idx in range(max_steps):
        step_start = time.time()

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )

        step_latency = round((time.time() - step_start) * 1000, 2)
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens

        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})

        action, action_input, final_answer = parse_action(assistant_text)

        # ── Final answer reached ──
        if final_answer:
            if _trace:
                _trace.steps.append(Step(
                    step_index=step_idx,
                    type="final_answer",
                    input=query,
                    output=final_answer,
                    latency_ms=step_latency,
                    tokens_used=response.usage.total_tokens,
                ))
                _trace.prompt_tokens = total_prompt_tokens
                _trace.completion_tokens = total_completion_tokens
                _trace.total_tokens = total_prompt_tokens + total_completion_tokens
            return final_answer

        # ── Tool call ──
        if action and action_input:
            tool_fn = TOOLS.get(action.lower())
            if tool_fn:
                observation = tool_fn(action_input)
            else:
                observation = f"Unknown tool: {action}"

            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })

            if _trace:
                _trace.steps.append(Step(
                    step_index=step_idx,
                    type="tool_call",
                    input=action_input,
                    output=observation,
                    latency_ms=step_latency,
                    tokens_used=response.usage.total_tokens,
                    tool_name=action,
                ))
        else:
            # Model didn't follow format — still record
            if _trace:
                _trace.steps.append(Step(
                    step_index=step_idx,
                    type="llm_call",
                    input=query,
                    output=assistant_text,
                    latency_ms=step_latency,
                    tokens_used=response.usage.total_tokens,
                ))

    # Max steps exceeded — return best effort
    if _trace:
        _trace.prompt_tokens = total_prompt_tokens
        _trace.completion_tokens = total_completion_tokens
        _trace.total_tokens = total_prompt_tokens + total_completion_tokens

    return "Agent could not complete the task within the step limit."


# ─── Test Suite ──────────────────────────────────────────────────────────────

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
        "rubric": "Did the agent correctly state Mars has no human population or handled the unknown correctly?",
    },
]


def run_react_eval():
    print("\n" + "="*60)
    print("  AgentLens — ReAct Agent Evaluation Suite")
    print("="*60)

    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}] {tc['query'][:60]}...")

        output = react_agent(tc["query"])

        from tracer.database import get_all_traces
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
        )
        results.append(eval_result)

    # Summary
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: {passed}/{len(results)} tests passed")
    print(f"  Avg Score: {round(sum(r.overall_score for r in results)/len(results), 3)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_react_eval()
