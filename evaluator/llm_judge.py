import os
import json
import asyncio
from dataclasses import dataclass
from typing import Optional
from groq import Groq, AsyncGroq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

JUDGE_MODEL = "qwen/qwen3.8-27b"


@dataclass
class JudgeResult:
    metric: str
    score: float
    reasoning: str
    passed: bool


# ─── Shared prompt builder ────────────────────────────────────────────────────

def _build_prompt(metric: str, **kwargs) -> str:
    prompts = {
        "task_success": f"""You are an evaluator judging whether an AI agent successfully completed a task.

Task/Query: {kwargs.get('input_query')}
Agent Output: {kwargs.get('output')}
Rubric: {kwargs.get('rubric') or 'Did the agent answer the question correctly and completely?'}

Respond ONLY with valid JSON:
{{"score": 0.0, "reasoning": "your reasoning here", "passed": false}}
Score 0.0-1.0. passed is true if score >= 0.7.""",

        "groundedness": f"""You are an evaluator checking if an AI response is grounded in the provided context.

Context: {kwargs.get('context')}
Agent Output: {kwargs.get('output')}

Is the output supported by the context? Does it avoid claims not in the context?

Respond ONLY with valid JSON:
{{"score": 0.0, "reasoning": "your reasoning here", "passed": false}}
Score 1.0 = fully grounded, 0.0 = hallucinated. passed is true if score >= 0.7.""",

        "hallucination": f"""You are an evaluator detecting hallucinations in AI output.

Known Facts: {kwargs.get('known_facts')}
Agent Output: {kwargs.get('output')}

Does the output contain claims that contradict or go beyond the known facts?

Respond ONLY with valid JSON:
{{"score": 0.0, "reasoning": "your reasoning here", "passed": false}}
Score 1.0 = no hallucination, 0.0 = severe hallucination. passed is true if score >= 0.7.""",

        "coherence": f"""You are an evaluator judging the coherence and clarity of an AI response.

Agent Output: {kwargs.get('output')}

Is the response coherent, well-structured, and easy to understand?

Respond ONLY with valid JSON:
{{"score": 0.0, "reasoning": "your reasoning here", "passed": false}}
Score 1.0 = perfectly coherent, 0.0 = incoherent. passed is true if score >= 0.7.""",
    }
    return prompts[metric]


def _parse_response(raw: str, metric: str) -> JudgeResult:
    """Parse LLM response into JudgeResult."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    result = json.loads(raw)
    return JudgeResult(
        metric=metric,
        score=float(result["score"]),
        reasoning=result["reasoning"],
        passed=bool(result["passed"])
    )


# ─── Sync versions (kept for backward compatibility) ─────────────────────────

def _call_judge_sync(prompt: str) -> dict:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def evaluate_task_success(input_query: str, output: str, rubric: str = "") -> JudgeResult:
    prompt = _build_prompt("task_success", input_query=input_query, output=output, rubric=rubric)
    result = _call_judge_sync(prompt)
    return JudgeResult(metric="task_success", score=float(result["score"]),
                       reasoning=result["reasoning"], passed=bool(result["passed"]))


def evaluate_groundedness(output: str, context: str) -> JudgeResult:
    prompt = _build_prompt("groundedness", output=output, context=context)
    result = _call_judge_sync(prompt)
    return JudgeResult(metric="groundedness", score=float(result["score"]),
                       reasoning=result["reasoning"], passed=bool(result["passed"]))


def evaluate_hallucination(output: str, known_facts: str) -> JudgeResult:
    prompt = _build_prompt("hallucination", output=output, known_facts=known_facts)
    result = _call_judge_sync(prompt)
    return JudgeResult(metric="hallucination", score=float(result["score"]),
                       reasoning=result["reasoning"], passed=bool(result["passed"]))


def evaluate_coherence(output: str) -> JudgeResult:
    prompt = _build_prompt("coherence", output=output)
    result = _call_judge_sync(prompt)
    return JudgeResult(metric="coherence", score=float(result["score"]),
                       reasoning=result["reasoning"], passed=bool(result["passed"]))


# ─── Async versions ───────────────────────────────────────────────────────────

async def _call_judge_async(prompt: str, metric: str) -> JudgeResult:
    """Single async judge call."""
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_response(raw, metric)


async def evaluate_all_async(
    input_query: str,
    output: str,
    rubric: str = "",
    context: Optional[str] = None,
    known_facts: Optional[str] = None,
) -> list:
    """
    Run ALL judge evaluations in parallel using asyncio.gather.
    3-4x faster than sequential calls.
    """
    tasks = []

    # Always run these two
    tasks.append(_call_judge_async(
        _build_prompt("task_success", input_query=input_query, output=output, rubric=rubric),
        "task_success"
    ))
    tasks.append(_call_judge_async(
        _build_prompt("coherence", output=output),
        "coherence"
    ))

    # Conditionally add groundedness and hallucination
    if context:
        tasks.append(_call_judge_async(
            _build_prompt("groundedness", output=output, context=context),
            "groundedness"
        ))
    if known_facts:
        tasks.append(_call_judge_async(
            _build_prompt("hallucination", output=output, known_facts=known_facts),
            "hallucination"
        ))

    # Run all in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions — return successful results only
    valid = []
    for r in results:
        if isinstance(r, Exception):
            print(f"  Judge error: {r}")
        else:
            valid.append(r)

    return valid


def run_judges_parallel(
    input_query: str,
    output: str,
    rubric: str = "",
    context: Optional[str] = None,
    known_facts: Optional[str] = None,
) -> list:
    """
    Sync wrapper around async judge — use this from sync code.
    Runs all judges in parallel, returns results.
    """
    return asyncio.run(evaluate_all_async(
        input_query=input_query,
        output=output,
        rubric=rubric,
        context=context,
        known_facts=known_facts,
    ))
