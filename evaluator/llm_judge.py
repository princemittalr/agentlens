import os
import json
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

JUDGE_MODEL = "qwen/qwen3.8-27b"


@dataclass
class JudgeResult:
    metric: str
    score: float        # 0.0 to 1.0
    reasoning: str
    passed: bool


def _call_judge(prompt: str) -> dict:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def evaluate_task_success(input_query: str, output: str, rubric: str = "") -> JudgeResult:
    prompt = f"""You are an evaluator judging whether an AI agent successfully completed a task.

Task/Query: {input_query}
Agent Output: {output}
Rubric: {rubric if rubric else "Did the agent answer the question correctly and completely?"}

Respond ONLY with valid JSON in this exact format:
{{"score": 0.0, "reasoning": "your reasoning here", "passed": false}}

Score must be between 0.0 and 1.0. passed is true if score >= 0.7."""

    result = _call_judge(prompt)
    return JudgeResult(
        metric="task_success",
        score=float(result["score"]),
        reasoning=result["reasoning"],
        passed=bool(result["passed"])
    )


def evaluate_groundedness(output: str, context: str) -> JudgeResult:
    prompt = f"""You are an evaluator checking if an AI response is grounded in the provided context.

Context: {context}
Agent Output: {output}

Is the output supported by the context? Does it avoid making claims not in the context?

Respond ONLY with valid JSON in this exact format:
{{"score": 0.0, "reasoning": "your reasoning here", "passed": false}}

Score 1.0 = fully grounded, 0.0 = completely hallucinated. passed is true if score >= 0.7."""

    result = _call_judge(prompt)
    return JudgeResult(
        metric="groundedness",
        score=float(result["score"]),
        reasoning=result["reasoning"],
        passed=bool(result["passed"])
    )


def evaluate_hallucination(output: str, known_facts: str) -> JudgeResult:
    prompt = f"""You are an evaluator detecting hallucinations in AI output.

Known Facts: {known_facts}
Agent Output: {output}

Does the output contain claims that contradict or go beyond the known facts?

Respond ONLY with valid JSON in this exact format:
{{"score": 0.0, "reasoning": "your reasoning here", "passed": false}}

Score 1.0 = no hallucination, 0.0 = severe hallucination. passed is true if score >= 0.7."""

    result = _call_judge(prompt)
    return JudgeResult(
        metric="hallucination",
        score=float(result["score"]),
        reasoning=result["reasoning"],
        passed=bool(result["passed"])
    )


def evaluate_coherence(output: str) -> JudgeResult:
    prompt = f"""You are an evaluator judging the coherence and clarity of an AI response.

Agent Output: {output}

Is the response coherent, well-structured, and easy to understand?

Respond ONLY with valid JSON in this exact format:
{{"score": 0.0, "reasoning": "your reasoning here", "passed": false}}

Score 1.0 = perfectly coherent, 0.0 = incoherent. passed is true if score >= 0.7."""

    result = _call_judge(prompt)
    return JudgeResult(
        metric="coherence",
        score=float(result["score"]),
        reasoning=result["reasoning"],
        passed=bool(result["passed"])
    )
