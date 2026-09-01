"""
Seed AgentLens DB with demo data for Railway deployment.
Run once after deploy: python seed_demo.py
"""
import sys
import os
sys.path.insert(0, "/home/prince-mittal/agentlens")

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from tracer import Tracer, Step
from evaluator import evaluate_trace
from tracer.database import get_all_traces

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tracer = Tracer(project="demo")

SYSTEM = "You are a helpful assistant. Answer clearly and completely."

@tracer.trace(agent="demo-agent", version="v1.0", model="qwen/qwen3.8-27b")
def demo_agent(query: str, _trace=None) -> str:
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": SYSTEM},
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

DEMOS = [
    ("What is the capital of France?",
     "Paris is the capital of France.",
     "Did the agent correctly identify Paris?"),
    ("Explain what an API is.",
     "An API allows software systems to communicate with each other.",
     "Did the agent explain API correctly?"),
    ("What is 144 divided by 12?",
     "144 / 12 = 12",
     "Did the agent correctly calculate 12?"),
    ("What does LLM stand for?",
     "LLM stands for Large Language Model.",
     "Did the agent correctly expand LLM?"),
    ("Name three programming languages.",
     "Python, JavaScript, and Java are three popular programming languages.",
     "Did the agent name three valid programming languages?"),
]

print("Seeding demo data...")
for query, facts, rubric in DEMOS:
    output = demo_agent(query)
    latest = get_all_traces()[0]
    evaluate_trace(
        run_id=latest["run_id"],
        agent_name=latest["agent_name"],
        agent_version=latest["agent_version"],
        input_query=query,
        output=output,
        latency_ms=latest["total_latency_ms"],
        error=latest["error"],
        known_facts=facts,
        rubric=rubric,
    )
    print(f"  ✅ {query[:50]}")

print("\nDemo data seeded successfully!")
