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

# ─── Fake Document Store ──────────────────────────────────────────────────────
# Simulates a vector DB retrieval with intentional gaps

DOCUMENTS = [
    {
        "id": "doc1",
        "title": "AgentLens Overview",
        "content": "AgentLens is an observability platform for LLM agents. It captures traces, computes metrics, and detects regressions. It was built in 2024 by Prince Mittal."
    },
    {
        "id": "doc2",
        "title": "Tracer Module",
        "content": "The Tracer module instruments any agent pipeline. It records latency, token usage, cost, and step-by-step execution. It uses SQLite for storage."
    },
    {
        "id": "doc3",
        "title": "Evaluation Engine",
        "content": "The Evaluation Engine runs rule-based checks and LLM-judge metrics. Rules include latency SLA, output length, and refusal detection. LLM judges score task success, coherence, and hallucination."
    },
    {
        "id": "doc4",
        "title": "Supported Models",
        "content": "AgentLens currently supports Groq-hosted models including qwen/qwen3.8-27b. Anthropic and OpenAI models can be added via the evaluator config."
    },
]

# ─── Simulated Retriever ──────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 2) -> list:
    """Keyword-based retrieval (simulates vector search)."""
    query_lower = query.lower()
    scored = []
    for doc in DOCUMENTS:
        score = sum(
            1 for word in query_lower.split()
            if word in doc["content"].lower() or word in doc["title"].lower()
        )
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [doc for score, doc in scored[:top_k] if score > 0]
    return top


# ─── RAG Agent ───────────────────────────────────────────────────────────────

def build_context(docs: list) -> str:
    if not docs:
        return "No relevant documents found."
    parts = []
    for doc in docs:
        parts.append(f"[{doc['title']}]\n{doc['content']}")
    return "\n\n".join(parts)


@tracer.trace(agent="rag-agent", version="v1.0", model="qwen/qwen3.8-27b")
def rag_agent(query: str, _trace=None) -> str:
    # Step 1 — Retrieve
    retrieve_start = time.time()
    docs = retrieve(query)
    retrieve_latency = round((time.time() - retrieve_start) * 1000, 2)
    context = build_context(docs)

    if _trace:
        _trace.steps.append(Step(
            step_index=0,
            type="tool_call",
            input=query,
            output=f"Retrieved {len(docs)} docs: {[d['title'] for d in docs]}",
            latency_ms=retrieve_latency,
            tokens_used=0,
            tool_name="retriever",
        ))

    # Step 2 — Generate answer from context
    system_prompt = """You are a helpful assistant that answers questions strictly based on the provided context.
If the context does not contain enough information, say: "I don't have enough information in the provided documents."
Do NOT use your own knowledge. Only use the context."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""

    llm_start = time.time()
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    llm_latency = round((time.time() - llm_start) * 1000, 2)

    output = response.choices[0].message.content

    if _trace:
        _trace.prompt_tokens = response.usage.prompt_tokens
        _trace.completion_tokens = response.usage.completion_tokens
        _trace.total_tokens = response.usage.total_tokens
        _trace.steps.append(Step(
            step_index=1,
            type="llm_call",
            input=user_prompt,
            output=output,
            latency_ms=llm_latency,
            tokens_used=response.usage.total_tokens,
        ))

    return output


# ─── Test Suite ──────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "query": "What does the Tracer module record?",
        "context_key": "tracer",
        "known_facts": "The Tracer module records latency, token usage, cost, and step-by-step execution. It uses SQLite for storage.",
        "rubric": "Did the agent correctly describe the Tracer module's capabilities from the document?",
    },
    {
        "query": "What LLM judges does the Evaluation Engine use?",
        "context_key": "evaluation",
        "known_facts": "LLM judges score task success, coherence, and hallucination.",
        "rubric": "Did the agent mention task success, coherence, and hallucination as judge metrics?",
    },
    {
        "query": "Who is the CEO of Anthropic?",
        "context_key": "none",
        "known_facts": "The documents do not contain information about the CEO of Anthropic. The agent should say it doesn't have enough information.",
        "rubric": "Did the agent correctly say it lacks information rather than hallucinating an answer?",
    },
    {
        "query": "What models does AgentLens support?",
        "context_key": "models",
        "known_facts": "AgentLens supports Groq-hosted models including qwen/qwen3.8-27b.",
        "rubric": "Did the agent mention qwen/qwen3.8-27b and Groq as the supported model/platform?",
    },
    {
        "query": "When was AgentLens built and by whom?",
        "context_key": "overview",
        "known_facts": "AgentLens was built in 2024 by Prince Mittal.",
        "rubric": "Did the agent correctly state it was built in 2024 by Prince Mittal?",
    },
]


def run_rag_eval():
    print("\n" + "="*60)
    print("  AgentLens — RAG Agent Evaluation Suite")
    print("="*60)

    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}] {tc['query'][:60]}...")

        output = rag_agent(tc["query"])

        # Get retrieved context for groundedness eval
        docs = retrieve(tc["query"])
        context = build_context(docs)

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
            context=context,
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

    # Per-metric breakdown
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
            print(f"    ❌ {metric}: failed {count}/{len(results)} times")
    else:
        print("    ✅ No metric failures detected")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_rag_eval()
