import os
import sys
import time
sys.path.insert(0, "/home/prince-mittal/agentlens")

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/prince-mittal/agentlens/.env")

from groq import Groq
from tracer import Tracer, Step
from tracer.retriever import FAISSRetriever, get_retriever
from evaluator import evaluate_trace

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tracer = Tracer(project="agentlens")

# ─── Knowledge Base (richer than before) ─────────────────────────────────────

DOCUMENTS = [
    {
        "id": "doc1",
        "title": "AgentLens Overview",
        "content": "AgentLens is an observability platform for LLM agents built by Prince Mittal in 2024. It captures traces, computes metrics, and detects regressions across agent versions."
    },
    {
        "id": "doc2",
        "title": "Tracer Module",
        "content": "The Tracer module instruments any agent pipeline using a decorator. It records latency, token usage, cost, and step-by-step execution. All data is stored in SQLite."
    },
    {
        "id": "doc3",
        "title": "Evaluation Engine",
        "content": "The Evaluation Engine runs rule-based checks and LLM-judge metrics. Rules include latency SLA, output length, and refusal detection. LLM judges score task success, coherence, groundedness, and hallucination."
    },
    {
        "id": "doc4",
        "title": "Supported Models",
        "content": "AgentLens currently supports Groq-hosted models including qwen/qwen3.8-27b. Anthropic and OpenAI models can be added via the evaluator config."
    },
    {
        "id": "doc5",
        "title": "Regression Detection",
        "content": "The regression engine compares metric distributions across two agent versions. It uses configurable thresholds to classify changes as critical, warning, improvement, or stable."
    },
    {
        "id": "doc6",
        "title": "Failure Clustering",
        "content": "Failure clustering embeds failed run outputs using sentence-transformers and groups them with HDBSCAN. Each cluster is auto-labeled by an LLM to describe the failure pattern."
    },
    {
        "id": "doc7",
        "title": "Dashboard",
        "content": "The dashboard is built with FastAPI and Jinja2. It includes a run explorer, trace viewer, regression page, failure cluster page, and charts with Chart.js."
    },
    {
        "id": "doc8",
        "title": "GitHub Actions CI",
        "content": "AgentLens has a GitHub Actions CI pipeline that runs an eval suite on every push to main. If any test scores below 0.7, the build fails automatically."
    },
    {
        "id": "doc9",
        "title": "FAISS Retriever",
        "content": "The FAISS retriever uses sentence-transformer embeddings and cosine similarity search. It returns a relevance score per document and supports a similarity threshold to reject irrelevant queries."
    },
    {
        "id": "doc10",
        "title": "Cost Tracking",
        "content": "AgentLens tracks cost per run based on token usage and model pricing. Total platform cost across all runs is displayed in the charts dashboard."
    },
]

# Similarity threshold — below this, retrieval is considered irrelevant
RELEVANCE_THRESHOLD = 0.15


# ─── Build retriever once ─────────────────────────────────────────────────────

def get_rag_retriever() -> FAISSRetriever:
    return get_retriever(documents=DOCUMENTS)


# ─── RAG Agent ───────────────────────────────────────────────────────────────

def build_context(docs) -> str:
    if not docs:
        return "No relevant documents found."
    parts = []
    for doc in docs:
        parts.append(f"[{doc.title}] (relevance: {doc.score:.3f})\n{doc.content}")
    return "\n\n".join(parts)


@tracer.trace(agent="rag-agent-v2", version="v2.0", model="qwen/qwen3.8-27b")
def rag_agent(query: str, _trace=None) -> str:
    retriever = get_rag_retriever()

    # Step 1 — FAISS retrieval
    retrieve_start = time.time()
    docs = retriever.retrieve(query, top_k=3)
    retrieve_latency = round((time.time() - retrieve_start) * 1000, 2)

    # Filter by relevance threshold
    relevant_docs = [d for d in docs if d.score >= RELEVANCE_THRESHOLD]
    context = build_context(relevant_docs)

    if _trace:
        _trace.steps.append(Step(
            step_index=0,
            type="tool_call",
            input=query,
            output=f"Retrieved {len(relevant_docs)} relevant docs "
                   f"(threshold={RELEVANCE_THRESHOLD}): "
                   f"{[d.title for d in relevant_docs]}",
            latency_ms=retrieve_latency,
            tokens_used=0,
            tool_name="faiss-retriever",
        ))

    # Step 2 — Generate answer
    system_prompt = """You are a helpful assistant that answers questions strictly based on the provided context.
If the context does not contain enough information, say exactly:
"I don't have enough information in the provided documents."
Do NOT use your own knowledge. Only use the context provided."""

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
            input=user_prompt[:200],
            output=output,
            latency_ms=llm_latency,
            tokens_used=response.usage.total_tokens,
        ))

    return output


# ─── Test Suite ───────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "query": "What does the Tracer module record?",
        "known_facts": "The Tracer module records latency, token usage, cost, and step-by-step execution. It uses SQLite.",
        "rubric": "Did the agent correctly describe Tracer module capabilities from the document?",
    },
    {
        "query": "How does failure clustering work?",
        "known_facts": "Failure clustering uses sentence-transformers and HDBSCAN. Each cluster is auto-labeled by an LLM.",
        "rubric": "Did the agent mention sentence-transformers, HDBSCAN, and LLM labeling?",
    },
    {
        "query": "What happens when CI fails?",
        "known_facts": "If any test scores below 0.7, the build fails automatically.",
        "rubric": "Did the agent correctly describe the CI failure condition (score below 0.7)?",
    },
    {
        "query": "Who is the CEO of OpenAI?",
        "known_facts": "The documents do not contain information about the CEO of OpenAI. The agent should say it lacks information.",
        "rubric": "Did the agent correctly refuse to answer due to lack of relevant context?",
    },
    {
        "query": "What similarity threshold does FAISS retriever use?",
        "known_facts": "The FAISS retriever uses a relevance score and supports a similarity threshold to reject irrelevant queries.",
        "rubric": "Did the agent mention the FAISS retriever's similarity threshold capability?",
    },
    {
        "query": "How much does running AgentLens cost?",
        "known_facts": "AgentLens tracks cost per run based on token usage. Total cost is shown in the charts dashboard.",
        "rubric": "Did the agent correctly describe how cost is tracked per run?",
    },
]


def run_rag_eval():
    print("\n" + "="*62)
    print("  AgentLens — RAG Agent v2 (FAISS) Evaluation Suite")
    print("="*62)

    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}] {tc['query'][:58]}...")

        output = rag_agent(tc["query"])

        # Get retrieved context for groundedness eval
        retriever = get_rag_retriever()
        docs = retriever.retrieve(tc["query"], top_k=3)
        relevant = [d for d in docs if d.score >= RELEVANCE_THRESHOLD]
        context = build_context(relevant)

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
    run_rag_eval()
