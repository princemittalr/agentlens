# ⬡ AgentLens

<div align="center">

**Production-grade observability and evaluation platform for LLM agent pipelines.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-Powered-orange?style=flat-square)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

[Features](#features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Dashboard](#dashboard) • [Bugs Caught](#real-bugs-surfaced) • [Regression Detection](#regression-detection)

</div>

---

## What is AgentLens?

AgentLens is a **standalone observability and evaluation platform** for LLM agent pipelines. It is not tied to any single agent framework — it instruments arbitrary pipelines using a simple decorator, then automatically:

- Captures full execution traces (latency, tokens, cost, step-by-step reasoning)
- Runs **rule-based** and **LLM-judge** metrics on every run
- Detects **regressions** across prompt versions and model changes
- Visualizes everything in a **live web dashboard**

Built to solve the *"eval design is the single biggest signal"* gap identified across AI engineering hiring research — the exact category of internal tooling that companies like Anthropic, Databricks, and Cohere build themselves.

---

## Features

### 🔭 Instrumentation Layer
- Zero-boilerplate `@tracer.trace()` decorator wraps any agent function
- Captures per-run: latency, prompt tokens, completion tokens, cost, steps, errors
- Records step-level granularity: LLM calls, tool calls, thoughts, final answers
- Persists everything to SQLite (upgradeable to PostgreSQL)

### 📊 Evaluation Engine
**Rule-based metrics (deterministic, instant):**
| Metric | Description |
|--------|-------------|
| `no_error` | Did the agent complete without an exception? |
| `output_not_empty` | Did the agent produce any output? |
| `latency_sla` | Was latency within the configured threshold (default 3s)? |
| `output_length` | Was output length within acceptable word range? |
| `no_refusal` | Did the agent avoid refusing to answer? |
| `json_format` | (Optional) Is the output valid JSON? |

**LLM-judge metrics (semantic, automatic):**
| Metric | Description |
|--------|-------------|
| `task_success` | Did the agent complete the task correctly per rubric? |
| `coherence` | Is the response coherent, structured, and clear? |
| `groundedness` | Is the response grounded in provided context? |
| `hallucination` | Does the response contradict known facts? |

### 🔁 Regression Detection
- Compares metric distributions between any two agent versions
- Configurable thresholds: `warn` and `critical` per metric
- Handles both quality metrics (lower = regression) and system metrics (higher = regression)
- Produces structured reports: verdict, per-metric delta, severity classification

### 🖥️ Live Dashboard
- **Run Explorer** — all runs with score, status, latency, cost, tokens
- **Trace Viewer** — step-by-step execution breakdown per run
- **Regression Page** — side-by-side version comparison with alerts
- Search, filter by agent, filter by pass/fail status
- REST API endpoints for programmatic access

---

## Architecture

```
agentlens/
│
├── tracer/                  # Instrumentation layer
│   ├── __init__.py
│   ├── tracer.py            # @tracer.trace() decorator
│   ├── models.py            # AgentTrace + Step dataclasses
│   ├── database.py          # SQLite read/write layer
│   └── regression.py        # Version comparison engine
│
├── evaluator/               # Evaluation engine
│   ├── __init__.py
│   ├── rules.py             # Rule-based metric checks
│   ├── llm_judge.py         # LLM-as-judge evaluators
│   └── scorer.py            # Unified scorer + DB persistence
│
├── agents/                  # 3 validated agent architectures
│   ├── react_agent.py       # ReAct loop with tool use
│   ├── rag_agent.py         # RAG with document retrieval
│   └── planner_agent.py     # Multi-step task planner
│
├── dashboard/               # Web UI
│   ├── app.py               # FastAPI routes
│   └── templates/           # Jinja2 HTML templates
│       ├── index.html        # Run explorer
│       ├── trace.html        # Trace detail view
│       └── regression.html   # Regression report view
│
├── tests/
│   └── test_regression.py   # Regression detection test
│
├── agentlens.db             # SQLite database (auto-created)
├── requirements.txt
└── .env                     # API keys (not committed)
```

### How the layers connect

```
Your Agent
    │
    ▼
@tracer.trace()              ← wraps any function, zero code changes
    │
    ├──► AgentTrace saved to SQLite (traces table)
    │
    ▼
evaluate_trace()             ← call after each run
    │
    ├──► Rule-based checks   (instant, deterministic)
    ├──► LLM judge calls     (semantic scoring via Groq)
    └──► EvalResult saved to SQLite (evaluations table)
                │
                ▼
        compare_versions()   ← regression detection
                │
                ▼
        Dashboard (FastAPI)  ← visualize everything
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/princemittalr/agentlens.git
cd agentlens

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run the included agents

```bash
# Agent 1 — ReAct agent with tool use
python3 agents/react_agent.py

# Agent 2 — RAG agent with document retrieval
python3 agents/rag_agent.py

# Agent 3 — Multi-step planner agent
python3 agents/planner_agent.py

# Run regression detection (v1.0 vs v2.0)
python3 tests/test_regression.py
```

### Start the dashboard

```bash
python3 -m uvicorn dashboard.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

---

## Instrument Your Own Agent

Adding AgentLens to any existing agent takes **3 lines of code**:

```python
from tracer import Tracer, Step

tracer = Tracer(project="my-project")

@tracer.trace(agent="my-agent", version="v1.0", model="qwen/qwen3.8-27b")
def my_agent(query: str, _trace=None) -> str:
    # ── your existing agent code, completely unchanged ──
    response = llm_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": query}]
    )

    # optionally record token usage
    if _trace:
        _trace.prompt_tokens = response.usage.prompt_tokens
        _trace.completion_tokens = response.usage.completion_tokens
        _trace.total_tokens = response.usage.total_tokens

    return response.choices[0].message.content
```

Then evaluate any run:

```python
from evaluator import evaluate_trace

evaluate_trace(
    run_id=latest_trace["run_id"],
    agent_name="my-agent",
    agent_version="v1.0",
    input_query=query,
    output=output,
    latency_ms=latency,
    known_facts="Ground truth to check against",
    rubric="Did the agent answer the question correctly and completely?",
)
```

Then detect regressions between versions:

```python
from tracer.regression import compare_versions

report = compare_versions(
    agent_name="my-agent",
    baseline_version="v1.0",
    candidate_version="v2.0",
)
# Prints full regression report with per-metric deltas and severity
```

---

## Dashboard

### Run Explorer
The main view shows all agent runs across all architectures with:
- Overall eval score (color-coded: green ≥ 0.8, yellow ≥ 0.6, red < 0.6)
- Pass/Fail badge per run
- Latency, token usage, cost per run
- Search by query, agent name, or run ID
- Filter by agent or pass/fail status

### Trace Viewer
Click **View →** on any run to see:
- Full trace metadata (run ID, agent, version, model, timestamps)
- Per-metric scores with visual progress bars
- LLM judge reasoning excerpts
- Step-by-step execution: every LLM call, tool call, and observation
- Error details for failed steps

### Regression Page
Select any two versions of the same agent to get:
- Per-metric comparison table with delta percentages
- Critical 🔴 / Warning 🟡 / Improvement 🟢 / Stable ✅ classification
- Structured alert messages for every detected regression
- Overall verdict: REGRESSION / WARNING / IMPROVEMENT / STABLE

---

## Real Bugs Surfaced

AgentLens was validated against 3 agent architectures and automatically caught **6 real bugs** — none of which would have been caught by simple pass/fail testing.

### Agent 1 — ReAct Agent

| Bug | Metric That Caught It | Details |
|-----|----------------------|---------|
| Step limit exhaustion | `task_success: 0.0` | Agent consumed all 5 steps on "Who founded Anthropic?" without producing an answer |
| Terse calculator output | `coherence: 0.0`, `output_length: 0.5` | Calculator returned bare "1200" — no context, flagged as incoherent by LLM judge |

### Agent 2 — RAG Agent

| Bug | Metric That Caught It | Details |
|-----|----------------------|---------|
| Speculative claim presented as fact | `hallucination: 0.0` | Agent repeated "Anthropic and OpenAI models can be added" from docs — unverified claim flagged |

### Agent 3 — Planner Agent

| Bug | Metric That Caught It | Details |
|-----|----------------------|---------|
| Broken input substitution | `task_success: 0.0` | Step inputs weren't passed correctly across chained steps — 3/4 runs failed |
| Fragment outputs | `coherence: 0.0` | Output "le texte en français" — a fragment with no actual translation |
| Fabricated completions | `hallucination: 0.0` | Agent reported completing steps it never actually executed |

---

## Regression Detection

AgentLens automatically detected a **critical quality regression** when comparing a degraded prompt (v2.0 — "one word only") against the baseline (v1.0):

```
═══════════════════════════════════════════════════════════════════
  AgentLens Regression Report
═══════════════════════════════════════════════════════════════════
  Agent     : react-agent
  Baseline  : v1.0 (4 runs)
  Candidate : v2.0 (4 runs)
  Verdict   : REGRESSION 🔴

  Metric           Baseline   Candidate    Delta%     Status
  ────────────────────────────────────────────────────────────
  coherence          0.5000     0.0000    -100.0%    🔴 critical
  hallucination      1.0000     0.7500     -25.0%    🔴 critical
  output_length      0.8750     0.5000     -42.9%    🔴 critical
  overall_score      0.8906     0.7656     -14.0%    🔴 critical
  latency_ms      1463.7ms    506.3ms     -65.4%    🟢 improvement
  cost_usd           0.0003     0.0000    -100.0%    🟢 improvement
  total_tokens     865.25      44.00      -94.9%    🟢 improvement

  ⚠ Regressions (4):
    • coherence: ↓ decreased by 100.0% (0.5000 → 0.0000) [CRITICAL]
    • hallucination: ↓ decreased by 25.0% (1.0000 → 0.7500) [CRITICAL]
    • output_length: ↓ decreased by 42.9% (0.8750 → 0.5000) [CRITICAL]
    • overall_score: ↓ decreased by 14.0% (0.8906 → 0.7656) [CRITICAL]

  ✨ Improvements (4):
    • latency_ms: ↓ decreased by 65.4% [IMPROVEMENT]
    • cost_usd: ↓ decreased by 100.0% [IMPROVEMENT]
    • total_tokens: ↓ decreased by 94.9% [IMPROVEMENT]
═══════════════════════════════════════════════════════════════════
```

**Key insight:** v2.0 was 65% faster and 100% cheaper — a naive cost-only monitor would have shipped it. AgentLens caught that it was fundamentally broken in quality.

---

## API Reference

AgentLens exposes a REST API for programmatic access:

```
GET  /api/traces              → all traces (JSON)
GET  /api/evals               → all evaluations (JSON)
GET  /api/regression          → regression report
     ?agent=react-agent
     &baseline=v1.0
     &candidate=v2.0
```

Example:
```bash
curl "http://localhost:8000/api/regression?agent=react-agent&baseline=v1.0&candidate=v2.0"
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| LLM Provider | Groq API (`qwen/qwen3.8-27b`) |
| Backend | FastAPI + Uvicorn |
| Templating | Jinja2 |
| Database | SQLite (zero-config, upgradeable) |
| Eval Pattern | LLM-as-judge + rule-based |
| Tracing | Decorator-based instrumentation |

---

## Roadmap

- [ ] PostgreSQL backend for production deployments
- [ ] Failure clustering with HDBSCAN (embed + cluster failed runs)
- [ ] Async evaluation jobs with Celery + Redis
- [ ] Prometheus metrics export
- [ ] GitHub Actions CI integration for automated regression checks
- [ ] Support for OpenAI, Anthropic, and LangChain agent instrumentation
- [ ] Multi-project support with project-level isolation

---

## Why This Project

Most AI engineers build agents. Very few build the infrastructure to **measure whether those agents actually work** — across versions, across architectures, at scale. Eval infrastructure is the hardest and most valuable part of shipping reliable AI products.

AgentLens is a demonstration that I can build both the agents *and* the platform to evaluate them — end to end, production-grade, from scratch.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Prince Mittal R**

Built as a portfolio project targeting production AI engineering and research roles.

- GitHub: [@princemittalr](https://github.com/princemittalr)
- Project: [github.com/princemittalr/agentlens](https://github.com/princemittalr/agentlens)