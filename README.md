# ⬡ AgentLens

<div align="center">

**Production-grade observability and evaluation platform for LLM agent pipelines.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-Powered-orange?style=flat-square)](https://groq.com)
[![CI](https://img.shields.io/github/actions/workflow/status/princemittalr/agentlens/eval.yml?style=flat-square&label=CI)](https://github.com/princemittalr/agentlens/actions)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

[Live Demo](#-live-demo) • [Features](#features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Bugs Caught](#real-bugs-surfaced) • [Regression Detection](#regression-detection)

</div>

---

## 🌐 Live Demo

**[https://agentlens-production-29ad.up.railway.app](https://agentlens-production-29ad.up.railway.app)**

The live dashboard shows real agent runs, eval scores, regression reports, failure clusters, and metric charts — no setup required.

---

## What is AgentLens?

AgentLens is a **standalone observability and evaluation platform** for LLM agent pipelines. It is not tied to any single agent framework — it instruments arbitrary pipelines using a simple decorator, then automatically:

- Captures full execution traces (latency, tokens, cost, step-by-step reasoning)
- Runs **rule-based** and **LLM-judge** metrics on every run
- Detects **regressions** across prompt versions and model changes with **statistical significance testing**
- Embeds failed outputs and clusters them with **HDBSCAN** — auto-labeled by an LLM
- Visualizes everything in a **live web dashboard** deployed on Railway

Built to solve the *"eval design is the single biggest signal"* gap in AI engineering hiring — the exact category of internal tooling that companies like Anthropic, Databricks, and Cohere build themselves.

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
- **Mann-Whitney U statistical significance testing** — only flags real regressions, not noise
- Configurable thresholds: warn and critical per metric
- Handles both quality metrics (lower = regression) and system metrics (higher = regression)
- Produces structured reports: verdict, per-metric delta, p-value, severity classification

### 🔬 Failure Clustering
- Embeds all failed/low-scoring run outputs using **sentence-transformers** (all-MiniLM-L6-v2)
- Groups semantically similar failures with **HDBSCAN**
- Auto-labels each cluster with an LLM
- Saves clusters to DB — pre-computed clusters shown in dashboard without re-running

### 🖥️ Live Dashboard (5 pages)
- **Run Explorer** — all runs with score, status, latency, cost, tokens, search + filter
- **Trace Viewer** — step-by-step execution breakdown with eval scores per run
- **Charts** — score trends, pass rates, latency, token usage, score distribution
- **Clusters** — failure cluster cards with LLM labels, sample outputs, run ID links
- **Regression** — side-by-side version comparison with statistical significance

### 📤 Export API
- GET /export/csv — download all eval results as CSV
- GET /export/json — download all eval results as JSON
- GET /api/traces — raw traces
- GET /api/evals — raw evaluations
- GET /api/regression — regression report

### 🔄 GitHub Actions CI
- Runs 5-test eval suite on every push to main
- Unit tests (87 passing) run before LLM eval suite
- Build fails if any score drops below 0.7 threshold
- Uploads CI report as workflow artifact

---

## Architecture

```
agentlens/
│
├── tracer/                  # Instrumentation layer
│   ├── tracer.py            # @tracer.trace() decorator
│   ├── models.py            # AgentTrace + Step dataclasses
│   ├── database.py          # SQLite read/write layer
│   ├── regression.py        # Version comparison + statistical testing
│   └── retriever.py         # FAISS vector retriever
│
├── evaluator/               # Evaluation engine
│   ├── rules.py             # Rule-based metric checks
│   ├── llm_judge.py         # LLM-as-judge evaluators
│   ├── scorer.py            # Unified scorer + DB persistence
│   └── clustering.py        # HDBSCAN failure clustering
│
├── agents/                  # 3 validated agent architectures
│   ├── react_agent.py       # ReAct loop with tool use
│   ├── rag_agent.py         # RAG with FAISS retrieval (v2.0)
│   └── planner_agent.py     # Multi-step task planner (v2.0 fixed)
│
├── dashboard/               # Web UI
│   ├── app.py               # FastAPI routes
│   └── templates/           # Jinja2 HTML templates (5 pages)
│
├── tests/
│   ├── test_rules.py        # 37 unit tests
│   ├── test_regression.py   # 18 unit tests
│   ├── test_tracer.py       # 15 unit tests
│   ├── test_retriever.py    # 17 unit tests
│   └── ci_eval.py           # GitHub Actions eval runner
│
├── agentlens.db             # SQLite database
├── Dockerfile               # Railway deployment
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
git clone https://github.com/princemittalr/agentlens.git
cd agentlens
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### Run agents

```bash
python3 agents/react_agent.py
python3 agents/rag_agent.py
python3 agents/planner_agent.py
python3 evaluator/clustering.py
```

### Run tests

```bash
python3 -m pytest tests/test_rules.py tests/test_regression.py tests/test_tracer.py -v
```

### Start dashboard

```bash
python3 -m uvicorn dashboard.app:app --reload --port 8000
```

---

## Instrument Your Own Agent

```python
from tracer import Tracer, Step

tracer = Tracer(project="my-project")

@tracer.trace(agent="my-agent", version="v1.0", model="qwen/qwen3.8-27b")
def my_agent(query: str, _trace=None) -> str:
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": query}]
    )
    if _trace:
        _trace.prompt_tokens = response.usage.prompt_tokens
        _trace.completion_tokens = response.usage.completion_tokens
        _trace.total_tokens = response.usage.total_tokens
    return response.choices[0].message.content
```

---

## Real Bugs Surfaced

AgentLens caught **6 real bugs** automatically across 3 agent architectures:

### ReAct Agent
| Bug | Metric | Details |
|-----|--------|---------|
| Step limit exhaustion | `task_success: 0.0` | Agent used all 5 steps without answering |
| Terse calculator output | `coherence: 0.0` | Bare "1200" flagged as incoherent |

### RAG Agent
| Bug | Metric | Details |
|-----|--------|---------|
| Speculative claim as fact | `hallucination: 0.0` | Unverified doc claim presented as truth |

### Planner Agent
| Bug | Metric | Details |
|-----|--------|---------|
| Broken input substitution | `task_success: 0.0` | Steps received wrong inputs (fixed in v2.0) |
| Fragment outputs | `coherence: 0.0` | "le texte en français" — not a translation |
| Fabricated completions | `hallucination: 0.0` | Reported completing steps never executed |

---

## Regression Detection

```
react-agent v1.0 → v2.0

  Metric         Baseline  Candidate   Delta%   Status
  coherence        0.5000    0.0000   -100.0%   🔴 CRITICAL
  hallucination    1.0000    0.7500    -25.0%   🔴 CRITICAL
  output_length    0.8750    0.5000    -42.9%   🔴 CRITICAL
  overall_score    0.8906    0.7656    -14.0%   🔴 CRITICAL
  latency_ms    1463.7ms   506.3ms    -65.4%   🟢 IMPROVEMENT
  cost_usd         0.0003    0.0000   -100.0%   🟢 IMPROVEMENT

  Verdict: REGRESSION 🔴
```

v2.0 was faster and cheaper — a naive monitor would have shipped it. AgentLens caught it was fundamentally broken.

---

## Failure Clustering

5 failure patterns discovered automatically:

| Cluster | Label | Runs |
|---------|-------|------|
| 0 | Fails by outputting only a single number or word | 3 |
| 1 | Refuses to answer due to insufficient document information | 2 |
| 2 | Fails by outputting fragments instead of complete sentences | 4 |
| 3 | Fails when empty input triggers JSON parse errors | 3 |
| 4 | Confuses general LLM concepts with specific tool capabilities | 3 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| LLM Provider | Groq API (qwen/qwen3.8-27b) |
| Backend | FastAPI + Uvicorn |
| Vector Search | FAISS + sentence-transformers |
| Clustering | HDBSCAN |
| Statistics | SciPy (Mann-Whitney U) |
| Database | SQLite |
| Deployment | Railway (Docker) |
| CI/CD | GitHub Actions |
| Testing | pytest (87 tests) |

---

## Roadmap

- [x] Decorator-based instrumentation
- [x] Rule-based + LLM-judge evaluation
- [x] Regression detection engine
- [x] FAISS vector retriever
- [x] HDBSCAN failure clustering
- [x] Live dashboard (5 pages)
- [x] GitHub Actions CI
- [x] Railway deployment
- [x] 87-test pytest suite
- [x] Export API (CSV + JSON)
- [x] Statistical significance testing
- [ ] Async evaluation pipeline
- [ ] Multi-model comparison
- [ ] Real-time WebSocket dashboard
- [ ] Prompt diff viewer
- [ ] PostgreSQL backend
- [ ] OpenTelemetry export

---

## Author

**Prince Mittal R**

- GitHub: [@princemittalr](https://github.com/princemittalr)
- Live Demo: [agentlens-production-29ad.up.railway.app](https://agentlens-production-29ad.up.railway.app)