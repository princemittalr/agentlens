import sys
import os
import json
sys.path.insert(0, "/home/prince-mittal/agentlens")

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/prince-mittal/agentlens/.env")

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
import sqlite3

from tracer.database import get_all_traces, get_trace_by_id
from tracer.regression import compare_versions, fetch_evals_for_version
from evaluator.clustering import run_failure_clustering, load_clusters

app = FastAPI(title="AgentLens Dashboard")

BASE_DIR = os.path.dirname(__file__)
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH = os.path.join(BASE_DIR, "..", "agentlens.db")

jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

# Add custom filter for JSON parsing in templates
jinja_env.filters['from_json'] = json.loads


def render(template_name: str, context: dict) -> HTMLResponse:
    template = jinja_env.get_template(template_name)
    html = template.render(**context)
    return HTMLResponse(content=html)


def get_all_evals():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_agent_versions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT agent_name, agent_version
        FROM evaluations
        ORDER BY agent_name, agent_version
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_distinct_agents():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT agent_name FROM evaluations ORDER BY agent_name"
    ).fetchall()
    conn.close()
    return [r["agent_name"] for r in rows]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    traces = get_all_traces()
    evals = get_all_evals()

    eval_map = {e["run_id"]: e for e in evals}
    for t in traces:
        ev = eval_map.get(t["run_id"])
        if ev:
            t["overall_score"] = round(ev["overall_score"], 3)
            t["eval_passed"] = bool(ev["passed"])
        else:
            t["overall_score"] = None
            t["eval_passed"] = None
        input_data = json.loads(t["input"])
        args = input_data.get("args", [])
        t["input_preview"] = args[0][:60] if args else ""

    total_runs = len(traces)
    evaluated = len([t for t in traces if t["overall_score"] is not None])
    passed = len([t for t in traces if t["eval_passed"]])
    avg_score = round(
        sum(t["overall_score"] for t in traces if t["overall_score"] is not None) / evaluated, 3
    ) if evaluated else 0
    avg_latency = round(
        sum(t["total_latency_ms"] for t in traces) / total_runs, 1
    ) if total_runs else 0

    agents = list(set(t["agent_name"] for t in traces))

    return render("index.html", {
        "traces": traces,
        "total_runs": total_runs,
        "evaluated": evaluated,
        "passed": passed,
        "avg_score": avg_score,
        "avg_latency": avg_latency,
        "agents": agents,
    })


@app.get("/trace/{run_id}", response_class=HTMLResponse)
async def trace_detail(request: Request, run_id: str):
    trace = get_trace_by_id(run_id)
    if not trace:
        return HTMLResponse("<h1>Trace not found</h1>", status_code=404)

    trace["steps"] = json.loads(trace["steps"])
    trace["input"] = json.loads(trace["input"])
    trace["tags"] = json.loads(trace.get("tags", "{}"))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()

    eval_data = None
    if row:
        eval_data = dict(row)
        eval_data["rule_results"] = json.loads(eval_data["rule_results"])
        eval_data["judge_results"] = json.loads(eval_data["judge_results"])

    return render("trace.html", {
        "trace": trace,
        "eval_data": eval_data,
    })


@app.get("/clusters", response_class=HTMLResponse)
async def clusters_view(request: Request, agent: str = None):
    agents = get_distinct_agents()
    clusters = load_clusters(agent_name=agent or "all")

    total_clustered = sum(
        len(json.loads(c["run_ids"])) for c in clusters
    )
    noise_count = 0  # updated after clustering run

    return render("clusters.html", {
        "clusters": clusters,
        "agents": agents,
        "selected_agent": agent or "all",
        "total_clustered": total_clustered,
        "noise_count": noise_count,
    })


@app.post("/api/clusters/run")
async def api_run_clustering(agent: str = "all"):
    try:
        target = None if agent == "all" else agent
        result = run_failure_clustering(
            agent_name=target,
            threshold=0.9,
            min_cluster_size=2,
            verbose=False
        )
        return {
            "num_clusters": result.num_clusters,
            "total_failures": result.total_failures,
            "noise_count": result.noise_count,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "size": c.size,
                    "avg_score": c.avg_score,
                }
                for c in result.clusters
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/regression", response_class=HTMLResponse)
async def regression_view(
    request: Request,
    agent: str = None,
    baseline: str = None,
    candidate: str = None,
):
    versions = get_agent_versions()
    report = None
    error = None

    if agent and baseline and candidate:
        try:
            report = compare_versions(agent, baseline, candidate, verbose=False)
        except Exception as e:
            error = str(e)

    return render("regression.html", {
        "versions": versions,
        "report": report,
        "error": error,
        "selected_agent": agent,
        "selected_baseline": baseline,
        "selected_candidate": candidate,
    })


@app.get("/api/traces")
async def api_traces():
    return get_all_traces()


@app.get("/api/evals")
async def api_evals():
    return get_all_evals()


@app.get("/api/regression")
async def api_regression(agent: str, baseline: str, candidate: str):
    try:
        report = compare_versions(agent, baseline, candidate, verbose=False)
        return {
            "verdict": report.verdict,
            "alerts": [a.__dict__ for a in report.alerts],
            "improved": [a.__dict__ for a in report.improved],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/charts", response_class=HTMLResponse)
async def charts_view(request: Request):
    traces = get_all_traces()
    evals = get_all_evals()

    eval_map = {e["run_id"]: e for e in evals}
    total_runs = len(traces)
    evaluated = len(evals)
    passed = sum(1 for e in evals if e["passed"])
    pass_rate = round((passed / evaluated * 100), 1) if evaluated else 0
    avg_score = round(
        sum(e["overall_score"] for e in evals) / evaluated, 3
    ) if evaluated else 0
    total_cost = round(sum(t["cost_usd"] for t in traces), 6)

    # Per-agent stats
    from collections import defaultdict
    agent_data = defaultdict(lambda: {
        "scores": [], "latencies": [],
        "prompt_tokens": [], "completion_tokens": [],
        "passed": 0, "total": 0
    })

    for t in traces:
        a = t["agent_name"]
        agent_data[a]["latencies"].append(t["total_latency_ms"])
        agent_data[a]["prompt_tokens"].append(t["prompt_tokens"])
        agent_data[a]["completion_tokens"].append(t["completion_tokens"])
        ev = eval_map.get(t["run_id"])
        if ev:
            agent_data[a]["scores"].append(ev["overall_score"])
            agent_data[a]["total"] += 1
            if ev["passed"]:
                agent_data[a]["passed"] += 1

    agent_stats = []
    for agent_name, d in sorted(agent_data.items()):
        scores = d["scores"]
        lats = d["latencies"]
        pt = d["prompt_tokens"]
        ct = d["completion_tokens"]
        agent_stats.append({
            "agent_name": agent_name,
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "pass_rate": round(d["passed"] / d["total"] * 100, 1) if d["total"] else 0,
            "avg_latency": round(sum(lats) / len(lats), 1) if lats else 0,
            "avg_prompt_tokens": round(sum(pt) / len(pt), 1) if pt else 0,
            "avg_completion_tokens": round(sum(ct) / len(ct), 1) if ct else 0,
        })

    # Score distribution
    all_scores = [e["overall_score"] for e in evals]
    score_distribution = {
        "excellent": sum(1 for s in all_scores if s >= 0.9),
        "good": sum(1 for s in all_scores if 0.7 <= s < 0.9),
        "poor": sum(1 for s in all_scores if s < 0.7),
    }

    # Token stats (same as agent_stats but for chart)
    token_stats = [
        {"agent": a["agent_name"],
         "prompt": a["avg_prompt_tokens"],
         "completion": a["avg_completion_tokens"]}
        for a in agent_stats
    ]

    return render("charts.html", {
        "total_runs": total_runs,
        "evaluated": evaluated,
        "passed": passed,
        "pass_rate": pass_rate,
        "avg_score": avg_score,
        "total_cost": total_cost,
        "agent_count": len(agent_stats),
        "agent_stats": agent_stats,
        "score_distribution": score_distribution,
        "token_stats": token_stats,
    })
