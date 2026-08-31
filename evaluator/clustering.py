import os
import sys
import json
import sqlite3
sys.path.insert(0, "/home/prince-mittal/agentlens")

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/prince-mittal/agentlens/.env")

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from groq import Groq

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "agentlens.db")


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class FailureCluster:
    cluster_id: int
    label: str
    size: int
    run_ids: List[str]
    sample_outputs: List[str]
    avg_score: float
    common_agent: str


@dataclass
class ClusteringResult:
    total_failures: int
    num_clusters: int
    noise_count: int
    clusters: List[FailureCluster]


# ─── Fetch Runs ───────────────────────────────────────────────────────────────

def fetch_failed_runs(agent_name: Optional[str] = None, threshold: float = 0.9) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if agent_name:
        rows = conn.execute("""
            SELECT e.run_id, e.agent_name, e.agent_version, e.overall_score,
                   t.output, t.error
            FROM evaluations e
            JOIN traces t ON e.run_id = t.run_id
            WHERE e.overall_score < ? AND e.agent_name = ?
            ORDER BY e.overall_score ASC
        """, (threshold, agent_name)).fetchall()
    else:
        rows = conn.execute("""
            SELECT e.run_id, e.agent_name, e.agent_version, e.overall_score,
                   t.output, t.error
            FROM evaluations e
            JOIN traces t ON e.run_id = t.run_id
            WHERE e.overall_score < ?
            ORDER BY e.overall_score ASC
        """, (threshold,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Embed ────────────────────────────────────────────────────────────────────

def embed_outputs(texts: List[str]) -> np.ndarray:
    print("  Loading embedding model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Embedding {len(texts)} outputs...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)


# ─── Cluster ─────────────────────────────────────────────────────────────────

def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    import hdbscan
    print(f"  Clustering {len(embeddings)} embeddings with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric="euclidean",
    )
    return clusterer.fit_predict(embeddings)


# ─── LLM Label ───────────────────────────────────────────────────────────────

def label_cluster(sample_outputs: List[str], agent_name: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    samples_text = "\n---\n".join(sample_outputs[:3])
    prompt = f"""You are analyzing failure patterns in an AI agent called '{agent_name}'.

Here are failed outputs from the same failure cluster:

{samples_text}

In ONE short sentence (under 12 words), describe the common failure pattern.
Examples:
- "Fails when output is too short or single-word"
- "Fails on multi-step reasoning with tool exhaustion"
- "Produces fragments instead of complete translations"

Respond with ONLY the one-sentence label, nothing else."""

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip().strip('"').strip("'")


# ─── DB ──────────────────────────────────────────────────────────────────────

def ensure_clusters_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failure_clusters (
            cluster_id INTEGER,
            agent_name TEXT,
            label TEXT,
            size INTEGER,
            run_ids TEXT,
            sample_outputs TEXT,
            avg_score REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_clusters(clusters: List[FailureCluster], agent_name: str):
    from datetime import datetime
    ensure_clusters_table()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM failure_clusters WHERE agent_name = ?", (agent_name,))
    for c in clusters:
        conn.execute("""
            INSERT INTO failure_clusters VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            c.cluster_id,
            agent_name,
            c.label,
            c.size,
            json.dumps(c.run_ids),
            json.dumps(c.sample_outputs),
            c.avg_score,
            datetime.utcnow().isoformat()
        ))
    conn.commit()
    conn.close()


def load_clusters(agent_name: Optional[str] = None) -> List[dict]:
    ensure_clusters_table()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if agent_name:
        rows = conn.execute(
            "SELECT * FROM failure_clusters WHERE agent_name = ? ORDER BY size DESC",
            (agent_name,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM failure_clusters ORDER BY size DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_failure_clustering(
    agent_name: Optional[str] = None,
    threshold: float = 0.9,
    min_cluster_size: int = 2,
    verbose: bool = True
) -> ClusteringResult:

    failed_runs = fetch_failed_runs(agent_name, threshold)

    if len(failed_runs) < 2:
        print(f"  Not enough runs to cluster ({len(failed_runs)} found).")
        return ClusteringResult(
            total_failures=len(failed_runs),
            num_clusters=0,
            noise_count=len(failed_runs),
            clusters=[]
        )

    print(f"\n  Found {len(failed_runs)} runs to cluster (score < {threshold}).")

    texts = [r["output"] or r["error"] or "empty output" for r in failed_runs]
    embeddings = embed_outputs(texts)
    labels = cluster_embeddings(embeddings, min_cluster_size)

    from collections import defaultdict
    cluster_map = defaultdict(list)
    for i, label in enumerate(labels):
        if label != -1:
            cluster_map[int(label)].append(i)

    noise_count = int(np.sum(labels == -1))
    clusters = []

    for cluster_id, indices in sorted(cluster_map.items()):
        runs_in_cluster = [failed_runs[i] for i in indices]
        sample_outputs = [r["output"] or "empty" for r in runs_in_cluster[:3]]
        avg_score = round(
            sum(r["overall_score"] for r in runs_in_cluster) / len(runs_in_cluster), 3
        )
        agent = runs_in_cluster[0]["agent_name"]

        print(f"\n  Labeling cluster {cluster_id} ({len(runs_in_cluster)} runs)...")
        label = label_cluster(sample_outputs, agent)

        clusters.append(FailureCluster(
            cluster_id=cluster_id,
            label=label,
            size=len(runs_in_cluster),
            run_ids=[r["run_id"] for r in runs_in_cluster],
            sample_outputs=sample_outputs,
            avg_score=avg_score,
            common_agent=agent,
        ))

    target_agent = agent_name or "all"
    save_clusters(clusters, target_agent)

    result = ClusteringResult(
        total_failures=len(failed_runs),
        num_clusters=len(clusters),
        noise_count=noise_count,
        clusters=clusters,
    )

    if verbose:
        _print_result(result)

    return result


def _print_result(result: ClusteringResult):
    print(f"\n{'='*62}")
    print(f"  AgentLens — Failure Cluster Analysis")
    print(f"{'='*62}")
    print(f"  Total low-scoring runs : {result.total_failures}")
    print(f"  Clusters found         : {result.num_clusters}")
    print(f"  Noise (unclustered)    : {result.noise_count}")

    if result.clusters:
        print(f"\n  Clusters:")
        for c in result.clusters:
            print(f"\n  ┌─ Cluster {c.cluster_id} [{c.size} runs] — avg score: {c.avg_score}")
            print(f"  │  Agent  : {c.common_agent}")
            print(f"  │  Label  : {c.label}")
            print(f"  └─ Sample : {c.sample_outputs[0][:80]}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    print("Running failure clustering on all agents...")
    run_failure_clustering(threshold=0.9, min_cluster_size=2)
