import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional
from .models import AgentTrace, Step


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "agentlens.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            run_id TEXT PRIMARY KEY,
            agent_name TEXT NOT NULL,
            agent_version TEXT NOT NULL,
            model TEXT NOT NULL,
            input TEXT NOT NULL,
            output TEXT NOT NULL,
            steps TEXT NOT NULL,
            total_latency_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            cost_usd REAL,
            success INTEGER,
            error TEXT,
            tags TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized at agentlens.db")


# WebSocket broadcast callback — set by dashboard on startup
_ws_broadcast_callback = None


def set_ws_callback(callback):
    global _ws_broadcast_callback
    _ws_broadcast_callback = callback


def save_trace(trace: AgentTrace):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO traces VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        trace.run_id,
        trace.agent_name,
        trace.agent_version,
        trace.model,
        json.dumps(trace.input),
        trace.output,
        json.dumps([s.__dict__ for s in trace.steps]),
        trace.total_latency_ms,
        trace.prompt_tokens,
        trace.completion_tokens,
        trace.total_tokens,
        trace.cost_usd,
        int(trace.success),
        trace.error,
        json.dumps(trace.tags),
        trace.timestamp.isoformat()
    ))

    conn.commit()
    conn.close()




def get_all_traces() -> List[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM traces ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_trace_by_id(run_id: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM traces WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
