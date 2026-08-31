"""
Unit tests for tracer/models.py and tracer/database.py
Tests AgentTrace creation, Step creation, and DB persistence.
"""
import sys
import os
import uuid
import tempfile
sys.path.insert(0, "/home/prince-mittal/agentlens")

import pytest
from datetime import datetime
from tracer.models import AgentTrace, Step


# ─── Step ────────────────────────────────────────────────────────────────────

class TestStep:
    def test_basic_creation(self):
        step = Step(
            step_index=0,
            type="llm_call",
            input="What is Python?",
            output="Python is a programming language.",
            latency_ms=250.5,
        )
        assert step.step_index == 0
        assert step.type == "llm_call"
        assert step.latency_ms == 250.5
        assert step.tokens_used == 0
        assert step.tool_name is None
        assert step.error is None

    def test_tool_call_step(self):
        step = Step(
            step_index=1,
            type="tool_call",
            input="Paris",
            output="Paris is the capital of France.",
            latency_ms=100.0,
            tool_name="search",
            tokens_used=50,
        )
        assert step.tool_name == "search"
        assert step.tokens_used == 50

    def test_step_with_error(self):
        step = Step(
            step_index=2,
            type="tool_call",
            input="bad input",
            output="",
            latency_ms=10.0,
            error="Connection timeout",
        )
        assert step.error == "Connection timeout"


# ─── AgentTrace ──────────────────────────────────────────────────────────────

class TestAgentTrace:
    def test_basic_creation(self):
        trace = AgentTrace(
            agent_name="test-agent",
            agent_version="v1.0",
            model="qwen/qwen3.8-27b",
            input={"query": "test"},
            output="test output",
        )
        assert trace.agent_name == "test-agent"
        assert trace.agent_version == "v1.0"
        assert trace.success is True
        assert trace.error is None
        assert trace.total_tokens == 0
        assert trace.cost_usd == 0.0

    def test_auto_run_id_generated(self):
        trace = AgentTrace(
            agent_name="agent",
            agent_version="v1.0",
            model="model",
            input={},
            output="",
        )
        assert trace.run_id is not None
        assert len(trace.run_id) == 36  # UUID format

    def test_unique_run_ids(self):
        t1 = AgentTrace(agent_name="a", agent_version="v1", model="m", input={}, output="")
        t2 = AgentTrace(agent_name="a", agent_version="v1", model="m", input={}, output="")
        assert t1.run_id != t2.run_id

    def test_auto_timestamp(self):
        before = datetime.utcnow()
        trace = AgentTrace(agent_name="a", agent_version="v1", model="m", input={}, output="")
        after = datetime.utcnow()
        assert before <= trace.timestamp <= after

    def test_steps_default_empty(self):
        trace = AgentTrace(agent_name="a", agent_version="v1", model="m", input={}, output="")
        assert trace.steps == []

    def test_steps_not_shared_between_instances(self):
        t1 = AgentTrace(agent_name="a", agent_version="v1", model="m", input={}, output="")
        t2 = AgentTrace(agent_name="b", agent_version="v1", model="m", input={}, output="")
        t1.steps.append(Step(0, "llm_call", "in", "out", 100))
        assert len(t2.steps) == 0

    def test_tags_default_empty(self):
        trace = AgentTrace(agent_name="a", agent_version="v1", model="m", input={}, output="")
        assert trace.tags == {}

    def test_failed_trace(self):
        trace = AgentTrace(
            agent_name="agent",
            agent_version="v1.0",
            model="model",
            input={"q": "test"},
            output="",
            success=False,
            error="API timeout",
        )
        assert trace.success is False
        assert trace.error == "API timeout"

    def test_with_token_counts(self):
        trace = AgentTrace(
            agent_name="a", agent_version="v1", model="m", input={}, output="result",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.0001,
        )
        assert trace.prompt_tokens == 100
        assert trace.completion_tokens == 50
        assert trace.total_tokens == 150
        assert trace.cost_usd == 0.0001


# ─── Cost calculation ────────────────────────────────────────────────────────

class TestCostCalculation:
    def test_cost_calculation(self):
        from tracer.tracer import calculate_cost
        cost = calculate_cost("qwen/qwen3.8-27b", prompt_tokens=1000, completion_tokens=500)
        assert cost > 0
        assert cost < 0.01  # sanity check

    def test_zero_tokens_zero_cost(self):
        from tracer.tracer import calculate_cost
        cost = calculate_cost("qwen/qwen3.8-27b", 0, 0)
        assert cost == 0.0

    def test_default_pricing_fallback(self):
        from tracer.tracer import calculate_cost
        cost = calculate_cost("unknown-model", 1000, 1000)
        assert cost > 0
