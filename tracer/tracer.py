import time
import functools
from typing import Callable, Any, Dict, Optional
from datetime import datetime
from .models import AgentTrace, Step
from .database import init_db, save_trace


# Groq pricing (per 1M tokens) — update as needed
COST_PER_1M_TOKENS = {
    "qwen/qwen3.8-27b": {"input": 0.29, "output": 0.59},
    "default": {"input": 0.30, "output": 0.60},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = COST_PER_1M_TOKENS.get(model, COST_PER_1M_TOKENS["default"])
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)


class Tracer:
    def __init__(self, project: str = "default"):
        self.project = project
        init_db()

    def trace(self, agent: str, version: str = "v1.0", model: str = "qwen/qwen3.8-27b", tags: Dict[str, str] = {}):
        """Decorator to wrap any agent function and capture its trace."""
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Build input dict
                input_data = {}
                if args:
                    input_data["args"] = [str(a) for a in args]
                if kwargs:
                    input_data["kwargs"] = {k: str(v) for k, v in kwargs.items()}
                if not input_data and args:
                    input_data["query"] = str(args[0])

                trace = AgentTrace(
                    agent_name=agent,
                    agent_version=version,
                    model=model,
                    input=input_data,
                    output="",
                    tags=tags,
                )

                try:
                    result = func(*args, **kwargs, _trace=trace)
                    trace.output = str(result) if result else ""
                    trace.success = True
                except Exception as e:
                    trace.output = ""
                    trace.error = str(e)
                    trace.success = False
                    result = None
                finally:
                    end_time = time.time()
                    trace.total_latency_ms = round((end_time - start_time) * 1000, 2)
                    trace.cost_usd = calculate_cost(model, trace.prompt_tokens, trace.completion_tokens)
                    save_trace(trace)
                    self._print_summary(trace)

                return result
            return wrapper
        return decorator

    def _print_summary(self, trace: AgentTrace):
        status = "✅ SUCCESS" if trace.success else "❌ FAILED"
        print(f"\n{'='*50}")
        print(f"AgentLens Trace [{status}]")
        print(f"  Run ID     : {trace.run_id}")
        print(f"  Agent      : {trace.agent_name} ({trace.agent_version})")
        print(f"  Model      : {trace.model}")
        print(f"  Latency    : {trace.total_latency_ms} ms")
        print(f"  Tokens     : {trace.total_tokens} (prompt={trace.prompt_tokens}, completion={trace.completion_tokens})")
        print(f"  Cost       : ${trace.cost_usd}")
        print(f"  Steps      : {len(trace.steps)}")
        if trace.error:
            print(f"  Error      : {trace.error}")
        print(f"{'='*50}\n")
