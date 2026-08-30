from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


@dataclass
class Step:
    step_index: int
    type: str                    # "llm_call", "tool_call", "thought"
    input: str
    output: str
    latency_ms: float
    tokens_used: int = 0
    tool_name: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentTrace:
    agent_name: str
    agent_version: str
    model: str
    input: Dict[str, Any]
    output: str
    steps: List[Step] = field(default_factory=list)
    
    # Auto-generated
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Performance
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    
    # Status
    success: bool = True
    error: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
