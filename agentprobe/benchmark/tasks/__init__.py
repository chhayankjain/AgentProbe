"""Standard benchmark tasks for AgentProbe.

Tasks are designed to be:
  1. Reproducible — deterministic inputs, public data sources
  2. Comparable — same inputs applied across all frameworks
  3. Realistic — representative of production agentic use cases

Task categories:
  - tool_use     : Single-agent, multi-tool calling
  - rag          : Retrieval-augmented generation
  - multi_agent  : Two-agent researcher + writer pipeline
"""

from agentprobe.benchmark.tasks.tool_use import ToolUseTask
from agentprobe.benchmark.tasks.rag import RAGTask
from agentprobe.benchmark.tasks.multi_agent import MultiAgentTask

__all__ = ["ToolUseTask", "RAGTask", "MultiAgentTask"]
