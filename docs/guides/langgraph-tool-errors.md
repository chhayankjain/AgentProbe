---
description: Test whether a LangGraph agent survives tool exceptions. ToolNode re-raises tool errors by default; fault tests show it and verify handle_tool_errors.
---

# LangGraph tool errors

By default, LangGraph's `ToolNode` only turns argument-validation errors into messages. **Any other exception a tool raises ends the run.** A tool that times out once can crash the whole agent. Fault tests make this visible and verify the fix.

## A small graph

This graph uses AgentProbe's scripted chat model, so it runs offline. In your tests, build your graph as usual, with your own model.

```python
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agentprobe.fake_llm import ScriptedToolCallingModel
from agentprobe.injector.tool_failure import ToolCallTimeout
from agentprobe.testing import probe


@tool
def payment_status(order_id: str) -> str:
    """Payment status for an order."""
    return f"Order {order_id} is paid."


def build_agent(tools, handle_tool_errors=None):
    model = ScriptedToolCallingModel(tool_name="payment_status", tool_args={"order_id": "42"})
    node = ToolNode(tools) if handle_tool_errors is None else ToolNode(
        tools, handle_tool_errors=handle_tool_errors
    )
    graph = StateGraph(MessagesState)
    graph.add_node("agent", lambda s: {"messages": [model.bind_tools(tools).invoke(s["messages"])]})
    graph.add_node("tools", node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


QUESTION = {"messages": [("user", "Did order 42 go through?")]}
```

## The default crashes on a tool timeout

```python
def test_default_tool_node_ends_the_run_on_a_timeout():
    with probe() as p:
        tools = p.wrap([payment_status])
        p.on("payment_status").fail_first(1)
        result = p.run(build_agent(tools), QUESTION)
    assert isinstance(result.exception, ToolCallTimeout)
    assert result.answer is None
```

In a real test suite you would write `result.assert_no_crash()` and watch it fail. The report shows the injected timeout as the last call and no answer.

## With error handling, the model sees the error

```python
def test_handled_tool_errors_reach_the_model():
    with probe() as p:
        tools = p.wrap([payment_status])
        p.on("payment_status").always_fail(with_="timeout")
        result = p.run(build_agent(tools, handle_tool_errors=True), QUESTION)
        result.assert_no_crash()
        result.assert_answer_contains("could not complete")
```

`handle_tool_errors=True` turns every tool exception into a `ToolMessage`, so the model can retry or explain. You can also pass an exception type or a tuple of types to handle only those, or a function that writes the message.

## Wrapping LangChain tools

`p.wrap([...])` returns copies of LangChain tools with the same name, description and argument schema, so the model sees exactly the same tools. Faults are declared by tool name. If your graph builds its tools at import time, see the module-swap pattern in the [case study](../case-studies/agents-from-scratch.md).
