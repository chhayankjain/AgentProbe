"""`pip install agentprobe-toolkit` (no extras) must be enough for agentprobe.testing.

Each test runs a fresh interpreter in which the optional dependencies cannot be
imported, as on a minimal install, even though the test environment has them.
CI also installs the wheel without extras in a clean virtualenv (ci.yml).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Everything that used to be a required dependency.
OPTIONAL = [
    "langchain",
    "langchain_core",
    "langgraph",
    "opentelemetry",
    "prometheus_client",
    "grpc",
    "pydantic",
    "rich",
    "numpy",
    "scipy",
]

#: What each extra installs, as measured from real installs of the wheel
#: (langchain-core brings pydantic; the OpenTelemetry exporter brings grpc).
PROVIDED = {
    "langchain": {"langchain", "langchain_core", "langgraph", "pydantic"},
    "observability": {"opentelemetry", "prometheus_client", "grpc"},
}

BLOCKER = """
import importlib.abc, sys
BLOCKED = {blocked!r}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None
sys.meta_path.insert(0, Blocker())
"""


def run_minimal(code: str, extra: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run *code* as if only the core (and *extra*, if given) were installed."""
    blocked = sorted(set(OPTIONAL) - PROVIDED.get(extra or "", set()))
    script = BLOCKER.format(blocked=blocked) + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_probe_works_without_optional_dependencies():
    result = run_minimal(
        """
        import sys
        import agentprobe
        from agentprobe.testing import ToolCallTimeout, probe

        def payment_status(order_id):
            return f"Order {order_id} is paid."

        def answer(order_id, tools):
            for attempt in range(2):
                try:
                    return tools["payment_status"](order_id)
                except TimeoutError:
                    if attempt == 1:
                        return "I couldn't confirm."

        with probe() as p:
            tools = p.wrap({"payment_status": payment_status})
            p.on("payment_status").fail_first(1)
            result = p.run(lambda q: answer(q, tools), "42")
            result.assert_no_crash().assert_answer_contains("is paid")
            result.assert_called_times("payment_status", 2)

        first = result.calls[0]
        assert first.exception_type == "ToolCallTimeout", first
        assert first.span_id == "0" * 16, first.span_id  # no-op tracer
        assert isinstance(result.to_dict(), dict)
        loaded = sorted({m.split(".")[0] for m in sys.modules} & set(BLOCKED))
        assert not loaded, loaded
        print("ok", agentprobe.__version__)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ok ")


def test_async_probe_works_without_optional_dependencies():
    result = run_minimal(
        """
        import asyncio
        from agentprobe.testing import probe

        async def search(q):
            return "results"

        async def main():
            async with probe() as p:
                tools = p.wrap({"search": search})
                p.on("search").returns("stale")
                result = await p.arun(lambda: tools["search"]("x"), timeout=5)
                result.assert_answer_contains("stale")

        asyncio.run(main())
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr


def test_research_features_name_the_extra_to_install():
    result = run_minimal(
        """
        import agentprobe
        needs = {
            "BenchmarkRunner": "research",
            "ToolFailureInjector": "observability",
            "AgentProbeTracer": "observability",
            "FailureClassifier": "observability",
        }
        for name, extra in needs.items():
            try:
                getattr(agentprobe, name)
            except ImportError as exc:
                assert f'pip install "agentprobe-toolkit[{extra}]"' in str(exc), (name, exc)
            else:
                raise AssertionError(f"{name} imported without its dependencies")
        for module in ["agentprobe.benchmark", "agentprobe.injector", "agentprobe.observer"]:
            try:
                __import__(module)
            except ImportError as exc:
                assert "agentprobe-toolkit[" in str(exc), exc
            else:
                raise AssertionError(f"{module} imported without its dependencies")
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr


def test_demo_explains_how_to_install_it():
    result = run_minimal("import runpy; runpy.run_module('agentprobe.demo', run_name='__main__')")
    assert result.returncode != 0
    assert 'pip install "agentprobe-toolkit[research]"' in result.stderr


def test_unrelated_missing_modules_are_not_disguised():
    from agentprobe._extras import missing_extra

    exc = ModuleNotFoundError("No module named 'agentprobe.nope'", name="agentprobe.nope")
    assert missing_extra(exc, "x") is exc


ADAPTERS_NEED_RESEARCH = """
for module in ["agentprobe.agents.langgraph_agent", "agentprobe.agents.langchain_agent"]:
    try:
        __import__(module)
    except ImportError as exc:
        assert 'pip install "agentprobe-toolkit[research]"' in str(exc), (module, exc)
    else:
        raise AssertionError(f"{module} imported without OpenTelemetry")
"""


def test_langchain_extra_runs_an_offline_langgraph_fault_test():
    result = run_minimal(
        textwrap.dedent("""
        import sys
        from langchain_core.tools import tool
        from langgraph.graph import START, MessagesState, StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition

        from agentprobe.fake_llm import ScriptedToolCallingModel
        from agentprobe.testing import probe

        @tool
        def payment_status(order_id: str) -> str:
            \"\"\"Payment status for an order.\"\"\"
            return f"Order {order_id} is paid."

        with probe() as p:
            tools = p.wrap([payment_status])
            p.on("payment_status").always_fail(with_="timeout")
            model = ScriptedToolCallingModel(
                tool_name="payment_status", tool_args={"order_id": "42"}
            ).bind_tools(tools)
            graph = StateGraph(MessagesState)
            graph.add_node("agent", lambda s: {"messages": [model.invoke(s["messages"])]})
            graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
            graph.add_edge(START, "agent")
            graph.add_conditional_edges("agent", tools_condition)
            graph.add_edge("tools", "agent")
            result = p.run(graph.compile(), {"messages": [("user", "Did order 42 go through?")]})
            result.assert_no_crash().assert_answer_contains("could not complete")

        assert "opentelemetry" not in {m.split(".")[0] for m in sys.modules}
        """)
        + ADAPTERS_NEED_RESEARCH,
        extra="langchain",
    )
    assert result.returncode == 0, result.stderr


def test_observability_extra_runs_the_tracer_injectors_and_classifier():
    result = run_minimal(
        textwrap.dedent("""
        from agentprobe import AgentProbeTracer, FailureClassifier, ToolFailureInjector
        from agentprobe.injector import LatencyFailureInjector
        from agentprobe.injector.tool_failure import ToolFailureConfig, ToolFailureType
        from agentprobe.observer.dashboard import PrometheusExporter

        tracer = AgentProbeTracer(service_name="extras-test", console=False)
        with tracer.span("check"):
            pass
        injector = ToolFailureInjector(ToolFailureConfig(failure_type=ToolFailureType.NONE))
        assert injector.call(lambda: 41) == 41
        assert FailureClassifier().classify_exception(TimeoutError("t")) is not None
        assert PrometheusExporter is not None and LatencyFailureInjector is not None

        import agentprobe
        try:
            agentprobe.BenchmarkRunner
        except ImportError as exc:
            assert 'pip install "agentprobe-toolkit[research]"' in str(exc), exc
        else:
            raise AssertionError("BenchmarkRunner imported without the research extra")
        """)
        + ADAPTERS_NEED_RESEARCH,
        extra="observability",
    )
    assert result.returncode == 0, result.stderr
