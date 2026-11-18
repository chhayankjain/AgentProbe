"""Rerun the model x prompt matrix behind the README table and save the evidence.

    python examples/real_model_agent/run_matrix.py --repeats 3

Each scenario applies the same faults and the same checks as test_shop_agent.py.
Every run is saved with its call log, answer, failure reasons, the prompt and
error-message variant used, model digests and package versions, so the table can
be checked without rerunning it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import urllib.request
from datetime import UTC, datetime
from importlib import metadata

from checks import recovered_correctly
from shop_agent import ERROR_MESSAGES, PROMPTS, TOOLS, build_agent

from agentprobe.testing import probe

QUESTION = {"messages": [("user", "Did my order 42 go through? If it did, send me the receipt.")]}
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _check(fn) -> str | None:
    try:
        fn()
    except AssertionError as exc:
        return str(exc).splitlines()[0]
    return None


def recover(p, result):
    return [
        _check(result.assert_no_crash),
        recovered_correctly(result.calls, order_id="42"),
    ]


def outage(p, result):
    return [
        _check(result.assert_no_crash),
        _check(lambda: result.assert_not_called("send_receipt")),
    ]


def pending(p, result):
    return [
        _check(result.assert_no_crash),
        _check(lambda: result.assert_not_called("send_receipt")),
    ]


SCENARIOS = {
    "recovers_from_one_timeout": (lambda p: p.on("payment_status").fail_first(1), recover),
    "outage_sends_no_receipt": (lambda p: p.on("payment_status").always_fail(), outage),
    "pending_gets_no_receipt": (
        lambda p: p.on("payment_status").returns("pending", times=None),
        pending,
    ),
}


def run_once(model, prompt, hint, scenario):
    setup, verdict = SCENARIOS[scenario]
    p = probe(enforce=False)
    with p:
        agent = build_agent(p.wrap(TOOLS), model=model, prompt=prompt, error_hint=hint)
        setup(p)
        result = p.run(agent, QUESTION, timeout=180)
        problems = [x for x in verdict(p, result) if x]
    gate = p.gate()
    problems += [f for f in gate.failures if not f.startswith("assertion ")]
    observed = result.to_dict()["observed"]
    return {
        "passed": not problems,
        "problems": problems,
        "answer": result.answer,
        "exception": observed["exception"],
        "elapsed_ms": observed["elapsed_ms"],
        "calls": observed["calls"],
        "scenarios": observed["scenarios"],
    }


def _get(path):
    with urllib.request.urlopen(f"{OLLAMA}{path}", timeout=5) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5:7b", "llama3.1:latest"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--out", default=os.path.join(os.path.dirname(__file__), "results", "matrix.json")
    )
    args = ap.parse_args()

    tags = {m["name"]: m for m in _get("/api/tags")["models"]}
    configs = [(m, pr, True, s) for m in args.models for pr in PROMPTS for s in SCENARIOS]
    # The error-message claim: the recovery scenario with the hint turned off.
    configs += [(m, pr, False, "recovers_from_one_timeout") for m in args.models for pr in PROMPTS]

    runs = []
    for model, prompt, hint, scenario in configs:
        for i in range(args.repeats):
            r = run_once(model, prompt, hint, scenario)
            runs.append(
                {
                    "model": model,
                    "prompt": prompt,
                    "error_hint": hint,
                    "scenario": scenario,
                    "repeat": i + 1,
                    **r,
                }
            )
            print(
                f"{model:16} {prompt:9} hint={int(hint)} {scenario:28} #{i + 1}: "
                + ("pass" if r["passed"] else "FAIL"),
                flush=True,
            )

    pkgs = ["agentprobe-toolkit", "langchain", "langchain-core", "langgraph", "langchain-ollama"]
    out = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": QUESTION["messages"][0][1],
        "temperature": 0,
        "repeats": args.repeats,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {p: metadata.version(p) for p in pkgs},
            "ollama_version": _get("/api/version").get("version"),
            "models": {
                m: {
                    "digest": tags[m]["digest"],
                    **{
                        k: tags[m]["details"].get(k)
                        for k in ("parameter_size", "quantization_level")
                    },
                }
                for m in args.models
            },
        },
        "prompts": PROMPTS,
        "error_messages": {"hint": ERROR_MESSAGES[True], "plain": ERROR_MESSAGES[False]},
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {len(runs)} runs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
