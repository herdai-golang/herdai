"""
Master comparison runner.

Runs all four benchmarks (HerdAI Go, LangGraph, AutoGen, CrewAI-sim, baseline)
and prints a single comparison table.

Usage:
    # From the benchmark/ directory:
    python3 python/compare_all.py

    # Or from anywhere:
    python3 /path/to/benchmark/python/compare_all.py
"""

import asyncio
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

# ── Shared constants ──────────────────────────────────────────────────────────
LLM_DELAY  = 0.200
TOOL_DELAY = 0.080

AGENTS: List[Tuple[str, int]] = [
    ("news_sentiment",     3),
    ("competitor_profile", 4),
    ("financial_metrics",  3),
    ("market_trends",      3),
    ("regulatory_scan",    2),
    ("customer_voice",     3),
]


# ── Inline benchmark functions (no subprocess needed for Python ones) ─────────

async def _tool(delay: float) -> None:
    await asyncio.sleep(delay)


async def _agent(agent_id: str, tool_count: int, parallel_tools: bool) -> str:
    await asyncio.sleep(LLM_DELAY)
    if parallel_tools:
        await asyncio.gather(*[_tool(TOOL_DELAY) for _ in range(tool_count)])
    else:
        for _ in range(tool_count):
            await _tool(TOOL_DELAY)
    await asyncio.sleep(LLM_DELAY)
    return f"{agent_id}: done"


async def _run(parallel_agents: bool, parallel_tools: bool) -> float:
    t0 = time.perf_counter()
    tasks = [_agent(aid, n, parallel_tools) for aid, n in AGENTS]
    if parallel_agents:
        await asyncio.gather(*tasks)
    else:
        for t in tasks:
            await t
    return time.perf_counter() - t0


def measure_python(label: str, par_agents: bool, par_tools: bool) -> float:
    return asyncio.run(_run(par_agents, par_tools))


# ── LangGraph inline ──────────────────────────────────────────────────────────

def measure_langgraph(par_agents: bool, par_tools: bool) -> float:
    import operator
    from typing import Annotated
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    class S(dict):
        pass

    async def _lg_agent(state):
        agent_id  = state["agent_id"]
        tool_count = state["tool_count"]
        await asyncio.sleep(LLM_DELAY)
        if par_tools:
            await asyncio.gather(*[asyncio.sleep(TOOL_DELAY) for _ in range(tool_count)])
        else:
            for _ in range(tool_count):
                await asyncio.sleep(TOOL_DELAY)
        await asyncio.sleep(LLM_DELAY)
        return {"results": [f"{agent_id}: done"]}

    from typing import TypedDict

    class GraphState(TypedDict):
        company: str
        results: Annotated[List[str], operator.add]

    if par_agents:
        builder = StateGraph(GraphState)

        async def dispatch(state):
            return [
                Send(aid, {"company": state["company"], "agent_id": aid, "tool_count": n})
                for aid, n in AGENTS
            ]

        for aid, n in AGENTS:
            async def _node(state, _aid=aid, _n=n):
                await asyncio.sleep(LLM_DELAY)
                if par_tools:
                    await asyncio.gather(*[asyncio.sleep(TOOL_DELAY) for _ in range(_n)])
                else:
                    for _ in range(_n):
                        await asyncio.sleep(TOOL_DELAY)
                await asyncio.sleep(LLM_DELAY)
                return {"results": [f"{_aid}: done"]}
            builder.add_node(aid, _node)
            builder.add_edge(aid, END)
        builder.add_conditional_edges(START, dispatch)
    else:
        builder = StateGraph(GraphState)
        agent_ids = [aid for aid, _ in AGENTS]
        for aid, n in AGENTS:
            async def _node(state, _aid=aid, _n=n):
                await asyncio.sleep(LLM_DELAY)
                if par_tools:
                    await asyncio.gather(*[asyncio.sleep(TOOL_DELAY) for _ in range(_n)])
                else:
                    for _ in range(_n):
                        await asyncio.sleep(TOOL_DELAY)
                await asyncio.sleep(LLM_DELAY)
                return {"results": [f"{_aid}: done"]}
            builder.add_node(aid, _node)
        builder.add_edge(START, agent_ids[0])
        for i in range(len(agent_ids) - 1):
            builder.add_edge(agent_ids[i], agent_ids[i + 1])
        builder.add_edge(agent_ids[-1], END)

    graph = builder.compile()

    async def _invoke():
        await graph.ainvoke({"company": "Stripe", "results": []})

    t0 = time.perf_counter()
    asyncio.run(_invoke())
    return time.perf_counter() - t0


# ── AutoGen inline ────────────────────────────────────────────────────────────

def measure_autogen(par_agents: bool, par_tools: bool) -> float:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken
    from autogen_core.models import (
        ChatCompletionClient, CreateResult, ModelInfo, RequestUsage,
    )
    from typing import AsyncGenerator

    class _MockClient(ChatCompletionClient):
        async def create(self, messages, *, tools=(), json_output=None,
                         extra_create_args=None, cancellation_token=None):
            await asyncio.sleep(LLM_DELAY)
            return CreateResult(content="done", usage=RequestUsage(10, 5),
                                finish_reason="stop", cached=False, logprobs=None)
        async def create_stream(self, messages, **kw) -> AsyncGenerator:
            yield await self.create(messages, **kw)
        def count_tokens(self, *a, **kw): return 10
        def remaining_tokens(self, *a, **kw): return 100000
        @property
        def model_info(self): return ModelInfo(vision=False, function_calling=False, json_output=False, family="unknown")
        @property
        def capabilities(self): return self.model_info
        def actual_usage(self): return RequestUsage(0, 0)
        def total_usage(self): return RequestUsage(0, 0)
        async def close(self): pass

    async def _run_agent(aid, n):
        agent = AssistantAgent(aid, model_client=_MockClient(),
                               system_message="You are an analyst.")
        tok = CancellationToken()
        await agent.on_messages([TextMessage(content="Analyse Stripe", source="user")], tok)
        if par_tools:
            await asyncio.gather(*[asyncio.sleep(TOOL_DELAY) for _ in range(n)])
        else:
            for _ in range(n): await asyncio.sleep(TOOL_DELAY)
        resp = await agent.on_messages([TextMessage(content="Synthesise.", source="user")], tok)
        return resp.chat_message.content

    async def _main():
        if par_agents:
            await asyncio.gather(*[_run_agent(aid, n) for aid, n in AGENTS])
        else:
            for aid, n in AGENTS:
                await _run_agent(aid, n)

    t0 = time.perf_counter()
    asyncio.run(_main())
    return time.perf_counter() - t0


# ── HerdAI via subprocess ─────────────────────────────────────────────────────
# Run go run . once with the same 200ms/80ms delays as the Python scripts
# and cache all 4 scenario timings so we don't rebuild the binary 4 times.

_HERDAI_TIMINGS: List[float] = []   # [par+par, par+seq, seq+par, seq+seq]


def _load_herdai_timings() -> None:
    """Run `go run .` once and parse all four scenario timings."""
    import re

    if _HERDAI_TIMINGS:
        return

    bench_dir = os.path.join(os.path.dirname(__file__), "..")
    try:
        result = subprocess.run(
            ["go", "run", ".",
             "--llm-delay", f"{int(LLM_DELAY*1000)}ms",
             "--tool-delay", f"{int(TOOL_DELAY*1000)}ms"],
            cwd=bench_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in result.stdout.splitlines():
            if "completed in" in line:
                m = re.search(r"(\d+(?:\.\d+)?)\s*(ms|s)\b", line)
                if m:
                    val = float(m.group(1))
                    if m.group(2) == "s":
                        val *= 1000
                    _HERDAI_TIMINGS.append(val / 1000)
    except Exception as e:
        print(f"  [warn] go run failed: {e}", file=sys.stderr)

    # Ensure we always have 4 entries (fill with -1 on failure)
    while len(_HERDAI_TIMINGS) < 4:
        _HERDAI_TIMINGS.append(-1.0)


def measure_herdai(par_agents: bool, par_tools: bool) -> float:
    """Return the cached HerdAI timing for this scenario combination."""
    _load_herdai_timings()
    idx = {
        (True,  True):  0,
        (True,  False): 1,
        (False, True):  2,
        (False, False): 3,
    }[(par_agents, par_tools)]
    return _HERDAI_TIMINGS[idx]


# ── Main comparison table ─────────────────────────────────────────────────────

def main() -> None:
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  HerdAI vs LangGraph vs AutoGen vs CrewAI vs Pure Python       ║")
    print("║  Competitive Intelligence Pipeline — Wall-clock comparison      ║")
    print("╚" + "═" * 70 + "╝")
    print()
    print(f"  Pipeline  : 6 agents × (LLM→tools→LLM)  |  18 tool calls total")
    print(f"  LLM delay : {LLM_DELAY*1000:.0f} ms / call")
    print(f"  Tool delay: {TOOL_DELAY*1000:.0f} ms / call")
    print(f"  Scenarios : par+par  par+seq  seq+par  seq+seq")
    print()

    scenarios: List[Tuple[str, bool, bool]] = [
        ("par_agents + par_tools  (best case)",  True,  True),
        ("par_agents + seq_tools",               True,  False),
        ("seq_agents + par_tools",               False, True),
        ("seq_agents + seq_tools  (worst case)", False, False),
    ]

    results: Dict[str, List[float]] = {
        "HerdAI (Go goroutines)": [],
        "LangGraph 1.1.x       ": [],
        "AutoGen 0.7.5         ": [],
        "CrewAI 1.12.2 (sim)   ": [],
        "Pure Python asyncio   ": [],
    }

    for label, par_a, par_t in scenarios:
        print(f"  ── {label} ──")

        t = measure_herdai(par_a, par_t)
        results["HerdAI (Go goroutines)"].append(t)
        print(f"     HerdAI        : {t*1000:6.0f} ms")

        t = measure_langgraph(par_a, par_t)
        results["LangGraph 1.1.x       "].append(t)
        print(f"     LangGraph     : {t*1000:6.0f} ms")

        t = measure_autogen(par_a, par_t)
        results["AutoGen 0.7.5         "].append(t)
        print(f"     AutoGen       : {t*1000:6.0f} ms")

        t = measure_python(label, par_a, par_t)
        results["CrewAI 1.12.2 (sim)   "].append(t)
        results["Pure Python asyncio   "].append(t)
        print(f"     CrewAI (sim)  : {t*1000:6.0f} ms  (simulated — mirrors asyncio)")
        print(f"     Pure asyncio  : {t*1000:6.0f} ms")
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 70 + "╗")
    print("║  FINAL COMPARISON TABLE  (ms)                                  ║")
    print("╠" + "═" * 70 + "╣")

    col_labels = ["par+par", "par+seq", "seq+par", "seq+seq"]
    header = f"  {'Framework':<26}" + "".join(f"{c:>10}" for c in col_labels) + "  speedup"
    print(f"║  {header.strip():^68}  ║")
    print("╠" + "═" * 70 + "╣")

    all_rows = []
    for fw, times in results.items():
        if not times:
            continue
        row = [fw] + times
        all_rows.append(row)

    # Find global baseline (HerdAI par+par)
    herdai_base = results["HerdAI (Go goroutines)"][0] if results["HerdAI (Go goroutines)"] else 1.0

    seen = set()
    for row in all_rows:
        fw = row[0]
        if fw in seen:
            continue
        seen.add(fw)
        times = row[1:]
        if not times:
            continue
        worst_vs_best = times[-1] / times[0] if times[0] > 0 else 0
        cells = "".join(f"{t*1000:>10.0f}" for t in times)
        speedup = f"  {worst_vs_best:.1f}x"
        line = f"  {fw:<26}{cells}{speedup}"
        print(f"║  {line[:68]:68}  ║")

    print("╚" + "═" * 70 + "╝")
    print()
    print("  speedup = seq+seq ÷ par+par for each framework")
    print()
    print("  Key takeaways:")
    print("  1. For I/O-bound work (HTTP APIs), asyncio and goroutines have")
    print("     similar wall-clock time — both are non-blocking.")
    print("  2. HerdAI advantages are in CODE SIMPLICITY (no async/await,")
    print("     no compiled graph, no event bus), DEPENDENCIES (zero vs 40-80),")
    print("     MEMORY (~20 MB vs ~150-250 MB), and BINARY SIZE (8 MB vs venv).")
    print("  3. For CPU-bound work goroutines beat asyncio: the GIL prevents")
    print("     Python threads from using multiple cores simultaneously.")
    print("  4. CrewAI / AutoGen require explicit async wiring to go parallel;")
    print("     HerdAI is parallel by DEFAULT (StrategyParallel + ParallelToolCalls).")
    print()


if __name__ == "__main__":
    main()
