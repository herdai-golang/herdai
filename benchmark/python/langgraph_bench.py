"""
LangGraph — Competitive Intelligence Pipeline benchmark.

LangGraph lets you define graph nodes as plain async functions; no LLM is
required for the nodes themselves (the graph orchestrates them).  This makes
it easy to benchmark the framework's concurrency model directly.

Two graphs are built and timed:
  1. PARALLEL  — all 6 agent nodes fan out from START simultaneously via Send.
                 Wall time ≈ slowest single agent (max, not sum).
  2. SEQUENTIAL — nodes are chained; each agent waits for the previous one.
                 Wall time ≈ sum of all agents.

The per-agent work is identical to the Go and baseline Python benchmarks:
  • LLM latency ×2  (one call before tools, one call to synthesise results)
  • Tool latency ×N  (parallel or sequential within each agent)

Run:
    python3 langgraph_bench.py
"""

import asyncio
import operator
import time
from typing import Annotated, List, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# ── Simulated latencies (same as Go benchmark defaults) ───────────────────────
LLM_DELAY  = 0.200   # seconds
TOOL_DELAY = 0.080   # seconds

# ── Pipeline definition ────────────────────────────────────────────────────────
AGENTS: List[Tuple[str, int]] = [
    ("news_sentiment",     3),
    ("competitor_profile", 4),
    ("financial_metrics",  3),
    ("market_trends",      3),
    ("regulatory_scan",    2),
    ("customer_voice",     3),
]
TOTAL_TOOLS = sum(n for _, n in AGENTS)


# ── LangGraph state definitions ───────────────────────────────────────────────

class PipelineState(TypedDict):
    """Shared state for the top-level graph.  Results are merged via reducer."""
    company: str
    results: Annotated[List[str], operator.add]   # reducer: concat all lists


class AgentState(TypedDict):
    """Per-agent state passed in via Send."""
    company:    str
    agent_id:   str
    tool_count: int


# ── Shared async helpers ──────────────────────────────────────────────────────

async def _simulate_tools(tool_count: int, parallel: bool) -> None:
    if parallel:
        await asyncio.gather(*[asyncio.sleep(TOOL_DELAY) for _ in range(tool_count)])
    else:
        for _ in range(tool_count):
            await asyncio.sleep(TOOL_DELAY)


def _make_agent_node(agent_id: str, tool_count: int, parallel_tools: bool):
    """Return a LangGraph node function for one specialist agent."""
    async def node(state: AgentState) -> dict:
        await asyncio.sleep(LLM_DELAY)                          # LLM turn 1
        await _simulate_tools(tool_count, parallel_tools)       # tool calls
        await asyncio.sleep(LLM_DELAY)                          # LLM turn 2
        return {"results": [f"[{agent_id}] analysis complete"]}
    node.__name__ = agent_id
    return node


# ── Graph 1: PARALLEL fan-out via Send ────────────────────────────────────────
#
# START → dispatch_node → [Send("news_sentiment"), Send("competitor_profile"), ...]
#                                ↓                       ↓                    ...
#                    news_sentiment_node   competitor_profile_node  ...
#                                ↓                       ↓
#                               END ←──────────────────────────── (all converge)
#
# LangGraph collects all node outputs using the Annotated list reducer.

def build_parallel_graph(parallel_tools: bool) -> StateGraph:
    """
    Parallel fan-out using add_conditional_edges returning a list of Send.
    LangGraph dispatches all Send targets concurrently as asyncio tasks.
    """
    builder = StateGraph(PipelineState)

    for agent_id, tool_count in AGENTS:
        node = _make_agent_node(agent_id, tool_count, parallel_tools)
        builder.add_node(agent_id, node)
        builder.add_edge(agent_id, END)

    # Conditional edge from START returns Send() for every agent → parallel fan-out.
    def dispatch(state: PipelineState):
        return [
            Send(agent_id, {
                "company":    state["company"],
                "agent_id":   agent_id,
                "tool_count": tool_count,
            })
            for agent_id, tool_count in AGENTS
        ]

    builder.add_conditional_edges(START, dispatch)
    return builder.compile()


# ── Graph 2: SEQUENTIAL chain ─────────────────────────────────────────────────
#
# START → news_sentiment → competitor_profile → ... → customer_voice → END
#
# Each agent's output is merged into PipelineState before the next starts.

def build_sequential_graph(parallel_tools: bool) -> StateGraph:
    builder = StateGraph(PipelineState)

    agent_ids = [aid for aid, _ in AGENTS]

    for agent_id, tool_count in AGENTS:
        node = _make_agent_node(agent_id, tool_count, parallel_tools)
        builder.add_node(agent_id, node)

    builder.add_edge(START, agent_ids[0])
    for i in range(len(agent_ids) - 1):
        builder.add_edge(agent_ids[i], agent_ids[i + 1])
    builder.add_edge(agent_ids[-1], END)

    return builder.compile()


# ── Benchmark harness ─────────────────────────────────────────────────────────

async def run_graph(graph, company: str) -> float:
    t0 = time.perf_counter()
    await graph.ainvoke({"company": company, "results": []})
    return time.perf_counter() - t0


def main() -> None:
    print("=" * 68)
    print("  LangGraph — Competitive Intelligence Pipeline benchmark")
    print("=" * 68)
    print(f"  Agents     : {len(AGENTS)}")
    print(f"  Tools total: {TOTAL_TOOLS}")
    print(f"  LLM delay  : {LLM_DELAY*1000:.0f} ms / call")
    print(f"  Tool delay : {TOOL_DELAY*1000:.0f} ms / call")
    print()
    print("  LangGraph concurrency model:")
    print("  • Nodes reached by Send() fan-out run in parallel (asyncio tasks).")
    print("  • Parallel = LangGraph StrategyParallel equivalent.")
    print("  • State reducer (operator.add) merges all node outputs.")
    print()

    scenarios = [
        ("Parallel agents + Parallel tools  [LangGraph fan-out + async tools]",
         build_parallel_graph(True)),
        ("Parallel agents + Sequential tools",
         build_parallel_graph(False)),
        ("Sequential agents + Parallel tools",
         build_sequential_graph(True)),
        ("Sequential agents + Sequential tools  [GIL worst-case analog]",
         build_sequential_graph(False)),
    ]

    timings = []
    for label, graph in scenarios:
        print(f"  Running: {label}")
        elapsed = asyncio.run(run_graph(graph, "Stripe"))
        timings.append((label, elapsed))
        print(f"  ✓  completed in {elapsed*1000:.0f} ms")
        print()

    baseline = timings[0][1]
    print("-" * 68)
    print("  RESULTS")
    print("-" * 68)
    for label, t in timings:
        mult = t / baseline
        if mult < 1.05:
            print(f"  {t*1000:6.0f} ms  {label}  (baseline)")
        else:
            print(f"  {t*1000:6.0f} ms  {label}  ({mult:.1f}x slower)")
    print()
    worst = timings[-1][1]
    print(f"  LangGraph speedup (seq+seq ÷ par+par): {worst/baseline:.1f}x")
    print()
    print("  LangGraph uses asyncio under the hood.  For I/O-bound workloads")
    print("  its parallel fan-out is comparable to HerdAI goroutines in raw")
    print("  wall-clock time.  The differences are:")
    print("    • Code complexity: LangGraph requires typed state, explicit")
    print("      edges/nodes, and a compiled graph object.  HerdAI needs only")
    print("      NewAgent + NewManager.")
    print("    • Dependencies: LangGraph pulls 40+ packages; HerdAI has zero.")
    print("    • CPU-bound work: asyncio is single-threaded; goroutines are not.")
    print("    • Memory: Go processes use ~10–30 MB; Python + LangGraph ~150 MB.")
    print("=" * 68)


if __name__ == "__main__":
    main()
