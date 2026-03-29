"""
Pure Python concurrency baseline — no framework.

Mirrors the same 6-agent, 18-tool Competitive Intelligence pipeline in:
  1. asyncio parallel agents + parallel tools (Python's theoretical best)
  2. asyncio parallel agents + sequential tools
  3. asyncio sequential agents + parallel tools
  4. asyncio sequential agents + sequential tools  (worst case / GIL analogy)
  5. threading parallel agents + parallel tools   (OS threads, GIL-limited for CPU)

This is the honest floor: the numbers here show what Python can do for
I/O-bound work (time.sleep) without any framework overhead.  Real LLM API
calls are network-I/O-bound, so asyncio is legitimately competitive.
The GIL difference shows up when agents need CPU work (tokenisation,
embedding, inference, JSON parsing at scale).

Run:
    python3 baseline.py
"""

import asyncio
import concurrent.futures
import time
from typing import List, Tuple

# ── Simulated latencies (match Go benchmark defaults) ─────────────────────────
LLM_DELAY  = 0.200   # seconds — one LLM API round-trip
TOOL_DELAY = 0.080   # seconds — one external API / DB / scrape call

# ── Pipeline definition (same as Go app.go) ───────────────────────────────────
AGENTS: List[Tuple[str, int]] = [
    ("news_sentiment",     3),   # 3 tool calls (Reuters, Bloomberg, Twitter)
    ("competitor_profile", 4),   # 4 tool calls (Crunchbase, LinkedIn, Website, Glassdoor)
    ("financial_metrics",  3),   # 3 tool calls (SEC, Revenue API, Market Cap)
    ("market_trends",      3),   # 3 tool calls (Google Trends, Gartner, Patents)
    ("regulatory_scan",    2),   # 2 tool calls (SEC EDGAR, Compliance DB)
    ("customer_voice",     3),   # 3 tool calls (G2, Gartner Peer, App Store)
]
TOTAL_TOOLS = sum(n for _, n in AGENTS)


# ── Async helpers ─────────────────────────────────────────────────────────────

async def simulate_tool(name: str) -> str:
    """Simulate one external API / DB / scrape call."""
    await asyncio.sleep(TOOL_DELAY)
    return f"{name}: data retrieved ✓"


async def simulate_agent(agent_id: str, tool_count: int, parallel_tools: bool) -> str:
    """
    Simulate one specialist agent:
      LLM call 1 → tool calls (parallel or sequential) → LLM call 2
    """
    await asyncio.sleep(LLM_DELAY)          # LLM turn 1: decides which tools to call

    tool_names = [f"{agent_id}_tool_{i}" for i in range(tool_count)]
    if parallel_tools:
        # All tool calls triggered in one LLM response → run concurrently.
        await asyncio.gather(*[simulate_tool(n) for n in tool_names])
    else:
        for n in tool_names:
            await simulate_tool(n)

    await asyncio.sleep(LLM_DELAY)          # LLM turn 2: synthesises tool results
    return f"[{agent_id}] analysis complete"


# ── Four concurrency scenarios ─────────────────────────────────────────────────

async def run_parallel_parallel() -> List[str]:
    """Manager runs all agents concurrently; each agent runs its tools concurrently."""
    return list(await asyncio.gather(
        *[simulate_agent(aid, n, True) for aid, n in AGENTS]
    ))


async def run_parallel_sequential() -> List[str]:
    """Manager runs all agents concurrently; tools run sequentially within each agent."""
    return list(await asyncio.gather(
        *[simulate_agent(aid, n, False) for aid, n in AGENTS]
    ))


async def run_sequential_parallel() -> List[str]:
    """Manager runs agents one at a time; each agent runs its tools concurrently."""
    results = []
    for aid, n in AGENTS:
        results.append(await simulate_agent(aid, n, True))
    return results


async def run_sequential_sequential() -> List[str]:
    """
    Everything serialised — equivalent to what a Python framework does when
    it uses a single-threaded executor with no async tool calls.
    This is the GIL-worst-case analog.
    """
    results = []
    for aid, n in AGENTS:
        results.append(await simulate_agent(aid, n, False))
    return results


# ── Threading variant ─────────────────────────────────────────────────────────

def _sync_agent(agent_id: str, tool_count: int) -> str:
    """Synchronous agent — used for the thread-pool experiment."""
    time.sleep(LLM_DELAY)
    # Tools run sequentially inside each thread.
    for i in range(tool_count):
        time.sleep(TOOL_DELAY)
    time.sleep(LLM_DELAY)
    return f"[{agent_id}] analysis complete (thread)"


def run_threading_parallel() -> List[str]:
    """
    Thread-per-agent with sequential tools.
    Threads share one GIL, so this helps for I/O-bound work but would
    NOT help for CPU-bound work (tokenisation, embedding inference, etc.).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENTS)) as pool:
        futures = [pool.submit(_sync_agent, aid, n) for aid, n in AGENTS]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


# ── Benchmark harness ─────────────────────────────────────────────────────────

def bench(label: str, coro_or_fn, is_async: bool = True) -> float:
    t0 = time.perf_counter()
    if is_async:
        asyncio.run(coro_or_fn())
    else:
        coro_or_fn()
    elapsed = time.perf_counter() - t0
    return elapsed


def main() -> None:
    print("=" * 68)
    print("  Pure Python — Competitive Intelligence Pipeline (no framework)")
    print("=" * 68)
    print(f"  Agents     : {len(AGENTS)}")
    print(f"  Tools total: {TOTAL_TOOLS}")
    print(f"  LLM delay  : {LLM_DELAY*1000:.0f} ms / call")
    print(f"  Tool delay : {TOOL_DELAY*1000:.0f} ms / call")
    print()

    scenarios = [
        ("asyncio  par agents + par tools  [Python best for I/O]",
         run_parallel_parallel, True),
        ("asyncio  par agents + seq tools",
         run_parallel_sequential, True),
        ("asyncio  seq agents + par tools",
         run_sequential_parallel, True),
        ("asyncio  seq agents + seq tools  [GIL worst-case analog]",
         run_sequential_sequential, True),
        ("threading par agents + seq tools (I/O-bound only)",
         run_threading_parallel, False),
    ]

    timings = []
    for label, fn, is_async in scenarios:
        elapsed = bench(label, fn, is_async)
        timings.append((label, elapsed))
        print(f"  ✓ {label}")
        print(f"    completed in {elapsed*1000:.0f} ms")
        print()

    baseline = timings[0][1]
    print("-" * 68)
    print("  SUMMARY (ms)")
    print("-" * 68)
    for label, t in timings:
        mult = t / baseline
        bar = "─" * int(mult * 10)
        if mult < 1.05:
            print(f"  {t*1000:6.0f} ms  {bar}  {label} (baseline)")
        else:
            print(f"  {t*1000:6.0f} ms  {bar}  {label} ({mult:.1f}x slower)")
    print()

    worst = timings[3][1]  # seq+seq
    print(f"  Python asyncio par+par vs seq+seq: {worst/baseline:.1f}x speedup")
    print()
    print("  Key insight: asyncio is single-threaded. For I/O-bound work")
    print("  (time.sleep, HTTP calls) it IS concurrent — comparable to Go.")
    print("  But for CPU-bound work (tokenisation, embedding, JSON parsing")
    print("  at scale) the GIL serialises all threads; only goroutines")
    print("  achieve true CPU parallelism inside one process.")
    print("=" * 68)


if __name__ == "__main__":
    main()
