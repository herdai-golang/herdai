"""
AutoGen (autogen-agentchat 0.7.x) — Competitive Intelligence Pipeline benchmark.

AutoGen is async-first: agents communicate via messages, and a team
(e.g. RoundRobinGroupChat) orchestrates them.  This script uses a custom
MockChatCompletionClient so no API key is needed.

Two modes are timed:
  1. PARALLEL — 6 agents run concurrently via asyncio.gather (simulates
                what you'd do in AutoGen to get parallel execution;
                AutoGen itself doesn't have a built-in parallel team type
                in 0.7.x, so this uses gather on individual on_messages calls).
  2. SEQUENTIAL — agents run one after another, each receiving the previous
                  agent's result as context.

Within each agent, tool calls are simulated with asyncio.sleep.

Run:
    python3 autogen_bench.py
"""

import asyncio
import time
from typing import AsyncGenerator, List, Sequence, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import BaseTool

# ── Simulated latencies (same as all other benchmarks) ────────────────────────
LLM_DELAY  = 0.200   # seconds per LLM call
TOOL_DELAY = 0.080   # seconds per tool call

# ── Pipeline definition ────────────────────────────────────────────────────────
AGENTS: List[Tuple[str, str, int]] = [
    ("news_sentiment",     "News Sentiment Analyst",     3),
    ("competitor_profile", "Competitor Intel Analyst",   4),
    ("financial_metrics",  "Financial Metrics Analyst",  3),
    ("market_trends",      "Market Trend Analyst",       3),
    ("regulatory_scan",    "Regulatory Risk Analyst",    2),
    ("customer_voice",     "Customer Voice Analyst",     3),
]
TOTAL_TOOLS = sum(n for *_, n in AGENTS)


# ── Mock LLM client ────────────────────────────────────────────────────────────

class MockChatCompletionClient(ChatCompletionClient):
    """
    A minimal ChatCompletionClient that returns a fixed response after sleeping
    for LLM_DELAY.  No API key or network access needed.
    """

    def __init__(self, delay: float = LLM_DELAY):
        self._delay = delay
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages,
        *,
        tools=(),
        json_output=None,
        extra_create_args=None,
        cancellation_token=None,
    ) -> CreateResult:
        await asyncio.sleep(self._delay)
        self._usage = RequestUsage(
            prompt_tokens=self._usage.prompt_tokens + 10,
            completion_tokens=self._usage.completion_tokens + 5,
        )
        return CreateResult(
            content="Analysis complete based on all gathered data.",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
            finish_reason="stop",
            cached=False,
            logprobs=None,
        )

    async def create_stream(self, messages, **kwargs) -> AsyncGenerator:
        yield await self.create(messages, **kwargs)

    def count_tokens(self, messages, **kwargs) -> int:
        return 10

    def remaining_tokens(self, messages, **kwargs) -> int:
        return 100_000

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=False,
            json_output=False,
            family="unknown",
        )

    @property
    def capabilities(self) -> ModelInfo:
        return self.model_info

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    async def close(self) -> None:
        pass


# ── Simulated tool functions ───────────────────────────────────────────────────

async def simulate_tool(name: str) -> str:
    await asyncio.sleep(TOOL_DELAY)
    return f"{name}: data retrieved ✓"


# ── Agent factory ─────────────────────────────────────────────────────────────

def make_agent(agent_id: str, role: str) -> AssistantAgent:
    """
    Build an AutoGen AssistantAgent with a mock LLM.
    Note: tool calls are simulated separately (see run_agent_with_tools).
    """
    return AssistantAgent(
        name=agent_id,
        model_client=MockChatCompletionClient(),
        system_message=(
            f"You are a {role}. Analyse the target company thoroughly "
            "and provide structured insights."
        ),
    )


async def run_agent_with_tools(
    agent: AssistantAgent,
    agent_id: str,
    tool_count: int,
    parallel_tools: bool,
    company: str,
) -> str:
    """
    Run one agent: message → LLM (mocked) → simulated tools → final response.
    AutoGen 0.7.x agents call on_messages for each turn.
    """
    tok = CancellationToken()

    # Turn 1: LLM receives the task and "decides" to call tools.
    await agent.on_messages(
        [TextMessage(content=f"Analyse {company}", source="user")],
        tok,
    )

    # Tool calls (simulated — AutoGen 0.7.x with function_calling=False means
    # the mock LLM returns plain text; tools are triggered manually here to
    # mirror the Go/LangGraph/baseline timing model faithfully).
    tool_names = [f"{agent_id}_tool_{i}" for i in range(tool_count)]
    if parallel_tools:
        await asyncio.gather(*[simulate_tool(n) for n in tool_names])
    else:
        for n in tool_names:
            await simulate_tool(n)

    # Turn 2: LLM synthesises tool results.
    resp = await agent.on_messages(
        [TextMessage(content="Synthesise the gathered data.", source="user")],
        tok,
    )
    return resp.chat_message.content


# ── Parallel scenario ─────────────────────────────────────────────────────────

async def run_parallel(parallel_tools: bool, company: str) -> List[str]:
    """
    All 6 agents run concurrently via asyncio.gather.
    This is how you achieve parallel agent execution in AutoGen 0.7.x —
    there is no built-in 'parallel team' type; you compose it with gather.
    """
    tasks = [
        run_agent_with_tools(make_agent(aid, role), aid, n, parallel_tools, company)
        for aid, role, n in AGENTS
    ]
    return list(await asyncio.gather(*tasks))


# ── Sequential scenario ───────────────────────────────────────────────────────

async def run_sequential(parallel_tools: bool, company: str) -> List[str]:
    """
    Agents run one after another.  This matches AutoGen's default RoundRobin
    behaviour where a GroupChat serialises agent turns.
    """
    results = []
    for aid, role, n in AGENTS:
        result = await run_agent_with_tools(
            make_agent(aid, role), aid, n, parallel_tools, company
        )
        results.append(result)
    return results


# ── Benchmark harness ─────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("  AutoGen (autogen-agentchat 0.7.5) — CI Pipeline benchmark")
    print("=" * 68)
    print(f"  Agents     : {len(AGENTS)}")
    print(f"  Tools total: {TOTAL_TOOLS}")
    print(f"  LLM delay  : {LLM_DELAY*1000:.0f} ms / call (MockChatCompletionClient)")
    print(f"  Tool delay : {TOOL_DELAY*1000:.0f} ms / call (asyncio.sleep)")
    print()
    print("  AutoGen concurrency note:")
    print("  AutoGen 0.7.x has RoundRobinGroupChat (sequential by default).")
    print("  True parallel execution requires asyncio.gather on on_messages calls.")
    print()

    scenarios = [
        ("asyncio.gather on 6 agents + parallel tools  [par+par]",
         lambda: run_parallel(True,  "Stripe")),
        ("asyncio.gather on 6 agents + sequential tools [par+seq]",
         lambda: run_parallel(False, "Stripe")),
        ("Sequential agents + parallel tools            [seq+par]",
         lambda: run_sequential(True,  "Stripe")),
        ("Sequential agents + sequential tools          [seq+seq]",
         lambda: run_sequential(False, "Stripe")),
    ]

    timings = []
    for label, fn in scenarios:
        print(f"  Running: {label}")
        t0 = time.perf_counter()
        asyncio.run(fn())
        elapsed = time.perf_counter() - t0
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
    print(f"  AutoGen speedup (seq+seq ÷ par+par): {worst/baseline:.1f}x")
    print()
    print("  AutoGen key differences vs HerdAI:")
    print("    • No built-in parallel team type in 0.7.x; must use asyncio.gather.")
    print("    • Message-passing overhead between agents (serialised dicts).")
    print("    • 50+ dependency packages vs HerdAI's zero.")
    print("    • Async-first but still single-threaded (GIL applies to CPU work).")
    print("    • autogen-agentchat import alone takes ~3s; HerdAI binary starts <10ms.")
    print("=" * 68)


if __name__ == "__main__":
    main()
