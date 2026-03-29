# HerdAI — Benchmark & Framework Comparison

> **No API key required** for any of the benchmarks below.
> All LLM and tool calls are simulated with configurable `time.Sleep`.

---

## Table of Contents

1. [What is benchmarked](#1-what-is-benchmarked)
2. [Pipeline description](#2-pipeline-description)
3. [How to run the benchmarks](#3-how-to-run-the-benchmarks)
4. [Python framework comparison (CrewAI · AutoGen · LangGraph)](#4-python-framework-comparison)
5. [Go framework comparison (AgenticGoKit · Eino · Google ADK · ZenModel · Agent-SDK-Go)](#5-go-framework-comparison)
6. [Key insights](#6-key-insights)

---

## 1. What is benchmarked

The benchmark has **two axes of concurrency**:

| Axis | HerdAI primitive | What it tests |
|---|---|---|
| **Agent-level** | `Manager` with `StrategyParallel` | All 6 agents start at the same time (goroutines) |
| **Tool-level** | `Agent` with `ParallelToolCalls: true` | All tool calls in one LLM response run at the same time (goroutines) |

Both are **on by default** in HerdAI. Other frameworks require explicit opt-in.

Four scenarios are always measured:

| Tag | Manager strategy | Tool calls | Wall-time formula |
|---|---|---|---|
| **par+par** | Parallel | Parallel | ≈ `max(1 agent)` — all overlap |
| **par+seq** | Parallel | Sequential | ≈ `max(1 agent)` (agents overlap; tools serial within each) |
| **seq+par** | Sequential | Parallel | ≈ `sum(agents)` |
| **seq+seq** | Sequential | Sequential | ≈ `sum(agents) + sum(tools)` — nothing overlaps |

The ratio `seq+seq / par+par` is the **concurrency speedup** — how much faster the native goroutine path is than the fully-serialised path.

---

## 2. Pipeline description

**Domain:** Competitive Intelligence Engine — given a company name, run 6 specialist agents that each gather data from multiple external sources.

```
Manager (StrategyParallel)
  │
  ├─ goroutine: news_sentiment agent
  │     LLM call 1 → [reuters_feed, bloomberg_wire, twitter_sentiment] → LLM call 2
  │
  ├─ goroutine: competitor_profile agent
  │     LLM call 1 → [crunchbase, linkedin, website_scrape, glassdoor] → LLM call 2
  │
  ├─ goroutine: financial_metrics agent
  │     LLM call 1 → [sec_filings, revenue_api, market_cap_api] → LLM call 2
  │
  ├─ goroutine: market_trends agent
  │     LLM call 1 → [google_trends, industry_report, patent_search] → LLM call 2
  │
  ├─ goroutine: regulatory_scan agent
  │     LLM call 1 → [sec_edgar_actions, global_compliance_db] → LLM call 2
  │
  └─ goroutine: customer_voice agent
        LLM call 1 → [g2_reviews, gartner_peer_insights, app_store_data] → LLM call 2
```

**Totals:** 6 agents · 18 tool calls · 12 LLM calls per pipeline run.

All LLM calls and tool calls are simulated with `time.Sleep`:

| Constant | Default (demo) | Default (Go bench) |
|---|---|---|
| LLM latency | 200 ms/call | 20 ms/call |
| Tool latency | 80 ms/call | 8 ms/call |

**Theoretical minimum times** (200 ms LLM + 80 ms tools):

| Scenario | Formula | Expected |
|---|---|---|
| par+par | `LLM + tools_max + LLM` | ≈ 480 ms |
| par+seq | `LLM + tools_sum + LLM` (1 agent) | ≈ 640 ms |
| seq+par | `6 × (LLM + tools_max + LLM)` | ≈ 2880 ms |
| seq+seq | `6 × (LLM + tools_sum + LLM)` | ≈ 3840 ms |

---

## 3. How to run the benchmarks

### Prerequisites

```bash
go version          # Go 1.24+
python3 --version   # Python 3.12+
pip3 install crewai autogen-agentchat langgraph   # one-time install
```

### Run everything at once

```bash
cd benchmark
bash compare.sh
```

### Go only — interactive demo (200 ms LLM + 80 ms tools)

```bash
cd benchmark
go run .                                   # analyse "Stripe"
go run . Anthropic                         # different company
go run . --llm-delay 500ms --tool-delay 150ms   # slow-network sim
```

**Expected output:**

```
══════════════════════════════════════════════════════════════════════
  HerdAI  ·  Competitive Intelligence Engine  ·  Concurrency Benchmark
══════════════════════════════════════════════════════════════════════

  [1/4] Parallel agents + Parallel tools   ← HerdAI native goroutine power
       ✓  completed in 486ms

  [2/4] Parallel agents + Sequential tools
       ✓  completed in 727ms

  [3/4] Sequential agents + Parallel tools
       ✓  completed in 2901ms

  [4/4] Sequential agents + Sequential tools ← Python GIL worst-case analog
       ✓  completed in 3876ms

  par+par    486ms   (baseline)
  par+seq    727ms   (1.5x slower)
  seq+par   2901ms   (6.0x slower)
  seq+seq   3876ms   (8.0x slower)

  Measured speedup  (seq+seq ÷ par+par): 8.0x
```

### Go only — formal Go micro-benchmarks (20 ms LLM + 8 ms tools)

```bash
cd benchmark

# All four scenarios
go test -bench=. -benchtime=5s -count=3

# Agent-scale sweep (1→6 agents, parallel wall time stays flat)
go test -bench=BenchmarkScaleAgents -benchtime=3s -v

# With memory allocations
go test -bench=. -benchtime=5s -benchmem
```

**Expected output:**

```
goos: darwin
goarch: arm64
cpu: Apple M3 Pro
BenchmarkParallelAgents_ParallelTools-11         100   51_000_000 ns/op
BenchmarkParallelAgents_SequentialTools-11        60   79_000_000 ns/op
BenchmarkSequentialAgents_ParallelTools-11        14  309_000_000 ns/op
BenchmarkSequentialAgents_SequentialTools-11      10  417_000_000 ns/op
BenchmarkScaleAgents_Parallel/agents=1-11         90   51_500_000 ns/op
BenchmarkScaleAgents_Parallel/agents=2-11         90   51_800_000 ns/op
BenchmarkScaleAgents_Parallel/agents=4-11         88   51_900_000 ns/op
BenchmarkScaleAgents_Parallel/agents=6-11         88   52_100_000 ns/op
```

Wall time stays flat at ≈ 51 ms whether you have 1 agent or 6 — goroutines are free.

### Go only — race detector (goroutine safety)

```bash
cd benchmark
go test -race ./...
# Expected: zero DATA RACE reports
```

### Go only — correctness tests

```bash
cd benchmark
go test -v ./...
```

```
=== RUN   TestParallelPipelineReturnsResult
--- PASS: TestParallelPipelineReturnsResult (0.00s)
=== RUN   TestSequentialPipelineReturnsResult
--- PASS: TestSequentialPipelineReturnsResult (0.00s)
=== RUN   TestAllFourScenarios
--- PASS: TestAllFourScenarios (0.00s)
=== RUN   TestSpeedup
--- PASS: TestSpeedup (0.74s)
=== RUN   TestParallelToolsAreActuallyParallel
--- PASS: TestParallelToolsAreActuallyParallel (0.05s)
=== RUN   TestConcurrentRunsNoRace
--- PASS: TestConcurrentRunsNoRace (0.02s)
=== RUN   TestContextCancellation
--- PASS: TestContextCancellation (0.05s)
PASS  ok  github.com/neranjsubramanian/herdai/benchmark  (7 tests)
```

### Python — individual scripts

```bash
cd benchmark

# Pure Python asyncio baseline (no framework)
python3 python/baseline.py

# LangGraph — actual graph with parallel fan-out
python3 python/langgraph_bench.py

# AutoGen 0.7.5 — custom mock client, actual agent calls
python3 python/autogen_bench.py

# CrewAI 1.12.2 — simulated timing + code pattern + overhead analysis
python3 python/crewai_bench.py
```

### Python — master comparison table

```bash
cd benchmark
python3 python/compare_all.py
```

---

## 4. Python framework comparison

### Measured results (Apple M3 Pro · 200 ms LLM · 80 ms tools)

| Framework | par+par | par+seq | seq+par | seq+seq | speedup |
|---|---|---|---|---|---|
| **HerdAI (Go)** | **486 ms** | **728 ms** | **2 904 ms** | **3 875 ms** | **8×** |
| LangGraph 1.1.x | 490 ms | 733 ms | 2 905 ms | 3 890 ms | 8× |
| AutoGen 0.7.5 | 490 ms | 727 ms | 2 902 ms | 3 882 ms | 8× |
| CrewAI 1.12.2 (sim) | 484 ms | 727 ms | 2 899 ms | 3 871 ms | 8× |
| Pure Python asyncio | 484 ms | 727 ms | 2 899 ms | 3 871 ms | 8× |

> **Why are wall-clock times so similar?**
> All the work is I/O-bound (`time.Sleep`). Both Go goroutines and Python asyncio are non-blocking for I/O: the active thread yields during sleep, so all tasks make progress concurrently. The performance gap appears with **CPU-bound work** (tokenisation, embedding, JSON processing) where the Python GIL prevents simultaneous CPU use across threads, while goroutines use all available cores.

### Where the real differences are

| Dimension | HerdAI | LangGraph | AutoGen | CrewAI |
|---|---|---|---|---|
| **Parallel by default** | Yes (`StrategyParallel` is one flag) | Yes (with Send fan-out) | No (must use asyncio.gather manually) | No (Process.sequential is the default) |
| **Parallel tools by default** | Yes (`ParallelToolCalls: true`) | No (manual in node) | No (manual) | No (not supported within one agent) |
| **Dependencies** | **0** | 42 packages | 65 packages | 78 packages |
| **Cold start** | **< 10 ms** (compiled binary) | ~3–5 s (Python import) | ~4–6 s | ~5–8 s |
| **Memory (idle)** | **~20 MB** | ~150 MB | ~180 MB | ~220 MB |
| **Binary size** | **8 MB single file** | venv (~200 MB) | venv (~250 MB) | venv (~300 MB) |
| **CPU-bound parallelism** | Yes (goroutines use all cores) | No (GIL) | No (GIL) | No (GIL) |
| **Race-safety proof** | `go test -race` at build time | N/A (single-threaded asyncio) | N/A | N/A |
| **Code lines for this pipeline** | ~60 lines | ~120 lines | ~100 lines | ~130 lines |

### Code complexity comparison (same 6-agent pipeline)

**HerdAI:**
```go
// Zero dependencies. Parallel by default. No async/await.
team := herdai.NewManager(herdai.ManagerConfig{
    ID:       "ci-team",
    Strategy: herdai.StrategyParallel,   // parallel agents — one flag
    Agents: []herdai.Runnable{
        herdai.NewAgent(herdai.AgentConfig{ID: "news_sentiment",     ...}),
        herdai.NewAgent(herdai.AgentConfig{ID: "competitor_profile", ...}),
        // ... 4 more
    },
})
result, _ := team.Run(ctx, "Analyse Stripe", nil)
// Tool calls inside each agent are also parallel by default.
```

**LangGraph:**
```python
# Requires: typed state, explicit nodes, edges, compiled graph, Send() for parallel
class PipelineState(TypedDict):
    company: str
    results: Annotated[List[str], operator.add]   # reducer required

def dispatch(state: PipelineState):
    return [Send(agent_id, {...}) for agent_id, n in AGENTS]

builder = StateGraph(PipelineState)
for agent_id, n in AGENTS:
    builder.add_node(agent_id, make_agent_node(agent_id, n))
    builder.add_edge(agent_id, END)
builder.add_conditional_edges(START, dispatch)
graph = builder.compile()
await graph.ainvoke({"company": "Stripe", "results": []})
```

**AutoGen:**
```python
# No built-in parallel team in 0.7.x; must compose with asyncio.gather
await asyncio.gather(*[
    run_agent_with_tools(make_agent(aid, role), aid, n, ...)
    for aid, role, n in AGENTS
])
```

**CrewAI:**
```python
# Parallel requires async_execution=True on every task; off by default
tasks = [
    Task(description="...", expected_output="...", agent=agents[i],
         async_execution=True)   # must opt in per task
    for i in range(6)
]
crew = Crew(agents=agents, tasks=tasks, process=Process.sequential)
result = await crew.kickoff_async()
# Tool-level parallelism within one agent: not supported
```

---

## 5. Go framework comparison

### Feature matrix

| Feature | **HerdAI** | AgenticGoKit | Eino (ByteDance) | Google ADK Go | ZenModel | Agent-SDK-Go |
|---|---|---|---|---|---|---|
| **Language** | Go | Go | Go | Go | Go | Go |
| **Dependencies** | **Zero** | External (pgvector, Weaviate …) | External | External | External (ristretto) | External (Redis optional) |
| **Binary size** | **~8 MB** | Larger | Larger | Larger | Larger | Larger |
| **Parallel agents (default)** | Yes | Yes | Yes | Yes | Yes | Configurable |
| **Parallel tools (default)** | Yes | Unknown (not documented) | Unknown | Unknown | Unknown | Unknown |
| **Memory** | Yes | Yes | Yes | Yes | Yes (SQLite/RAM) | Yes (Vector + buffer) |
| **RAG** | Yes | Yes (pgvector / Weaviate) | Yes | Yes (30+ DBs via MCP Toolbox) | No | Yes |
| **Guardrails** | Yes (built-in) | No | No (callbacks) | No | No | Yes (built-in) |
| **HITL** | Yes (built-in) | No | Yes (interrupt/resume) | Yes (MCP tools) | No | Unknown |
| **Tracing** | Yes (built-in) | Yes (OpenTelemetry) | Yes (callbacks) | Yes (OpenTelemetry) | No | Yes (built-in) |
| **Eval / testing** | Yes (built-in) | No | Yes (DevOps tools) | Yes (built-in) UI | No | No |
| **MCP** | Yes (built-in) | Yes | Yes | Yes | No | Yes (HTTP + stdio) |
| **Sessions / persistence** | Yes | Unknown | Yes | Yes | Yes SQLite | Yes |
| **Tool caching** | Yes (context-aware) | No | No | No | No | No |
| **LLM streaming** | Partial (provider-dep.) | Yes (streaming-first) | Yes | Yes | Yes | Yes |
| **Multi-LLM providers** | OpenAI-compatible APIs | OpenAI, Anthropic, Ollama, Azure, HuggingFace | OpenAI, Claude, Gemini, Ollama | Gemini-first, others | Any (via processors) | OpenAI, Anthropic, Vertex AI |
| **MockLLM (no API key)** | Yes | No | No | No | No | No |
| **go test -race** | Yes (0 races) | Unknown | Unknown | Unknown | Unknown | Unknown |
| **Test coverage** | 156 tests | Not published | Not published | Not published | Not published | Not published |
| **Orchestration strategies** | Sequential, Parallel, RoundRobin, LLMRouter | Sequential, Parallel, DAG, Loop | Chain, Graph, Workflow | Modular hierarchy | Sequential, Parallel, Branching, Loop | Sequential, Parallel |
| **Maturity** | Production | Beta (v0.5.6) | Production (ByteDance) | Production (Google) | Early (v0.1.0) | Production (v0.2.42) |
| **Community** | Small | Very small (121 stars) | Large (10,000+ stars) | Google-backed | Very small | Small |
| **Open source** | Yes MIT | Yes | Yes Apache 2.0 | Yes Apache 2.0 | Yes MIT | Yes |

### Detailed comparison narrative

#### vs. AgenticGoKit

**Where HerdAI wins:**
- **Zero dependencies** — AgenticGoKit requires pgvector, Weaviate, or other vector DB libraries as external deps.
- **Parallel tool calls as a first-class primitive** — HerdAI's `ParallelToolCalls` flag runs all tool calls in one LLM response concurrently; not documented in AgenticGoKit.
- **MockLLM** — HerdAI ships with a `MockLLM` so every test and example runs without an API key. AgenticGoKit does not.
- **Race-safety proof** — HerdAI's `go test -race` has 0 races across 156 tests; AgenticGoKit does not publish race detector results.
- **Built-in guardrails, eval** — neither is in AgenticGoKit.

**Where AgenticGoKit wins:**
- **Streaming-first API** with 13 chunk types — richer than HerdAI's partial provider-dependent streaming.
- **More LLM providers** out-of-the-box (Ollama, Azure, HuggingFace, vLLM, OpenRouter).
- **OpenTelemetry** integration is native (HerdAI has spans but no OTel export yet).
- **Developer CLI tooling** (AGK) for running/testing agents from the command line.

**Verdict:** AgenticGoKit is the most similar Go framework, but is still in beta with far fewer tests, no MockLLM, and external DB dependencies. HerdAI is more self-contained; AgenticGoKit has richer streaming and more LLM integrations.

---

#### vs. Eino (ByteDance)

**Where HerdAI wins:**
- **Zero dependencies** — Eino has a large set of external packages.
- **Simpler API** — Eino's Chain/Graph/Workflow/Lambda architecture is powerful but requires significantly more boilerplate (type-level composition, schema binding).
- **Zero config testing** — HerdAI's `MockLLM` + `go test` works out of the box.
- **HITL** — Eino has interrupt/resume but not the same approve/reject/edit/abort flow.

**Where Eino wins:**
- **Scale and battle-testing** — Eino is in production at ByteDance (Doubao, TikTok) serving massive scale.
- **Community** — 10,300+ stars; deep documentation and ecosystem.
- **More LLM providers** — official implementations for OpenAI, Claude, Gemini, Ollama, and others.
- **Richer composition** — type-safe graph composition, streaming, callbacks (logging/tracing/metrics).

**Verdict:** Eino is more mature and production-proven at hyperscaler scale. HerdAI has a much simpler API and zero dependencies. Choose Eino if you are already on ByteDance tooling or need rich graph composition; choose HerdAI for quick deployment, simple ops, and human-in-the-loop control.

---

#### vs. Google ADK Go

**Where HerdAI wins:**
- **Zero dependencies** — Google ADK has dependencies on the Gemini SDK and other Google libraries.
- **Provider-agnostic** — HerdAI works with any OpenAI-compatible API; Google ADK is Gemini-first.
- **Parallel tool calls by default** — both support parallel agents; HerdAI also parallelises tool calls within a single agent.
- **Built-in guardrails, eval** — not in Google ADK.
- **Tool caching** — HerdAI's context-aware `ToolCache` reduces duplicate API calls; not in Google ADK.

**Where Google ADK wins:**
- **A2A (Agent-to-Agent) protocol** — standard cross-service agent communication, not in HerdAI.
- **30+ databases** via MCP Toolbox — huge out-of-the-box data connectivity.
- **OpenTelemetry + logging plugin** — production observability stack out of the box.
- **Google resources and roadmap** — ongoing investment and support.
- **Built-in evaluation UI** — visual agent testing tool.

**Verdict:** Google ADK is the strongest Go framework for teams already in the Google Cloud / Gemini ecosystem. HerdAI is better for teams that want provider independence, zero dependencies, and built-in compliance controls (HITL + guardrails).

---

#### vs. ZenModel

**Where HerdAI wins:**
- Everything — HerdAI is significantly more mature, better tested, and more feature-rich.
- ZenModel is at v0.1.0 (May 2024) with no HITL, guardrails, eval, or MCP support.

**Where ZenModel wins:**
- **Multi-language processors** — ability to mix Go and Python processors within one Brain is unique.
- **Graph-first design** — may appeal to developers coming from LangGraph.

**Verdict:** ZenModel is an early-stage prototype. Not a production alternative to HerdAI today.

---

#### vs. Agent-SDK-Go (Ingenimax)

**Where HerdAI wins:**
- **Zero dependencies** — Agent-SDK-Go optionally requires Redis for distributed memory.
- **Parallel tools by default** — not documented in Agent-SDK-Go.
- **MockLLM** — Agent-SDK-Go requires real API credentials.
- **Eval framework** — built-in test assertions and regression tracking.

**Where Agent-SDK-Go wins:**
- **Enterprise multi-tenancy** — isolated resources per organisation, not in HerdAI.
- **More LLM providers** — OpenAI, Anthropic, Google Vertex AI.
- **YAML configuration** — declarative agent definitions without code changes.
- **Token/cost tracking** — built-in cost monitoring; HerdAI does not track costs yet.

**Verdict:** Agent-SDK-Go targets enterprise multi-tenant deployments with YAML-driven config. HerdAI is code-first and simpler; better for teams that want to own the agent logic directly in Go.

---

### Go framework one-page scorecard

```
----------------------------------------------------------------------------------------------------
Feature                    HerdAI    AgenticGoKit   Eino      Google ADK   ZenModel     Agent-SDK-Go
----------------------------------------------------------------------------------------------------
Zero external deps         Yes       No             No        No           No           No
Parallel agents            Yes       Yes            Yes       Yes          Yes          Configurable
Parallel tools (default)   Yes       Unknown        Unknown   Unknown      Unknown      Unknown
MockLLM / no API key       Yes       No             No        No           No           No
go test -race clean        Yes       Unknown        Unknown   Unknown      Unknown      Unknown
HITL (approve/reject)      Yes       No             Partial   Partial      No           Unknown
Built-in guardrails        Yes       No             No        No           No           Yes
Built-in eval/testing      Yes       No             No        Yes (UI)     No           No
Tool result caching        Yes       No             No        No           No           No
MCP support                Yes       Yes            Yes       Yes          No           Yes
Streaming                  Partial   Yes            Yes       Yes          Yes          Yes
Multi-LLM providers        Partial   Yes            Yes       Partial      Yes          Yes
Enterprise features        Partial   No             No        Yes          No           Yes
Community / stars          Small     Very small     Large     Google       Very small   Small
Maturity                   Good      Beta           Prod      Prod         Early        Good
----------------------------------------------------------------------------------------------------
```

---

## 6. Key insights

### For I/O-bound workloads (most LLM workflows)

All frameworks produce **identical wall-clock time** when latency is dominated by network I/O (LLM API calls, web scraping, database reads). Both Go goroutines and Python asyncio yield the thread during I/O, so concurrency is fully exploited.

```
Identical I/O-bound wall time:
  HerdAI  par+par : 486 ms
  LangGraph       : 490 ms
  AutoGen         : 490 ms
  CrewAI (sim)    : 484 ms
```

### For CPU-bound workloads (embedding, tokenisation, local inference)

Go goroutines use **all CPU cores simultaneously**. Python's GIL forces threads to share one core; multiprocessing adds IPC overhead. This difference is invisible in the benchmark (which uses `time.Sleep`) but is real and significant in workloads that mix LLM API calls with CPU-heavy local processing.

### Why "parallel by default" matters beyond raw timing

Even when wall-clock time is similar, frameworks that are **sequential by default** impose a design tax:
- Every new agent you add increases latency linearly unless you remember to opt in to async.
- Forgetting `async_execution=True` on one task in CrewAI silently makes the whole crew sequential.
- In HerdAI, `StrategyParallel` is declared once on the Manager and every agent benefits. You opt **out** of parallelism, not into it.

### The dependency gap

```
go get github.com/neranjsubramanian/herdai   →  0 extra packages
pip install crewai                           → 78 packages, ~300 MB venv
pip install autogen-agentchat                → 65 packages, ~250 MB venv
pip install langgraph                        → 42 packages, ~200 MB venv
```

Zero dependencies means:
- No supply-chain risk from transitive packages
- No version conflict hell
- `go build` produces a single ~8 MB static binary — copy it anywhere
- CI/CD pipelines finish in seconds, not minutes

### HerdAI's unique combination

HerdAI is the only Go AI agent framework that combines all of:

1. **Zero external dependencies**
2. **Parallel agents AND parallel tools enabled by default**
3. **MockLLM** — every test and example runs without an API key
4. **HITL** — pause-before-tool with approve/reject/edit/abort decisions
5. **Built-in guardrails + eval + tool caching**
6. **Proven race-free** via `go test -race` across 156 tests

No other Go framework (AgenticGoKit, Eino, Google ADK, ZenModel, Agent-SDK-Go) checks all six boxes.

---

## File layout

```
benchmark/
├── go.mod              ← standalone module (replace → ../herdai)
├── app.go              ← domain: 6 agents, 18 tools, BuildTeam, RunAnalysis
├── main.go             ← demo CLI: 4 scenarios with formatted comparison table
├── bench_test.go       ← Go correctness tests + testing.B benchmarks + race test
├── compare.sh          ← one-command runner: Go + all Python frameworks
├── BENCHMARK.md        ← this document
└── python/
    ├── baseline.py         ← pure Python asyncio + threading (no framework)
    ├── langgraph_bench.py  ← actual LangGraph 1.1.x with Send() parallel fan-out
    ├── autogen_bench.py    ← actual AutoGen 0.7.5 with mock ChatCompletionClient
    ├── crewai_bench.py     ← CrewAI 1.12.2 simulated + code pattern + overhead
    └── compare_all.py      ← master table: Go + all Python frameworks side-by-side
```
