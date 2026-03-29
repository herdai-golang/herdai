# Why We Need HerdAI

A concise comparison with CrewAI and other frameworks, the key differentiators HerdAI offers, and where it can improve.

---

## Can HerdAI Be Used the Same as CrewAI?

**Yes, for the core use case.** Both are multi-agent frameworks: you define agents, give them roles and tools, and orchestrate them to complete tasks. You can build the same kinds of workflows in HerdAI that you would in CrewAI — researcher + writer pipelines, parallel analysts, LLM-routed flows, and RAG-augmented assistants.

**Main differences:**

| Aspect | CrewAI | HerdAI |
|--------|--------|-----------|
| **Language** | Python | Go |
| **Runtime** | Interpreted, large dependency tree | Single binary, zero external dependencies |
| **Deployment** | Container or venv with 50+ packages | One binary (~8 MB), drop on server |
| **Concurrency** | GIL-limited; true parallelism via processes | Native goroutines — agents and tools run in parallel by default |
| **Typing** | Optional (gradual typing) | Static typing; refactors and APIs are safer |
| **Ecosystem** | Rich Python ML/AI ecosystem | Smaller Go AI ecosystem; OpenAI-compatible APIs cover most needs |

**When to choose HerdAI over CrewAI:** You want a single binary, minimal ops, native parallel execution, or a Go-based stack. When you need Python-only libraries or a Python team, CrewAI (or AutoGen) remains a good fit.

---

## Key Differentiators HerdAI Offers

### 1. **Parallel Execution of Agents**

Managers can run multiple agents **concurrently** with the same input and merge results. No process spawning — lightweight goroutines.

```go
team := herdai.NewManager(herdai.ManagerConfig{
    ID:       "analysis-team",
    Strategy: herdai.StrategyParallel,
    Agents:   []herdai.Runnable{porter, swot, pestel, bmc, vrio, blueOcean},
})
result, _ := team.Run(ctx, "Analyze this market", conv)
// All 6 agents run at once; total time ≈ slowest agent, not sum of all.
```

**CrewAI:** Agents are typically sequential or require process-based parallelism; Python’s GIL limits true CPU parallelism in a single process.

---

### 2. **Parallel Execution of Tools**

When an LLM returns **multiple tool calls in one response**, HerdAI can execute them **concurrently** (configurable per agent). Example: “get_weather(London)” and “get_weather(Tokyo)” run in parallel.

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:                 "research",
    ParallelToolCalls:   &trueVal,  // run 2+ tool calls in parallel
    Tools:              tools,
    LLM:                llm,
})
```

So both **agent-level** (Manager) and **tool-level** (Agent) parallelism are first-class.

---

### 3. **Scalability**

- **Binary size:** ~8 MB static binary. No Python runtime, no pip installs in production.
- **Memory:** Go’s small stacks and efficient concurrency; many agents/tools can run in one process without the overhead of multiple Python processes.
- **Deployment:** Single binary; easy to scale horizontally (multiple instances behind a load balancer) or embed in larger Go services.
- **Cold start:** No interpreter or heavy framework init; process startup is fast.

This makes HerdAI suitable for **high-throughput or latency-sensitive** workloads (e.g. APIs, batch jobs) where Python agent frameworks are often heavier.

---

### 4. **Zero External Dependencies**

The core framework has **no third-party Go dependencies**. You get agents, tools, managers, MCP, memory, RAG, guardrails, HITL, tracing, and eval without pulling in a long dependency chain. Your app’s `go.mod` only adds what you need (e.g. LLM client, SQLite). That reduces supply-chain risk and upgrade friction.

---

### 5. **Human-in-the-Loop (HITL)**

HerdAI bakes in **human-in-the-loop** so you can pause before tool execution and let a human approve, reject, edit, or abort. That’s why it fits production and compliance-sensitive flows.

**Why HITL is in the framework:**

- **Safety:** High-impact tools (e.g. send email, delete data, run code) don’t run until a human approves.
- **Control:** Users can correct tool arguments (e.g. fix a wrong account ID) before the call runs.
- **Compliance and audit:** Every tool call can be gated by an approval step and logged.
- **UX:** UIs (web, Slack, CLI) plug in via a single handler; the framework handles the protocol.

**What you get:**

- **Decisions:** Approve (run as-is), Reject (skip and optionally send feedback to the agent), Edit (run with modified args), ApproveAll (auto-approve rest of run), Abort (stop the run).
- **Policies:** None (no HITL), AllTools (approve every call), Dangerous (only listed tools), or Custom (your own function).
- **Handlers:** Implement one callback; use it from a CLI prompt, WebSocket, or Slack. The framework sends `HITLRequest` (tool name, args, reason) and waits for `HITLResponse` (decision + optional edited args).

So “human in the flow” isn’t an afterthought — it’s a first-class path from agent → pause → human → continue.

---

### 6. **Intelligent Tool Result Caching**

HerdAI includes **context-aware caching** for tool results so you run expensive tools only when the input actually changes.

**Why caching is in the framework:**

- **Cost and latency:** LLM-backed or external tools (search, DB, APIs) are slow and costly. Reusing results for the same or similar context cuts both.
- **Consistency:** Re-asking “who are my competitors?” in the same session should return the same answer unless the user pivots.
- **Selective invalidation:** When the user changes one part of the context (e.g. “customer” from “architects” to “hospitals”), only tools that depend on that field are invalidated; others stay cached.

**What you get:**

| Feature | Description |
|--------|-------------|
| **Context key** | Each cache entry is keyed by tool name + context string (e.g. intake summary). Same context → cache hit. |
| **NewWordThreshold** | Configurable (default 3). If N or more *meaningful* words change (added, removed, or replaced) between cached context and new context, the cache is treated as stale. Trivial rephrases (“the” / “a”) don’t invalidate. |
| **Field-aware invalidation** | Optional `ToolDeps`: map each tool to the context fields it depends on (e.g. `financial_analysis` → `idea`, `industry`, `revenue`, `customer`). When you call `SetContextFields(fields)`, only tools whose dependent fields changed are invalidated. Others keep their cache. |
| **MaxAge** | Optional TTL; entries older than MaxAge are considered stale. |
| **MaxEntries** | Cap on total cached results; oldest evicted when full. |
| **Wrap** | `cache.Wrap(toolName, handler)` returns a tool handler that checks cache first and bypasses on `refresh: true` in args. |
| **Tracing** | Cache hits are recorded on spans (`cached: true`) so you can see reuse in traces. |

Example: in a strategy app, the user changes “customer” from “architects” to “hospitals”. With `ToolDeps`, `competitor_intel` and `gtm_analysis` (which depend on customer) are invalidated; `strategic_analysis` (idea, industry, problem) can stay cached. So caching is both **smart** (context diff) and **selective** (per-tool deps).

---

### 7. **Built-in Guardrails, Eval, and Tracing**

- **Guardrails:** Input/output validation (PII, injection, length, keywords) without extra services.
- **Eval:** Declarative test cases and assertions (ContainsText, ToolUsed, MaxDuration, etc.) and regression tracking.
- **Tracing:** Hierarchical spans for agents, tools, LLM calls, RAG; export for observability. Cache hits are visible as `cached: true` on tool spans.

These are built in rather than bolted on, so you can ship production-grade behavior without assembling multiple libraries.

---

### 8. **Four Orchestration Strategies**

| Strategy | Use case |
|----------|----------|
| **Sequential** | Pipelines: researcher → writer → editor |
| **Parallel** | Independent analyses merged in one shot (e.g. 6 strategy frameworks) |
| **Round-robin** | Iterative refinement: propose → critique → refine |
| **LLM router** | Dynamic flow: LLM decides which agent runs next |

CrewAI and others offer similar ideas; HerdAI implements them in one Manager API with a small, consistent surface.

---

### 9. **MCP and Session Persistence Built In**

- **MCP (Model Context Protocol):** Connect to external tool servers (stdio or HTTP); tools are discovered at runtime. No plugin layer to install separately.
- **Sessions:** Save/resume conversations and checkpoints so you can pause and continue later or replay flows.

---

## Room for Improvement

HerdAI is strong in runtime behavior, deployment, and concurrency; these are areas where it can grow:

| Area | Current state | Possible improvement |
|------|----------------|----------------------|
| **Ecosystem** | Smaller than Python’s; OpenAI-compatible APIs cover most LLMs | More out-of-the-box adapters (e.g. Anthropic, Gemini) and community examples |
| **Python users** | Go-only; no official Python SDK | A thin Python wrapper or separate Python SDK that calls a Go backend or REST API |
| **Low-code / UI** | Code-first; no visual flow builder | Optional UI or DSL for defining crews/flows and deploying them |
| **Streaming** | LLM streaming is provider-dependent | Standardized streaming API across providers for token-by-token or chunked output |
| **Observability** | Tracing export exists; no built-in dashboard | Export to OpenTelemetry/backend of choice; or a reference dashboard for spans |
| **Docs and examples** | README + docs; examples in separate repo or full source tree | More “recipes” (e.g. customer support bot, code review pipeline) and video walkthroughs |
| **RAG** | Built-in chunking, vector store, embedders | More embedders (e.g. Cohere, Voyager), optional reranking, and doc format support |
| **Cost control** | No built-in token/cost tracking | Optional cost hooks per LLM call and per-session summaries |

---

## Summary

- **Can you use HerdAI like CrewAI?** Yes — for multi-agent workflows, tools, and RAG-style apps. You trade Python and its ecosystem for Go, a single binary, and native parallelism.
- **Why choose HerdAI?** Parallel execution of agents and tools; scalability (binary size, memory, deployment); zero framework dependencies; **human-in-the-loop (HITL)** so humans can approve/reject/edit tool calls before they run; **intelligent tool result caching** (context-aware and field-aware invalidation) to save cost and latency; and built-in guardrails, eval, and tracing.
- **Where to improve?** Ecosystem breadth, Python-friendly options, low-code/UI tooling, streaming standardization, observability UX, RAG options, and cost visibility.

HerdAI is aimed at teams that want production-grade agent behavior, human oversight in the flow, smart caching, and clear control over execution and deployment — with the understanding that the Go ecosystem and code-first approach are the right fit.
