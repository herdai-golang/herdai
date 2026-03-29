# HerdAI

[![Go 1.24+](https://img.shields.io/badge/Go-1.24+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 156 passing](https://img.shields.io/badge/Tests-156%20passing-brightgreen)]()
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-blue)]()

## HerdAI — what it is

**HerdAI** is a production-oriented AI agent framework for Go: you build agents that call LLMs, use tools, run in teams, and optionally use memory, RAG, guardrails, and human approval — with no extra runtime dependencies in the library itself.

### Features

- **Agents** — Role, goal, optional backstory; `Run` with timeouts, max tool calls, and structured logging.
- **Tools** — First-class function calling: name, description, parameters, `Execute`; LLM decides when to invoke.
- **Parallel tool execution** — Multiple tool calls from one LLM turn can run concurrently (configurable).
- **Multi-agent managers** — Four strategies: sequential (pipeline), parallel (fan-out), round-robin (turn-taking), LLM router (supervisor picks the next agent). Managers are nestable (e.g. parallel team inside a sequential pipeline).
- **Conversations** — Thread-safe transcript; pass `nil` if you do not need history.
- **LLM providers** — OpenAI-compatible stack: OpenAI, Mistral, Groq, Ollama, etc.; per-agent LLM choice; `MockLLM` for tests without API keys.
- **RAG** — Ingest from files, URLs, strings, directories; in-memory or pluggable stores; `SimpleRAG` (keyword) or hybrid/embeddings; optional citations, `MinScore`, query rewriting; documents can be added mid-session.
- **Memory** — Multi-layer store (facts, episodes, instructions, summaries) with search, tags, TTL, scoping, import/export.
- **Guardrails** — Chains on input and output (length, patterns, keywords, JSON shape, PII/injection filters, redaction, custom rules).
- **Human-in-the-loop (HITL)** — Pause before tool execution: approve, reject, edit args, approve-all, or abort; policies (none / all / dangerous list / custom); CLI-style or channel-based handlers for UIs.
- **Tracing** — Hierarchical spans (agent, LLM, tool, manager, MCP, memory, RAG, etc.) with stats and JSON export.
- **Sessions** — Persist and resume conversations (e.g. file-backed store), checkpoints and lifecycle states.
- **Eval harness** — Suites with multiple cases and built-in assertions (content, length, JSON, tools used, duration, custom), reports and baseline comparison.
- **MCP** — Connect to Model Context Protocol servers so tools are discovered at runtime (multiple servers per agent).
- **Tool cache (optional)** — Cache tool results with invalidation when context shifts.
- **Tests & examples** — Broad test coverage and runnable examples (minimal, tools, supervisor, benchmarks, HITL channel, concurrent runs, RAG, etc.).

### Install

```bash
go get github.com/herdai-golang/herdai@latest
```

```go
import "github.com/herdai-golang/herdai"
```

---

### Runnable examples (in this repo)

All examples work **without an API key** — they use `MockLLM`. Clone the repo and run any of them:

| Directory | What it shows |
|-----------|----------------|
| [`examples/hello_minimal`](examples/hello_minimal) | Smallest program: one agent, one answer |
| [`examples/single_agent_tools`](examples/single_agent_tools) | One agent, multiple tools — two parallel tool calls then a final reply |
| [`examples/supervisor_three_agents`](examples/supervisor_three_agents) | **Supervisor pattern**: `StrategyLLMRouter` picks among three specialists |
| [`examples/concurrent_questions`](examples/concurrent_questions) | 8 goroutines each using their own agent simultaneously — race-safe pattern |
| [`examples/hitl_channel`](examples/hitl_channel) | **Human-in-the-loop** via `ChannelHITLHandler` — simulates a UI/WebSocket approval flow |
| [`examples/rag_simple`](examples/rag_simple) | Keyword RAG with `SimpleRAG` and an in-memory vector store |
| [`examples/concurrency_benchmark`](examples/concurrency_benchmark) | Parallel vs sequential benchmark; see also [`benchmark/`](benchmark/) for the full suite |

```bash
cd examples/hello_minimal          && go run .
cd examples/single_agent_tools     && go run .
cd examples/supervisor_three_agents && go run .
cd examples/concurrent_questions   && go run .
cd examples/hitl_channel           && go run .
cd examples/rag_simple             && go run .
```

---

## Table of Contents

- [HerdAI — what it is](#herdai--what-it-is)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Agents](#agents)
  - [Tools](#tools)
  - [Managers (Multi-Agent)](#managers-multi-agent)
  - [Conversations](#conversations)
- [LLM Providers](#llm-providers)
- [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
- [Memory](#memory)
- [Guardrails](#guardrails)
- [Human-in-the-Loop](#human-in-the-loop)
- [Tracing](#tracing)
- [Tool Caching](#tool-caching)
- [Sessions (Persistence)](#sessions-persistence)
- [Eval / Testing](#eval--testing)
- [MCP Integration](#mcp-integration)
- [Benchmarks](#benchmarks)
- [Comparison with Other Frameworks](#comparison-with-other-frameworks)
- [Running Tests](#running-tests)

---

## Project Structure

This repository contains **only the framework** — the library you import. Applications that use HerdAI live in **separate Go modules**.

```
herdai/
├── *.go                 ← The framework. You import this.
├── *_test.go            ← 156 unit and integration tests
├── go.mod               ← Module: github.com/herdai-golang/herdai
├── examples/            ← Runnable examples (all MockLLM, no API key)
├── benchmark/           ← Concurrency benchmark suite (Go + Python comparison)
├── docs/                ← Long-form documentation
├── LICENSE
└── README.md
```

For local development in your own module:

```
work/
├── herdai/              ← This library
└── my-app/              ← Your service (go.mod: replace github.com/herdai-golang/herdai => ../herdai)
```

---

## Quick Start

### 1. Minimal agent (no API key)

```go
package main

import (
    "context"
    "fmt"

    "github.com/herdai-golang/herdai"
)

func main() {
    mock := &herdai.MockLLM{}
    mock.PushResponse(herdai.LLMResponse{
        Content: "Go is great for building fast, concurrent backend services.",
    })

    agent := herdai.NewAgent(herdai.AgentConfig{
        ID:   "assistant",
        Role: "Helpful Assistant",
        Goal: "Answer questions clearly and concisely.",
        LLM:  mock,
    })

    result, err := agent.Run(context.Background(), "Why should I use Go?", nil)
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Content)
}
```

### 2. With a real LLM (Mistral)

```go
llm := herdai.NewMistral(herdai.OpenAIConfig{
    Model: "mistral-small-latest",
})

agent := herdai.NewAgent(herdai.AgentConfig{
    ID:   "researcher",
    Role: "Research Assistant",
    Goal: "Help users understand technical topics.",
    LLM:  llm,
})

result, _ := agent.Run(ctx, "Explain microservices vs monoliths", nil)
fmt.Println(result.Content)
```

Set your key first: `export MISTRAL_API_KEY=your-key-here`

---

## Core Concepts

### Agents

An `Agent` is the basic unit. It has a role, goal, optional tools, and an LLM. Call `agent.Run(ctx, input, conv)` and it returns a `*Result`.

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:           "analyst",
    Role:         "Market Analyst",
    Goal:         "Provide data-driven market insights.",
    Backstory:    "You are a senior analyst at a top consulting firm.",
    LLM:          llm,
    Timeout:      60 * time.Second, // default: 2 minutes
    MaxToolCalls: 5,                // default: 10
})
```

### Tools

Tools give agents capabilities beyond text generation. `Parameters` is a slice of `ToolParam`; `Execute` always receives a `context.Context` as its first argument:

```go
weatherTool := herdai.Tool{
    Name:        "get_weather",
    Description: "Get the current weather for a city",
    Parameters: []herdai.ToolParam{
        {Name: "city", Type: "string", Description: "City name", Required: true},
    },
    Execute: func(ctx context.Context, args map[string]any) (string, error) {
        city := args["city"].(string)
        return fmt.Sprintf("Weather in %s: 22°C, sunny", city), nil
    },
}

agent := herdai.NewAgent(herdai.AgentConfig{
    ID:    "weather-bot",
    Role:  "Weather Assistant",
    Goal:  "Tell users the weather",
    LLM:   llm,
    Tools: []herdai.Tool{weatherTool},
})
```

When the LLM returns multiple tool calls in one response, HerdAI executes them **concurrently by default** (`ParallelToolCalls: true`). To disable:

```go
seqOnly := false
agent := herdai.NewAgent(herdai.AgentConfig{
    ParallelToolCalls: &seqOnly,
    // ...
})
```

> **Example:** [`examples/single_agent_tools`](examples/single_agent_tools) — two tools called in parallel, result merged.

### Managers (Multi-Agent)

A `Manager` orchestrates multiple agents. Pick a strategy:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `StrategySequential` | Agents run one after another; each gets the previous agent's output | Pipelines (research → write → edit) |
| `StrategyParallel` | All agents run concurrently in goroutines; results are merged | Independent analyses (SWOT + Porter + PESTEL) |
| `StrategyRoundRobin` | Agents take turns until done or max turns reached | Iterative refinement (propose → critique → refine) |
| `StrategyLLMRouter` | An LLM picks which agent runs next | Dynamic / supervisor workflows |

**Sequential pipeline:**

```go
pipeline := herdai.NewManager(herdai.ManagerConfig{
    ID:       "report-pipeline",
    Strategy: herdai.StrategySequential,
    Agents:   []herdai.Runnable{researcherAgent, writerAgent},
})
result, _ := pipeline.Run(ctx, "Analyze the AI code review market", conv)
```

**Parallel analysis (all agents start at the same time):**

```go
team := herdai.NewManager(herdai.ManagerConfig{
    ID:       "analysis-team",
    Strategy: herdai.StrategyParallel,
    Agents:   []herdai.Runnable{porterAgent, swotAgent, pestelAgent},
})
// All three run in goroutines; wall time ≈ slowest agent, not sum of all.
```

**Nested managers (hierarchical teams):**

`Manager` implements `Runnable`, so you can nest them freely:

```go
researchTeam := herdai.NewManager(herdai.ManagerConfig{
    ID:       "research-team",
    Strategy: herdai.StrategyParallel,
    Agents:   []herdai.Runnable{marketAgent, techAgent, competitorAgent},
})

fullPipeline := herdai.NewManager(herdai.ManagerConfig{
    ID:       "pipeline",
    Strategy: herdai.StrategySequential,
    Agents:   []herdai.Runnable{researchTeam, synthesizerAgent},
})
```

> **Example:** [`examples/supervisor_three_agents`](examples/supervisor_three_agents) — `StrategyLLMRouter` supervisor picks among specialists.

### Conversations

A `Conversation` is a thread-safe message history shared between agents:

```go
conv := herdai.NewConversation()

result1, _ := researcher.Run(ctx, "Find market data", conv)
result2, _ := writer.Run(ctx, "Write a report based on the research", conv)
```

Pass `nil` when you don't need conversation history. All `Conversation` methods are safe for concurrent use — required by `StrategyParallel` where multiple agents write simultaneously.

---

## LLM Providers

HerdAI works with any OpenAI-compatible API. Each agent can use a different provider.

```go
// OpenAI
openaiLLM := herdai.NewOpenAI(herdai.OpenAIConfig{Model: "gpt-4o-mini"})

// Mistral
mistralLLM := herdai.NewMistral(herdai.OpenAIConfig{Model: "mistral-small-latest"})

// Groq (OpenAI-compatible endpoint)
groqLLM := herdai.NewOpenAI(herdai.OpenAIConfig{
    BaseURL: "https://api.groq.com/openai/v1",
    APIKey:  os.Getenv("GROQ_API_KEY"),
    Model:   "llama-3.1-70b-versatile",
})

// Ollama (local, no API key)
ollamaLLM := herdai.NewOpenAI(herdai.OpenAIConfig{
    BaseURL: "http://localhost:11434/v1",
    Model:   "llama3.1",
    APIKey:  "ollama",
})

// MockLLM (testing — no API key, deterministic responses)
mock := herdai.NewMockLLM(
    herdai.MockResponse{Content: "Hello!"},
    herdai.MockResponse{Content: "Second response."},
)
```

**Per-agent providers** — mix LLMs in one team:

```go
analyst  := herdai.NewAgent(herdai.AgentConfig{LLM: mistralLLM, ...})
writer   := herdai.NewAgent(herdai.AgentConfig{LLM: openaiLLM, ...})
reviewer := herdai.NewAgent(herdai.AgentConfig{LLM: ollamaLLM, ...})
```

---

## RAG (Retrieval-Augmented Generation)

RAG lets agents answer questions grounded in your documents. HerdAI supports loading from **files**, **strings**, **URLs**, **directories**, and `io.Reader` — and you can add documents at any point during a conversation.

### How it works

1. **Ingest** documents at startup (split into chunks, embed, store in vector store)
2. **Query** — on each agent call, retrieve the most relevant chunks
3. **Generate** — the LLM answers using the retrieved context

### Load from files

```go
store    := herdai.NewInMemoryVectorStore()
loader   := herdai.NewTextLoader("docs/product.md")
pipeline := herdai.NewIngestionPipeline(herdai.IngestionConfig{
    Loader:   loader,
    Chunker:  herdai.DefaultChunker(),
    Embedder: herdai.NewNoOpEmbedder(), // keyword search (no API key)
    Store:    store,
})
pipeline.Ingest(ctx)
```

### Load from a URL

```go
urlLoader := herdai.NewURLLoader("https://example.com/docs/api-reference")
docs, _   := urlLoader.Load(ctx)
herdai.IngestDocuments(ctx, store, herdai.DefaultChunker(), herdai.NewNoOpEmbedder(), docs...)

// Multiple URLs at once:
loader := herdai.NewMultiURLLoader("https://example.com/page1", "https://example.com/page2")
```

### Load from strings or a directory

```go
herdai.NewStringsLoader(map[string]string{
    "policy.md": "All employees must...",
    "faq.md":    "Q: How does billing work?...",
})

herdai.NewDirectoryLoader("docs/", []string{".md", ".txt"})
```

### Attach RAG to an agent

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:   "doc-assistant",
    Role: "Documentation Assistant",
    Goal: "Answer questions using the knowledge base.",
    LLM:  llm,
    RAG:  herdai.SimpleRAG(store, 5), // retrieve top 5 chunks per query
})
```

### Advanced RAG configuration

```go
RAG: &herdai.RAGConfig{
    Retriever:   herdai.NewHybridRetriever(store, 0.7), // keyword + vector blend
    TopK:        10,
    MinScore:    0.3,
    CiteSources: true,
    QueryRewriter: func(input string) string {
        return "technical documentation: " + input
    },
}
```

### Semantic search with embeddings

The default `NoOpEmbedder` uses keyword matching. For semantic similarity:

```go
embedder := herdai.NewMistralEmbedder(herdai.EmbedderConfig{Model: "mistral-embed"})
// or
embedder := herdai.NewOpenAIEmbedder(herdai.EmbedderConfig{Model: "text-embedding-3-small"})
```

### Add documents mid-conversation

```go
newDoc := herdai.Document{Content: "Version 3.0 adds streaming support...", Source: "v3-notes.md"}
herdai.IngestDocuments(ctx, store, herdai.DefaultChunker(), herdai.NewNoOpEmbedder(), newDoc)
// The agent's next query automatically searches the new document.
```

---

## Memory

Multi-layer memory gives agents context from past interactions. Memories are recalled automatically before each LLM call.

```go
memory := herdai.NewInMemoryStore()

memory.Store(ctx, herdai.MemoryEntry{
    Kind:    herdai.MemoryFact,
    Content: "The user prefers Go over Python",
    Tags:    []string{"preference"},
})
memory.Store(ctx, herdai.MemoryEntry{
    Kind:    herdai.MemoryInstruction,
    Content: "Always format code examples with syntax highlighting",
})

agent := herdai.NewAgent(herdai.AgentConfig{
    Memory: memory,
    // ...
})
```

**4 memory kinds:**

| Kind | Purpose |
|------|---------|
| `MemoryFact` | Things the agent should know ("user is on the Pro plan") |
| `MemoryEpisode` | Past events ("analyzed competitor X on Jan 5") |
| `MemoryInstruction` | Standing orders ("always respond in bullet points") |
| `MemorySummary` | Compressed conversation histories |

Features: keyword search with relevance scoring, tag filtering, TTL expiration, session/agent scoping, export/import to JSON.

---

## Guardrails

Validate and transform inputs before the LLM sees them, and outputs before they reach the user:

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    InputGuardrails: herdai.NewGuardrailChain(
        herdai.ContentFilter("injection", "pii"),
        herdai.MaxLength(10000),
    ),
    OutputGuardrails: herdai.NewGuardrailChain(
        herdai.RedactPII(),
        herdai.MinLength(20),
        herdai.BlockKeywords("confidential", "internal only"),
    ),
    // ...
})
```

**10 built-in guardrails:**

| Guardrail | What it does |
|-----------|-------------|
| `MaxLength(n)` | Reject if input/output exceeds n characters |
| `MinLength(n)` | Reject if too short |
| `BlockPatterns(regex...)` | Block matching regex patterns |
| `RequirePatterns(regex...)` | Require matching patterns |
| `BlockKeywords(words...)` | Block specific words |
| `RequireJSON(fields...)` | Require valid JSON with specific fields |
| `ContentFilter(types...)` | Block PII, prompt injection |
| `RedactPII()` | Replace emails/phones/SSNs with `[REDACTED]` |
| `TrimWhitespace()` | Clean up whitespace |
| `CustomGuardrail(fn)` | Your own validation logic |

---

## Human-in-the-Loop

Pause agent execution for human approval before tool calls:

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    HITL: &herdai.HITLConfig{
        Policy:         herdai.HITLPolicyDangerous,
        DangerousTools: []string{"delete_file", "execute_command"},
        Handler:        herdai.NewCLIApprovalHandler(), // prompts in terminal
    },
    // ...
})
```

**Decisions:** `Approve`, `Reject`, `Edit` (modify tool arguments before running), `ApproveAll` (skip future prompts this run), `Abort`.

**Policies:**

| Policy | Behavior |
|--------|----------|
| `HITLPolicyNone` | Never ask (default) |
| `HITLPolicyAllTools` | Ask before every tool call |
| `HITLPolicyDangerous` | Only ask for tools in the dangerous list |
| `HITLPolicyCustom` | Your own function decides |

For WebSocket or UI integration, use `herdai.NewChannelHITLHandler()` instead of CLI.

> **Example:** [`examples/hitl_channel`](examples/hitl_channel) — channel-based approval handler.

---

## Tracing

Hierarchical span tracing for every agent, tool, LLM call, and RAG retrieval:

```go
tracer := herdai.NewTracer()
ctx    := herdai.ContextWithTracer(context.Background(), tracer)

result, _ := pipeline.Run(ctx, "Analyze the market", conv)

fmt.Println(tracer.Summary())
// ✓ [manager] pipeline       (2.3s) ok
//   ✓ [agent]   researcher   (1.1s) ok
//     ✓ [llm]   chat         (800ms) ok
//     ✓ [tool]  web_search   (300ms) ok  cached=false
//   ✓ [agent]   writer       (1.2s) ok
//     ✓ [llm]   chat         (1.1s) ok
//     ✓ [custom] rag:retrieve (5ms)  ok

stats := tracer.Stats()
// LLMCalls: 3, ToolCalls: 1, TotalSpans: 7

data := tracer.Export() // JSON for dashboards or offline analysis
```

**8 span kinds:** `agent`, `tool`, `llm`, `manager`, `mcp`, `memory`, `session`, `custom`

---

## Tool Caching

Context-aware caching for tool results. Automatically re-runs tools only when the context changes meaningfully.

```go
cache := herdai.NewToolCache(herdai.ToolCacheConfig{
    NewWordThreshold: 3,              // invalidate if 3+ meaningful words change
    MaxAge:           10*time.Minute, // optional TTL
    MaxEntries:       50,

    // Selective invalidation: only re-run tools whose fields actually changed
    ToolDeps: map[string][]string{
        "financial_analysis": {"idea", "industry", "revenue"},
        "competitor_intel":   {"idea", "industry", "customer"},
        "gtm_analysis":       {"idea", "customer", "geography"},
    },
})

// Wrap individual tool handlers with the cache:
myTool := herdai.Tool{
    Name:    "financial_analysis",
    Execute: cache.Wrap("financial_analysis", expensiveAPICall),
}

agent := herdai.NewAgent(herdai.AgentConfig{
    Tools:     []herdai.Tool{myTool},
    ToolCache: cache,
    // ...
})
```

When the user changes "customer" from "architects" to "hospitals", only tools whose `ToolDeps` include `"customer"` are invalidated; tools that don't depend on that field keep their cached results. Cache hits are visible as `cached: true` on tracing spans.

---

## Sessions (Persistence)

Save and resume agent conversations across restarts:

```go
store, _ := herdai.NewFileSessionStore("./sessions")

// Run and save
session := herdai.NewSession("market-analysis")
conv    := session.GetConversation()
result, _ := agent.Run(ctx, "Analyze AI market", conv)
session.AddResult(result)
store.Save(session)

// Resume exactly where you left off
loaded, _ := store.Load(session.ID)
loaded.Resume()
conv = loaded.GetConversation() // full history is restored
result, _ = agent.Run(ctx, "Now analyze competitors", conv)
```

Supports checkpoints, metadata, and 4 lifecycle states (`active`, `paused`, `completed`, `failed`).

---

## Eval / Testing

Built-in evaluation harness for testing agent quality and tracking regressions:

```go
suite := herdai.NewEvalSuite("quality-tests", agent)

suite.AddCase(herdai.EvalCase{
    Name:  "Answers about Go",
    Input: "What is Go good for?",
    Tags:  []string{"basic"},
    Assertions: []herdai.Assertion{
        herdai.AssertContains("concurrent"),
        herdai.AssertMinLength(100),
        herdai.AssertMaxDuration(10 * time.Second),
    },
})

report := suite.Run(ctx)
fmt.Println(report.Summary())
// ╔══════════════════════════════════╗
// ║ EVAL REPORT: quality-tests       ║
// ║ Total: 5  Passed: 4  Failed: 1   ║
// ║ Pass Rate: 80.0%   Duration: 3.2s ║
// ╚══════════════════════════════════╝

// Export and compare against a baseline to catch regressions:
report.ExportJSON("results/v1.json")
baseline, _ := herdai.LoadReport("results/v1.json")
current     := suite.Run(ctx)
fmt.Println(herdai.CompareReports(baseline, current))
```

**12 built-in assertions:** `AssertContains`, `AssertNotContains`, `AssertMinLength`, `AssertMaxLength`, `AssertJSON`, `AssertToolUsed`, `AssertToolNotUsed`, `AssertMaxToolCalls`, `AssertMaxDuration`, `AssertCustom`, and more.

---

## MCP Integration

[Model Context Protocol](https://spec.modelcontextprotocol.io/) lets agents discover tools from external servers at runtime:

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:   "web-researcher",
    Role: "Web Researcher",
    Goal: "Search the web and analyze findings",
    LLM:  llm,
    MCPServers: []herdai.MCPServerConfig{
        {Name: "search", Command: "./search_server"},
    },
})
// Agent auto-discovers tools from the MCP server via JSON-RPC 2.0
```

Multiple MCP servers per agent, manager-level propagation to all agents, and `DisableMCP: true` per agent to opt out.

---

## Benchmarks

HerdAI is built on two stacked layers of native Go concurrency:

- **Manager-level:** `StrategyParallel` runs all agents as goroutines simultaneously.
  Wall time ≈ slowest agent, not the sum.
- **Agent-level:** `ParallelToolCalls` (on by default) runs all tool calls in one LLM
  response as goroutines simultaneously. Wall time ≈ slowest tool, not the sum.

Both layers are **on by default**. Other frameworks require explicit opt-in.

### Competitive Intelligence Engine — measured results

**6 agents · 18 tool calls · 12 LLM calls per run.
Simulated latency: 200 ms per LLM call · 80 ms per tool call.
Hardware: Apple M3 Pro.**

| Scenario | HerdAI | LangGraph 1.1 | AutoGen 0.7 | CrewAI 1.12 | Pure asyncio |
|---|---|---|---|---|---|
| **par agents + par tools** | **486 ms** | 490 ms | 490 ms | ~484 ms | 484 ms |
| par agents + seq tools | **728 ms** | 733 ms | 727 ms | ~727 ms | 727 ms |
| seq agents + par tools | 2 904 ms | 2 905 ms | 2 902 ms | ~2 899 ms | 2 899 ms |
| **seq agents + seq tools** | 3 875 ms | 3 890 ms | 3 882 ms | ~3 871 ms | 3 871 ms |
| **Speedup (seq÷par)** | **8×** | 8× | 8× | 8× | 8× |

> **Key insight:** For I/O-bound workloads (network API calls), Go goroutines and Python asyncio achieve the same wall-clock time. The difference appears with CPU-bound work (embedding, tokenisation, local inference) where the Python GIL prevents simultaneous multi-core use — goroutines do not have this constraint.

### Formal Go micro-benchmarks (20 ms LLM · 8 ms tools)

```
BenchmarkParallelAgents_ParallelTools-11         100    51 ms/op
BenchmarkParallelAgents_SequentialTools-11        60    79 ms/op
BenchmarkSequentialAgents_ParallelTools-11        14   309 ms/op
BenchmarkSequentialAgents_SequentialTools-11      10   417 ms/op
```

**Scale sweep — adding agents costs nothing in parallel mode:**

```
BenchmarkScaleAgents_Parallel/agents=1   90   51.5 ms/op
BenchmarkScaleAgents_Parallel/agents=2   90   51.7 ms/op
BenchmarkScaleAgents_Parallel/agents=4   88   51.9 ms/op
BenchmarkScaleAgents_Parallel/agents=6   88   52.1 ms/op
```

Wall time is flat at ≈ 51 ms whether there is 1 agent or 6.

### Run the benchmarks yourself

```bash
cd benchmark

# Full comparison: Go demo app + all Python frameworks
bash compare.sh

# Go demo — 4 scenarios with detailed table (no API key)
go run . --llm-delay 200ms --tool-delay 80ms

# Formal Go micro-benchmarks
go test -bench=. -benchtime=5s -count=3

# Race detector — proves goroutine safety (expect: zero DATA RACE reports)
go test -race ./...

# Python frameworks individually
python3 python/baseline.py        # pure asyncio baseline
python3 python/langgraph_bench.py # actual LangGraph 1.1.x
python3 python/autogen_bench.py   # actual AutoGen 0.7.5
python3 python/crewai_bench.py    # CrewAI 1.12.2 timing + code pattern

# Master comparison table (Go + all Python)
python3 python/compare_all.py
```

See [`benchmark/BENCHMARK.md`](benchmark/BENCHMARK.md) for detailed methodology, expected outputs, and the full Go framework comparison.

---

## Comparison with Other Frameworks

### vs. Python frameworks (CrewAI · AutoGen · LangGraph)

| Feature | **HerdAI** | CrewAI | AutoGen | LangGraph |
|---------|-----------|--------|---------|-----------|
| **Language** | Go | Python | Python | Python |
| **Dependencies** | **Zero** | 78 packages | 65 packages | 42 packages |
| **Cold start** | **< 10 ms** (compiled binary) | ~5–8 s | ~4–6 s | ~3–5 s |
| **Memory (idle)** | **~20 MB** | ~220 MB | ~180 MB | ~150 MB |
| **Binary size** | **~8 MB** | ~300 MB venv | ~250 MB venv | ~200 MB venv |
| **Parallel agents (default)** | Yes | No (sequential by default) | No (must use asyncio.gather) | Yes (via Send fan-out) |
| **Parallel tools (default)** | Yes | No (not supported within one agent) | No (manual) | No (manual) |
| **CPU-bound parallelism** | Yes goroutines use all cores | No (GIL) | No (GIL) | No (GIL) |
| **HITL** | **Approve/Reject/Edit/Abort** | No | Basic | No |
| **Guardrails** | **10 built-in** | No | No | No |
| **Eval harness** | **Built-in** (12 assertions) | No | No | No |
| **Tool caching** | **Context-aware, field-aware** | No | No | No |
| **MockLLM (no API key)** | Yes | No | No | No |
| **Race-safe proof** | **`go test -race` (0 races)** | N/A | N/A | N/A |
| **Sessions** | **Save/Resume** | No | No | Checkpoints |
| **MCP** | **Built-in** | Plugin | Plugin | External |

### vs. Go frameworks (AgenticGoKit · Eino · Google ADK · ZenModel · Agent-SDK-Go)

| Feature | **HerdAI** | AgenticGoKit | Eino (ByteDance) | Google ADK Go | ZenModel | Agent-SDK-Go |
|---|---|---|---|---|---|---|
| **Zero external dependencies** | Yes | No | No | No | No | No |
| **Parallel agents (default)** | Yes | Yes | Yes | Yes | Yes | Configurable |
| **Parallel tools (default)** | Yes | Not documented | Not documented | Not documented | Not documented | Not documented |
| **MockLLM — no API key needed** | Yes | No | No | No | No | No |
| **`go test -race` clean** | **Yes (156 tests)** | Not published | Not published | Not published | Not published | Not published |
| **HITL (approve/reject/edit/abort)** | Yes | No | Partial (interrupt) | Partial (MCP confirmation) | No | Partial |
| **Built-in guardrails** | Yes | No | No | No | No | Yes |
| **Built-in eval / testing** | Yes | No | No | Yes (UI) | No | No |
| **Tool result caching** | Yes | No | No | No | No | No |
| **MCP** | Yes | Yes | Yes | Yes | No | Yes |
| **Tracing** | Yes spans | Yes (OpenTelemetry) | Yes (callbacks) | Yes (OpenTelemetry) | No | Yes |
| **Sessions / persistence** | Yes | Partial | Yes | Yes | Yes (SQLite) | Yes |
| **Streaming** | Provider-dep. | **Yes (streaming-first)** | Yes | Yes | Yes | Yes |
| **LLM providers** | OpenAI-compatible | OpenAI, Anthropic, Ollama, Azure, HF | OpenAI, Claude, Gemini, Ollama | Gemini-first | Any (processors) | OpenAI, Anthropic, Vertex |
| **Orchestration strategies** | Sequential, Parallel, RoundRobin, LLMRouter | Sequential, Parallel, DAG, Loop | Chain, Graph, Workflow | Modular hierarchy | Sequential, Parallel, Branching, Loop | Sequential, Parallel |
| **Maturity** | Good | Beta (v0.5.6) | Production (ByteDance) | Production (Google) | Early (v0.1.0) | Good (v0.2.x) |
| **Community stars** | Small | Very small (~120) | **Large (10 K+)** | **Google-backed** | Very small | Small |

**HerdAI is the only Go AI agent framework with all of:**
1. Zero external dependencies
2. Parallel agents AND parallel tools enabled by default
3. `MockLLM` — every example runs without an API key
4. HITL with approve / reject / edit / abort decisions
5. Built-in guardrails, eval harness, and context-aware tool caching
6. Proven race-free via `go test -race` across 156 tests

**Where other Go frameworks lead:**
- **Eino** — battle-tested at ByteDance (Doubao, TikTok) at massive scale; richest graph composition
- **Google ADK** — A2A protocol, 30+ databases via MCP Toolbox, Google Cloud integration
- **AgenticGoKit** — streaming-first API, broader LLM provider support, developer CLI tooling

---

## Running Tests

```bash
# All 156 tests (no API key or network access needed)
go test ./...

# Verbose
go test -v ./...

# Race detector — expect: zero DATA RACE reports
go test -race ./...

# Specific test
go test -run TestAgentWithRAG -v ./...
```

The test suite covers: agents, managers, tools, MCP, memory, tracing, HITL, sessions, guardrails, eval, RAG, tool cache, and conversations — all offline.

---

## License

MIT License. See [LICENSE](LICENSE).
