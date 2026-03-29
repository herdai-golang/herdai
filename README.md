# HerdAI

**A production-grade AI agent framework for Go.**

[![Go 1.24+](https://img.shields.io/badge/Go-1.24+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 156 passing](https://img.shields.io/badge/Tests-156%20passing-brightgreen)]()
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero-blue)]()

HerdAI lets you build AI agents in Go that can use tools, call LLMs, work in teams, remember context, validate inputs/outputs, and retrieve knowledge from documents — all with zero external dependencies.

### Install

```bash
go get github.com/herdai-golang/herdai@latest
```

```go
import "github.com/herdai-golang/herdai"
```

### Runnable examples (in this repo)

All of these work **without an API key** (they use `MockLLM`). From a clone of the repository:

| Directory | What it shows |
|-----------|----------------|
| [`examples/hello_minimal`](examples/hello_minimal) | Smallest program: one agent, one answer |
| [`examples/single_agent_tools`](examples/single_agent_tools) | One agent with **multiple tools** (two tool calls in parallel, then a final reply) |
| [`examples/supervisor_three_agents`](examples/supervisor_three_agents) | **Supervisor pattern**: `StrategyLLMRouter` picks among three specialists, then `FINISH` |
| [`examples/concurrency_benchmark`](examples/concurrency_benchmark) | Benchmark: parallel vs sequential managers (see that folder’s README) |

```bash
cd examples/hello_minimal && go run .
cd examples/single_agent_tools && go run .
cd examples/supervisor_three_agents && go run .
```

In your own module, after `go get`, copy an example `main.go` or follow [Quick Start](#quick-start) below.

---

## Table of Contents

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
- [Sessions (Persistence)](#sessions-persistence)
- [Eval / Testing](#eval--testing)
- [MCP Integration](#mcp-integration)
- [All Examples](#all-examples)
- [Comparison with Other Frameworks](#comparison-with-other-frameworks)
- [Running Tests](#running-tests)

---

## Project Structure

This repository contains **only the framework** — the library you import. Applications that use HerdAI (e.g. the Strategy Advisor) live in **separate Go modules** and depend on this library.

```
herdai/
├── *.go                 ← The framework. You import this.
├── *_test.go            ← Unit and integration tests
├── go.mod               ← Module: github.com/herdai-golang/herdai
├── examples/            ← Runnable examples (hello, tools, supervisor, benchmark)
├── docs/                ← Long-form documentation
├── LICENSE
└── README.md
```

**To use the framework**, import it in your own Go module:

```go
import "github.com/herdai-golang/herdai"
```

**Example layout** (library + your app):

```
work/
├── herdai/              ← This library
└── my-app/              ← Your service or CLI (imports herdai)
```

For local development, in `my-app/go.mod` you can use:

`replace github.com/herdai-golang/herdai => ../herdai`

The **Strategy Advisor** (a production-style web UI built on HerdAI) is maintained as a **separate module** (not in this tree).

---

## Quick Start

### 1. Minimal Agent (no API key needed)

Same idea as [`examples/hello_minimal`](examples/hello_minimal).

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

### 2. With a Real LLM (Mistral)

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

An Agent is the basic unit. It has a role, goal, optional tools, and an LLM. You call `agent.Run(ctx, input, conversation)` and it returns a `*Result`.

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:        "analyst",
    Role:      "Market Analyst",
    Goal:      "Provide data-driven market insights.",
    Backstory: "You are a senior analyst at a top consulting firm.",
    LLM:       llm,
    Timeout:   60 * time.Second,  // default: 2 minutes
    MaxToolCalls: 5,              // default: 10
})
```

### Tools

Tools give agents capabilities beyond text generation. Define a tool with a name, description, parameters, and an execute function:

```go
weatherTool := herdai.Tool{
    Name:        "get_weather",
    Description: "Get the current weather for a city",
    Parameters: map[string]herdai.ToolParam{
        "city": {Type: "string", Description: "City name", Required: true},
    },
    Execute: func(args map[string]any) (string, error) {
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

When the LLM decides it needs weather data, it calls the tool automatically through function calling.

> **Example:** [`examples/single_agent_tools`](examples/single_agent_tools) — one agent, multiple tools (mock LLM).

### Managers (Multi-Agent)

A Manager orchestrates multiple agents. There are 4 strategies:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `StrategySequential` | Agents run one after another. Each gets the previous agent's output. | Pipelines (research → write → edit) |
| `StrategyParallel` | All agents run concurrently. Results are merged. | Independent analyses (SWOT + Porter + PESTEL) |
| `StrategyRoundRobin` | Agents take turns until done. | Iterative refinement (propose → critique → refine) |
| `StrategyLLMRouter` | An LLM picks which agent runs next (supervisor). | Dynamic workflows |

> **Example:** [`examples/supervisor_three_agents`](examples/supervisor_three_agents) — supervisor + three specialists using `StrategyLLMRouter`.

**Sequential pipeline:**

```go
researcher := herdai.NewAgent(herdai.AgentConfig{
    ID: "researcher", Role: "Researcher", Goal: "Find key facts", LLM: llm,
})
writer := herdai.NewAgent(herdai.AgentConfig{
    ID: "writer", Role: "Writer", Goal: "Turn research into a clear report", LLM: llm,
})

pipeline := herdai.NewManager(herdai.ManagerConfig{
    ID:       "report-pipeline",
    Strategy: herdai.StrategySequential,
    Agents:   []herdai.Runnable{researcher, writer},
})

result, _ := pipeline.Run(ctx, "Analyze the AI code review market", conv)
```

**Parallel analysis:**

```go
porter := herdai.NewAgent(herdai.AgentConfig{
    ID: "porter", Role: "Porter's Five Forces Analyst", Goal: "Analyze competitive forces", LLM: llm,
})
swot := herdai.NewAgent(herdai.AgentConfig{
    ID: "swot", Role: "SWOT Analyst", Goal: "Identify strengths, weaknesses, opportunities, threats", LLM: llm,
})

team := herdai.NewManager(herdai.ManagerConfig{
    ID:       "analysis-team",
    Strategy: herdai.StrategyParallel,
    Agents:   []herdai.Runnable{porter, swot},
})
```

**Nested managers (hierarchical teams):**

Managers implement `Runnable`, so you can nest them:

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

Use [`examples/supervisor_three_agents`](examples/supervisor_three_agents) for LLM routing; compose `StrategySequential` / `StrategyParallel` managers as in the code above.

### Conversations

A `Conversation` is a thread-safe message history shared between agents:

```go
conv := herdai.NewConversation()

// First agent writes to it
result1, _ := researcher.Run(ctx, "Find market data", conv)

// Second agent reads the history
result2, _ := writer.Run(ctx, "Write a report based on the research", conv)
```

Pass `nil` if you don't need conversation history.

---

## LLM Providers

HerdAI works with any OpenAI-compatible API. Each agent can use a different provider.

```go
// OpenAI
openaiLLM := herdai.NewOpenAI(herdai.OpenAIConfig{
    Model: "gpt-4o-mini",
})

// Mistral
mistralLLM := herdai.NewMistral(herdai.OpenAIConfig{
    Model: "mistral-small-latest",
})

// Groq
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

// Mock (for testing, no API key needed)
mock := &herdai.MockLLM{}
mock.PushResponse(herdai.LLMResponse{Content: "Hello!"})
```

**Per-agent providers** — one team, multiple LLMs:

```go
analyst  := herdai.NewAgent(herdai.AgentConfig{LLM: mistralLLM, ...})
writer   := herdai.NewAgent(herdai.AgentConfig{LLM: openaiLLM, ...})
reviewer := herdai.NewAgent(herdai.AgentConfig{LLM: ollamaLLM, ...})
```

---

## RAG (Retrieval-Augmented Generation)

RAG lets agents answer questions grounded in your documents. HerdAI supports loading documents from **files**, **strings**, **URLs**, and **io.Reader** — and you can add new documents at any time during a conversation.

### How It Works

1. **Ingest** documents at startup (split into chunks, store in vector store)
2. **Query** — when the agent receives a question, it retrieves relevant chunks
3. **Generate** — the LLM answers using the retrieved context

### Load From Files

```go
store := herdai.NewInMemoryVectorStore()

loader := herdai.NewTextLoader("docs/product.md")
pipeline := herdai.NewIngestionPipeline(herdai.IngestionConfig{
    Loader:   loader,
    Chunker:  herdai.DefaultChunker(),
    Embedder: herdai.NewNoOpEmbedder(),
    Store:    store,
})
pipeline.Ingest(ctx)
```

### Load From a URL

```go
urlLoader := herdai.NewURLLoader("https://example.com/docs/api-reference")
docs, _ := urlLoader.Load(ctx)

herdai.IngestDocuments(ctx, store, herdai.DefaultChunker(), herdai.NewNoOpEmbedder(), docs...)
```

Load from multiple URLs at once:

```go
loader := herdai.NewMultiURLLoader(
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3",
)
```

### Load From Strings (In-Memory)

```go
loader := herdai.NewStringsLoader(map[string]string{
    "company-policy.md": "All employees must...",
    "product-faq.md":    "Q: How does billing work?...",
})
```

### Load an Entire Directory

```go
loader := herdai.NewDirectoryLoader("docs/", []string{".md", ".txt"})
```

### Create the RAG-Enabled Agent

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:   "doc-assistant",
    Role: "Documentation Assistant",
    Goal: "Answer questions using the knowledge base.",
    LLM:  llm,
    RAG:  herdai.SimpleRAG(store, 5), // retrieve top 5 chunks per query
})

result, _ := agent.Run(ctx, "How do I deploy the app?", conv)
```

### Add Documents Mid-Conversation

You don't need to restart. Add documents dynamically at any time:

```go
newDoc := herdai.Document{
    Content: "Version 3.0 adds streaming support and...",
    Source:  "release-notes-v3.md",
}
herdai.IngestDocuments(ctx, store, herdai.DefaultChunker(), herdai.NewNoOpEmbedder(), newDoc)

// The agent's next query will now search the new document too
result, _ := agent.Run(ctx, "What's new in version 3.0?", conv)
```

### Advanced RAG Configuration

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    // ...
    RAG: &herdai.RAGConfig{
        Retriever:   herdai.NewHybridRetriever(store, 0.7), // keyword + vector mix
        TopK:        10,
        MinScore:    0.3,
        CiteSources: true,
        QueryRewriter: func(input string) string {
            return "technical documentation: " + input
        },
    },
})
```

### Using Embeddings (for Semantic Search)

The default `NoOpEmbedder` uses keyword matching. For semantic similarity, plug in an embedder:

```go
embedder := herdai.NewMistralEmbedder(herdai.EmbedderConfig{
    Model: "mistral-embed",
})
// or
embedder := herdai.NewOpenAIEmbedder(herdai.EmbedderConfig{
    Model: "text-embedding-3-small",
})
```

---

## Memory

Multi-layer memory gives agents context from past interactions. Memories are automatically recalled before each LLM call.

```go
memory := herdai.NewInMemoryStore()

// Pre-load facts the agent should know
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
    ID:     "assistant",
    Role:   "Personal Assistant",
    Goal:   "Help the user with context from past interactions",
    LLM:    llm,
    Memory: memory,
})
```

**4 memory kinds:**

| Kind | Purpose |
|------|---------|
| `MemoryFact` | Things the agent should know ("user is on the Pro plan") |
| `MemoryEpisode` | Past events ("analyzed competitor X on Jan 5") |
| `MemoryInstruction` | Standing orders ("always respond in bullet points") |
| `MemorySummary` | Compressed histories |

Features: keyword search with relevance scoring, tag filtering, TTL expiration, session/agent scoping, export/import to JSON.

---

## Guardrails

Validate and transform inputs before the LLM sees them, and outputs before they reach the user:

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID: "safe-agent", Role: "Assistant", Goal: "Help safely", LLM: llm,

    InputGuardrails: herdai.NewGuardrailChain(
        herdai.ContentFilter("injection", "pii"), // block prompt injection & PII
        herdai.MaxLength(10000),
    ),

    OutputGuardrails: herdai.NewGuardrailChain(
        herdai.RedactPII(),       // remove emails, phone numbers, SSNs
        herdai.MinLength(20),
        herdai.BlockKeywords("confidential", "internal only"),
    ),
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
    ID: "careful-agent", Role: "Analyst", Goal: "Analyze with oversight",
    LLM: llm, Tools: tools,

    HITL: &herdai.HITLConfig{
        Policy:         herdai.HITLPolicyDangerous,
        DangerousTools: []string{"delete_file", "execute_command"},
        Handler:        herdai.NewCLIApprovalHandler(), // prompts in terminal
    },
})
```

**Decisions:** Approve, Reject, Edit (modify tool arguments), ApproveAll (skip future prompts), Abort.

**Policies:**

| Policy | Behavior |
|--------|----------|
| `HITLPolicyNone` | Never ask (default) |
| `HITLPolicyAllTools` | Ask before every tool call |
| `HITLPolicyDangerous` | Only ask for tools in the dangerous list |
| `HITLPolicyCustom` | Your own function decides |

For WebSocket/UI integration, use `herdai.NewChannelHITLHandler()` instead of CLI.

> **Example:** The **Strategy Advisor** app (separate module) is a full web chat built on HerdAI with human-in-the-loop.

---

## Tracing

OpenTelemetry-style hierarchical tracing for every agent, tool, LLM call, and RAG retrieval:

```go
tracer := herdai.NewTracer()
ctx := herdai.ContextWithTracer(context.Background(), tracer)

result, _ := pipeline.Run(ctx, "Analyze the market", conv)

fmt.Println(tracer.Summary())
```

Output:

```
✓ [manager] pipeline (2.3s) ok
  ✓ [agent] researcher (1.1s) ok
    ✓ [llm] chat (800ms) ok
    ✓ [tool] web_search (300ms) ok
  ✓ [agent] writer (1.2s) ok
    ✓ [llm] chat (1.1s) ok
    ✓ [custom] rag:retrieve (5ms) ok
```

```go
stats := tracer.Stats()
fmt.Printf("LLM calls: %d, Tool calls: %d, Total spans: %d\n",
    stats.LLMCalls, stats.ToolCalls, stats.TotalSpans)

data := tracer.Export() // JSON for analysis/dashboards
```

**8 span kinds:** `agent`, `tool`, `llm`, `manager`, `mcp`, `memory`, `session`, `custom`

---

## Sessions (Persistence)

Save and resume agent conversations across restarts:

```go
store, _ := herdai.NewFileSessionStore("./sessions")

// Create session and run
session := herdai.NewSession("market-analysis")
conv := session.GetConversation()
result, _ := agent.Run(ctx, "Analyze AI market", conv)
session.AddResult(result)
store.Save(session)

// Later: resume exactly where you left off
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
```

Output:

```
╔══════════════════════════════════════════════════════╗
║  EVAL REPORT: quality-tests                         ║
║  Total: 5   Passed: 4   Failed: 1                   ║
║  Pass Rate: 80.0%    Duration: 3.2s                  ║
╚══════════════════════════════════════════════════════╝
```

```go
report.ExportJSON("results/v1.json")

// Compare with a baseline to catch regressions
baseline, _ := herdai.LoadReport("results/v1.json")
current := suite.Run(ctx)
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

Multiple MCP servers per agent, manager-level propagation, and `DisableMCP: true` to opt out.

---

## All examples

This repository ships **runnable** examples under `examples/` (see the [table at the top](#runnable-examples-in-this-repo)). They use `MockLLM` unless you swap in `NewMistral` / `NewOpenAI` with a real API key.

| App | Description |
|-----|-------------|
| **Strategy Advisor** (optional, separate repo) | Full web app for startup idea validation — not part of this module. |

Patterns such as RAG, MCP, streaming, and eval are covered in this README and in `docs/`; start from the snippets here and the [`examples/`](examples/) programs above.

---

## Comparison with Other Frameworks

| Feature | HerdAI | CrewAI (Python) | AutoGen (Python) | Eino (Go) | Google ADK (Go) |
|---------|-----------|----------------|-----------------|-----------|----------------|
| Language | **Go** | Python | Python | Go | Go |
| Dependencies | **Zero** | 50+ | 30+ | CloudWeGo | Google Cloud |
| Multi-agent strategies | **4** (seq, parallel, round-robin, LLM router) | Yes | Yes | Chain/Graph | Sub-agents |
| Multi-layer memory | **Yes** (4 kinds, TTL, tags, search) | Basic | Basic | Redis | Vertex RAG |
| RAG | **Built-in** (files, URLs, strings, embeddings) | Plugin | Plugin | External | Vertex |
| OTel-style tracing | **Yes** (hierarchical spans) | No | No | Callbacks | OpenTelemetry |
| HITL | **Approve/Reject/Edit/Abort** | No | Basic | Interrupt | Confirmation |
| Guardrails | **10 built-in** | No | No | Composition | Plugins |
| Eval harness | **Built-in** (12 assertions, regression) | No | No | External | Dev UI |
| Session persistence | **Save/Resume** | No | No | Context | Events |
| MCP | **Built-in** | Plugin | Plugin | Extension | Toolbox |
| Per-agent LLM | **Yes** | No | No | Yes | Yes |
| Binary size | **~8 MB** | ~200 MB+ | ~200 MB+ | Varies | Varies |
| Test suite | **156 tests** | Varies | Varies | Varies | Varies |

---

## Running Tests

```bash
# All 156 tests
go test ./...

# Verbose output
go test -v ./...

# Run a specific test
go test -run TestAgentWithRAG -v ./...
```

The tests cover: agents, managers, tools, MCP, memory, tracing, HITL, sessions, guardrails, eval, and RAG — all without any API keys or network access.

---

## License

MIT License. See [LICENSE](LICENSE).
