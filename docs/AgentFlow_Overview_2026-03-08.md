# HerdAI — Framework & App Overview
**Date: March 8, 2026**

---

## What is HerdAI?

HerdAI is a production-grade multi-agent framework written in Go. It provides the building blocks for creating AI-powered applications where one or more LLM-driven agents use tools, memory, and structured orchestration to accomplish complex tasks.

The framework is split into two parts:
1. **`herdai/`** — the reusable Go framework (any app can import and use it)
2. **`herdai/examples/webui/`** — a Strategy Advisor app built on top of it

---

## Part 1: The HerdAI Framework

### All Framework Features

#### 1. Agents
An `Agent` is an autonomous unit with a role, goal, backstory, and access to tools. It runs a ReAct loop (Thought → Action → Observation) powered by an LLM, calling tools as needed until it has a final answer.

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:    "analyst",
    Role:  "Market Analyst",
    Goal:  "Analyze competitive landscape",
    LLM:   llm,
    Tools: []herdai.Tool{searchTool, dbTool},
})
result, _ := agent.Run(ctx, "Analyze BIM market", nil)
```

#### 2. Tools
Tools are capabilities exposed to agents. Each has a name, description, typed parameters, and an execute function. The LLM decides when to call them based on the description.

```go
tool := herdai.Tool{
    Name:        "web_search",
    Description: "Search the internet",
    Parameters:  []herdai.ToolParam{{Name: "query", Type: "string", Required: true}},
    Execute:     func(ctx context.Context, args map[string]any) (string, error) { ... },
}
```

#### 3. Tool Caching (Smart Cache)
Framework-level caching for tool results with three layers of intelligence:

- **Word-drift detection**: Counts meaningful words added/removed (filters stop words) to detect when context changed
- **Structured context fields**: Tracks named fields (`customer`, `industry`, etc.) and compares field-by-field
- **Tool dependency mapping**: Each tool declares which fields it depends on — changing `customer` only invalidates tools that depend on `customer`, not all tools

```go
cache := herdai.NewToolCache(herdai.ToolCacheConfig{
    NewWordThreshold: 3,
    MaxAge:           30 * time.Minute,
    ToolDeps: map[string][]string{
        "financial_analysis": {"idea", "industry", "revenue", "customer"},
        "competitor_intel":   {"idea", "industry", "customer"},
    },
})

// User changes target customer mid-conversation:
invalidated := cache.SetContextFields(map[string]string{"customer": "hospitals"})
// Only financial_analysis and competitor_intel are invalidated
// strategic_analysis stays cached (doesn't depend on customer)
```

#### 4. MCP (Model Context Protocol)
Connect to external MCP servers via stdio or HTTP transport. Tools are auto-discovered and made available to agents.

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    MCPServers: []herdai.MCPServerConfig{
        {Name: "web-search", Command: "./mcp-server"},          // stdio
        {Name: "company-db", URL: "http://localhost:9477/mcp"}, // HTTP
    },
})
```

#### 5. Memory
Pluggable memory store with support for facts, episodes, instructions, and summaries. Includes search, tagging, TTL expiry, and session/agent scoping.

```go
mem := herdai.NewInMemoryStore()
mem.Store(ctx, herdai.MemoryEntry{
    Kind:    herdai.MemoryFact,
    Content: "User targets enterprise healthcare",
    Tags:    []string{"customer", "market"},
})
results, _ := mem.Search(ctx, "healthcare", 5)
```

#### 6. HITL (Human-in-the-Loop)
Pause before tool execution for human approval. Supports five decisions: Approve, Reject, Edit (modify args), ApproveAll, and Abort.

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    HITL: &herdai.HITLConfig{
        Policy:         herdai.HITLDangerous,
        DangerousTools: []string{"delete_data", "send_email"},
        Handler:        herdai.CLIApprovalHandler(),
        Timeout:        2 * time.Minute,
    },
})
```

#### 7. Guardrails
Input/output validation chains. Run before LLM sees input and after output is generated. 10+ built-in guardrails including PII redaction, content filtering, pattern blocking, JSON validation.

```go
chain := herdai.NewGuardrailChain(
    herdai.ContentFilter("injection"),
    herdai.BlockKeywords("password", "secret"),
    herdai.MaxLength(5000),
    herdai.RedactPII(),
)
agent := herdai.NewAgent(herdai.AgentConfig{
    InputGuardrails:  chain,
    OutputGuardrails: herdai.NewGuardrailChain(herdai.MinLength(50)),
})
```

#### 8. RAG (Retrieval-Augmented Generation)
Full Load → Chunk → Embed → Store → Retrieve pipeline.

- **Loaders**: Text, Directory, URL, Multi-URL, Reader, Strings
- **Chunkers**: FixedSize, Paragraph, Markdown, Recursive
- **Embedders**: OpenAI, Mistral, NoOp
- **Stores**: InMemoryVectorStore
- **Retrievers**: Keyword, Vector, Hybrid (combines both)

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    RAG: &herdai.RAGConfig{
        Retriever:  herdai.NewHybridRetriever(store, embedder, 0.5),
        TopK:       5,
        MinScore:   0.3,
        CiteSources: true,
    },
})
```

#### 9. Tracing
OpenTelemetry-style tracing with spans for every operation (agent runs, LLM calls, tool executions, guardrail checks, RAG retrieval). Supports nested spans, events, attributes, and JSON export.

```go
tracer := herdai.NewTracer()
ctx := herdai.ContextWithTracer(context.Background(), tracer)
agent.Run(ctx, "analyze", nil)
fmt.Println(tracer.Summary())
// Agent: 1 span, LLM: 3 calls, Tool: 2 calls, duration: 4.2s
```

#### 10. Evaluation Suite
Test harness for agents with declarative assertions and regression tracking.

```go
suite := herdai.NewEvalSuite(agent, "market-analysis-tests")
suite.AddCase(herdai.EvalCase{
    Name:  "BIM market analysis",
    Input: "Analyze the BIM market",
    Assertions: []herdai.Assertion{
        herdai.AssertContains("market size"),
        herdai.AssertToolUsed("web_search"),
        herdai.AssertMaxDuration(30 * time.Second),
    },
})
report := suite.Run(ctx)
```

#### 11. Manager (Multi-Agent Orchestration)
Orchestrate multiple agents with configurable strategies:

- **Sequential**: Agents run one after another, each receiving the previous output
- **Parallel**: All agents run concurrently, results synthesized by LLM
- **RoundRobin**: Agents take turns in a conversation loop
- **LLMRouter**: An LLM dynamically picks which agent should handle each input

```go
mgr := herdai.NewManager(herdai.ManagerConfig{
    Strategy:        herdai.StrategyParallel,
    Agents:          []herdai.Runnable{analystAgent, financeAgent},
    SynthesisPrompt: "Combine the analyses into a single coherent report.",
    LLM:             llm,
})
```

#### 12. Sessions & Checkpointing
Persist conversation state, agent results, memory, and checkpoints for crash recovery. Supports file-based and in-memory stores.

```go
store := herdai.NewFileSessionStore("./sessions")
session := herdai.NewSession(store)
session.SetCheckpoint("analyst", map[string]any{"progress": 50})
// On crash recovery:
mgr.ResumeRun(ctx, sessionID, store)
```

#### 13. Parallel Tool Execution
When the LLM returns multiple tool calls in a single response, they execute concurrently via goroutines. Results are collected in the original order.

#### 14. Conversations
Thread-safe transcript shared across agents and managers. Tracks turns, tool calls, durations, and cached status.

#### 15. LLM Integration
Pluggable LLM interface. Ships with OpenAI-compatible client (works with OpenAI, Mistral, and any OpenAI-compatible API) and a full MockLLM for deterministic testing.

---

## Part 2: The Strategy Advisor App (WebUI)

### What the App Does

The Strategy Advisor is an AI-powered business idea evaluation tool. A founder submits their idea through a structured intake form, and a Chief Strategy Officer (CSO) agent orchestrates 5 specialized analysis tools to deliver a comprehensive evaluation — then continues a conversational advisory session.

### The Architecture

```
User (Browser) ←→ WebSocket ←→ CSO Agent (Mistral Small)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              ToolCache        5 Analysis       MCP Servers
          (field-aware)          Tools          (web search,
                                    │           company DB)
                    ┌───────┬───────┼───────┬────────┐
                    │       │       │       │        │
               Strategic Financial Competitor GTM  Consulting
               Analysis  Analysis   Intel   Analysis Evaluation
                    │
            ┌───┬───┼───┬───┬───┐
            │   │   │   │   │   │
          Porter SWOT BMC PESTEL VRIO Blue
           5F              Ocean
           (all 6 run in parallel)
```

### The 5 Analysis Tools

| Tool | What It Does | Depends On |
|------|-------------|------------|
| `strategic_analysis` | Runs 6 frameworks (Porter's, SWOT, BMC, PESTEL, VRIO, Blue Ocean) concurrently as sub-agents | idea, industry, problem, differentiator |
| `financial_analysis` | TAM/SAM/SOM, unit economics, revenue projections, break-even with real web data | idea, industry, revenue, customer, geography |
| `competitor_intel` | Searches 15,000+ companies (YC, a16z, Product Hunt, SEC, GitHub) via MCP + web search | idea, industry, customer, competitors, differentiator |
| `gtm_analysis` | Go-to-market: channels, pricing, sales cycle, launch plan with real benchmarks | idea, customer, geography, revenue, stage |
| `consulting_evaluation` | McKinsey-grade: 7S alignment, TEMPO risk matrix, readiness scoring, Go/No-Go verdict | idea, industry, customer, revenue, team, stage, problem |

### Memory System (4 Layers)

| Layer | Type | What It Does |
|-------|------|-------------|
| 1 | Working Memory | Rolling window of last 20 conversation turns |
| 2 | Semantic Memory | LLM-extracted categorized facts (idea, market, team, competitor, pivot, financial, etc.) |
| 3 | Episodic Memory | Compressed 2-3 sentence summaries of older conversation turns |
| 4 | Tool Cache | Cached analysis results with field-aware smart invalidation |

### Session Persistence

Every session is captured as a structured JSON record (`SessionRecord`) and persisted to SQLite. The record includes:
- Session ID, timestamps, status
- Intake form data
- `IdeaContext` — the living structured snapshot of the idea (used for cache invalidation)
- All conversation turns with tool call metadata
- Memory snapshot (facts + episodes)
- All analysis results
- LLM-generated session summary

---

## Sample Flows

### Flow 1: First-Time Idea Evaluation

```
1. User opens http://localhost:8080
2. Intake form appears
3. User fills in:
   - Idea: "AI-powered BIM tool that uses natural language to generate building designs"
   - Customer: "Architecture firms"
   - Industry: "Construction"
   - Revenue: "SaaS subscription"
4. User submits → IdeaContext baseline set on ToolCache
5. CSO receives the idea → calls ALL 5 tools in parallel:
   - strategic_analysis → spawns 6 sub-agents (Porter, SWOT, BMC, PESTEL, VRIO, Blue Ocean)
   - financial_analysis → agent with web search for real market data
   - competitor_intel → agent with company DB MCP + web search
   - gtm_analysis → agent with web search for channel/pricing data
   - consulting_evaluation → McKinsey 7S + TEMPO risk agent
6. All results cached in ToolCache with IdeaContext snapshot
7. CSO synthesizes everything into a conversational response:
   - 3-4 sentence summary paragraph
   - Detailed analysis with headings
   - Source attribution for all numbers
   - Confidence ratings
8. SessionRecord updated with turns + analysis results
```

### Flow 2: Follow-Up Question (Cache Hit)

```
1. User asks: "Tell me more about the financial opportunity"
2. CSO sees cached financial_analysis → presents it directly
3. No tools called → instant response
4. Memory extracts facts from the exchange
```

### Flow 3: Mid-Conversation Pivot (Selective Invalidation)

```
1. User says: "Actually, let's target hospitals instead of architects"
2. extractAndStoreFacts detects: FIELD_CHANGE|customer|hospitals
3. ToolCache.SetContextFields({"customer": "hospitals"})
4. Framework compares: customer changed → who depends on "customer"?
   - financial_analysis → INVALIDATED
   - competitor_intel → INVALIDATED
   - gtm_analysis → INVALIDATED
   - consulting_evaluation → INVALIDATED
   - strategic_analysis → STAYS CACHED (doesn't depend on customer)
5. CSO calls the 4 invalidated tools with updated context
6. strategic_analysis serves from cache (saves ~7 seconds)
7. CSO synthesizes fresh + cached results
8. IdeaContext updated in SessionRecord
```

### Flow 4: New Chat

```
1. User clicks "+" button in header
2. Session saved to SQLite with status "completed"
3. All caches cleared, memory cleared, history cleared
4. Fresh intake form appears
5. New SessionRecord created
```

---

## Part 3: Framework Comparison

### HerdAI vs CrewAI vs AutoGen vs LangGraph

| Feature | HerdAI (Go) | CrewAI (Python) | AutoGen (Python) | LangGraph (Python) |
|---------|:-:|:-:|:-:|:-:|
| **Language** | Go | Python | Python/.NET | Python |
| **Agent abstraction** | Agent with Role/Goal/Backstory + Tools | Agent with Role/Goal/Backstory + Tools | AssistantAgent with system message | Graph nodes (functions) |
| **Orchestration** | Manager (Sequential, Parallel, RoundRobin, LLMRouter) | Crew (Sequential, Hierarchical) | GroupChat with speaker selection | StateGraph with conditional edges |
| **Parallel tool execution** | Built-in (goroutines, automatic) | No | No | Manual (parallel fan-out nodes) |
| **Tool caching** | Framework-level with smart invalidation | Basic exact-match `cache` flag | None | None |
| **Field-aware cache invalidation** | Yes — tools declare field dependencies, selective invalidation | No | No | No |
| **Context drift detection** | Yes — word-level + field-level | No | No | No |
| **MCP support** | Built-in (stdio + HTTP transport) | Via plugin | No | Via LangChain integration |
| **Memory** | MemoryStore with facts, episodes, instructions, TTL, tags | Short-term + episodic + persistent | ListMemory (chronological) | Short-term (thread) + long-term (store) |
| **HITL** | 5 decisions (Approve/Reject/Edit/ApproveAll/Abort) + policies | Flow-based + webhook (Enterprise) | GroupChat interruption | Checkpoint-based interruption |
| **Guardrails** | 10+ built-in (PII, injection, patterns, JSON, length) | Basic validation | None built-in | None built-in |
| **RAG** | Full pipeline (Load→Chunk→Embed→Store→Retrieve) with 4 loaders, 4 chunkers, 3 embedders, 3 retrievers | Via tools/plugins | Memory protocol with RAG | Via LangChain integration |
| **Tracing** | OpenTelemetry-style spans, nested, with export | Verbose logging | OpenTelemetry support | LangSmith integration |
| **Evaluation** | Built-in EvalSuite with assertions and regression tracking | No built-in | No built-in | Via LangSmith |
| **Sessions/Checkpoints** | Built-in with file/memory stores + crash recovery | No built-in | State serialization | Checkpointer (Postgres, SQLite) |
| **Conversation history** | Thread-safe Conversation with tool call records | Task context passing | GroupChat message history | State-based |
| **Typing/Safety** | Go's static type system | Python dynamic | Python with type hints | Python with TypedDict |
| **Performance** | Compiled Go, goroutines for concurrency | Python (slower) | Python async | Python async |
| **Learning curve** | Moderate | Low (best onboarding) | Moderate | High (graph abstraction) |

### Where Each Framework Excels

**HerdAI** excels when you need:
- Production Go services with compiled performance and type safety
- Smart tool caching that knows which tools to re-run and which to skip
- A single framework that includes agents, tools, memory, HITL, guardrails, RAG, tracing, evaluation, sessions, and caching — without external dependencies
- MCP server integration (stdio + HTTP)
- Parallel tool execution without manual orchestration

**CrewAI** excels when you need:
- Fastest prototype-to-demo path
- Non-engineers reading agent definitions (role/goal/backstory is intuitive)
- Simple sequential or hierarchical workflows
- Python ecosystem integration

**AutoGen** excels when you need:
- Conversational multi-agent patterns (agents debating each other)
- Microsoft ecosystem integration
- Cross-language support (Python + .NET)
- Simple chat-based agent interactions

**LangGraph** excels when you need:
- Complex graph-shaped workflows with cycles and conditional branching
- Explicit, inspectable state machines
- LangSmith for production observability
- LangChain ecosystem integration

### What Only HerdAI Has

These features exist in HerdAI but not in any of the three comparison frameworks:

1. **Smart tool caching with field-aware invalidation** — no other framework caches tool results at the framework level, let alone selectively invalidates based on which structured context fields changed
2. **Tool dependency declarations** — `ToolDeps` mapping so the framework knows which tools care about which fields
3. **Built-in evaluation suite** — declarative test cases with assertions (ContainsText, ToolUsed, MaxDuration, etc.) and regression tracking with JSON export
4. **10+ built-in guardrails** — PII redaction, injection detection, keyword blocking, JSON validation, pattern matching — all composable in chains
5. **Full RAG pipeline in the same package** — loaders, chunkers, embedders, vector store, and 3 retriever types without external dependencies
6. **Compiled Go performance** — goroutine-based parallelism for tool execution, no GIL, no async/await complexity

---

## Running the App

```bash
# Set your Mistral API key
export MISTRAL_API_KEY=your_key_here

# Optional: set company database MCP URL
export COMPANY_DB_URL=http://localhost:9477/mcp

# Build and run
cd herdai/examples/webui
go build -o webui . && ./webui

# Open in browser
# http://localhost:8080
```

---

*Generated on March 8, 2026 — HerdAI v1.0*
