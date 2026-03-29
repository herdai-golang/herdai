# Agents vs Tools: When to Use What

## The Core Principle

**A Tool is a capability. An Agent is a decision-maker.**

A tool does one thing when asked. An agent decides what to do, when to do it, and how to combine results. If you're unsure, start with a tool. Promote to an agent only when the task requires reasoning.

---

## Decision Framework

```
Does this task require the LLM to THINK, PLAN, or make DECISIONS?
│
├─ NO  → Make it a TOOL
│        Examples: search the web, query a database, run a calculation,
│        fetch a URL, call an API, read a file, execute code
│
└─ YES → Does it need its OWN reasoning loop (multiple steps, self-correction)?
         │
         ├─ NO  → Still a TOOL (the calling agent will do the reasoning)
         │        Examples: summarize text, translate, extract entities,
         │        format data — these are single LLM calls, not loops
         │
         └─ YES → Make it an AGENT
                  Examples: research a topic (search → read → search again → synthesize),
                  write code (plan → write → test → fix), analyze a company
                  (gather data → apply framework → iterate)
```

---

## Tools: The Default Choice

A tool is a function the agent can call. It takes input, does work, returns output. No reasoning loop.

### When to create a tool

| Situation | Example |
|---|---|
| Accessing external data | `web_search`, `query_database`, `fetch_url` |
| Performing a computation | `calculate_roi`, `convert_currency` |
| Calling an API | `send_email`, `create_ticket`, `post_to_slack` |
| Transforming data | `parse_csv`, `extract_json`, `resize_image` |
| Wrapping an agent as a capability | `analyze_porter` (runs a sub-agent internally) |

### Tool characteristics

- **Deterministic or near-deterministic** — same input, same output
- **No decision-making** — it doesn't choose what to do next
- **Stateless** — it doesn't remember previous calls
- **Fast** — ideally completes in seconds
- **Self-contained** — doesn't need to call other tools

### Code pattern

```go
tool := herdai.Tool{
    Name:        "web_search",
    Description: "Search the web for current information",
    Parameters: []herdai.ToolParam{
        {Name: "query", Type: "string", Description: "Search query", Required: true},
    },
    Execute: func(ctx context.Context, args map[string]any) (string, error) {
        query := args["query"].(string)
        return searchWeb(query)
    },
}
```

---

## Agents: When Reasoning is Required

An agent has an LLM at its core. It receives a goal, reasons about how to achieve it, calls tools, observes results, and decides what to do next. This is the ReAct loop.

### When to create an agent

| Situation | Example |
|---|---|
| Multi-step research | "Find competitors, analyze their strengths, compare pricing" |
| Tasks requiring judgment | "Review this code for security issues" |
| Open-ended exploration | "Investigate why our conversion rate dropped" |
| Self-correcting workflows | "Write tests, run them, fix failures" |
| Different persona/expertise | "You are a patent attorney — evaluate this IP" |

### Agent characteristics

- **Has a reasoning loop** — think → act → observe → repeat
- **Makes decisions** — chooses which tools to call and when to stop
- **Has a persona** — role, goal, backstory shape its behavior
- **Can fail and recover** — tries alternative approaches
- **More expensive** — each step is an LLM call

### Code pattern

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:    "researcher",
    Role:  "Market Research Analyst",
    Goal:  "Find and analyze the top 5 competitors in this space",
    LLM:   llm,
    Tools: []herdai.Tool{webSearchTool, fetchURLTool},
})
```

---

## The Hybrid: Agent-as-Tool

The most powerful pattern is wrapping an agent inside a tool. The outer agent (orchestrator) decides WHEN to call the inner agent, but the inner agent handles the HOW.

This is what the webui does: the CSO agent has `analyze_porter` as a tool. The tool internally runs a Porter's Five Forces agent with web search. The CSO decides when to invoke it.

### When to use agent-as-tool

- The outer agent needs multiple specialized capabilities
- Each capability requires its own reasoning (not just a function call)
- You want the orchestrator to decide which analyses to run
- You want caching — run once, reuse, refresh when context changes

### Code pattern

```go
porterTool := herdai.Tool{
    Name:        "analyze_porter",
    Description: "Run a Porter's Five Forces analysis on the business",
    Parameters:  []herdai.ToolParam{{Name: "context", Type: "string", Required: true}},
    Execute: func(ctx context.Context, args map[string]any) (string, error) {
        // This tool runs a FULL AGENT internally
        agent := herdai.NewAgent(herdai.AgentConfig{
            ID:     "porter",
            Role:   "Porter's Five Forces Analyst",
            Goal:   "Analyze competitive forces...",
            LLM:    llm,
            Tools:  []herdai.Tool{webSearchTool},
        })
        defer agent.Close()
        result, err := agent.Run(ctx, args["context"].(string), herdai.NewConversation())
        if err != nil {
            return "", err
        }
        return result.Content, nil
    },
}

// The CSO gets this as one of its tools
cso := herdai.NewAgent(herdai.AgentConfig{
    ID:    "cso",
    Role:  "Chief Strategy Officer",
    Tools: []herdai.Tool{porterTool, swotTool, pestelTool, webSearchTool},
    LLM:   llm,
})
```

### Caching tool results

HerdAI’s **ToolCache** lets you run expensive tools once per “context” and reuse results until the context changes meaningfully. That fits agent-as-tool especially well: e.g. “Porter analysis for this idea” is cached until the idea or its dependencies change.

**Behavior:**

- **Context key:** Entries are keyed by tool name + context string (e.g. the intake or prompt text). Same context → cache hit; no second execution.
- **NewWordThreshold (default 3):** If N or more *meaningful* words change (added, removed, or replaced) between the cached context and the new request, the cache is invalidated. Small rephrases (“the idea” vs “our idea”) still hit the cache.
- **Field-aware invalidation (ToolDeps):** You can map each tool to the context *fields* it depends on (e.g. `financial_analysis` → `idea`, `industry`, `revenue`, `customer`). When you update context with `SetContextFields(fields)`, only tools whose dependent fields changed are invalidated; others stay cached. So changing “customer” from “architects” to “hospitals” can invalidate GTM and competitor intel but leave strategic_analysis cached.
- **MaxAge / MaxEntries:** Optional TTL and size cap; entries can expire or be evicted when the cache is full.
- **Wrap:** `cache.Wrap(toolName, handler)` returns a tool that checks the cache before calling the handler; you can pass `refresh: true` in args to force a re-run.
- **Tracing:** Cache hits are recorded on spans (`cached: true`) so you can see reuse in traces.

Use a shared `ToolCache` (and optional `ToolDeps`) on the agent so that repeated questions or follow-ups reuse results and only re-run when the user pivots.

---

## Anti-Patterns: What NOT to Do

### 1. Agent when a tool would suffice

**Bad:** Creating an agent just to call one API.
```go
// DON'T: This is an agent that just calls an API — no reasoning needed
weatherAgent := herdai.NewAgent(herdai.AgentConfig{
    ID:   "weather",
    Role: "Weather Reporter",
    Goal: "Get the weather for a city",
    LLM:  llm,
})
```

**Good:** Make it a tool.
```go
// DO: Simple function, no LLM needed
weatherTool := herdai.Tool{
    Name: "get_weather",
    Execute: func(ctx context.Context, args map[string]any) (string, error) {
        return fetchWeatherAPI(args["city"].(string))
    },
}
```

### 2. Router/classifier before every response

**Bad:** Using an LLM call to classify every user message before acting.
```go
// DON'T: Extra LLM call that adds latency with no benefit
classifyResp := llm.Chat(ctx, "Is this a question or new info?")
if classifyResp == "question" {
    handleQuestion()
} else {
    handleUpdate()
}
```

**Good:** Let the agent decide naturally through tool selection.
```go
// DO: The agent has tools and decides when to use them
agent := herdai.NewAgent(herdai.AgentConfig{
    Tools: []herdai.Tool{searchTool, analysisTool, updateTool},
})
// The LLM naturally picks the right tool based on the conversation
```

### 3. Multi-agent pipeline for a conversational UI

**Bad:** Agent → Router → Specialist Agent → Synthesizer Agent (4 LLM calls).

**Good:** One agent with specialist tools (1 LLM call + tool calls when needed).

### 4. Hardcoding which agents/tools to call

**Bad:** `if userAsksAboutCompetitors { callYCScout() }`

**Good:** Give the agent tools with clear descriptions. The LLM picks the right one.

---

## The Anthropic Rule

From Anthropic's "Building Effective Agents":

> *"Start with the simplest solution possible, and only increase complexity when needed."*

The progression:

1. **Single LLM call** — just a prompt, no tools, no agents
2. **LLM + tools** — one model with functions it can call
3. **Agent** — LLM with tools in a reasoning loop
4. **Agent with agent-tools** — orchestrator agent delegating to specialist agents
5. **Multi-agent manager** — multiple agents coordinated by a manager

Most applications need level 2 or 3. Move to 4-5 only when you have genuinely different specializations that require separate reasoning loops.

---

## Quick Reference

| I need to... | Use |
|---|---|
| Call an API | Tool |
| Search the web | Tool |
| Run a database query | Tool |
| Parse/transform data | Tool |
| Summarize a document | Tool (single LLM call wrapped as tool) |
| Research a topic (multi-step) | Agent |
| Analyze with a specific framework | Agent (wrapped as tool) |
| Write and iterate on code | Agent |
| Have a conversation with a user | Agent (the outer orchestrator) |
| Coordinate multiple analyses | Agent with agent-tools |
| Run a pipeline with human checkpoints | Manager with agents |

---

## How This Maps to HerdAI

| HerdAI Concept | Role |
|---|---|
| `Tool` | A capability — function the agent can call |
| `Agent` | A decision-maker — LLM with tools in a ReAct loop |
| `Manager` | An orchestrator — coordinates multiple agents with a strategy |
| `Agent-as-Tool` | A specialist agent wrapped as a tool for another agent |
| `MCP Server` | External tools discovered dynamically via protocol |

The webui example demonstrates the ideal pattern: **one conversational agent (CSO) with 7 framework agent-tools + MCP web search**. The CSO decides what to analyze, when to refresh, and how to synthesize — just like ChatGPT decides when to search the web.

---

## Parallel Tool Execution

When an LLM returns multiple tool calls in a single response, they are **independent by definition** — if one tool's output were needed by another, the LLM would return them in separate rounds. HerdAI exploits this by executing all tool calls within a batch **concurrently** using goroutines.

This is enabled by default. To disable:

```go
seqOnly := false
agent := herdai.NewAgent(herdai.AgentConfig{
    ParallelToolCalls: &seqOnly,
})
```

### How it works

```
LLM response: [analyze_porter, analyze_swot, analyze_pestel, analyze_tam]

Sequential (old):   porter → swot → pestel → tam   = ~120s total
Parallel (default): porter ┐
                    swot   ├─ all at once            = ~30s total
                    pestel ┤
                    tam    ┘
```

### Guarantees

1. **Order preserved** — tool results are fed back to the LLM in the same order the calls were requested, regardless of which tool finishes first.
2. **HITL still works** — human approval checks happen sequentially *before* execution. Only approved tools run in parallel.
3. **Error isolation** — if one tool fails, the others still complete. The LLM receives all results (successes and errors) and decides how to proceed.
4. **Single tool = no overhead** — parallelism only activates when there are 2+ pending tool calls.
