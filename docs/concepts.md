# Core Concepts

HerdAI has five core concepts: **Agent**, **Tool**, **MCP**, **Manager**, and **Conversation**.

## Agent

An Agent is an autonomous unit that uses an LLM and optional tools to accomplish a goal.

### Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| ID | Yes | — | Unique identifier |
| Role | Yes | — | What the agent does (e.g. "Market Analyst") |
| Goal | Yes | — | What it's trying to achieve |
| Backstory | No | — | Additional context for the LLM |
| Tools | No | — | Capabilities the agent can use |
| LLM | Yes | — | Language model provider |
| MaxToolCalls | No | 10 | Hard limit on tool invocations per run |
| Timeout | No | 2 min | Hard deadline per run |
| Logger | No | slog.Default() | Structured logger |
| MCPServers | No | — | MCP servers to connect to (tools auto-discovered) |
| DisableMCP | No | false | Skip manager-level MCP servers |

### How It Works

1. Agent builds a system prompt from Role, Goal, and Backstory
2. Adds conversation context (last 20 turns) and the current input
3. Calls the LLM
4. If LLM requests tool calls → executes them → feeds results back → calls LLM again
5. When LLM returns a final message (no tool calls) → returns Result
6. Records its output in the Conversation

### Safety

- **Timeout**: Every run has a `context.WithTimeout`. If the LLM or tools take too long, the agent returns an error — never hangs.
- **MaxToolCalls**: If the LLM keeps requesting tools beyond the limit, the agent stops with an error.
- **Context cancellation**: If the parent context is cancelled, the agent stops immediately.

## Tool

A Tool is a capability that an agent can use during execution.

### Fields

| Field | Description |
|-------|-------------|
| Name | Identifier (e.g. "web_search") — sent to LLM |
| Description | What it does — sent to LLM so it knows when to use it |
| Parameters | List of parameters with name, type, description, required |
| Execute | Function that runs when the LLM invokes this tool |

### How It Works

1. Tool definitions (Name, Description, Parameters) are sent to the LLM as available functions
2. The LLM decides when to call a tool based on the description
3. The agent runtime calls `Execute` with the LLM's arguments
4. The result string is fed back to the LLM as a tool response
5. The LLM uses the result to form its answer

### External APIs

Tools are how agents call external APIs:

```go
marketAPI := herdai.Tool{
    Name:        "market_data",
    Description: "Get market size and growth data for an industry",
    Parameters: []herdai.ToolParam{
        {Name: "industry", Type: "string", Description: "Industry name", Required: true},
    },
    Execute: func(ctx context.Context, args map[string]any) (string, error) {
        industry := args["industry"].(string)
        resp, err := http.Get("https://api.example.com/market/" + industry)
        // parse and return
        return parsedResult, nil
    },
}
```

## MCP (Model Context Protocol)

MCP lets agents connect to external tool servers. Instead of defining tools manually, MCP tools are auto-discovered from the server and work identically to regular tools.

### MCPServerConfig

| Field | Required | Description |
|-------|----------|-------------|
| Name | Yes | Human-readable name for logging |
| Command | Yes | Command to start the MCP server (e.g. "npx", "python") |
| Args | No | Command arguments |
| Env | No | Extra environment variables |

### How It Works

1. Agent starts (or `ConnectMCPWithTransport` is called explicitly)
2. MCP client launches the server process (stdio transport)
3. Performs JSON-RPC initialization handshake
4. Calls `tools/list` to discover all tools the server provides
5. Converts each MCP tool into an herdai `Tool` with an auto-generated `Execute` handler
6. The handler calls `tools/call` on the MCP server when invoked
7. From the agent's perspective, MCP tools behave identically to regular tools

### Multiple MCP Servers

Each agent can connect to any number of MCP servers. All tools from all servers are merged into the agent's tool list:

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    MCPServers: []herdai.MCPServerConfig{
        {Name: "filesystem", Command: "npx", Args: []string{"-y", "@modelcontextprotocol/server-filesystem"}},
        {Name: "database",   Command: "db-mcp-server"},
        {Name: "web-search", Command: "python", Args: []string{"search.py"}},
    },
})
```

### Manager-Level MCP Propagation

When a Manager has MCPServers, they are automatically propagated to all agents (unless the agent has `DisableMCP: true`):

```go
mgr := herdai.NewManager(herdai.ManagerConfig{
    MCPServers: mcpServers,
    Agents:     []herdai.Runnable{agent1, agent2, agent3},
})
// agent1, agent2, agent3 all get mcpServers tools automatically
```

### Custom Transports

For non-stdio transports (HTTP, WebSocket) or testing, implement the `MCPTransport` interface:

```go
type MCPTransport interface {
    Start(ctx context.Context) error
    Send(msg json.RawMessage) error
    Receive() (json.RawMessage, error)
    Close() error
}
```

### Cleanup

Always call `Close()` when done to terminate MCP server processes:

```go
defer agent.Close()   // or
defer mgr.Close()     // closes all agents' MCP connections
```

## Manager

A Manager orchestrates a group of agents using a chosen strategy.

### Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| ID | Yes | — | Unique identifier |
| Strategy | Yes | — | How to orchestrate (Sequential, Parallel, RoundRobin, LLMRouter) |
| Agents | Yes | — | List of Runnable (Agent or Manager) |
| MaxTurns | No | 20 | Max iterations for RoundRobin/LLMRouter |
| Timeout | No | 10 min | Hard deadline for the entire run |
| LLM | Only for LLMRouter | — | LLM for routing decisions |
| Logger | No | slog.Default() | Structured logger |
| MCPServers | No | — | MCP servers shared across all agents |

### Strategies

**Sequential**: Runs agents in order. Each agent gets the previous agent's output as input.
- Best for: pipelines (research → write → edit)

**Parallel**: Runs all agents concurrently via goroutines. Merges all results.
- Best for: independent analyses (Porter + SWOT + BMC + PESTEL + VRIO + Blue Ocean)

**RoundRobin**: Cycles through agents until MaxTurns or a `[DONE]`/`[FINISH]` signal.
- Best for: iterative refinement, debates

**LLMRouter**: An LLM decides which agent to call next (or finish). Most flexible.
- Best for: dynamic workflows where the next step depends on results so far

### Hierarchy

Because Manager implements `Runnable`, you can nest Managers:

```
TopManager (Sequential)
├── IndustryTeam (Parallel)
│   ├── PorterAgent
│   └── PESTELAgent
└── InternalTeam (Parallel)
    ├── SWOTAgent
    └── VRIOAgent
```

## Conversation

A thread-safe transcript of all turns across all agents in a run.

### Turn Fields

| Field | Description |
|-------|-------------|
| ID | Auto-generated unique ID |
| AgentID | Which agent produced this turn |
| Role | "user", "assistant", "tool" |
| Content | The message content |
| ToolCalls | Records of tool invocations (name, input, output, duration, error) |
| Timestamp | When this turn was recorded |

### Thread Safety

All Conversation methods are safe for concurrent use (protected by `sync.RWMutex`).
This matters in Parallel strategy where multiple agents write to the same conversation.

## The Runnable Interface

Both Agent and Manager implement `Runnable`:

```go
type Runnable interface {
    GetID() string
    Run(ctx context.Context, input string, conv *Conversation) (*Result, error)
}
```

This is what enables hierarchical composition: anywhere you can use an Agent, you can also use a Manager.
