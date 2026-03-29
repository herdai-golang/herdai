# How-To Guides

## How to Create an Agent

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:        "my-agent",
    Role:      "Market Analyst",
    Goal:      "Analyze market trends and competitors",
    Backstory: "You are a senior analyst with 10 years of experience",
    Tools:     []herdai.Tool{searchTool, apiTool},
    LLM:       herdai.NewOpenAI(herdai.OpenAIConfig{}),
    Timeout:   3 * time.Minute,
})

result, err := agent.Run(ctx, "Analyze the SaaS market", nil)
```

## How to Define a Tool

Every tool needs: Name (for the LLM), Description (so the LLM knows when to use it), and Execute (the actual implementation).

```go
tool := herdai.Tool{
    Name:        "get_weather",
    Description: "Get current weather for a city. Use when the user asks about weather.",
    Parameters: []herdai.ToolParam{
        {Name: "city", Type: "string", Description: "City name", Required: true},
    },
    Execute: func(ctx context.Context, args map[string]any) (string, error) {
        city := args["city"].(string)
        // Make HTTP call to weather API
        return fmt.Sprintf("Weather in %s: 72°F, sunny", city), nil
    },
}
```

## How to Call External APIs from an Agent

The agent decides when to call the API via the tool description. You handle HTTP + parsing in the Execute handler.

```go
apiTool := herdai.Tool{
    Name:        "competitors_api",
    Description: "Look up competitors for a given industry. Returns JSON with company names and market share.",
    Parameters: []herdai.ToolParam{
        {Name: "industry", Type: "string", Description: "Industry to search", Required: true},
    },
    Execute: func(ctx context.Context, args map[string]any) (string, error) {
        industry := args["industry"].(string)

        req, _ := http.NewRequestWithContext(ctx, "GET",
            "https://api.example.com/competitors?industry="+industry, nil)
        resp, err := http.DefaultClient.Do(req)
        if err != nil {
            return "", fmt.Errorf("API call failed: %w", err)
        }
        defer resp.Body.Close()

        body, _ := io.ReadAll(resp.Body)
        return string(body), nil // agent gets the raw JSON and can reason about it
    },
}
```

## How to Build a Multi-Agent System

### Sequential Pipeline (A → B → C)

```go
mgr := herdai.NewManager(herdai.ManagerConfig{
    ID:       "pipeline",
    Strategy: herdai.StrategySequential,
    Agents:   []herdai.Runnable{researcher, analyst, writer},
})
result, _ := mgr.Run(ctx, "Research and write about AI", nil)
```

### Parallel Fan-Out (all agents at once)

```go
mgr := herdai.NewManager(herdai.ManagerConfig{
    ID:       "parallel-analysis",
    Strategy: herdai.StrategyParallel,
    Agents:   []herdai.Runnable{porter, swot, bmc, pestel, vrio, blueOcean},
})
result, _ := mgr.Run(ctx, "Analyze the industry", nil)
// result.Content has merged output from all 6 agents
```

### Dynamic LLM Router (manager decides)

```go
mgr := herdai.NewManager(herdai.ManagerConfig{
    ID:       "smart-manager",
    Strategy: herdai.StrategyLLMRouter,
    Agents:   []herdai.Runnable{analyst, researcher, writer},
    LLM:      routerLLM, // the LLM that makes routing decisions
})
result, _ := mgr.Run(ctx, "Analyze opportunity and write report", nil)
// Manager calls agents in whatever order makes sense, then finishes
```

## How to Build Hierarchical Teams

Create sub-managers and add them as agents to a top-level manager:

```go
// Sub-team 1: Industry analysis (runs Porter + PESTEL in parallel)
industryTeam := herdai.NewManager(herdai.ManagerConfig{
    ID:       "industry",
    Strategy: herdai.StrategyParallel,
    Agents:   []herdai.Runnable{porterAgent, pestelAgent},
})

// Sub-team 2: Internal analysis (runs SWOT + VRIO in parallel)
internalTeam := herdai.NewManager(herdai.ManagerConfig{
    ID:       "internal",
    Strategy: herdai.StrategyParallel,
    Agents:   []herdai.Runnable{swotAgent, vrioAgent},
})

// Top-level: run industry first, then internal (sequential)
top := herdai.NewManager(herdai.ManagerConfig{
    ID:       "top",
    Strategy: herdai.StrategySequential,
    Agents:   []herdai.Runnable{industryTeam, internalTeam},
})
```

## How to Add Agents at Runtime

```go
mgr := herdai.NewManager(herdai.ManagerConfig{
    ID:       "dynamic",
    Strategy: herdai.StrategySequential,
    Agents:   []herdai.Runnable{agent1},
})

// Later, based on some condition:
mgr.AddAgent(agent2)
mgr.AddAgent(agent3)

result, _ := mgr.Run(ctx, "Go", nil)
```

## How to Inspect the Conversation

```go
conv := herdai.NewConversation()
result, _ := mgr.Run(ctx, "Analyze", conv)

for _, turn := range conv.GetTurns() {
    fmt.Printf("[%s] %s (%s): %s\n",
        turn.Timestamp.Format("15:04:05"),
        turn.AgentID,
        turn.Role,
        turn.Content[:min(len(turn.Content), 100)],
    )
    for _, tc := range turn.ToolCalls {
        fmt.Printf("  → Tool: %s (%v)\n", tc.ToolName, tc.Duration)
    }
}
```

## How to Use a Custom LLM

Implement the `LLM` interface:

```go
type LLM interface {
    Chat(ctx context.Context, messages []Message, tools []Tool) (*LLMResponse, error)
}
```

Example for Anthropic:

```go
type AnthropicLLM struct {
    apiKey string
    model  string
}

func (a *AnthropicLLM) Chat(ctx context.Context, messages []Message, tools []Tool) (*LLMResponse, error) {
    // Convert messages/tools to Anthropic API format
    // Make HTTP call to https://api.anthropic.com/v1/messages
    // Parse response and return *LLMResponse
}
```

## How to Set Timeouts and Limits

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    // ...
    Timeout:      3 * time.Minute,  // agent must finish within 3 minutes
    MaxToolCalls: 5,                 // max 5 tool invocations per run
})

mgr := herdai.NewManager(herdai.ManagerConfig{
    // ...
    Timeout:  15 * time.Minute,     // entire pipeline must finish within 15 minutes
    MaxTurns: 10,                   // max 10 iterations for RoundRobin/LLMRouter
})
```

## How to Connect MCP Servers

### Single MCP Server on an Agent

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:   "researcher",
    Role: "Market Researcher",
    LLM:  llm,
    MCPServers: []herdai.MCPServerConfig{
        {
            Name:    "web-search",
            Command: "npx",
            Args:    []string{"-y", "@anthropic/mcp-server-web-search"},
            Env:     map[string]string{"API_KEY": "..."},
        },
    },
})
defer agent.Close()

// MCP tools are auto-discovered on first Run
result, _ := agent.Run(ctx, "Search for AI trends", conv)
```

### Multiple MCP Servers on One Agent

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:   "analyst",
    Role: "Data Analyst",
    LLM:  llm,
    MCPServers: []herdai.MCPServerConfig{
        {Name: "filesystem", Command: "npx", Args: []string{"-y", "@modelcontextprotocol/server-filesystem", "/data"}},
        {Name: "database",   Command: "db-mcp-server", Args: []string{"--host", "localhost"}},
        {Name: "web",        Command: "python", Args: []string{"web_search_mcp.py"}},
    },
    // Agent can also have regular tools alongside MCP tools
    Tools: []herdai.Tool{myCustomTool},
})
defer agent.Close()
```

### Share MCP Servers Across All Agents via Manager

```go
sharedMCP := []herdai.MCPServerConfig{
    {Name: "company-db", Command: "db-server"},
    {Name: "web-search", Command: "search-server"},
}

mgr := herdai.NewManager(herdai.ManagerConfig{
    ID:         "analysis-team",
    Strategy:   herdai.StrategyParallel,
    Agents:     []herdai.Runnable{agent1, agent2, agent3},
    MCPServers: sharedMCP, // all 3 agents get these MCP tools
})
defer mgr.Close()
```

### Opt an Agent Out of Manager MCP

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:         "simple-worker",
    Role:       "Text Writer",
    LLM:        llm,
    DisableMCP: true, // won't receive manager-level MCP servers
})
```

### Use a Custom Transport (Testing or Non-Stdio)

```go
// Create a mock MCP transport for testing
mock := herdai.NewMockMCPTransport(
    herdai.MCPToolDef{
        Name:        "search",
        Description: "Search for info",
        InputSchema: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "query": map[string]any{"type": "string"},
            },
        },
        Handler: func(args map[string]any) string {
            return "mock search result"
        },
    },
)

agent.ConnectMCPWithTransport(ctx, "test-server", mock)
defer agent.Close()
```

### Standalone MCP Connection (Without Agent)

```go
tools, clients, err := herdai.ConnectMCP(ctx, []herdai.MCPServerConfig{
    {Name: "my-server", Command: "my-mcp-server"},
}, logger)
defer func() {
    for _, c := range clients {
        c.Close()
    }
}()

// Use discovered tools with any agent
agent := herdai.NewAgent(herdai.AgentConfig{
    Tools: append(myTools, tools...),
    // ...
})
```
