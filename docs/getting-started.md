# Getting Started with HerdAI

## Prerequisites

- Go 1.21 or later
- (Optional) OpenAI API key for real LLM calls

## Installation

```bash
# Add to your Go module
go get herdai
```

Or clone and use locally:

```bash
git clone <repo-url> herdai
cd herdai
go mod tidy
```

## Your First Agent (5 minutes)

### Step 1: Create an LLM

For testing (no API key needed):

```go
mock := herdai.NewMockLLM(
    herdai.MockResponse{Content: "Hello! I'm your first agent."},
)
```

For production:

```go
llm := herdai.NewOpenAI(herdai.OpenAIConfig{
    Model: "gpt-4o-mini", // or "gpt-4o", "gpt-3.5-turbo", etc.
})
// Set OPENAI_API_KEY environment variable, or pass APIKey in config
```

### Step 2: Create an Agent

```go
agent := herdai.NewAgent(herdai.AgentConfig{
    ID:   "my-agent",
    Role: "Helpful Assistant",
    Goal: "Answer user questions clearly",
    LLM:  mock, // or llm for real calls
})
```

### Step 3: Run It

```go
result, err := agent.Run(context.Background(), "What is HerdAI?", nil)
if err != nil {
    log.Fatal(err)
}
fmt.Println(result.Content)
```

## Your First Multi-Agent System (10 minutes)

### Step 1: Create Multiple Agents

```go
researcher := herdai.NewAgent(herdai.AgentConfig{
    ID:   "researcher",
    Role: "Researcher",
    Goal: "Gather information",
    LLM:  researcherLLM,
})

writer := herdai.NewAgent(herdai.AgentConfig{
    ID:   "writer",
    Role: "Writer",
    Goal: "Write clear reports from research",
    LLM:  writerLLM,
})
```

### Step 2: Create a Manager

```go
mgr := herdai.NewManager(herdai.ManagerConfig{
    ID:       "pipeline",
    Strategy: herdai.StrategySequential, // researcher then writer
    Agents:   []herdai.Runnable{researcher, writer},
})
```

### Step 3: Run

```go
result, err := mgr.Run(context.Background(), "Research and report on AI trends", nil)
fmt.Println(result.Content) // Writer's polished output
```

## Adding Tools

```go
searchTool := herdai.Tool{
    Name:        "web_search",
    Description: "Search the web for information",
    Parameters: []herdai.ToolParam{
        {Name: "query", Type: "string", Description: "Search query", Required: true},
    },
    Execute: func(ctx context.Context, args map[string]any) (string, error) {
        query := args["query"].(string)
        // Make your HTTP call here
        return "search results...", nil
    },
}

agent := herdai.NewAgent(herdai.AgentConfig{
    ID:    "researcher",
    Role:  "Researcher",
    Goal:  "Find information using web search",
    Tools: []herdai.Tool{searchTool},
    LLM:   llm,
})
```

## Running the Examples

All examples use MockLLM and require no API key:

```bash
cd examples/single_agent && go run main.go     # Single agent with tool
cd examples/two_agent && go run main.go          # Sequential pipeline
cd examples/parallel_agents && go run main.go    # 6 agents in parallel
cd examples/llm_router && go run main.go         # Dynamic routing
cd examples/external_api && go run main.go       # External API calls
```

## Next Steps

- Read [Concepts](concepts.md) to understand Agent, Tool, Manager, Conversation
- Read [How-To Guides](howto.md) for advanced patterns (hierarchy, LLM router, external APIs)
