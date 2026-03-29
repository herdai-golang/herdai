package herdai

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// --- MockMCPTransport Tests ---

func TestMockMCPTransport_Initialize(t *testing.T) {
	mock := NewMockMCPTransport()
	client := NewMCPClient("test-server", mock, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	defer client.Close()

	if !client.connected {
		t.Fatal("expected client to be connected")
	}
}

func TestMockMCPTransport_ToolDiscovery(t *testing.T) {
	mock := NewMockMCPTransport(
		MCPToolDef{
			Name:        "web_search",
			Description: "Search the web for information",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Search query",
					},
				},
				"required": []any{"query"},
			},
		},
		MCPToolDef{
			Name:        "read_file",
			Description: "Read contents of a file",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path": map[string]any{
						"type":        "string",
						"description": "File path",
					},
				},
				"required": []any{"path"},
			},
		},
	)

	client := NewMCPClient("test-server", mock, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	defer client.Close()

	tools := client.Tools()
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}

	if tools[0].Name != "web_search" {
		t.Errorf("expected tool name 'web_search', got '%s'", tools[0].Name)
	}
	if tools[1].Name != "read_file" {
		t.Errorf("expected tool name 'read_file', got '%s'", tools[1].Name)
	}

	if tools[0].Description != "Search the web for information" {
		t.Errorf("unexpected description: %s", tools[0].Description)
	}

	// Check parameters were extracted
	hasQuery := false
	for _, p := range tools[0].Parameters {
		if p.Name == "query" && p.Required {
			hasQuery = true
		}
	}
	if !hasQuery {
		t.Error("expected 'query' required parameter on web_search tool")
	}
}

func TestMockMCPTransport_ToolInvocation(t *testing.T) {
	mock := NewMockMCPTransport(
		MCPToolDef{
			Name:        "calculator",
			Description: "Perform calculations",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"expression": map[string]any{
						"type":        "string",
						"description": "Math expression",
					},
				},
			},
			Handler: func(args map[string]any) string {
				expr, _ := args["expression"].(string)
				return fmt.Sprintf("Result of %s = 42", expr)
			},
		},
	)

	client := NewMCPClient("calc-server", mock, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	defer client.Close()

	tools := client.Tools()
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	// Execute the discovered tool
	result, err := tools[0].Execute(ctx, map[string]any{"expression": "2+2"})
	if err != nil {
		t.Fatalf("tool execute failed: %v", err)
	}

	if !strings.Contains(result, "42") {
		t.Errorf("expected result to contain '42', got: %s", result)
	}
}

// --- Agent + MCP Integration Tests ---

func TestAgent_MCPToolDiscovery(t *testing.T) {
	mock := NewMockMCPTransport(
		MCPToolDef{
			Name:        "mcp_search",
			Description: "MCP search tool",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
			Handler: func(args map[string]any) string {
				return "mcp search result"
			},
		},
	)

	mockLLM := NewMockLLM()
	mockLLM.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{{
			ID: "tc1", Function: "mcp_search", Args: map[string]any{},
		}},
	})
	mockLLM.PushResponse(LLMResponse{Content: "Found via MCP"})

	agent := NewAgent(AgentConfig{
		ID:   "mcp-agent",
		Role: "Researcher",
		Goal: "Search using MCP tools",
		LLM:  mockLLM,
	})

	// Manually connect MCP via transport
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := agent.ConnectMCPWithTransport(ctx, "test-server", mock); err != nil {
		t.Fatalf("ConnectMCPWithTransport failed: %v", err)
	}
	defer agent.Close()

	conv := NewConversation()
	result, err := agent.Run(ctx, "search for something", conv)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if result.Content != "Found via MCP" {
		t.Errorf("unexpected content: %s", result.Content)
	}

	// Verify the MCP tool was called
	turns := conv.GetTurns()
	foundToolCall := false
	for _, turn := range turns {
		for _, tc := range turn.ToolCalls {
			if tc.ToolName == "mcp_search" {
				foundToolCall = true
			}
		}
	}
	if !foundToolCall {
		t.Error("expected MCP tool call to be recorded in conversation")
	}
}

func TestAgent_MCPWithRegularTools(t *testing.T) {
	mock := NewMockMCPTransport(
		MCPToolDef{
			Name:        "mcp_tool",
			Description: "Tool from MCP",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
			Handler: func(args map[string]any) string {
				return "mcp result"
			},
		},
	)

	regularTool := Tool{
		Name:        "regular_tool",
		Description: "A regular tool",
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			return "regular result", nil
		},
	}

	mockLLM := NewMockLLM()
	// Call regular tool first
	mockLLM.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{{
			ID: "tc1", Function: "regular_tool", Args: map[string]any{},
		}},
	})
	// Then call MCP tool
	mockLLM.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{{
			ID: "tc2", Function: "mcp_tool", Args: map[string]any{},
		}},
	})
	mockLLM.PushResponse(LLMResponse{Content: "Combined results"})

	agent := NewAgent(AgentConfig{
		ID:    "combined-agent",
		Role:  "Analyst",
		Goal:  "Use both regular and MCP tools",
		Tools: []Tool{regularTool},
		LLM:   mockLLM,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := agent.ConnectMCPWithTransport(ctx, "test-server", mock); err != nil {
		t.Fatalf("ConnectMCPWithTransport failed: %v", err)
	}
	defer agent.Close()

	conv := NewConversation()
	result, err := agent.Run(ctx, "analyze data", conv)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if result.Content != "Combined results" {
		t.Errorf("unexpected content: %s", result.Content)
	}

	// Both tools should have been called
	turns := conv.GetTurns()
	toolNames := make(map[string]bool)
	for _, turn := range turns {
		for _, tc := range turn.ToolCalls {
			toolNames[tc.ToolName] = true
		}
	}

	if !toolNames["regular_tool"] {
		t.Error("expected regular_tool to be called")
	}
	if !toolNames["mcp_tool"] {
		t.Error("expected mcp_tool to be called")
	}
}

func TestAgent_DisableMCP(t *testing.T) {
	agent := NewAgent(AgentConfig{
		ID:         "no-mcp-agent",
		Role:       "Worker",
		Goal:       "Work without MCP",
		LLM:        NewMockLLM(),
		DisableMCP: true,
	})

	if !agent.disableMCP {
		t.Error("expected DisableMCP to be true")
	}
}

func TestAgent_MultipleMCPServers(t *testing.T) {
	mock1 := NewMockMCPTransport(
		MCPToolDef{
			Name:        "server1_tool",
			Description: "Tool from server 1",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
			Handler:     func(args map[string]any) string { return "result from server 1" },
		},
	)

	mock2 := NewMockMCPTransport(
		MCPToolDef{
			Name:        "server2_tool",
			Description: "Tool from server 2",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
			Handler:     func(args map[string]any) string { return "result from server 2" },
		},
	)

	mockLLM := NewMockLLM()
	mockLLM.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{{
			ID: "tc1", Function: "server1_tool", Args: map[string]any{},
		}},
	})
	mockLLM.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{{
			ID: "tc2", Function: "server2_tool", Args: map[string]any{},
		}},
	})
	mockLLM.PushResponse(LLMResponse{Content: "Combined from both servers"})

	agent := NewAgent(AgentConfig{
		ID:   "multi-mcp-agent",
		Role: "Multi-Server Agent",
		Goal: "Use tools from multiple MCP servers",
		LLM:  mockLLM,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := agent.ConnectMCPWithTransport(ctx, "server-1", mock1); err != nil {
		t.Fatalf("Connect server 1 failed: %v", err)
	}
	if err := agent.ConnectMCPWithTransport(ctx, "server-2", mock2); err != nil {
		t.Fatalf("Connect server 2 failed: %v", err)
	}
	defer agent.Close()

	conv := NewConversation()
	result, err := agent.Run(ctx, "use both servers", conv)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if result.Content != "Combined from both servers" {
		t.Errorf("unexpected content: %s", result.Content)
	}
}

// --- Manager + MCP Propagation Tests ---

func TestManager_MCPPropagation(t *testing.T) {
	mcpServers := []MCPServerConfig{
		{Name: "shared-server", Command: "echo", Args: []string{"test"}},
	}

	mockLLM := NewMockLLM()
	mockLLM.PushResponse(LLMResponse{Content: "agent1 done"})
	agent1 := NewAgent(AgentConfig{
		ID:   "agent-1",
		Role: "Worker 1",
		Goal: "Do work",
		LLM:  mockLLM,
	})

	mockLLM2 := NewMockLLM()
	mockLLM2.PushResponse(LLMResponse{Content: "agent2 done"})
	agent2 := NewAgent(AgentConfig{
		ID:         "agent-2",
		Role:       "Worker 2",
		Goal:       "Do work",
		LLM:        mockLLM2,
		DisableMCP: true,
	})

	_ = NewManager(ManagerConfig{
		ID:         "manager-1",
		Strategy:   StrategySequential,
		Agents:     []Runnable{agent1, agent2},
		MCPServers: mcpServers,
	})

	// agent1 should have MCP servers propagated
	if len(agent1.mcpServers) != 1 {
		t.Errorf("expected agent1 to have 1 MCP server, got %d", len(agent1.mcpServers))
	}
	if agent1.mcpServers[0].Name != "shared-server" {
		t.Errorf("expected 'shared-server', got '%s'", agent1.mcpServers[0].Name)
	}

	// agent2 opted out, should have no MCP servers
	if len(agent2.mcpServers) != 0 {
		t.Errorf("expected agent2 to have 0 MCP servers (DisableMCP=true), got %d", len(agent2.mcpServers))
	}
}

func TestManager_AddAgent_MCPPropagation(t *testing.T) {
	mcpServers := []MCPServerConfig{
		{Name: "shared-server", Command: "echo"},
	}

	mgr := NewManager(ManagerConfig{
		ID:         "manager",
		Strategy:   StrategySequential,
		Agents:     []Runnable{},
		MCPServers: mcpServers,
	})

	mockLLM := NewMockLLM()
	mockLLM.PushResponse(LLMResponse{Content: "done"})
	newAgent := NewAgent(AgentConfig{
		ID:   "new-agent",
		Role: "New Worker",
		Goal: "Do work",
		LLM:  mockLLM,
	})

	mgr.AddAgent(newAgent)

	if len(newAgent.mcpServers) != 1 {
		t.Errorf("expected dynamically added agent to have MCP server, got %d", len(newAgent.mcpServers))
	}
}

func TestManager_Close(t *testing.T) {
	mock := NewMockMCPTransport()

	mockLLM := NewMockLLM()
	agent := NewAgent(AgentConfig{
		ID:   "closeable-agent",
		Role: "Worker",
		Goal: "Work",
		LLM:  mockLLM,
	})

	ctx := context.Background()
	if err := agent.ConnectMCPWithTransport(ctx, "test", mock); err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	mgr := NewManager(ManagerConfig{
		ID:       "manager",
		Strategy: StrategySequential,
		Agents:   []Runnable{agent},
	})

	if err := mgr.Close(); err != nil {
		t.Fatalf("Manager.Close failed: %v", err)
	}

	if agent.mcpReady {
		t.Error("expected mcpReady to be false after Close")
	}
}

// --- ConnectMCPWithTransport helper ---

func TestConnectMCPWithTransport(t *testing.T) {
	mock := NewMockMCPTransport(
		MCPToolDef{
			Name:        "helper_tool",
			Description: "A helper tool",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	client, tools, err := ConnectMCPWithTransport(ctx, "helper", mock, nil)
	if err != nil {
		t.Fatalf("ConnectMCPWithTransport failed: %v", err)
	}
	defer client.Close()

	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Name != "helper_tool" {
		t.Errorf("expected 'helper_tool', got '%s'", tools[0].Name)
	}
}

// --- Concurrent MCP tool access ---

func TestMCPClient_ConcurrentToolCalls(t *testing.T) {
	callCount := 0
	var mu sync.Mutex

	mock := NewMockMCPTransport(
		MCPToolDef{
			Name:        "concurrent_tool",
			Description: "Tool for concurrent testing",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
			Handler: func(args map[string]any) string {
				mu.Lock()
				callCount++
				mu.Unlock()
				return "concurrent result"
			},
		},
	)

	client := NewMCPClient("concurrent-server", mock, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	defer client.Close()

	tools := client.Tools()
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	// Call the tool from multiple goroutines
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := tools[0].Execute(ctx, map[string]any{})
			if err != nil {
				t.Errorf("concurrent call failed: %v", err)
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	if callCount != 5 {
		t.Errorf("expected 5 calls, got %d", callCount)
	}
	mu.Unlock()
}

func TestMCPClient_Close_Idempotent(t *testing.T) {
	mock := NewMockMCPTransport()
	client := NewMCPClient("test", mock, nil)

	ctx := context.Background()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	// Close multiple times should not panic
	if err := client.Close(); err != nil {
		t.Fatalf("first Close failed: %v", err)
	}
	if err := client.Close(); err != nil {
		t.Fatalf("second Close failed: %v", err)
	}
}

func TestMCPClient_AlreadyConnected(t *testing.T) {
	mock := NewMockMCPTransport()
	client := NewMCPClient("test", mock, nil)

	ctx := context.Background()
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("first Connect failed: %v", err)
	}
	defer client.Close()

	// Second connect should be a no-op
	if err := client.Connect(ctx); err != nil {
		t.Fatalf("second Connect failed: %v", err)
	}
}

func TestExtractParams(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "Search query",
			},
			"limit": map[string]any{
				"type":        "number",
				"description": "Max results",
			},
		},
		"required": []any{"query"},
	}

	params := extractParams(schema)
	if len(params) != 2 {
		t.Fatalf("expected 2 params, got %d", len(params))
	}

	paramMap := make(map[string]ToolParam)
	for _, p := range params {
		paramMap[p.Name] = p
	}

	q, ok := paramMap["query"]
	if !ok {
		t.Fatal("expected 'query' param")
	}
	if q.Type != "string" {
		t.Errorf("expected type 'string', got '%s'", q.Type)
	}
	if !q.Required {
		t.Error("expected 'query' to be required")
	}

	l, ok := paramMap["limit"]
	if !ok {
		t.Fatal("expected 'limit' param")
	}
	if l.Type != "number" {
		t.Errorf("expected type 'number', got '%s'", l.Type)
	}
	if l.Required {
		t.Error("expected 'limit' to not be required")
	}
}

func TestExtractParams_NilSchema(t *testing.T) {
	params := extractParams(nil)
	if params != nil {
		t.Errorf("expected nil params for nil schema, got %v", params)
	}
}

// --- Full end-to-end: Agent with MCP tools runs via Manager ---

func TestManager_AgentsWithMCP_EndToEnd(t *testing.T) {
	mock1 := NewMockMCPTransport(
		MCPToolDef{
			Name:        "analyze",
			Description: "Analyze data",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
			Handler:     func(args map[string]any) string { return "analysis complete" },
		},
	)

	mock2 := NewMockMCPTransport(
		MCPToolDef{
			Name:        "summarize",
			Description: "Summarize text",
			InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
			Handler:     func(args map[string]any) string { return "summary ready" },
		},
	)

	llm1 := NewMockLLM()
	llm1.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{{ID: "tc1", Function: "analyze", Args: map[string]any{}}},
	})
	llm1.PushResponse(LLMResponse{Content: "Agent 1: analysis complete"})

	llm2 := NewMockLLM()
	llm2.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{{ID: "tc2", Function: "summarize", Args: map[string]any{}}},
	})
	llm2.PushResponse(LLMResponse{Content: "Agent 2: summary ready"})

	agent1 := NewAgent(AgentConfig{
		ID:   "analyst",
		Role: "Data Analyst",
		Goal: "Analyze data using MCP tools",
		LLM:  llm1,
	})

	agent2 := NewAgent(AgentConfig{
		ID:   "summarizer",
		Role: "Summarizer",
		Goal: "Summarize analysis using MCP tools",
		LLM:  llm2,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Connect each agent to its own MCP server
	if err := agent1.ConnectMCPWithTransport(ctx, "analysis-server", mock1); err != nil {
		t.Fatalf("agent1 MCP connect failed: %v", err)
	}
	if err := agent2.ConnectMCPWithTransport(ctx, "summary-server", mock2); err != nil {
		t.Fatalf("agent2 MCP connect failed: %v", err)
	}

	mgr := NewManager(ManagerConfig{
		ID:       "mcp-manager",
		Strategy: StrategySequential,
		Agents:   []Runnable{agent1, agent2},
	})
	defer mgr.Close()

	conv := NewConversation()
	result, err := mgr.Run(ctx, "Process this data", conv)
	if err != nil {
		t.Fatalf("Manager run failed: %v", err)
	}

	// Sequential manager returns the last agent's result
	if result.Content != "Agent 2: summary ready" {
		t.Errorf("unexpected result: %s", result.Content)
	}

	// Verify both agents' turns were recorded in conversation
	turns := conv.GetTurns()
	foundAgent1 := false
	foundAgent2 := false
	for _, turn := range turns {
		if turn.AgentID == "analyst" {
			foundAgent1 = true
		}
		if turn.AgentID == "summarizer" {
			foundAgent2 = true
		}
	}
	if !foundAgent1 {
		t.Error("expected agent1 (analyst) turns in conversation")
	}
	if !foundAgent2 {
		t.Error("expected agent2 (summarizer) turns in conversation")
	}
}
