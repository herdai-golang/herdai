package herdai

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestIntegrationFullPipeline(t *testing.T) {
	searchTool := Tool{
		Name:        "web_search",
		Description: "Search the web for information",
		Parameters:  []ToolParam{{Name: "query", Type: "string", Description: "search query", Required: true}},
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			query, _ := args["query"].(string)
			return fmt.Sprintf("Search results for '%s': Found 3 competitors in the market", query), nil
		},
	}

	porterMock := NewMockLLM(
		MockResponse{
			ToolCalls: []ToolCall{{ID: "c1", Function: "web_search", Args: map[string]any{"query": "competitors"}}},
		},
		MockResponse{Content: "Porter's Five Forces: High rivalry, moderate barriers to entry"},
	)
	porter := NewAgent(AgentConfig{
		ID:    "porter",
		Role:  "Porter's Five Forces Analyst",
		Goal:  "Analyze industry attractiveness using Porter's Five Forces",
		Tools: []Tool{searchTool},
		LLM:   porterMock,
	})

	swotMock := NewMockLLM(
		MockResponse{Content: "SWOT: Strengths - strong team; Weaknesses - limited funding"},
	)
	swot := NewAgent(AgentConfig{
		ID:   "swot",
		Role: "SWOT Analyst",
		Goal: "Identify strengths, weaknesses, opportunities, and threats",
		LLM:  swotMock,
	})

	bmcMock := NewMockLLM(
		MockResponse{Content: "BMC: Value prop is AI-powered analytics; Revenue via SaaS subscription"},
	)
	bmc := NewAgent(AgentConfig{
		ID:   "bmc",
		Role: "Business Model Canvas Analyst",
		Goal: "Map out the full business model",
		LLM:  bmcMock,
	})

	mgr := NewManager(ManagerConfig{
		ID:       "pipeline-manager",
		Strategy: StrategyParallel,
		Agents:   []Runnable{porter, swot, bmc},
		Timeout:  30 * time.Second,
	})

	conv := NewConversation()
	result, err := mgr.Run(context.Background(), "Analyze the industry for a new SaaS startup", conv)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "porter") {
		t.Fatal("missing porter results")
	}
	if !strings.Contains(result.Content, "swot") {
		t.Fatal("missing swot results")
	}
	if !strings.Contains(result.Content, "bmc") {
		t.Fatal("missing bmc results")
	}

	turns := conv.GetTurns()
	if len(turns) < 4 {
		t.Fatalf("expected at least 4 turns (1 input + 3 agents), got %d", len(turns))
	}

	t.Logf("Pipeline completed with %d turns", len(turns))
	t.Logf("Result length: %d chars", len(result.Content))
}

func TestIntegrationHierarchicalPipeline(t *testing.T) {
	a1, _ := newTestAgent("porter", "Porter Analyst", "Five Forces", "Forces analyzed")
	a2, _ := newTestAgent("pestel", "PESTEL Analyst", "PESTEL", "PESTEL done")

	industryTeam := NewManager(ManagerConfig{
		ID:       "industry-team",
		Strategy: StrategyParallel,
		Agents:   []Runnable{a1, a2},
	})

	a3, _ := newTestAgent("swot", "SWOT Analyst", "SWOT", "SWOT complete")
	a4, _ := newTestAgent("vrio", "VRIO Analyst", "VRIO", "VRIO complete")

	internalTeam := NewManager(ManagerConfig{
		ID:       "internal-team",
		Strategy: StrategyParallel,
		Agents:   []Runnable{a3, a4},
	})

	topManager := NewManager(ManagerConfig{
		ID:       "top-manager",
		Strategy: StrategySequential,
		Agents:   []Runnable{industryTeam, internalTeam},
	})

	conv := NewConversation()
	result, err := topManager.Run(context.Background(), "Full analysis", conv)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "swot") && !strings.Contains(result.Content, "SWOT") {
		t.Fatal("expected internal team results")
	}

	t.Logf("Hierarchical pipeline completed: %d turns", conv.Len())
}

func TestIntegrationLLMRouterWithTools(t *testing.T) {
	apiTool := Tool{
		Name:        "market_api",
		Description: "Call market data API",
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			return `{"market_size": "$5B", "growth": "12% CAGR"}`, nil
		},
	}

	analystMock := NewMockLLM(
		MockResponse{
			ToolCalls: []ToolCall{{ID: "c1", Function: "market_api", Args: map[string]any{}}},
		},
		MockResponse{Content: "Market is $5B with 12% CAGR growth"},
	)
	analyst := NewAgent(AgentConfig{
		ID:    "analyst",
		Role:  "Market Analyst",
		Goal:  "Analyze market size",
		Tools: []Tool{apiTool},
		LLM:   analystMock,
	})

	writerMock := NewMockLLM(
		MockResponse{Content: "Report: The market presents significant opportunity at $5B"},
	)
	writer := NewAgent(AgentConfig{
		ID:   "writer",
		Role: "Report Writer",
		Goal: "Write the final report",
		LLM:  writerMock,
	})

	routerMock := NewMockLLM(
		MockResponse{Content: `{"agent_id": "analyst", "instruction": "Analyze the market size and growth"}`},
		MockResponse{Content: `{"agent_id": "writer", "instruction": "Write a summary report based on the analysis"}`},
		MockResponse{Content: `{"agent_id": "FINISH", "instruction": "Analysis and report complete. Market is $5B with 12% growth."}`},
	)

	mgr := NewManager(ManagerConfig{
		ID:       "smart-manager",
		Strategy: StrategyLLMRouter,
		Agents:   []Runnable{analyst, writer},
		LLM:      routerMock,
	})

	conv := NewConversation()
	result, err := mgr.Run(context.Background(), "Analyze the market and write a report", conv)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "$5B") {
		t.Fatalf("expected market data in result, got: %s", result.Content)
	}

	t.Logf("LLM Router completed: %d turns, result: %s", conv.Len(), truncate(result.Content, 100))
}

func TestIntegrationNoHanging(t *testing.T) {
	done := make(chan struct{})

	go func() {
		defer close(done)

		agents := make([]Runnable, 10)
		for i := 0; i < 10; i++ {
			mock := NewMockLLM(MockResponse{Content: fmt.Sprintf("agent-%d done", i)})
			agents[i] = NewAgent(AgentConfig{
				ID:   fmt.Sprintf("agent-%d", i),
				Role: "Worker",
				Goal: "Do work",
				LLM:  mock,
			})
		}

		mgr := NewManager(ManagerConfig{
			ID:       "no-hang-manager",
			Strategy: StrategyParallel,
			Agents:   agents,
			Timeout:  5 * time.Second,
		})

		_, err := mgr.Run(context.Background(), "Go", nil)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}()

	select {
	case <-done:
		// success
	case <-time.After(10 * time.Second):
		t.Fatal("TEST HUNG: Manager did not complete within 10 seconds")
	}
}

func TestIntegrationSixAgentsParallel(t *testing.T) {
	frameworks := []struct{ id, role, goal, response string }{
		{"porter", "Porter's Five Forces Analyst", "Industry analysis", "Five forces complete"},
		{"swot", "SWOT Analyst", "Internal/external analysis", "SWOT complete"},
		{"bmc", "Business Model Canvas Analyst", "Business model", "BMC complete"},
		{"pestel", "PESTEL Analyst", "Macro environment", "PESTEL complete"},
		{"vrio", "VRIO Analyst", "Resource analysis", "VRIO complete"},
		{"blueocean", "Blue Ocean Strategist", "Differentiation strategy", "Blue ocean complete"},
	}

	agents := make([]Runnable, len(frameworks))
	for i, f := range frameworks {
		mock := NewMockLLM(MockResponse{Content: f.response})
		agents[i] = NewAgent(AgentConfig{
			ID:   f.id,
			Role: f.role,
			Goal: f.goal,
			LLM:  mock,
		})
	}

	mgr := NewManager(ManagerConfig{
		ID:       "six-framework-manager",
		Strategy: StrategyParallel,
		Agents:   agents,
		Timeout:  30 * time.Second,
	})

	start := time.Now()
	result, err := mgr.Run(context.Background(), "Full strategic analysis", nil)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, f := range frameworks {
		if !strings.Contains(result.Content, f.id) {
			t.Fatalf("missing results from %s", f.id)
		}
	}

	t.Logf("6 agents parallel completed in %v", duration)
	t.Logf("Result: %d chars", len(result.Content))
}
