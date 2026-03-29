package herdai

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

func TestAgentBasicRun(t *testing.T) {
	mock := NewMockLLM(MockResponse{Content: "Hello from agent!"})

	agent := NewAgent(AgentConfig{
		ID:   "test-agent",
		Role: "Greeter",
		Goal: "Greet the user",
		LLM:  mock,
	})

	result, err := agent.Run(context.Background(), "Say hello", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.AgentID != "test-agent" {
		t.Fatalf("expected agent_id 'test-agent', got '%s'", result.AgentID)
	}
	if result.Content != "Hello from agent!" {
		t.Fatalf("expected 'Hello from agent!', got '%s'", result.Content)
	}
	if mock.CallCount() != 1 {
		t.Fatalf("expected 1 LLM call, got %d", mock.CallCount())
	}
}

func TestAgentWithConversation(t *testing.T) {
	mock := NewMockLLM(MockResponse{Content: "I see the context"})

	agent := NewAgent(AgentConfig{
		ID:   "ctx-agent",
		Role: "Analyst",
		Goal: "Analyze with context",
		LLM:  mock,
	})

	conv := NewConversation()
	conv.AddTurn(Turn{AgentID: "other-agent", Role: "assistant", Content: "prior work"})

	result, err := agent.Run(context.Background(), "Continue", conv)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "I see the context" {
		t.Fatalf("unexpected content: %s", result.Content)
	}

	if conv.Len() != 2 {
		t.Fatalf("expected 2 turns (1 prior + 1 agent), got %d", conv.Len())
	}
}

func TestAgentWithToolCalls(t *testing.T) {
	toolCalled := false

	searchTool := Tool{
		Name:        "web_search",
		Description: "Search the web",
		Parameters:  []ToolParam{{Name: "query", Type: "string", Description: "search query", Required: true}},
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			toolCalled = true
			return "Search results: competitor A, competitor B", nil
		},
	}

	mock := NewMockLLM(
		MockResponse{
			ToolCalls: []ToolCall{{ID: "call_1", Function: "web_search", Args: map[string]any{"query": "competitors"}}},
		},
		MockResponse{Content: "Based on search: competitor A and B found"},
	)

	agent := NewAgent(AgentConfig{
		ID:    "search-agent",
		Role:  "Researcher",
		Goal:  "Find competitors",
		Tools: []Tool{searchTool},
		LLM:   mock,
	})

	result, err := agent.Run(context.Background(), "Find competitors", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !toolCalled {
		t.Fatal("expected tool to be called")
	}
	if result.Content != "Based on search: competitor A and B found" {
		t.Fatalf("unexpected content: %s", result.Content)
	}
	if mock.CallCount() != 2 {
		t.Fatalf("expected 2 LLM calls (1 with tool call + 1 final), got %d", mock.CallCount())
	}

	tc, ok := result.Metadata["tool_calls"]
	if !ok || tc.(int) != 1 {
		t.Fatalf("expected 1 tool call in metadata, got %v", tc)
	}
}

func TestAgentTimeout(t *testing.T) {
	slowLLM := NewMockLLM(MockResponse{Content: "should not reach"})
	slowLLM.responses[0] = MockResponse{
		Error: context.DeadlineExceeded,
	}

	agent := NewAgent(AgentConfig{
		ID:      "slow-agent",
		Role:    "Slow",
		Goal:    "Be slow",
		LLM:     slowLLM,
		Timeout: 50 * time.Millisecond,
	})

	_, err := agent.Run(context.Background(), "Do something", nil)
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestAgentMaxToolCalls(t *testing.T) {
	dummyTool := Tool{
		Name:        "dummy",
		Description: "dummy tool",
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			return "ok", nil
		},
	}

	responses := make([]MockResponse, 20)
	for i := range responses {
		responses[i] = MockResponse{
			ToolCalls: []ToolCall{{ID: fmt.Sprintf("call_%d", i), Function: "dummy", Args: map[string]any{}}},
		}
	}

	mock := NewMockLLM(responses...)

	agent := NewAgent(AgentConfig{
		ID:           "greedy-agent",
		Role:         "Greedy",
		Goal:         "Use too many tools",
		Tools:        []Tool{dummyTool},
		LLM:          mock,
		MaxToolCalls: 3,
	})

	_, err := agent.Run(context.Background(), "Go", nil)
	if err == nil {
		t.Fatal("expected max tool calls error")
	}
}

func TestAgentUnknownTool(t *testing.T) {
	mock := NewMockLLM(
		MockResponse{
			ToolCalls: []ToolCall{{ID: "call_1", Function: "nonexistent", Args: map[string]any{}}},
		},
		MockResponse{Content: "Recovered after unknown tool"},
	)

	agent := NewAgent(AgentConfig{
		ID:   "recover-agent",
		Role: "Recoverer",
		Goal: "Handle unknown tools",
		LLM:  mock,
	})

	result, err := agent.Run(context.Background(), "Try something", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "Recovered after unknown tool" {
		t.Fatalf("unexpected content: %s", result.Content)
	}
}

func TestAgentToolError(t *testing.T) {
	failTool := Tool{
		Name:        "fail_tool",
		Description: "always fails",
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			return "", fmt.Errorf("tool broke")
		},
	}

	mock := NewMockLLM(
		MockResponse{
			ToolCalls: []ToolCall{{ID: "call_1", Function: "fail_tool", Args: map[string]any{}}},
		},
		MockResponse{Content: "Recovered after tool error"},
	)

	agent := NewAgent(AgentConfig{
		ID:    "error-agent",
		Role:  "Error handler",
		Goal:  "Handle tool errors",
		Tools: []Tool{failTool},
		LLM:   mock,
	})

	result, err := agent.Run(context.Background(), "Try", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "Recovered after tool error" {
		t.Fatalf("unexpected content: %s", result.Content)
	}
}

func TestAgentLLMError(t *testing.T) {
	mock := NewMockLLM(MockResponse{Error: fmt.Errorf("API rate limited")})

	agent := NewAgent(AgentConfig{
		ID:   "err-agent",
		Role: "Errored",
		Goal: "Fail",
		LLM:  mock,
	})

	_, err := agent.Run(context.Background(), "Go", nil)
	if err == nil {
		t.Fatal("expected error from LLM")
	}
}

func TestAgentContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	mock := NewMockLLM(MockResponse{Content: "should not reach"})

	agent := NewAgent(AgentConfig{
		ID:   "cancel-agent",
		Role: "Cancelled",
		Goal: "Be cancelled",
		LLM:  mock,
	})

	_, err := agent.Run(ctx, "Go", nil)
	if err == nil {
		t.Fatal("expected cancellation error")
	}
}

func TestAgentParallelToolCalls(t *testing.T) {
	var running atomic.Int32
	var peak atomic.Int32

	makeTool := func(name string, delay time.Duration) Tool {
		return Tool{
			Name:        name,
			Description: name + " tool",
			Execute: func(ctx context.Context, args map[string]any) (string, error) {
				cur := running.Add(1)
				for {
					old := peak.Load()
					if cur <= old || peak.CompareAndSwap(old, cur) {
						break
					}
				}
				time.Sleep(delay)
				running.Add(-1)
				return name + " result", nil
			},
		}
	}

	toolA := makeTool("tool_a", 100*time.Millisecond)
	toolB := makeTool("tool_b", 100*time.Millisecond)
	toolC := makeTool("tool_c", 100*time.Millisecond)

	mock := NewMockLLM(
		MockResponse{
			ToolCalls: []ToolCall{
				{ID: "c1", Function: "tool_a", Args: map[string]any{}},
				{ID: "c2", Function: "tool_b", Args: map[string]any{}},
				{ID: "c3", Function: "tool_c", Args: map[string]any{}},
			},
		},
		MockResponse{Content: "All done"},
	)

	agent := NewAgent(AgentConfig{
		ID:    "parallel-agent",
		Role:  "Tester",
		Goal:  "Test parallel",
		Tools: []Tool{toolA, toolB, toolC},
		LLM:   mock,
	})

	start := time.Now()
	result, err := agent.Run(context.Background(), "Go", nil)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "All done" {
		t.Fatalf("unexpected content: %s", result.Content)
	}

	// With parallel execution, 3 tools at 100ms each should take ~100ms, not ~300ms.
	if elapsed > 250*time.Millisecond {
		t.Fatalf("expected parallel execution (~100ms), took %v — tools ran sequentially", elapsed)
	}

	// Verify at least 2 tools ran concurrently at the same time.
	if peak.Load() < 2 {
		t.Fatalf("expected peak concurrency >= 2, got %d", peak.Load())
	}

	// Verify all tool results were fed back to the LLM in correct order.
	if mock.CallCount() != 2 {
		t.Fatalf("expected 2 LLM calls, got %d", mock.CallCount())
	}
	secondCall := mock.Calls[1]
	toolMsgs := 0
	for _, m := range secondCall.Messages {
		if m.Role == RoleTool {
			toolMsgs++
		}
	}
	if toolMsgs != 3 {
		t.Fatalf("expected 3 tool result messages in second LLM call, got %d", toolMsgs)
	}
}

func TestAgentSequentialToolCallsWhenDisabled(t *testing.T) {
	var running atomic.Int32
	var peak atomic.Int32

	makeTool := func(name string) Tool {
		return Tool{
			Name:        name,
			Description: name + " tool",
			Execute: func(ctx context.Context, args map[string]any) (string, error) {
				cur := running.Add(1)
				for {
					old := peak.Load()
					if cur <= old || peak.CompareAndSwap(old, cur) {
						break
					}
				}
				time.Sleep(50 * time.Millisecond)
				running.Add(-1)
				return name + " result", nil
			},
		}
	}

	mock := NewMockLLM(
		MockResponse{
			ToolCalls: []ToolCall{
				{ID: "c1", Function: "tool_a", Args: map[string]any{}},
				{ID: "c2", Function: "tool_b", Args: map[string]any{}},
			},
		},
		MockResponse{Content: "Done sequentially"},
	)

	seqFalse := false
	agent := NewAgent(AgentConfig{
		ID:                "seq-agent",
		Role:              "Tester",
		Goal:              "Test sequential",
		Tools:             []Tool{makeTool("tool_a"), makeTool("tool_b")},
		LLM:               mock,
		ParallelToolCalls: &seqFalse,
	})

	result, err := agent.Run(context.Background(), "Go", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "Done sequentially" {
		t.Fatalf("unexpected content: %s", result.Content)
	}

	// With parallel disabled, peak concurrency must be 1.
	if peak.Load() != 1 {
		t.Fatalf("expected peak concurrency of 1 (sequential), got %d", peak.Load())
	}
}

func TestAgentParallelToolResultOrder(t *testing.T) {
	// Tool A is slow, Tool B is fast. Verify results arrive in call order, not completion order.
	toolA := Tool{
		Name:        "slow_tool",
		Description: "slow",
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			time.Sleep(100 * time.Millisecond)
			return "slow_result", nil
		},
	}
	toolB := Tool{
		Name:        "fast_tool",
		Description: "fast",
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			return "fast_result", nil
		},
	}

	mock := NewMockLLM(
		MockResponse{
			ToolCalls: []ToolCall{
				{ID: "c1", Function: "slow_tool", Args: map[string]any{}},
				{ID: "c2", Function: "fast_tool", Args: map[string]any{}},
			},
		},
		MockResponse{Content: "Order verified"},
	)

	agent := NewAgent(AgentConfig{
		ID:    "order-agent",
		Role:  "Tester",
		Goal:  "Test order",
		Tools: []Tool{toolA, toolB},
		LLM:   mock,
	})

	result, err := agent.Run(context.Background(), "Go", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "Order verified" {
		t.Fatalf("unexpected content: %s", result.Content)
	}

	// Check the tool messages sent to the LLM are in the original call order.
	secondCall := mock.Calls[1]
	var toolResults []string
	for _, m := range secondCall.Messages {
		if m.Role == RoleTool {
			toolResults = append(toolResults, m.Content)
		}
	}
	if len(toolResults) != 2 {
		t.Fatalf("expected 2 tool results, got %d", len(toolResults))
	}
	if toolResults[0] != "slow_result" || toolResults[1] != "fast_result" {
		t.Fatalf("tool results out of order: got %v", toolResults)
	}
}
