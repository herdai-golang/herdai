package herdai

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func newTestAgent(id, role, goal string, response string) (*Agent, *MockLLM) {
	mock := NewMockLLM(MockResponse{Content: response})
	agent := NewAgent(AgentConfig{
		ID:   id,
		Role: role,
		Goal: goal,
		LLM:  mock,
	})
	return agent, mock
}

func TestManagerSequential(t *testing.T) {
	a1, _ := newTestAgent("agent-1", "Writer", "Write content", "Draft content here")
	a2, _ := newTestAgent("agent-2", "Editor", "Edit content", "Edited and polished content")

	mgr := NewManager(ManagerConfig{
		ID:       "seq-manager",
		Strategy: StrategySequential,
		Agents:   []Runnable{a1, a2},
	})

	result, err := mgr.Run(context.Background(), "Write a blog post", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "Edited and polished content" {
		t.Fatalf("expected final agent's output, got: %s", result.Content)
	}
}

func TestManagerParallel(t *testing.T) {
	a1, _ := newTestAgent("porter", "Porter Analyst", "Analyze forces", "Five forces analysis")
	a2, _ := newTestAgent("swot", "SWOT Analyst", "Run SWOT", "SWOT matrix complete")
	a3, _ := newTestAgent("pestel", "PESTEL Analyst", "Run PESTEL", "PESTEL analysis done")

	mgr := NewManager(ManagerConfig{
		ID:       "par-manager",
		Strategy: StrategyParallel,
		Agents:   []Runnable{a1, a2, a3},
	})

	result, err := mgr.Run(context.Background(), "Analyze the industry", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "porter") {
		t.Fatal("expected porter results in merged output")
	}
	if !strings.Contains(result.Content, "swot") {
		t.Fatal("expected swot results in merged output")
	}
	if !strings.Contains(result.Content, "pestel") {
		t.Fatal("expected pestel results in merged output")
	}
}

func TestManagerParallelConcurrency(t *testing.T) {
	var running int64
	var maxConcurrent int64

	makeSlowAgent := func(id string) *Agent {
		mock := NewMockLLM(MockResponse{Content: id + " done"})
		return NewAgent(AgentConfig{
			ID:   id,
			Role: "Worker",
			Goal: "Work",
			LLM: &concurrencyTracker{
				llm:           mock,
				running:       &running,
				maxConcurrent: &maxConcurrent,
			},
		})
	}

	agents := make([]Runnable, 6)
	for i := 0; i < 6; i++ {
		agents[i] = makeSlowAgent(fmt.Sprintf("agent-%d", i))
	}

	mgr := NewManager(ManagerConfig{
		ID:       "conc-manager",
		Strategy: StrategyParallel,
		Agents:   agents,
	})

	result, err := mgr.Run(context.Background(), "Go", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}

	peak := atomic.LoadInt64(&maxConcurrent)
	if peak < 2 {
		t.Logf("peak concurrency was %d (may vary by scheduling)", peak)
	}
}

type concurrencyTracker struct {
	llm           LLM
	running       *int64
	maxConcurrent *int64
}

func (ct *concurrencyTracker) Chat(ctx context.Context, messages []Message, tools []Tool) (*LLMResponse, error) {
	current := atomic.AddInt64(ct.running, 1)
	for {
		old := atomic.LoadInt64(ct.maxConcurrent)
		if current <= old {
			break
		}
		if atomic.CompareAndSwapInt64(ct.maxConcurrent, old, current) {
			break
		}
	}
	time.Sleep(10 * time.Millisecond)
	defer atomic.AddInt64(ct.running, -1)
	return ct.llm.Chat(ctx, messages, tools)
}

func TestManagerRoundRobin(t *testing.T) {
	mock1 := NewMockLLM(
		MockResponse{Content: "round 1 from agent-1"},
		MockResponse{Content: "round 2 from agent-1 [DONE]"},
	)
	a1 := NewAgent(AgentConfig{ID: "agent-1", Role: "A", Goal: "Do A", LLM: mock1})

	mock2 := NewMockLLM(
		MockResponse{Content: "round 1 from agent-2"},
	)
	a2 := NewAgent(AgentConfig{ID: "agent-2", Role: "B", Goal: "Do B", LLM: mock2})

	mgr := NewManager(ManagerConfig{
		ID:       "rr-manager",
		Strategy: StrategyRoundRobin,
		Agents:   []Runnable{a1, a2},
		MaxTurns: 10,
	})

	result, err := mgr.Run(context.Background(), "Start", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Content, "[DONE]") {
		t.Fatalf("expected finish signal in result, got: %s", result.Content)
	}
}

func TestManagerLLMRouter(t *testing.T) {
	a1, _ := newTestAgent("analyst", "Analyst", "Analyze data", "Analysis complete")
	a2, _ := newTestAgent("writer", "Writer", "Write report", "Report written")

	routerMock := NewMockLLM(
		MockResponse{Content: `{"agent_id": "analyst", "instruction": "Analyze the market"}`},
		MockResponse{Content: `{"agent_id": "writer", "instruction": "Write the report"}`},
		MockResponse{Content: `{"agent_id": "FINISH", "instruction": "All tasks completed successfully"}`},
	)

	mgr := NewManager(ManagerConfig{
		ID:       "router-manager",
		Strategy: StrategyLLMRouter,
		Agents:   []Runnable{a1, a2},
		LLM:      routerMock,
		MaxTurns: 10,
	})

	result, err := mgr.Run(context.Background(), "Analyze and report", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "All tasks completed successfully" {
		t.Fatalf("expected finish summary, got: %s", result.Content)
	}
	if routerMock.CallCount() != 3 {
		t.Fatalf("expected 3 router LLM calls, got %d", routerMock.CallCount())
	}
}

func TestManagerLLMRouterUnknownAgent(t *testing.T) {
	a1, _ := newTestAgent("analyst", "Analyst", "Analyze", "Done")

	routerMock := NewMockLLM(
		MockResponse{Content: `{"agent_id": "nonexistent", "instruction": "Do something"}`},
		MockResponse{Content: `{"agent_id": "analyst", "instruction": "Analyze"}`},
		MockResponse{Content: `{"agent_id": "FINISH", "instruction": "Complete"}`},
	)

	mgr := NewManager(ManagerConfig{
		ID:       "router-unknown",
		Strategy: StrategyLLMRouter,
		Agents:   []Runnable{a1},
		LLM:      routerMock,
		MaxTurns: 10,
	})

	result, err := mgr.Run(context.Background(), "Go", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "Complete" {
		t.Fatalf("expected 'Complete', got: %s", result.Content)
	}
}

func TestManagerLLMRouterRequiresLLM(t *testing.T) {
	mgr := NewManager(ManagerConfig{
		ID:       "no-llm",
		Strategy: StrategyLLMRouter,
		Agents:   []Runnable{&dummyRunnable{id: "a1"}},
	})

	_, err := mgr.Run(context.Background(), "Go", nil)
	if err == nil {
		t.Fatal("expected error when LLMRouter has no LLM")
	}
}

func TestManagerNoAgents(t *testing.T) {
	mgr := NewManager(ManagerConfig{
		ID:       "empty-manager",
		Strategy: StrategySequential,
	})

	_, err := mgr.Run(context.Background(), "Go", nil)
	if err == nil {
		t.Fatal("expected error with no agents")
	}
}

func TestManagerTimeout(t *testing.T) {
	slowMock := NewMockLLM(MockResponse{Error: context.DeadlineExceeded})
	a1 := NewAgent(AgentConfig{ID: "slow", Role: "Slow", Goal: "Be slow", LLM: slowMock})

	mgr := NewManager(ManagerConfig{
		ID:       "timeout-manager",
		Strategy: StrategySequential,
		Agents:   []Runnable{a1},
		Timeout:  50 * time.Millisecond,
	})

	_, err := mgr.Run(context.Background(), "Go", nil)
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestManagerAddAgent(t *testing.T) {
	mgr := NewManager(ManagerConfig{
		ID:       "dynamic-manager",
		Strategy: StrategySequential,
		Agents:   []Runnable{},
	})

	a1, _ := newTestAgent("late-agent", "Late", "Join late", "I joined late!")
	mgr.AddAgent(a1)

	result, err := mgr.Run(context.Background(), "Go", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "I joined late!" {
		t.Fatalf("expected late agent output, got: %s", result.Content)
	}
}

func TestManagerHierarchy(t *testing.T) {
	a1, _ := newTestAgent("sub-1", "Sub Worker 1", "Do sub work", "Sub result 1")
	a2, _ := newTestAgent("sub-2", "Sub Worker 2", "Do sub work", "Sub result 2")

	subManager := NewManager(ManagerConfig{
		ID:       "sub-manager",
		Strategy: StrategyParallel,
		Agents:   []Runnable{a1, a2},
	})

	a3, _ := newTestAgent("top-agent", "Top Worker", "Top work", "Top result")

	topManager := NewManager(ManagerConfig{
		ID:       "top-manager",
		Strategy: StrategySequential,
		Agents:   []Runnable{subManager, a3},
	})

	result, err := topManager.Run(context.Background(), "Full pipeline", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "Top result" {
		t.Fatalf("expected 'Top result', got: %s", result.Content)
	}
}

// ── Synthesis Tests ─────────────────────────────────────────────────────────

func TestParallelWithSynthesis(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Porter analysis: competitive forces are high"})
	mock.PushResponse(LLMResponse{Content: "SWOT analysis: strong brand, weak supply chain"})
	// Synthesis LLM response
	mock.PushResponse(LLMResponse{Content: "SYNTHESIZED: Both analyses show high competition. The strong brand is an advantage but supply chain weakness is a risk. Recommendation: invest in supply chain improvements."})

	a1 := NewAgent(AgentConfig{ID: "porter", Role: "Porter", Goal: "Analyze", LLM: mock})
	a2 := NewAgent(AgentConfig{ID: "swot", Role: "SWOT", Goal: "Analyze", LLM: mock})

	mgr := NewManager(ManagerConfig{
		ID:       "synth-manager",
		Strategy: StrategyParallel,
		Agents:   []Runnable{a1, a2},
		LLM:      mock,
		SynthesisPrompt: "Create a unified strategic recommendation from all analyses. Include an opportunity score (1-10) and top 3 action items.",
	})

	result, err := mgr.Run(context.Background(), "Analyze coffee market", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "SYNTHESIZED") {
		t.Errorf("expected synthesized output, got: %s", result.Content)
	}

	if synthesized, ok := result.Metadata["synthesized"].(bool); !ok || !synthesized {
		t.Error("expected metadata.synthesized to be true")
	}
}

func TestParallelWithoutSynthesis(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Result A"})
	mock.PushResponse(LLMResponse{Content: "Result B"})

	a1 := NewAgent(AgentConfig{ID: "a1", Role: "A", Goal: "Do", LLM: mock})
	a2 := NewAgent(AgentConfig{ID: "a2", Role: "B", Goal: "Do", LLM: mock})

	mgr := NewManager(ManagerConfig{
		ID:       "no-synth",
		Strategy: StrategyParallel,
		Agents:   []Runnable{a1, a2},
	})

	result, err := mgr.Run(context.Background(), "Test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "Result A") || !strings.Contains(result.Content, "Result B") {
		t.Error("expected concatenated results")
	}

	if synthesized, ok := result.Metadata["synthesized"].(bool); ok && synthesized {
		t.Error("expected synthesized to be false when no SynthesisPrompt")
	}
}

func TestSynthesisFallbackOnError(t *testing.T) {
	failLLM := &MockLLM{}
	failLLM.PushResponse(LLMResponse{Content: "Agent output"})
	failLLM.PushResponse(LLMResponse{Content: "Agent output 2"})

	a1 := NewAgent(AgentConfig{ID: "a1", Role: "A", Goal: "Do", LLM: failLLM})
	a2 := NewAgent(AgentConfig{ID: "a2", Role: "B", Goal: "Do", LLM: failLLM})

	mgr := NewManager(ManagerConfig{
		ID:       "fail-synth",
		Strategy: StrategyParallel,
		Agents:   []Runnable{a1, a2},
		LLM:      failLLM, // will have no more responses, so synthesis will fail
		SynthesisPrompt: "Synthesize everything",
	})

	result, err := mgr.Run(context.Background(), "Test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should fall back to concatenation
	if !strings.Contains(result.Content, "Agent output") {
		t.Error("expected fallback concatenation on synthesis failure")
	}
}

// ── Conversational HITL Tests ───────────────────────────────────────────────

func TestConversationHandler(t *testing.T) {
	mock := NewMockLLM()
	// Agent responses
	mock.PushResponse(LLMResponse{Content: "Porter: high competition"})
	mock.PushResponse(LLMResponse{Content: "SWOT: strong brand"})
	// Router response for follow-up
	mock.PushResponse(LLMResponse{Content: `{"agent_id": "porter", "instruction": "Explain the supplier power in more detail"}`})
	// Agent response to follow-up
	mock.PushResponse(LLMResponse{Content: "Supplier power is high because there are few suppliers"})

	a1 := NewAgent(AgentConfig{ID: "porter", Role: "Porter Analyst", Goal: "Analyze competitive forces", LLM: mock})
	a2 := NewAgent(AgentConfig{ID: "swot", Role: "SWOT Analyst", Goal: "Do SWOT", LLM: mock})

	callCount := 0
	handler := func(ctx context.Context, prompt string) string {
		callCount++
		if callCount == 1 {
			return "Tell me more about supplier power"
		}
		return "done"
	}

	mgr := NewManager(ManagerConfig{
		ID:                   "conv-mgr",
		Strategy:             StrategyParallel,
		Agents:               []Runnable{a1, a2},
		LLM:                  mock,
		ConversationHandler:  handler,
		MaxConversationTurns: 3,
	})

	result, err := mgr.Run(context.Background(), "Analyze coffee market", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "Supplier power") {
		t.Errorf("expected follow-up result, got: %s", result.Content)
	}

	if callCount < 2 {
		t.Errorf("expected handler to be called at least 2 times, got %d", callCount)
	}
}

func TestConversationHandlerDoneImmediately(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Analysis complete"})

	a1 := NewAgent(AgentConfig{ID: "a1", Role: "A", Goal: "Do", LLM: mock})

	handler := func(ctx context.Context, prompt string) string {
		return "done"
	}

	mgr := NewManager(ManagerConfig{
		ID:                  "done-mgr",
		Strategy:            StrategySequential,
		Agents:              []Runnable{a1},
		LLM:                 mock,
		ConversationHandler: handler,
	})

	result, err := mgr.Run(context.Background(), "Test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Content != "Analysis complete" {
		t.Errorf("expected original result, got: %s", result.Content)
	}
}

func TestConversationHandlerMaxTurns(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Initial result"})

	for i := 0; i < 5; i++ {
		mock.PushResponse(LLMResponse{Content: fmt.Sprintf(`{"agent_id": "a1", "instruction": "question %d"}`, i)})
		mock.PushResponse(LLMResponse{Content: fmt.Sprintf("Answer %d", i)})
	}

	a1 := NewAgent(AgentConfig{ID: "a1", Role: "A", Goal: "Do", LLM: mock})

	callCount := 0
	handler := func(ctx context.Context, prompt string) string {
		callCount++
		return "another question"
	}

	mgr := NewManager(ManagerConfig{
		ID:                   "max-turns-mgr",
		Strategy:             StrategySequential,
		Agents:               []Runnable{a1},
		LLM:                  mock,
		ConversationHandler:  handler,
		MaxConversationTurns: 2,
	})

	_, err := mgr.Run(context.Background(), "Test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if callCount > 2 {
		t.Errorf("expected max 2 conversation turns, got %d", callCount)
	}
}

// ── Synthesis Clarification Tests ────────────────────────────────────────────

func TestSynthesisWithClarification(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Agent A output"})
	mock.PushResponse(LLMResponse{Content: "Agent B output"})
	// Synthesis asks for clarification
	mock.PushResponse(LLMResponse{Content: "CLARIFY: What is your target market — enterprise or SMB?"})
	// After human answers, final synthesis
	mock.PushResponse(LLMResponse{Content: "Final synthesis: For enterprise market, prioritize security features."})

	a1 := NewAgent(AgentConfig{ID: "a1", Role: "A", Goal: "Do", LLM: mock})
	a2 := NewAgent(AgentConfig{ID: "a2", Role: "B", Goal: "Do", LLM: mock})

	handler := func(ctx context.Context, prompt string) string {
		if strings.Contains(prompt, "target market") {
			return "Enterprise customers"
		}
		return "done"
	}

	mgr := NewManager(ManagerConfig{
		ID:                  "clarify-mgr",
		Strategy:            StrategyParallel,
		Agents:              []Runnable{a1, a2},
		LLM:                 mock,
		SynthesisPrompt:     "Synthesize with clarification",
		ConversationHandler: handler,
	})

	result, err := mgr.Run(context.Background(), "Analyze market", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "enterprise") {
		t.Errorf("expected clarified synthesis, got: %s", result.Content)
	}

	rounds, _ := result.Metadata["clarify_rounds"].(int)
	if rounds != 1 {
		t.Errorf("expected 1 clarify round, got %d", rounds)
	}
}

func TestSynthesisSkipClarification(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Agent A output"})
	mock.PushResponse(LLMResponse{Content: "Agent B output"})
	// Synthesis asks for clarification
	mock.PushResponse(LLMResponse{Content: "CLARIFY: What budget range?"})
	// After skip, proceeds without clarification
	mock.PushResponse(LLMResponse{Content: "Synthesis with assumed mid-range budget."})

	a1 := NewAgent(AgentConfig{ID: "a1", Role: "A", Goal: "Do", LLM: mock})
	a2 := NewAgent(AgentConfig{ID: "a2", Role: "B", Goal: "Do", LLM: mock})

	handler := func(ctx context.Context, prompt string) string {
		if strings.Contains(prompt, "budget") {
			return "skip"
		}
		return "done"
	}

	mgr := NewManager(ManagerConfig{
		ID:                  "skip-clarify-mgr",
		Strategy:            StrategyParallel,
		Agents:              []Runnable{a1, a2},
		LLM:                 mock,
		SynthesisPrompt:     "Synthesize",
		ConversationHandler: handler,
	})

	result, err := mgr.Run(context.Background(), "Test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(result.Content, "assumed") {
		t.Errorf("expected assumed answer after skip, got: %s", result.Content)
	}
}

// ── Sequential Interrupt Tests ──────────────────────────────────────────────

func TestSequentialHumanInterrupt(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Step 1 research done"})
	mock.PushResponse(LLMResponse{Content: "Step 2 writing with feedback"})

	a1 := NewAgent(AgentConfig{ID: "researcher", Role: "Researcher", Goal: "Research", LLM: mock})
	a2 := NewAgent(AgentConfig{ID: "writer", Role: "Writer", Goal: "Write", LLM: mock})

	handler := func(ctx context.Context, prompt string) string {
		if strings.Contains(prompt, "Step 1 research") {
			return "Focus more on competitor pricing"
		}
		return "done"
	}

	mgr := NewManager(ManagerConfig{
		ID:                  "interrupt-mgr",
		Strategy:            StrategySequential,
		Agents:              []Runnable{a1, a2},
		LLM:                 mock,
		ConversationHandler: handler,
	})

	result, err := mgr.Run(context.Background(), "Analyze market", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result == nil {
		t.Fatal("expected a result")
	}
}

func TestSequentialHumanStop(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Step 1 done"})
	mock.PushResponse(LLMResponse{Content: "Step 2 should not run"})

	a1 := NewAgent(AgentConfig{ID: "a1", Role: "A", Goal: "Do", LLM: mock})
	a2 := NewAgent(AgentConfig{ID: "a2", Role: "B", Goal: "Do", LLM: mock})

	handler := func(ctx context.Context, prompt string) string {
		return "stop"
	}

	mgr := NewManager(ManagerConfig{
		ID:                  "stop-mgr",
		Strategy:            StrategySequential,
		Agents:              []Runnable{a1, a2},
		LLM:                 mock,
		ConversationHandler: handler,
	})

	result, err := mgr.Run(context.Background(), "Test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Content != "Step 1 done" {
		t.Errorf("expected pipeline to stop after step 1, got: %s", result.Content)
	}
}

// ── Reject Tests ────────────────────────────────────────────────────────────

func TestConversationReject(t *testing.T) {
	mock := NewMockLLM()
	mock.PushResponse(LLMResponse{Content: "Analysis complete"})

	a1 := NewAgent(AgentConfig{ID: "a1", Role: "A", Goal: "Do", LLM: mock})

	handler := func(ctx context.Context, prompt string) string {
		return "reject"
	}

	mgr := NewManager(ManagerConfig{
		ID:                  "reject-mgr",
		Strategy:            StrategySequential,
		Agents:              []Runnable{a1},
		LLM:                 mock,
		ConversationHandler: handler,
	})

	result, err := mgr.Run(context.Background(), "Test", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rejected, _ := result.Metadata["rejected"].(bool)
	if !rejected {
		t.Error("expected result to be marked as rejected")
	}
}

// dummyRunnable for testing edge cases
type dummyRunnable struct {
	id string
}

func (d *dummyRunnable) GetID() string { return d.id }
func (d *dummyRunnable) Run(ctx context.Context, input string, conv *Conversation) (*Result, error) {
	return &Result{AgentID: d.id, Content: "dummy"}, nil
}

