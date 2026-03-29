package herdai

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestToolCacheHitAndMiss(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{NewWordThreshold: 5})

	_, ok := cache.Get("test", "hello world")
	if ok {
		t.Fatal("expected cache miss on empty cache")
	}

	cache.Set("test", "hello world", "result-1")

	result, ok := cache.Get("test", "hello world")
	if !ok {
		t.Fatal("expected cache hit")
	}
	if result != "result-1" {
		t.Fatalf("expected result-1, got %s", result)
	}
}

func TestToolCacheAutoRefreshOnNewContext(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{NewWordThreshold: 3})

	cache.Set("test", "BIM tool for architects", "old-result")

	// Same context — should hit
	_, ok := cache.Get("test", "BIM tool for architects")
	if !ok {
		t.Fatal("expected cache hit for same context")
	}

	// Stop-word-only change — should still hit (stop words are filtered)
	_, ok = cache.Get("test", "a BIM tool for the architects")
	if !ok {
		t.Fatal("expected cache hit for stop-word-only change")
	}

	// Significant addition (3+ new meaningful words) — should miss
	_, ok = cache.Get("test", "BIM tool for architects targeting enterprise electrical engineers")
	if ok {
		t.Fatal("expected cache miss for significant context addition")
	}
}

func TestToolCacheDetectsConceptReplacement(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{NewWordThreshold: 3})

	cache.Set("test", "BIM tool for architects in construction", "old-result")

	// Concept replacement: "architects" → "hospitals", "construction" → "healthcare"
	// That's 2 removed + 2 added = 4 drift, which exceeds threshold of 3
	_, ok := cache.Get("test", "BIM tool for hospitals in healthcare")
	if ok {
		t.Fatal("expected cache miss when key concepts are replaced")
	}
}

func TestToolCacheSmallPivotDetection(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{NewWordThreshold: 3})

	cache.Set("test", "AI powered design tool for small architecture firms", "old-result")

	// User changes target customer: "small" → "enterprise" and "architecture" → "engineering"
	// removed: "small", "architecture" + added: "enterprise", "engineering" = 4 drift
	_, ok := cache.Get("test", "AI powered design tool for enterprise engineering firms")
	if ok {
		t.Fatal("expected cache miss when target customer changes (small→enterprise, architecture→engineering)")
	}
}

func TestToolCacheTrivialRephraseStaysHit(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{NewWordThreshold: 3})

	cache.Set("test", "AI powered BIM tool for architects", "old-result")

	// Trivial rephrase — same meaningful words, just different stop words
	_, ok := cache.Get("test", "an AI powered BIM tool for the architects")
	if !ok {
		t.Fatal("expected cache hit for trivial rephrase (only stop words changed)")
	}
}

func TestToolCacheMaxAge(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{
		NewWordThreshold: 5,
		MaxAge:           50 * time.Millisecond,
	})

	cache.Set("test", "ctx", "result")

	_, ok := cache.Get("test", "ctx")
	if !ok {
		t.Fatal("expected hit before expiry")
	}

	time.Sleep(60 * time.Millisecond)

	_, ok = cache.Get("test", "ctx")
	if ok {
		t.Fatal("expected miss after expiry")
	}
}

func TestToolCacheInvalidate(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{})

	cache.Set("tool-a", "ctx", "result-a")
	cache.Set("tool-b", "ctx", "result-b")

	cache.Invalidate("tool-a")

	_, ok := cache.Get("tool-a", "ctx")
	if ok {
		t.Fatal("expected miss after invalidate")
	}

	_, ok = cache.Get("tool-b", "ctx")
	if !ok {
		t.Fatal("expected hit for non-invalidated tool")
	}

	cache.InvalidateAll()
	_, ok = cache.Get("tool-b", "ctx")
	if ok {
		t.Fatal("expected miss after invalidate all")
	}
}

func TestToolCacheWrap(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{NewWordThreshold: 5})

	var callCount atomic.Int32
	original := func(ctx context.Context, args map[string]any) (string, error) {
		callCount.Add(1)
		return "computed-result", nil
	}

	wrapped := cache.Wrap("my_tool", original)

	// First call — should execute
	result, err := wrapped(context.Background(), map[string]any{"context": "test idea"})
	if err != nil {
		t.Fatal(err)
	}
	if result != "computed-result" {
		t.Fatalf("expected computed-result, got %s", result)
	}
	if callCount.Load() != 1 {
		t.Fatalf("expected 1 call, got %d", callCount.Load())
	}

	// Second call same context — should use cache
	result, err = wrapped(context.Background(), map[string]any{"context": "test idea"})
	if err != nil {
		t.Fatal(err)
	}
	if result != "computed-result" {
		t.Fatalf("expected computed-result from cache, got %s", result)
	}
	if callCount.Load() != 1 {
		t.Fatalf("expected still 1 call (cached), got %d", callCount.Load())
	}

	// Third call with refresh=true — should bypass cache
	result, err = wrapped(context.Background(), map[string]any{"context": "test idea", "refresh": true})
	if err != nil {
		t.Fatal(err)
	}
	if callCount.Load() != 2 {
		t.Fatalf("expected 2 calls (refresh bypassed cache), got %d", callCount.Load())
	}
}

func TestToolCacheMaxEntries(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{MaxEntries: 3})

	cache.Set("tool-1", "ctx", "r1")
	cache.Set("tool-2", "ctx", "r2")
	cache.Set("tool-3", "ctx", "r3")
	cache.Set("tool-4", "ctx", "r4")

	entries := cache.Entries()
	if len(entries) > 3 {
		t.Fatalf("expected max 3 entries, got %d", len(entries))
	}
}

func TestToolCacheFieldAwareInvalidation(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{
		ToolDeps: map[string][]string{
			"strategic_analysis":    {"idea", "industry"},
			"financial_analysis":    {"idea", "industry", "revenue", "customer"},
			"competitor_intel":      {"idea", "industry", "customer", "competitors"},
			"gtm_analysis":         {"idea", "customer", "geography", "revenue"},
			"consulting_evaluation": {"idea", "industry", "customer", "revenue"},
		},
	})

	// Set initial context and populate cache
	cache.SetContextFields(map[string]string{
		"idea":     "BIM tool for architects",
		"customer": "small architecture firms",
		"industry": "construction",
		"revenue":  "SaaS subscription",
	})
	cache.Set("strategic_analysis", "ctx", "strategic-result")
	cache.Set("financial_analysis", "ctx", "financial-result")
	cache.Set("competitor_intel", "ctx", "competitor-result")
	cache.Set("gtm_analysis", "ctx", "gtm-result")
	cache.Set("consulting_evaluation", "ctx", "consulting-result")

	// Changing "customer" should invalidate tools that depend on it
	// but NOT strategic_analysis (which only depends on idea, industry)
	invalidated := cache.SetContextFields(map[string]string{
		"idea":     "BIM tool for architects",
		"customer": "large hospital networks",
		"industry": "construction",
		"revenue":  "SaaS subscription",
	})

	// strategic_analysis should survive (doesn't depend on customer)
	if _, ok := cache.Get("strategic_analysis", "ctx"); !ok {
		t.Fatal("strategic_analysis should NOT be invalidated when only customer changed")
	}

	// These should be invalidated (depend on customer)
	if _, ok := cache.Get("financial_analysis", "ctx"); ok {
		t.Fatal("financial_analysis should be invalidated (depends on customer)")
	}
	if _, ok := cache.Get("competitor_intel", "ctx"); ok {
		t.Fatal("competitor_intel should be invalidated (depends on customer)")
	}
	if _, ok := cache.Get("gtm_analysis", "ctx"); ok {
		t.Fatal("gtm_analysis should be invalidated (depends on customer)")
	}
	if _, ok := cache.Get("consulting_evaluation", "ctx"); ok {
		t.Fatal("consulting_evaluation should be invalidated (depends on customer)")
	}

	if len(invalidated) != 4 {
		t.Fatalf("expected 4 invalidated tools, got %d: %v", len(invalidated), invalidated)
	}
}

func TestToolCacheFieldNoChangeNoInvalidation(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{
		ToolDeps: map[string][]string{
			"tool_a": {"idea"},
			"tool_b": {"customer"},
		},
	})

	cache.SetContextFields(map[string]string{"idea": "BIM tool", "customer": "architects"})
	cache.Set("tool_a", "ctx", "result-a")
	cache.Set("tool_b", "ctx", "result-b")

	// Same fields, no change → nothing invalidated
	invalidated := cache.SetContextFields(map[string]string{"idea": "BIM tool", "customer": "architects"})
	if len(invalidated) != 0 {
		t.Fatalf("expected 0 invalidated, got %d", len(invalidated))
	}

	if _, ok := cache.Get("tool_a", "ctx"); !ok {
		t.Fatal("tool_a should still be cached")
	}
	if _, ok := cache.Get("tool_b", "ctx"); !ok {
		t.Fatal("tool_b should still be cached")
	}
}

func TestToolCacheFieldIdeaChangeInvalidatesAll(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{
		ToolDeps: map[string][]string{
			"strategic_analysis": {"idea", "industry"},
			"financial_analysis": {"idea", "revenue"},
			"competitor_intel":   {"idea", "customer"},
		},
	})

	cache.SetContextFields(map[string]string{"idea": "BIM tool for architects"})
	cache.Set("strategic_analysis", "ctx", "r1")
	cache.Set("financial_analysis", "ctx", "r2")
	cache.Set("competitor_intel", "ctx", "r3")

	// Changing "idea" should invalidate ALL tools (all depend on idea)
	invalidated := cache.SetContextFields(map[string]string{"idea": "AI hospital management system"})
	if len(invalidated) != 3 {
		t.Fatalf("expected 3 invalidated (all depend on idea), got %d: %v", len(invalidated), invalidated)
	}
}

func TestToolCacheFieldValueMatchIgnoresCase(t *testing.T) {
	cache := NewToolCache(ToolCacheConfig{
		ToolDeps: map[string][]string{"tool_a": {"industry"}},
	})

	cache.SetContextFields(map[string]string{"industry": "Construction"})
	cache.Set("tool_a", "ctx", "result")

	// Same value, different case → no invalidation
	invalidated := cache.SetContextFields(map[string]string{"industry": "construction"})
	if len(invalidated) != 0 {
		t.Fatalf("expected 0 invalidated (case difference only), got %d", len(invalidated))
	}
}

func TestAgentWithToolCache(t *testing.T) {
	var toolExecCount atomic.Int32

	analysisTool := Tool{
		Name:        "analyze",
		Description: "Analyze a topic",
		Parameters:  []ToolParam{{Name: "context", Type: "string", Description: "context", Required: true}},
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			toolExecCount.Add(1)
			return "analysis-result-for-" + args["context"].(string), nil
		},
	}

	cache := NewToolCache(ToolCacheConfig{NewWordThreshold: 3})

	// 1st run: LLM calls tool, tool executes, result cached
	mock1 := NewMockLLM(
		MockResponse{ToolCalls: []ToolCall{{ID: "c1", Function: "analyze", Args: map[string]any{"context": "BIM for architects"}}}},
		MockResponse{Content: "Here is the analysis."},
	)
	agent1 := NewAgent(AgentConfig{
		ID: "test", Role: "analyst", Goal: "analyze",
		Tools: []Tool{analysisTool}, LLM: mock1, ToolCache: cache,
	})
	_, err := agent1.Run(context.Background(), "analyze BIM", nil)
	if err != nil {
		t.Fatal(err)
	}
	if toolExecCount.Load() != 1 {
		t.Fatalf("expected tool to execute once, got %d", toolExecCount.Load())
	}

	// 2nd run with same context: tool should be served from cache
	mock2 := NewMockLLM(
		MockResponse{ToolCalls: []ToolCall{{ID: "c2", Function: "analyze", Args: map[string]any{"context": "BIM for architects"}}}},
		MockResponse{Content: "Cached analysis."},
	)
	agent2 := NewAgent(AgentConfig{
		ID: "test", Role: "analyst", Goal: "analyze",
		Tools: []Tool{analysisTool}, LLM: mock2, ToolCache: cache,
	})
	_, err = agent2.Run(context.Background(), "analyze BIM again", nil)
	if err != nil {
		t.Fatal(err)
	}
	if toolExecCount.Load() != 1 {
		t.Fatalf("expected tool NOT to execute again (cached), got %d", toolExecCount.Load())
	}

	// 3rd run with significantly different context: tool should re-execute
	mock3 := NewMockLLM(
		MockResponse{ToolCalls: []ToolCall{{ID: "c3", Function: "analyze", Args: map[string]any{"context": "BIM for enterprise electrical engineers construction"}}}},
		MockResponse{Content: "Fresh analysis."},
	)
	agent3 := NewAgent(AgentConfig{
		ID: "test", Role: "analyst", Goal: "analyze",
		Tools: []Tool{analysisTool}, LLM: mock3, ToolCache: cache,
	})
	_, err = agent3.Run(context.Background(), "analyze BIM for enterprise", nil)
	if err != nil {
		t.Fatal(err)
	}
	if toolExecCount.Load() != 2 {
		t.Fatalf("expected tool to re-execute on context change, got %d", toolExecCount.Load())
	}
}
