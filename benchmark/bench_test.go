// Tests and benchmarks for the Competitive Intelligence Engine.
//
// Run options:
//
//	go test ./...                          — correctness tests only
//	go test -run TestSpeedup -v            — proves parallel is faster
//	go test -bench=. -benchtime=5s         — formal Go micro-benchmarks
//	go test -bench=. -benchtime=5s -count=3 -benchmem  — with allocations
//	go test -race ./...                    — race detector (goroutine safety)
//
// Reading benchmark output:
//
//	BenchmarkParallelAgents_ParallelTools-10   200   6_100_000 ns/op
//	BenchmarkSequentialAgents_SequentialTools-10  20  53_000_000 ns/op
//
// The ns/op for the sequential run should be ≈ agentCount × toolsPerAgent
// times larger than the fully-parallel run — that ratio is exactly what the
// Python GIL costs you in a comparable workload.
package main

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/herdai-golang/herdai"
)

// ─── Benchmark delays ─────────────────────────────────────────────────────────
// Small enough that benchmarks finish quickly; large enough to produce a clear
// speedup signal over OS scheduling jitter.

const (
	benchLLMDelay  = 20 * time.Millisecond
	benchToolDelay = 8 * time.Millisecond
)

// ─── Correctness tests ────────────────────────────────────────────────────────

// TestParallelPipelineReturnsResult verifies the happy path: all 6 agents run,
// all tools execute, and a non-empty merged result is produced.
func TestParallelPipelineReturnsResult(t *testing.T) {
	cfg := AnalysisConfig{
		LLMDelay:      0, // no artificial delay in unit tests
		ToolDelay:     0,
		ParallelTools: true,
		Strategy:      herdai.StrategyParallel,
	}

	result, elapsed, err := RunAnalysis(context.Background(), "TestCorp", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.Content == "" {
		t.Fatal("expected non-empty result content")
	}
	t.Logf("parallel pipeline completed in %v, result length: %d chars", elapsed, len(result.Content))
}

// TestSequentialPipelineReturnsResult does the same for the sequential strategy
// to confirm both paths produce results.
func TestSequentialPipelineReturnsResult(t *testing.T) {
	cfg := AnalysisConfig{
		LLMDelay:      0,
		ToolDelay:     0,
		ParallelTools: false,
		Strategy:      herdai.StrategySequential,
	}

	result, _, err := RunAnalysis(context.Background(), "TestCorp", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Content == "" {
		t.Fatal("expected non-empty result from sequential pipeline")
	}
}

// TestAllFourScenarios runs every combination and asserts none error out.
func TestAllFourScenarios(t *testing.T) {
	type scenario struct {
		name          string
		parallelTools bool
		strategy      herdai.Strategy
	}
	scenarios := []scenario{
		{"par_agents+par_tools", true, herdai.StrategyParallel},
		{"par_agents+seq_tools", false, herdai.StrategyParallel},
		{"seq_agents+par_tools", true, herdai.StrategySequential},
		{"seq_agents+seq_tools", false, herdai.StrategySequential},
	}

	for _, sc := range scenarios {
		sc := sc
		t.Run(sc.name, func(t *testing.T) {
			t.Parallel()
			cfg := AnalysisConfig{
				LLMDelay:      0,
				ToolDelay:     0,
				ParallelTools: sc.parallelTools,
				Strategy:      sc.strategy,
			}
			result, elapsed, err := RunAnalysis(context.Background(), "AcmeCorp", cfg)
			if err != nil {
				t.Fatalf("scenario %q failed: %v", sc.name, err)
			}
			if result == nil || result.Content == "" {
				t.Fatalf("scenario %q returned empty result", sc.name)
			}
			t.Logf("%s: %v", sc.name, elapsed)
		})
	}
}

// ─── Speedup test ─────────────────────────────────────────────────────────────

// TestSpeedup is the key assertion: parallel execution must be measurably
// faster than fully-sequential execution.  We use realistic-ish delays so the
// signal dominates OS scheduling noise.
//
// Expected:
//
//	par+par  ≈  1 × (llm + tools + llm) — agents and tools overlap
//	seq+seq  ≈  6 × (llm + tools + llm) — nothing overlaps
//	speedup  ≈  6× (± scheduling jitter)
//
// We assert at least 3× to keep the test robust on loaded CI machines.
func TestSpeedup(t *testing.T) {
	const (
		llmD  = 30 * time.Millisecond
		toolD = 15 * time.Millisecond
	)

	parCfg := AnalysisConfig{LLMDelay: llmD, ToolDelay: toolD, ParallelTools: true, Strategy: herdai.StrategyParallel}
	seqCfg := AnalysisConfig{LLMDelay: llmD, ToolDelay: toolD, ParallelTools: false, Strategy: herdai.StrategySequential}

	ctx := context.Background()

	_, parTime, err := RunAnalysis(ctx, "Corp", parCfg)
	if err != nil {
		t.Fatalf("parallel run: %v", err)
	}
	_, seqTime, err := RunAnalysis(ctx, "Corp", seqCfg)
	if err != nil {
		t.Fatalf("sequential run: %v", err)
	}

	speedup := float64(seqTime) / float64(parTime)
	t.Logf("parallel: %v  |  sequential: %v  |  speedup: %.2fx", parTime, seqTime, speedup)

	if speedup < 3.0 {
		t.Errorf("expected at least 3x speedup, got %.2fx  (par=%v  seq=%v)", speedup, parTime, seqTime)
	}
}

// TestParallelToolsAreActuallyParallel measures tool execution directly to
// confirm that with ParallelToolCalls=true the wall time is close to one
// tool's latency, not four tools' latency summed.
func TestParallelToolsAreActuallyParallel(t *testing.T) {
	const toolDelay = 50 * time.Millisecond
	const toolCount = 4

	// Build one agent with 4 tools that each sleep for toolDelay.
	tools := make([]herdai.Tool, toolCount)
	calls := make([]herdai.ToolCall, toolCount)
	for i := 0; i < toolCount; i++ {
		name := fmt.Sprintf("slow_tool_%d", i+1)
		delay := toolDelay
		tools[i] = herdai.Tool{
			Name:        name,
			Description: "Simulated slow API",
			Execute: func(ctx context.Context, args map[string]any) (string, error) {
				time.Sleep(delay)
				return name + " done", nil
			},
		}
		calls[i] = herdai.ToolCall{ID: fmt.Sprintf("c%d", i+1), Function: name, Args: map[string]any{}}
	}

	mock := herdai.NewMockLLM(
		herdai.MockResponse{ToolCalls: calls},
		herdai.MockResponse{Content: "all done"},
	)

	parallelOn := true
	agent := herdai.NewAgent(herdai.AgentConfig{
		ID:                "tool-bench-agent",
		Role:              "Tester",
		Goal:              "Call all tools",
		Tools:             tools,
		LLM:               mock,
		ParallelToolCalls: &parallelOn,
	})

	start := time.Now()
	result, err := agent.Run(context.Background(), "go", nil)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Content != "all done" {
		t.Fatalf("unexpected content: %s", result.Content)
	}

	// With full parallelism: wall time ≈ toolDelay (not toolCount × toolDelay).
	maxExpected := toolDelay * 2 // generous headroom for scheduling
	t.Logf("4 tools × %v = %v serial; parallel elapsed: %v", toolDelay, toolDelay*toolCount, elapsed)
	if elapsed > maxExpected {
		t.Errorf("tools ran too slowly: %v > %v — likely executing sequentially", elapsed, maxExpected)
	}
}

// ─── Race-detector test ───────────────────────────────────────────────────────

// TestConcurrentRunsNoRace launches several RunAnalysis calls at the same time
// to stress the race detector.  Run with:  go test -race ./...
//
// This validates two claims:
//  1. Multiple independent analyses can run concurrently (no global state).
//  2. Inside each analysis, goroutines share no unprotected mutable state.
func TestConcurrentRunsNoRace(t *testing.T) {
	const concurrency = 8
	cfg := AnalysisConfig{
		LLMDelay:      5 * time.Millisecond,
		ToolDelay:     3 * time.Millisecond,
		ParallelTools: true,
		Strategy:      herdai.StrategyParallel,
	}

	var wg sync.WaitGroup
	errs := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		company := fmt.Sprintf("Company-%d", i)
		go func() {
			defer wg.Done()
			_, _, err := RunAnalysis(context.Background(), company, cfg)
			if err != nil {
				errs <- err
			}
		}()
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("concurrent run failed: %v", err)
	}
}

// TestContextCancellation confirms that a cancelled context stops the pipeline
// promptly — important for timeout handling in production services.
func TestContextCancellation(t *testing.T) {
	cfg := AnalysisConfig{
		LLMDelay:      500 * time.Millisecond, // long enough to hit the cancel
		ToolDelay:     500 * time.Millisecond,
		ParallelTools: true,
		Strategy:      herdai.StrategyParallel,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, _, err := RunAnalysis(ctx, "Corp", cfg)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected an error due to context cancellation, got nil")
	}
	if elapsed > 500*time.Millisecond {
		t.Errorf("context cancellation took too long: %v (expected < 500ms)", elapsed)
	}
	t.Logf("cancelled after %v with: %v", elapsed, err)
}

// ─── Go micro-benchmarks (testing.B) ─────────────────────────────────────────
// Run: go test -bench=. -benchtime=5s -count=3
//
// Interpret ns/op as wall-clock time per full 6-agent pipeline run.
// The ratio  BenchmarkSequentialAgents_SequentialTools / BenchmarkParallelAgents_ParallelTools
// is the concurrency speedup — compare this to the theoretical max (≈ agentCount).

func BenchmarkParallelAgents_ParallelTools(b *testing.B) {
	cfg := AnalysisConfig{
		LLMDelay:      benchLLMDelay,
		ToolDelay:     benchToolDelay,
		ParallelTools: true,
		Strategy:      herdai.StrategyParallel,
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := RunAnalysis(ctx, "BenchCorp", cfg)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkParallelAgents_SequentialTools(b *testing.B) {
	cfg := AnalysisConfig{
		LLMDelay:      benchLLMDelay,
		ToolDelay:     benchToolDelay,
		ParallelTools: false,
		Strategy:      herdai.StrategyParallel,
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := RunAnalysis(ctx, "BenchCorp", cfg)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSequentialAgents_ParallelTools(b *testing.B) {
	cfg := AnalysisConfig{
		LLMDelay:      benchLLMDelay,
		ToolDelay:     benchToolDelay,
		ParallelTools: true,
		Strategy:      herdai.StrategySequential,
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := RunAnalysis(ctx, "BenchCorp", cfg)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSequentialAgents_SequentialTools(b *testing.B) {
	cfg := AnalysisConfig{
		LLMDelay:      benchLLMDelay,
		ToolDelay:     benchToolDelay,
		ParallelTools: false,
		Strategy:      herdai.StrategySequential,
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := RunAnalysis(ctx, "BenchCorp", cfg)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkScaleAgents sweeps the number of agents to show that parallel
// wall time stays roughly flat while sequential grows linearly — the classic
// O(1) vs O(n) story for goroutine fan-out.
func BenchmarkScaleAgents_Parallel(b *testing.B) {
	for _, n := range []int{1, 2, 4, 6} {
		n := n
		b.Run(fmt.Sprintf("agents=%d", n), func(b *testing.B) {
			blueprints := competitorIntelBlueprints()
			if n > len(blueprints) {
				n = len(blueprints)
			}
			limited := blueprints[:n]

			cfg := AnalysisConfig{
				LLMDelay:      benchLLMDelay,
				ToolDelay:     benchToolDelay,
				ParallelTools: true,
				Strategy:      herdai.StrategyParallel,
			}
			ctx := context.Background()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				agents := make([]herdai.Runnable, len(limited))
				for j, bp := range limited {
					agents[j] = buildAgent(bp, cfg)
				}
				mgr := herdai.NewManager(herdai.ManagerConfig{
					ID:       "scale-team",
					Strategy: herdai.StrategyParallel,
					Agents:   agents,
				})
				_, err := mgr.Run(ctx, "Analyze BenchCorp", nil)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
