// What this example does
//
// Benchmark (not a tutorial): measures wall time for (1) a parallel vs sequential
// Manager over N agents with simulated LLM delay, and (2) parallel vs sequential
// tool execution inside one Agent. Use it to see that HerdAI can overlap work; compare
// to Python stacks where work is often serialized.
//
// Run: go run .
// Run: go run . -quiet
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/herdai-golang/herdai"
)

func main() {
	quiet := flag.Bool("quiet", false, "suppress non-result output")
	nAgents := flag.Int("agents", 6, "number of parallel specialist agents")
	toolCount := flag.Int("tools", 4, "number of tools invoked in one LLM response")
	agentDelay := flag.Duration("agent-delay", 50*time.Millisecond, "simulated LLM latency per agent")
	toolDelay := flag.Duration("tool-delay", 40*time.Millisecond, "simulated I/O latency per tool")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))

	if !*quiet {
		fmt.Println("HerdAI concurrency benchmark (no API key — MockLLM + simulated latency)")
		fmt.Println("Compare: parallel wall time should track max(latency), not sum(latency).")
		fmt.Println()
	}

	// --- 1) Manager: parallel vs sequential agents ---
	parAgents, seqAgents := benchManagerAgents(log, *nAgents, *agentDelay)
	if !*quiet {
		fmt.Printf("--- Multi-agent manager (%d agents, %s LLM delay each) ---\n", *nAgents, agentDelay)
		fmt.Printf("  StrategyParallel:   %v\n", parAgents)
		fmt.Printf("  StrategySequential: %v\n", seqAgents)
		if seqAgents > 0 {
			fmt.Printf("  speedup (seq/par):  %.2fx\n", float64(seqAgents)/float64(parAgents))
		}
		fmt.Println()
	}

	// --- 2) Agent: parallel vs sequential tool calls ---
	parTools, seqTools := benchAgentTools(log, *toolCount, *toolDelay)
	if !*quiet {
		fmt.Printf("--- Single agent, %d tools in one LLM response (%s each) ---\n", *toolCount, toolDelay)
		fmt.Printf("  ParallelToolCalls=true:  %v\n", parTools)
		fmt.Printf("  ParallelToolCalls=false: %v\n", seqTools)
		if seqTools > 0 {
			fmt.Printf("  speedup (seq/par):       %.2fx\n", float64(seqTools)/float64(parTools))
		}
		fmt.Println()
	}

	// Minimal machine-readable line for scripts
	fmt.Printf("RESULTS agents_parallel=%d agents_sequential=%d tools_parallel=%d tools_sequential=%d\n",
		parAgents.Milliseconds(), seqAgents.Milliseconds(),
		parTools.Milliseconds(), seqTools.Milliseconds(),
	)
}

// delayLLM wraps an LLM and adds a fixed delay per Chat (simulates network/model latency).
type delayLLM struct {
	inner herdai.LLM
	delay time.Duration
}

func (d *delayLLM) Chat(ctx context.Context, messages []herdai.Message, tools []herdai.Tool) (*herdai.LLMResponse, error) {
	time.Sleep(d.delay)
	return d.inner.Chat(ctx, messages, tools)
}

func benchManagerAgents(log *slog.Logger, n int, delay time.Duration) (parallel, sequential time.Duration) {
	agents := make([]herdai.Runnable, n)
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("analyst-%d", i+1)
		mock := herdai.NewMockLLM(herdai.MockResponse{Content: fmt.Sprintf("[%s] analysis complete", id)})
		agents[i] = herdai.NewAgent(herdai.AgentConfig{
			ID:     id,
			Role:   "Analyst",
			Goal:   "Produce a short analysis fragment",
			LLM:    &delayLLM{inner: mock, delay: delay},
			Logger: log,
		})
	}

	ctx := context.Background()
	input := "Analyze market signals for benchmarking."

	parallelMgr := herdai.NewManager(herdai.ManagerConfig{
		ID:       "parallel-team",
		Strategy: herdai.StrategyParallel,
		Agents:   agents,
		Logger:   log,
	})
	t0 := time.Now()
	_, err := parallelMgr.Run(ctx, input, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parallel manager: %v\n", err)
		os.Exit(1)
	}
	parallel = time.Since(t0)

	seqAgents := make([]herdai.Runnable, n)
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("analyst-%d", i+1)
		mock := herdai.NewMockLLM(herdai.MockResponse{Content: fmt.Sprintf("[%s] analysis complete", id)})
		seqAgents[i] = herdai.NewAgent(herdai.AgentConfig{
			ID:     id,
			Role:   "Analyst",
			Goal:   "Produce a short analysis fragment",
			LLM:    &delayLLM{inner: mock, delay: delay},
			Logger: log,
		})
	}
	seqMgr := herdai.NewManager(herdai.ManagerConfig{
		ID:       "sequential-team",
		Strategy: herdai.StrategySequential,
		Agents:   seqAgents,
		Logger:   log,
	})
	t1 := time.Now()
	_, err = seqMgr.Run(ctx, input, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sequential manager: %v\n", err)
		os.Exit(1)
	}
	sequential = time.Since(t1)

	return parallel, sequential
}

func benchAgentTools(log *slog.Logger, n int, delay time.Duration) (parallel, sequential time.Duration) {
	if n < 2 {
		n = 2
	}

	makeTool := func(name string) herdai.Tool {
		return herdai.Tool{
			Name:        name,
			Description: "Simulated API / retrieval",
			Execute: func(ctx context.Context, args map[string]any) (string, error) {
				time.Sleep(delay)
				return name + ": ok", nil
			},
		}
	}

	tools := make([]herdai.Tool, n)
	calls := make([]herdai.ToolCall, n)
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("fetch_%d", i+1)
		tools[i] = makeTool(name)
		calls[i] = herdai.ToolCall{
			ID:       fmt.Sprintf("call_%d", i+1),
			Function: name,
			Args:     map[string]any{},
		}
	}

	mockResponses := []herdai.MockResponse{
		{ToolCalls: calls},
		{Content: "Synthesized from all parallel fetches."},
	}

	ctx := context.Background()

	mockPar := herdai.NewMockLLM(mockResponses...)
	parallelOn := true
	agentPar := herdai.NewAgent(herdai.AgentConfig{
		ID:                "researcher",
		Role:              "Researcher",
		Goal:              "Gather sources",
		Tools:             tools,
		LLM:               mockPar,
		ParallelToolCalls: &parallelOn,
		Logger:            log,
	})
	t0 := time.Now()
	_, err := agentPar.Run(ctx, "Gather data", nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parallel tools: %v\n", err)
		os.Exit(1)
	}
	parallel = time.Since(t0)

	mockSeq := herdai.NewMockLLM(mockResponses...)
	parallelOff := false
	agentSeq := herdai.NewAgent(herdai.AgentConfig{
		ID:                "researcher-seq",
		Role:              "Researcher",
		Goal:              "Gather sources",
		Tools:             tools,
		LLM:               mockSeq,
		ParallelToolCalls: &parallelOff,
		Logger:            log,
	})
	t1 := time.Now()
	_, err = agentSeq.Run(ctx, "Gather data", nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sequential tools: %v\n", err)
		os.Exit(1)
	}
	sequential = time.Since(t1)

	return parallel, sequential
}
