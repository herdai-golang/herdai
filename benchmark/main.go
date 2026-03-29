// Competitive Intelligence Engine — HerdAI concurrency benchmark demo.
//
// Runs the same 6-agent, 18-tool pipeline in all four parallelism combinations
// so you can see wall-clock time shrink as more goroutine concurrency is
// enabled.  No API key needed — all LLM calls and tool calls are simulated
// with realistic latency via time.Sleep.
//
// Usage:
//
//	go run . [company]
//	go run . --llm-delay 200ms --tool-delay 80ms Stripe
//
// To run formal Go benchmarks:
//
//	go test -bench=. -benchtime=5s -count=3
//
// To run with the race detector (proves goroutine safety):
//
//	go test -race ./...
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/herdai-golang/herdai"
)

func main() {
	llmDelay := flag.Duration("llm-delay", 200*time.Millisecond,
		"Simulated LLM API latency per agent (represents one round-trip to OpenAI / Mistral / etc.)")
	toolDelay := flag.Duration("tool-delay", 80*time.Millisecond,
		"Simulated external tool latency per call (represents one HTTP/DB/scrape round-trip)")
	flag.Parse()

	company := "Stripe"
	if flag.NArg() > 0 {
		company = strings.Join(flag.Args(), " ")
	}

	printBanner(company, *llmDelay, *toolDelay)

	// The 6 specialist agents, the total tool count, and how time is dominated by latency.
	blueprints := competitorIntelBlueprints()
	totalTools := 0
	for _, bp := range blueprints {
		totalTools += len(bp.toolDefs)
	}
	fmt.Printf("  Pipeline: %d specialist agents × (1 LLM call + up to %d tools + 1 LLM call)\n",
		len(blueprints), totalTools/len(blueprints))
	fmt.Printf("  Total tool calls per run: %d\n\n", totalTools)

	type scenario struct {
		tag           string
		label         string
		parallelTools bool
		strategy      herdai.Strategy
	}

	scenarios := []scenario{
		{
			"par_par",
			"Parallel agents + Parallel tools   ← HerdAI native goroutine power",
			true, herdai.StrategyParallel,
		},
		{
			"par_seq",
			"Parallel agents + Sequential tools  (agents concurrent, tools serial)",
			false, herdai.StrategyParallel,
		},
		{
			"seq_par",
			"Sequential agents + Parallel tools  (one agent at a time, tools concurrent)",
			true, herdai.StrategySequential,
		},
		{
			"seq_seq",
			"Sequential agents + Sequential tools ← Python GIL worst-case analog",
			false, herdai.StrategySequential,
		},
	}

	results := make([]runResult, 0, len(scenarios))

	for i, sc := range scenarios {
		fmt.Printf("  [%d/4] %s\n", i+1, sc.label)

		cfg := AnalysisConfig{
			LLMDelay:      *llmDelay,
			ToolDelay:     *toolDelay,
			ParallelTools: sc.parallelTools,
			Strategy:      sc.strategy,
		}

		_, elapsed, err := RunAnalysis(context.Background(), company, cfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "       ✗  error: %v\n\n", err)
			os.Exit(1)
		}
		fmt.Printf("       ✓  completed in %v\n\n", elapsed.Round(time.Millisecond))
		results = append(results, runResult{sc.tag, sc.label, elapsed})
	}

	printSummary(results, *llmDelay, *toolDelay, len(blueprints), totalTools)
}

// ─── Output helpers ───────────────────────────────────────────────────────────

func printBanner(company string, llmDelay, toolDelay time.Duration) {
	line := strings.Repeat("═", 70)
	fmt.Println(line)
	fmt.Println("  HerdAI  ·  Competitive Intelligence Engine  ·  Concurrency Benchmark")
	fmt.Println(line)
	fmt.Printf("  Company      : %s\n", company)
	fmt.Printf("  LLM latency  : %v / agent (simulated API round-trip)\n", llmDelay)
	fmt.Printf("  Tool latency : %v / call  (simulated HTTP / DB / scrape)\n", toolDelay)
	fmt.Println()
	fmt.Println("  What this proves:")
	fmt.Println("  • Go goroutines let HerdAI run all agents simultaneously in one process.")
	fmt.Println("  • Python GIL prevents the same within a single process;")
	fmt.Println("    CrewAI / AutoGen / LangGraph rely on async or multiprocessing instead.")
	fmt.Println()
	fmt.Println("  Running four scenarios …")
	fmt.Println()
}

type runResult struct {
	tag     string
	label   string
	elapsed time.Duration
}

func printSummary(results []runResult, llmDelay, toolDelay time.Duration, agentCount, toolCount int) {
	if len(results) == 0 {
		return
	}

	line := strings.Repeat("─", 70)
	fmt.Println(line)
	fmt.Println("  RESULTS")
	fmt.Println(line)

	baseline := results[0].elapsed

	for _, r := range results {
		mult := float64(r.elapsed) / float64(baseline)
		if mult < 1.05 {
			fmt.Printf("  %-54s  %6v\n", r.label, r.elapsed.Round(time.Millisecond))
		} else {
			fmt.Printf("  %-54s  %6v   (%.1fx slower)\n",
				r.label, r.elapsed.Round(time.Millisecond), mult)
		}
	}

	worst := results[len(results)-1].elapsed
	speedup := float64(worst) / float64(baseline)

	fmt.Println()

	// Theoretical expectations
	perAgent := llmDelay*2 + toolDelay // 2 LLM calls + 1 parallel tool batch
	thParPar := perAgent               // all agents overlap → cost of one
	thSeqPar := time.Duration(agentCount) * perAgent
	thParSeq := llmDelay*2 + time.Duration(toolCount/agentCount)*toolDelay
	thSeqSeq := time.Duration(agentCount) * (llmDelay*2 + time.Duration(toolCount/agentCount)*toolDelay)

	fmt.Println("  Theoretical minimums (ideal scheduling, no OS jitter):")
	fmt.Printf("    par+par   %v  (wall ≈ one agent's cost)\n", thParPar)
	fmt.Printf("    par+seq   %v  (wall ≈ one agent's cost, tools serial)\n", thParSeq)
	fmt.Printf("    seq+par   %v  (agents add up, tools overlap)\n", thSeqPar)
	fmt.Printf("    seq+seq   %v  (everything adds up)\n", thSeqSeq)
	fmt.Println()
	fmt.Printf("  Measured speedup  (seq+seq ÷ par+par): %.1fx\n", speedup)
	fmt.Println()
	fmt.Println("  This gap grows with more agents and more tools per agent.")
	fmt.Println("  Compare to Python: the GIL forces threads to share one CPU core;")
	fmt.Println("  true parallelism requires spawning separate OS processes, which adds")
	fmt.Println("  startup cost, IPC overhead, and memory duplication.")
	fmt.Println()
	fmt.Println("  HerdAI goroutines share memory safely, start in microseconds, and")
	fmt.Println("  are scheduled across all available CPU cores by the Go runtime.")
	fmt.Println(line)
}
