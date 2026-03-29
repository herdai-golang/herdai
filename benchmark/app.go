// Package main implements a Competitive Intelligence Engine — a realistic
// multi-agent pipeline that showcases HerdAI's two levels of native goroutine
// concurrency:
//
//  1. Manager-level parallelism: 6 specialist agents all run at the same time
//     via StrategyParallel, so total wall time ≈ slowest agent, not sum of all.
//
//  2. Agent-level tool parallelism: when the LLM returns multiple tool calls in
//     one response, HerdAI executes them concurrently (ParallelToolCalls=true).
//
// Both layers stack: the outer Manager fans out to 6 goroutines; inside each
// goroutine the Agent fans out again across its tool calls.
//
// All LLM calls and external API calls are simulated with time.Sleep so the
// benchmark requires no real API key and the latency is deterministic.
package main

import (
	"context"
	"fmt"
	"time"

	"github.com/herdai-golang/herdai"
)

// ─── Configuration ────────────────────────────────────────────────────────────

// AnalysisConfig controls how the pipeline is parallelised and what latencies
// are simulated. Swap Strategy and ParallelTools to emulate different runtimes.
type AnalysisConfig struct {
	LLMDelay      time.Duration   // round-trip latency per LLM call (per agent)
	ToolDelay     time.Duration   // round-trip latency per external tool call
	ParallelTools bool            // run concurrent tool calls inside one agent
	Strategy      herdai.Strategy // orchestration strategy for the Manager
}

// ─── Domain blueprints ────────────────────────────────────────────────────────

// agentBlueprint is a pure-data description of one specialist agent.
// The Execute functions are wired in at build time so latency is configurable.
type agentBlueprint struct {
	id       string
	role     string
	goal     string
	toolDefs []toolDef
}

type toolDef struct {
	name        string
	description string
}

// competitorIntelBlueprints returns the 6 specialist agents used by the engine.
// They map to common real-world analyst roles: news sentiment, competitor
// profiling, financial metrics, market trends, regulatory risk, and customer
// voice. Each agent calls 2–4 external data sources.
func competitorIntelBlueprints() []agentBlueprint {
	return []agentBlueprint{
		{
			id:   "news_sentiment",
			role: "News Sentiment Analyst",
			goal: "Measure media sentiment and news coverage about the target company",
			toolDefs: []toolDef{
				{"reuters_feed", "Pull latest Reuters news articles"},
				{"bloomberg_wire", "Fetch Bloomberg financial news"},
				{"twitter_sentiment", "Aggregate Twitter/X real-time sentiment score"},
			},
		},
		{
			id:   "competitor_profile",
			role: "Competitor Intelligence Analyst",
			goal: "Profile direct competitors on funding, team size, and product positioning",
			toolDefs: []toolDef{
				{"crunchbase_lookup", "Fetch funding rounds from Crunchbase"},
				{"linkedin_company", "Get headcount and hiring signals from LinkedIn"},
				{"website_scrape", "Parse product pages and public pricing"},
				{"glassdoor_data", "Retrieve employer ratings and culture signals"},
			},
		},
		{
			id:   "financial_metrics",
			role: "Financial Metrics Analyst",
			goal: "Extract revenue, margins, and valuation multiples",
			toolDefs: []toolDef{
				{"sec_filings", "Download 10-K / 10-Q data from SEC EDGAR"},
				{"revenue_api", "Fetch ARR / revenue estimates from Visible Alpha"},
				{"market_cap_api", "Retrieve live market capitalisation"},
			},
		},
		{
			id:   "market_trends",
			role: "Market Trend Analyst",
			goal: "Identify macro trends and innovation signals in the sector",
			toolDefs: []toolDef{
				{"google_trends", "Query search-trend velocity for key terms"},
				{"industry_report", "Download latest Gartner / IDC report extracts"},
				{"patent_search", "Search recent patent filings in the domain"},
			},
		},
		{
			id:   "regulatory_scan",
			role: "Regulatory Risk Analyst",
			goal: "Flag open regulatory actions, fines, and compliance risks",
			toolDefs: []toolDef{
				{"sec_edgar_actions", "Scan SEC enforcement actions and comment letters"},
				{"global_compliance_db", "Check GDPR, CCPA, and sector-specific filings"},
			},
		},
		{
			id:   "customer_voice",
			role: "Customer Voice Analyst",
			goal: "Synthesise customer satisfaction, NPS, and switching signals",
			toolDefs: []toolDef{
				{"g2_reviews", "Scrape G2 review scores and qualitative themes"},
				{"gartner_peer_insights", "Fetch Gartner Peer Insights ratings"},
				{"app_store_data", "Aggregate iOS/Android app-store ratings and reviews"},
			},
		},
	}
}

// ─── Slow LLM wrapper ─────────────────────────────────────────────────────────

// slowLLM wraps any LLM and inserts a fixed sleep before each response,
// faithfully simulating real API round-trip latency in benchmarks.
type slowLLM struct {
	inner herdai.LLM
	delay time.Duration
}

func (s *slowLLM) Chat(ctx context.Context, msgs []herdai.Message, tools []herdai.Tool) (*herdai.LLMResponse, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(s.delay):
	}
	return s.inner.Chat(ctx, msgs, tools)
}

// ─── Agent factory ────────────────────────────────────────────────────────────

// buildAgent constructs a fully-wired herdai.Agent from a blueprint.
//
// MockLLM is pre-loaded with two responses:
//  1. A ToolCalls response — triggers every tool in the blueprint at once,
//     exactly as a real LLM would do when it decides to gather data.
//  2. A Content response — the agent's synthesised analysis after tools return.
//
// The Execute handlers sleep for cfg.ToolDelay, simulating network I/O (HTTP
// APIs, database lookups, web scraping, etc.).
func buildAgent(bp agentBlueprint, cfg AnalysisConfig) *herdai.Agent {
	tools := make([]herdai.Tool, len(bp.toolDefs))
	toolCalls := make([]herdai.ToolCall, len(bp.toolDefs))

	for i, td := range bp.toolDefs {
		toolDelay := cfg.ToolDelay
		name := td.name // capture per-iteration value for the closure

		tools[i] = herdai.Tool{
			Name:        name,
			Description: td.description,
			Parameters: []herdai.ToolParam{
				{Name: "query", Type: "string", Description: "Search or lookup query", Required: true},
			},
			Execute: func(ctx context.Context, args map[string]any) (string, error) {
				// Simulate the latency of an external HTTP / DB call.
				select {
				case <-ctx.Done():
					return "", ctx.Err()
				case <-time.After(toolDelay):
				}
				return fmt.Sprintf("%s: data retrieved ✓", name), nil
			},
		}

		toolCalls[i] = herdai.ToolCall{
			ID:       fmt.Sprintf("%s_call_%d", bp.id, i+1),
			Function: name,
			Args:     map[string]any{"query": "target-company"},
		}
	}

	// The mock LLM simulates two turns:
	//   Turn 1 — the LLM reads the task and decides to call all tools at once.
	//   Turn 2 — the LLM receives all tool results and writes the final analysis.
	mock := herdai.NewMockLLM(
		herdai.MockResponse{ToolCalls: toolCalls},
		herdai.MockResponse{
			Content: fmt.Sprintf(
				"[%s] Analysis complete: all data sources reconciled and cross-referenced.",
				bp.role,
			),
		},
	)

	parallel := cfg.ParallelTools
	return herdai.NewAgent(herdai.AgentConfig{
		ID:                bp.id,
		Role:              bp.role,
		Goal:              bp.goal,
		Tools:             tools,
		LLM:               &slowLLM{inner: mock, delay: cfg.LLMDelay},
		ParallelToolCalls: &parallel,
	})
}

// ─── Team builder ─────────────────────────────────────────────────────────────

// BuildTeam creates the Manager that orchestrates all 6 specialist agents.
// Strategy controls whether agents run in parallel (goroutines) or sequentially.
func BuildTeam(cfg AnalysisConfig) *herdai.Manager {
	blueprints := competitorIntelBlueprints()
	agents := make([]herdai.Runnable, len(blueprints))
	for i, bp := range blueprints {
		agents[i] = buildAgent(bp, cfg)
	}
	return herdai.NewManager(herdai.ManagerConfig{
		ID:       "market-intelligence-team",
		Strategy: cfg.Strategy,
		Agents:   agents,
	})
}

// ─── Entry point for both main and tests ──────────────────────────────────────

// RunAnalysis builds a fresh team and runs the full competitive intelligence
// pipeline for the given company. Returns the merged result and wall-clock
// elapsed time. Building a fresh team per call ensures MockLLM state is clean,
// which is essential for benchmark loops (b.N iterations).
func RunAnalysis(ctx context.Context, company string, cfg AnalysisConfig) (*herdai.Result, time.Duration, error) {
	team := BuildTeam(cfg)
	start := time.Now()
	result, err := team.Run(
		ctx,
		fmt.Sprintf("Perform a full competitive intelligence analysis for: %s", company),
		nil,
	)
	return result, time.Since(start), err
}
