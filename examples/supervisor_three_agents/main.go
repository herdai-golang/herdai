// What this example does
//
// Multi-agent routing: a Manager uses StrategyLLMRouter so an LLM “supervisor” picks
// which specialist agent runs next (researcher, writer, qa). When finished, the router
// returns FINISH. Production uses a real LLM on ManagerConfig.LLM; this demo uses a
// MockLLM with fixed routing JSON so it runs without an API key.
//
// Run: go run .
package main

import (
	"context"
	"fmt"

	"github.com/herdai-golang/herdai"
)

func main() {
	researcherLLM := herdai.NewMockLLM(herdai.MockResponse{
		Content: "Research: key points — Go is fast, simple concurrency, strong stdlib.",
	})
	writerLLM := herdai.NewMockLLM(herdai.MockResponse{
		Content: "Draft: HerdAI fits teams that want agents without heavy Python dependencies.",
	})
	qaLLM := herdai.NewMockLLM(herdai.MockResponse{
		Content: "QA: Approved. Mention pkg.go.dev and zero third-party deps in README.",
	})

	researcher := herdai.NewAgent(herdai.AgentConfig{
		ID:   "researcher",
		Role: "Researcher",
		Goal: "Collect factual bullets for the topic.",
		LLM:  researcherLLM,
	})
	writer := herdai.NewAgent(herdai.AgentConfig{
		ID:   "writer",
		Role: "Writer",
		Goal: "Turn research into a short pitch.",
		LLM:  writerLLM,
	})
	qa := herdai.NewAgent(herdai.AgentConfig{
		ID:   "qa",
		Role: "QA reviewer",
		Goal: "Check tone and completeness.",
		LLM:  qaLLM,
	})

	// Supervisor: routes to specialists, then FINISH with a summary.
	supervisor := herdai.NewMockLLM(
		herdai.MockResponse{Content: `{"agent_id": "researcher", "instruction": "Research why Go is a good fit for AI agent backends."}`},
		herdai.MockResponse{Content: `{"agent_id": "writer", "instruction": "Write a 2-sentence pitch using the research."}`},
		herdai.MockResponse{Content: `{"agent_id": "qa", "instruction": "Review the draft for clarity."}`},
		herdai.MockResponse{Content: `{"agent_id": "FINISH", "instruction": "Pipeline complete. Ship the pitch after QA notes."}`},
	)

	mgr := herdai.NewManager(herdai.ManagerConfig{
		ID:       "team",
		Strategy: herdai.StrategyLLMRouter,
		Agents:   []herdai.Runnable{researcher, writer, qa},
		LLM:      supervisor,
		MaxTurns: 10,
	})

	result, err := mgr.Run(context.Background(), "Prepare a pitch for HerdAI on a team slide.", nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Content)
}
