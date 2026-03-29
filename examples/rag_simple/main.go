// Minimal RAG: load a few chunks into an in-memory store, use keyword SimpleRAG,
// and answer with a mock LLM (no API key). Swap the LLM for Mistral/OpenAI in production.
//
// Run: go run .
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/herdai-golang/herdai"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
	ctx := context.Background()
	store := herdai.NewInMemoryVectorStore()
	if err := store.Add(ctx, []herdai.Chunk{
		{ID: "c1", Content: "HerdAI managers support Sequential, Parallel, RoundRobin, and LLMRouter strategies.", Source: "architecture.md"},
		{ID: "c2", Content: "Human-in-the-loop can gate dangerous tools before they run.", Source: "safety.md"},
	}); err != nil {
		panic(err)
	}

	mock := herdai.NewMockLLM(herdai.MockResponse{
		Content: "HerdAI offers four manager strategies: Sequential, Parallel, RoundRobin, and LLMRouter.",
	})

	agent := herdai.NewAgent(herdai.AgentConfig{
		ID:     "kb-assistant",
		Role:   "Knowledge assistant",
		Goal:   "Ground answers in the knowledge base when possible.",
		LLM:    mock,
		Logger: log,
		RAG:    herdai.SimpleRAG(store, 3),
	})

	result, err := agent.Run(ctx, "What manager strategies does HerdAI support?", nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Content)
}
