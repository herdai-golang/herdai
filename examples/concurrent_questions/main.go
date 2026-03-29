// Concurrent questions: many goroutines ask the same agent *configuration* at once.
//
// Race safety: each goroutine uses its own Agent and MockLLM. A single Agent with a
// shared MockLLM would interleave queued mock responses and break runs. Each goroutine
// also uses its own Conversation so transcripts stay isolated.
//
// Verify with: go test -race .
//
// Run: go run .
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"

	"github.com/herdai-golang/herdai"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))

	const workers = 8
	ctx := context.Background()
	var wg sync.WaitGroup
	errs := make(chan error, workers)

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			// Dedicated LLM queue per request — safe under concurrency.
			mock := herdai.NewMockLLM(herdai.MockResponse{
				Content: fmt.Sprintf("Handled question %d in isolation.", id),
			})
			agent := herdai.NewAgent(herdai.AgentConfig{
				ID:     fmt.Sprintf("assistant-%d", id),
				Role:   "Assistant",
				Goal:   "Answer briefly.",
				LLM:    mock,
				Logger: log,
			})
			conv := herdai.NewConversation()
			q := fmt.Sprintf("What is the answer for concurrent request %d?", id)
			res, err := agent.Run(ctx, q, conv)
			if err != nil {
				errs <- err
				return
			}
			fmt.Printf("[%s] %s\n", conv.ID(), res.Content)
		}(i)
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			panic(err)
		}
	}
}
