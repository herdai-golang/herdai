package main

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/herdai-golang/herdai"
)

func TestConcurrentQuestionsRaceSafe(t *testing.T) {
	const workers = 32
	ctx := context.Background()
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			mock := herdai.NewMockLLM(herdai.MockResponse{
				Content: fmt.Sprintf("ok-%d", id),
			})
			agent := herdai.NewAgent(herdai.AgentConfig{
				ID:   fmt.Sprintf("a-%d", id),
				Role: "Assistant",
				Goal: "Answer.",
				LLM:  mock,
			})
			conv := herdai.NewConversation()
			res, err := agent.Run(ctx, "ping", conv)
			if err != nil {
				errs <- fmt.Errorf("run %d: %w", id, err)
				return
			}
			want := fmt.Sprintf("ok-%d", id)
			if res.Content != want {
				errs <- fmt.Errorf("run %d: got %q want %q", id, res.Content, want)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}
