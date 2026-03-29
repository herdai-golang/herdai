// What this example does
//
// Smallest possible HerdAI program: create one Agent with a MockLLM, call Run once with
// no tools and no conversation history. Shows the basic Run → Result flow only.
//
// Run: go run .
//
// After publishing: go get github.com/herdai-golang/herdai@latest
package main

import (
	"context"
	"fmt"

	"github.com/herdai-golang/herdai"
)

func main() {
	mock := &herdai.MockLLM{}
	mock.PushResponse(herdai.LLMResponse{
		Content: "HerdAI is a Go library for building AI agents with tools, multi-agent managers, RAG, and more.",
	})

	agent := herdai.NewAgent(herdai.AgentConfig{
		ID:   "assistant",
		Role: "Assistant",
		Goal: "Answer briefly.",
		LLM:  mock,
	})

	result, err := agent.Run(context.Background(), "What is HerdAI?", nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Content)
}
