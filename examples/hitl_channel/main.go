// Human-in-the-loop using ChannelHITLHandler: the approval path is a request channel
// (outbound to your UI) and a response channel (inbound from the human). This matches
// WebSocket or desktop apps better than stdin prompts.
//
// Run: go run .
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/herdai-golang/herdai"
)

func main() {
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))

	handler, respCh, reqCh := herdai.ChannelHITLHandler()

	// Simulate a UI layer: receive pending tool calls and send decisions.
	// In production this bridges to your front-end instead of stdout.
	go func() {
		req := <-reqCh
		fmt.Printf("[HITL] approve tool %q (agent=%s) args=%v\n", req.ToolName, req.AgentID, req.Args)
		respCh <- herdai.HITLResponse{
			Decision:    herdai.HITLApprove,
			RespondedAt: time.Now(),
		}
	}()

	mock := herdai.NewMockLLM(
		herdai.MockResponse{
			ToolCalls: []herdai.ToolCall{
				{ID: "t1", Function: "delete_file", Args: map[string]any{"path": "/tmp/old.log"}},
			},
		},
		herdai.MockResponse{Content: "File removal was approved and completed (simulated)."},
	)

	agent := herdai.NewAgent(herdai.AgentConfig{
		ID:     "file-assistant",
		Role:   "File helper",
		Goal:   "Help with file operations safely.",
		LLM:    mock,
		Logger: log,
		Tools: []herdai.Tool{
			{
				Name:        "delete_file",
				Description: "Delete a file by path",
				Execute: func(_ context.Context, args map[string]any) (string, error) {
					return fmt.Sprintf("deleted %v", args["path"]), nil
				},
			},
		},
		HITL: &herdai.HITLConfig{
			Policy:         herdai.HITLPolicyDangerous,
			DangerousTools: []string{"delete_file"},
			Handler:        handler,
		},
	})

	result, err := agent.Run(context.Background(), "Remove /tmp/old.log", nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Content)
}
