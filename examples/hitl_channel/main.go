// What this example does
//
// Demonstrates Human-in-the-Loop (HITL) gating of tool calls using ChannelHITLHandler,
// which is what you wire to a WebSocket, desktop UI, or ticket queue (not stdin).
//
// Where HITL appears in the execution flow
//
//  1. AgentConfig.HITL installs a HITLController with your Handler and Policy.
//  2. The mock LLM returns a tool call for delete_file (simulating the model deciding to delete).
//  3. Before Execute runs, the runtime checks HITLPolicyDangerous + DangerousTools — delete_file matches.
//  4. RequestApproval calls your Handler: ChannelHITLHandler sends the request on reqCh and blocks until respCh.
//  5. The goroutine below plays “the human/UI”: it prints the pending tool and args, then sends HITLApprove.
//  6. Only after approval does the tool Execute run; then the LLM is called again for the final message.
//
// If you rejected (HITLReject) or aborted (HITLAbort), the tool would not run (or the run would stop).
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

	// --- HITL “human / UI” side: approve or deny what the agent wants to run. ---
	go func() {
		// Block 1: agent hit a dangerous tool → request arrives here (would be your WebSocket write).
		req := <-reqCh
		fmt.Printf("[HITL] pending approval: tool=%q agent=%s args=%v\n", req.ToolName, req.AgentID, req.Args)
		fmt.Println("[HITL] decision: APPROVE (tool will now execute)")
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
		// HITL: only dangerous tools listed here pause for the handler above.
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
