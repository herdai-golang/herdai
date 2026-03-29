package herdai

import (
	"context"
	"testing"
)

func TestHITL_AutoApproveHandler(t *testing.T) {
	handler := AutoApproveHandler()
	resp, err := handler(context.Background(), HITLRequest{
		ToolName: "dangerous_tool",
		Args:     map[string]any{"cmd": "rm -rf /"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Decision != HITLApprove {
		t.Errorf("expected approve, got %s", resp.Decision)
	}
}

func TestHITL_ChannelHandler(t *testing.T) {
	handler, respCh, reqCh := ChannelHITLHandler()

	go func() {
		req := <-reqCh
		if req.ToolName != "web_search" {
			t.Errorf("expected web_search, got %s", req.ToolName)
		}
		respCh <- HITLResponse{Decision: HITLApprove}
	}()

	resp, err := handler(context.Background(), HITLRequest{
		ToolName: "web_search",
		Args:     map[string]any{"q": "test"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Decision != HITLApprove {
		t.Errorf("expected approve, got %s", resp.Decision)
	}
}

func TestHITL_ChannelHandler_Reject(t *testing.T) {
	handler, respCh, reqCh := ChannelHITLHandler()

	go func() {
		<-reqCh
		respCh <- HITLResponse{Decision: HITLReject, Feedback: "Not safe"}
	}()

	resp, err := handler(context.Background(), HITLRequest{ToolName: "delete_file"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Decision != HITLReject {
		t.Errorf("expected reject, got %s", resp.Decision)
	}
}

func TestHITLController_PolicyNone(t *testing.T) {
	ctrl := NewHITLController(HITLConfig{
		Policy:  HITLPolicyNone,
		Handler: AutoApproveHandler(),
	})

	if ctrl.NeedsApproval(ToolCall{Function: "anything"}) {
		t.Error("policy=none should never need approval")
	}
}

func TestHITLController_PolicyAllTools(t *testing.T) {
	ctrl := NewHITLController(HITLConfig{
		Policy:  HITLPolicyAllTools,
		Handler: AutoApproveHandler(),
	})

	if !ctrl.NeedsApproval(ToolCall{Function: "search"}) {
		t.Error("policy=all should need approval for everything")
	}
}

func TestHITLController_PolicyDangerous(t *testing.T) {
	ctrl := NewHITLController(HITLConfig{
		Policy:         HITLPolicyDangerous,
		Handler:        AutoApproveHandler(),
		DangerousTools: []string{"delete_file", "execute_command"},
	})

	if !ctrl.NeedsApproval(ToolCall{Function: "delete_file"}) {
		t.Error("delete_file should need approval")
	}
	if ctrl.NeedsApproval(ToolCall{Function: "web_search"}) {
		t.Error("web_search should NOT need approval")
	}
}

func TestHITLController_PolicyCustom(t *testing.T) {
	ctrl := NewHITLController(HITLConfig{
		Policy:  HITLPolicyCustom,
		Handler: AutoApproveHandler(),
		ShouldApprove: func(tc ToolCall) bool {
			_, hasCmd := tc.Args["command"]
			return hasCmd
		},
	})

	if ctrl.NeedsApproval(ToolCall{Function: "search", Args: map[string]any{"q": "test"}}) {
		t.Error("should NOT need approval without 'command' arg")
	}
	if !ctrl.NeedsApproval(ToolCall{Function: "exec", Args: map[string]any{"command": "ls"}}) {
		t.Error("should need approval with 'command' arg")
	}
}

func TestHITLController_ApproveAll(t *testing.T) {
	callCount := 0
	handler := func(_ context.Context, _ HITLRequest) (*HITLResponse, error) {
		callCount++
		return &HITLResponse{Decision: HITLApproveAll}, nil
	}

	ctrl := NewHITLController(HITLConfig{
		Policy:  HITLPolicyAllTools,
		Handler: handler,
	})

	// First call: needs approval
	if !ctrl.NeedsApproval(ToolCall{Function: "tool1"}) {
		t.Error("first call should need approval")
	}

	_, err := ctrl.RequestApproval(context.Background(), "agent-1", ToolCall{Function: "tool1"})
	if err != nil {
		t.Fatal(err)
	}

	// After ApproveAll, subsequent calls should auto-approve
	if ctrl.NeedsApproval(ToolCall{Function: "tool2"}) {
		t.Error("after approve_all, should NOT need approval")
	}
}

func TestHITLController_History(t *testing.T) {
	ctrl := NewHITLController(HITLConfig{
		Policy:  HITLPolicyAllTools,
		Handler: AutoApproveHandler(),
	})

	_, _ = ctrl.RequestApproval(context.Background(), "a1", ToolCall{Function: "t1"})
	_, _ = ctrl.RequestApproval(context.Background(), "a1", ToolCall{Function: "t2"})

	history := ctrl.History()
	if len(history) != 2 {
		t.Fatalf("expected 2 history records, got %d", len(history))
	}
	if history[0].Request.ToolName != "t1" {
		t.Errorf("expected t1, got %s", history[0].Request.ToolName)
	}
}

func TestAgentWithHITL_Approve(t *testing.T) {
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{
			{ID: "tc1", Function: "search", Args: map[string]any{"q": "test"}},
		},
	})
	mock.PushResponse(LLMResponse{Content: "search result based on findings"})

	agent := NewAgent(AgentConfig{
		ID:   "hitl-test",
		Role: "Tester",
		Goal: "Test HITL",
		LLM:  mock,
		Tools: []Tool{
			{
				Name:        "search",
				Description: "Search the web",
				Execute: func(_ context.Context, args map[string]any) (string, error) {
					return "found: " + args["q"].(string), nil
				},
			},
		},
		HITL: &HITLConfig{
			Policy:  HITLPolicyAllTools,
			Handler: AutoApproveHandler(),
		},
	})

	result, err := agent.Run(context.Background(), "search for test", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content == "" {
		t.Fatal("expected non-empty result")
	}
}

func TestAgentWithHITL_Reject(t *testing.T) {
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{
			{ID: "tc1", Function: "dangerous", Args: map[string]any{}},
		},
	})
	mock.PushResponse(LLMResponse{Content: "ok, I won't do that"})

	rejectHandler := func(_ context.Context, _ HITLRequest) (*HITLResponse, error) {
		return &HITLResponse{Decision: HITLReject, Feedback: "too dangerous"}, nil
	}

	agent := NewAgent(AgentConfig{
		ID:   "hitl-reject",
		Role: "Tester",
		Goal: "Test rejection",
		LLM:  mock,
		Tools: []Tool{
			{
				Name:        "dangerous",
				Description: "A dangerous operation",
				Execute: func(_ context.Context, _ map[string]any) (string, error) {
					t.Fatal("tool should NOT have been executed after rejection")
					return "", nil
				},
			},
		},
		HITL: &HITLConfig{
			Policy:  HITLPolicyAllTools,
			Handler: rejectHandler,
		},
	})

	result, err := agent.Run(context.Background(), "do something dangerous", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "ok, I won't do that" {
		t.Errorf("unexpected: %s", result.Content)
	}
}

func TestAgentWithHITL_Abort(t *testing.T) {
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{
			{ID: "tc1", Function: "risky", Args: map[string]any{}},
		},
	})

	abortHandler := func(_ context.Context, _ HITLRequest) (*HITLResponse, error) {
		return &HITLResponse{Decision: HITLAbort}, nil
	}

	agent := NewAgent(AgentConfig{
		ID:   "hitl-abort",
		Role: "Tester",
		Goal: "Test abort",
		LLM:  mock,
		Tools: []Tool{
			{Name: "risky", Description: "Risky op", Execute: func(_ context.Context, _ map[string]any) (string, error) {
				return "", nil
			}},
		},
		HITL: &HITLConfig{
			Policy:  HITLPolicyAllTools,
			Handler: abortHandler,
		},
	})

	_, err := agent.Run(context.Background(), "do risky thing", nil)
	if err == nil {
		t.Fatal("expected error from abort")
	}
}

func TestHITLDecision_String(t *testing.T) {
	tests := []struct {
		d    HITLDecision
		want string
	}{
		{HITLApprove, "approve"},
		{HITLReject, "reject"},
		{HITLEdit, "edit"},
		{HITLApproveAll, "approve_all"},
		{HITLAbort, "abort"},
	}
	for _, tt := range tests {
		if got := tt.d.String(); got != tt.want {
			t.Errorf("%d.String() = %s, want %s", tt.d, got, tt.want)
		}
	}
}
