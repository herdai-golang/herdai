package herdai

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// HITL — Human-in-the-Loop approval system (Cursor/Copilot style)
//
// When enabled, the agent pauses before executing tool calls and waits for
// human approval. The human can:
//   - Approve the action (proceed as-is)
//   - Reject the action (skip it, agent gets a rejection message)
//   - Edit the action (modify args, agent uses edited version)
//   - Approve all remaining actions in this run (auto-approve mode)
//   - Abort the entire run
// ═══════════════════════════════════════════════════════════════════════════════

// HITLDecision is the human's response to a proposed action.
type HITLDecision int

const (
	HITLApprove    HITLDecision = iota // proceed with this action
	HITLReject                         // skip this action
	HITLEdit                           // proceed with modified args
	HITLApproveAll                     // approve this + all future actions
	HITLAbort                          // abort the entire agent run
)

func (d HITLDecision) String() string {
	switch d {
	case HITLApprove:
		return "approve"
	case HITLReject:
		return "reject"
	case HITLEdit:
		return "edit"
	case HITLApproveAll:
		return "approve_all"
	case HITLAbort:
		return "abort"
	default:
		return "unknown"
	}
}

// HITLRequest is sent to the human when approval is needed.
type HITLRequest struct {
	ID        string         `json:"id"`
	AgentID   string         `json:"agent_id"`
	ToolName  string         `json:"tool_name"`
	Args      map[string]any `json:"args"`
	Reason    string         `json:"reason,omitempty"` // why the agent wants to call this
	Timestamp time.Time      `json:"timestamp"`
}

// HITLResponse is the human's answer.
type HITLResponse struct {
	Decision    HITLDecision   `json:"decision"`
	EditedArgs  map[string]any `json:"edited_args,omitempty"` // only used with HITLEdit
	Feedback    string         `json:"feedback,omitempty"`    // optional message back to agent
	RespondedAt time.Time     `json:"responded_at"`
}

// HITLHandler is the callback the user implements to handle approval requests.
// This is where UI integration happens — CLI prompt, WebSocket, Slack bot, etc.
type HITLHandler func(ctx context.Context, req HITLRequest) (*HITLResponse, error)

// HITLPolicy controls which actions require approval.
type HITLPolicy int

const (
	HITLPolicyNone        HITLPolicy = iota // no approvals needed
	HITLPolicyAllTools                       // approve every tool call
	HITLPolicyDangerous                      // only approve tools in DangerousTools list
	HITLPolicyCustom                         // use custom ShouldApprove function
)

// HITLConfig configures human-in-the-loop behavior for an agent.
type HITLConfig struct {
	Policy         HITLPolicy            // when to ask for approval
	Handler        HITLHandler           // callback for approval requests
	DangerousTools []string              // tool names that need approval (for HITLPolicyDangerous)
	ShouldApprove  func(ToolCall) bool   // custom approval predicate (for HITLPolicyCustom)
	Timeout        time.Duration         // max time to wait for human response (default: 5min)
}

// HITLController manages the approval flow for a single agent run.
type HITLController struct {
	config      HITLConfig
	autoApprove bool
	history     []HITLRecord
	mu          sync.Mutex
}

// HITLRecord captures the full request/response pair for audit.
type HITLRecord struct {
	Request  HITLRequest  `json:"request"`
	Response HITLResponse `json:"response"`
	WaitTime time.Duration `json:"wait_time"`
}

// NewHITLController creates a controller from config.
func NewHITLController(cfg HITLConfig) *HITLController {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Minute
	}
	return &HITLController{config: cfg}
}

// NeedsApproval checks if a given tool call requires human approval.
func (h *HITLController) NeedsApproval(tc ToolCall) bool {
	if h == nil || h.config.Policy == HITLPolicyNone || h.config.Handler == nil {
		return false
	}

	h.mu.Lock()
	auto := h.autoApprove
	h.mu.Unlock()
	if auto {
		return false
	}

	switch h.config.Policy {
	case HITLPolicyAllTools:
		return true
	case HITLPolicyDangerous:
		for _, name := range h.config.DangerousTools {
			if name == tc.Function {
				return true
			}
		}
		return false
	case HITLPolicyCustom:
		if h.config.ShouldApprove != nil {
			return h.config.ShouldApprove(tc)
		}
		return false
	default:
		return false
	}
}

// RequestApproval sends the tool call to the human and waits for a decision.
// Returns the (possibly edited) args and whether to proceed.
func (h *HITLController) RequestApproval(ctx context.Context, agentID string, tc ToolCall) (*HITLResponse, error) {
	req := HITLRequest{
		ID:        generateID(),
		AgentID:   agentID,
		ToolName:  tc.Function,
		Args:      tc.Args,
		Timestamp: time.Now(),
	}

	ctx, cancel := context.WithTimeout(ctx, h.config.Timeout)
	defer cancel()

	resp, err := h.config.Handler(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("hitl handler error: %w", err)
	}
	if resp == nil {
		return &HITLResponse{Decision: HITLApprove, RespondedAt: time.Now()}, nil
	}

	resp.RespondedAt = time.Now()

	h.mu.Lock()
	h.history = append(h.history, HITLRecord{
		Request:  req,
		Response: *resp,
		WaitTime: resp.RespondedAt.Sub(req.Timestamp),
	})
	if resp.Decision == HITLApproveAll {
		h.autoApprove = true
	}
	h.mu.Unlock()

	return resp, nil
}

// History returns the audit trail of all approval decisions.
func (h *HITLController) History() []HITLRecord {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]HITLRecord, len(h.history))
	copy(out, h.history)
	return out
}

// ── CLI HITL handler — simple terminal-based approval ──────────────────────

// CLIApprovalHandler returns an HITLHandler that prompts on stdin.
// Pass a custom prompt writer for integration with different terminals.
func CLIApprovalHandler(promptFn func(req HITLRequest) string, readFn func() (string, error)) HITLHandler {
	return func(ctx context.Context, req HITLRequest) (*HITLResponse, error) {
		prompt := promptFn(req)
		fmt.Print(prompt)

		responseCh := make(chan string, 1)
		errCh := make(chan error, 1)

		go func() {
			line, err := readFn()
			if err != nil {
				errCh <- err
				return
			}
			responseCh <- line
		}()

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case err := <-errCh:
			return nil, err
		case line := <-responseCh:
			switch line {
			case "y", "yes", "approve", "":
				return &HITLResponse{Decision: HITLApprove}, nil
			case "n", "no", "reject":
				return &HITLResponse{Decision: HITLReject}, nil
			case "a", "all", "approve_all":
				return &HITLResponse{Decision: HITLApproveAll}, nil
			case "x", "abort":
				return &HITLResponse{Decision: HITLAbort}, nil
			default:
				return &HITLResponse{
					Decision: HITLEdit,
					Feedback: line,
				}, nil
			}
		}
	}
}

// ── Channel-based HITL handler — for WebSocket/UI integration ──────────────

// ChannelHITLHandler creates an HITLHandler that communicates via channels.
// Send requests to the returned request channel, read responses from response channel.
func ChannelHITLHandler() (HITLHandler, chan<- HITLResponse, <-chan HITLRequest) {
	reqCh := make(chan HITLRequest, 1)
	respCh := make(chan HITLResponse, 1)

	handler := func(ctx context.Context, req HITLRequest) (*HITLResponse, error) {
		select {
		case reqCh <- req:
		case <-ctx.Done():
			return nil, ctx.Err()
		}

		select {
		case resp := <-respCh:
			return &resp, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	return handler, respCh, reqCh
}

// AutoApproveHandler returns an HITLHandler that approves everything.
// Useful for testing.
func AutoApproveHandler() HITLHandler {
	return func(_ context.Context, _ HITLRequest) (*HITLResponse, error) {
		return &HITLResponse{Decision: HITLApprove, RespondedAt: time.Now()}, nil
	}
}
