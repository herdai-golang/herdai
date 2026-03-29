package herdai

import (
	"context"
	"crypto/rand"
	"encoding/hex"
)

// Runnable is the core interface implemented by both Agent and Manager.
// This enables hierarchical composition: a Manager's agents can be other Managers.
type Runnable interface {
	GetID() string
	Run(ctx context.Context, input string, conv *Conversation) (*Result, error)
}

// Result holds the output of an Agent or Manager execution.
type Result struct {
	AgentID  string         `json:"agent_id"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func generateID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
