package herdai

import (
	"sync"
	"time"
)

// ToolCallRecord captures the details of a single tool invocation within a turn.
type ToolCallRecord struct {
	ToolName string         `json:"tool_name"`
	Input    map[string]any `json:"input"`
	Output   string         `json:"output"`
	Duration time.Duration  `json:"duration_ms"`
	Error    string         `json:"error,omitempty"`
	Cached   bool           `json:"cached,omitempty"`
}

// Turn represents one step in the conversation transcript.
type Turn struct {
	ID        string           `json:"id"`
	AgentID   string           `json:"agent_id"`
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []ToolCallRecord `json:"tool_calls,omitempty"`
	Timestamp time.Time        `json:"timestamp"`
}

// Conversation is a thread-safe transcript of all turns.
// It is shared across agents and the manager within a single run.
type Conversation struct {
	mu    sync.RWMutex
	id    string
	turns []Turn
}

// NewConversation creates an empty conversation with a unique ID.
func NewConversation() *Conversation {
	return &Conversation{
		id:    generateID(),
		turns: make([]Turn, 0),
	}
}

// ID returns the conversation's unique identifier.
func (c *Conversation) ID() string {
	return c.id
}

// AddTurn appends a turn to the transcript (thread-safe).
func (c *Conversation) AddTurn(t Turn) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if t.ID == "" {
		t.ID = generateID()
	}
	if t.Timestamp.IsZero() {
		t.Timestamp = time.Now()
	}
	c.turns = append(c.turns, t)
}

// GetTurns returns a copy of all turns (thread-safe).
func (c *Conversation) GetTurns() []Turn {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]Turn, len(c.turns))
	copy(out, c.turns)
	return out
}

// LastN returns the last n turns (or all turns if fewer than n exist).
func (c *Conversation) LastN(n int) []Turn {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if n >= len(c.turns) {
		out := make([]Turn, len(c.turns))
		copy(out, c.turns)
		return out
	}
	out := make([]Turn, n)
	copy(out, c.turns[len(c.turns)-n:])
	return out
}

// Len returns the number of turns.
func (c *Conversation) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.turns)
}
