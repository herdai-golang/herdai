package herdai

import "context"

// Role represents a message role in the conversation.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message represents a single message in the LLM conversation.
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolCall represents the LLM's request to invoke a tool.
type ToolCall struct {
	ID       string         `json:"id"`
	Function string         `json:"function"`
	Args     map[string]any `json:"arguments"`
}

// LLMResponse is what the LLM returns after a Chat call.
type LLMResponse struct {
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// LLM is the interface for language model providers.
// Implement this to plug in OpenAI, Anthropic, local models, or mocks.
type LLM interface {
	Chat(ctx context.Context, messages []Message, tools []Tool) (*LLMResponse, error)
}

// StreamingLLM is an optional extension for providers that can stream
// partial output tokens. Implementations should invoke onToken for each
// incremental token/chunk and return the final assembled response.
type StreamingLLM interface {
	LLM
	ChatStream(ctx context.Context, messages []Message, tools []Tool, onToken func(string)) (*LLMResponse, error)
}
