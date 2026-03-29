package herdai

import "context"

// ToolHandler executes a tool with the given arguments and returns a string result.
type ToolHandler func(ctx context.Context, args map[string]any) (string, error)

// ToolParam describes a single parameter for a tool (used to generate JSON Schema for the LLM).
type ToolParam struct {
	Name        string `json:"name"`
	Type        string `json:"type"` // "string", "number", "boolean", "object", "array"
	Description string `json:"description"`
	Required    bool   `json:"required"`
}

// Tool represents a capability available to an Agent.
// Name and Description are sent to the LLM so it knows when to invoke the tool.
// Execute is called by the agent runtime when the LLM requests a tool call.
type Tool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  []ToolParam `json:"parameters"`
	Execute     ToolHandler `json:"-"`
}
