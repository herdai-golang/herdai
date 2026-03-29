// What this example does
//
// One agent registers two tools (add, to_upper). The mock LLM first returns two tool
// calls in a single LLM turn (they may run in parallel), then a second LLM response
// with the final natural-language answer. Use this to see the tool loop end-to-end.
//
// Run: go run .
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/herdai-golang/herdai"
)

func main() {
	addTool := herdai.Tool{
		Name:        "add",
		Description: "Add two numbers and return the sum as text.",
		Parameters: []herdai.ToolParam{
			{Name: "a", Type: "number", Description: "first number", Required: true},
			{Name: "b", Type: "number", Description: "second number", Required: true},
		},
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			a, _ := toFloat(args["a"])
			b, _ := toFloat(args["b"])
			return fmt.Sprintf("%.0f", a+b), nil
		},
	}

	upperTool := herdai.Tool{
		Name:        "to_upper",
		Description: "Convert text to uppercase.",
		Parameters: []herdai.ToolParam{
			{Name: "text", Type: "string", Description: "input text", Required: true},
		},
		Execute: func(ctx context.Context, args map[string]any) (string, error) {
			s, _ := args["text"].(string)
			return strings.ToUpper(s), nil
		},
	}

	mock := herdai.NewMockLLM(
		herdai.MockResponse{
			ToolCalls: []herdai.ToolCall{
				{ID: "t1", Function: "add", Args: map[string]any{"a": 20.0, "b": 22.0}},
				{ID: "t2", Function: "to_upper", Args: map[string]any{"text": "herdai"}},
			},
		},
		herdai.MockResponse{
			Content: "Sum is 42. Uppercase brand: HERDAI. Done.",
		},
	)

	agent := herdai.NewAgent(herdai.AgentConfig{
		ID:    "helper",
		Role:  "Helper",
		Goal:  "Use tools when useful, then give a short final answer.",
		Tools: []herdai.Tool{addTool, upperTool},
		LLM:   mock,
	})

	result, err := agent.Run(context.Background(), "Add 20 and 22, and uppercase the word herdai.", nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Content)
}

func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
