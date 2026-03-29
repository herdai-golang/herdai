package herdai

import (
	"context"
	"fmt"
	"sync"
)

// MockResponse defines a single canned response for MockLLM.
type MockResponse struct {
	Content   string
	ToolCalls []ToolCall
	Error     error
}

// MockCall records one invocation of MockLLM.Chat for test assertions.
type MockCall struct {
	Messages []Message
	Tools    []Tool
}

// MockLLM is a deterministic LLM implementation for testing.
// It returns responses in order and records all calls for assertions.
type MockLLM struct {
	mu        sync.Mutex
	responses []MockResponse
	callIndex int
	Calls     []MockCall
}

// NewMockLLM creates a MockLLM that returns the given responses in order.
func NewMockLLM(responses ...MockResponse) *MockLLM {
	return &MockLLM{
		responses: responses,
		Calls:     make([]MockCall, 0),
	}
}

// Chat returns the next canned response and records the call.
func (m *MockLLM) Chat(ctx context.Context, messages []Message, tools []Tool) (*LLMResponse, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.Calls = append(m.Calls, MockCall{Messages: messages, Tools: tools})

	if m.callIndex >= len(m.responses) {
		return nil, fmt.Errorf("mock LLM exhausted: no response for call %d (have %d responses)", m.callIndex+1, len(m.responses))
	}

	resp := m.responses[m.callIndex]
	m.callIndex++

	if resp.Error != nil {
		return nil, resp.Error
	}

	return &LLMResponse{
		Content:   resp.Content,
		ToolCalls: resp.ToolCalls,
	}, nil
}

// CallCount returns the number of Chat calls made so far.
func (m *MockLLM) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.Calls)
}

// PushResponse appends a response to the queue (useful for building responses incrementally).
func (m *MockLLM) PushResponse(resp LLMResponse) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responses = append(m.responses, MockResponse{
		Content:   resp.Content,
		ToolCalls: resp.ToolCalls,
	})
}

// Reset clears recorded calls and resets the response index.
func (m *MockLLM) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callIndex = 0
	m.Calls = make([]MockCall, 0)
}
