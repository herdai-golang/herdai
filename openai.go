package herdai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"
)

// OpenAIConfig holds configuration for the OpenAI LLM client.
type OpenAIConfig struct {
	APIKey      string
	Model       string
	BaseURL     string
	HTTPClient  *http.Client
	Temperature float64
	// Logger receives HTTP-level events (e.g. 429 retries). Optional.
	Logger *slog.Logger
}

// OpenAI implements the LLM interface using the OpenAI Chat Completions API.
type OpenAI struct {
	apiKey      string
	model       string
	baseURL     string
	client      *http.Client
	temperature float64
	logger      *slog.Logger
}

// NewOpenAI creates an OpenAI client with sensible defaults.
// If APIKey is empty, it reads from OPENAI_API_KEY environment variable.
func NewOpenAI(cfg OpenAIConfig) *OpenAI {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("OPENAI_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = &http.Client{Timeout: 60 * time.Second}
	}
	if cfg.Temperature == 0 {
		cfg.Temperature = 0.7
	}
	return &OpenAI{
		apiKey:      cfg.APIKey,
		model:       cfg.Model,
		baseURL:     cfg.BaseURL,
		client:      cfg.HTTPClient,
		temperature: cfg.Temperature,
		logger:      cfg.Logger,
	}
}

// NewMistral creates an LLM client for Mistral AI.
// Mistral's API is OpenAI-compatible, so this uses the same client with Mistral's base URL.
// If APIKey is empty, it reads from MISTRAL_API_KEY environment variable.
func NewMistral(cfg OpenAIConfig) *OpenAI {
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("MISTRAL_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "mistral-small-latest"
	}
	cfg.BaseURL = "https://api.mistral.ai/v1"
	return NewOpenAI(cfg)
}

// --- OpenAI API request/response types ---

type oaiRequest struct {
	Model       string       `json:"model"`
	Messages    []oaiMessage `json:"messages"`
	Tools       []oaiTool    `json:"tools,omitempty"`
	Temperature float64      `json:"temperature"`
}

type oaiMessage struct {
	Role       string        `json:"role"`
	Content    *string       `json:"content"`
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	Name       string        `json:"name,omitempty"`
}

type oaiToolCall struct {
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function oaiToolFunction `json:"function"`
}

type oaiToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type oaiTool struct {
	Type     string    `json:"type"`
	Function oaiToolDef `json:"function"`
}

type oaiToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type oaiResponse struct {
	Choices []oaiChoice `json:"choices"`
	Error   *oaiError   `json:"error,omitempty"`
}

type oaiChoice struct {
	Message oaiResponseMessage `json:"message"`
}

// oaiResponseMessage is used only when unmarshaling API responses. The "content"
// field may be a string, null, or an array of parts (OpenAI multimodal / some
// Mistral responses) — unmarshaling into *string fails on arrays.
type oaiResponseMessage struct {
	Role       string          `json:"role"`
	Content    oaiFlexContent  `json:"content"`
	ToolCalls  []oaiToolCall   `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

// oaiFlexContent accepts JSON null, a string, or an array of content blocks
// like [{"type":"text","text":"..."}].
type oaiFlexContent struct {
	Text string
}

func (f *oaiFlexContent) UnmarshalJSON(b []byte) error {
	f.Text = ""
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		f.Text = s
		return nil
	}
	var parts []map[string]any
	if err := json.Unmarshal(b, &parts); err != nil {
		return nil // leave empty; caller may still have tool_calls
	}
	var sb strings.Builder
	for _, p := range parts {
		if t, ok := p["text"].(string); ok {
			sb.WriteString(t)
			continue
		}
		// Some providers nest content
		if nested, ok := p["content"].(string); ok {
			sb.WriteString(nested)
		}
	}
	f.Text = sb.String()
	return nil
}

type oaiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

// Chat sends a request to the OpenAI Chat Completions API.
func (o *OpenAI) Chat(ctx context.Context, messages []Message, tools []Tool) (*LLMResponse, error) {
	oaiMessages := make([]oaiMessage, 0, len(messages))
	for _, m := range messages {
		msg := oaiMessage{
			ToolCallID: m.ToolCallID,
			Name:       m.Name,
		}
		if m.Content != "" || len(m.ToolCalls) == 0 {
			s := m.Content
			msg.Content = &s
		}
		msg.Role = string(m.Role)

		for _, tc := range m.ToolCalls {
			argsJSON, _ := json.Marshal(tc.Args)
			msg.ToolCalls = append(msg.ToolCalls, oaiToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: oaiToolFunction{
					Name:      tc.Function,
					Arguments: string(argsJSON),
				},
			})
		}
		oaiMessages = append(oaiMessages, msg)
	}

	var oaiTools []oaiTool
	for _, t := range tools {
		props := make(map[string]any)
		required := make([]string, 0)
		for _, p := range t.Parameters {
			props[p.Name] = map[string]any{
				"type":        p.Type,
				"description": p.Description,
			}
			if p.Required {
				required = append(required, p.Name)
			}
		}
		oaiTools = append(oaiTools, oaiTool{
			Type: "function",
			Function: oaiToolDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters: map[string]any{
					"type":       "object",
					"properties": props,
					"required":   required,
				},
			},
		})
	}

	reqBody := oaiRequest{
		Model:       o.model,
		Messages:    oaiMessages,
		Temperature: o.temperature,
	}
	if len(oaiTools) > 0 {
		reqBody.Tools = oaiTools
	}

	bodyJSON, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	var resp *http.Response
	var respBody []byte

	maxRetries := 5
	lastAttempt := 0
	for attempt := 0; attempt <= maxRetries; attempt++ {
		lastAttempt = attempt
		reqCopy, _ := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/chat/completions", bytes.NewReader(bodyJSON))
		reqCopy.Header.Set("Content-Type", "application/json")
		reqCopy.Header.Set("Authorization", "Bearer "+o.apiKey)

		resp, err = o.client.Do(reqCopy)
		if err != nil {
			return nil, fmt.Errorf("http request failed: %w", err)
		}

		respBody, err = io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("read response body: %w", err)
		}

		if resp.StatusCode == 429 && attempt < maxRetries {
			backoff := time.Duration(1<<uint(attempt)) * time.Second
			if backoff > 30*time.Second {
				backoff = 30 * time.Second
			}
			if o.logger != nil {
				o.logger.Warn("LLM HTTP 429 rate limit — sleeping before retry",
					"http_attempt", attempt+1,
					"http_attempts_max", maxRetries+1,
					"backoff", backoff.String(),
					"model", o.model,
				)
			}
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
				continue
			}
		}
		break
	}

	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusTooManyRequests && o.logger != nil {
			o.logger.Error("LLM HTTP 429 — retries exhausted for this Chat() call",
				"http_attempts", lastAttempt+1,
				"model", o.model,
			)
		}
		return nil, fmt.Errorf("openai api error (status %d): %s (http_attempts=%d) body=%s",
			resp.StatusCode, http.StatusText(resp.StatusCode), lastAttempt+1, string(respBody))
	}

	var oaiResp oaiResponse
	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}
	if oaiResp.Error != nil {
		return nil, fmt.Errorf("openai error: %s (%s)", oaiResp.Error.Message, oaiResp.Error.Type)
	}
	if len(oaiResp.Choices) == 0 {
		return nil, fmt.Errorf("openai returned no choices")
	}

	choice := oaiResp.Choices[0]
	llmResp := &LLMResponse{}
	llmResp.Content = choice.Message.Content.Text

	for _, tc := range choice.Message.ToolCalls {
		var args map[string]any
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			args = map[string]any{"_raw": tc.Function.Arguments}
		}
		llmResp.ToolCalls = append(llmResp.ToolCalls, ToolCall{
			ID:       tc.ID,
			Function: tc.Function.Name,
			Args:     args,
		})
	}

	return llmResp, nil
}

// ChatStream streams response chunks via onToken and returns the final result.
// This implementation is provider-agnostic and works even when the upstream API
// stream format differs: it first gets the final answer through Chat(), then
// emits incremental chunks to the callback.
func (o *OpenAI) ChatStream(ctx context.Context, messages []Message, tools []Tool, onToken func(string)) (*LLMResponse, error) {
	resp, err := o.Chat(ctx, messages, tools)
	if err != nil {
		return nil, err
	}
	if onToken != nil && resp.Content != "" {
		parts := strings.Fields(resp.Content)
		for i, p := range parts {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			default:
			}
			if i == 0 {
				onToken(p)
			} else {
				onToken(" " + p)
			}
		}
	}
	return resp, nil
}
