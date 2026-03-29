package herdai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// MCPServerConfig describes how to connect to an MCP server.
// Set Command for stdio transport, or URL for Streamable HTTP transport.
type MCPServerConfig struct {
	Name    string            // human-readable name (e.g. "filesystem", "web-search")
	Command string            // stdio: command to run (e.g. "npx", "python")
	Args    []string          // stdio: command arguments
	Env     map[string]string // stdio: extra environment variables
	URL     string            // http: server endpoint (e.g. "http://localhost:8000/mcp")
}

// MCPTransport abstracts the communication layer to an MCP server.
// Implement this interface for custom transports (HTTP, WebSocket, etc.).
type MCPTransport interface {
	Start(ctx context.Context) error
	Send(msg json.RawMessage) error
	Receive() (json.RawMessage, error)
	Close() error
}

// --- Stdio Transport (most common for MCP) ---

// StdioTransport communicates with an MCP server process via stdin/stdout.
// Each JSON-RPC message is newline-delimited (ndjson).
type StdioTransport struct {
	command string
	args    []string
	env     map[string]string
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	scanner *bufio.Scanner
	stderr  io.ReadCloser
}

// NewStdioTransport creates a transport that runs the given command.
func NewStdioTransport(command string, args []string, env map[string]string) *StdioTransport {
	return &StdioTransport{
		command: command,
		args:    args,
		env:     env,
	}
}

func (t *StdioTransport) Start(ctx context.Context) error {
	if strings.TrimSpace(t.command) == "" {
		return fmt.Errorf("stdio MCP transport: empty command")
	}
	t.cmd = exec.CommandContext(ctx, t.command, t.args...)

	// Set up environment
	t.cmd.Env = os.Environ()
	for k, v := range t.env {
		t.cmd.Env = append(t.cmd.Env, k+"="+v)
	}

	var err error
	t.stdin, err = t.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("stdin pipe: %w", err)
	}

	stdout, err := t.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	t.scanner = bufio.NewScanner(stdout)
	t.scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer for large responses

	t.stderr, err = t.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := t.cmd.Start(); err != nil {
		return fmt.Errorf("start command '%s': %w", t.command, err)
	}

	// Drain stderr in background to prevent blocking
	go func() {
		scanner := bufio.NewScanner(t.stderr)
		for scanner.Scan() {
			// MCP servers often log to stderr; we ignore it
		}
	}()

	return nil
}

func (t *StdioTransport) Send(msg json.RawMessage) error {
	line := append(msg, '\n')
	_, err := t.stdin.Write(line)
	return err
}

func (t *StdioTransport) Receive() (json.RawMessage, error) {
	if !t.scanner.Scan() {
		if err := t.scanner.Err(); err != nil {
			return nil, fmt.Errorf("read: %w", err)
		}
		return nil, io.EOF
	}
	return json.RawMessage(t.scanner.Bytes()), nil
}

func (t *StdioTransport) Close() error {
	if t.stdin != nil {
		t.stdin.Close()
	}
	if t.cmd != nil && t.cmd.Process != nil {
		t.cmd.Process.Kill()
		t.cmd.Wait()
	}
	return nil
}

// --- Streamable HTTP Transport ---

// HTTPTransport communicates with an MCP server over HTTP using the
// Streamable HTTP transport (JSON-RPC over POST, SSE responses).
type HTTPTransport struct {
	url       string
	sessionID string
	mu        sync.Mutex
	client    *http.Client
	pending   chan json.RawMessage
	closed    bool
	ctx       context.Context // set by Start(); propagated to every HTTP request
}

// NewHTTPTransport creates a transport that talks to an MCP server over HTTP.
// Per-request timeout is short (15s) so a hanging MCP server fails fast
// instead of blocking the whole agent for minutes.
func NewHTTPTransport(url string) *HTTPTransport {
	return &HTTPTransport{
		url:    url,
		client: &http.Client{Timeout: 15 * time.Second},
	}
}

func (t *HTTPTransport) Start(ctx context.Context) error {
	t.ctx = ctx
	t.pending = make(chan json.RawMessage, 100)
	return nil
}

func (t *HTTPTransport) Send(msg json.RawMessage) error {
	ctx := t.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	return t.sendWithContext(ctx, msg)
}

func (t *HTTPTransport) sendWithContext(ctx context.Context, msg json.RawMessage) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return fmt.Errorf("transport closed")
	}

	// Detect notifications (no response expected)
	var check struct {
		Method string `json:"method"`
	}
	json.Unmarshal(msg, &check)
	isNotification := strings.HasPrefix(check.Method, "notifications/")

	req, err := http.NewRequestWithContext(ctx, "POST", t.url, bytes.NewReader(msg))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if t.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", t.sessionID)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("http post: %w", err)
	}
	defer resp.Body.Close()

	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		t.sessionID = sid
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	if isNotification {
		return nil
	}

	if resp.StatusCode >= 400 {
		return fmt.Errorf("http %d: %s", resp.StatusCode, string(body))
	}

	// Parse SSE or plain JSON
	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "text/event-stream") {
		for _, line := range strings.Split(string(body), "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "data: ") {
				t.pending <- json.RawMessage(strings.TrimPrefix(line, "data: "))
				return nil
			}
		}
		return fmt.Errorf("SSE response contained no data lines")
	}

	t.pending <- json.RawMessage(body)
	return nil
}

func (t *HTTPTransport) Receive() (json.RawMessage, error) {
	msg, ok := <-t.pending
	if !ok {
		return nil, io.EOF
	}
	return msg, nil
}

func (t *HTTPTransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.closed {
		t.closed = true
		close(t.pending)
	}
	return nil
}

// --- Mock Transport (for testing) ---

// MockMCPTransport simulates an MCP server for testing.
type MockMCPTransport struct {
	mu        sync.Mutex
	tools     []MCPToolDef
	responses chan json.RawMessage
	closed    bool
}

// MCPToolDef defines a tool for use with MockMCPTransport.
type MCPToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
	Handler     func(args map[string]any) string
}

// NewMockMCPTransport creates a mock transport with the given tools.
func NewMockMCPTransport(tools ...MCPToolDef) *MockMCPTransport {
	return &MockMCPTransport{
		tools:     tools,
		responses: make(chan json.RawMessage, 100),
	}
}

func (m *MockMCPTransport) Start(ctx context.Context) error { return nil }

func (m *MockMCPTransport) Send(msg json.RawMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return fmt.Errorf("transport closed")
	}

	var req jrpcRequest
	if err := json.Unmarshal(msg, &req); err != nil {
		return err
	}

	var resp jrpcResponse
	resp.JSONRPC = "2.0"
	resp.ID = req.ID

	switch req.Method {
	case "initialize":
		result, _ := json.Marshal(map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"serverInfo":      map[string]any{"name": "mock-server", "version": "1.0.0"},
		})
		resp.Result = result

	case "notifications/initialized":
		return nil // notifications don't get responses

	case "tools/list":
		toolList := make([]map[string]any, len(m.tools))
		for i, t := range m.tools {
			toolList[i] = map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"inputSchema": t.InputSchema,
			}
		}
		result, _ := json.Marshal(map[string]any{"tools": toolList})
		resp.Result = result

	case "tools/call":
		var params mcpCallParams
		if raw, ok := req.Params.(json.RawMessage); ok {
			json.Unmarshal(raw, &params)
		} else {
			b, _ := json.Marshal(req.Params)
			json.Unmarshal(b, &params)
		}

		output := "tool not found"
		for _, t := range m.tools {
			if t.Name == params.Name {
				if t.Handler != nil {
					output = t.Handler(params.Arguments)
				} else {
					output = fmt.Sprintf("mock result from %s", t.Name)
				}
				break
			}
		}
		result, _ := json.Marshal(map[string]any{
			"content": []map[string]any{{"type": "text", "text": output}},
			"isError": false,
		})
		resp.Result = result
	}

	respJSON, _ := json.Marshal(resp)
	m.responses <- json.RawMessage(respJSON)
	return nil
}

func (m *MockMCPTransport) Receive() (json.RawMessage, error) {
	msg, ok := <-m.responses
	if !ok {
		return nil, io.EOF
	}
	return msg, nil
}

func (m *MockMCPTransport) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.closed {
		m.closed = true
		close(m.responses)
	}
	return nil
}

// --- JSON-RPC types ---

type jrpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jrpcError      `json:"error,omitempty"`
}

type jrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type mcpCallParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type mcpToolsResult struct {
	Tools []struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		InputSchema map[string]any `json:"inputSchema"`
	} `json:"tools"`
}

type mcpCallResult struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	IsError bool `json:"isError"`
}

// --- MCP Client ---

// MCPClient manages a connection to a single MCP server.
// It handles initialization, tool discovery, and tool invocation.
type MCPClient struct {
	name      string
	transport MCPTransport
	mu        sync.Mutex
	nextID    int64
	tools     []Tool
	log       *slog.Logger
	connected bool
}

// NewMCPClient creates a client for the given transport.
func NewMCPClient(name string, transport MCPTransport, logger *slog.Logger) *MCPClient {
	if logger == nil {
		logger = slog.Default()
	}
	return &MCPClient{
		name:      name,
		transport: transport,
		log:       logger.With("component", "mcp_client", "mcp_server", name),
	}
}

// Connect starts the transport, performs MCP initialization, and discovers tools.
func (c *MCPClient) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected {
		return nil
	}

	c.log.Info("connecting to MCP server")
	start := time.Now()

	if err := c.transport.Start(ctx); err != nil {
		return fmt.Errorf("mcp %s: start transport: %w", c.name, err)
	}

	// Initialize
	if err := c.initialize(ctx); err != nil {
		c.transport.Close()
		return fmt.Errorf("mcp %s: initialize: %w", c.name, err)
	}

	// Discover tools
	tools, err := c.discoverTools(ctx)
	if err != nil {
		c.transport.Close()
		return fmt.Errorf("mcp %s: discover tools: %w", c.name, err)
	}

	c.tools = tools
	c.connected = true
	c.log.Info("MCP server connected",
		"tools_discovered", len(tools),
		"duration", time.Since(start),
	)

	return nil
}

// Tools returns the tools discovered from this MCP server as herdai Tools.
// Each tool's Execute handler routes to the MCP server.
func (c *MCPClient) Tools() []Tool {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Tool, len(c.tools))
	copy(out, c.tools)
	return out
}

// Close disconnects from the MCP server.
func (c *MCPClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.connected = false
	c.log.Info("disconnecting from MCP server")
	return c.transport.Close()
}

func (c *MCPClient) sendRequest(method string, params any) (json.RawMessage, error) {
	c.nextID++
	req := jrpcRequest{
		JSONRPC: "2.0",
		ID:      c.nextID,
		Method:  method,
		Params:  params,
	}

	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	if err := c.transport.Send(reqJSON); err != nil {
		return nil, fmt.Errorf("send: %w", err)
	}

	respJSON, err := c.transport.Receive()
	if err != nil {
		return nil, fmt.Errorf("receive: %w", err)
	}

	var resp jrpcResponse
	if err := json.Unmarshal(respJSON, &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("rpc error %d: %s", resp.Error.Code, resp.Error.Message)
	}

	return resp.Result, nil
}

func (c *MCPClient) sendNotification(method string, params any) error {
	req := jrpcRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}
	reqJSON, _ := json.Marshal(req)
	return c.transport.Send(reqJSON)
}

func (c *MCPClient) initialize(ctx context.Context) error {
	params := map[string]any{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "herdai", "version": "1.0.0"},
	}

	_, err := c.sendRequest("initialize", params)
	if err != nil {
		return err
	}

	return c.sendNotification("notifications/initialized", map[string]any{})
}

func (c *MCPClient) discoverTools(ctx context.Context) ([]Tool, error) {
	result, err := c.sendRequest("tools/list", map[string]any{})
	if err != nil {
		return nil, err
	}

	var toolsResult mcpToolsResult
	if err := json.Unmarshal(result, &toolsResult); err != nil {
		return nil, fmt.Errorf("parse tools list: %w", err)
	}

	tools := make([]Tool, 0, len(toolsResult.Tools))
	for _, t := range toolsResult.Tools {
		mcpToolName := t.Name
		serverName := c.name

		params := extractParams(t.InputSchema)

		tool := Tool{
			Name:        mcpToolName,
			Description: t.Description,
			Parameters:  params,
			Execute:     c.makeToolHandler(serverName, mcpToolName),
		}
		tools = append(tools, tool)

		c.log.Info("discovered MCP tool",
			"tool", mcpToolName,
			"params", len(params),
		)
	}

	return tools, nil
}

func (c *MCPClient) makeToolHandler(serverName, toolName string) ToolHandler {
	return func(ctx context.Context, args map[string]any) (string, error) {
		c.mu.Lock()
		defer c.mu.Unlock()

		c.log.Info("calling MCP tool", "server", serverName, "tool", toolName)

		result, err := c.sendRequest("tools/call", mcpCallParams{
			Name:      toolName,
			Arguments: args,
		})
		if err != nil {
			return "", fmt.Errorf("mcp tool %s/%s: %w", serverName, toolName, err)
		}

		var callResult mcpCallResult
		if err := json.Unmarshal(result, &callResult); err != nil {
			return "", fmt.Errorf("parse tool result: %w", err)
		}

		if callResult.IsError {
			var texts []string
			for _, c := range callResult.Content {
				texts = append(texts, c.Text)
			}
			return "", fmt.Errorf("mcp tool error: %s", strings.Join(texts, "; "))
		}

		var texts []string
		for _, c := range callResult.Content {
			if c.Type == "text" {
				texts = append(texts, c.Text)
			}
		}
		return strings.Join(texts, "\n"), nil
	}
}

// extractParams converts a JSON Schema inputSchema to herdai ToolParams.
func extractParams(schema map[string]any) []ToolParam {
	if schema == nil {
		return nil
	}

	props, _ := schema["properties"].(map[string]any)
	requiredList, _ := schema["required"].([]any)
	requiredSet := make(map[string]bool)
	for _, r := range requiredList {
		if s, ok := r.(string); ok {
			requiredSet[s] = true
		}
	}

	var params []ToolParam
	for name, v := range props {
		prop, ok := v.(map[string]any)
		if !ok {
			continue
		}
		p := ToolParam{
			Name:     name,
			Type:     "string",
			Required: requiredSet[name],
		}
		if t, ok := prop["type"].(string); ok {
			p.Type = t
		}
		if d, ok := prop["description"].(string); ok {
			p.Description = d
		}
		params = append(params, p)
	}
	return params
}

// --- High-level helpers ---

// MCPServerConfigIsRunnable returns true if the server has an HTTP URL or a non-empty stdio Command.
func MCPServerConfigIsRunnable(srv MCPServerConfig) bool {
	return strings.TrimSpace(srv.URL) != "" || strings.TrimSpace(srv.Command) != ""
}

// FilterRunnableMCPServers drops entries with neither URL nor Command (avoids exec: no command).
func FilterRunnableMCPServers(servers []MCPServerConfig) []MCPServerConfig {
	var out []MCPServerConfig
	for _, srv := range servers {
		if MCPServerConfigIsRunnable(srv) {
			out = append(out, srv)
		}
	}
	return out
}

// ConnectMCP connects to multiple MCP servers and returns all discovered tools
// and the clients (for cleanup). Tools from all servers are merged.
func ConnectMCP(ctx context.Context, servers []MCPServerConfig, logger *slog.Logger) ([]Tool, []*MCPClient, error) {
	if logger == nil {
		logger = slog.Default()
	}

	var allTools []Tool
	var clients []*MCPClient

	for _, srv := range servers {
		// Unconfigured stdio (empty Command) and no HTTP URL — skip; caller may still use built-in tools.
		if !MCPServerConfigIsRunnable(srv) {
			continue
		}
		var transport MCPTransport
		if srv.URL != "" {
			transport = NewHTTPTransport(srv.URL)
		} else {
			transport = NewStdioTransport(srv.Command, srv.Args, srv.Env)
		}
		client := NewMCPClient(srv.Name, transport, logger)

		if err := client.Connect(ctx); err != nil {
			// Log and skip — don't let one unavailable server block everything.
			// The agent will simply not have that server's tools available.
			logger.Warn("MCP server unavailable, skipping",
				"server", srv.Name,
				"url", srv.URL,
				"command", srv.Command,
				"error", err,
			)
			continue
		}

		clients = append(clients, client)
		allTools = append(allTools, client.Tools()...)
	}

	return allTools, clients, nil
}

// ConnectMCPWithTransport connects to an MCP server using a custom transport.
// Useful for testing or non-stdio transports.
func ConnectMCPWithTransport(ctx context.Context, name string, transport MCPTransport, logger *slog.Logger) (*MCPClient, []Tool, error) {
	client := NewMCPClient(name, transport, logger)
	if err := client.Connect(ctx); err != nil {
		return nil, nil, err
	}
	return client, client.Tools(), nil
}
