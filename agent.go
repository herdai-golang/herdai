package herdai

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"
)

// AgentConfig holds all parameters needed to create an Agent.
type AgentConfig struct {
	ID           string            // unique identifier
	Role         string            // what this agent does (e.g. "Market Analyst")
	Goal         string            // what it's trying to achieve
	Backstory    string            // optional context that shapes behavior
	Tools        []Tool            // capabilities this agent can use
	LLM          LLM               // language model provider
	MaxToolCalls int               // hard limit on tool invocations per run (default: 10)
	Timeout      time.Duration     // hard deadline per run (default: 2m)
	Logger       *slog.Logger      // structured logger (default: slog.Default())
	MCPServers   []MCPServerConfig // MCP servers to connect to (tools auto-discovered)
	DisableMCP   bool              // if true, skip manager-level MCP servers (default: false)

	// ParallelToolCalls controls whether tool calls within a single LLM response
	// are executed concurrently. When the LLM returns multiple tool calls in one
	// response, they are independent by definition (otherwise the LLM would return
	// them in separate rounds). Default: true.
	ParallelToolCalls *bool

	// Memory — multi-layer memory store (optional)
	Memory MemoryStore

	// HITL — human-in-the-loop approval for tool calls (optional)
	HITL *HITLConfig

	// Guardrails — input/output validation (optional)
	InputGuardrails  *GuardrailChain // validated before LLM sees the input
	OutputGuardrails *GuardrailChain // validated before result is returned

	// RAG — retrieval-augmented generation (optional)
	// Documents are loaded once at setup via IngestionPipeline, but the store is
	// live — you can add more documents anytime with IngestDocuments().
	RAG *RAGConfig

	// ToolCache — smart result caching for tool calls (optional).
	// When set, the agent automatically caches tool results and reuses them
	// when the same tool is called with similar context. Results auto-invalidate
	// when context changes meaningfully (new information detected).
	ToolCache *ToolCache
}

// Agent is an autonomous unit that uses an LLM and tools to accomplish a goal.
// It implements the Runnable interface so it can be composed into Managers.
type Agent struct {
	id                string
	role              string
	goal              string
	backstory         string
	tools             []Tool
	toolMap           map[string]Tool
	llm               LLM
	maxToolCalls      int
	parallelToolCalls bool
	timeout           time.Duration
	log               *slog.Logger
	mcpServers        []MCPServerConfig
	mcpClients        []*MCPClient
	mcpReady          bool
	disableMCP        bool
	mu                sync.Mutex

	memory           MemoryStore
	hitl             *HITLController
	inputGuardrails  *GuardrailChain
	outputGuardrails *GuardrailChain
	rag              *RAGConfig
	toolCache        *ToolCache
}

// NewAgent creates an Agent with the given config and sensible defaults.
func NewAgent(cfg AgentConfig) *Agent {
	if cfg.MaxToolCalls <= 0 {
		cfg.MaxToolCalls = 10
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 2 * time.Minute
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}

	toolMap := make(map[string]Tool, len(cfg.Tools))
	for _, t := range cfg.Tools {
		toolMap[t.Name] = t
	}

	parallel := true
	if cfg.ParallelToolCalls != nil {
		parallel = *cfg.ParallelToolCalls
	}

	a := &Agent{
		id:                cfg.ID,
		role:              cfg.Role,
		goal:              cfg.Goal,
		backstory:         cfg.Backstory,
		tools:             cfg.Tools,
		toolMap:           toolMap,
		llm:               cfg.LLM,
		maxToolCalls:      cfg.MaxToolCalls,
		parallelToolCalls: parallel,
		timeout:           cfg.Timeout,
		log:               cfg.Logger.With("component", "agent", "agent_id", cfg.ID),
		mcpServers:        cfg.MCPServers,
		disableMCP:        cfg.DisableMCP,
		memory:            cfg.Memory,
	}

	if cfg.HITL != nil && cfg.HITL.Handler != nil {
		a.hitl = NewHITLController(*cfg.HITL)
	}
	if cfg.InputGuardrails != nil && cfg.InputGuardrails.Len() > 0 {
		a.inputGuardrails = cfg.InputGuardrails
	}
	if cfg.OutputGuardrails != nil && cfg.OutputGuardrails.Len() > 0 {
		a.outputGuardrails = cfg.OutputGuardrails
	}
	if cfg.RAG != nil && cfg.RAG.Retriever != nil {
		a.rag = cfg.RAG
		if a.rag.TopK <= 0 {
			a.rag.TopK = 5
		}
	}
	if cfg.ToolCache != nil {
		a.toolCache = cfg.ToolCache
	}

	return a
}

// MCPServerConfigs returns the agent's configured MCP servers.
func (a *Agent) MCPServerConfigs() []MCPServerConfig {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]MCPServerConfig, len(a.mcpServers))
	copy(out, a.mcpServers)
	return out
}

// AddMCPServers adds MCP server configs to this agent.
// Tools from these servers will be discovered on the next Run (or via ConnectMCP).
func (a *Agent) AddMCPServers(servers ...MCPServerConfig) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.mcpServers = append(a.mcpServers, servers...)
	a.mcpReady = false // force rediscovery
}

// ConnectMCPWithTransport connects to an MCP server using a custom transport.
// Useful for testing or when you want explicit control over the connection.
func (a *Agent) ConnectMCPWithTransport(ctx context.Context, name string, transport MCPTransport) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	client, tools, err := ConnectMCPWithTransport(ctx, name, transport, a.log)
	if err != nil {
		return err
	}

	a.mcpClients = append(a.mcpClients, client)
	for _, t := range tools {
		a.tools = append(a.tools, t)
		a.toolMap[t.Name] = t
	}
	return nil
}

// Close disconnects from all MCP servers. Call this when done with the agent.
func (a *Agent) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	var lastErr error
	for _, c := range a.mcpClients {
		if err := c.Close(); err != nil {
			lastErr = err
		}
	}
	a.mcpClients = nil
	a.mcpReady = false
	return lastErr
}

func (a *Agent) ensureMCPConnected(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.mcpReady || len(a.mcpServers) == 0 {
		return nil
	}

	runnable := FilterRunnableMCPServers(a.mcpServers)
	if len(runnable) == 0 {
		// Had placeholder configs (e.g. empty Command) — nothing to connect; don't retry every Run.
		a.mcpReady = true
		return nil
	}

	a.log.Info("connecting to MCP servers", "count", len(runnable))

	tools, clients, err := ConnectMCP(ctx, runnable, a.log)
	if err != nil {
		return fmt.Errorf("agent %s: mcp connect: %w", a.id, err)
	}

	a.mcpClients = append(a.mcpClients, clients...)
	for _, t := range tools {
		a.tools = append(a.tools, t)
		a.toolMap[t.Name] = t
	}
	a.mcpReady = true

	a.log.Info("MCP tools added", "new_tools", len(tools), "total_tools", len(a.tools))
	return nil
}

// GetID returns the agent's unique identifier.
func (a *Agent) GetID() string { return a.id }

// Describe returns a human-readable summary (used by the LLM Router).
func (a *Agent) Describe() string {
	return fmt.Sprintf("%s: %s (Goal: %s)", a.id, a.role, a.goal)
}

func (a *Agent) buildSystemPrompt() string {
	prompt := fmt.Sprintf("You are %s.\n\nRole: %s\nGoal: %s", a.role, a.role, a.goal)
	if a.backstory != "" {
		prompt += fmt.Sprintf("\nBackstory: %s", a.backstory)
	}
	if len(a.tools) > 0 {
		prompt += "\n\nYou have access to tools. Use them when needed to accomplish your goal."
		prompt += " After gathering information via tools, provide a clear final answer."
	}
	prompt += "\n\nProvide clear, structured, and actionable responses."
	return prompt
}

// Run executes the agent: builds context from the conversation, calls the LLM
// (with tool loop if needed), and returns a structured Result.
// Every LLM call and tool call respects the context deadline — no hanging.
// If a Tracer is in the context, all operations are traced as spans.
// If HITL is configured, tool calls pause for human approval.
// If Memory is configured, relevant memories are injected into context.
func (a *Agent) Run(ctx context.Context, input string, conv *Conversation) (*Result, error) {
	ctx, cancel := context.WithTimeout(ctx, a.timeout)
	defer cancel()

	// Start tracing span
	agentSpan, ctx := StartSpanFromContext(ctx, "agent:"+a.id, SpanKindAgent)
	agentSpan.SetAttribute("agent.id", a.id)
	agentSpan.SetAttribute("agent.role", a.role)
	agentSpan.SetAttribute("input.length", len(input))

	// Auto-connect to MCP servers if configured
	if err := a.ensureMCPConnected(ctx); err != nil {
		a.log.Error("MCP connection failed", "error", err)
		agentSpan.EndError(err)
		return nil, err
	}

	a.log.Info("agent started",
		"input_length", len(input),
		"tool_count", len(a.tools),
	)
	start := time.Now()

	// Input guardrails — validate/transform input before LLM sees it
	if a.inputGuardrails != nil {
		guardSpan := agentSpan.StartChild("guardrail:input", SpanKindCustom)
		validated, err := a.inputGuardrails.Run(ctx, input)
		if err != nil {
			a.log.Warn("input guardrail blocked", "error", err)
			guardSpan.EndError(err)
			agentSpan.EndError(err)
			return nil, fmt.Errorf("agent %s: input guardrail: %w", a.id, err)
		}
		if validated != input {
			a.log.Info("input modified by guardrail", "original_len", len(input), "modified_len", len(validated))
			input = validated
		}
		guardSpan.EndOK()
	}

	messages := []Message{
		{Role: RoleSystem, Content: a.buildSystemPrompt()},
	}

	// Inject relevant memories into context
	if a.memory != nil {
		memSpan := agentSpan.StartChild("memory:recall", SpanKindMemory)
		memories, err := a.memory.Search(ctx, input, 5)
		if err != nil {
			a.log.Warn("memory search failed", "error", err)
			memSpan.EndError(err)
		} else if len(memories) > 0 {
			var memContext string
			for _, m := range memories {
				memContext += fmt.Sprintf("- [%s] %s\n", m.Kind, m.Content)
			}
			messages = append(messages, Message{
				Role:    RoleSystem,
				Content: "Relevant memories from previous interactions:\n" + memContext,
			})
			memSpan.SetAttribute("memory.count", len(memories))
			memSpan.EndOK()
		} else {
			memSpan.EndOK()
		}
	}

	// RAG — retrieve relevant document chunks and inject as context
	if a.rag != nil {
		ragSpan := agentSpan.StartChild("rag:retrieve", SpanKindCustom)
		ragQuery := input
		if a.rag.QueryRewriter != nil {
			ragQuery = a.rag.QueryRewriter(input)
		}

		chunks, err := a.rag.Retriever.Retrieve(ctx, ragQuery, a.rag.TopK)
		if err != nil {
			a.log.Warn("RAG retrieval failed", "error", err)
			ragSpan.EndError(err)
		} else if len(chunks) > 0 {
			var filtered []Chunk
			for _, ch := range chunks {
				if ch.Score >= a.rag.MinScore {
					filtered = append(filtered, ch)
				}
			}

			if len(filtered) > 0 {
				var ragContext strings.Builder
				ragContext.WriteString("Relevant context from your knowledge base:\n\n")
				for i, ch := range filtered {
					fmt.Fprintf(&ragContext, "[%d] ", i+1)
					if a.rag.CiteSources && ch.Source != "" {
						fmt.Fprintf(&ragContext, "(source: %s) ", ch.Source)
					}
					ragContext.WriteString(ch.Content)
					ragContext.WriteString("\n\n")
				}
				ragContext.WriteString("Use the above context to ground your response. If the context doesn't contain relevant information, say so.")
				messages = append(messages, Message{
					Role:    RoleSystem,
					Content: ragContext.String(),
				})
				ragSpan.SetAttribute("chunks.count", len(filtered))
				ragSpan.SetAttribute("chunks.top_score", filtered[0].Score)
			}
			ragSpan.EndOK()
		} else {
			ragSpan.EndOK()
		}
	}

	if conv != nil {
		for _, t := range conv.LastN(20) {
			role := RoleUser
			if t.AgentID == a.id {
				role = RoleAssistant
			}
			content := t.Content
			if role == RoleUser && t.AgentID != "" {
				content = fmt.Sprintf("[%s]: %s", t.AgentID, t.Content)
			}
			messages = append(messages, Message{Role: role, Content: content})
		}
	}

	messages = append(messages, Message{Role: RoleUser, Content: input})

	toolCallCount := 0
	var toolRecords []ToolCallRecord
	llmRound := 0

	for {
		select {
		case <-ctx.Done():
			a.log.Error("agent timed out",
				"tool_calls_made", toolCallCount,
				"llm_rounds", llmRound,
				"duration", time.Since(start),
			)
			agentSpan.SetAttribute("tool_calls", toolCallCount)
			agentSpan.EndError(ctx.Err())
			return nil, fmt.Errorf("agent %s timed out after %v: %w", a.id, time.Since(start), ctx.Err())
		default:
		}

		llmRound++
		callStart := time.Now()
		approxChars := approxAgentPromptChars(messages)

		a.log.Info("LLM round starting",
			"agent_id", a.id,
			"llm_round", llmRound,
			"tool_calls_so_far", toolCallCount,
			"message_count", len(messages),
			"approx_context_chars", approxChars,
			"tools_registered", len(a.tools),
		)

		llmSpan := agentSpan.StartChild("llm:chat", SpanKindLLM)
		llmSpan.SetAttribute("message_count", len(messages))
		llmSpan.SetAttribute("llm.round", llmRound)

		resp, err := a.llm.Chat(ctx, messages, a.tools)
		callDur := time.Since(callStart)
		if err != nil {
			rateLimited := strings.Contains(err.Error(), "429") ||
				strings.Contains(strings.ToLower(err.Error()), "rate limit")
			a.log.Error("LLM call failed",
				"agent_id", a.id,
				"llm_round", llmRound,
				"duration_this_call", callDur,
				"duration_agent_run", time.Since(start),
				"tool_calls_so_far", toolCallCount,
				"approx_context_chars", approxChars,
				"rate_limit_or_429", rateLimited,
				"error", err,
			)
			llmSpan.EndError(err)
			agentSpan.EndError(err)
			return nil, fmt.Errorf("agent %s: llm error: %w", a.id, err)
		}
		llmSpan.SetAttribute("response.length", len(resp.Content))
		llmSpan.SetAttribute("tool_calls.count", len(resp.ToolCalls))
		llmSpan.EndOK()

		a.log.Info("LLM round completed",
			"agent_id", a.id,
			"llm_round", llmRound,
			"duration_this_call", callDur,
			"returned_tool_calls", len(resp.ToolCalls),
			"content_len", len(resp.Content),
		)

		if len(resp.ToolCalls) == 0 {
			a.log.Info("agent completed",
				"duration", time.Since(start),
				"llm_rounds", llmRound,
				"tool_calls", toolCallCount,
				"response_length", len(resp.Content),
			)

			finalContent := resp.Content

			// Output guardrails — validate/transform output before returning
			if a.outputGuardrails != nil {
				guardSpan := agentSpan.StartChild("guardrail:output", SpanKindCustom)
				validated, err := a.outputGuardrails.Run(ctx, finalContent)
				if err != nil {
					a.log.Warn("output guardrail blocked", "error", err)
					guardSpan.EndError(err)
					agentSpan.EndError(err)
					return nil, fmt.Errorf("agent %s: output guardrail: %w", a.id, err)
				}
				if validated != finalContent {
					a.log.Info("output modified by guardrail", "original_len", len(finalContent), "modified_len", len(validated))
					finalContent = validated
				}
				guardSpan.EndOK()
			}

			if conv != nil {
				conv.AddTurn(Turn{
					AgentID:   a.id,
					Role:      string(RoleAssistant),
					Content:   finalContent,
					ToolCalls: toolRecords,
				})
			}

			// Store result as a memory episode
			if a.memory != nil {
				_ = a.memory.Store(ctx, MemoryEntry{
					AgentID:   a.id,
					Kind:      MemoryEpisode,
					Content:   fmt.Sprintf("Task: %s\nResult: %s", truncate(input, 200), truncate(finalContent, 500)),
					Tags:      []string{"episode", a.id},
					CreatedAt: time.Now(),
				})
			}

			agentSpan.SetAttribute("tool_calls", toolCallCount)
			agentSpan.SetAttribute("response.length", len(finalContent))
			agentSpan.SetAttribute("duration_ms", time.Since(start).Milliseconds())
			agentSpan.EndOK()

			return &Result{
				AgentID: a.id,
				Content: finalContent,
				Metadata: map[string]any{
					"tool_calls":  toolCallCount,
					"duration_ms": time.Since(start).Milliseconds(),
				},
			}, nil
		}

		if toolCallCount+len(resp.ToolCalls) > a.maxToolCalls {
			a.log.Warn("max tool calls exceeded",
				"limit", a.maxToolCalls,
				"attempted", toolCallCount+len(resp.ToolCalls),
			)
			agentSpan.SetAttribute("tool_calls", toolCallCount)
			agentSpan.EndError(fmt.Errorf("exceeded max tool calls"))
			return nil, fmt.Errorf("agent %s: exceeded max tool calls (%d)", a.id, a.maxToolCalls)
		}

		messages = append(messages, Message{
			Role:      RoleAssistant,
			ToolCalls: resp.ToolCalls,
		})

		// Phase 1: HITL checks and pre-validation (sequential — human interaction).
		// Build an execution plan for each tool call.
		type toolExecSlot struct {
			tc       ToolCall
			tool     Tool
			resolved bool // true means we already have a message (rejected / error)
			msg      Message
			record   ToolCallRecord
		}
		slots := make([]toolExecSlot, len(resp.ToolCalls))

		for i, tc := range resp.ToolCalls {
			toolCallCount++
			slots[i].tc = tc

			// HITL check
			if a.hitl != nil && a.hitl.NeedsApproval(tc) {
				hitlSpan := agentSpan.StartChild("hitl:"+tc.Function, SpanKindCustom)
				hitlResp, err := a.hitl.RequestApproval(ctx, a.id, tc)
				if err != nil {
					hitlSpan.EndError(err)
					agentSpan.EndError(err)
					return nil, fmt.Errorf("agent %s: hitl error for %s: %w", a.id, tc.Function, err)
				}
				hitlSpan.SetAttribute("decision", hitlResp.Decision.String())
				hitlSpan.EndOK()

				switch hitlResp.Decision {
				case HITLAbort:
					agentSpan.AddEvent("hitl_abort", map[string]any{"tool": tc.Function})
					agentSpan.EndOK()
					return nil, fmt.Errorf("agent %s: run aborted by human at tool call %s", a.id, tc.Function)
				case HITLReject:
					a.log.Info("tool call rejected by human", "tool", tc.Function)
					feedback := "Tool call was rejected by the user."
					if hitlResp.Feedback != "" {
						feedback += " Feedback: " + hitlResp.Feedback
					}
					slots[i].resolved = true
					slots[i].record = ToolCallRecord{ToolName: tc.Function, Input: tc.Args, Error: "rejected by human"}
					slots[i].msg = Message{Role: RoleTool, Content: feedback, ToolCallID: tc.ID, Name: tc.Function}
					continue
				case HITLEdit:
					if hitlResp.EditedArgs != nil {
						slots[i].tc.Args = hitlResp.EditedArgs
					}
				case HITLApprove, HITLApproveAll:
					// proceed
				}
			}

			tool, ok := a.toolMap[tc.Function]
			if !ok {
				a.log.Error("unknown tool requested", "tool", tc.Function)
				slots[i].resolved = true
				slots[i].record = ToolCallRecord{ToolName: tc.Function, Input: tc.Args, Error: "unknown tool"}
				slots[i].msg = Message{
					Role: RoleTool, ToolCallID: tc.ID, Name: tc.Function,
					Content: fmt.Sprintf("Error: unknown tool '%s'. Available tools: %s", tc.Function, a.availableToolNames()),
				}
				continue
			}
			if tool.Execute == nil {
				a.log.Error("tool has no handler", "tool", tc.Function)
				slots[i].resolved = true
				slots[i].record = ToolCallRecord{ToolName: tc.Function, Input: tc.Args, Error: "tool has no execute handler"}
				slots[i].msg = Message{
					Role: RoleTool, ToolCallID: tc.ID, Name: tc.Function,
					Content: fmt.Sprintf("Error: tool '%s' has no execute handler", tc.Function),
				}
				continue
			}
			slots[i].tool = tool
		}

		// Phase 2: Execute approved tools.
		// If parallel is enabled and there are 2+ tools to execute, run them concurrently.
		pendingCount := 0
		for _, s := range slots {
			if !s.resolved {
				pendingCount++
			}
		}

		if a.parallelToolCalls && pendingCount >= 2 {
			a.log.Info("executing tools in parallel", "count", pendingCount)
			var wg sync.WaitGroup
			for i := range slots {
				if slots[i].resolved {
					continue
				}
				wg.Add(1)
				go func(idx int) {
					defer wg.Done()
					s := &slots[idx]
					toolStart := time.Now()
					toolSpan := agentSpan.StartChild("tool:"+s.tc.Function, SpanKindTool)
					toolSpan.SetAttribute("tool.name", s.tc.Function)
					toolSpan.SetAttribute("parallel", true)

					a.log.Info("executing tool", "tool", s.tc.Function, "parallel", true)
					output, cached, err := a.executeTool(ctx, s.tool, s.tc)
					toolDuration := time.Since(toolStart)

					s.record = ToolCallRecord{
						ToolName: s.tc.Function,
						Input:    s.tc.Args,
						Output:   output,
						Duration: toolDuration,
					}
					if cached {
						s.record.Cached = true
						toolSpan.SetAttribute("cached", true)
					}
					if err != nil {
						a.log.Error("tool execution failed", "tool", s.tc.Function, "error", err, "duration", toolDuration)
						s.record.Error = err.Error()
						toolSpan.EndError(err)
						s.msg = Message{Role: RoleTool, Content: fmt.Sprintf("Tool error: %s", err.Error()), ToolCallID: s.tc.ID, Name: s.tc.Function}
					} else {
						a.log.Info("tool completed", "tool", s.tc.Function, "duration", toolDuration, "output_length", len(output), "cached", cached)
						toolSpan.SetAttribute("output.length", len(output))
						toolSpan.EndOK()
						s.msg = Message{Role: RoleTool, Content: output, ToolCallID: s.tc.ID, Name: s.tc.Function}
					}
					s.resolved = true
				}(i)
			}
			wg.Wait()
		} else {
			// Sequential fallback (single tool, or parallel disabled)
			for i := range slots {
				if slots[i].resolved {
					continue
				}
				s := &slots[i]
				toolStart := time.Now()
				toolSpan := agentSpan.StartChild("tool:"+s.tc.Function, SpanKindTool)
				toolSpan.SetAttribute("tool.name", s.tc.Function)

				a.log.Info("executing tool", "tool", s.tc.Function)
				output, cached, err := a.executeTool(ctx, s.tool, s.tc)
				toolDuration := time.Since(toolStart)

				s.record = ToolCallRecord{
					ToolName: s.tc.Function,
					Input:    s.tc.Args,
					Output:   output,
					Duration: toolDuration,
				}
				if cached {
					s.record.Cached = true
					toolSpan.SetAttribute("cached", true)
				}
				if err != nil {
					a.log.Error("tool execution failed", "tool", s.tc.Function, "error", err, "duration", toolDuration)
					s.record.Error = err.Error()
					toolSpan.EndError(err)
					s.msg = Message{Role: RoleTool, Content: fmt.Sprintf("Tool error: %s", err.Error()), ToolCallID: s.tc.ID, Name: s.tc.Function}
				} else {
					a.log.Info("tool completed", "tool", s.tc.Function, "duration", toolDuration, "output_length", len(output), "cached", cached)
					toolSpan.SetAttribute("output.length", len(output))
					toolSpan.EndOK()
					s.msg = Message{Role: RoleTool, Content: output, ToolCallID: s.tc.ID, Name: s.tc.Function}
				}
				s.resolved = true
			}
		}

		// Phase 3: Collect results in original order.
		for _, s := range slots {
			toolRecords = append(toolRecords, s.record)
			messages = append(messages, s.msg)
		}
	}
}

// GetMemory returns the agent's memory store (nil if none configured).
func (a *Agent) GetMemory() MemoryStore {
	return a.memory
}

// SetMemory sets or replaces the agent's memory store.
func (a *Agent) SetMemory(mem MemoryStore) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.memory = mem
}

// GetHITLHistory returns the audit trail of HITL decisions (nil if no HITL).
func (a *Agent) GetHITLHistory() []HITLRecord {
	if a.hitl == nil {
		return nil
	}
	return a.hitl.History()
}

// GetToolCache returns the agent's tool cache (nil if none configured).
func (a *Agent) GetToolCache() *ToolCache { return a.toolCache }

// SetToolCache sets or replaces the agent's tool cache.
func (a *Agent) SetToolCache(tc *ToolCache) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.toolCache = tc
}

// executeTool runs a tool, checking the cache first if configured.
func (a *Agent) executeTool(ctx context.Context, tool Tool, tc ToolCall) (string, bool, error) {
	if a.toolCache == nil {
		out, err := tool.Execute(ctx, tc.Args)
		return out, false, err
	}

	toolCtx := toolArgsContext(tc.Args)
	refresh, _ := tc.Args["refresh"].(bool)

	if !refresh {
		if cached, ok := a.toolCache.Get(tc.Function, toolCtx); ok {
			a.log.Info("tool cache hit", "tool", tc.Function)
			return cached, true, nil
		}
	}

	out, err := tool.Execute(ctx, tc.Args)
	if err != nil {
		return "", false, err
	}

	a.toolCache.Set(tc.Function, toolCtx, out)
	return out, false, nil
}

// toolArgsContext builds a context string from tool args for cache comparison.
// Excludes control parameters like "refresh".
func toolArgsContext(args map[string]any) string {
	var parts []string
	for k, v := range args {
		if k == "refresh" {
			continue
		}
		if s, ok := v.(string); ok {
			parts = append(parts, s)
		} else {
			parts = append(parts, fmt.Sprintf("%v", v))
		}
	}
	return strings.Join(parts, " ")
}

// approxAgentPromptChars estimates total characters going to the model (rough proxy for tokens).
func approxAgentPromptChars(msgs []Message) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content)
		for _, tc := range m.ToolCalls {
			n += len(tc.Function) + len(tc.ID)
			if b, err := json.Marshal(tc.Args); err == nil {
				n += len(b)
			}
		}
	}
	return n
}

func (a *Agent) availableToolNames() string {
	names := ""
	for i, t := range a.tools {
		if i > 0 {
			names += ", "
		}
		names += t.Name
	}
	return names
}
