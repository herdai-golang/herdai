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

// Strategy defines how the Manager orchestrates its agents.
type Strategy int

const (
	// StrategySequential runs agents one after another, passing each output as the next input.
	StrategySequential Strategy = iota
	// StrategyParallel runs all agents concurrently with the same input and merges results.
	StrategyParallel
	// StrategyRoundRobin cycles through agents in order until max turns or a finish signal.
	StrategyRoundRobin
	// StrategyLLMRouter uses an LLM to dynamically decide which agent speaks next.
	StrategyLLMRouter
)

func (s Strategy) String() string {
	switch s {
	case StrategySequential:
		return "sequential"
	case StrategyParallel:
		return "parallel"
	case StrategyRoundRobin:
		return "round_robin"
	case StrategyLLMRouter:
		return "llm_router"
	default:
		return "unknown"
	}
}

// ConversationHandler is called by the manager to get human input during execution.
// The manager sends a prompt (e.g. agent results so far) and expects a human response.
// Return empty string to skip/continue without input.
type ConversationHandler func(ctx context.Context, prompt string) string

// ManagerConfig holds all parameters needed to create a Manager.
type ManagerConfig struct {
	ID         string            // unique identifier
	Strategy   Strategy          // orchestration strategy
	Agents     []Runnable        // agents (or sub-managers) to orchestrate
	MaxTurns   int               // max iterations for RoundRobin/LLMRouter (default: 20)
	Timeout    time.Duration     // hard deadline for the entire run (default: 10m)
	LLM        LLM               // required for StrategyLLMRouter and synthesis
	Logger     *slog.Logger      // structured logger
	MCPServers []MCPServerConfig // MCP servers shared across all agents (unless agent has DisableMCP)

	// SynthesisPrompt enables LLM-powered synthesis of parallel/merged results.
	// When set (non-empty), the manager uses its LLM to synthesize all agent outputs
	// into a coherent answer instead of concatenating them.
	// The synthesis LLM may ask up to 2 targeted clarifying questions via the
	// ConversationHandler (if set) when it needs critical information.
	// The human can reply or say "skip" to proceed with assumptions.
	// Requires LLM to be set.
	SynthesisPrompt string

	// ConversationHandler enables human-in-the-loop at the manager level.
	// It is used in three ways:
	//   1. Sequential interrupt: between agent steps, the human can give feedback,
	//      say "skip"/"continue" to proceed, or "stop" to end the pipeline.
	//   2. Synthesis clarification: the synthesizer may ask the human for input
	//      when it needs critical information to produce a better answer.
	//   3. Post-run conversation: after agents complete, the human can ask follow-ups
	//      (routed to the right agent), say "confirm"/"accept" to finalize,
	//      or "reject" to discard the result.
	// Requires LLM to be set for routing and synthesis.
	ConversationHandler ConversationHandler

	// MaxConversationTurns limits how many post-run follow-up interactions the human
	// can have. Default: 5. Only used with ConversationHandler.
	MaxConversationTurns int

	// Session enables durable checkpointing for crash recovery.
	// When set with a SessionStore, the manager saves progress after each agent step.
	// On crash, call ResumeRun with the session ID to continue from the last completed step.
	Session      *Session
	SessionStore SessionStore
}

// Manager orchestrates a group of agents using a chosen strategy.
// It implements Runnable so it can be nested inside another Manager (hierarchy).
type Manager struct {
	id         string
	strategy   Strategy
	agents     []Runnable
	agentMap   map[string]Runnable
	maxTurns   int
	timeout    time.Duration
	llm        LLM
	log        *slog.Logger
	mcpServers []MCPServerConfig

	synthesisPrompt      string
	conversationHandler  ConversationHandler
	maxConversationTurns int

	session      *Session
	sessionStore SessionStore
}

// NewManager creates a Manager with the given config and sensible defaults.
func NewManager(cfg ManagerConfig) *Manager {
	if cfg.MaxTurns <= 0 {
		cfg.MaxTurns = 20
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Minute
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}

	agentMap := make(map[string]Runnable, len(cfg.Agents))
	for _, a := range cfg.Agents {
		agentMap[a.GetID()] = a
	}

	maxConvTurns := cfg.MaxConversationTurns
	if maxConvTurns <= 0 {
		maxConvTurns = 5
	}

	mgr := &Manager{
		id:                   cfg.ID,
		strategy:             cfg.Strategy,
		agents:               cfg.Agents,
		agentMap:             agentMap,
		maxTurns:             cfg.MaxTurns,
		timeout:              cfg.Timeout,
		llm:                  cfg.LLM,
		log:                  cfg.Logger.With("component", "manager", "manager_id", cfg.ID),
		mcpServers:           cfg.MCPServers,
		synthesisPrompt:      cfg.SynthesisPrompt,
		conversationHandler:  cfg.ConversationHandler,
		maxConversationTurns: maxConvTurns,
		session:              cfg.Session,
		sessionStore:         cfg.SessionStore,
	}

	// Propagate MCP servers to all agents that support it (unless they opted out)
	if len(cfg.MCPServers) > 0 {
		mgr.propagateMCP()
	}

	return mgr
}

// GetID returns the manager's unique identifier.
func (m *Manager) GetID() string { return m.id }

// Describe returns a human-readable summary.
func (m *Manager) Describe() string {
	ids := make([]string, len(m.agents))
	for i, a := range m.agents {
		ids[i] = a.GetID()
	}
	return fmt.Sprintf("Manager(%s) [%s] agents: %s", m.id, m.strategy, strings.Join(ids, ", "))
}

// propagateMCP adds the manager's MCP servers to all agents that accept them.
func (m *Manager) propagateMCP() {
	for _, r := range m.agents {
		if agent, ok := r.(*Agent); ok {
			if !agent.disableMCP {
				agent.AddMCPServers(m.mcpServers...)
				m.log.Info("propagated MCP servers to agent",
					"agent_id", agent.id,
					"mcp_count", len(m.mcpServers),
				)
			} else {
				m.log.Info("agent opted out of MCP",
					"agent_id", agent.id,
				)
			}
		} else if subMgr, ok := r.(*Manager); ok {
			subMgr.mcpServers = append(subMgr.mcpServers, m.mcpServers...)
			subMgr.propagateMCP()
		}
	}
}

// Close disconnects all agents from their MCP servers.
func (m *Manager) Close() error {
	var lastErr error
	for _, r := range m.agents {
		if agent, ok := r.(*Agent); ok {
			if err := agent.Close(); err != nil {
				lastErr = err
			}
		} else if subMgr, ok := r.(*Manager); ok {
			if err := subMgr.Close(); err != nil {
				lastErr = err
			}
		}
	}
	return lastErr
}

// AddAgent adds an agent (or sub-manager) to this manager at runtime.
func (m *Manager) AddAgent(agent Runnable) {
	m.agents = append(m.agents, agent)
	m.agentMap[agent.GetID()] = agent

	// Propagate MCP servers to the new agent
	if len(m.mcpServers) > 0 {
		if a, ok := agent.(*Agent); ok && !a.disableMCP {
			a.AddMCPServers(m.mcpServers...)
		}
	}

	m.log.Info("agent added", "agent_id", agent.GetID(), "total_agents", len(m.agents))
}

// managerCheckpoint captures the manager's progress for crash recovery.
type managerCheckpoint struct {
	Strategy    Strategy  `json:"strategy"`
	CurrentStep int       `json:"current_step"`
	LastOutput  string    `json:"last_output"`
	Results     []*Result `json:"results,omitempty"`
}

func (m *Manager) saveCheckpoint(step int, lastOutput string, results []*Result) {
	if m.session == nil || m.sessionStore == nil {
		return
	}
	cp := managerCheckpoint{
		Strategy:    m.strategy,
		CurrentStep: step,
		LastOutput:  lastOutput,
		Results:     results,
	}
	if err := m.session.SetCheckpoint(m.id, cp); err != nil {
		m.log.Warn("checkpoint: save failed", "error", err)
		return
	}
	if err := m.sessionStore.Save(m.session); err != nil {
		m.log.Warn("checkpoint: persist failed", "error", err)
	}
}

func (m *Manager) loadCheckpoint() (managerCheckpoint, bool) {
	if m.session == nil {
		return managerCheckpoint{}, false
	}
	var cp managerCheckpoint
	if err := m.session.GetCheckpoint(m.id, &cp); err != nil {
		return managerCheckpoint{}, false
	}
	return cp, true
}

func (m *Manager) persistSession(result *Result, err error) {
	if m.session == nil || m.sessionStore == nil {
		return
	}
	if err != nil {
		m.session.Fail()
	} else {
		m.session.Complete()
	}
	if result != nil {
		m.session.AddResult(result)
	}
	if saveErr := m.sessionStore.Save(m.session); saveErr != nil {
		m.log.Warn("session: final save failed", "error", saveErr)
	}
}

// Run orchestrates all agents using the configured strategy.
// The entire run is bounded by the configured timeout — no hanging.
// If a Tracer is in the context, all operations are traced as spans.
func (m *Manager) Run(ctx context.Context, input string, conv *Conversation) (*Result, error) {
	ctx, cancel := context.WithTimeout(ctx, m.timeout)
	defer cancel()

	// Start tracing span
	mgrSpan, ctx := StartSpanFromContext(ctx, "manager:"+m.id, SpanKindManager)
	mgrSpan.SetAttribute("manager.id", m.id)
	mgrSpan.SetAttribute("strategy", m.strategy.String())
	mgrSpan.SetAttribute("agent_count", len(m.agents))

	if conv == nil {
		conv = NewConversation()
	}

	if m.session != nil {
		m.session.Conversation = conv
	}

	if len(m.agents) == 0 {
		err := fmt.Errorf("manager %s: no agents configured", m.id)
		mgrSpan.EndError(err)
		return nil, err
	}

	m.log.Info("manager started",
		"strategy", m.strategy.String(),
		"agent_count", len(m.agents),
		"max_turns", m.maxTurns,
		"timeout", m.timeout,
	)
	start := time.Now()

	_, resuming := m.loadCheckpoint()
	if !resuming {
		conv.AddTurn(Turn{
			AgentID: m.id,
			Role:    string(RoleUser),
			Content: input,
		})
	}

	var result *Result
	var err error

	switch m.strategy {
	case StrategySequential:
		result, err = m.runSequential(ctx, input, conv)
	case StrategyParallel:
		result, err = m.runParallel(ctx, input, conv)
	case StrategyRoundRobin:
		result, err = m.runRoundRobin(ctx, input, conv)
	case StrategyLLMRouter:
		result, err = m.runLLMRouter(ctx, input, conv)
	default:
		err = fmt.Errorf("unknown strategy: %d", m.strategy)
	}

	duration := time.Since(start)
	mgrSpan.SetAttribute("duration_ms", duration.Milliseconds())
	mgrSpan.SetAttribute("turns", conv.Len())

	if err != nil {
		m.log.Error("manager failed",
			"strategy", m.strategy.String(),
			"duration", duration,
			"error", err,
		)
		m.persistSession(nil, err)
		mgrSpan.EndError(err)
		return nil, fmt.Errorf("manager %s: %w", m.id, err)
	}

	// Conversational HITL: let the human ask follow-up questions, route to the right agent
	if m.conversationHandler != nil && m.llm != nil && result != nil {
		convResult, convErr := m.runConversationLoop(ctx, input, result, conv)
		if convErr != nil {
			m.log.Warn("conversation loop error", "error", convErr)
		} else if convResult != nil {
			result = convResult
		}
	}

	m.persistSession(result, nil)

	m.log.Info("manager completed",
		"strategy", m.strategy.String(),
		"duration", time.Since(start),
		"turns", conv.Len(),
	)
	mgrSpan.EndOK()
	return result, nil
}

// ResumeRun loads a previously checkpointed session and continues from
// where it left off. The manager resumes the same strategy, skipping
// already-completed steps.
func (m *Manager) ResumeRun(ctx context.Context, sessionID string, store SessionStore) (*Result, error) {
	session, err := store.Load(sessionID)
	if err != nil {
		return nil, fmt.Errorf("resume: load session: %w", err)
	}

	m.session = session
	m.sessionStore = store
	session.Resume()

	conv := session.GetConversation()
	turns := conv.GetTurns()
	if len(turns) == 0 {
		return nil, fmt.Errorf("resume: session %s has no turns to resume from", sessionID)
	}

	originalInput := turns[0].Content
	return m.Run(ctx, originalInput, conv)
}

// --- Sequential: agent1 -> agent2 -> ... -> agentN ---

func (m *Manager) runSequential(ctx context.Context, input string, conv *Conversation) (*Result, error) {
	m.log.Info("sequential: starting", "agent_count", len(m.agents))

	var lastResult *Result
	currentInput := input
	startStep := 0

	if cp, ok := m.loadCheckpoint(); ok && cp.Strategy == StrategySequential {
		startStep = cp.CurrentStep
		currentInput = cp.LastOutput
		m.log.Info("sequential: resuming from checkpoint", "step", startStep)
	}

	for i := startStep; i < len(m.agents); i++ {
		agent := m.agents[i]

		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("sequential cancelled at step %d/%d: %w", i+1, len(m.agents), ctx.Err())
		default:
		}

		m.log.Info("sequential: running agent",
			"step", i+1,
			"total", len(m.agents),
			"agent_id", agent.GetID(),
		)

		result, err := agent.Run(ctx, currentInput, conv)
		if err != nil {
			return nil, fmt.Errorf("sequential step %d, agent %s: %w", i+1, agent.GetID(), err)
		}

		lastResult = result
		currentInput = result.Content
		m.saveCheckpoint(i+1, result.Content, nil)

		// Human interrupt point between agents: confirm, redirect, or stop
		if m.conversationHandler != nil && i < len(m.agents)-1 {
			prompt := fmt.Sprintf("[Agent %s completed step %d/%d]\n\n%s\n\n"+
				"Options: send feedback to adjust the next step, 'skip' to continue as-is, or 'stop' to end here.",
				agent.GetID(), i+1, len(m.agents), truncate(result.Content, 1500))

			humanInput := m.conversationHandler(ctx, prompt)
			humanInput = strings.TrimSpace(humanInput)

			if strings.EqualFold(humanInput, "stop") || strings.EqualFold(humanInput, "abort") {
				m.log.Info("sequential: human stopped pipeline", "at_step", i+1)
				break
			}

			if humanInput != "" && !strings.EqualFold(humanInput, "skip") &&
				!strings.EqualFold(humanInput, "continue") && !strings.EqualFold(humanInput, "") {
				m.log.Info("sequential: human provided feedback", "step", i+1)
				conv.AddTurn(Turn{AgentID: "human", Role: string(RoleUser), Content: humanInput})
				currentInput = fmt.Sprintf("Previous agent output:\n%s\n\nHuman feedback: %s",
					truncate(result.Content, 1000), humanInput)
			}
		}
	}

	return lastResult, nil
}

// --- Parallel: fan-out all agents, fan-in results ---

func (m *Manager) runParallel(ctx context.Context, input string, conv *Conversation) (*Result, error) {
	m.log.Info("parallel: starting", "agent_count", len(m.agents))

	type agentResult struct {
		result *Result
		err    error
		id     string
	}

	resultsCh := make(chan agentResult, len(m.agents))
	var wg sync.WaitGroup

	for _, agent := range m.agents {
		agent := agent
		wg.Add(1)
		go func() {
			defer wg.Done()

			m.log.Info("parallel: starting agent", "agent_id", agent.GetID())
			start := time.Now()

			result, err := agent.Run(ctx, input, conv)
			if err != nil {
				m.log.Error("parallel: agent failed",
					"agent_id", agent.GetID(),
					"duration", time.Since(start),
					"error", err,
				)
				resultsCh <- agentResult{err: err, id: agent.GetID()}
				return
			}

			m.log.Info("parallel: agent completed",
				"agent_id", agent.GetID(),
				"duration", time.Since(start),
			)
			resultsCh <- agentResult{result: result, id: agent.GetID()}
		}()
	}

	go func() {
		wg.Wait()
		close(resultsCh)
	}()

	var results []*Result
	var errors []string
	for ar := range resultsCh {
		if ar.err != nil {
			errors = append(errors, fmt.Sprintf("%s: %v", ar.id, ar.err))
		} else {
			results = append(results, ar.result)
		}
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("parallel execution: all agents failed: %s", strings.Join(errors, "; "))
	}

	if len(errors) > 0 {
		m.log.Warn("parallel: some agents failed",
			"succeeded", len(results),
			"failed", len(errors),
			"errors", strings.Join(errors, "; "),
		)
	}

	return m.mergeResults(results, conv)
}

// --- RoundRobin: cycle through agents until max turns or finish signal ---

func (m *Manager) runRoundRobin(ctx context.Context, input string, conv *Conversation) (*Result, error) {
	m.log.Info("round_robin: starting", "max_turns", m.maxTurns)

	var lastResult *Result
	currentInput := input
	startTurn := 0

	if cp, ok := m.loadCheckpoint(); ok && cp.Strategy == StrategyRoundRobin {
		startTurn = cp.CurrentStep
		currentInput = cp.LastOutput
		m.log.Info("round_robin: resuming from checkpoint", "turn", startTurn)
	}

	for turn := startTurn; turn < m.maxTurns; turn++ {
		agent := m.agents[turn%len(m.agents)]

		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("round_robin cancelled at turn %d: %w", turn+1, ctx.Err())
		default:
		}

		m.log.Info("round_robin: turn",
			"turn", turn+1,
			"max", m.maxTurns,
			"agent_id", agent.GetID(),
		)

		result, err := agent.Run(ctx, currentInput, conv)
		if err != nil {
			return nil, fmt.Errorf("round_robin turn %d, agent %s: %w", turn+1, agent.GetID(), err)
		}

		lastResult = result
		currentInput = result.Content
		m.saveCheckpoint(turn+1, result.Content, nil)

		lower := strings.ToLower(result.Content)
		if strings.Contains(lower, "[done]") || strings.Contains(lower, "[finish]") {
			m.log.Info("round_robin: finish signal received", "turn", turn+1)
			break
		}
	}

	if lastResult == nil {
		return nil, fmt.Errorf("round_robin produced no results")
	}
	return lastResult, nil
}

// --- LLMRouter: LLM decides who speaks next ---

func (m *Manager) runLLMRouter(ctx context.Context, input string, conv *Conversation) (*Result, error) {
	if m.llm == nil {
		return nil, fmt.Errorf("LLMRouter strategy requires an LLM in ManagerConfig")
	}

	m.log.Info("llm_router: starting", "max_turns", m.maxTurns, "agent_count", len(m.agents))

	agentDescs := make([]string, len(m.agents))
	for i, a := range m.agents {
		if agent, ok := a.(*Agent); ok {
			agentDescs[i] = fmt.Sprintf("- %s: %s (Goal: %s)", agent.id, agent.role, agent.goal)
		} else {
			agentDescs[i] = fmt.Sprintf("- %s", a.GetID())
		}
	}

	var results []*Result
	startTurn := 0

	if cp, ok := m.loadCheckpoint(); ok && cp.Strategy == StrategyLLMRouter {
		startTurn = cp.CurrentStep
		results = cp.Results
		m.log.Info("llm_router: resuming from checkpoint", "turn", startTurn, "existing_results", len(results))
	}

	for turn := startTurn; turn < m.maxTurns; turn++ {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("llm_router cancelled at turn %d: %w", turn+1, ctx.Err())
		default:
		}

		routerPrompt := fmt.Sprintf(
			"You are a manager coordinating a team of specialized agents.\n\n"+
				"Available agents:\n%s\n\n"+
				"Original task: %s\n\n"+
				"Conversation so far:\n%s\n\n"+
				"Decide which agent should act next, or if the task is complete.\n"+
				"Respond with ONLY a JSON object (no other text):\n"+
				`{"agent_id": "<agent_id>", "instruction": "<what to tell the agent>"}`+"\n"+
				"OR to finish:\n"+
				`{"agent_id": "FINISH", "instruction": "<final summary of all findings>"}`,
			strings.Join(agentDescs, "\n"),
			input,
			m.buildTranscriptSummary(conv),
		)

		m.log.Debug("llm_router: asking LLM for next agent", "turn", turn+1)

		routerResp, err := m.llm.Chat(ctx, []Message{
			{Role: RoleSystem, Content: "You are a manager agent. You coordinate other agents to accomplish tasks. Always respond with valid JSON."},
			{Role: RoleUser, Content: routerPrompt},
		}, nil)
		if err != nil {
			return nil, fmt.Errorf("llm_router: router LLM call failed at turn %d: %w", turn+1, err)
		}

		decision, err := m.parseRouterDecision(routerResp.Content)
		if err != nil {
			m.log.Warn("llm_router: failed to parse decision, finishing",
				"turn", turn+1,
				"raw_response", truncate(routerResp.Content, 200),
				"error", err,
			)
			break
		}

		if decision.AgentID == "FINISH" {
			m.log.Info("llm_router: decided to finish",
				"turn", turn+1,
				"summary_length", len(decision.Instruction),
			)
			conv.AddTurn(Turn{
				AgentID: m.id,
				Role:    string(RoleAssistant),
				Content: decision.Instruction,
			})
			return &Result{
				AgentID: m.id,
				Content: decision.Instruction,
				Metadata: map[string]any{
					"turns":    turn + 1,
					"strategy": "llm_router",
				},
			}, nil
		}

		agent, ok := m.agentMap[decision.AgentID]
		if !ok {
			m.log.Warn("llm_router: selected unknown agent, skipping",
				"selected", decision.AgentID,
				"turn", turn+1,
			)
			conv.AddTurn(Turn{
				AgentID: m.id,
				Role:    string(RoleAssistant),
				Content: fmt.Sprintf("Error: agent '%s' not found", decision.AgentID),
			})
			continue
		}

		m.log.Info("llm_router: selected agent",
			"turn", turn+1,
			"agent_id", decision.AgentID,
			"instruction_length", len(decision.Instruction),
		)

		result, err := agent.Run(ctx, decision.Instruction, conv)
		if err != nil {
			m.log.Error("llm_router: agent failed",
				"agent_id", decision.AgentID,
				"turn", turn+1,
				"error", err,
			)
			conv.AddTurn(Turn{
				AgentID: m.id,
				Role:    string(RoleAssistant),
				Content: fmt.Sprintf("Agent %s failed: %s", decision.AgentID, err.Error()),
			})
			continue
		}

		results = append(results, result)
		m.saveCheckpoint(turn+1, "", results)
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("llm_router: no results after %d turns", m.maxTurns)
	}
	return m.mergeResults(results, conv)
}

// --- Helpers ---

type routerDecision struct {
	AgentID     string `json:"agent_id"`
	Instruction string `json:"instruction"`
}

func (m *Manager) parseRouterDecision(raw string) (*routerDecision, error) {
	raw = strings.TrimSpace(raw)

	start := strings.Index(raw, "{")
	end := strings.LastIndex(raw, "}")
	if start == -1 || end == -1 || end <= start {
		return nil, fmt.Errorf("no JSON object found in: %s", truncate(raw, 100))
	}

	jsonStr := raw[start : end+1]
	var decision routerDecision
	if err := json.Unmarshal([]byte(jsonStr), &decision); err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}
	if decision.AgentID == "" {
		return nil, fmt.Errorf("empty agent_id in decision")
	}
	return &decision, nil
}

func (m *Manager) buildTranscriptSummary(conv *Conversation) string {
	turns := conv.GetTurns()
	if len(turns) == 0 {
		return "(no conversation yet)"
	}

	var sb strings.Builder
	for _, t := range turns {
		sb.WriteString(fmt.Sprintf("[%s] (%s): %s\n", t.AgentID, t.Role, truncate(t.Content, 500)))
	}
	return sb.String()
}

func (m *Manager) mergeResults(results []*Result, conv *Conversation) (*Result, error) {
	nonNil := make([]*Result, 0, len(results))
	for _, r := range results {
		if r != nil {
			nonNil = append(nonNil, r)
		}
	}
	if len(nonNil) == 0 {
		return nil, fmt.Errorf("no results to merge")
	}
	if len(nonNil) == 1 {
		return nonNil[0], nil
	}

	// Build the raw combined output
	var sb strings.Builder
	for _, r := range nonNil {
		sb.WriteString(fmt.Sprintf("=== %s ===\n%s\n\n", r.AgentID, r.Content))
	}
	rawCombined := sb.String()

	// If synthesis LLM is configured, use it to create a coherent summary
	if m.synthesisPrompt != "" && m.llm != nil {
		return m.synthesizeResults(rawCombined, nonNil, conv)
	}

	// Otherwise, plain concatenation
	merged := &Result{
		AgentID: m.id,
		Content: "=== Combined Results ===\n\n" + rawCombined,
		Metadata: map[string]any{
			"agent_count": len(nonNil),
			"merged":      true,
			"synthesized": false,
		},
	}

	conv.AddTurn(Turn{
		AgentID: m.id,
		Role:    string(RoleAssistant),
		Content: merged.Content,
	})

	return merged, nil
}

func (m *Manager) synthesizeResults(rawCombined string, results []*Result, conv *Conversation) (*Result, error) {
	synthSpan, ctx := StartSpanFromContext(context.Background(), "synthesis:"+m.id, SpanKindCustom)
	synthSpan.SetAttribute("agent_count", len(results))

	m.log.Info("synthesizing agent results with LLM", "agent_count", len(results))

	messages := []Message{
		{Role: RoleSystem, Content: "You are a senior strategist who synthesizes multiple expert analyses into decisive, actionable recommendations. " +
			"When you need critical information to make a good recommendation, ask ONE focused clarifying question. " +
			"Prefix your question with CLARIFY: so the system knows to ask the human. " +
			"Only ask when truly necessary — prefer giving a strong answer with stated assumptions."},
		{Role: RoleUser, Content: fmt.Sprintf(
			"Synthesize the outputs of %d specialist agents into one coherent response.\n\n"+
				"Instructions: %s\n\n"+
				"Agent outputs:\n%s\n\n"+
				"Produce a single, well-structured response that answers the original question. "+
				"Connect insights across agents. Highlight agreements and contradictions. "+
				"End with clear recommendations or decisions.\n\n"+
				"If you need ONE critical piece of information from the user to give a better answer, "+
				"start your response with CLARIFY: followed by a short, specific question. Otherwise, give your full synthesis.",
			len(results), m.synthesisPrompt, rawCombined,
		)},
	}

	resp, err := m.llm.Chat(ctx, messages, nil)
	if err != nil {
		m.log.Error("synthesis LLM call failed, falling back to concatenation", "error", err)
		synthSpan.EndError(err)
		merged := &Result{
			AgentID:  m.id,
			Content:  "=== Combined Results ===\n\n" + rawCombined,
			Metadata: map[string]any{"agent_count": len(results), "merged": true, "synthesized": false},
		}
		conv.AddTurn(Turn{AgentID: m.id, Role: string(RoleAssistant), Content: merged.Content})
		return merged, nil
	}

	// Check if synthesis needs clarification (max 2 rounds to avoid over-asking)
	content := resp.Content
	clarifyRounds := 0
	for strings.HasPrefix(strings.TrimSpace(content), "CLARIFY:") && m.conversationHandler != nil && clarifyRounds < 2 {
		clarifyRounds++
		question := strings.TrimPrefix(strings.TrimSpace(content), "CLARIFY:")
		question = strings.TrimSpace(question)

		m.log.Info("synthesis: asking human for clarification", "round", clarifyRounds, "question", truncate(question, 100))

		humanAnswer := m.conversationHandler(ctx, question)
		humanAnswer = strings.TrimSpace(humanAnswer)

		if humanAnswer == "" || strings.EqualFold(humanAnswer, "skip") {
			messages = append(messages,
				Message{Role: RoleAssistant, Content: content},
				Message{Role: RoleUser, Content: "The user chose to skip this question. Proceed with your best synthesis using reasonable assumptions."},
			)
		} else {
			conv.AddTurn(Turn{AgentID: "human", Role: string(RoleUser), Content: humanAnswer})
			messages = append(messages,
				Message{Role: RoleAssistant, Content: content},
				Message{Role: RoleUser, Content: "Human's answer: " + humanAnswer + "\n\nNow produce your full synthesis incorporating this information. Do not ask more questions."},
			)
		}

		resp, err = m.llm.Chat(ctx, messages, nil)
		if err != nil {
			m.log.Warn("synthesis: follow-up LLM call failed, using previous response", "error", err)
			break
		}
		content = resp.Content
	}

	synthSpan.SetAttribute("response.length", len(content))
	synthSpan.SetAttribute("clarify_rounds", clarifyRounds)
	synthSpan.EndOK()

	synthesized := &Result{
		AgentID: m.id,
		Content: content,
		Metadata: map[string]any{
			"agent_count":    len(results),
			"merged":         true,
			"synthesized":    true,
			"clarify_rounds": clarifyRounds,
		},
	}

	conv.AddTurn(Turn{
		AgentID: m.id,
		Role:    string(RoleAssistant),
		Content: synthesized.Content,
	})

	return synthesized, nil
}

// runConversationLoop lets the human interact after agents complete.
// The human can: ask follow-up questions (routed to right agent), interrupt, confirm, or stop.
func (m *Manager) runConversationLoop(ctx context.Context, originalInput string, lastResult *Result, conv *Conversation) (*Result, error) {
	agentDescs := make([]string, len(m.agents))
	for i, a := range m.agents {
		if agent, ok := a.(*Agent); ok {
			agentDescs[i] = fmt.Sprintf("- %s: %s (Goal: %s)", agent.id, agent.role, agent.goal)
		} else {
			agentDescs[i] = fmt.Sprintf("- %s", a.GetID())
		}
	}
	agentList := strings.Join(agentDescs, "\n")

	result := lastResult

	for turn := 0; turn < m.maxConversationTurns; turn++ {
		select {
		case <-ctx.Done():
			return result, nil
		default:
		}

		prompt := fmt.Sprintf(
			"[Result from agents]\n\n%s\n\n"+
				"You can: ask a follow-up question, say 'confirm' to accept, or 'reject' to discard.",
			truncate(result.Content, 2000),
		)

		humanInput := m.conversationHandler(ctx, prompt)
		humanInput = strings.TrimSpace(humanInput)

		lower := strings.ToLower(humanInput)

		// End signals
		if humanInput == "" || lower == "done" || lower == "exit" ||
			lower == "quit" || lower == "confirm" || lower == "accept" ||
			lower == "ok" || lower == "yes" || lower == "lgtm" {
			m.log.Info("conversation: human confirmed/ended", "turn", turn+1, "signal", lower)
			break
		}

		// Reject: discard the result
		if lower == "reject" || lower == "no" || lower == "discard" {
			m.log.Info("conversation: human rejected result", "turn", turn+1)
			result = &Result{
				AgentID: m.id,
				Content: "[Result rejected by human. The analysis was discarded.]",
				Metadata: map[string]any{"rejected": true},
			}
			conv.AddTurn(Turn{AgentID: "human", Role: string(RoleUser), Content: "I reject this analysis."})
			break
		}

		conv.AddTurn(Turn{
			AgentID: "human",
			Role:    string(RoleUser),
			Content: humanInput,
		})

		m.log.Info("conversation: human follow-up", "turn", turn+1, "input_length", len(humanInput))

		// Route the question to the best agent
		routerPrompt := fmt.Sprintf(
			"A human has a follow-up after a multi-agent analysis.\n\n"+
				"Original task: %s\n\n"+
				"Available agents:\n%s\n\n"+
				"Current result summary:\n%s\n\n"+
				"Human says: %s\n\n"+
				"Which agent is best suited to handle this? Respond with ONLY JSON:\n"+
				`{"agent_id": "<id>", "instruction": "<the question rephrased for the agent with relevant context>"}`,
			truncate(originalInput, 500),
			agentList,
			truncate(result.Content, 1000),
			humanInput,
		)

		routerResp, err := m.llm.Chat(ctx, []Message{
			{Role: RoleSystem, Content: "You route human questions to the most appropriate specialist agent. Always respond with valid JSON."},
			{Role: RoleUser, Content: routerPrompt},
		}, nil)
		if err != nil {
			m.log.Error("conversation: router LLM failed", "error", err)
			continue
		}

		decision, err := m.parseRouterDecision(routerResp.Content)
		if err != nil {
			m.log.Warn("conversation: failed to parse routing decision", "error", err)
			continue
		}

		agent, ok := m.agentMap[decision.AgentID]
		if !ok {
			m.log.Warn("conversation: routed to unknown agent, trying first available",
				"selected", decision.AgentID)
			agent = m.agents[0]
		}

		m.log.Info("conversation: routing to agent",
			"turn", turn+1,
			"agent_id", agent.GetID(),
		)

		agentResult, err := agent.Run(ctx, decision.Instruction, conv)
		if err != nil {
			m.log.Error("conversation: agent failed", "agent_id", agent.GetID(), "error", err)
			continue
		}

		result = agentResult
	}

	return result, nil
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
