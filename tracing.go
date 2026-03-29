package herdai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Tracing — OpenTelemetry-style per-agent, per-tool span tracing
// ═══════════════════════════════════════════════════════════════════════════════

// SpanKind categorizes what a span represents.
type SpanKind string

const (
	SpanKindAgent    SpanKind = "agent"
	SpanKindTool     SpanKind = "tool"
	SpanKindLLM      SpanKind = "llm"
	SpanKindManager  SpanKind = "manager"
	SpanKindMCP      SpanKind = "mcp"
	SpanKindMemory   SpanKind = "memory"
	SpanKindSession  SpanKind = "session"
	SpanKindCustom   SpanKind = "custom"
)

// SpanStatus is the outcome of a span.
type SpanStatus string

const (
	SpanStatusOK    SpanStatus = "ok"
	SpanStatusError SpanStatus = "error"
)

// Span represents a single timed operation in the trace tree.
// Spans nest hierarchically: manager → agent → llm / tool.
type Span struct {
	ID         string            `json:"id"`
	ParentID   string            `json:"parent_id,omitempty"`
	TraceID    string            `json:"trace_id"`
	Name       string            `json:"name"`
	Kind       SpanKind          `json:"kind"`
	Status     SpanStatus        `json:"status"`
	StartTime  time.Time         `json:"start_time"`
	EndTime    time.Time         `json:"end_time"`
	Duration   time.Duration     `json:"duration_ms"`
	Attributes map[string]any    `json:"attributes,omitempty"`
	Events     []SpanEvent       `json:"events,omitempty"`
	Children   []*Span           `json:"children,omitempty"`

	mu      sync.Mutex
	ended   bool
	tracer  *Tracer
}

// SpanEvent is a timestamped annotation within a span.
type SpanEvent struct {
	Name       string         `json:"name"`
	Timestamp  time.Time      `json:"timestamp"`
	Attributes map[string]any `json:"attributes,omitempty"`
}

// SetAttribute adds a key-value attribute to the span.
func (s *Span) SetAttribute(key string, value any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Attributes == nil {
		s.Attributes = make(map[string]any)
	}
	s.Attributes[key] = value
}

// AddEvent records a timestamped event within the span.
func (s *Span) AddEvent(name string, attrs map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Events = append(s.Events, SpanEvent{
		Name:       name,
		Timestamp:  time.Now(),
		Attributes: attrs,
	})
}

// End completes the span with the given status.
func (s *Span) End(status SpanStatus) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ended {
		return
	}
	s.ended = true
	s.EndTime = time.Now()
	s.Duration = s.EndTime.Sub(s.StartTime)
	s.Status = status

	if s.tracer != nil {
		s.tracer.record(s)
	}
}

// EndOK is a convenience for End(SpanStatusOK).
func (s *Span) EndOK() { s.End(SpanStatusOK) }

// EndError is a convenience for End(SpanStatusError) with an error attribute.
func (s *Span) EndError(err error) {
	s.SetAttribute("error", err.Error())
	s.End(SpanStatusError)
}

// StartChild creates a new child span under this span.
func (s *Span) StartChild(name string, kind SpanKind) *Span {
	child := &Span{
		ID:        generateID(),
		ParentID:  s.ID,
		TraceID:   s.TraceID,
		Name:      name,
		Kind:      kind,
		StartTime: time.Now(),
		tracer:    s.tracer,
	}
	s.mu.Lock()
	s.Children = append(s.Children, child)
	s.mu.Unlock()
	return child
}

// ── Tracer — collects and queries spans ────────────────────────────────────

// Tracer collects spans from agent/manager execution for debugging and analysis.
type Tracer struct {
	mu      sync.RWMutex
	spans   []*Span
	spanMap map[string]*Span
	traceID string
}

// NewTracer creates a new tracer with a unique trace ID.
func NewTracer() *Tracer {
	return &Tracer{
		spanMap: make(map[string]*Span),
		traceID: generateID(),
	}
}

// StartSpan creates a new root span (no parent).
func (t *Tracer) StartSpan(name string, kind SpanKind) *Span {
	s := &Span{
		ID:        generateID(),
		TraceID:   t.traceID,
		Name:      name,
		Kind:      kind,
		StartTime: time.Now(),
		tracer:    t,
	}
	return s
}

func (t *Tracer) record(s *Span) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.spans = append(t.spans, s)
	t.spanMap[s.ID] = s
}

// TraceID returns the trace ID for this tracer.
func (t *Tracer) TraceID() string { return t.traceID }

// Spans returns all completed spans in order.
func (t *Tracer) Spans() []*Span {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]*Span, len(t.spans))
	copy(out, t.spans)
	return out
}

// SpansByKind returns spans matching the given kind.
func (t *Tracer) SpansByKind(kind SpanKind) []*Span {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var out []*Span
	for _, s := range t.spans {
		if s.Kind == kind {
			out = append(out, s)
		}
	}
	return out
}

// RootSpans returns only spans with no parent.
func (t *Tracer) RootSpans() []*Span {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var out []*Span
	for _, s := range t.spans {
		if s.ParentID == "" {
			out = append(out, s)
		}
	}
	return out
}

// Summary returns a human-readable timeline of the trace.
func (t *Tracer) Summary() string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var sb strings.Builder
	fmt.Fprintf(&sb, "Trace %s (%d spans)\n", t.traceID[:12], len(t.spans))
	sb.WriteString(strings.Repeat("─", 80) + "\n")

	for _, s := range t.spans {
		indent := ""
		if s.ParentID != "" {
			indent = "  "
			if s.Kind == SpanKindTool || s.Kind == SpanKindLLM {
				indent = "    "
			}
		}

		status := "✓"
		if s.Status == SpanStatusError {
			status = "✗"
		}

		fmt.Fprintf(&sb, "%s%s [%s] %s (%s) %s\n",
			indent, status, s.Kind, s.Name,
			s.Duration.Round(time.Millisecond), s.Status)

		if errAttr, ok := s.Attributes["error"]; ok {
			fmt.Fprintf(&sb, "%s  error: %v\n", indent, errAttr)
		}

		for _, e := range s.Events {
			fmt.Fprintf(&sb, "%s  → %s\n", indent, e.Name)
		}
	}
	return sb.String()
}

// TotalDuration returns the wall-clock duration of the entire trace.
func (t *Tracer) TotalDuration() time.Duration {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if len(t.spans) == 0 {
		return 0
	}
	earliest := t.spans[0].StartTime
	latest := t.spans[0].EndTime
	for _, s := range t.spans[1:] {
		if s.StartTime.Before(earliest) {
			earliest = s.StartTime
		}
		if s.EndTime.After(latest) {
			latest = s.EndTime
		}
	}
	return latest.Sub(earliest)
}

// Export serializes the trace to JSON.
func (t *Tracer) Export() ([]byte, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return json.MarshalIndent(t.spans, "", "  ")
}

// Stats returns aggregate statistics.
func (t *Tracer) Stats() TraceStats {
	t.mu.RLock()
	defer t.mu.RUnlock()

	stats := TraceStats{
		TraceID:    t.traceID,
		TotalSpans: len(t.spans),
		ByKind:     make(map[SpanKind]int),
	}

	for _, s := range t.spans {
		stats.ByKind[s.Kind]++
		if s.Status == SpanStatusError {
			stats.Errors++
		}
		stats.TotalDuration += s.Duration

		if s.Kind == SpanKindLLM {
			stats.LLMCalls++
			stats.LLMDuration += s.Duration
		} else if s.Kind == SpanKindTool {
			stats.ToolCalls++
			stats.ToolDuration += s.Duration
		}
	}
	return stats
}

// TraceStats provides aggregate statistics about a trace.
type TraceStats struct {
	TraceID       string              `json:"trace_id"`
	TotalSpans    int                 `json:"total_spans"`
	Errors        int                 `json:"errors"`
	TotalDuration time.Duration       `json:"total_duration"`
	LLMCalls      int                 `json:"llm_calls"`
	LLMDuration   time.Duration       `json:"llm_duration"`
	ToolCalls     int                 `json:"tool_calls"`
	ToolDuration  time.Duration       `json:"tool_duration"`
	ByKind        map[SpanKind]int    `json:"by_kind"`
}

// ── Context-based tracing (like OpenTelemetry) ─────────────────────────────

type tracerKey struct{}
type spanKey struct{}

// ContextWithTracer returns a new context carrying the tracer.
func ContextWithTracer(ctx context.Context, t *Tracer) context.Context {
	return context.WithValue(ctx, tracerKey{}, t)
}

// TracerFromContext extracts the tracer from context (nil if none).
func TracerFromContext(ctx context.Context) *Tracer {
	t, _ := ctx.Value(tracerKey{}).(*Tracer)
	return t
}

// ContextWithSpan returns a new context carrying the current span.
func ContextWithSpan(ctx context.Context, s *Span) context.Context {
	return context.WithValue(ctx, spanKey{}, s)
}

// SpanFromContext extracts the current span from context (nil if none).
func SpanFromContext(ctx context.Context) *Span {
	s, _ := ctx.Value(spanKey{}).(*Span)
	return s
}

// StartSpanFromContext creates a child span under the current context span,
// or a root span if no parent span exists. Returns the span and a new context.
func StartSpanFromContext(ctx context.Context, name string, kind SpanKind) (*Span, context.Context) {
	tracer := TracerFromContext(ctx)
	parent := SpanFromContext(ctx)

	var span *Span
	if parent != nil {
		span = parent.StartChild(name, kind)
	} else if tracer != nil {
		span = tracer.StartSpan(name, kind)
	} else {
		span = &Span{
			ID:        generateID(),
			TraceID:   generateID(),
			Name:      name,
			Kind:      kind,
			StartTime: time.Now(),
		}
	}

	return span, ContextWithSpan(ctx, span)
}
