package herdai

import (
	"context"
	"testing"
	"time"
)

func TestTracer_BasicSpans(t *testing.T) {
	tracer := NewTracer()

	root := tracer.StartSpan("manager:test", SpanKindManager)
	root.SetAttribute("agent_count", 2)

	child1 := root.StartChild("agent:analyst", SpanKindAgent)
	time.Sleep(1 * time.Millisecond)
	child1.EndOK()

	child2 := root.StartChild("agent:writer", SpanKindAgent)
	llmSpan := child2.StartChild("llm:chat", SpanKindLLM)
	time.Sleep(1 * time.Millisecond)
	llmSpan.EndOK()

	toolSpan := child2.StartChild("tool:search", SpanKindTool)
	toolSpan.SetAttribute("query", "golang frameworks")
	time.Sleep(1 * time.Millisecond)
	toolSpan.EndOK()

	child2.EndOK()
	root.EndOK()

	spans := tracer.Spans()
	if len(spans) != 5 {
		t.Fatalf("expected 5 spans, got %d", len(spans))
	}

	agentSpans := tracer.SpansByKind(SpanKindAgent)
	if len(agentSpans) != 2 {
		t.Fatalf("expected 2 agent spans, got %d", len(agentSpans))
	}

	roots := tracer.RootSpans()
	if len(roots) != 1 {
		t.Fatalf("expected 1 root span, got %d", len(roots))
	}
}

func TestTracer_ErrorSpan(t *testing.T) {
	tracer := NewTracer()

	span := tracer.StartSpan("failing-op", SpanKindTool)
	span.EndError(context.DeadlineExceeded)

	spans := tracer.Spans()
	if len(spans) != 1 {
		t.Fatal("expected 1 span")
	}
	if spans[0].Status != SpanStatusError {
		t.Errorf("expected error status, got %s", spans[0].Status)
	}
	if spans[0].Attributes["error"] != "context deadline exceeded" {
		t.Errorf("expected error attribute, got %v", spans[0].Attributes)
	}
}

func TestTracer_Events(t *testing.T) {
	tracer := NewTracer()

	span := tracer.StartSpan("with-events", SpanKindAgent)
	span.AddEvent("started processing", map[string]any{"items": 42})
	span.AddEvent("checkpoint reached", nil)
	span.EndOK()

	spans := tracer.Spans()
	if len(spans[0].Events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(spans[0].Events))
	}
	if spans[0].Events[0].Attributes["items"] != 42 {
		t.Error("expected items attribute on first event")
	}
}

func TestTracer_Stats(t *testing.T) {
	tracer := NewTracer()

	mgr := tracer.StartSpan("manager", SpanKindManager)
	llm := mgr.StartChild("llm", SpanKindLLM)
	time.Sleep(1 * time.Millisecond)
	llm.EndOK()
	tool := mgr.StartChild("tool", SpanKindTool)
	time.Sleep(1 * time.Millisecond)
	tool.EndOK()
	mgr.EndOK()

	stats := tracer.Stats()
	if stats.TotalSpans != 3 {
		t.Fatalf("expected 3 spans, got %d", stats.TotalSpans)
	}
	if stats.LLMCalls != 1 {
		t.Errorf("expected 1 LLM call, got %d", stats.LLMCalls)
	}
	if stats.ToolCalls != 1 {
		t.Errorf("expected 1 tool call, got %d", stats.ToolCalls)
	}
	if stats.Errors != 0 {
		t.Errorf("expected 0 errors, got %d", stats.Errors)
	}
}

func TestTracer_Summary(t *testing.T) {
	tracer := NewTracer()

	span := tracer.StartSpan("test-op", SpanKindAgent)
	time.Sleep(1 * time.Millisecond)
	span.EndOK()

	summary := tracer.Summary()
	if summary == "" {
		t.Fatal("expected non-empty summary")
	}
}

func TestTracer_Export(t *testing.T) {
	tracer := NewTracer()

	span := tracer.StartSpan("exportable", SpanKindAgent)
	span.EndOK()

	data, err := tracer.Export()
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("expected non-empty export")
	}
}

func TestContext_TracerAndSpan(t *testing.T) {
	tracer := NewTracer()
	ctx := ContextWithTracer(context.Background(), tracer)

	// StartSpanFromContext should use the tracer
	span1, ctx := StartSpanFromContext(ctx, "op1", SpanKindAgent)
	span1.EndOK()

	// Nested span should be a child
	span2, _ := StartSpanFromContext(ctx, "op2", SpanKindTool)
	span2.EndOK()

	if span2.ParentID != span1.ID {
		t.Errorf("expected span2 to be child of span1, got parent=%s", span2.ParentID)
	}

	if span2.TraceID != tracer.TraceID() {
		t.Errorf("expected same trace ID")
	}

	spans := tracer.Spans()
	if len(spans) != 2 {
		t.Fatalf("expected 2 spans recorded, got %d", len(spans))
	}
}

func TestTracer_TotalDuration(t *testing.T) {
	tracer := NewTracer()

	s := tracer.StartSpan("timed", SpanKindAgent)
	time.Sleep(5 * time.Millisecond)
	s.EndOK()

	dur := tracer.TotalDuration()
	if dur < 5*time.Millisecond {
		t.Errorf("expected at least 5ms, got %v", dur)
	}
}

func TestAgentWithTracer(t *testing.T) {
	tracer := NewTracer()
	ctx := ContextWithTracer(context.Background(), tracer)

	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "traced output"})

	agent := NewAgent(AgentConfig{
		ID:   "traced-agent",
		Role: "Test",
		Goal: "Test tracing",
		LLM:  mock,
	})

	result, err := agent.Run(ctx, "test input", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "traced output" {
		t.Errorf("expected 'traced output', got %s", result.Content)
	}

	spans := tracer.Spans()
	if len(spans) < 2 {
		t.Fatalf("expected at least 2 spans (agent + llm), got %d", len(spans))
	}

	// First completed span should be the LLM call
	foundLLM := false
	foundAgent := false
	for _, s := range spans {
		if s.Kind == SpanKindLLM {
			foundLLM = true
		}
		if s.Kind == SpanKindAgent {
			foundAgent = true
		}
	}
	if !foundLLM {
		t.Error("expected an LLM span")
	}
	if !foundAgent {
		t.Error("expected an agent span")
	}
}
