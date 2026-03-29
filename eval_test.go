package herdai

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestAssertContains(t *testing.T) {
	a := AssertContains("hello")
	r := a.Check(&Result{Content: "Hello World"}, nil)
	if !r.Passed {
		t.Error("case-insensitive 'hello' should match 'Hello World'")
	}

	r = a.Check(&Result{Content: "goodbye"}, nil)
	if r.Passed {
		t.Error("'goodbye' should not match 'hello'")
	}
}

func TestAssertNotContains(t *testing.T) {
	a := AssertNotContains("error")
	r := a.Check(&Result{Content: "all good"}, nil)
	if !r.Passed {
		t.Error("'all good' should pass not_contains('error')")
	}

	r = a.Check(&Result{Content: "an error occurred"}, nil)
	if r.Passed {
		t.Error("should fail when content contains 'error'")
	}
}

func TestAssertMinLength(t *testing.T) {
	a := AssertMinLength(10)
	r := a.Check(&Result{Content: "short"}, nil)
	if r.Passed {
		t.Error("5 chars should fail min_length(10)")
	}

	r = a.Check(&Result{Content: "this is long enough"}, nil)
	if !r.Passed {
		t.Error("19 chars should pass min_length(10)")
	}
}

func TestAssertMaxLength(t *testing.T) {
	a := AssertMaxLength(10)
	r := a.Check(&Result{Content: "short"}, nil)
	if !r.Passed {
		t.Error("5 chars should pass max_length(10)")
	}

	r = a.Check(&Result{Content: "this is way too long"}, nil)
	if r.Passed {
		t.Error("20 chars should fail max_length(10)")
	}
}

func TestAssertJSON(t *testing.T) {
	a := AssertJSON()
	r := a.Check(&Result{Content: `{"key": "value"}`}, nil)
	if !r.Passed {
		t.Error("valid JSON should pass")
	}

	r = a.Check(&Result{Content: "not json"}, nil)
	if r.Passed {
		t.Error("invalid JSON should fail")
	}
}

func TestAssertToolUsed(t *testing.T) {
	conv := NewConversation()
	conv.AddTurn(Turn{
		AgentID: "agent-1",
		Role:    "assistant",
		Content: "done",
		ToolCalls: []ToolCallRecord{
			{ToolName: "search", Output: "found it"},
		},
	})

	a := AssertToolUsed("search")
	r := a.Check(&Result{AgentID: "agent-1"}, conv)
	if !r.Passed {
		t.Error("should pass when tool was used")
	}

	a2 := AssertToolUsed("delete")
	r2 := a2.Check(&Result{AgentID: "agent-1"}, conv)
	if r2.Passed {
		t.Error("should fail when tool was not used")
	}
}

func TestAssertToolNotUsed(t *testing.T) {
	conv := NewConversation()
	conv.AddTurn(Turn{
		AgentID: "agent-1",
		Role:    "assistant",
		ToolCalls: []ToolCallRecord{
			{ToolName: "search"},
		},
	})

	a := AssertToolNotUsed("delete")
	r := a.Check(&Result{AgentID: "agent-1"}, conv)
	if !r.Passed {
		t.Error("should pass when tool was NOT used")
	}

	a2 := AssertToolNotUsed("search")
	r2 := a2.Check(&Result{AgentID: "agent-1"}, conv)
	if r2.Passed {
		t.Error("should fail when tool WAS used")
	}
}

func TestAssertMaxToolCalls(t *testing.T) {
	a := AssertMaxToolCalls(3)
	r := a.Check(&Result{Metadata: map[string]any{"tool_calls": 2}}, nil)
	if !r.Passed {
		t.Error("2 calls should pass max_tool_calls(3)")
	}

	r = a.Check(&Result{Metadata: map[string]any{"tool_calls": 5}}, nil)
	if r.Passed {
		t.Error("5 calls should fail max_tool_calls(3)")
	}
}

func TestAssertMaxDuration(t *testing.T) {
	a := AssertMaxDuration(5 * time.Second)
	r := a.Check(&Result{Metadata: map[string]any{"duration_ms": int64(2000)}}, nil)
	if !r.Passed {
		t.Error("2s should pass max_duration(5s)")
	}

	r = a.Check(&Result{Metadata: map[string]any{"duration_ms": int64(10000)}}, nil)
	if r.Passed {
		t.Error("10s should fail max_duration(5s)")
	}
}

func TestAssertCustom(t *testing.T) {
	a := AssertCustom("has_bullet_points", func(output string) (bool, string) {
		if len(output) > 0 && (output[0] == '-' || output[0] == '*') {
			return true, ""
		}
		return false, "output should start with a bullet point"
	})

	r := a.Check(&Result{Content: "- first point"}, nil)
	if !r.Passed {
		t.Error("bullet point output should pass")
	}

	r = a.Check(&Result{Content: "no bullets here"}, nil)
	if r.Passed {
		t.Error("non-bullet output should fail")
	}
}

func TestEvalSuite_BasicRun(t *testing.T) {
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "Go is a great language for building systems"})
	mock.PushResponse(LLMResponse{Content: "Python is popular for data science"})

	agent := NewAgent(AgentConfig{
		ID:   "eval-agent",
		Role: "Analyst",
		Goal: "Answer questions",
		LLM:  mock,
	})

	suite := NewEvalSuite("basic-eval", agent)
	suite.AddCase(EvalCase{
		ID:    "q1",
		Name:  "Go question",
		Input: "Tell me about Go",
		Assertions: []Assertion{
			AssertContains("go"),
			AssertMinLength(10),
		},
		Tags: []string{"language"},
	})
	suite.AddCase(EvalCase{
		ID:    "q2",
		Name:  "Python question",
		Input: "Tell me about Python",
		Assertions: []Assertion{
			AssertContains("python"),
			AssertMinLength(10),
		},
		Tags: []string{"language"},
	})

	report := suite.Run(context.Background())

	if report.TotalCases != 2 {
		t.Fatalf("expected 2 cases, got %d", report.TotalCases)
	}
	if report.Passed != 2 {
		t.Fatalf("expected 2 passed, got %d passed / %d failed", report.Passed, report.Failed)
	}
	if report.PassRate != 100 {
		t.Errorf("expected 100%% pass rate, got %.1f%%", report.PassRate)
	}
}

func TestEvalSuite_FailingCase(t *testing.T) {
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "short"})

	agent := NewAgent(AgentConfig{
		ID:   "eval-fail",
		Role: "Test",
		Goal: "Test failure",
		LLM:  mock,
	})

	suite := NewEvalSuite("fail-eval", agent)
	suite.AddCase(EvalCase{
		ID:    "f1",
		Name:  "Requires long output",
		Input: "Say something",
		Assertions: []Assertion{
			AssertMinLength(100),
			AssertContains("specific keyword"),
		},
	})

	report := suite.Run(context.Background())

	if report.Passed != 0 {
		t.Error("case should have failed")
	}
	if report.Failed != 1 {
		t.Fatalf("expected 1 failure, got %d", report.Failed)
	}

	failedAssertions := 0
	for _, a := range report.Results[0].Assertions {
		if !a.Passed {
			failedAssertions++
		}
	}
	if failedAssertions != 2 {
		t.Errorf("expected 2 failed assertions, got %d", failedAssertions)
	}
}

func TestEvalSuite_RunByTag(t *testing.T) {
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "Go is compiled"})

	agent := NewAgent(AgentConfig{
		ID:   "tag-agent",
		Role: "Test",
		Goal: "Test tags",
		LLM:  mock,
	})

	suite := NewEvalSuite("tag-eval", agent)
	suite.AddCase(EvalCase{
		ID:    "t1",
		Name:  "Tagged case",
		Input: "Go?",
		Tags:  []string{"fast"},
		Assertions: []Assertion{
			AssertContains("go"),
		},
	})
	suite.AddCase(EvalCase{
		ID:    "t2",
		Name:  "Untagged case",
		Input: "Python?",
		Tags:  []string{"slow"},
		Assertions: []Assertion{
			AssertContains("python"),
		},
	})

	report := suite.RunByTag(context.Background(), "fast")

	if report.TotalCases != 1 {
		t.Fatalf("expected 1 case for tag 'fast', got %d", report.TotalCases)
	}
	if report.Passed != 1 {
		t.Errorf("expected 1 pass, got %d", report.Passed)
	}
}

func TestEvalReport_Summary(t *testing.T) {
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "answer"})

	agent := NewAgent(AgentConfig{
		ID:   "summary-agent",
		Role: "Test",
		Goal: "Test summary",
		LLM:  mock,
	})

	suite := NewEvalSuite("summary-eval", agent)
	suite.AddCase(EvalCase{
		ID:         "s1",
		Name:       "Summary test",
		Input:      "test",
		Assertions: []Assertion{AssertContains("answer")},
	})

	report := suite.Run(context.Background())
	summary := report.Summary()

	if summary == "" {
		t.Fatal("expected non-empty summary")
	}
	if len(summary) < 50 {
		t.Error("summary seems too short")
	}
}

func TestEvalReport_ExportAndLoad(t *testing.T) {
	report := &EvalReport{
		Timestamp:  time.Now(),
		TotalCases: 2,
		Passed:     1,
		Failed:     1,
		PassRate:   50,
		Duration:   time.Second,
		Results: []EvalResult{
			{CaseID: "1", CaseName: "test1", Passed: true},
			{CaseID: "2", CaseName: "test2", Passed: false},
		},
	}

	path := t.TempDir() + "/report.json"
	if err := report.ExportJSON(path); err != nil {
		t.Fatalf("export failed: %v", err)
	}

	loaded, err := LoadReport(path)
	if err != nil {
		t.Fatalf("load failed: %v", err)
	}

	if loaded.TotalCases != 2 {
		t.Errorf("expected 2 cases, got %d", loaded.TotalCases)
	}
	if loaded.PassRate != 50 {
		t.Errorf("expected 50%% pass rate, got %.1f%%", loaded.PassRate)
	}
}

func TestCompareReports(t *testing.T) {
	baseline := &EvalReport{
		TotalCases: 3,
		Passed:     2,
		Failed:     1,
		PassRate:   66.7,
		Results: []EvalResult{
			{CaseID: "1", CaseName: "test1", Passed: true},
			{CaseID: "2", CaseName: "test2", Passed: true},
			{CaseID: "3", CaseName: "test3", Passed: false},
		},
	}

	current := &EvalReport{
		TotalCases: 4,
		Passed:     3,
		Failed:     1,
		PassRate:   75,
		Results: []EvalResult{
			{CaseID: "1", CaseName: "test1", Passed: true},
			{CaseID: "2", CaseName: "test2", Passed: false}, // regression
			{CaseID: "3", CaseName: "test3", Passed: true},  // fixed
			{CaseID: "4", CaseName: "test4", Passed: true},  // new
		},
	}

	diff := CompareReports(baseline, current)
	if diff == "" {
		t.Fatal("expected non-empty diff")
	}
	// Should mention the regression and fix
	if !contains(diff, "REGR") {
		t.Error("diff should mention regression for test2")
	}
	if !contains(diff, "FIXED") {
		t.Error("diff should mention fix for test3")
	}
	if !contains(diff, "NEW") {
		t.Error("diff should mention new test4")
	}
}

func TestLoadReport_NotFound(t *testing.T) {
	_, err := LoadReport("/nonexistent/report.json")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestEvalSuite_ErrorCase(t *testing.T) {
	mock := &MockLLM{} // no responses pushed — will error

	agent := NewAgent(AgentConfig{
		ID:      "error-agent",
		Role:    "Test",
		Goal:    "Test errors",
		LLM:     mock,
		Timeout: 100 * time.Millisecond,
	})

	suite := NewEvalSuite("error-eval", agent)
	suite.AddCase(EvalCase{
		ID:         "e1",
		Name:       "Error case",
		Input:      "test",
		Assertions: []Assertion{AssertContains("never")},
	})

	report := suite.Run(context.Background())

	if report.Errored != 1 {
		t.Fatalf("expected 1 error, got %d", report.Errored)
	}
	if report.Results[0].Error == "" {
		t.Error("expected error message")
	}
}

func TestEvalSuite_WithToolAssertions(t *testing.T) {
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{
		ToolCalls: []ToolCall{
			{ID: "tc1", Function: "search", Args: map[string]any{"q": "go"}},
		},
	})
	mock.PushResponse(LLMResponse{Content: "Go is great"})

	agent := NewAgent(AgentConfig{
		ID:   "tool-eval",
		Role: "Test",
		Goal: "Test tool assertions",
		LLM:  mock,
		Tools: []Tool{
			{
				Name:        "search",
				Description: "Search",
				Execute: func(_ context.Context, _ map[string]any) (string, error) {
					return "result", nil
				},
			},
		},
	})

	suite := NewEvalSuite("tool-eval", agent)
	suite.AddCase(EvalCase{
		ID:    "tools1",
		Name:  "Uses search tool",
		Input: "Search for Go",
		Assertions: []Assertion{
			AssertToolUsed("search"),
			AssertToolNotUsed("delete"),
			AssertContains("go"),
		},
	})

	report := suite.Run(context.Background())

	if report.Passed != 1 {
		t.Fatalf("expected 1 pass, got %d passed / %d failed", report.Passed, report.Failed)
		for _, r := range report.Results {
			for _, a := range r.Assertions {
				if !a.Passed {
					t.Logf("  FAIL: %s — %s", a.Name, a.Reason)
				}
			}
		}
	}
}

func TestExportJSON_TempFile(t *testing.T) {
	report := &EvalReport{TotalCases: 1, Passed: 1, PassRate: 100}
	path := t.TempDir() + "/test_report.json"

	if err := report.ExportJSON(path); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("exported file should not be empty")
	}
}

func contains(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 && len(s) >= len(substr) && containsStr(s, substr)
}

func containsStr(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
