package herdai

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Eval — structured testing harness for agent behavior
//
// Define test cases with expected outcomes, run them against agents, and get
// scored results with pass/fail assertions. Supports regression tracking
// by exporting results to JSON.
//
// This is the first eval framework for Go AI agents.
// ═══════════════════════════════════════════════════════════════════════════════

// ── Test Case Definition ───────────────────────────────────────────────────

// EvalCase is a single test scenario for an agent.
type EvalCase struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Input       string         `json:"input"`
	Tags        []string       `json:"tags,omitempty"`
	Assertions  []Assertion    `json:"-"` // programmatic checks
	Timeout     time.Duration  `json:"timeout,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// Assertion is a single check on an agent's output.
type Assertion struct {
	Name  string
	Check func(result *Result, conv *Conversation) *AssertionResult
}

// AssertionResult is the outcome of a single assertion.
type AssertionResult struct {
	Passed  bool   `json:"passed"`
	Name    string `json:"name"`
	Reason  string `json:"reason,omitempty"`
	Got     string `json:"got,omitempty"`
	Want    string `json:"want,omitempty"`
}

// ── Built-in Assertions ────────────────────────────────────────────────────

// AssertContains checks that the output contains a substring.
func AssertContains(substring string) Assertion {
	return Assertion{
		Name: fmt.Sprintf("contains(%q)", substring),
		Check: func(result *Result, _ *Conversation) *AssertionResult {
			if strings.Contains(strings.ToLower(result.Content), strings.ToLower(substring)) {
				return &AssertionResult{Passed: true, Name: fmt.Sprintf("contains(%q)", substring)}
			}
			return &AssertionResult{
				Passed: false,
				Name:   fmt.Sprintf("contains(%q)", substring),
				Reason: fmt.Sprintf("output does not contain %q", substring),
				Got:    truncate(result.Content, 200),
				Want:   substring,
			}
		},
	}
}

// AssertNotContains checks that the output does NOT contain a substring.
func AssertNotContains(substring string) Assertion {
	return Assertion{
		Name: fmt.Sprintf("not_contains(%q)", substring),
		Check: func(result *Result, _ *Conversation) *AssertionResult {
			if !strings.Contains(strings.ToLower(result.Content), strings.ToLower(substring)) {
				return &AssertionResult{Passed: true, Name: fmt.Sprintf("not_contains(%q)", substring)}
			}
			return &AssertionResult{
				Passed: false,
				Name:   fmt.Sprintf("not_contains(%q)", substring),
				Reason: fmt.Sprintf("output should not contain %q", substring),
				Got:    truncate(result.Content, 200),
			}
		},
	}
}

// AssertMinLength checks that the output has at least n characters.
func AssertMinLength(n int) Assertion {
	return Assertion{
		Name: fmt.Sprintf("min_length(%d)", n),
		Check: func(result *Result, _ *Conversation) *AssertionResult {
			if len(result.Content) >= n {
				return &AssertionResult{Passed: true, Name: fmt.Sprintf("min_length(%d)", n)}
			}
			return &AssertionResult{
				Passed: false,
				Name:   fmt.Sprintf("min_length(%d)", n),
				Reason: fmt.Sprintf("output length %d is below minimum %d", len(result.Content), n),
				Got:    fmt.Sprintf("%d chars", len(result.Content)),
				Want:   fmt.Sprintf(">= %d chars", n),
			}
		},
	}
}

// AssertMaxLength checks that the output has at most n characters.
func AssertMaxLength(n int) Assertion {
	return Assertion{
		Name: fmt.Sprintf("max_length(%d)", n),
		Check: func(result *Result, _ *Conversation) *AssertionResult {
			if len(result.Content) <= n {
				return &AssertionResult{Passed: true, Name: fmt.Sprintf("max_length(%d)", n)}
			}
			return &AssertionResult{
				Passed: false,
				Name:   fmt.Sprintf("max_length(%d)", n),
				Reason: fmt.Sprintf("output length %d exceeds maximum %d", len(result.Content), n),
				Got:    fmt.Sprintf("%d chars", len(result.Content)),
				Want:   fmt.Sprintf("<= %d chars", n),
			}
		},
	}
}

// AssertJSON checks that the output is valid JSON.
func AssertJSON() Assertion {
	return Assertion{
		Name: "valid_json",
		Check: func(result *Result, _ *Conversation) *AssertionResult {
			content := strings.TrimSpace(result.Content)
			var js json.RawMessage
			if json.Unmarshal([]byte(content), &js) == nil {
				return &AssertionResult{Passed: true, Name: "valid_json"}
			}
			return &AssertionResult{
				Passed: false,
				Name:   "valid_json",
				Reason: "output is not valid JSON",
				Got:    truncate(content, 200),
			}
		},
	}
}

// AssertToolUsed checks that the agent used a specific tool at least once.
func AssertToolUsed(toolName string) Assertion {
	return Assertion{
		Name: fmt.Sprintf("tool_used(%q)", toolName),
		Check: func(result *Result, conv *Conversation) *AssertionResult {
			if conv == nil {
				return &AssertionResult{
					Passed: false,
					Name:   fmt.Sprintf("tool_used(%q)", toolName),
					Reason: "no conversation provided to check tool usage",
				}
			}
			for _, turn := range conv.GetTurns() {
				if turn.AgentID == result.AgentID {
					for _, tc := range turn.ToolCalls {
						if tc.ToolName == toolName {
							return &AssertionResult{Passed: true, Name: fmt.Sprintf("tool_used(%q)", toolName)}
						}
					}
				}
			}
			return &AssertionResult{
				Passed: false,
				Name:   fmt.Sprintf("tool_used(%q)", toolName),
				Reason: fmt.Sprintf("agent %s did not use tool %q", result.AgentID, toolName),
			}
		},
	}
}

// AssertToolNotUsed checks that the agent did NOT use a specific tool.
func AssertToolNotUsed(toolName string) Assertion {
	return Assertion{
		Name: fmt.Sprintf("tool_not_used(%q)", toolName),
		Check: func(result *Result, conv *Conversation) *AssertionResult {
			if conv == nil {
				return &AssertionResult{Passed: true, Name: fmt.Sprintf("tool_not_used(%q)", toolName)}
			}
			for _, turn := range conv.GetTurns() {
				if turn.AgentID == result.AgentID {
					for _, tc := range turn.ToolCalls {
						if tc.ToolName == toolName {
							return &AssertionResult{
								Passed: false,
								Name:   fmt.Sprintf("tool_not_used(%q)", toolName),
								Reason: fmt.Sprintf("agent %s used tool %q (should not have)", result.AgentID, toolName),
							}
						}
					}
				}
			}
			return &AssertionResult{Passed: true, Name: fmt.Sprintf("tool_not_used(%q)", toolName)}
		},
	}
}

// AssertMaxToolCalls checks that the agent made at most n tool calls.
func AssertMaxToolCalls(n int) Assertion {
	return Assertion{
		Name: fmt.Sprintf("max_tool_calls(%d)", n),
		Check: func(result *Result, _ *Conversation) *AssertionResult {
			tc, _ := result.Metadata["tool_calls"].(int)
			if tc <= n {
				return &AssertionResult{Passed: true, Name: fmt.Sprintf("max_tool_calls(%d)", n)}
			}
			return &AssertionResult{
				Passed: false,
				Name:   fmt.Sprintf("max_tool_calls(%d)", n),
				Reason: fmt.Sprintf("agent made %d tool calls, max is %d", tc, n),
				Got:    fmt.Sprintf("%d", tc),
				Want:   fmt.Sprintf("<= %d", n),
			}
		},
	}
}

// AssertMaxDuration checks the agent completed within a time limit.
func AssertMaxDuration(d time.Duration) Assertion {
	return Assertion{
		Name: fmt.Sprintf("max_duration(%s)", d),
		Check: func(result *Result, _ *Conversation) *AssertionResult {
			ms, _ := result.Metadata["duration_ms"].(int64)
			actual := time.Duration(ms) * time.Millisecond
			if actual <= d {
				return &AssertionResult{Passed: true, Name: fmt.Sprintf("max_duration(%s)", d)}
			}
			return &AssertionResult{
				Passed: false,
				Name:   fmt.Sprintf("max_duration(%s)", d),
				Reason: fmt.Sprintf("agent took %s, max is %s", actual, d),
				Got:    actual.String(),
				Want:   fmt.Sprintf("<= %s", d),
			}
		},
	}
}

// AssertCustom creates an assertion from a simple pass/fail function on the output.
func AssertCustom(name string, check func(output string) (pass bool, reason string)) Assertion {
	return Assertion{
		Name: name,
		Check: func(result *Result, _ *Conversation) *AssertionResult {
			pass, reason := check(result.Content)
			return &AssertionResult{Passed: pass, Name: name, Reason: reason}
		},
	}
}

// ── Eval Runner ────────────────────────────────────────────────────────────

// EvalResult is the outcome of running a single test case.
type EvalResult struct {
	CaseID     string            `json:"case_id"`
	CaseName   string            `json:"case_name"`
	Passed     bool              `json:"passed"`
	Assertions []AssertionResult `json:"assertions"`
	Output     string            `json:"output"`
	Duration   time.Duration     `json:"duration"`
	Error      string            `json:"error,omitempty"`
	Metadata   map[string]any    `json:"metadata,omitempty"`
}

// EvalReport is the aggregate result of running all test cases.
type EvalReport struct {
	Timestamp   time.Time    `json:"timestamp"`
	TotalCases  int          `json:"total_cases"`
	Passed      int          `json:"passed"`
	Failed      int          `json:"failed"`
	Errored     int          `json:"errored"`
	PassRate    float64      `json:"pass_rate"`
	Duration    time.Duration `json:"total_duration"`
	Results     []EvalResult `json:"results"`
	AgentID     string       `json:"agent_id,omitempty"`
	Tags        []string     `json:"tags,omitempty"`
}

// EvalSuite is the test runner that executes eval cases against an agent.
type EvalSuite struct {
	name  string
	cases []EvalCase
	agent Runnable
}

// NewEvalSuite creates a new evaluation suite for an agent.
func NewEvalSuite(name string, agent Runnable) *EvalSuite {
	return &EvalSuite{name: name, agent: agent}
}

// AddCase adds a test case to the suite.
func (es *EvalSuite) AddCase(c EvalCase) {
	if c.ID == "" {
		c.ID = generateID()
	}
	es.cases = append(es.cases, c)
}

// AddCases adds multiple test cases at once.
func (es *EvalSuite) AddCases(cases ...EvalCase) {
	for _, c := range cases {
		es.AddCase(c)
	}
}

// Run executes all test cases and returns a report.
func (es *EvalSuite) Run(ctx context.Context) *EvalReport {
	start := time.Now()
	report := &EvalReport{
		Timestamp:  start,
		TotalCases: len(es.cases),
		AgentID:    es.agent.GetID(),
	}

	for _, tc := range es.cases {
		result := es.runCase(ctx, tc)
		report.Results = append(report.Results, result)
		if result.Error != "" {
			report.Errored++
		} else if result.Passed {
			report.Passed++
		} else {
			report.Failed++
		}
	}

	report.Duration = time.Since(start)
	if report.TotalCases > 0 {
		report.PassRate = float64(report.Passed) / float64(report.TotalCases) * 100
	}

	return report
}

// RunByTag runs only test cases matching the given tag.
func (es *EvalSuite) RunByTag(ctx context.Context, tag string) *EvalReport {
	start := time.Now()

	var filtered []EvalCase
	for _, tc := range es.cases {
		for _, t := range tc.Tags {
			if t == tag {
				filtered = append(filtered, tc)
				break
			}
		}
	}

	report := &EvalReport{
		Timestamp:  start,
		TotalCases: len(filtered),
		AgentID:    es.agent.GetID(),
		Tags:       []string{tag},
	}

	for _, tc := range filtered {
		result := es.runCase(ctx, tc)
		report.Results = append(report.Results, result)
		if result.Error != "" {
			report.Errored++
		} else if result.Passed {
			report.Passed++
		} else {
			report.Failed++
		}
	}

	report.Duration = time.Since(start)
	if report.TotalCases > 0 {
		report.PassRate = float64(report.Passed) / float64(report.TotalCases) * 100
	}

	return report
}

func (es *EvalSuite) runCase(ctx context.Context, tc EvalCase) EvalResult {
	if tc.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, tc.Timeout)
		defer cancel()
	}

	start := time.Now()
	conv := NewConversation()

	result, err := es.agent.Run(ctx, tc.Input, conv)
	dur := time.Since(start)

	if err != nil {
		return EvalResult{
			CaseID:   tc.ID,
			CaseName: tc.Name,
			Passed:   false,
			Duration: dur,
			Error:    err.Error(),
		}
	}

	var assertions []AssertionResult
	allPassed := true

	for _, a := range tc.Assertions {
		ar := a.Check(result, conv)
		assertions = append(assertions, *ar)
		if !ar.Passed {
			allPassed = false
		}
	}

	return EvalResult{
		CaseID:     tc.ID,
		CaseName:   tc.Name,
		Passed:     allPassed,
		Assertions: assertions,
		Output:     result.Content,
		Duration:   dur,
		Metadata:   result.Metadata,
	}
}

// ── Report Output ──────────────────────────────────────────────────────────

// Summary returns a human-readable summary of the eval report.
func (r *EvalReport) Summary() string {
	var sb strings.Builder

	fmt.Fprintf(&sb, "╔══════════════════════════════════════════════════════╗\n")
	fmt.Fprintf(&sb, "║  EVAL REPORT                                       ║\n")
	fmt.Fprintf(&sb, "╠══════════════════════════════════════════════════════╣\n")
	fmt.Fprintf(&sb, "║  Total: %-4d  Passed: %-4d  Failed: %-4d  Err: %-3d  ║\n",
		r.TotalCases, r.Passed, r.Failed, r.Errored)
	fmt.Fprintf(&sb, "║  Pass Rate: %5.1f%%           Duration: %-12s  ║\n",
		r.PassRate, r.Duration.Round(time.Millisecond))
	fmt.Fprintf(&sb, "╚══════════════════════════════════════════════════════╝\n\n")

	for _, res := range r.Results {
		status := "✓ PASS"
		if res.Error != "" {
			status = "✗ ERR "
		} else if !res.Passed {
			status = "✗ FAIL"
		}

		fmt.Fprintf(&sb, "%s  %s (%s)\n", status, res.CaseName, res.Duration.Round(time.Millisecond))

		if res.Error != "" {
			fmt.Fprintf(&sb, "        error: %s\n", res.Error)
		}

		for _, a := range res.Assertions {
			if a.Passed {
				fmt.Fprintf(&sb, "        ✓ %s\n", a.Name)
			} else {
				fmt.Fprintf(&sb, "        ✗ %s: %s\n", a.Name, a.Reason)
				if a.Got != "" {
					fmt.Fprintf(&sb, "          got:  %s\n", a.Got)
				}
				if a.Want != "" {
					fmt.Fprintf(&sb, "          want: %s\n", a.Want)
				}
			}
		}
	}

	return sb.String()
}

// ExportJSON saves the report as a JSON file for regression tracking.
func (r *EvalReport) ExportJSON(path string) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal eval report: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// ── Regression Comparison ──────────────────────────────────────────────────

// CompareReports compares two eval reports and returns a diff summary.
func CompareReports(baseline, current *EvalReport) string {
	var sb strings.Builder

	fmt.Fprintf(&sb, "Regression Comparison\n")
	fmt.Fprintf(&sb, "═════════════════════\n")
	fmt.Fprintf(&sb, "Baseline: %d/%d passed (%.1f%%)\n", baseline.Passed, baseline.TotalCases, baseline.PassRate)
	fmt.Fprintf(&sb, "Current:  %d/%d passed (%.1f%%)\n", current.Passed, current.TotalCases, current.PassRate)

	diff := current.PassRate - baseline.PassRate
	if diff > 0 {
		fmt.Fprintf(&sb, "Change:   +%.1f%% ↑ IMPROVED\n\n", diff)
	} else if diff < 0 {
		fmt.Fprintf(&sb, "Change:   %.1f%% ↓ REGRESSION\n\n", diff)
	} else {
		fmt.Fprintf(&sb, "Change:   0%% → NO CHANGE\n\n")
	}

	baselineMap := make(map[string]EvalResult)
	for _, r := range baseline.Results {
		baselineMap[r.CaseID] = r
	}

	for _, cur := range current.Results {
		prev, found := baselineMap[cur.CaseID]
		if !found {
			fmt.Fprintf(&sb, "  NEW   %s\n", cur.CaseName)
			continue
		}
		if prev.Passed && !cur.Passed {
			fmt.Fprintf(&sb, "  REGR  %s (was passing, now failing)\n", cur.CaseName)
		} else if !prev.Passed && cur.Passed {
			fmt.Fprintf(&sb, "  FIXED %s (was failing, now passing)\n", cur.CaseName)
		}
	}

	return sb.String()
}

// LoadReport loads a previously exported eval report from JSON.
func LoadReport(path string) (*EvalReport, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read eval report: %w", err)
	}
	var report EvalReport
	if err := json.Unmarshal(data, &report); err != nil {
		return nil, fmt.Errorf("unmarshal eval report: %w", err)
	}
	return &report, nil
}
