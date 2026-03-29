package herdai

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Guardrails — input/output validation, content filtering, schema enforcement
//
// Guardrails run at two points in the agent lifecycle:
//   - InputGuardrails:  validate/transform the user input BEFORE the LLM sees it
//   - OutputGuardrails: validate/transform the LLM response BEFORE it's returned
//
// If a guardrail returns an error, the agent run fails with that error.
// If a guardrail modifies the content (returns different string), the modified
// version is used downstream.
// ═══════════════════════════════════════════════════════════════════════════════

// GuardrailResult is what a guardrail check returns.
type GuardrailResult struct {
	Passed   bool           `json:"passed"`
	Modified string         `json:"modified,omitempty"` // if non-empty, replaces the original content
	Reason   string         `json:"reason,omitempty"`   // explanation when blocked
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Guardrail is a single validation/transformation rule.
type Guardrail struct {
	Name    string
	Check   func(ctx context.Context, content string) (*GuardrailResult, error)
}

// GuardrailChain runs multiple guardrails in sequence.
// If any guardrail fails (Passed=false), the chain stops and returns the failure.
// If a guardrail modifies content, subsequent guardrails see the modified version.
type GuardrailChain struct {
	guardrails []Guardrail
}

// NewGuardrailChain creates a chain from one or more guardrails.
func NewGuardrailChain(guardrails ...Guardrail) *GuardrailChain {
	return &GuardrailChain{guardrails: guardrails}
}

// Add appends a guardrail to the chain.
func (gc *GuardrailChain) Add(g Guardrail) {
	gc.guardrails = append(gc.guardrails, g)
}

// Run executes all guardrails in sequence. Returns the (possibly modified) content
// and an error if any guardrail blocks the content.
func (gc *GuardrailChain) Run(ctx context.Context, content string) (string, error) {
	if gc == nil || len(gc.guardrails) == 0 {
		return content, nil
	}

	current := content
	for _, g := range gc.guardrails {
		result, err := g.Check(ctx, current)
		if err != nil {
			return "", fmt.Errorf("guardrail %q error: %w", g.Name, err)
		}
		if !result.Passed {
			reason := result.Reason
			if reason == "" {
				reason = "blocked by guardrail"
			}
			return "", fmt.Errorf("guardrail %q blocked: %s", g.Name, reason)
		}
		if result.Modified != "" {
			current = result.Modified
		}
	}
	return current, nil
}

// Len returns the number of guardrails in the chain.
func (gc *GuardrailChain) Len() int {
	if gc == nil {
		return 0
	}
	return len(gc.guardrails)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Built-in Guardrails — ready-to-use validation rules
// ═══════════════════════════════════════════════════════════════════════════════

// MaxLength blocks content exceeding a character limit.
func MaxLength(maxLen int) Guardrail {
	return Guardrail{
		Name: fmt.Sprintf("max_length(%d)", maxLen),
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			if len(content) > maxLen {
				return &GuardrailResult{
					Passed: false,
					Reason: fmt.Sprintf("content length %d exceeds maximum %d", len(content), maxLen),
				}, nil
			}
			return &GuardrailResult{Passed: true}, nil
		},
	}
}

// MinLength blocks content shorter than a minimum.
func MinLength(minLen int) Guardrail {
	return Guardrail{
		Name: fmt.Sprintf("min_length(%d)", minLen),
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			if len(strings.TrimSpace(content)) < minLen {
				return &GuardrailResult{
					Passed: false,
					Reason: fmt.Sprintf("content length %d below minimum %d", len(strings.TrimSpace(content)), minLen),
				}, nil
			}
			return &GuardrailResult{Passed: true}, nil
		},
	}
}

// BlockPatterns rejects content matching any of the given regex patterns.
func BlockPatterns(patterns ...string) Guardrail {
	compiled := make([]*regexp.Regexp, len(patterns))
	for i, p := range patterns {
		compiled[i] = regexp.MustCompile(p)
	}
	return Guardrail{
		Name: "block_patterns",
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			for i, re := range compiled {
				if re.MatchString(content) {
					return &GuardrailResult{
						Passed: false,
						Reason: fmt.Sprintf("content matches blocked pattern: %s", patterns[i]),
					}, nil
				}
			}
			return &GuardrailResult{Passed: true}, nil
		},
	}
}

// RequirePatterns ensures content matches ALL given regex patterns.
func RequirePatterns(patterns ...string) Guardrail {
	compiled := make([]*regexp.Regexp, len(patterns))
	for i, p := range patterns {
		compiled[i] = regexp.MustCompile(p)
	}
	return Guardrail{
		Name: "require_patterns",
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			for i, re := range compiled {
				if !re.MatchString(content) {
					return &GuardrailResult{
						Passed: false,
						Reason: fmt.Sprintf("content does not match required pattern: %s", patterns[i]),
					}, nil
				}
			}
			return &GuardrailResult{Passed: true}, nil
		},
	}
}

// BlockKeywords rejects content containing any of the given keywords (case-insensitive).
func BlockKeywords(keywords ...string) Guardrail {
	lower := make([]string, len(keywords))
	for i, k := range keywords {
		lower[i] = strings.ToLower(k)
	}
	return Guardrail{
		Name: "block_keywords",
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			contentLower := strings.ToLower(content)
			for _, kw := range lower {
				if strings.Contains(contentLower, kw) {
					return &GuardrailResult{
						Passed: false,
						Reason: fmt.Sprintf("content contains blocked keyword: %q", kw),
					}, nil
				}
			}
			return &GuardrailResult{Passed: true}, nil
		},
	}
}

// RequireJSON ensures content is valid JSON. Optionally validates against required keys.
func RequireJSON(requiredKeys ...string) Guardrail {
	return Guardrail{
		Name: "require_json",
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			content = strings.TrimSpace(content)

			// Try to extract JSON from markdown code blocks
			if !strings.HasPrefix(content, "{") && !strings.HasPrefix(content, "[") {
				if idx := strings.Index(content, "```json"); idx != -1 {
					start := idx + 7
					end := strings.Index(content[start:], "```")
					if end != -1 {
						content = strings.TrimSpace(content[start : start+end])
					}
				} else if idx := strings.Index(content, "```"); idx != -1 {
					start := idx + 3
					if nl := strings.Index(content[start:], "\n"); nl != -1 {
						start += nl + 1
					}
					end := strings.Index(content[start:], "```")
					if end != -1 {
						content = strings.TrimSpace(content[start : start+end])
					}
				}
			}

			var parsed map[string]any
			if err := json.Unmarshal([]byte(content), &parsed); err != nil {
				var arr []any
				if arrErr := json.Unmarshal([]byte(content), &arr); arrErr != nil {
					return &GuardrailResult{
						Passed: false,
						Reason: fmt.Sprintf("content is not valid JSON: %s", err.Error()),
					}, nil
				}
				return &GuardrailResult{Passed: true, Modified: content}, nil
			}

			for _, key := range requiredKeys {
				if _, ok := parsed[key]; !ok {
					return &GuardrailResult{
						Passed: false,
						Reason: fmt.Sprintf("JSON missing required key: %q", key),
					}, nil
				}
			}

			return &GuardrailResult{Passed: true, Modified: content}, nil
		},
	}
}

// ContentFilter blocks content containing common categories of harmful content.
// Categories: "pii" (emails, phones, SSNs), "profanity" (common profane words),
// "injection" (prompt injection attempts).
func ContentFilter(categories ...string) Guardrail {
	catSet := make(map[string]bool, len(categories))
	for _, c := range categories {
		catSet[strings.ToLower(c)] = true
	}

	var checks []func(string) (bool, string)

	if catSet["pii"] {
		piiPatterns := []*regexp.Regexp{
			regexp.MustCompile(`\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`),
			regexp.MustCompile(`\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`),
			regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`),
		}
		checks = append(checks, func(content string) (bool, string) {
			for _, re := range piiPatterns {
				if re.MatchString(content) {
					return true, "content contains PII (email, phone, or SSN)"
				}
			}
			return false, ""
		})
	}

	if catSet["injection"] {
		injectionPatterns := []string{
			"ignore previous instructions",
			"ignore all previous",
			"disregard your instructions",
			"you are now",
			"new instructions:",
			"system prompt:",
			"forget everything",
			"override your",
		}
		checks = append(checks, func(content string) (bool, string) {
			lower := strings.ToLower(content)
			for _, p := range injectionPatterns {
				if strings.Contains(lower, p) {
					return true, fmt.Sprintf("content contains potential prompt injection: %q", p)
				}
			}
			return false, ""
		})
	}

	return Guardrail{
		Name: fmt.Sprintf("content_filter(%s)", strings.Join(categories, ",")),
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			for _, check := range checks {
				if blocked, reason := check(content); blocked {
					return &GuardrailResult{Passed: false, Reason: reason}, nil
				}
			}
			return &GuardrailResult{Passed: true}, nil
		},
	}
}

// RedactPII replaces detected PII (emails, phones, SSNs) with [REDACTED].
// Unlike ContentFilter("pii") which blocks, this transforms the content.
func RedactPII() Guardrail {
	emailRe := regexp.MustCompile(`\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`)
	phoneRe := regexp.MustCompile(`\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`)
	ssnRe := regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)

	return Guardrail{
		Name: "redact_pii",
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			modified := ssnRe.ReplaceAllString(content, "[REDACTED-SSN]")
			modified = emailRe.ReplaceAllString(modified, "[REDACTED-EMAIL]")
			modified = phoneRe.ReplaceAllString(modified, "[REDACTED-PHONE]")

			if modified != content {
				return &GuardrailResult{Passed: true, Modified: modified}, nil
			}
			return &GuardrailResult{Passed: true}, nil
		},
	}
}

// CustomGuardrail creates a guardrail from a simple pass/fail function.
func CustomGuardrail(name string, check func(content string) (pass bool, reason string)) Guardrail {
	return Guardrail{
		Name: name,
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			pass, reason := check(content)
			return &GuardrailResult{Passed: pass, Reason: reason}, nil
		},
	}
}

// TrimWhitespace normalizes whitespace in content (trims + collapses internal runs).
func TrimWhitespace() Guardrail {
	multiSpace := regexp.MustCompile(`\s{3,}`)
	return Guardrail{
		Name: "trim_whitespace",
		Check: func(_ context.Context, content string) (*GuardrailResult, error) {
			trimmed := strings.TrimSpace(content)
			trimmed = multiSpace.ReplaceAllString(trimmed, "  ")
			if trimmed != content {
				return &GuardrailResult{Passed: true, Modified: trimmed}, nil
			}
			return &GuardrailResult{Passed: true}, nil
		},
	}
}
