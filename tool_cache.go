package herdai

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"sync"
	"time"
)

// ToolCache provides intelligent caching for tool results at the framework level.
// It tracks what context each result was computed from and auto-invalidates
// when the context changes meaningfully.
//
// Usage:
//
//	cache := herdai.NewToolCache(herdai.ToolCacheConfig{
//	    NewWordThreshold: 3,       // re-run if 3+ meaningful words changed (added or replaced)
//	    MaxAge:           10*time.Minute,
//	    MaxEntries:       50,
//	})
//
//	// Wrap a tool with caching:
//	tool := herdai.Tool{
//	    Name: "my_tool",
//	    Execute: cache.Wrap("my_tool", originalHandler),
//	}
type ToolCache struct {
	mu            sync.RWMutex
	entries       map[string]*CacheEntry
	config        ToolCacheConfig
	contextFields map[string]string // current structured context (field→value)
}

// ToolCacheConfig controls cache behavior.
type ToolCacheConfig struct {
	// NewWordThreshold is the minimum number of meaningful changed words
	// (added OR removed/replaced) that triggers a cache refresh. Default: 3.
	// The system checks both directions: words appearing in new context that
	// weren't in old, AND words that disappeared from old context. This catches
	// concept replacements like "architects" → "hospitals" (1 removed + 1 added = 2 changes).
	// This is the fallback when no ContextFields are provided.
	NewWordThreshold int

	// MaxAge is the maximum time a cached result is valid. Default: 0 (no expiry).
	MaxAge time.Duration

	// MaxEntries caps the number of cached tool results. Default: 100.
	MaxEntries int

	// ToolDeps maps tool names to the context field keys they depend on.
	// When SetContextFields is called, only tools whose dependent fields changed
	// are selectively invalidated — other tools keep their cache.
	//
	// Example:
	//   ToolDeps: map[string][]string{
	//       "financial_analysis": {"idea", "industry", "revenue", "customer"},
	//       "competitor_intel":   {"idea", "industry", "customer", "competitors"},
	//       "gtm_analysis":      {"idea", "customer", "geography", "revenue"},
	//   }
	ToolDeps map[string][]string
}

// CacheEntry stores a tool result alongside the context it was computed from.
type CacheEntry struct {
	ToolName      string            `json:"tool_name"`
	Result        string            `json:"result"`
	ContextKey    string            `json:"context_key"`
	Context       string            `json:"context"`
	ContextFields map[string]string `json:"context_fields,omitempty"` // structured fields snapshot at cache time
	CreatedAt     time.Time         `json:"created_at"`
	HitCount      int               `json:"hit_count"`
}

// NewToolCache creates a new cache with the given configuration.
func NewToolCache(cfg ToolCacheConfig) *ToolCache {
	if cfg.NewWordThreshold <= 0 {
		cfg.NewWordThreshold = 3
	}
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = 100
	}
	return &ToolCache{
		entries:       make(map[string]*CacheEntry),
		config:        cfg,
		contextFields: make(map[string]string),
	}
}

// Get returns a cached result if it exists and is still valid for the given context.
// Returns the result and true if cache hit, empty string and false if miss.
func (tc *ToolCache) Get(toolName, context string) (string, bool) {
	tc.mu.RLock()
	entry, exists := tc.entries[toolName]
	tc.mu.RUnlock()

	if !exists {
		return "", false
	}

	if tc.config.MaxAge > 0 && time.Since(entry.CreatedAt) > tc.config.MaxAge {
		return "", false
	}

	if tc.shouldRefresh(entry.Context, context) {
		return "", false
	}

	tc.mu.Lock()
	entry.HitCount++
	tc.mu.Unlock()

	return entry.Result, true
}

// Set stores a tool result with its associated context.
func (tc *ToolCache) Set(toolName, context, result string) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	if len(tc.entries) >= tc.config.MaxEntries {
		tc.evictOldest()
	}

	fields := make(map[string]string, len(tc.contextFields))
	for k, v := range tc.contextFields {
		fields[k] = v
	}

	tc.entries[toolName] = &CacheEntry{
		ToolName:      toolName,
		Result:        result,
		ContextKey:    hashContext(context),
		Context:       context,
		ContextFields: fields,
		CreatedAt:     time.Now(),
	}
}

// Invalidate removes a specific tool's cached result.
func (tc *ToolCache) Invalidate(toolName string) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	delete(tc.entries, toolName)
}

// InvalidateAll clears the entire cache and context fields.
func (tc *ToolCache) InvalidateAll() {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	tc.entries = make(map[string]*CacheEntry)
	tc.contextFields = make(map[string]string)
}

// SetContextFields updates the structured context fields and selectively
// invalidates tools whose dependent fields changed. This is the Level 2
// field-aware cache invalidation — it knows WHICH tools care about WHICH
// fields, so changing "customer" only invalidates competitor_intel, gtm, etc.
// but leaves strategic_analysis cached if it doesn't depend on "customer".
//
// Returns the list of tool names that were invalidated.
func (tc *ToolCache) SetContextFields(fields map[string]string) []string {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	// Find which fields actually changed (value is different or new)
	changedFields := map[string]bool{}
	for key, newVal := range fields {
		oldVal, existed := tc.contextFields[key]
		newVal = strings.TrimSpace(newVal)
		if newVal == "" {
			continue
		}
		if !existed || !fieldValuesMatch(oldVal, newVal) {
			changedFields[key] = true
		}
	}

	// Update stored fields
	for k, v := range fields {
		v = strings.TrimSpace(v)
		if v != "" {
			tc.contextFields[k] = v
		}
	}

	if len(changedFields) == 0 {
		return nil
	}

	// Determine which tools to invalidate based on ToolDeps
	var invalidated []string
	if len(tc.config.ToolDeps) > 0 {
		for toolName, deps := range tc.config.ToolDeps {
			for _, dep := range deps {
				if changedFields[dep] {
					delete(tc.entries, toolName)
					invalidated = append(invalidated, toolName)
					break
				}
			}
		}
	} else {
		// No deps configured — any field change invalidates everything
		for k := range tc.entries {
			invalidated = append(invalidated, k)
		}
		tc.entries = make(map[string]*CacheEntry)
	}

	return invalidated
}

// GetContextFields returns a copy of the current context fields.
func (tc *ToolCache) GetContextFields() map[string]string {
	tc.mu.RLock()
	defer tc.mu.RUnlock()
	out := make(map[string]string, len(tc.contextFields))
	for k, v := range tc.contextFields {
		out[k] = v
	}
	return out
}

// fieldValuesMatch compares two field values ignoring case, whitespace, and stop words.
func fieldValuesMatch(a, b string) bool {
	a = strings.TrimSpace(strings.ToLower(a))
	b = strings.TrimSpace(strings.ToLower(b))
	if a == b {
		return true
	}
	aWords := meaningfulWords(a)
	bWords := meaningfulWords(b)
	if len(aWords) != len(bWords) {
		return false
	}
	aSet := make(map[string]bool, len(aWords))
	for _, w := range aWords {
		aSet[w] = true
	}
	for _, w := range bWords {
		if !aSet[w] {
			return false
		}
	}
	return true
}

// Wrap returns a ToolHandler that transparently caches results.
// The wrapped handler extracts the "context" argument to determine cache validity.
// If "refresh" is true in args, the cache is bypassed.
func (tc *ToolCache) Wrap(toolName string, handler ToolHandler) ToolHandler {
	return func(ctx context.Context, args map[string]any) (string, error) {
		inputCtx, _ := args["context"].(string)
		refresh, _ := args["refresh"].(bool)

		if !refresh {
			if result, ok := tc.Get(toolName, inputCtx); ok {
				return result, nil
			}
		}

		result, err := handler(ctx, args)
		if err != nil {
			return "", err
		}

		tc.Set(toolName, inputCtx, result)
		return result, nil
	}
}

// Entries returns a snapshot of all cache entries (for inspection/debugging).
func (tc *ToolCache) Entries() map[string]CacheEntry {
	tc.mu.RLock()
	defer tc.mu.RUnlock()
	out := make(map[string]CacheEntry, len(tc.entries))
	for k, v := range tc.entries {
		out[k] = *v
	}
	return out
}

// shouldRefresh detects whether the context has changed meaningfully.
// It checks three signals:
//  1. New concepts added (words in new context absent from old)
//  2. Concepts removed or replaced (words in old context absent from new)
//  3. Significant length change
//
// Stop words (the, a, an, for, etc.) are filtered so only meaningful
// term changes trigger a refresh. This catches small but critical pivots
// like "architects" → "hospitals" which is only 2 words changed but
// fundamentally alters the analysis.
func (tc *ToolCache) shouldRefresh(cachedCtx, newCtx string) bool {
	if cachedCtx == "" {
		return true
	}
	if cachedCtx == newCtx {
		return false
	}

	if len(newCtx) > len(cachedCtx)+100 {
		return true
	}

	oldWords := meaningfulWords(cachedCtx)
	newWords := meaningfulWords(newCtx)

	oldSet := make(map[string]bool, len(oldWords))
	for _, w := range oldWords {
		oldSet[w] = true
	}
	newSet := make(map[string]bool, len(newWords))
	for _, w := range newWords {
		newSet[w] = true
	}

	added := 0
	for w := range newSet {
		if !oldSet[w] {
			added++
		}
	}

	removed := 0
	for w := range oldSet {
		if !newSet[w] {
			removed++
		}
	}

	// Total concept drift = new terms + replaced/removed terms.
	// Even a single word swap ("architects" → "hospitals") produces
	// added=1 + removed=1 = 2, which clears the default threshold of 3
	// when combined with any other small change (e.g., "small" → "enterprise").
	totalDrift := added + removed
	return totalDrift >= tc.config.NewWordThreshold
}

// stopWords are filtered out during context comparison so only
// meaningful business terms trigger cache invalidation.
var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "shall": true, "can": true,
	"for": true, "and": true, "but": true, "or": true, "nor": true,
	"not": true, "so": true, "yet": true, "both": true, "either": true,
	"neither": true, "each": true, "every": true,
	"in": true, "on": true, "at": true, "to": true, "of": true,
	"by": true, "with": true, "from": true, "up": true, "about": true,
	"into": true, "over": true, "after": true, "before": true,
	"between": true, "under": true, "above": true, "through": true,
	"during": true, "without": true, "within": true,
	"i": true, "me": true, "my": true, "we": true, "our": true,
	"you": true, "your": true, "he": true, "she": true, "it": true,
	"its": true, "they": true, "them": true, "their": true,
	"this": true, "that": true, "these": true, "those": true,
	"what": true, "which": true, "who": true, "whom": true,
	"how": true, "when": true, "where": true, "why": true,
	"if": true, "then": true, "than": true, "as": true, "also": true,
	"just": true, "very": true, "too": true, "more": true, "most": true,
	"some": true, "any": true, "all": true, "no": true, "only": true,
	"own": true, "same": true, "other": true, "such": true,
}

// meaningfulWords extracts lowercase words from text, filtering stop words.
func meaningfulWords(text string) []string {
	raw := strings.Fields(strings.ToLower(text))
	out := make([]string, 0, len(raw))
	for _, w := range raw {
		w = strings.Trim(w, ".,;:!?\"'()-[]{}/*")
		if len(w) < 2 || stopWords[w] {
			continue
		}
		out = append(out, w)
	}
	return out
}

func (tc *ToolCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time
	for k, v := range tc.entries {
		if oldestKey == "" || v.CreatedAt.Before(oldestTime) {
			oldestKey = k
			oldestTime = v.CreatedAt
		}
	}
	if oldestKey != "" {
		delete(tc.entries, oldestKey)
	}
}

func hashContext(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:8])
}
