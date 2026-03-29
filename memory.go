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
// Memory — multi-layer memory system with short-term + long-term + search
// ═══════════════════════════════════════════════════════════════════════════════

// MemoryEntry is a single piece of remembered information.
type MemoryEntry struct {
	ID        string         `json:"id"`
	SessionID string         `json:"session_id"`
	AgentID   string         `json:"agent_id"`
	Kind      MemoryKind     `json:"kind"` // fact, episode, instruction, summary
	Content   string         `json:"content"`
	Tags      []string       `json:"tags,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	ExpiresAt *time.Time     `json:"expires_at,omitempty"`
	Score     float64        `json:"score,omitempty"` // relevance score from search
}

// MemoryKind categorizes what kind of memory this is.
type MemoryKind string

const (
	MemoryFact        MemoryKind = "fact"        // a learned fact ("user prefers Go over Python")
	MemoryEpisode     MemoryKind = "episode"     // a conversation summary
	MemoryInstruction MemoryKind = "instruction" // a persistent instruction ("always use metric units")
	MemorySummary     MemoryKind = "summary"     // compressed summary of older conversation
)

// MemoryStore is the interface for pluggable memory backends.
// Implement this for Redis, PostgreSQL, vector DBs, etc.
type MemoryStore interface {
	// Store saves a memory entry.
	Store(ctx context.Context, entry MemoryEntry) error

	// Search finds memories relevant to a query string.
	// Returns entries sorted by relevance, up to limit.
	Search(ctx context.Context, query string, limit int) ([]MemoryEntry, error)

	// GetBySession returns all memories for a session.
	GetBySession(ctx context.Context, sessionID string) ([]MemoryEntry, error)

	// GetByAgent returns all memories for an agent.
	GetByAgent(ctx context.Context, agentID string) ([]MemoryEntry, error)

	// GetByTags returns memories matching ALL given tags.
	GetByTags(ctx context.Context, tags []string, limit int) ([]MemoryEntry, error)

	// Delete removes a memory entry by ID.
	Delete(ctx context.Context, id string) error

	// Clear removes all memories (optionally scoped to a session).
	Clear(ctx context.Context, sessionID string) error
}

// ── InMemoryStore — full-featured in-process memory store ──────────────────

// InMemoryStore is a thread-safe in-memory implementation of MemoryStore.
// Supports keyword search, tag filtering, and TTL expiration.
// Suitable for development, testing, and single-process deployments.
type InMemoryStore struct {
	mu      sync.RWMutex
	entries map[string]MemoryEntry // id → entry
	index   []string              // ordered by creation time
}

// NewInMemoryStore creates an empty in-memory store.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		entries: make(map[string]MemoryEntry),
	}
}

func (s *InMemoryStore) Store(_ context.Context, entry MemoryEntry) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry.ID == "" {
		entry.ID = generateID()
	}
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = time.Now()
	}

	if _, exists := s.entries[entry.ID]; !exists {
		s.index = append(s.index, entry.ID)
	}
	s.entries[entry.ID] = entry
	return nil
}

func (s *InMemoryStore) Search(_ context.Context, query string, limit int) ([]MemoryEntry, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	query = strings.ToLower(query)
	words := strings.Fields(query)
	now := time.Now()

	type scored struct {
		entry MemoryEntry
		score float64
	}

	var results []scored
	for _, id := range s.index {
		entry := s.entries[id]
		if entry.ExpiresAt != nil && now.After(*entry.ExpiresAt) {
			continue
		}
		score := s.computeRelevance(entry, words)
		if score > 0 {
			e := entry
			e.Score = score
			results = append(results, scored{entry: e, score: score})
		}
	}

	// Sort by score descending (simple insertion sort for small N)
	for i := 1; i < len(results); i++ {
		for j := i; j > 0 && results[j].score > results[j-1].score; j-- {
			results[j], results[j-1] = results[j-1], results[j]
		}
	}

	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}

	out := make([]MemoryEntry, len(results))
	for i, r := range results {
		out[i] = r.entry
	}
	return out, nil
}

func (s *InMemoryStore) computeRelevance(entry MemoryEntry, queryWords []string) float64 {
	content := strings.ToLower(entry.Content)
	tags := strings.ToLower(strings.Join(entry.Tags, " "))
	combined := content + " " + tags

	var score float64
	for _, w := range queryWords {
		if strings.Contains(combined, w) {
			score += 1.0
		}
	}
	// Boost instructions and facts
	if entry.Kind == MemoryInstruction {
		score *= 1.5
	} else if entry.Kind == MemoryFact {
		score *= 1.2
	}
	return score
}

func (s *InMemoryStore) GetBySession(_ context.Context, sessionID string) ([]MemoryEntry, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	now := time.Now()
	var out []MemoryEntry
	for _, id := range s.index {
		e := s.entries[id]
		if e.SessionID == sessionID {
			if e.ExpiresAt != nil && now.After(*e.ExpiresAt) {
				continue
			}
			out = append(out, e)
		}
	}
	return out, nil
}

func (s *InMemoryStore) GetByAgent(_ context.Context, agentID string) ([]MemoryEntry, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	now := time.Now()
	var out []MemoryEntry
	for _, id := range s.index {
		e := s.entries[id]
		if e.AgentID == agentID {
			if e.ExpiresAt != nil && now.After(*e.ExpiresAt) {
				continue
			}
			out = append(out, e)
		}
	}
	return out, nil
}

func (s *InMemoryStore) GetByTags(_ context.Context, tags []string, limit int) ([]MemoryEntry, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	now := time.Now()
	var out []MemoryEntry
	for _, id := range s.index {
		e := s.entries[id]
		if e.ExpiresAt != nil && now.After(*e.ExpiresAt) {
			continue
		}
		if hasAllTags(e.Tags, tags) {
			out = append(out, e)
			if limit > 0 && len(out) >= limit {
				break
			}
		}
	}
	return out, nil
}

func (s *InMemoryStore) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.entries, id)
	for i, idx := range s.index {
		if idx == id {
			s.index = append(s.index[:i], s.index[i+1:]...)
			break
		}
	}
	return nil
}

func (s *InMemoryStore) Clear(_ context.Context, sessionID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if sessionID == "" {
		s.entries = make(map[string]MemoryEntry)
		s.index = nil
		return nil
	}

	var kept []string
	for _, id := range s.index {
		if s.entries[id].SessionID == sessionID {
			delete(s.entries, id)
		} else {
			kept = append(kept, id)
		}
	}
	s.index = kept
	return nil
}

// Count returns the total number of entries (thread-safe).
func (s *InMemoryStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

// All returns all non-expired entries in creation order.
func (s *InMemoryStore) All() []MemoryEntry {
	s.mu.RLock()
	defer s.mu.RUnlock()

	now := time.Now()
	var out []MemoryEntry
	for _, id := range s.index {
		e := s.entries[id]
		if e.ExpiresAt != nil && now.After(*e.ExpiresAt) {
			continue
		}
		out = append(out, e)
	}
	return out
}

// Export serializes all entries to JSON.
func (s *InMemoryStore) Export() ([]byte, error) {
	return json.Marshal(s.All())
}

// Import loads entries from JSON, merging with existing.
func (s *InMemoryStore) Import(data []byte) error {
	var entries []MemoryEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return fmt.Errorf("import memory: %w", err)
	}
	ctx := context.Background()
	for _, e := range entries {
		if err := s.Store(ctx, e); err != nil {
			return err
		}
	}
	return nil
}

func hasAllTags(entryTags, requiredTags []string) bool {
	set := make(map[string]bool, len(entryTags))
	for _, t := range entryTags {
		set[strings.ToLower(t)] = true
	}
	for _, t := range requiredTags {
		if !set[strings.ToLower(t)] {
			return false
		}
	}
	return true
}
