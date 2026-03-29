package herdai

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// Session — save/resume conversations and agent state to disk
//
// Sessions persist:
//   - Conversation transcript (all turns)
//   - Memory entries (short-term + long-term)
//   - Agent results and metadata
//   - Session-level metadata (timestamps, status, custom data)
//
// Sessions can be saved, loaded, listed, and deleted.
// ═══════════════════════════════════════════════════════════════════════════════

// SessionStatus is the lifecycle state of a session.
type SessionStatus string

const (
	SessionActive    SessionStatus = "active"
	SessionPaused    SessionStatus = "paused"
	SessionCompleted SessionStatus = "completed"
	SessionFailed    SessionStatus = "failed"
)

// Session captures the full state of an agent pipeline run.
type Session struct {
	mu sync.RWMutex

	ID           string            `json:"id"`
	Name         string            `json:"name,omitempty"`
	Status       SessionStatus     `json:"status"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
	ResumedAt    *time.Time        `json:"resumed_at,omitempty"`
	Metadata     map[string]any    `json:"metadata,omitempty"`
	Conversation *Conversation     `json:"-"` // serialized separately

	// Serializable form of conversation
	Turns   []Turn        `json:"turns"`
	Results []*Result     `json:"results,omitempty"`
	Memory  []MemoryEntry `json:"memory,omitempty"`

	// Checkpoint support — agents can save arbitrary state
	Checkpoints map[string]json.RawMessage `json:"checkpoints,omitempty"`
}

// NewSession creates a fresh session.
func NewSession(name string) *Session {
	now := time.Now()
	return &Session{
		ID:           generateID(),
		Name:         name,
		Status:       SessionActive,
		CreatedAt:    now,
		UpdatedAt:    now,
		Metadata:     make(map[string]any),
		Conversation: NewConversation(),
		Checkpoints:  make(map[string]json.RawMessage),
	}
}

// GetConversation returns the session's conversation.
func (s *Session) GetConversation() *Conversation {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.Conversation == nil {
		s.mu.RUnlock()
		s.mu.Lock()
		if s.Conversation == nil {
			s.Conversation = NewConversation()
			for _, t := range s.Turns {
				s.Conversation.AddTurn(t)
			}
		}
		s.mu.Unlock()
		s.mu.RLock()
	}
	return s.Conversation
}

// AddResult appends an agent result to the session.
func (s *Session) AddResult(r *Result) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Results = append(s.Results, r)
	s.UpdatedAt = time.Now()
}

// SetCheckpoint saves arbitrary agent state for later resumption.
func (s *Session) SetCheckpoint(agentID string, state any) error {
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal checkpoint for %s: %w", agentID, err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Checkpoints[agentID] = data
	s.UpdatedAt = time.Now()
	return nil
}

// GetCheckpoint loads a previously saved agent checkpoint.
func (s *Session) GetCheckpoint(agentID string, dest any) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	data, ok := s.Checkpoints[agentID]
	if !ok {
		return fmt.Errorf("no checkpoint for agent %s", agentID)
	}
	return json.Unmarshal(data, dest)
}

// Pause marks the session as paused (can be resumed later).
func (s *Session) Pause() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Status = SessionPaused
	s.UpdatedAt = time.Now()
}

// Resume marks the session as active.
func (s *Session) Resume() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Status = SessionActive
	now := time.Now()
	s.ResumedAt = &now
	s.UpdatedAt = now
}

// Complete marks the session as completed.
func (s *Session) Complete() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Status = SessionCompleted
	s.UpdatedAt = time.Now()
}

// Fail marks the session as failed.
func (s *Session) Fail() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Status = SessionFailed
	s.UpdatedAt = time.Now()
}

// SetMeta sets a metadata key-value pair.
func (s *Session) SetMeta(key string, value any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Metadata == nil {
		s.Metadata = make(map[string]any)
	}
	s.Metadata[key] = value
	s.UpdatedAt = time.Now()
}

// prepareForSave syncs the conversation turns into the serializable field.
func (s *Session) prepareForSave() {
	if s.Conversation != nil {
		s.Turns = s.Conversation.GetTurns()
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// SessionStore — interface for pluggable session backends
// ═══════════════════════════════════════════════════════════════════════════════

// SessionStore is the interface for saving/loading sessions.
// Implement for Redis, S3, database, etc.
type SessionStore interface {
	Save(session *Session) error
	Load(id string) (*Session, error)
	List() ([]*SessionSummary, error)
	Delete(id string) error
}

// SessionSummary is a lightweight listing entry (no full conversation).
type SessionSummary struct {
	ID        string        `json:"id"`
	Name      string        `json:"name"`
	Status    SessionStatus `json:"status"`
	TurnCount int           `json:"turn_count"`
	CreatedAt time.Time     `json:"created_at"`
	UpdatedAt time.Time     `json:"updated_at"`
}

// ── FileSessionStore — saves sessions as JSON files on disk ────────────────

// FileSessionStore saves each session as a JSON file in a directory.
type FileSessionStore struct {
	dir string
}

// NewFileSessionStore creates a file-based session store.
// The directory is created if it doesn't exist.
func NewFileSessionStore(dir string) (*FileSessionStore, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create session dir: %w", err)
	}
	return &FileSessionStore{dir: dir}, nil
}

func (fs *FileSessionStore) sessionPath(id string) string {
	return filepath.Join(fs.dir, id+".json")
}

func (fs *FileSessionStore) Save(session *Session) error {
	session.mu.Lock()
	session.prepareForSave()
	session.mu.Unlock()

	session.mu.RLock()
	data, err := json.MarshalIndent(session, "", "  ")
	session.mu.RUnlock()
	if err != nil {
		return fmt.Errorf("marshal session: %w", err)
	}

	path := fs.sessionPath(session.ID)
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write session %s: %w", session.ID, err)
	}
	return nil
}

func (fs *FileSessionStore) Load(id string) (*Session, error) {
	path := fs.sessionPath(id)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("session %s not found", id)
		}
		return nil, fmt.Errorf("read session %s: %w", id, err)
	}

	var session Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("unmarshal session %s: %w", id, err)
	}

	// Rebuild conversation from turns
	session.Conversation = NewConversation()
	for _, t := range session.Turns {
		session.Conversation.AddTurn(t)
	}

	if session.Checkpoints == nil {
		session.Checkpoints = make(map[string]json.RawMessage)
	}
	if session.Metadata == nil {
		session.Metadata = make(map[string]any)
	}

	return &session, nil
}

func (fs *FileSessionStore) List() ([]*SessionSummary, error) {
	entries, err := os.ReadDir(fs.dir)
	if err != nil {
		return nil, fmt.Errorf("read session dir: %w", err)
	}

	var summaries []*SessionSummary
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		id := entry.Name()[:len(entry.Name())-5] // strip .json
		session, err := fs.Load(id)
		if err != nil {
			continue
		}

		summaries = append(summaries, &SessionSummary{
			ID:        session.ID,
			Name:      session.Name,
			Status:    session.Status,
			TurnCount: len(session.Turns),
			CreatedAt: session.CreatedAt,
			UpdatedAt: session.UpdatedAt,
		})
	}
	return summaries, nil
}

func (fs *FileSessionStore) Delete(id string) error {
	path := fs.sessionPath(id)
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("delete session %s: %w", id, err)
	}
	return nil
}

// ── InMemorySessionStore — for testing ─────────────────────────────────────

// InMemorySessionStore keeps sessions in memory (no persistence across restarts).
type InMemorySessionStore struct {
	mu       sync.RWMutex
	sessions map[string]*Session
}

// NewInMemorySessionStore creates an in-memory session store.
func NewInMemorySessionStore() *InMemorySessionStore {
	return &InMemorySessionStore{
		sessions: make(map[string]*Session),
	}
}

func (ms *InMemorySessionStore) Save(session *Session) error {
	session.mu.Lock()
	session.prepareForSave()
	session.mu.Unlock()

	// Deep copy via JSON round-trip
	session.mu.RLock()
	data, err := json.Marshal(session)
	session.mu.RUnlock()
	if err != nil {
		return err
	}

	var copy Session
	if err := json.Unmarshal(data, &copy); err != nil {
		return err
	}

	ms.mu.Lock()
	defer ms.mu.Unlock()
	ms.sessions[session.ID] = &copy
	return nil
}

func (ms *InMemorySessionStore) Load(id string) (*Session, error) {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	s, ok := ms.sessions[id]
	if !ok {
		return nil, fmt.Errorf("session %s not found", id)
	}

	// Deep copy via JSON round-trip
	data, _ := json.Marshal(s)
	var copy Session
	json.Unmarshal(data, &copy)

	copy.Conversation = NewConversation()
	for _, t := range copy.Turns {
		copy.Conversation.AddTurn(t)
	}

	if copy.Checkpoints == nil {
		copy.Checkpoints = make(map[string]json.RawMessage)
	}
	if copy.Metadata == nil {
		copy.Metadata = make(map[string]any)
	}

	return &copy, nil
}

func (ms *InMemorySessionStore) List() ([]*SessionSummary, error) {
	ms.mu.RLock()
	defer ms.mu.RUnlock()

	var out []*SessionSummary
	for _, s := range ms.sessions {
		out = append(out, &SessionSummary{
			ID:        s.ID,
			Name:      s.Name,
			Status:    s.Status,
			TurnCount: len(s.Turns),
			CreatedAt: s.CreatedAt,
			UpdatedAt: s.UpdatedAt,
		})
	}
	return out, nil
}

func (ms *InMemorySessionStore) Delete(id string) error {
	ms.mu.Lock()
	defer ms.mu.Unlock()
	delete(ms.sessions, id)
	return nil
}
