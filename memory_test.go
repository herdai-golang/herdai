package herdai

import (
	"context"
	"testing"
	"time"
)

func TestInMemoryStore_StoreAndSearch(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	_ = store.Store(ctx, MemoryEntry{
		ID:      "1",
		AgentID: "agent-a",
		Kind:    MemoryFact,
		Content: "The user prefers Go over Python",
		Tags:    []string{"preference", "language"},
	})
	_ = store.Store(ctx, MemoryEntry{
		ID:      "2",
		AgentID: "agent-a",
		Kind:    MemoryEpisode,
		Content: "Discussed REST API design with the user",
		Tags:    []string{"api", "design"},
	})
	_ = store.Store(ctx, MemoryEntry{
		ID:      "3",
		AgentID: "agent-b",
		Kind:    MemoryInstruction,
		Content: "Always use metric units in responses",
		Tags:    []string{"format"},
	})

	if store.Count() != 3 {
		t.Fatalf("expected 3 entries, got %d", store.Count())
	}

	results, err := store.Search(ctx, "Go Python language", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected search results")
	}
	if results[0].ID != "1" {
		t.Errorf("expected first result to be entry 1, got %s", results[0].ID)
	}

	results, err = store.Search(ctx, "metric units", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected search results for metric units")
	}
	// Instructions get a 1.5x boost
	if results[0].Kind != MemoryInstruction {
		t.Errorf("expected instruction to rank higher, got %s", results[0].Kind)
	}
}

func TestInMemoryStore_GetBySession(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	_ = store.Store(ctx, MemoryEntry{ID: "1", SessionID: "s1", Content: "session 1 fact"})
	_ = store.Store(ctx, MemoryEntry{ID: "2", SessionID: "s2", Content: "session 2 fact"})
	_ = store.Store(ctx, MemoryEntry{ID: "3", SessionID: "s1", Content: "another s1 fact"})

	results, err := store.GetBySession(ctx, "s1")
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 entries for session s1, got %d", len(results))
	}
}

func TestInMemoryStore_GetByAgent(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	_ = store.Store(ctx, MemoryEntry{ID: "1", AgentID: "a1", Content: "fact 1"})
	_ = store.Store(ctx, MemoryEntry{ID: "2", AgentID: "a2", Content: "fact 2"})

	results, err := store.GetByAgent(ctx, "a1")
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1, got %d", len(results))
	}
}

func TestInMemoryStore_Tags(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	_ = store.Store(ctx, MemoryEntry{ID: "1", Content: "tagged item", Tags: []string{"go", "api"}})
	_ = store.Store(ctx, MemoryEntry{ID: "2", Content: "another item", Tags: []string{"python", "api"}})

	results, err := store.GetByTags(ctx, []string{"go", "api"}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != "1" {
		t.Fatalf("expected entry 1 with both tags, got %v", results)
	}

	results, err = store.GetByTags(ctx, []string{"api"}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 entries with 'api' tag, got %d", len(results))
	}
}

func TestInMemoryStore_TTL(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	past := time.Now().Add(-1 * time.Hour)
	_ = store.Store(ctx, MemoryEntry{
		ID:        "expired",
		Content:   "old data",
		ExpiresAt: &past,
	})
	_ = store.Store(ctx, MemoryEntry{ID: "valid", Content: "fresh data"})

	results, err := store.Search(ctx, "data", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].ID != "valid" {
		t.Fatalf("expected only 'valid' entry, got %v", results)
	}
}

func TestInMemoryStore_DeleteAndClear(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	_ = store.Store(ctx, MemoryEntry{ID: "1", SessionID: "s1", Content: "a"})
	_ = store.Store(ctx, MemoryEntry{ID: "2", SessionID: "s1", Content: "b"})
	_ = store.Store(ctx, MemoryEntry{ID: "3", SessionID: "s2", Content: "c"})

	_ = store.Delete(ctx, "1")
	if store.Count() != 2 {
		t.Fatalf("expected 2 after delete, got %d", store.Count())
	}

	_ = store.Clear(ctx, "s1")
	if store.Count() != 1 {
		t.Fatalf("expected 1 after clear s1, got %d", store.Count())
	}

	_ = store.Clear(ctx, "")
	if store.Count() != 0 {
		t.Fatalf("expected 0 after clear all, got %d", store.Count())
	}
}

func TestInMemoryStore_ExportImport(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	_ = store.Store(ctx, MemoryEntry{ID: "1", Content: "first fact", Kind: MemoryFact})
	_ = store.Store(ctx, MemoryEntry{ID: "2", Content: "second fact", Kind: MemoryFact})

	data, err := store.Export()
	if err != nil {
		t.Fatal(err)
	}

	store2 := NewInMemoryStore()
	if err := store2.Import(data); err != nil {
		t.Fatal(err)
	}
	if store2.Count() != 2 {
		t.Fatalf("expected 2 entries after import, got %d", store2.Count())
	}
}

func TestAgentWithMemory(t *testing.T) {
	store := NewInMemoryStore()
	ctx := context.Background()

	_ = store.Store(ctx, MemoryEntry{
		Kind:    MemoryFact,
		Content: "The user's name is Neraj",
		Tags:    []string{"user"},
	})

	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "Hello Neraj, based on your preferences..."})

	agent := NewAgent(AgentConfig{
		ID:     "greeter",
		Role:   "Greeter",
		Goal:   "Greet the user",
		LLM:    mock,
		Memory: store,
	})

	result, err := agent.Run(ctx, "Hi there", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content == "" {
		t.Fatal("expected non-empty response")
	}

	// Verify that a memory episode was stored
	episodes, _ := store.GetByAgent(ctx, "greeter")
	found := false
	for _, e := range episodes {
		if e.Kind == MemoryEpisode {
			found = true
		}
	}
	if !found {
		t.Error("expected agent to store an episode memory after running")
	}
}
