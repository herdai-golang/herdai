package herdai

import (
	"os"
	"testing"
)

func TestSession_Lifecycle(t *testing.T) {
	s := NewSession("test session")

	if s.Status != SessionActive {
		t.Errorf("expected active, got %s", s.Status)
	}

	conv := s.GetConversation()
	conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "hello"})
	conv.AddTurn(Turn{AgentID: "a1", Role: "assistant", Content: "hi there"})

	if conv.Len() != 2 {
		t.Fatalf("expected 2 turns, got %d", conv.Len())
	}

	s.AddResult(&Result{AgentID: "a1", Content: "result 1"})
	if len(s.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(s.Results))
	}

	s.Pause()
	if s.Status != SessionPaused {
		t.Errorf("expected paused, got %s", s.Status)
	}

	s.Resume()
	if s.Status != SessionActive {
		t.Errorf("expected active after resume, got %s", s.Status)
	}
	if s.ResumedAt == nil {
		t.Error("expected ResumedAt to be set")
	}

	s.Complete()
	if s.Status != SessionCompleted {
		t.Errorf("expected completed, got %s", s.Status)
	}
}

func TestSession_Checkpoints(t *testing.T) {
	s := NewSession("checkpoint test")

	type AgentState struct {
		Step     int    `json:"step"`
		LastTool string `json:"last_tool"`
	}

	state := AgentState{Step: 3, LastTool: "search"}
	if err := s.SetCheckpoint("analyst", state); err != nil {
		t.Fatal(err)
	}

	var loaded AgentState
	if err := s.GetCheckpoint("analyst", &loaded); err != nil {
		t.Fatal(err)
	}
	if loaded.Step != 3 || loaded.LastTool != "search" {
		t.Errorf("expected {3, search}, got %+v", loaded)
	}

	if err := s.GetCheckpoint("nonexistent", &loaded); err == nil {
		t.Error("expected error for nonexistent checkpoint")
	}
}

func TestSession_Metadata(t *testing.T) {
	s := NewSession("meta test")
	s.SetMeta("model", "mistral-small-latest")
	s.SetMeta("temperature", 0.7)

	if s.Metadata["model"] != "mistral-small-latest" {
		t.Errorf("expected mistral-small-latest, got %v", s.Metadata["model"])
	}
}

func TestFileSessionStore_SaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	store, err := NewFileSessionStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	s := NewSession("file test")
	conv := s.GetConversation()
	conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "hello"})
	conv.AddTurn(Turn{AgentID: "a1", Role: "assistant", Content: "hi"})
	s.AddResult(&Result{AgentID: "a1", Content: "test result"})
	s.SetMeta("key", "value")
	_ = s.SetCheckpoint("a1", map[string]int{"step": 1})

	if err := store.Save(s); err != nil {
		t.Fatal(err)
	}

	loaded, err := store.Load(s.ID)
	if err != nil {
		t.Fatal(err)
	}

	if loaded.Name != "file test" {
		t.Errorf("expected 'file test', got %s", loaded.Name)
	}
	if len(loaded.Turns) != 2 {
		t.Errorf("expected 2 turns, got %d", len(loaded.Turns))
	}
	if len(loaded.Results) != 1 {
		t.Errorf("expected 1 result, got %d", len(loaded.Results))
	}
	if loaded.Metadata["key"] != "value" {
		t.Errorf("expected metadata key=value")
	}

	// Verify conversation was reconstructed
	loadedConv := loaded.GetConversation()
	if loadedConv.Len() != 2 {
		t.Errorf("expected reconstructed conversation with 2 turns, got %d", loadedConv.Len())
	}
}

func TestFileSessionStore_List(t *testing.T) {
	dir := t.TempDir()
	store, _ := NewFileSessionStore(dir)

	s1 := NewSession("first")
	s2 := NewSession("second")
	_ = store.Save(s1)
	_ = store.Save(s2)

	summaries, err := store.List()
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 2 {
		t.Fatalf("expected 2 summaries, got %d", len(summaries))
	}
}

func TestFileSessionStore_Delete(t *testing.T) {
	dir := t.TempDir()
	store, _ := NewFileSessionStore(dir)

	s := NewSession("to delete")
	_ = store.Save(s)

	if err := store.Delete(s.ID); err != nil {
		t.Fatal(err)
	}

	_, err := store.Load(s.ID)
	if err == nil {
		t.Error("expected error loading deleted session")
	}
}

func TestFileSessionStore_LoadNotFound(t *testing.T) {
	dir := t.TempDir()
	store, _ := NewFileSessionStore(dir)

	_, err := store.Load("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent session")
	}
}

func TestFileSessionStore_DirCreation(t *testing.T) {
	dir := t.TempDir() + "/nested/sessions"
	store, err := NewFileSessionStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	s := NewSession("nested test")
	if err := store.Save(s); err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(dir); os.IsNotExist(err) {
		t.Error("directory should have been created")
	}
}

func TestInMemorySessionStore_SaveLoadDelete(t *testing.T) {
	store := NewInMemorySessionStore()

	s := NewSession("in-memory test")
	conv := s.GetConversation()
	conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "hello"})

	if err := store.Save(s); err != nil {
		t.Fatal(err)
	}

	loaded, err := store.Load(s.ID)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Name != "in-memory test" {
		t.Errorf("expected 'in-memory test', got %s", loaded.Name)
	}

	// Verify deep copy — modifying loaded should not affect stored
	loaded.Name = "modified"
	original, _ := store.Load(s.ID)
	if original.Name == "modified" {
		t.Error("store should deep copy — modifying loaded should not affect stored")
	}

	summaries, _ := store.List()
	if len(summaries) != 1 {
		t.Fatalf("expected 1 summary, got %d", len(summaries))
	}

	_ = store.Delete(s.ID)
	_, err = store.Load(s.ID)
	if err == nil {
		t.Error("expected error after delete")
	}
}

func TestSession_FailStatus(t *testing.T) {
	s := NewSession("fail test")
	s.Fail()
	if s.Status != SessionFailed {
		t.Errorf("expected failed, got %s", s.Status)
	}
}
