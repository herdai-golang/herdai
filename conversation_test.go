package herdai

import (
	"sync"
	"testing"
)

func TestNewConversation(t *testing.T) {
	conv := NewConversation()
	if conv.ID() == "" {
		t.Fatal("expected non-empty conversation ID")
	}
	if conv.Len() != 0 {
		t.Fatalf("expected 0 turns, got %d", conv.Len())
	}
}

func TestConversationAddAndGet(t *testing.T) {
	conv := NewConversation()

	conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "hello"})
	conv.AddTurn(Turn{AgentID: "a2", Role: "assistant", Content: "hi there"})
	conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "how are you"})

	if conv.Len() != 3 {
		t.Fatalf("expected 3 turns, got %d", conv.Len())
	}

	turns := conv.GetTurns()
	if len(turns) != 3 {
		t.Fatalf("expected 3 turns from GetTurns, got %d", len(turns))
	}
	if turns[0].Content != "hello" {
		t.Fatalf("expected 'hello', got '%s'", turns[0].Content)
	}
	if turns[1].Content != "hi there" {
		t.Fatalf("expected 'hi there', got '%s'", turns[1].Content)
	}
}

func TestConversationAutoFields(t *testing.T) {
	conv := NewConversation()
	conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "test"})

	turns := conv.GetTurns()
	if turns[0].ID == "" {
		t.Fatal("expected auto-generated turn ID")
	}
	if turns[0].Timestamp.IsZero() {
		t.Fatal("expected auto-generated timestamp")
	}
}

func TestConversationLastN(t *testing.T) {
	conv := NewConversation()
	for i := 0; i < 10; i++ {
		conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "msg"})
	}

	last3 := conv.LastN(3)
	if len(last3) != 3 {
		t.Fatalf("expected 3, got %d", len(last3))
	}

	last20 := conv.LastN(20)
	if len(last20) != 10 {
		t.Fatalf("expected 10 (all), got %d", len(last20))
	}

	last0 := conv.LastN(0)
	if len(last0) != 0 {
		t.Fatalf("expected 0, got %d", len(last0))
	}
}

func TestConversationGetTurnsReturnsCopy(t *testing.T) {
	conv := NewConversation()
	conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "original"})

	turns := conv.GetTurns()
	turns[0].Content = "modified"

	fresh := conv.GetTurns()
	if fresh[0].Content != "original" {
		t.Fatal("GetTurns should return a copy, not a reference")
	}
}

func TestConversationConcurrentAccess(t *testing.T) {
	conv := NewConversation()
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			conv.AddTurn(Turn{AgentID: "a1", Role: "user", Content: "concurrent"})
		}(i)
	}

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = conv.GetTurns()
			_ = conv.LastN(5)
			_ = conv.Len()
		}()
	}

	wg.Wait()

	if conv.Len() != 100 {
		t.Fatalf("expected 100 turns after concurrent writes, got %d", conv.Len())
	}
}
