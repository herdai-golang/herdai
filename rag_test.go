package herdai

import (
	"context"
	"strings"
	"testing"
	"time"
)

// ── Chunker Tests ──────────────────────────────────────────────────────────

func TestFixedSizeChunker(t *testing.T) {
	doc := Document{ID: "d1", Content: "ABCDEFGHIJKLMNOPQRSTUVWXYZ", Source: "test"}
	chunker := NewFixedSizeChunker(10, 2)
	chunks := chunker.Chunk(doc)

	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 chunks from 26 chars with size=10 overlap=2, got %d", len(chunks))
	}
	if chunks[0].Content != "ABCDEFGHIJ" {
		t.Errorf("first chunk: got %q", chunks[0].Content)
	}
	if chunks[0].DocumentID != "d1" {
		t.Error("chunks should carry document ID")
	}
}

func TestFixedSizeChunker_EmptyDoc(t *testing.T) {
	doc := Document{ID: "d1", Content: "", Source: "test"}
	chunks := NewFixedSizeChunker(10, 2).Chunk(doc)
	if len(chunks) != 0 {
		t.Errorf("expected 0 chunks for empty doc, got %d", len(chunks))
	}
}

func TestParagraphChunker(t *testing.T) {
	doc := Document{
		ID:     "d1",
		Source: "test",
		Content: "First paragraph about Go.\n\n" +
			"Second paragraph about agents.\n\n" +
			"Third paragraph about tools.",
	}
	chunker := NewParagraphChunker(1000, 0)
	chunks := chunker.Chunk(doc)

	// All paragraphs fit in one chunk since target is 1000
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk (all fit), got %d", len(chunks))
	}

	// With small target, should split
	chunker2 := NewParagraphChunker(40, 0)
	chunks2 := chunker2.Chunk(doc)
	if len(chunks2) < 2 {
		t.Fatalf("expected multiple chunks with small target, got %d", len(chunks2))
	}
}

func TestMarkdownChunker(t *testing.T) {
	doc := Document{
		ID:     "d1",
		Source: "readme.md",
		Content: "# Title\n\nIntro text.\n\n" +
			"## Section 1\n\nSection 1 content.\n\n" +
			"## Section 2\n\nSection 2 content.",
	}
	chunker := NewMarkdownChunker(1000)
	chunks := chunker.Chunk(doc)

	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks (by headers), got %d", len(chunks))
	}

	// Check metadata has section info
	foundSection := false
	for _, ch := range chunks {
		if sec, ok := ch.Metadata["section"]; ok {
			if sec == "Section 1" || sec == "Section 2" {
				foundSection = true
			}
		}
	}
	if !foundSection {
		t.Error("expected section metadata on markdown chunks")
	}
}

func TestRecursiveChunker(t *testing.T) {
	doc := Document{
		ID:     "d1",
		Source: "test",
		Content: "Short paragraph one.\n\n" +
			"Short paragraph two.\n\n" +
			strings.Repeat("Very long paragraph that should be split. ", 50),
	}
	chunker := NewRecursiveChunker(200, 20)
	chunks := chunker.Chunk(doc)

	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	for _, ch := range chunks {
		if len(ch.Content) > 600 { // some tolerance for boundaries
			t.Errorf("chunk too large: %d chars", len(ch.Content))
		}
	}
}

// ── Loader Tests ───────────────────────────────────────────────────────────

func TestStringsLoader(t *testing.T) {
	loader := NewStringsLoader(map[string]string{
		"doc1.txt": "Hello world",
		"doc2.txt": "Goodbye world",
	})

	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 docs, got %d", len(docs))
	}
}

func TestReaderLoader(t *testing.T) {
	reader := strings.NewReader("Content from a reader")
	loader := NewReaderLoader(reader, "uploaded.pdf")

	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 {
		t.Fatal("expected 1 doc")
	}
	if docs[0].Content != "Content from a reader" {
		t.Errorf("unexpected content: %s", docs[0].Content)
	}
	if docs[0].Source != "uploaded.pdf" {
		t.Errorf("expected source 'uploaded.pdf', got %s", docs[0].Source)
	}
}

func TestTextLoader_NotFound(t *testing.T) {
	loader := NewTextLoader("/nonexistent/file.txt")
	_, err := loader.Load(context.Background())
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

// ── Vector Store Tests ─────────────────────────────────────────────────────

func TestInMemoryVectorStore_KeywordSearch(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	_ = store.Add(ctx, []Chunk{
		{ID: "c1", Content: "Go is a compiled programming language designed at Google"},
		{ID: "c2", Content: "Python is an interpreted language popular for data science"},
		{ID: "c3", Content: "Rust is a systems programming language focused on safety"},
	})

	if store.Count() != 3 {
		t.Fatalf("expected 3 chunks, got %d", store.Count())
	}

	results, err := store.Search(ctx, "Go programming Google", nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected results for 'Go programming Google'")
	}
	if results[0].ID != "c1" {
		t.Errorf("expected c1 as top result, got %s", results[0].ID)
	}
}

func TestInMemoryVectorStore_CosineSimilarity(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	_ = store.Add(ctx, []Chunk{
		{ID: "c1", Content: "about cats", Embedding: []float64{1, 0, 0}},
		{ID: "c2", Content: "about dogs", Embedding: []float64{0, 1, 0}},
		{ID: "c3", Content: "about cats and dogs", Embedding: []float64{0.7, 0.7, 0}},
	})

	// Query embedding close to "cats" (c1)
	results, err := store.Search(ctx, "", []float64{0.9, 0.1, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	if results[0].ID != "c1" {
		t.Errorf("expected c1 as closest to cat-like embedding, got %s", results[0].ID)
	}
}

func TestInMemoryVectorStore_Clear(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	_ = store.Add(ctx, []Chunk{{ID: "c1", Content: "test"}})
	_ = store.Clear(ctx)
	if store.Count() != 0 {
		t.Error("expected 0 after clear")
	}
}

// ── Retriever Tests ────────────────────────────────────────────────────────

func TestKeywordRetriever(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	_ = store.Add(ctx, []Chunk{
		{ID: "c1", Content: "HerdAI is a Go framework for AI agents"},
		{ID: "c2", Content: "Python has many machine learning libraries"},
	})

	retriever := NewKeywordRetriever(store)
	results, err := retriever.Retrieve(ctx, "Go framework agents", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	if results[0].ID != "c1" {
		t.Errorf("expected c1, got %s", results[0].ID)
	}
}

// ── Ingestion Pipeline Tests ───────────────────────────────────────────────

func TestIngestionPipeline(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	loader := NewStringsLoader(map[string]string{
		"doc1.md": "# Architecture\n\nHerdAI uses agents and managers.\n\nAgents have tools and LLMs.",
		"doc2.md": "# Testing\n\nUse MockLLM for deterministic tests.\n\nThe eval harness runs assertions.",
	})

	pipeline := NewIngestionPipeline(IngestionConfig{
		Loader:   loader,
		Chunker:  NewParagraphChunker(200, 0),
		Embedder: NewNoOpEmbedder(),
		Store:    store,
	})

	stats, err := pipeline.Ingest(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Documents != 2 {
		t.Errorf("expected 2 documents, got %d", stats.Documents)
	}
	if stats.Chunks < 2 {
		t.Errorf("expected at least 2 chunks, got %d", stats.Chunks)
	}
	if store.Count() != stats.Chunks {
		t.Errorf("store count (%d) != stats chunks (%d)", store.Count(), stats.Chunks)
	}
}

func TestIngestDocuments_Dynamic(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	// Initial ingestion
	_, _ = IngestDocuments(ctx, store, DefaultChunker(), NewNoOpEmbedder(),
		Document{Content: "Initial knowledge about Go", Source: "initial.txt"},
	)

	if store.Count() == 0 {
		t.Fatal("expected chunks after initial ingestion")
	}
	initialCount := store.Count()

	// Dynamic addition — like user uploading a PDF mid-conversation
	_, _ = IngestDocuments(ctx, store, DefaultChunker(), NewNoOpEmbedder(),
		Document{Content: "New knowledge about Rust added later", Source: "upload.pdf"},
	)

	if store.Count() <= initialCount {
		t.Error("expected more chunks after dynamic addition")
	}

	// Search should find content from both
	results, _ := store.Search(ctx, "Go", nil, 5)
	if len(results) == 0 {
		t.Error("expected Go-related results")
	}

	results, _ = store.Search(ctx, "Rust", nil, 5)
	if len(results) == 0 {
		t.Error("expected Rust-related results from dynamically added doc")
	}
}

func TestSimpleRAG(t *testing.T) {
	store := NewInMemoryVectorStore()
	cfg := SimpleRAG(store, 3)

	if cfg.TopK != 3 {
		t.Errorf("expected TopK=3, got %d", cfg.TopK)
	}
	if !cfg.CiteSources {
		t.Error("expected CiteSources=true")
	}
	if cfg.Retriever == nil {
		t.Error("expected non-nil retriever")
	}
}

// ── Agent + RAG Integration Tests ──────────────────────────────────────────

func TestAgentWithRAG(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	_ = store.Add(ctx, []Chunk{
		{ID: "c1", Content: "HerdAI supports 4 manager strategies: Sequential, Parallel, RoundRobin, and LLMRouter.", Source: "docs/architecture.md"},
		{ID: "c2", Content: "The eval harness supports 12 built-in assertions for testing agent behavior.", Source: "docs/testing.md"},
		{ID: "c3", Content: "Guardrails run at two points: input validation and output validation.", Source: "docs/guardrails.md"},
	})

	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "HerdAI has 4 strategies: Sequential, Parallel, RoundRobin, and LLMRouter."})

	agent := NewAgent(AgentConfig{
		ID:   "rag-agent",
		Role: "Documentation Assistant",
		Goal: "Answer questions about HerdAI using the knowledge base",
		LLM:  mock,
		RAG:  SimpleRAG(store, 3),
	})

	result, err := agent.Run(ctx, "What manager strategies does HerdAI support?", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "Sequential") {
		t.Error("expected grounded answer about strategies")
	}
}

func TestAgentWithRAG_NoResults(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	// Empty store — no documents loaded
	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "I don't have specific information about that."})

	agent := NewAgent(AgentConfig{
		ID:   "rag-empty",
		Role: "Assistant",
		Goal: "Answer questions",
		LLM:  mock,
		RAG:  SimpleRAG(store, 3),
	})

	result, err := agent.Run(ctx, "What is the meaning of life?", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content == "" {
		t.Error("expected some response even without RAG context")
	}
}

func TestAgentWithRAG_MinScore(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	_ = store.Add(ctx, []Chunk{
		{ID: "c1", Content: "Completely unrelated content about cooking recipes"},
	})

	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "No relevant context found."})

	agent := NewAgent(AgentConfig{
		ID:   "rag-minsc",
		Role: "Assistant",
		Goal: "Answer questions",
		LLM:  mock,
		RAG: &RAGConfig{
			Retriever: NewKeywordRetriever(store),
			TopK:      3,
			MinScore:  10.0, // very high threshold — nothing will pass
		},
	})

	result, err := agent.Run(ctx, "quantum physics", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content == "" {
		t.Error("expected response even when no chunks pass MinScore")
	}
}

func TestAgentWithRAG_WithTracing(t *testing.T) {
	store := NewInMemoryVectorStore()
	ctx := context.Background()

	_ = store.Add(ctx, []Chunk{
		{ID: "c1", Content: "Go is a compiled language", Source: "go.md"},
	})

	tracer := NewTracer()
	ctx = ContextWithTracer(ctx, tracer)

	mock := &MockLLM{}
	mock.PushResponse(LLMResponse{Content: "Go is compiled"})

	agent := NewAgent(AgentConfig{
		ID:   "rag-traced",
		Role: "Test",
		Goal: "Test RAG tracing",
		LLM:  mock,
		RAG:  SimpleRAG(store, 3),
	})

	_, err := agent.Run(ctx, "What is Go?", nil)
	if err != nil {
		t.Fatal(err)
	}

	// Check that a RAG span was created
	spans := tracer.Spans()
	foundRAG := false
	for _, s := range spans {
		if strings.Contains(s.Name, "rag") {
			foundRAG = true
		}
	}
	if !foundRAG {
		t.Error("expected a RAG span in the trace")
	}
}

// ── Helper Tests ───────────────────────────────────────────────────────────

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors
	if s := cosineSimilarity([]float64{1, 0, 0}, []float64{1, 0, 0}); s < 0.999 {
		t.Errorf("identical vectors should have similarity ~1.0, got %f", s)
	}

	// Orthogonal vectors
	if s := cosineSimilarity([]float64{1, 0, 0}, []float64{0, 1, 0}); s > 0.001 {
		t.Errorf("orthogonal vectors should have similarity ~0.0, got %f", s)
	}

	// Empty/mismatched
	if s := cosineSimilarity(nil, nil); s != 0 {
		t.Errorf("nil vectors should return 0, got %f", s)
	}
}

func TestKeywordScore(t *testing.T) {
	score := keywordScore("Go programming language", "Go is a compiled programming language")
	if score <= 0 {
		t.Error("expected positive score for matching content")
	}

	noMatch := keywordScore("quantum physics", "Go is a compiled programming language")
	if noMatch >= score {
		t.Error("non-matching should score lower than matching")
	}
}

func TestURLLoader(t *testing.T) {
	loader := NewURLLoader("https://httpbin.org/robots.txt")
	loader.Timeout = 10 * time.Second
	docs, err := loader.Load(context.Background())
	if err != nil {
		t.Skipf("network unavailable: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("expected 1 doc, got %d", len(docs))
	}
	if docs[0].Source != "https://httpbin.org/robots.txt" {
		t.Errorf("wrong source: %s", docs[0].Source)
	}
	if docs[0].Metadata["loader"] != "url" {
		t.Error("expected loader=url in metadata")
	}
}

func TestURLLoaderBadURL(t *testing.T) {
	loader := NewURLLoader("http://localhost:1/nonexistent")
	loader.Timeout = 1 * time.Second
	_, err := loader.Load(context.Background())
	if err == nil {
		t.Error("expected error for bad URL")
	}
}

func TestMultiURLLoader(t *testing.T) {
	loader := NewMultiURLLoader(
		"http://localhost:1/bad1",
		"http://localhost:1/bad2",
	)
	_, err := loader.Load(context.Background())
	if err == nil {
		t.Error("expected error when all URLs fail")
	}
}

func TestStripHTML(t *testing.T) {
	input := "<html><body><h1>Hello</h1><p>World</p></body></html>"
	out := stripHTML(input)
	if strings.Contains(out, "<") || strings.Contains(out, ">") {
		t.Errorf("HTML tags not stripped: %s", out)
	}
	if !strings.Contains(out, "Hello") || !strings.Contains(out, "World") {
		t.Errorf("content lost: %s", out)
	}
}

func TestDefaultChunker(t *testing.T) {
	chunker := DefaultChunker()
	if chunker == nil {
		t.Fatal("DefaultChunker should not return nil")
	}

	doc := Document{ID: "d1", Content: "Some content that should be chunked properly", Source: "test"}
	chunks := chunker.Chunk(doc)
	if len(chunks) == 0 {
		t.Error("expected at least 1 chunk")
	}
}

func TestNoOpEmbedder(t *testing.T) {
	embedder := NewNoOpEmbedder()
	results, err := embedder.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if embedder.Dimensions() != 0 {
		t.Errorf("NoOp should have 0 dimensions, got %d", embedder.Dimensions())
	}
}
