package herdai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// RAG — Retrieval-Augmented Generation
//
// Documents are loaded ONCE at setup time via the ingestion pipeline, but the
// vector store is live — you can add more documents at any time.
//
// Architecture:
//   Loader → Chunker → Embedder → VectorStore ← Retriever ← Agent
// ═══════════════════════════════════════════════════════════════════════════════

// ── Core Types ─────────────────────────────────────────────────────────────

// Document represents a loaded document before chunking.
type Document struct {
	ID       string         `json:"id"`
	Content  string         `json:"content"`
	Source   string         `json:"source"`   // file path, URL, or identifier
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Chunk is a piece of a document, ready for embedding and retrieval.
type Chunk struct {
	ID         string         `json:"id"`
	Content    string         `json:"content"`
	DocumentID string         `json:"document_id"`
	Source     string         `json:"source"`
	Index      int            `json:"index"`    // position within the document
	Metadata   map[string]any `json:"metadata,omitempty"`
	Embedding  []float64      `json:"embedding,omitempty"`
	Score      float64        `json:"score,omitempty"` // relevance score from retrieval
}

// RAGConfig configures retrieval-augmented generation for an agent.
type RAGConfig struct {
	Retriever   Retriever              // how to find relevant chunks
	TopK        int                    // number of chunks to inject (default: 5)
	MinScore    float64                // minimum relevance threshold (default: 0.0)
	Template    string                 // format template for context injection (optional)
	CiteSources bool                   // include source references in context
	QueryRewriter func(input string) string // optional query transformation before retrieval
}

// ── Interfaces ─────────────────────────────────────────────────────────────

// Loader loads raw documents from a source.
type Loader interface {
	Load(ctx context.Context) ([]Document, error)
}

// Chunker splits a document into smaller chunks.
type Chunker interface {
	Chunk(doc Document) []Chunk
}

// Embedder converts text into vector embeddings.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float64, error)
	Dimensions() int
}

// VectorStore stores chunks with optional embeddings and supports search.
type VectorStore interface {
	Add(ctx context.Context, chunks []Chunk) error
	Search(ctx context.Context, query string, embedding []float64, topK int) ([]Chunk, error)
	Count() int
	Clear(ctx context.Context) error
}

// Retriever finds relevant chunks for a query.
type Retriever interface {
	Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Loaders
// ═══════════════════════════════════════════════════════════════════════════════

// TextLoader loads a single file as a document.
type TextLoader struct {
	Path string
}

func NewTextLoader(path string) *TextLoader {
	return &TextLoader{Path: path}
}

func (l *TextLoader) Load(_ context.Context) ([]Document, error) {
	data, err := os.ReadFile(l.Path)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", l.Path, err)
	}
	return []Document{{
		ID:      generateID(),
		Content: string(data),
		Source:  l.Path,
		Metadata: map[string]any{
			"loader": "text",
			"path":   l.Path,
		},
	}}, nil
}

// DirectoryLoader loads all files matching a glob pattern from a directory.
type DirectoryLoader struct {
	Dir     string
	Pattern string // e.g. "*.md", "*.txt"
}

func NewDirectoryLoader(dir, pattern string) *DirectoryLoader {
	return &DirectoryLoader{Dir: dir, Pattern: pattern}
}

func (l *DirectoryLoader) Load(_ context.Context) ([]Document, error) {
	matches, err := filepath.Glob(filepath.Join(l.Dir, l.Pattern))
	if err != nil {
		return nil, fmt.Errorf("glob %s/%s: %w", l.Dir, l.Pattern, err)
	}

	var docs []Document
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		docs = append(docs, Document{
			ID:      generateID(),
			Content: string(data),
			Source:  path,
			Metadata: map[string]any{
				"loader": "directory",
				"path":   path,
			},
		})
	}
	return docs, nil
}

// StringsLoader loads documents from in-memory strings.
// Useful for testing and dynamic content.
type StringsLoader struct {
	Documents []Document
}

func NewStringsLoader(sources map[string]string) *StringsLoader {
	var docs []Document
	for source, content := range sources {
		docs = append(docs, Document{
			ID:      generateID(),
			Content: content,
			Source:  source,
			Metadata: map[string]any{"loader": "strings"},
		})
	}
	return &StringsLoader{Documents: docs}
}

func (l *StringsLoader) Load(_ context.Context) ([]Document, error) {
	return l.Documents, nil
}

// URLLoader fetches a document from a URL.
// Use for loading online articles, docs, APIs, or any web page.
type URLLoader struct {
	URL     string
	Timeout time.Duration
}

func NewURLLoader(url string) *URLLoader {
	return &URLLoader{URL: url, Timeout: 30 * time.Second}
}

func (l *URLLoader) Load(ctx context.Context) ([]Document, error) {
	client := &http.Client{Timeout: l.Timeout}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, l.URL, nil)
	if err != nil {
		return nil, fmt.Errorf("url loader: %w", err)
	}
	req.Header.Set("User-Agent", "HerdAI/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch %s: %w", l.URL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetch %s: status %d", l.URL, resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", l.URL, err)
	}

	content := string(body)

	// Strip HTML tags if the response looks like HTML
	if strings.Contains(resp.Header.Get("Content-Type"), "text/html") || strings.HasPrefix(strings.TrimSpace(content), "<") {
		content = stripHTML(content)
	}

	return []Document{{
		ID:      generateID(),
		Content: content,
		Source:  l.URL,
		Metadata: map[string]any{
			"loader":       "url",
			"url":          l.URL,
			"content_type": resp.Header.Get("Content-Type"),
			"fetched_at":   time.Now().Format(time.RFC3339),
		},
	}}, nil
}

// MultiURLLoader loads documents from multiple URLs.
func NewMultiURLLoader(urls ...string) *multiURLLoader {
	return &multiURLLoader{urls: urls}
}

type multiURLLoader struct {
	urls []string
}

func (l *multiURLLoader) Load(ctx context.Context) ([]Document, error) {
	var allDocs []Document
	for _, u := range l.urls {
		loader := NewURLLoader(u)
		docs, err := loader.Load(ctx)
		if err != nil {
			continue // skip failed URLs
		}
		allDocs = append(allDocs, docs...)
	}
	if len(allDocs) == 0 {
		return nil, fmt.Errorf("all %d URLs failed to load", len(l.urls))
	}
	return allDocs, nil
}

// ReaderLoader loads a document from any io.Reader.
// Use for dynamic uploads (PDF extraction output, HTTP responses, etc.)
type ReaderLoader struct {
	Reader io.Reader
	Source string
}

func NewReaderLoader(r io.Reader, source string) *ReaderLoader {
	return &ReaderLoader{Reader: r, Source: source}
}

func (l *ReaderLoader) Load(_ context.Context) ([]Document, error) {
	data, err := io.ReadAll(l.Reader)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", l.Source, err)
	}
	return []Document{{
		ID:      generateID(),
		Content: string(data),
		Source:  l.Source,
		Metadata: map[string]any{
			"loader": "reader",
			"source": l.Source,
		},
	}}, nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// Chunkers
// ═══════════════════════════════════════════════════════════════════════════════

// FixedSizeChunker splits text into chunks of a fixed character size with overlap.
type FixedSizeChunker struct {
	Size    int // characters per chunk
	Overlap int // overlapping characters between chunks
}

func NewFixedSizeChunker(size, overlap int) *FixedSizeChunker {
	if overlap >= size {
		overlap = size / 4
	}
	return &FixedSizeChunker{Size: size, Overlap: overlap}
}

func (c *FixedSizeChunker) Chunk(doc Document) []Chunk {
	text := doc.Content
	if len(text) == 0 {
		return nil
	}

	var chunks []Chunk
	step := c.Size - c.Overlap
	if step <= 0 {
		step = c.Size
	}

	for i := 0; i < len(text); i += step {
		end := i + c.Size
		if end > len(text) {
			end = len(text)
		}
		chunk := text[i:end]
		if strings.TrimSpace(chunk) == "" {
			continue
		}
		chunks = append(chunks, Chunk{
			ID:         generateID(),
			Content:    chunk,
			DocumentID: doc.ID,
			Source:     doc.Source,
			Index:      len(chunks),
			Metadata:   copyMeta(doc.Metadata),
		})
		if end >= len(text) {
			break
		}
	}
	return chunks
}

// ParagraphChunker splits text on double newlines (paragraphs).
// Merges small paragraphs together until reaching the target size.
type ParagraphChunker struct {
	TargetSize int // target characters per chunk
	Overlap    int // overlap by repeating the last N chars of prev chunk
}

func NewParagraphChunker(targetSize, overlap int) *ParagraphChunker {
	return &ParagraphChunker{TargetSize: targetSize, Overlap: overlap}
}

func (c *ParagraphChunker) Chunk(doc Document) []Chunk {
	paragraphs := splitParagraphs(doc.Content)
	if len(paragraphs) == 0 {
		return nil
	}

	var chunks []Chunk
	var current strings.Builder
	prevTail := ""

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}

		if current.Len() > 0 && current.Len()+len(para)+2 > c.TargetSize {
			content := current.String()
			if prevTail != "" {
				content = prevTail + "\n\n" + content
			}
			chunks = append(chunks, Chunk{
				ID:         generateID(),
				Content:    strings.TrimSpace(content),
				DocumentID: doc.ID,
				Source:     doc.Source,
				Index:      len(chunks),
				Metadata:   copyMeta(doc.Metadata),
			})
			if c.Overlap > 0 && len(current.String()) > c.Overlap {
				s := current.String()
				prevTail = s[len(s)-c.Overlap:]
			} else {
				prevTail = ""
			}
			current.Reset()
		}

		if current.Len() > 0 {
			current.WriteString("\n\n")
		}
		current.WriteString(para)
	}

	if current.Len() > 0 {
		content := current.String()
		if prevTail != "" {
			content = prevTail + "\n\n" + content
		}
		chunks = append(chunks, Chunk{
			ID:         generateID(),
			Content:    strings.TrimSpace(content),
			DocumentID: doc.ID,
			Source:     doc.Source,
			Index:      len(chunks),
			Metadata:   copyMeta(doc.Metadata),
		})
	}

	return chunks
}

// MarkdownChunker splits on markdown headers (## lines).
// Each section becomes a chunk with the header preserved.
type MarkdownChunker struct {
	MaxSize int // max characters per chunk (splits large sections further)
}

func NewMarkdownChunker(maxSize int) *MarkdownChunker {
	if maxSize <= 0 {
		maxSize = 1000
	}
	return &MarkdownChunker{MaxSize: maxSize}
}

func (c *MarkdownChunker) Chunk(doc Document) []Chunk {
	lines := strings.Split(doc.Content, "\n")
	var chunks []Chunk
	var current strings.Builder
	currentHeader := ""

	flush := func() {
		text := strings.TrimSpace(current.String())
		if text == "" {
			return
		}
		meta := copyMeta(doc.Metadata)
		if currentHeader != "" {
			meta["section"] = currentHeader
		}
		chunks = append(chunks, Chunk{
			ID:         generateID(),
			Content:    text,
			DocumentID: doc.ID,
			Source:     doc.Source,
			Index:      len(chunks),
			Metadata:   meta,
		})
		current.Reset()
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			flush()
			currentHeader = strings.TrimLeft(trimmed, "# ")
		}

		if current.Len()+len(line)+1 > c.MaxSize && current.Len() > 0 {
			flush()
		}

		current.WriteString(line)
		current.WriteString("\n")
	}
	flush()

	return chunks
}

// RecursiveChunker tries paragraph splitting first, falls back to fixed-size.
type RecursiveChunker struct {
	TargetSize int
	Overlap    int
}

func NewRecursiveChunker(targetSize, overlap int) *RecursiveChunker {
	return &RecursiveChunker{TargetSize: targetSize, Overlap: overlap}
}

func (c *RecursiveChunker) Chunk(doc Document) []Chunk {
	para := NewParagraphChunker(c.TargetSize, c.Overlap)
	chunks := para.Chunk(doc)

	// Split any chunks that are still too large
	var result []Chunk
	fixed := NewFixedSizeChunker(c.TargetSize, c.Overlap)
	for _, ch := range chunks {
		if len(ch.Content) > c.TargetSize*2 {
			subDoc := Document{ID: ch.DocumentID, Content: ch.Content, Source: ch.Source, Metadata: ch.Metadata}
			subChunks := fixed.Chunk(subDoc)
			result = append(result, subChunks...)
		} else {
			result = append(result, ch)
		}
	}
	return result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Embedders
// ═══════════════════════════════════════════════════════════════════════════════

// NoOpEmbedder skips embedding entirely. Use for keyword-only RAG.
type NoOpEmbedder struct{}

func NewNoOpEmbedder() *NoOpEmbedder { return &NoOpEmbedder{} }

func (e *NoOpEmbedder) Embed(_ context.Context, texts []string) ([][]float64, error) {
	result := make([][]float64, len(texts))
	return result, nil
}

func (e *NoOpEmbedder) Dimensions() int { return 0 }

// OpenAIEmbedder calls the OpenAI-compatible embeddings API.
// Works with OpenAI, Mistral, and any compatible provider.
type OpenAIEmbedder struct {
	baseURL string
	apiKey  string
	model   string
	dims    int
	client  *http.Client
}

type EmbedderConfig struct {
	BaseURL string // default: "https://api.openai.com/v1"
	APIKey  string // default: OPENAI_API_KEY env var
	Model   string // default: "text-embedding-3-small"
}

func NewOpenAIEmbedder(cfg EmbedderConfig) *OpenAIEmbedder {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.APIKey == "" {
		cfg.APIKey = os.Getenv("OPENAI_API_KEY")
	}
	if cfg.Model == "" {
		cfg.Model = "text-embedding-3-small"
	}

	dims := 1536
	if strings.Contains(cfg.Model, "3-small") {
		dims = 1536
	} else if strings.Contains(cfg.Model, "3-large") {
		dims = 3072
	}

	return &OpenAIEmbedder{
		baseURL: strings.TrimRight(cfg.BaseURL, "/"),
		apiKey:  cfg.APIKey,
		model:   cfg.Model,
		dims:    dims,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

func NewMistralEmbedder(apiKey string) *OpenAIEmbedder {
	if apiKey == "" {
		apiKey = os.Getenv("MISTRAL_API_KEY")
	}
	return &OpenAIEmbedder{
		baseURL: "https://api.mistral.ai/v1",
		apiKey:  apiKey,
		model:   "mistral-embed",
		dims:    1024,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	body := map[string]any{
		"model": e.model,
		"input": texts,
	}
	bodyJSON, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embeddings", bytes.NewReader(bodyJSON))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embeddings request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embeddings response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embeddings API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse embeddings response: %w", err)
	}

	embeddings := make([][]float64, len(result.Data))
	for i, d := range result.Data {
		embeddings[i] = d.Embedding
	}
	return embeddings, nil
}

func (e *OpenAIEmbedder) Dimensions() int { return e.dims }

// ═══════════════════════════════════════════════════════════════════════════════
// InMemoryVectorStore — zero-dep vector store with keyword + cosine search
// ═══════════════════════════════════════════════════════════════════════════════

type InMemoryVectorStore struct {
	mu     sync.RWMutex
	chunks []Chunk
}

func NewInMemoryVectorStore() *InMemoryVectorStore {
	return &InMemoryVectorStore{}
}

func (s *InMemoryVectorStore) Add(_ context.Context, chunks []Chunk) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.chunks = append(s.chunks, chunks...)
	return nil
}

func (s *InMemoryVectorStore) Search(_ context.Context, query string, embedding []float64, topK int) ([]Chunk, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.chunks) == 0 {
		return nil, nil
	}

	type scored struct {
		chunk Chunk
		score float64
	}

	var results []scored

	hasEmbedding := len(embedding) > 0

	for _, ch := range s.chunks {
		var score float64
		if hasEmbedding && len(ch.Embedding) > 0 {
			score = cosineSimilarity(embedding, ch.Embedding)
		} else {
			score = keywordScore(query, ch.Content)
		}

		if score > 0 {
			c := ch
			c.Score = score
			results = append(results, scored{chunk: c, score: score})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}

	out := make([]Chunk, len(results))
	for i, r := range results {
		out[i] = r.chunk
	}
	return out, nil
}

func (s *InMemoryVectorStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.chunks)
}

func (s *InMemoryVectorStore) Clear(_ context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.chunks = nil
	return nil
}

// AllChunks returns a copy of all stored chunks (for inspection/debugging).
func (s *InMemoryVectorStore) AllChunks() []Chunk {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Chunk, len(s.chunks))
	copy(out, s.chunks)
	return out
}

// ═══════════════════════════════════════════════════════════════════════════════
// Retrievers
// ═══════════════════════════════════════════════════════════════════════════════

// KeywordRetriever does keyword-based search (no embeddings needed).
type KeywordRetriever struct {
	Store VectorStore
}

func NewKeywordRetriever(store VectorStore) *KeywordRetriever {
	return &KeywordRetriever{Store: store}
}

func (r *KeywordRetriever) Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error) {
	return r.Store.Search(ctx, query, nil, topK)
}

// VectorRetriever does cosine similarity search using embeddings.
type VectorRetriever struct {
	Store    VectorStore
	Embedder Embedder
}

func NewVectorRetriever(store VectorStore, embedder Embedder) *VectorRetriever {
	return &VectorRetriever{Store: store, Embedder: embedder}
}

func (r *VectorRetriever) Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error) {
	embeddings, err := r.Embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	if len(embeddings) == 0 || len(embeddings[0]) == 0 {
		return r.Store.Search(ctx, query, nil, topK)
	}
	return r.Store.Search(ctx, query, embeddings[0], topK)
}

// HybridRetriever combines keyword and vector search, deduplicates and merges scores.
type HybridRetriever struct {
	Keyword *KeywordRetriever
	Vector  *VectorRetriever
	Alpha   float64 // weight for vector score vs keyword (0=keyword only, 1=vector only, 0.5=equal)
}

func NewHybridRetriever(store VectorStore, embedder Embedder, alpha float64) *HybridRetriever {
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	return &HybridRetriever{
		Keyword: NewKeywordRetriever(store),
		Vector:  NewVectorRetriever(store, embedder),
		Alpha:   alpha,
	}
}

func (r *HybridRetriever) Retrieve(ctx context.Context, query string, topK int) ([]Chunk, error) {
	fetchK := topK * 3

	kwResults, err := r.Keyword.Retrieve(ctx, query, fetchK)
	if err != nil {
		return nil, err
	}

	vecResults, err := r.Vector.Retrieve(ctx, query, fetchK)
	if err != nil {
		kwResults = applyScoreLimit(kwResults, topK)
		return kwResults, nil
	}

	merged := make(map[string]Chunk)
	for _, ch := range kwResults {
		existing, ok := merged[ch.ID]
		if ok {
			existing.Score = existing.Score*(1-r.Alpha) + ch.Score*(1-r.Alpha)
			merged[ch.ID] = existing
		} else {
			ch.Score = ch.Score * (1 - r.Alpha)
			merged[ch.ID] = ch
		}
	}
	for _, ch := range vecResults {
		existing, ok := merged[ch.ID]
		if ok {
			existing.Score += ch.Score * r.Alpha
			merged[ch.ID] = existing
		} else {
			ch.Score = ch.Score * r.Alpha
			merged[ch.ID] = ch
		}
	}

	var all []Chunk
	for _, ch := range merged {
		all = append(all, ch)
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].Score > all[j].Score
	})

	return applyScoreLimit(all, topK), nil
}

func applyScoreLimit(chunks []Chunk, topK int) []Chunk {
	if topK > 0 && len(chunks) > topK {
		return chunks[:topK]
	}
	return chunks
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ingestion Pipeline
// ═══════════════════════════════════════════════════════════════════════════════

// IngestionConfig defines how documents are loaded, chunked, embedded, and stored.
type IngestionConfig struct {
	Loader   Loader
	Chunker  Chunker
	Embedder Embedder // nil or NoOpEmbedder for keyword-only
	Store    VectorStore
}

// IngestionStats reports what the pipeline processed.
type IngestionStats struct {
	Documents int           `json:"documents"`
	Chunks    int           `json:"chunks"`
	Duration  time.Duration `json:"duration"`
}

// IngestionPipeline loads, chunks, embeds, and stores documents.
type IngestionPipeline struct {
	config IngestionConfig
}

func NewIngestionPipeline(cfg IngestionConfig) *IngestionPipeline {
	return &IngestionPipeline{config: cfg}
}

// Ingest runs the full pipeline: load → chunk → embed → store.
func (p *IngestionPipeline) Ingest(ctx context.Context) (*IngestionStats, error) {
	start := time.Now()

	docs, err := p.config.Loader.Load(ctx)
	if err != nil {
		return nil, fmt.Errorf("load documents: %w", err)
	}

	var allChunks []Chunk
	for _, doc := range docs {
		chunks := p.config.Chunker.Chunk(doc)
		allChunks = append(allChunks, chunks...)
	}

	if len(allChunks) == 0 {
		return &IngestionStats{Documents: len(docs), Duration: time.Since(start)}, nil
	}

	// Embed chunks if an embedder is provided (not NoOp)
	if p.config.Embedder != nil && p.config.Embedder.Dimensions() > 0 {
		batchSize := 100
		for i := 0; i < len(allChunks); i += batchSize {
			end := i + batchSize
			if end > len(allChunks) {
				end = len(allChunks)
			}

			texts := make([]string, end-i)
			for j, ch := range allChunks[i:end] {
				texts[j] = ch.Content
			}

			embeddings, err := p.config.Embedder.Embed(ctx, texts)
			if err != nil {
				return nil, fmt.Errorf("embed chunks %d-%d: %w", i, end, err)
			}

			for j, emb := range embeddings {
				allChunks[i+j].Embedding = emb
			}
		}
	}

	if err := p.config.Store.Add(ctx, allChunks); err != nil {
		return nil, fmt.Errorf("store chunks: %w", err)
	}

	return &IngestionStats{
		Documents: len(docs),
		Chunks:    len(allChunks),
		Duration:  time.Since(start),
	}, nil
}

// IngestDocuments is a convenience for adding documents without a full loader.
// Use this to add documents dynamically (e.g., user uploads mid-conversation).
func IngestDocuments(ctx context.Context, store VectorStore, chunker Chunker, embedder Embedder, docs ...Document) (*IngestionStats, error) {
	loader := &StringsLoader{Documents: docs}
	pipeline := NewIngestionPipeline(IngestionConfig{
		Loader:   loader,
		Chunker:  chunker,
		Embedder: embedder,
		Store:    store,
	})
	return pipeline.Ingest(ctx)
}

// SimpleRAG creates a minimal RAGConfig with keyword-only retrieval.
// This is the fastest way to add RAG — no embeddings, no API calls.
func SimpleRAG(store VectorStore, topK int) *RAGConfig {
	if topK <= 0 {
		topK = 5
	}
	return &RAGConfig{
		Retriever:   NewKeywordRetriever(store),
		TopK:        topK,
		CiteSources: true,
	}
}

// DefaultChunker returns a RecursiveChunker with sensible defaults.
func DefaultChunker() Chunker {
	return NewRecursiveChunker(500, 50)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func keywordScore(query, content string) float64 {
	queryLower := strings.ToLower(query)
	contentLower := strings.ToLower(content)
	words := strings.Fields(queryLower)

	if len(words) == 0 {
		return 0
	}

	var score float64
	for _, w := range words {
		if len(w) < 2 {
			continue
		}
		count := strings.Count(contentLower, w)
		if count > 0 {
			score += 1.0 + math.Log(float64(count))
		}
	}

	// Normalize by number of query words
	return score / float64(len(words))
}

func splitParagraphs(text string) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	return strings.Split(text, "\n\n")
}

func stripHTML(s string) string {
	var b strings.Builder
	inTag := false
	for _, r := range s {
		switch {
		case r == '<':
			inTag = true
		case r == '>':
			inTag = false
			b.WriteRune(' ')
		case !inTag:
			b.WriteRune(r)
		}
	}
	// Collapse whitespace
	result := b.String()
	for strings.Contains(result, "  ") {
		result = strings.ReplaceAll(result, "  ", " ")
	}
	for strings.Contains(result, "\n\n\n") {
		result = strings.ReplaceAll(result, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(result)
}

func copyMeta(m map[string]any) map[string]any {
	if m == nil {
		return make(map[string]any)
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
