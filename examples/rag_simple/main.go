// What this example does
//
// This program demonstrates retrieval-augmented generation (RAG) against a real PDF:
// it downloads the public “Recipe Book” PDF, extracts plain text, chunks it into an
// in-memory vector store, runs a keyword retriever (SimpleRAG / no embeddings), then
// runs an agent query. The LLM is mocked so no API key is required; stdout shows
// ingestion stats, retrieved chunks for your question (what RAG injects into the
// agent), and the mock final answer.
//
// Default: fetches the PDF over the network (~18 MB download; may take a minute).
// Offline / CI: go run . --demo  (uses a tiny embedded excerpt, no network).
//
// Run: go run .
// Run: go run . --demo
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/dslipak/pdf"
	"github.com/herdai-golang/herdai"
)

// Default recipe PDF (public). Trailing/leading spaces are trimmed.
const recipeBookURL = "https://www.i4n.in/wp-content/uploads/2023/05/Recipe-Book.pdf"

// offlineDemoText is a short stand-in when --demo is used (no download).
const offlineDemoText = `Recipe collection: East India recipes include dishes from Bengal and Odisha.
FOOD AND RELATED PRODUCTS — compiled for educational use.
Sample terms: rice, lentils, vegetables, regional spices.`

func main() {
	demo := flag.Bool("demo", false, "use embedded offline text instead of downloading the PDF (no network)")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
	ctx := context.Background()

	var text string
	var source string
	if *demo {
		if _, err := fmt.Fprintf(os.Stderr, "Mode: --demo (offline excerpt, no PDF download)\n"); err != nil {
			panic(err)
		}
		text = offlineDemoText
		source = "embedded-demo.txt"
	} else {
		if _, err := fmt.Fprintf(os.Stderr, "Downloading PDF (~18 MB) and extracting text…\n"); err != nil {
			panic(err)
		}
		var err error
		text, err = fetchPDFPlainText(ctx, recipeBookURL)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		source = recipeBookURL
	}

	store := herdai.NewInMemoryVectorStore()
	stats, err := herdai.IngestDocuments(ctx, store, herdai.DefaultChunker(), nil, herdai.Document{
		Content: text,
		Source:  source,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("Ingested: %d document(s), %d chunk(s), in %s\n", stats.Documents, stats.Chunks, stats.Duration.Round(time.Millisecond))

	query := "What regions or themes does this recipe book cover? Mention East India if relevant."
	retriever := herdai.NewKeywordRetriever(store)
	topChunks, err := retriever.Retrieve(ctx, query, 3)
	if err != nil {
		panic(err)
	}
	fmt.Println("\nRetrieved chunks (keyword RAG — what the agent injects as context):")
	for i, ch := range topChunks {
		excerpt := ch.Content
		if len(excerpt) > 220 {
			excerpt = excerpt[:220] + "…"
		}
		fmt.Printf("  [%d] score=%.3f source=%s\n      %s\n", i+1, ch.Score, ch.Source, excerpt)
	}

	// Mock LLM: fixed answer; a real LLM would read the RAG messages built by the agent.
	preview := ""
	if len(topChunks) > 0 {
		preview = strings.TrimSpace(topChunks[0].Content)
		if len(preview) > 160 {
			preview = preview[:160] + "…"
		}
	}
	mock := herdai.NewMockLLM(herdai.MockResponse{
		Content: fmt.Sprintf("[MockLLM] Example answer only. A real LLM would cite the retrieved context. Top match starts with: %q", preview),
	})

	agent := herdai.NewAgent(herdai.AgentConfig{
		ID:     "recipe-assistant",
		Role:   "Recipe assistant",
		Goal:   "Answer using the knowledge base when possible.",
		LLM:    mock,
		Logger: log,
		RAG:    herdai.SimpleRAG(store, 3),
	})

	fmt.Println("\nAgent run (RAG injected into the agent turn; tool calls: none):")
	result, err := agent.Run(ctx, query, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Content)
}

func fetchPDFPlainText(ctx context.Context, rawURL string) (string, error) {
	u := strings.TrimSpace(rawURL)
	client := &http.Client{Timeout: 5 * time.Minute}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "HerdAI-rag_simple/1.0")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch pdf: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("fetch pdf: status %d", resp.StatusCode)
	}

	tmp, err := os.CreateTemp("", "herdai-recipe-*.pdf")
	if err != nil {
		return "", err
	}
	path := tmp.Name()
	defer func() { _ = os.Remove(path) }()

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		_ = tmp.Close()
		return "", fmt.Errorf("save pdf: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}

	r, err := pdf.Open(path)
	if err != nil {
		return "", fmt.Errorf("open pdf: %w", err)
	}
	rd, err := r.GetPlainText()
	if err != nil {
		return "", fmt.Errorf("pdf plain text: %w", err)
	}
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(rd); err != nil {
		return "", err
	}
	s := strings.TrimSpace(buf.String())
	if s == "" {
		return "", fmt.Errorf("extracted text from pdf is empty")
	}
	return s, nil
}
