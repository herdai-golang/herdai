# Examples

Run from any folder with `go run .` (no API key — examples use `herdai.MockLLM`).

| Folder | Description |
|--------|-------------|
| [hello_minimal](hello_minimal) | One agent, one response |
| [single_agent_tools](single_agent_tools) | One agent, multiple tools |
| [supervisor_three_agents](supervisor_three_agents) | LLM supervisor + three specialists (`StrategyLLMRouter`) |
| [concurrency_benchmark](concurrency_benchmark) | Parallel vs sequential benchmark |
| [hitl_channel](hitl_channel) | Human-in-the-loop via `ChannelHITLHandler` (UI / WebSocket style) |
| [concurrent_questions](concurrent_questions) | Many goroutines, each with its own agent + conversation (race-safe pattern; `go test -race`) |
| [rag_simple](rag_simple) | Keyword RAG with `SimpleRAG` and an in-memory store |

Use the published module:

```bash
go get github.com/herdai-golang/herdai@latest
```

```go
import "github.com/herdai-golang/herdai"
```
