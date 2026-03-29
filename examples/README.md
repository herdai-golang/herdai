# Examples

Each folder is its own small Go module (`go run .`). All use `MockLLM` unless noted — **no API key** for the demos below.

| Folder | What it does |
|--------|----------------|
| [hello_minimal](hello_minimal) | Minimal program: one agent, one `Run`, no tools. |
| [single_agent_tools](single_agent_tools) | One agent, two tools; mock requests both tools then returns a final answer. |
| [supervisor_three_agents](supervisor_three_agents) | `StrategyLLMRouter`: supervisor LLM routes to three specialists, then `FINISH`. |
| [concurrency_benchmark](concurrency_benchmark) | Timings for parallel vs sequential managers and parallel vs sequential tools. |
| [hitl_channel](hitl_channel) | **HITL** with `ChannelHITLHandler`: dangerous tool (`delete_file`) pauses for approval before `Execute`. |
| [concurrent_questions](concurrent_questions) | Many goroutines each with their own agent + `Conversation` + mock (race-safe); `go test -race .` |
| [rag_simple](rag_simple) | **RAG**: ingests the public [Recipe Book PDF](https://www.i4n.in/wp-content/uploads/2023/05/Recipe-Book.pdf) (text extract + chunk + keyword retrieval). Offline: `go run . --demo`. Extra dep: `github.com/dslipak/pdf` for PDF text. |

Use the published library from your own module:

```bash
go get github.com/herdai-golang/herdai@latest
```

```go
import "github.com/herdai-golang/herdai"
```
