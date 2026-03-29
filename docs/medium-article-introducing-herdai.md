<!--
  MEDIUM PUBLISHING CHECKLIST
  ───────────────────────────
  1. Create story → paste everything BELOW the line "=== START PASTE ==="
  2. In Medium: set the headline + subtitle using the TITLE / SUBTITLE lines (or type them in Medium’s fields and delete those two lines from the body)
  3. Tags (pick 5): Golang, Artificial Intelligence, Programming, Open Source, Software Development
-->

TITLE (Medium headline field):
Introducing HerdAI: A Production-Grade AI Agent Framework for Go

SUBTITLE (Medium kicker / subtitle field):
How to build LLM agents, tool use, and multi-agent teams without dragging half of PyPI into production.

=== START PASTE (delete the TITLE/SUBTITLE block above after you’ve set them in Medium) ===

Most AI agent frameworks live in Python—and for good reason. The notebooks are there, the models are there, and the tutorials assume Python first.

But a lot of real systems run on **Go**: APIs, workers, data pipelines, and infra that already need **low cold starts**, **small binaries**, **strict concurrency**, and **minimal dependencies**. If your agents sit next to that code, shipping a Python sidecar or a second runtime isn’t always what you want.

**HerdAI** is an open-source AI agent framework written in **Go**. It gives you agents, tools, multi-agent orchestration, RAG, memory, guardrails, human-in-the-loop, tracing, MCP, and an eval harness—while keeping the **library itself** free of extra third-party Go modules.


**Why Go for agents?**

• **One binary, predictable deploys** — no virtualenv, no import storms on startup.

• **Goroutines** — parallel tools and parallel agents are first-class patterns, not a special case.

• **Operational fit** — the same language as your HTTP services and workers.

If you’re comparing to Python stacks like CrewAI or LangGraph, the tradeoff is ecosystem: Python still owns most notebooks and ML glue. HerdAI is aimed at teams that want **agents inside Go services**, not a second language runtime.


**What you get**

• **Agents** — role, goal, optional backstory; timeouts; max tool calls; structured logging.

• **Tools** — OpenAI-style function calling: the LLM decides when to invoke your Go functions.

• **Parallel tool calls** — multiple tools from one LLM turn can run concurrently (configurable).

• **Multi-agent managers** — **sequential** pipelines, **parallel** fan-out, **round-robin** turn-taking, and an **LLM router** (supervisor picks the next agent). Managers can nest.

• **RAG** — ingest from files, URLs, strings, directories; keyword or hybrid retrieval; optional citations.

• **Memory** — layered store with search, tags, TTL, and scoping.

• **Guardrails** — input/output chains (length, patterns, PII, injection, JSON shape, custom).

• **Human-in-the-loop** — approve, reject, or edit tool calls before they run.

• **Tracing** — hierarchical spans (agent, LLM, tool, manager, MCP, memory, RAG).

• **Sessions** — persist and resume runs.

• **Eval harness** — suites, assertions, baselines for regression checks.

• **MCP** — connect to Model Context Protocol servers so tools appear at runtime.

The **LLM** is pluggable: OpenAI-compatible APIs (OpenAI, Mistral, Groq, Ollama, etc.), plus **`MockLLM`** for tests and examples with **no API key**.


**Install**

```bash
go get github.com/herdai-golang/herdai@latest
```

```go
import "github.com/herdai-golang/herdai"
```

**Docs:** https://pkg.go.dev/github.com/herdai-golang/herdai

**Source & examples:** https://github.com/herdai-golang/herdai


**Minimal example (no API key)**

The repo includes runnable examples that use `MockLLM`. The smallest is a single agent and one reply:

```go
package main

import (
    "context"
    "fmt"

    "github.com/herdai-golang/herdai"
)

func main() {
    mock := &herdai.MockLLM{}
    mock.PushResponse(herdai.LLMResponse{
        Content: "HerdAI is a Go library for building AI agents with tools and multi-agent teams.",
    })

    agent := herdai.NewAgent(herdai.AgentConfig{
        ID:   "assistant",
        Role: "Assistant",
        Goal: "Answer briefly.",
        LLM:  mock,
    })

    result, err := agent.Run(context.Background(), "What is HerdAI?", nil)
    if err != nil {
        panic(err)
    }
    fmt.Println(result.Content)
}
```

From a clone: `cd examples/hello_minimal && go run .`

There are also examples for **multiple tools in one agent**, a **supervisor + three specialists** (`StrategyLLMRouter`), **concurrent agents**, **HITL over a channel**, and **RAG on a PDF**—all documented in the README.


**Pass a question, get an answer (and multi-turn chat)**

HerdAI doesn’t hide “chat” behind a separate product API. Each turn is **`agent.Run(ctx, userMessage, conversation)`**: you pass the user’s question (or instruction) as a string; you get back a **`Result`** with the model’s reply in **`Content`**.

• **Single question, no history:** pass `nil` for the conversation — `agent.Run(ctx, "Explain this error…", nil)`.

• **Back-and-forth chat:** create **`herdai.NewConversation()`** once, then pass that same `*Conversation` into every `Run`. The agent includes recent turns in context, so follow-ups stay coherent.

A minimal **read–eval loop** (stdin in, answers out) looks like this:

```go
// imports: "bufio", "context", "fmt", "log", "os", "strings", plus herdai
ctx := context.Background()
conv := herdai.NewConversation()
scanner := bufio.NewScanner(os.Stdin)
fmt.Println("Type a message (empty line exits).")
for fmt.Print("You: "); scanner.Scan(); {
    line := strings.TrimSpace(scanner.Text())
    if line == "" {
        break
    }
    res, err := agent.Run(ctx, line, conv)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Assistant:", res.Content)
}
```

Swap **`MockLLM`** for **`herdai.NewMistral(...)`** or **`herdai.NewOpenAI(...)`** with your API key, and you have a local CLI chat against a real model.

**More relevant, document-grounded answers:** pair the same pattern with **RAG** (`examples/rag_simple`): ingest your PDFs or URLs, then ask questions—the retriever pulls relevant chunks before the LLM answers, so replies track your source material.

There is **no hosted web chat** in the library repo itself; you wrap `Run` in your own HTTP or WebSocket handler when you embed agents in a service. Questions and discussion: **https://github.com/herdai-golang/herdai/issues**


**Philosophy**

HerdAI is **not** trying to replace Python for every research workflow. It is trying to make **agents a first-class citizen in Go codebases**: testable, deployable, and operable next to the rest of your stack.

If you care about **dependency count**, **binary size**, and **runtime behavior**, it’s worth a look.


**License**

MIT — see the repo on GitHub: https://github.com/herdai-golang/herdai


**About the project**

**HerdAI** is developed as open source under the **herdai-golang** organization on GitHub. Feedback, issues, and PRs are welcome.
