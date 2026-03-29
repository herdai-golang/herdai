# Strategy Advisor Build Retrospective (Interview Notes)

Date: 2026-03-08
Project: HerdAI `examples/webui` (AI Strategy Advisor)

## 1) What We Built

We started with a multi-agent strategy analysis app and evolved it into a production-style conversational advisor:

- A single front-door CSO (Chief Strategy Officer) experience for users
- Five analysis capabilities:
  - `strategic_analysis`
  - `financial_analysis`
  - `competitor_intel`
  - `gtm_analysis`
  - `consulting_evaluation`
- Real web and company intelligence via MCP tools
- Session persistence, memory layers, and tool-result caching
- Follow-up question handling with selective tool reruns

The key objective shifted from "run many analyses" to "feel like a reliable, fast, conversational advisor."

## 2) Biggest Learnings

### Learning A: Architecture quality matters more than prompts

Early behavior issues looked like prompt problems, but many were architecture problems:

- ReAct-in-ReAct caused high latency and unstable behavior
- Reconnecting MCP servers on each tool invocation added avoidable overhead
- Async memory extraction caused stale follow-up context

Prompt tuning helped, but architecture changes fixed the root causes.

### Learning B: Caching must be semantic, not only lexical

Simple "new words count" invalidation was not enough. We moved to:

- Word-level meaningful drift detection
- Field-aware invalidation tied to intake context (`idea`, `customer`, `industry`, etc.)
- Tool dependency mapping (`ToolDeps`) so only affected tools rerun

This made follow-ups both faster and more accurate.

### Learning C: UX speed is mostly perceived speed

Users judge responsiveness by time-to-first-feedback, not only total completion time.

- Better orchestration + tighter tool selection improved real latency
- Progressive rendering improved perceived smoothness
- Reducing repeated full-report responses improved trust and usability

## 3) Major Optimization Changes Made

## Orchestration and Tooling

- Moved away from always-running all analyses on follow-ups
- Added router-style decision flow: direct response vs targeted tool calls
- Executed selected tools in parallel where independent

## MCP and External Tool Calls

- Added session-level MCP pooling and tool reuse
- Avoided repeated `initialize` + `tools/list` handshakes per sub-agent run
- Added shorter/faster failure behavior for problematic MCP paths

## Cache and Context

- Migrated to framework-level `ToolCache`
- Added field-aware cache invalidation from intake and detected pivots
- Added structured context snapshots to preserve correctness and explainability

## Memory

- Implemented short-term + semantic + episodic memory layers
- Changed memory extraction from async race-prone behavior to bounded synchronous update

## Model Usage

- Split model roles:
  - Smaller model for tool decision/routing
  - Stronger model for final synthesis
- Kept tool execution model cost-aware while improving response quality

## 4) Mistakes We Made (and What We Fixed)

### Mistake 1: Repeating full report on follow-ups

Symptom: user asked "who are competitors?" and got complete long-form analysis again.

Fix:
- Narrowed follow-up response style
- Stopped injecting full cached analysis dumps into every prompt
- Routed follow-ups to only relevant tools

### Mistake 2: Treating "MCP is slow" as the first diagnosis

Symptom: competitor flow felt slow.

Reality:
- A lot of delay came from extra LLM hops and ReAct loops
- Query strategy also mattered (short indexed terms worked better than long natural phrases)

Fix:
- Better query strategy for company DB
- Fewer LLM hops
- Better orchestration path

### Mistake 3: Not separating first-analysis mode from follow-up mode

Symptom: same response pattern in all turns.

Fix:
- First-turn behavior and follow-up behavior explicitly separated
- Different routing expectations and output formats

### Mistake 4: Allowing memory freshness lag

Symptom: turn N facts sometimes available only by turn N+1 or N+2.

Fix:
- Bounded synchronous extraction step to make memory available for immediate follow-up

## 5) Trade-offs We Took (and Why)

### Trade-off A: Rich analysis vs response latency

- More tools + deeper reasoning gives stronger output but increases latency.
- Decision: run full stack on first analysis, selective tools on follow-ups.

Why: preserves depth when needed and speed for conversational turns.

### Trade-off B: Stronger model quality vs cost

- Better synthesis model improves instruction following and output quality.
- Decision: use stronger model for CSO synthesis, smaller model for routing/tool decisions.

Why: best quality/cost balance for production.

### Trade-off C: Aggressive caching vs correctness

- Reusing cached outputs is fast but risky if context changed.
- Decision: field-aware invalidation using structured context + tool dependencies.

Why: speed without silently serving stale analysis.

### Trade-off D: Simplicity vs observability

- Faster iteration tempted minimal instrumentation.
- Decision: keep architecture simple first, then add targeted logging/metrics.

Why: avoid over-engineering early, but keep enough visibility to debug real bottlenecks.

### Trade-off E: One-agent purity vs practical tool orchestration

- Fully pure "single agent does everything" sounds elegant but can become slow/unstable.
- Decision: CSO as front door, with disciplined tool orchestration behind it.

Why: keeps UX coherent while retaining engineering control.

## 6) App-Specific Insights (Useful in Interviews)

- The hardest production issue was not model IQ, but orchestration correctness under latency pressure.
- "Same answer repeatedly" was mostly architecture and prompt-context pollution, not just bad prompts.
- Structured intake was foundational for both quality and cache invalidation.
- Parallel execution and selective reruns mattered more than adding more frameworks.
- Transparency rules (sources, uncertainty, failure disclosure) directly increased user trust.

## 7) What I Would Do Next

1. True SSE token streaming end-to-end (not only progressive rendering fallback)
2. Add clear latency metrics:
   - router decision latency
   - per-tool latency
   - synthesis latency
   - p50/p95 turn latency
3. Add reliability policies:
   - retry matrix by tool type
   - graceful degradation when specific MCP tools fail
4. Add evaluation harness for follow-up quality:
   - "answers only asked question"
   - "no hallucinated competitors"
   - "correct rerun when context fields change"

## 8) Interview-Ready One-Minute Summary

I built a strategy advisor app on top of a Go multi-agent framework and learned that production quality comes from orchestration, not just prompts. We moved from a slower ReAct-in-ReAct design to a two-stage flow: lightweight routing and selective parallel tool execution, then higher-quality synthesis. I implemented field-aware cache invalidation, session-level MCP pooling, bounded memory updates, and model role-splitting for quality/cost control. The key trade-off was balancing depth vs speed while keeping correctness under context changes. The result was a faster, more conversational, and more trustworthy system.

