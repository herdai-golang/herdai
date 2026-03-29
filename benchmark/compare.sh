#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# compare.sh — Full comparison: HerdAI Go + Python frameworks
#
# Usage:
#   cd benchmark && bash compare.sh          # full comparison table
#   cd benchmark && bash compare.sh --go     # Go benchmark only
#   cd benchmark && bash compare.sh --py     # Python benchmarks only
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GO_ONLY=false
PY_ONLY=false
for arg in "$@"; do
  case $arg in
    --go) GO_ONLY=true ;;
    --py) PY_ONLY=true ;;
  esac
done

LINE="════════════════════════════════════════════════════════════════════════"

echo ""
echo "$LINE"
echo "  HerdAI Concurrency Benchmark — Full Framework Comparison"
echo "$LINE"
echo ""

# ── 1. HerdAI (Go) ────────────────────────────────────────────────────────────
if [ "$PY_ONLY" = false ]; then
  echo "▶ HerdAI (Go) — demo app (200ms LLM + 80ms tools)"
  echo "─────────────────────────────────────────────────────────────"
  go run . --quiet 2>/dev/null
  echo ""

  echo "▶ HerdAI (Go) — formal Go benchmarks (20ms LLM + 8ms tools)"
  echo "─────────────────────────────────────────────────────────────"
  go test -bench='^Benchmark(Parallel|Sequential)Agents' \
    -benchtime=3s -count=1 2>/dev/null \
    | grep -E "^(Benchmark|PASS|ok)" | column -t
  echo ""
fi

# ── 2. Python baseline (asyncio, no framework) ────────────────────────────────
if [ "$GO_ONLY" = false ]; then
  echo "▶ Pure Python asyncio — baseline (no framework, 200ms LLM + 80ms tools)"
  echo "─────────────────────────────────────────────────────────────"
  python3 python/baseline.py 2>/dev/null
  echo ""

  # ── 3. LangGraph ─────────────────────────────────────────────────────────
  echo "▶ LangGraph 1.1.x — parallel graph nodes"
  echo "─────────────────────────────────────────────────────────────"
  python3 python/langgraph_bench.py 2>/dev/null
  echo ""

  # ── 4. AutoGen ───────────────────────────────────────────────────────────
  echo "▶ AutoGen 0.7.5 — async agents"
  echo "─────────────────────────────────────────────────────────────"
  python3 python/autogen_bench.py 2>/dev/null
  echo ""

  # ── 5. CrewAI ────────────────────────────────────────────────────────────
  echo "▶ CrewAI 1.12.2 — simulated timing + code pattern + overhead"
  echo "─────────────────────────────────────────────────────────────"
  python3 python/crewai_bench.py 2>/dev/null
  echo ""

  # ── 6. Master comparison table ────────────────────────────────────────────
  echo "$LINE"
  echo "▶ MASTER COMPARISON TABLE"
  echo "$LINE"
  python3 python/compare_all.py 2>/dev/null
fi
