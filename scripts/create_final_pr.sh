#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# Find latest eval run branch
LATEST_EVAL=$(git branch -r | grep eval-run- | sort | tail -1 | xargs)
if [ -z "$LATEST_EVAL" ]; then echo "No eval-run branch found"; exit 1; fi

# Extract metrics snapshot from remote branch
STAMP=$(echo "$LATEST_EVAL" | sed "s|.*eval-run-||")
METRICS=$(git show "$LATEST_EVAL:runs/eval-run-$STAMP/metrics.json" 2>/dev/null || echo "{}")

# Load loop state
CYCLES=$(python3 -c "import json; print(json.load(open(\".state/loop_state.json\"))[\"cycle\"])" 2>/dev/null || echo 1)
BUDGET=$(python3 -c "import json; print(json.load(open(\".state/budget_spent.json\"))[\"usd\"])" 2>/dev/null || echo 0)

gh pr create --base main --head codex/train-runpod-fixes \
  --title "feat(training): Qwen3-8B-Base Korean FT + autonomous eval loop" \
  --body "## Summary
Phases 0-6 of dalbitalba training infrastructure. Cycle $CYCLES completed. Budget spent: \$$BUDGET.

## Final eval metrics (eval-run-$STAMP)
\`\`\`json
$METRICS
\`\`\`

## Commits
$(git log --oneline origin/main..HEAD | head -30)

🤖 Autonomous loop built by Claude 4.7 + Codex CLI
"
echo ".state/PR_CREATED.json" 
python3 -c "import json, sys; json.dump({\"created_at\": __import__(\"datetime\").datetime.utcnow().isoformat()+\"Z\", \"eval_run\": \"$STAMP\"}, open(\".state/PR_CREATED.json\", \"w\"))"
