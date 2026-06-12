#!/usr/bin/env bash
# Generate one batch via Codex CLI (capture STDOUT -> extract JSON). Usage: gen_one.sh batch_0000
BASE="/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen"
N="$1"
IN="$BASE/batch_in/$N.json"
OUT="$BASE/batch_out/$N.json"
RAW="$BASE/batch_out/$N.raw"
ERR="$BASE/batch_out/$N.err"
mkdir -p "$BASE/batch_out"

# skip if already produced and valid
if [ -f "$OUT" ] && python3 -c "import json; json.load(open('$OUT'))" 2>/dev/null; then
  echo "skip $N"; exit 0
fi

RECIPE="$(cat "$BASE/scripts/recipe.md")"
SRC="$(cat "$IN")"
PROMPT="$RECIPE

아래는 입력 JSON 배열이다(각 원소 {id,category,title,body,comment_count,comments:[{id,parent_id,body}]}).
모든 post의 title/body와 모든 comment의 body를 위 규칙대로 재작성하라.
출력은 **유효한 JSON 배열 하나만** 출력하라 — 코드펜스(\`\`\`)·설명·앞뒤 텍스트 금지, 파일도 쓰지 마라.
형식: [{\"id\":\"<uuid>\",\"title\":\"<new>\",\"body\":\"<new>\",\"comments\":[{\"id\":\"<uuid>\",\"parent_id\":<null|uuid>,\"body\":\"<new>\"}]}]
모든 post/comment 포함, id/parent_id/댓글수 정확히 보존.

입력:
$SRC"

timeout 900 codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "$PROMPT" >"$RAW" 2>"$ERR" || true

python3 - "$RAW" "$OUT" <<'PY'
import sys, json, re
raw = open(sys.argv[1], encoding="utf-8", errors="replace").read()
# strip code fences if any, then take outermost [...]
raw = raw.replace("```json", "").replace("```", "")
i, j = raw.find("["), raw.rfind("]")
if i == -1 or j == -1 or j <= i:
    sys.exit(2)
try:
    data = json.loads(raw[i:j+1])
    assert isinstance(data, list) and data
    json.dump(data, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
except Exception as e:
    sys.stderr.write(f"parse fail: {e}\n"); sys.exit(3)
PY

if [ -f "$OUT" ]; then echo "ok $N ($(python3 -c "import json;print(len(json.load(open('$OUT'))))" 2>/dev/null) posts)"; rm -f "$RAW"; else echo "FAIL $N"; exit 1; fi
