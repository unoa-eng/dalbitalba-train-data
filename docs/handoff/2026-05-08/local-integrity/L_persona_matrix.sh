#!/bin/bash
# Stage L: persona generation matrix — 5 personas x 3 prompts = 15 generations
# Uses Phase-5.6 0.6B-Base + smoke adapter (already trained).
set -e
cd /Users/unoa/dalbitalba-train-data
MLX=/Users/unoa/projects/dalbitalba-train-data/.venv-mlx312/bin/mlx_lm.generate
ADAPTER=runs/local-smoke-2026-05-08-comprehensive/adapters
OUT=runs/local-integrity-2026-05-08/L_persona_matrix.txt
> "$OUT"

# 5 distinct personas (id, name, tone, mood, particles, trait)
declare -a PERSONAS=(
  "1|강남3년차|존댓말|resigned|ㅋㅋ ㅠ ㅠㅠ ㅇㅇ|경력자 체념 톤"
  "2|초보22|반말|nervous|ㄷㄷ ㅠㅠ ㅎㅎ|입문자 떨림"
  "3|텐카언니|반말|상냥|ㅋㅋ ㅎㅎ ㅋㅋㅋ|친절한 멘토"
  "4|쩜오씁쓸|반말|jaded|ㅋㅋ ㅎ|차가운 베테랑"
  "5|보도왕초|존댓말|상남|ㅋㅋㅋ ㅋㅋ ㅂㅂ|호탕 형님"
)

# 3 prompts
declare -a PROMPTS=(
  "다음 한국어 댓글을 cb2 밤문화 커뮤니티 톤으로 한 줄 생성:"
  "이번달 매상 너무 안나오는데 어떡해야 좋을까요"
  "초보입니다 손님 어떻게 응대해야 하나요"
)

for p in "${PERSONAS[@]}"; do
  IFS='|' read -r pid pname ptone pmood ppart ptrait <<< "$p"
  PERSONA_BLOCK="[PERSONA] id=${pid} name=${pname} tone=${ptone} mood=${pmood} particles=${ppart} trait=${ptrait} [/PERSONA]"
  for prompt in "${PROMPTS[@]}"; do
    full="${PERSONA_BLOCK}\n${prompt}"
    echo "===== persona=${pname} prompt=${prompt:0:30}... =====" >> "$OUT"
    echo "PROMPT:" >> "$OUT"
    echo -e "$full" >> "$OUT"
    echo "OUTPUT:" >> "$OUT"
    "$MLX" \
      --model Qwen/Qwen3-0.6B-Base \
      --adapter-path "$ADAPTER" \
      --prompt "$(echo -e "$full")" \
      --max-tokens 80 \
      --temp 0.8 \
      --top-p 0.9 \
      2>/dev/null | tail -n +2 >> "$OUT" || echo "[gen-failed]" >> "$OUT"
    echo "" >> "$OUT"
  done
done
echo "DONE -> $OUT"
