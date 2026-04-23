# Scale-up bench v3 — 20260421

- n = 25 (human=25, ai=0)
- seed = 42
- crawl = /mnt/c/Users/mapdr/Desktop/queenalba-crawler/crawled-data-v2
- ai-dir = (unset — populate later)

## 판정 실행
```bash
node apps/web/scripts/llm-judge.mjs --judge claude-opus-4-7 --samples bench/v3-20260421/samples.jsonl --out bench/v3-20260421/claude.json
node apps/web/scripts/llm-judge.mjs --judge gpt-5           --samples bench/v3-20260421/samples.jsonl --out bench/v3-20260421/gpt.json
node apps/web/scripts/llm-judge.mjs --judge hf-roberta      --samples bench/v3-20260421/samples.jsonl --out bench/v3-20260421/hf.json
node apps/web/scripts/llm-judge.mjs --consensus bench/v3-20260421/claude.json bench/v3-20260421/gpt.json bench/v3-20260421/hf.json
```