# Structure Comparison

Date: 2026-04-29
Model: `Qwen/Qwen3-0.6B` via MLX LoRA (`iters=200`, `batch=1`, `num_layers=8`)

## Setup

- Splits are deterministic with seed-pinned 500-row samples per dataset, then `450/25/25` train/valid/test because `mlx_lm lora` requires a test split directory.
- A4 reuses the exact same sampled rows as A1 and only rewrites post rows from `title\nbody` to `제목: title\nbody`.
- Prompts are shared at the semantic level via [prompt_set.jsonl](/Users/unoa/projects/dalbitalba-train-data/runs/ablation/prompt_set.jsonl) and rendered in the closest native format for each dataset family.

## Validation Loss

| Experiment | Final val loss | Test loss | Informal ratio | Domain hits (TC/밀빵/케어/ㅋㅋ/ㅠㅠ) | Notes |
|---|---:|---:|---:|---|---|
| A1-flat-cpt-v2 | 4.531 | 4.379 | 0.85 | 2/0/2/3/11 | flat v2 baseline |
| A2-structured-cpt-v3 | 3.410 | 3.377 | 1.00 | 2/0/2/2/14 | structured closure 16/20 |
| A3-context-stream | 4.332 | 4.477 | 0.85 | 2/0/2/1/13 | mixed context_comment stream |
| A4-flat-cpt-v2-title-prepended | 4.363 | 4.344 | 0.65 | 2/0/2/3/10 | same rows as A1 with explicit title marker |

## Sample Quality

### A1-flat-cpt-v2

- `초이스 너무 안 잡히는 날엔 / 더 나빠요…`
- `텐카 첫 출근인데 / 오늘은 객미러진 수 really 높었습니다.`
- Informal outputs: `85.00%` of samples avoided the coarse formal-style regex.

### A2-structured-cpt-v3

- `<|post|>제목: 초이스 너무 안 잡히는 날엔 / 말인이 갈콜러씨 수 있는거 알려주실래요? ㅜㅜ<|/post|>`
- `<|post|>제목: 텐카 첫 출근인데 / 말인가요? 봤소니 놀ㅓㄴ / 지금일은 도워서 한시간을 보자하고 딜에요만 하겠지 / 인디언 텐은 불가능해요 / 말초에 학입에 10분 더남 ㅠㅠ / 그런데 원래 몇 짓이야요? ??? / ㅅㅂ<|/post|>`
- Structured closure fidelity: `16/20` prompts produced a matching closing tag.
- Informal outputs: `100.00%` of samples avoided the coarse formal-style regex.

### A3-context-stream

- `제목: 초이스 너무 안 잡히는 날엔 / 원글: 간단히 음악달바기ㅓ븼 현청일손님들 / 주니는 토포에 가는거 태어나다.. 토포 토살낤테이블이 진짜하면서도 토살낤테이블이 진짜먼 몸세는데 토포래면 그거까지 왜 토포가 안되어야하지요? 토포래면 그거까지 왜 토포가 안`
- `제목: 텐카 첫 출근인데 / 원글: 간단하게 음악을 비기고 턱이가 뺍업 한게요. / 원글 딱에요? 하겠나요? 진짜 텐빠죠 텐개날테요 / 원글에 ㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅊㅅ오자 / 댓글: 죽마먹이 텐나 멍을 모시고 꽁 잎이 뺍업 한게요ㅠ`
- Informal outputs: `85.00%` of samples avoided the coarse formal-style regex.

### A4-flat-cpt-v2-title-prepended

- `제목: 초이스 너무 안 잡히는 날엔 / 일인공차술이랑 음악이 비일까요?`
- `제목: 텐카 첫 출근인데 / 일인공차술이랑 음악이랑 ㅓ븼 때 쯔ㄱㄱㄱㄱㄱㄱㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ`
- Informal outputs: `65.00%` of samples avoided the coarse formal-style regex.

## Takeaways

- Lowest validation loss in this local 0.6B matrix: `A2-structured-cpt-v3` at `3.410`.
- A1 vs A4 isolates whether an explicit `제목:` marker helps beyond the already-merged v2 title/body text.
- A2 is the only condition where structure-token closure can be measured directly; compare its closure count against its raw readability when choosing whether explicit wrappers are worth the complexity.
- A3 indicates whether the mixed context stream preserves reply tone better without forcing wrapper tokens.

