# Final Eval Readiness

- Date: 2026-04-28
- Repo: `/Users/unoa/projects/dalbitalba-train-data`
- Source crawl: `/Users/unoa/Downloads/crawled-data-v2`
- CPT input used for final GAP check: `cpt_context_stream.jsonl`

## Eval pipeline status

- `scripts/phase6_generate.py` now emits a `kind` field in generated JSONL rows.
- `scripts/phase6_generate.py` derives `kind` from explicit metadata first, then prompt/input hints, then reply-tag shape as a fallback.
- `scripts/phase6_eval.py` now normalizes and uses `kind` metadata explicitly for per-kind stratified metrics when both AI and raw inputs provide it.
- `scripts/refinement_loop.py` already had kind-stratified GAP analysis. It now also accepts `--cpt-jsonl` and respects `TRAIN_CPT_JSONL` / `cpt_context_stream.jsonl` selection for reproducible final checks.

## Verification

- Syntax check: `python3 -m py_compile scripts/phase6_generate.py scripts/phase6_eval.py scripts/refinement_loop.py`
- `phase6_generate` helper smoke test: PASS
- `phase6_eval` kind-breakdown smoke test: PASS

## Final GAP run

Command:

```bash
python3 scripts/refinement_loop.py \
  --source-dir /Users/unoa/Downloads/crawled-data-v2 \
  --cycle 1 \
  --cpt-jsonl cpt_context_stream.jsonl
```

Run output:

- Selected CPT corpus: `/Users/unoa/projects/dalbitalba-train-data/cpt_context_stream.jsonl`
- Run directory: `/Users/unoa/projects/dalbitalba-train-data/runs/refinement-20260428-165117/cycle-1`
- Source texts: `67,129` (`11,280` posts, `55,849` comments)
- Training texts: `134,841`
- Training kind counts:
  - `post`: `54,537`
  - `comment`: `68,290`
- Coverage: `93.9%` of source posts represented in training

## GAP status

- Overall status: `GAPS REMAIN`
- Total gaps: `17`
- Critical: `0`
- Moderate: `17`
- Minor: `0`

Top remaining gaps from the final run:

- `post/업소유형/도파민` is over-represented (`ratio=2.3189`)
- `comment/업소유형/쩜오` is under-represented (`ratio=0.3692`)
- `comment/직업용어/TC` is under-represented (`ratio=0.1566`)
- `comment/금전/빠꾸` is over-represented (`ratio=2.9312`)
- `comment/금전/밀빵` is under-represented (`ratio=0.2198`)
- `comment/avg_sentences_per_text` remains structurally low (`source=4.74`, `train=2.09`)

Detailed machine-readable report:

- `/Users/unoa/projects/dalbitalba-train-data/runs/refinement-20260428-165117/cycle-1/diagnostic.json`
