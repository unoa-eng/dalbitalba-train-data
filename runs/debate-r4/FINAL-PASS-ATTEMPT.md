# Round 4 Final Pass Attempt

- Timestamp: `2026-04-29T06:19Z`
- Verdict: `FAIL`
- Confidence: `high`

## What Passed

- `python3 -m py_compile scripts/launch_eval_pod.py scripts/launch_train_pod.py scripts/merge_cpt_to_fp16.py scripts/phase6_eval.py scripts/phase6_generate.py train_cpt.py` → pass
- `set -a && source recipes/budget30_v2.env && set +a && python3 scripts/local_verification_loop.py --profile budget30` → pass
  - report: `runs/local-verification-20260429-061601/report.md`
- `set -a && source recipes/budget30_v2.env && set +a && python3 scripts/launch_train_pod.py --dry-run` → pass
- `set -a && source recipes/budget30_v2.env && export CPT_MERGED_REPO=UNOA/test && set +a && python3 scripts/launch_eval_pod.py --dry-run` → pass
- `bash -n scripts/run_eval.sh chain_train.sh scripts/autonomous_loop.sh` → pass
- `git diff --check` → pass

## Critical Remaining Issue

`scripts/phase6_eval.py` added `korean_retention_ppl` as a required gate metric, but the gate is fail-open when that metric is unavailable.

Evidence:

- `maybe_korean_retention_ppl()` explicitly returns `None` with `"status": "skipped"` when:
  - the replay corpus is missing or empty
  - no adapted model source is configured
  - any model/tokenizer/perplexity exception occurs
- Source: `scripts/phase6_eval.py:475-506`
- The gate definition includes `korean_retention_ppl` at `<= 1.50`
- Source: `scripts/phase6_eval.py:521-529`
- But `evaluate_gate()` silently skips any metric whose value is `None`
- Source: `scripts/phase6_eval.py:533-543`

Minimal reproducer:

```python
from scripts.phase6_eval import evaluate_gate
metrics = {
    "bigram_jsd": 0.01,
    "length_kl": 0.01,
    "digit_density_delta": 0.0,
    "english_density_delta": 0.0,
    "domain_keyword_alignment": 0.9,
    "tone_distribution_match": 0.0,
    "korean_retention_ppl": None,
    "mauve_score": None,
}
print(evaluate_gate(metrics))
```

Observed output:

```python
('PASS', [])
```

Why this is critical:

- The new design relies on the retention check to catch catastrophic forgetting after the replay / packing / tokenizer / warmup changes.
- In the actual pod path, any missing replay corpus, HF auth/load problem, or runtime failure in the retention calculation can still produce an overall phase-6 `PASS`.
- That means the evaluation contract is stricter on paper than in execution.

## 0.6B MLX Proxy Caveat

Using the 0.6B MLX proxy is reasonable for:

- format fidelity
- structural-token direction checks
- quick ranking of obviously bad vs obviously better recipe variants

It is not strong evidence for absolute 8B outcome prediction.

Why:

- Brown et al. (GPT-3) used different optimal learning rates across model sizes, from `6e-4` at `125M` down to `6e-5` at `175B`, showing LR optima shift materially with scale.
  - https://arxiv.org/abs/2005.14165
- Chen et al. show LoRA learning-rate behavior depends on both adapter rank and model width, so rank/LR conclusions are not universally scale-invariant.
  - https://arxiv.org/abs/2602.06204
- Koh et al. show small proxy models can be useful, but specifically note that some behaviors emerge reliably only above `7B`, so proxy transfer needs caution.
  - https://arxiv.org/abs/2509.21013
- The UK AISI evaluation paper also warns that some capabilities only emerge at greater scale, which limits confidence in small-model-only evaluation.
  - https://arxiv.org/abs/2305.15324

Operational conclusion:

- The 0.6B sweep is valid as format/direction evidence.
- It is not sufficient to justify the final 8B LR/rank/replay decision without target-scale validation.

## Final Decision

Overall design verdict for this pass attempt: `FAIL`.

Reason:

- The newly introduced retention gate is not actually enforced under skipped/error conditions, so the current evaluation path can silently certify a bad model.
