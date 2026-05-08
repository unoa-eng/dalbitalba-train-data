# Round-2 Research Protocol

Verdict: RunPod launch is allowed only as a research run when this protocol and
the prelaunch checker pass.

## Primary Endpoint

Primary endpoint: blinded native-Korean pairwise indistinguishability on held-out
forum-style prompts.

Acceptance target: generated samples should be identified as AI no more than
40% of the time with a Wilson 95% confidence upper bound below 50%. Ties and
judge disagreements are reported separately.

## Required Comparisons

Every final report must include the same prompt set and seeds for:

- `raw-vs-raw` ceiling.
- selected base model prompt-only.
- prior round-1 best adapter when available.
- round2 CPT-only.
- round2 CPT+DoRA.
- round2 CPT+DoRA+TC-SFT.
- round2 full CPT+DoRA+TC-SFT+ORPO.

## Metrics

Engineering gates:

- `phase6_eval.py` base metrics.
- `phase6_eval_v2.py` punctuation, choseong, reply-depth, and persona metadata metrics.
- hard-fail filters for AI-disclosure, formal-template drift, PII/contact leakage,
  promo/operator templates, and unsafe rows.

Research gates:

- stratified randomized eval sampling by `kind`, length bucket, reply depth, and
  domain keyword density.
- at least 3 generation seeds.
- bootstrap confidence intervals for deterministic metrics.
- blind/native judging for the primary endpoint.

## Monitoring

W&B is mandatory for paper-grade runs. Each phase must log under:

- project: `dalbitalba-round2`
- group: `round2-qwen3-8b-paper`
- phase-specific run names: `phase1-cpt-broad`, `phase2-cpt-dora`,
  `phase3-tc-sft`, `phase4-orpo`.

Each persisted run artifact must include git SHA, dirty flag, recipe snapshot,
dataset SHA256s, row counts, base model, seeds, thresholds, package versions,
and final eval JSON.
