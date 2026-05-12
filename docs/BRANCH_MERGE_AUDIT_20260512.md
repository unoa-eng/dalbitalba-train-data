# Branch Merge Audit — 2026-05-12

## Main Status

- Local `main` and `origin/main` are synchronized at `9aeeddd`.
- `git rev-list --left-right --count main...origin/main` returned `0 0`.
- PR #8, `merge/main-consolidation`, was merged into `main` on 2026-05-08.
- The remote branch `origin/merge/main-consolidation` has been deleted and was
  pruned locally during this audit.

## PR State

- PR #8: merged. Consolidated PRs #2-#7 into main.
- PR #1: merged. `codex/budget30-macmini-loop`.
- PR #2-#7: closed after consolidation. Their feature branches are not direct
  ancestors of `main`, but their useful tree content was consolidated through
  PR #8 and subsequent hardening commits.

## Residual Branch Classification

- `origin/codex/budget30-macmini-loop`,
  `origin/codex/qwen3-verification-loop`, and
  `origin/codex/train-runpod-fixes`: no commits ahead of current `main`.
- `origin/feat/p0-data-hardening`,
  `origin/feat/p1-thread-aware-data`,
  `origin/feat/p1-train-sft-upgrade`, and
  `origin/feat/round2-runpod-process`: superseded by PR #8 plus later main
  hardening. Direct checkout/diff would delete newer main artifacts.
- `origin/feat/p1-style-classifier`: contains standalone classifier scripts,
  but current main already has the v2 full retrain handoff/results. No raw
  branch cherry-pick was applied because the branch version is older than the
  final v2 handoff.
- `origin/budget30-pre-launch`: contains older Obsidian auxiliary experiments,
  v3-data variants, and many historical run reports. Full cherry-pick is unsafe
  because it would reintroduce stale recipes/run artifacts. The useful
  policy-level Obsidian decision was ported into `docs/OBSIDIAN_SCOPE_POLICY.md`
  and the verifier now checks Obsidian scope mechanically.

## Train/Eval Run Branches

- `origin/train-run-*`: historical RunPod artifact branches. Statuses include
  `preflight_failed`, `cpt_failed`, `data_missing`, `done_cpt_only`, and one
  `done_ok` branch (`train-run-20260430-154836`). These are evidence branches,
  not feature branches to merge.
- `origin/eval-run-*`: historical eval artifact branches with manifests and
  metric-like outputs. They are not mainline process code and should remain
  archival unless a specific metric artifact is cited.

## Obsidian Scope Verification

- `research/obsidian-ref`: 18 markdown files.
- `research/obsidian-export`: 510 markdown files.
- `runs/round2-obsidian-synthesis/persona-30-extracted.json`: 30 accepted
  personas.
- Inline persona rows missing `tone/mood/trait` were repaired during this audit.
- `scripts/local_verification_loop.py` now fails if Obsidian reference/export
  scope or persona-30 coverage is materially missing.

## Cherry-Pick Outcome

Applied selectively:

- Ported the Obsidian auxiliary policy from the old `budget30-pre-launch`
  branch into `docs/OBSIDIAN_SCOPE_POLICY.md`.
- Added Obsidian scope validation to `scripts/local_verification_loop.py`.
- Repaired `runs/round2-obsidian-synthesis/persona-30-extracted.json` metadata
  so persona-conditioned SFT rows do not receive empty tone/mood fields.

Not applied:

- Stale run artifacts and old recipes from `budget30-pre-launch`.
- Older classifier v1 branch scripts, because current main contains a newer v2
  full-retrain record.
- Historical train/eval artifact branches.
