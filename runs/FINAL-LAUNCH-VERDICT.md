# Final Launch Verdict

Date: 2026-04-29
Branch: `budget30-pre-launch`
Verdict: `NO-GO`

## Required Checks

| Check | Result | Evidence |
| --- | --- | --- |
| Recipe loaded | `PASS` | `source recipes/budget30_v2.env` succeeded in all launch checks |
| Train env check | `PASS` | `python3 scripts/check_env.py --target train` reported `[result] environment is ready` |
| Launch dry-run | `PASS` | `python3 scripts/launch_train_pod.py --dry-run` rendered a valid RunPod payload with the `budget30-pre-launch` clone command |
| Local verification loop | `PASS` | `python3 scripts/local_verification_loop.py --profile budget30` returned `{"verdict":"PASS"}` and wrote `runs/local-verification-20260429-051233/report.md` |
| Verification report | `PASS` | Report shows `Severe: 0`, `Warnings: 0`, CPT/SFT/val/CAI datasets clean, total train estimate `60.12h` / `$47.49` |
| CPT learning rate | `PASS` | `recipes/budget30_v2.env` now contains `CPT_LR=5e-5` |

## Launch Blockers

1. Actual pod launch clones GitHub, not this local working tree.
   `scripts/launch_train_pod.py` builds `git clone --branch budget30-pre-launch --single-branch "https://x-access-token:${GITHUB_TOKEN}@github.com/unoa-eng/dalbitalba-train-data.git" /workspace/repo`
2. The verified local branch is ahead of remote by 4 commits.
   `git rev-list --left-right --count origin/budget30-pre-launch...HEAD` returned `0 4`
3. The recipe change that sets `CPT_LR=5e-5` is still uncommitted locally.
   `git diff -- recipes/budget30_v2.env` shows `CPT_LR` changed from `1e-4` to `5e-5`
4. The working tree still contains additional uncommitted launch-related artifacts.
   `git status -sb` shows local modifications including `.gitignore`, `recipes/budget30_v2.env`, and untracked `runs/ablation/` content

## Decision

`NO-GO` for the real RunPod launch from the current repository state.

The technical gates passed locally, but the launch target is still the remote `budget30-pre-launch` branch. Until the required v3 artifacts and recipe changes are committed and pushed, the pod would launch stale code/data instead of the verified local state.

## Flip To GO When

1. Worker-1's v3 artifact commit lands.
2. The branch is pushed so GitHub contains the same verified recipe and artifacts.
3. `git rev-list --left-right --count origin/budget30-pre-launch...HEAD` returns `0 0` or otherwise confirms the remote launch ref is current.
