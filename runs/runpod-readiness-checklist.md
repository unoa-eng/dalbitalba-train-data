# RunPod Readiness Checklist

Date: 2026-04-27

Validated locally with:

- `bash -n chain_train.sh`
- `python3 -m py_compile scripts/launch_train_pod.py train_cpt.py train_sft.py`
- `RUNPOD_API_KEY=dummy HF_TOKEN=dummy HF_USERNAME=unoa GITHUB_TOKEN=dummy python3 scripts/launch_train_pod.py --dry-run`

Official RunPod references checked:

- Pod create API: https://docs.runpod.io/api-reference/pods/POST/pods
- Pod lifecycle / stop vs terminate: https://docs.runpod.io/pods/manage-pods

## Checklist

| Item | Status | Evidence |
| --- | --- | --- |
| `launch_train_pod.py` is compatible with the current Pod create API shape | `PASS` | The script still uses documented request fields such as `cloudType`, `computeType`, `gpuTypeIds`, `dockerStartCmd`, `env`, `volumeInGb`, and `volumeMountPath`. |
| Launcher auto-selects `cpt_corpus.v3.jsonl` when present | `PASS` | `resolve_workspace_data_path()` now prefers `cpt_corpus.v3.jsonl` over `cpt_corpus.v2.jsonl`. Local check resolved `/workspace/data/cpt_corpus.v3.jsonl`. |
| Dataset env aliases are consistent across launcher and wrapper | `PASS` | The launcher now injects both explicit `TRAIN_*` names and legacy aliases (`INPUT_JSONL`, `SFT_PAIR_JSONL`, `CPT_VAL_JSONL`). `chain_train.sh` syncs those aliases back to a single resolved set of paths. |
| `chain_train.sh` no longer hardcodes `v2` as the only CPT default | `PASS` | The wrapper now prefers `/workspace/data/cpt_corpus.v3.jsonl` when it exists and logs the resolved CPT/SFT/val files at startup. |
| CPT-only runs do not fail just because SFT pairs are absent | `PASS` | The wrapper now requires `TRAIN_SFT_PAIR_JSONL` only when `SKIP_SFT` is not set. CPT-only `budget30` launches no longer have a false dependency on reply pairs. |
| RunPod CLI stop command matches current docs | `PASS` | `chain_train.sh` now tries `runpodctl pod stop` first, with the older `runpodctl stop pod` kept as fallback. Current docs show `runpodctl pod stop`. |
| Current stop behavior matches the current pod storage mode | `PASS` | The launcher uses `volumeInGb` and does not attach `networkVolumeId`, so the documented “pods with network volumes cannot be stopped” restriction does not apply to the current spec. |
| Price ceiling is enforced by the launcher itself | `FAIL` | The launcher still relies on static local estimates and GPU type pinning. It does not enforce a runtime `costPerHr` ceiling from the Pod create response, and it does not pass a documented max-price control into the REST create payload. |
| The pre-launch verifier is fully `v3`-aware | `FAIL` | `scripts/local_verification_loop.py` still reads `cpt_corpus.v2.jsonl`, `sft_pairs.v2.jsonl`, and `val_set.v2.jsonl` by default. A `budget30` PASS from that script can still describe the old corpus rather than the `v3` corpus the launcher now prefers. |

## Bottom Line

- The RunPod launch path itself is now ready for `v3`.
- Two remaining non-launch gaps are still open:
  - the verifier/reporting path is still `v2`-bound;
  - the launcher has no hard guard against live hourly price drift.

If we launch today, the operationally safe stance is:

1. Treat the launch wrapper as `v3`-ready.
2. Treat the verifier as advisory until it is taught `v3`.
3. Keep `budget30` on a conservative CPT cap (`1 epoch`, or an explicit `CPT_MAX_STEPS` cap) rather than assuming the old `v2` budget still applies.
