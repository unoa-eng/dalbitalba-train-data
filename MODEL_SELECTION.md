# Round-2 Base Model Selection

Verdict: `Qwen/Qwen3-8B-Base` is the paper-grade default. `Qwen/Qwen3-0.6B-Base`
is retained only as a smoke/proxy candidate.

## Decision Criteria

The project target is native-reader indistinguishability for Korean forum posts
and replies, not only domain-keyword mimicry. The base model must therefore
score well on:

- Korean/multilingual prior capability.
- Tokenizer fit for Hangul, compatibility jamo, slang, and domain argot.
- 2k+ thread-conditioned context under QLoRA/DoRA.
- Baseline comparability with prior repo recipes and audits.
- Capacity headroom for persona/register/thread structure after CPT + SFT + ORPO.
- Reproducible monitoring and eval traceability.

## Candidate Matrix

| Candidate | Role | Decision | Rationale |
| --- | --- | --- | --- |
| `Qwen/Qwen3-0.6B-Base` | smoke/proxy | Reject as primary | Cheap, same tokenizer family, but local smoke artifacts fail style gates badly and capacity is not justified for native indistinguishability. |
| `Qwen/Qwen3-4B-Base` | mid-cost ablation | Keep as ablation | Likely useful if 8B cost is constrained, but no repo baseline currently anchors it. |
| `Qwen/Qwen3-8B-Base` | primary | Select | Existing recipes/rulebook already use it, tokenizer matches smaller Qwen3 candidates, context length is sufficient, and capacity is most aligned with the stated quality target. |

## Required Evidence Before Launch

The launch path must record:

- `BASE_MODEL=Qwen/Qwen3-8B-Base` in `recipes/round2-cycle1.env`.
- Qwen3 tokenizer audit over representative Korean/domain terms.
- W&B tracking enabled for every train phase.
- Ablation plan present for base, CPT-only, CPT+DoRA, CPT+DoRA+TC-SFT, and full ORPO.
- Evaluation protocol present with primary endpoint, bootstrap intervals, and blind/native acceptance plan.

If the selected base changes, this file and `scripts/prelaunch_research_check.py`
must be updated in the same commit.
