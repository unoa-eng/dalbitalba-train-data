#!/usr/bin/env python3
"""
recipe_mutator.py — apply docs/RECIPE_MUTATION_RULEBOOK.md rules to mutate
the training recipe based on the latest eval metrics. Claude-authored
judgment layer; deterministic when invoked.

Usage:
  python3 scripts/recipe_mutator.py --metrics runs/eval-run-XYZ/metrics.json

Reads/writes .state/loop_state.json. Emits `export KEY=VALUE` lines on stdout
that the autonomous_loop.sh supervisor can `eval` to inject into the next
launch_train_pod.py invocation.

Exit codes:
  0  — normal mutation, loop should continue
  1  — input error
  2  — stop signal (cycle cap, budget cap, convergence, rollback)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
STATE = REPO / ".state" / "loop_state.json"
STOP_FILE = REPO / ".state" / "STOP"
BUDGET_FILE = REPO / ".state" / "budget_spent.json"

CYCLE_CAP = 5
BUDGET_CAP_USD = 25.0
CONVERGENCE_BIGRAM_JSD = 0.04
STAGNATION_DELTA = 0.01
# R8 — if the previous cycle's recipe is identical and bigram_jsd barely
# moved despite the recipe-mutator emitting *something*, the bottleneck is
# not in the recipe knobs — it is in the data. Trigger a regen cycle.
DATA_REGEN_DELTA = 0.005

DEFAULT_RECIPE = {
    "BASE_MODEL": "Qwen/Qwen3-8B-Base",
    "CPT_NUM_EPOCHS": 1,
    "SFT_NUM_EPOCHS": 2,
    "LORA_R": 64,
    "LORA_ALPHA": 64,
    "CPT_USE_DORA": 0,
    "CPT_LR": "2e-4",
    "SFT_LR": "5e-5",
}


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {
        "cycle": 0,
        "recipe": dict(DEFAULT_RECIPE),
        "history": [],
        "budget_spent_usd": 0.2,
        "stop_reason": None,
        "data_regen": {},
    }


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_budget() -> float:
    if BUDGET_FILE.exists():
        return float(json.loads(BUDGET_FILE.read_text(encoding="utf-8")).get("usd", 0.0))
    return 0.0


def write_stop(reason: str) -> None:
    STOP_FILE.write_text(
        json.dumps(
            {"stopped_at": datetime.now(timezone.utc).isoformat(), "reason": reason},
            indent=2,
        ),
        encoding="utf-8",
    )


def emit_exports(changes: dict) -> None:
    for k, v in changes.items():
        print(f"export {k}={v}")


def check_stagnation(history: list[dict]) -> bool:
    """If last 2 cycles show bigram_jsd movement < STAGNATION_DELTA, stagnant."""
    if len(history) < 2:
        return False
    recent = history[-2:]
    jsds = [
        h.get("metrics_snapshot", {}).get("bigram_jsd")
        for h in recent
    ]
    if any(j is None for j in jsds):
        return False
    return abs(jsds[-1] - jsds[0]) < STAGNATION_DELTA


def _data_regen_stagnant(history: list[dict]) -> bool:
    """R8 trigger: identical recipe across last 2 cycles AND bigram_jsd moved
    by less than DATA_REGEN_DELTA. Recipe-knob mutation is exhausted; the
    actual bottleneck is the corpus (dedup, ad spam, jamo consistency).
    """
    if len(history) < 2:
        return False
    a, b = history[-2], history[-1]
    a_recipe = a.get("recipe_changes", {})
    b_recipe = b.get("recipe_changes", {})
    if a_recipe or b_recipe:
        # recipe still being mutated; not a data-bound stagnation
        return False
    a_jsd = a.get("metrics_snapshot", {}).get("bigram_jsd")
    b_jsd = b.get("metrics_snapshot", {}).get("bigram_jsd")
    if a_jsd is None or b_jsd is None:
        return False
    return abs(b_jsd - a_jsd) < DATA_REGEN_DELTA


def apply_rules(
    metrics: dict, recipe: dict, data_regen: dict, history: list[dict] | None = None
) -> tuple[dict, dict, str, str]:
    """Return (recipe_changes, data_regen_changes, rule_id, rationale)."""
    m = metrics
    history = history or []

    # R8 — data-bound stagnation: recipe knobs are exhausted but bigram_jsd
    # still won't move. Emit a data-regen request and pause training so the
    # autonomous loop can rebuild cpt_corpus.v3 / sft_pairs.v3 with global
    # MinHash dedup + extended ad regex + entropy gate before relaunching.
    # Checked before R7 because data fixes are cheaper than architecture
    # switches, and a re-run on cleaner data may resolve the stagnation
    # without an architecture change.
    if _data_regen_stagnant(history):
        return (
            {},
            {
                "REGEN_DATA": 1,
                "REGEN_REASON": "stagnation",
                "REGEN_ENABLE_MINHASH": 1,
                "REGEN_ENABLE_ENTROPY_FILTER": 1,
                "REGEN_MIN_ENTROPY": "2.8",
            },
            "R8",
            (
                "bigram_jsd stagnant across two cycles with no recipe changes; "
                "regenerate data with MinHash dedup + entropy gate before "
                "next training launch"
            ),
        )

    # R7 — architecture switch: when LoRA-CPT keeps underfitting at r=128 +
    # DoRA already enabled, escalate to a full fp16 fine-tune (or r=256
    # rsLoRA on capable hardware). LoRA-CPT is documented to be ~16× data
    # inefficient vs full fine-tune (arXiv:2405.09673); persisting at r≤128
    # against a domain corpus is wrong-axis mutation.
    if (
        m.get("bigram_jsd", 0) > 0.15
        and recipe.get("LORA_R", 64) >= 128
        and recipe.get("CPT_USE_DORA", 0) == 1
        and recipe.get("CPT_NUM_EPOCHS", 1) >= 2
    ):
        return (
            {
                # Surfaces an env signal that chain_train.sh / launch_train_pod.py
                # can read to switch to a full-FT recipe (or r=256 rsLoRA).
                "CPT_FULL_FT": 1,
                "LORA_R": 256,
                "LORA_ALPHA": 256,
            },
            {},
            "R7",
            (
                "LoRA-CPT cap reached (r=128 + DoRA + 2 epochs) with "
                "bigram_jsd > 0.15; escalate to fp16 full fine-tune or "
                "r=256 rsLoRA per arXiv:2405.09673"
            ),
        )

    # R1 — high bigram JSD
    if m.get("bigram_jsd", 0) > 0.15:
        if recipe.get("CPT_NUM_EPOCHS", 1) == 1:
            return (
                {"CPT_NUM_EPOCHS": 2},
                {},
                "R1",
                "bigram_jsd > 0.15, first escalation: extend CPT to 2 epochs",
            )
        if recipe.get("LORA_R", 64) < 128:
            return (
                {"LORA_R": 128, "LORA_ALPHA": 128},
                {},
                "R1b",
                "bigram_jsd still > 0.15 at 2 epochs, bump LoRA r to 128",
            )
        if recipe.get("CPT_USE_DORA", 0) == 0:
            return (
                {"CPT_USE_DORA": 1},
                {},
                "R1c",
                "bigram_jsd persistent, enable DoRA",
            )
        # R1 levers exhausted — fall through to R7 on next cycle (above) by
        # returning a sentinel that autonomous_loop can interpret. We no
        # longer emit R1_EXHAUSTED as a hard stop because R7 supersedes it.
        return (
            {},
            {},
            "R1_EXHAUSTED",
            "bigram_jsd not responsive to LoRA-only mutations; R7 next cycle",
        )
    # R2 — length distribution mismatch
    if m.get("length_kl", 0) > 0.10:
        current = data_regen.get("OVERSAMPLE_LG_XXL", 2)
        if current < 3:
            return (
                {},
                {"OVERSAMPLE_LG_XXL": 3},
                "R2",
                "length_kl > 0.10, bump lg/xl/xxl oversample to 3x",
            )
        if current < 4:
            return (
                {"SEQ_LEN": 2048},
                {"OVERSAMPLE_LG_XXL": 4},
                "R2b",
                "length_kl still > 0.10, oversample 4x + seq_len 2048",
            )
    # R3 — digit/english density
    if m.get("digit_density_delta", 0) > 0.03 or m.get("english_density_delta", 0) > 0.02:
        current = data_regen.get("OVERSAMPLE_DIGIT_ENG", 2)
        if current < 4:
            return (
                {},
                {"OVERSAMPLE_DIGIT_ENG": 4},
                "R3",
                "digit/english delta above threshold, oversample 4x",
            )
    # R4 — MAUVE semantic gap
    mauve = m.get("mauve_score")
    if mauve is not None and mauve < 0.80 and recipe.get("CPT_USE_DORA", 0) == 0:
        return (
            {"CPT_USE_DORA": 1},
            {},
            "R4",
            "mauve < 0.80, enable DoRA for semantic match",
        )
    if (
        mauve is not None
        and mauve < 0.80
        and recipe.get("SFT_NUM_EPOCHS", 2) == 2
    ):
        return (
            {"SFT_NUM_EPOCHS": 3},
            {},
            "R4b",
            "mauve < 0.80 with DoRA on, extend SFT to 3 epochs",
        )
    # R5 — boundary case: many metrics close to threshold
    thresholds = {
        "bigram_jsd": 0.15,
        "length_kl": 0.10,
        "digit_density_delta": 0.03,
        "english_density_delta": 0.02,
    }
    close_count = sum(
        1
        for k, t in thresholds.items()
        if 0.9 * t < m.get(k, 0) <= t
    )
    if close_count >= 3 and recipe.get("CPT_NUM_EPOCHS", 1) == 1:
        return (
            {"CPT_NUM_EPOCHS": 2},
            {},
            "R5",
            "many metrics close to threshold, cheap global improvement",
        )
    # R6 — default parameter expansion
    if recipe.get("LORA_R", 64) < 128:
        return (
            {"LORA_R": 128, "LORA_ALPHA": 128},
            {},
            "R6",
            "no dominant metric; expand LoRA capacity",
        )
    return (
        {},
        {},
        "NO_RULE",
        "no applicable mutation rule — escalate",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    try:
        metrics_doc = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    except Exception as exc:
        sys.stderr.write(f"[error] cannot read metrics {args.metrics}: {exc}\n")
        return 1

    metrics = metrics_doc.get("metrics", {})
    gate = metrics_doc.get("gate", {})
    verdict = gate.get("verdict")

    state = load_state()
    state["budget_spent_usd"] = load_budget()

    # R0 — stop checks
    if state["cycle"] >= CYCLE_CAP:
        reason = f"cycle cap reached ({state['cycle']} >= {CYCLE_CAP})"
        state["stop_reason"] = reason
        save_state(state)
        write_stop(reason)
        sys.stderr.write(f"[stop] {reason}\n")
        print("export LOOP_STOP=1")
        return 2
    if state["budget_spent_usd"] >= BUDGET_CAP_USD:
        reason = f"budget cap reached (${state['budget_spent_usd']:.2f})"
        state["stop_reason"] = reason
        save_state(state)
        write_stop(reason)
        sys.stderr.write(f"[stop] {reason}\n")
        print("export LOOP_STOP=1")
        return 2
    if check_stagnation(state["history"]):
        reason = "bigram_jsd stagnant across last 2 cycles"
        state["stop_reason"] = reason
        save_state(state)
        write_stop(reason)
        sys.stderr.write(f"[stop] {reason}\n")
        print("export LOOP_STOP=1")
        return 2

    # PASS path
    if verdict == "PASS":
        if metrics.get("bigram_jsd", 1) < CONVERGENCE_BIGRAM_JSD:
            reason = "CONVERGED — bigram_jsd below raw-vs-raw×2 ceiling"
            state["stop_reason"] = reason
            save_state(state)
            write_stop(reason)
        print("export LOOP_PASS=1")
        sys.stderr.write("[pass] gate satisfied, loop should create PR\n")
        return 0

    # FAIL — apply rules
    recipe_changes, data_changes, rule_id, rationale = apply_rules(
        metrics,
        state["recipe"],
        state.get("data_regen", {}),
        state.get("history", []),
    )

    if rule_id == "NO_RULE":
        reason = f"no mutation available ({rule_id}); escalate"
        state["stop_reason"] = reason
        save_state(state)
        write_stop(reason)
        sys.stderr.write(f"[stop] {reason}\n")
        print("export LOOP_STOP=1")
        return 2

    if rule_id == "R1_EXHAUSTED":
        # Soft stop: surface the signal so the supervisor's next cycle can
        # see R7 fire, but do not write a hard STOP file. The next cycle's
        # apply_rules() will return R7 because LORA_R≥128 + DoRA on +
        # epochs≥2 is now true.
        state["history"].append(
            {
                "cycle": state["cycle"],
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "rule_id": rule_id,
                "rationale": rationale,
                "recipe_changes": {},
                "data_regen_changes": {},
                "metrics_snapshot": metrics,
            }
        )
        save_state(state)
        sys.stderr.write(
            "[mutate] R1_EXHAUSTED: LoRA levers exhausted; next cycle will "
            "evaluate R7 architecture switch.\n"
        )
        # Emit no exports; supervisor relaunches with current recipe so R7
        # can be cleanly triggered after the stable measurement.
        return 0

    # Apply changes
    state["recipe"].update(recipe_changes)
    state.setdefault("data_regen", {}).update(data_changes)
    state["cycle"] += 1
    state["history"].append(
        {
            "cycle": state["cycle"],
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "rule_id": rule_id,
            "rationale": rationale,
            "recipe_changes": recipe_changes,
            "data_regen_changes": data_changes,
            "metrics_snapshot": metrics,
        }
    )
    save_state(state)

    sys.stderr.write(f"[mutate] {rule_id}: {rationale}\n")
    emit_exports(recipe_changes)
    # data_regen changes go to env too so autonomous_loop can trigger regen
    for k, v in data_changes.items():
        print(f"export {k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
