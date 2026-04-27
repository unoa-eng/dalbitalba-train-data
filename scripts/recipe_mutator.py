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


def apply_rules(
    metrics: dict, recipe: dict, data_regen: dict
) -> tuple[dict, dict, str, str]:
    """Return (recipe_changes, data_regen_changes, rule_id, rationale)."""
    m = metrics
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
        # All cheap levers exhausted — escalate stop
        return (
            {},
            {},
            "R1_EXHAUSTED",
            "bigram_jsd not responsive to any cheap lever; escalate",
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
        metrics, state["recipe"], state.get("data_regen", {})
    )

    if rule_id in ("R1_EXHAUSTED", "NO_RULE"):
        reason = f"no mutation available ({rule_id}); escalate"
        state["stop_reason"] = reason
        save_state(state)
        write_stop(reason)
        sys.stderr.write(f"[stop] {reason}\n")
        print("export LOOP_STOP=1")
        return 2

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
