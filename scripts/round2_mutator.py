#!/usr/bin/env python3
"""round2_mutator.py — Round-2 R7-R11 mutator extension.

Stacks on top of `recipe_mutator.py` (R1-R6). Adds rules that respond to the
new phase6_eval_v2 metrics:

  R7  domain_keyword_alignment < 0.40   ->  argot loss weight 1.5x -> 2.0x
  R8  reply_depth_kl > 0.20             ->  SEQ_LEN 2048 -> 3072 + sample more chains
  R9  persona_consistency < 0.75        ->  2x sample weight on persona-tagged rows
  R10 punct_ratio_match_max > 0.20      ->  drop rows with promo_likeness=전형광고
  R11 choseong_marker_match_max > 0.25  ->  regenerate SFT outputs forcing argot anchors

Reuses `recipe_mutator.{load_state,save_state,emit_exports,write_stop}` for
state IO. Returns dict of mutations to merge with R1-R6 output.

CLI:
  python3 scripts/round2_mutator.py --metrics runs/eval-run-XYZ/metrics-v2.json

Exit codes:
  0  normal mutation (or no-op)
  1  input error
  2  stop signal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import recipe_mutator as rm  # type: ignore  # noqa: E402


R7_THRESHOLD_DKA = 0.40
R8_THRESHOLD_RDKL = 0.20
R9_THRESHOLD_PCONS = 0.75
R10_THRESHOLD_PUNCT = 0.20
R11_THRESHOLD_CHOSEONG = 0.25

DEFAULT_RECIPE_V2 = {
    **rm.DEFAULT_RECIPE,
    "SFT_LOSS_WEIGHT_ARGOT": 1.5,
    "SEQ_LEN": 2048,
    "SFT_PERSONA_BOOST": 1.0,
    "DROP_FORMAL_PROMO": 0,
    "REGEN_SFT_FORCE_ARGOT": 0,
}


def apply_round2_rules(metrics_v2: dict, recipe: dict) -> tuple[dict, str, str]:
    """Return (recipe_changes, rule_id, rationale)."""
    m = metrics_v2 or {}
    base = m.get("base", {}).get("metrics", {})
    v2 = m.get("v2", {}).get("metrics", {})

    # R7
    dka = base.get("domain_keyword_alignment", 1.0)
    if dka < R7_THRESHOLD_DKA:
        cur = float(recipe.get("SFT_LOSS_WEIGHT_ARGOT", 1.5))
        if cur < 2.0:
            return ({"SFT_LOSS_WEIGHT_ARGOT": 2.0}, "R7",
                    f"DKA={dka:.3f} < {R7_THRESHOLD_DKA}, escalate argot loss weight 1.5x -> 2.0x")

    # R8
    rdkl = v2.get("reply_depth_kl", 0.0)
    if rdkl > R8_THRESHOLD_RDKL:
        cur = int(recipe.get("SEQ_LEN", 2048))
        if cur < 3072:
            return ({"SEQ_LEN": 3072}, "R8",
                    f"reply_depth_kl={rdkl:.3f} > {R8_THRESHOLD_RDKL}, extend SEQ_LEN 2048 -> 3072")

    # R9
    pcons = v2.get("persona_consistency", 1.0)
    if pcons < R9_THRESHOLD_PCONS:
        cur = float(recipe.get("SFT_PERSONA_BOOST", 1.0))
        if cur < 2.0:
            return ({"SFT_PERSONA_BOOST": 2.0}, "R9",
                    f"persona_consistency={pcons:.3f} < {R9_THRESHOLD_PCONS}, "
                    f"2x sample weight on persona-tagged rows")

    # R10
    punct = v2.get("punct_ratio_match_max", 0.0)
    if punct > R10_THRESHOLD_PUNCT:
        if int(recipe.get("DROP_FORMAL_PROMO", 0)) == 0:
            return ({"DROP_FORMAL_PROMO": 1}, "R10",
                    f"punct_ratio_match_max={punct:.3f} > {R10_THRESHOLD_PUNCT}, "
                    f"drop rows with promo_likeness=전형광고")

    # R11
    cs = v2.get("choseong_marker_match_max", 0.0)
    if cs > R11_THRESHOLD_CHOSEONG:
        if int(recipe.get("REGEN_SFT_FORCE_ARGOT", 0)) == 0:
            return ({"REGEN_SFT_FORCE_ARGOT": 1}, "R11",
                    f"choseong_marker_match_max={cs:.3f} > {R11_THRESHOLD_CHOSEONG}, "
                    f"regenerate SFT outputs forcing argot anchors")

    return ({}, "R7-R11_NOOP", "all v2 metrics within round-2 thresholds")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=Path,
                    help="phase6_eval_v2.py JSON output")
    ap.add_argument("--state", type=Path, default=None,
                    help="override state path (default: rm.STATE)")
    args = ap.parse_args()

    if not args.metrics.exists():
        print(f"metrics path missing: {args.metrics}", file=sys.stderr)
        return 1
    try:
        metrics_v2 = json.loads(args.metrics.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"metrics not valid JSON: {exc}", file=sys.stderr)
        return 1

    if args.state:
        rm.STATE = args.state  # type: ignore[attr-defined]
    state = rm.load_state()
    recipe = {**DEFAULT_RECIPE_V2, **(state.get("recipe") or {})}
    changes, rule_id, rationale = apply_round2_rules(metrics_v2, recipe)
    if changes:
        recipe.update(changes)
        state["recipe"] = recipe
        history = state.setdefault("history", [])
        history.append({"rule": rule_id, "rationale": rationale,
                        "round2_changes": changes})
        rm.save_state(state)
        rm.emit_exports(changes)
    else:
        print(f"# {rule_id}: {rationale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
