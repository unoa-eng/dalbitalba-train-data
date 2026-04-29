#!/usr/bin/env python3
"""Unit tests for round2_mutator R7-R11.

Run:
  python3 scripts/test_round2_mutator.py
  exit 0 = pass
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round2_mutator as r2m  # type: ignore


def case_R7():
    metrics_v2 = {"base": {"metrics": {"domain_keyword_alignment": 0.30}}, "v2": {"metrics": {}}}
    recipe = {**r2m.DEFAULT_RECIPE_V2}
    changes, rule, _ = r2m.apply_round2_rules(metrics_v2, recipe)
    assert rule == "R7", f"expected R7 rule, got {rule}"
    assert changes == {"SFT_LOSS_WEIGHT_ARGOT": 2.0}, f"unexpected changes: {changes}"
    print("R7 ok")


def case_R7_already_max():
    metrics_v2 = {"base": {"metrics": {"domain_keyword_alignment": 0.30}}, "v2": {"metrics": {}}}
    recipe = {**r2m.DEFAULT_RECIPE_V2, "SFT_LOSS_WEIGHT_ARGOT": 2.0}
    changes, rule, _ = r2m.apply_round2_rules(metrics_v2, recipe)
    # R7 cap reached -> falls through to next rule which is no-op since other metrics missing
    assert rule != "R7" or changes == {}, "R7 should not re-fire when at cap"
    print("R7-cap ok")


def case_R8():
    metrics_v2 = {"base": {"metrics": {"domain_keyword_alignment": 0.6}},
                  "v2": {"metrics": {"reply_depth_kl": 0.30}}}
    recipe = {**r2m.DEFAULT_RECIPE_V2}
    changes, rule, _ = r2m.apply_round2_rules(metrics_v2, recipe)
    assert rule == "R8", f"expected R8, got {rule}"
    assert changes == {"SEQ_LEN": 3072}
    print("R8 ok")


def case_R9():
    metrics_v2 = {"base": {"metrics": {"domain_keyword_alignment": 0.6}},
                  "v2": {"metrics": {"reply_depth_kl": 0.10, "persona_consistency": 0.50}}}
    recipe = {**r2m.DEFAULT_RECIPE_V2}
    changes, rule, _ = r2m.apply_round2_rules(metrics_v2, recipe)
    assert rule == "R9", f"expected R9, got {rule}"
    assert changes == {"SFT_PERSONA_BOOST": 2.0}
    print("R9 ok")


def case_R10():
    metrics_v2 = {"base": {"metrics": {"domain_keyword_alignment": 0.6}},
                  "v2": {"metrics": {"reply_depth_kl": 0.10, "persona_consistency": 0.9,
                                     "punct_ratio_match_max": 0.30}}}
    recipe = {**r2m.DEFAULT_RECIPE_V2}
    changes, rule, _ = r2m.apply_round2_rules(metrics_v2, recipe)
    assert rule == "R10", f"expected R10, got {rule}"
    assert changes == {"DROP_FORMAL_PROMO": 1}
    print("R10 ok")


def case_R11():
    metrics_v2 = {"base": {"metrics": {"domain_keyword_alignment": 0.6}},
                  "v2": {"metrics": {"reply_depth_kl": 0.10, "persona_consistency": 0.9,
                                     "punct_ratio_match_max": 0.10,
                                     "choseong_marker_match_max": 0.30}}}
    recipe = {**r2m.DEFAULT_RECIPE_V2}
    changes, rule, _ = r2m.apply_round2_rules(metrics_v2, recipe)
    assert rule == "R11", f"expected R11, got {rule}"
    assert changes == {"REGEN_SFT_FORCE_ARGOT": 1}
    print("R11 ok")


def case_NOOP():
    # All metrics within thresholds
    metrics_v2 = {"base": {"metrics": {"domain_keyword_alignment": 0.6}},
                  "v2": {"metrics": {"reply_depth_kl": 0.10, "persona_consistency": 0.9,
                                     "punct_ratio_match_max": 0.10,
                                     "choseong_marker_match_max": 0.10}}}
    recipe = {**r2m.DEFAULT_RECIPE_V2}
    changes, rule, _ = r2m.apply_round2_rules(metrics_v2, recipe)
    assert rule == "R7-R11_NOOP", f"expected noop, got {rule}"
    assert changes == {}
    print("noop ok")


def main() -> int:
    case_R7()
    case_R7_already_max()
    case_R8()
    case_R9()
    case_R10()
    case_R11()
    case_NOOP()
    print("\nall round2 mutator tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
