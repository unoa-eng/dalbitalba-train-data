#!/usr/bin/env python3
"""Data path coherence gate for paper-grade RunPod launches.

Invariant 1: all paths referenced in the recipe exist on disk
Invariant 2: no .v2.jsonl for strict profiles (paper8b, budget30)
Invariant 3: SFT data schema sniff (output key must be present in first row)
Invariant 4: SHA256 matches DATA_CARD.md pinned hashes
Invariant 5: recipe SFT_DATA <-> launcher SFT_PAIR_JSONL coherence

Exit codes:
  0 - all invariants PASS
  1 - WARN only (non-strict mode violations exist)
  2 - FAIL (violations in strict mode, or violations in non-strict that are hard errors)
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "recipes"
DATA_CARD = REPO_ROOT / "docs" / "DATA_CARD.md"
STRICT_PROFILES = {"paper8b", "budget30"}

# Data keys inspected across recipes and launcher
DATA_KEYS = [
    "CPT_PHASE_1_DATA",
    "CPT_PHASE_2_DATA",
    "SFT_DATA",
    "SFT_EVAL_DATA",
    "EVAL_INPUT_DATA",
]

# Keys that the launcher always resolves (parallel to SFT_DATA / SFT_EVAL_DATA)
LAUNCHER_SFT_KEY = "TRAIN_SFT_PAIR_JSONL"
LAUNCHER_VAL_KEY = "TRAIN_VAL_JSONL"


def parse_env_file(path: Path) -> dict:
    """Parse a shell .env file, stripping comments and quotes."""
    env: dict = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        # Strip surrounding quotes (single or double)
        if len(v) >= 2 and v[0] in ('"', "'") and v[-1] == v[0]:
            v = v[1:-1]
        env[k] = v
    return env


def parse_recipe(profile: str) -> dict:
    """Load recipe env for a given profile.

    paper8b -> recipes/round2-cycle1.env (BUDGET_PROFILE=paper8b inside)
    budget30 -> recipes/budget30.env
    smoke    -> recipes/smoke.env
    """
    if profile == "paper8b":
        p = RECIPES_DIR / "round2-cycle1.env"
    else:
        p = RECIPES_DIR / f"{profile}.env"
    return parse_env_file(p)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_data_card_shas() -> dict:
    """Extract filename -> sha256 mappings from DATA_CARD.md.

    The card has a table row like:
      | `cpt_enriched.jsonl` | ... | `68a60f...` |
    We look for any 64-char lowercase hex that appears on the same line as a
    .jsonl filename (possibly with backtick formatting).
    """
    if not DATA_CARD.exists():
        return {}
    text = DATA_CARD.read_text(encoding="utf-8")
    shas: dict = {}
    # Match lines containing a .jsonl name and a 64-char hex token
    for line in text.splitlines():
        m_name = re.search(r'`?([\w.]+\.jsonl)`?', line)
        m_sha = re.search(r'([0-9a-fA-F]{64})', line)
        if m_name and m_sha:
            fname = m_name.group(1)
            sha = m_sha.group(1).lower()
            shas.setdefault(fname, sha)
    return shas


def resolve_data_path(recipe: dict, key: str) -> str | None:
    """Return the value for a data key from the recipe, or None."""
    return recipe.get(key) or None


def check_launcher_coherence(recipe: dict, verbose: bool) -> list:
    """Invariant 5: recipe SFT_DATA must be reachable via launcher resolution.

    The launcher (launch_train_pod.py) resolves SFT file via:
      resolve_workspace_data_path(("TRAIN_SFT_PAIR_JSONL","SFT_PAIR_JSONL","SFT_DATA"), ...)
    so as long as SFT_DATA is in the env tuple the launcher will pick it up.
    We verify:
      (a) launcher source code references SFT_DATA (not dropped),
      (b) if recipe has SFT_DATA, launcher's default fallback list does NOT
          resolve to a .v2. path when SFT_DATA is absent.
    """
    warnings: list = []
    launcher = REPO_ROOT / "scripts" / "launch_train_pod.py"
    if not launcher.exists():
        warnings.append("launcher scripts/launch_train_pod.py not found — skipping coherence check")
        return warnings

    ltext = launcher.read_text(encoding="utf-8")

    # Check (a): SFT_DATA is referenced in the launcher resolution tuple
    if '"SFT_DATA"' not in ltext and "'SFT_DATA'" not in ltext:
        warnings.append(
            "launcher does not reference SFT_DATA in env resolution — "
            "recipe SFT_DATA will not reach the pod (recipe↔launcher drift)"
        )
    elif verbose:
        print("  [I5a] launcher references SFT_DATA — OK")

    # Check (b): default fallback chain in launcher
    # Look for sft_pairs.v2.jsonl as a fallback default
    if "sft_pairs.v2.jsonl" in ltext:
        # Only a warning if recipe overrides it; violation if no SFT_DATA in recipe.
        # SKIP_SFT=1 (e.g. budget30 CPT-only probe) is a legitimate "SFT_DATA absent" case.
        recipe_sft = recipe.get("SFT_DATA", "")
        skip_sft = recipe.get("SKIP_SFT", "0").strip("\"'") in ("1", "true", "yes")
        if not recipe_sft and not skip_sft:
            warnings.append(
                "launcher fallback default includes sft_pairs.v2.jsonl and recipe has no SFT_DATA — "
                "pod may silently use v2 SFT pairs"
            )
        elif skip_sft and verbose:
            print("  [I5b] SKIP_SFT=1 — SFT path absence is intentional, OK")
        elif verbose:
            print(
                f"  [I5b] launcher has v2 fallback but recipe overrides with SFT_DATA={recipe_sft} — OK"
            )

    # Check (c): EVAL_INPUT_DATA consistency
    recipe_eval = recipe.get("EVAL_INPUT_DATA", "")
    if recipe_eval and "EVAL_INPUT_DATA" not in ltext:
        warnings.append(
            f"recipe sets EVAL_INPUT_DATA={recipe_eval} but launcher does not pass it — drift risk"
        )
    elif verbose and recipe_eval:
        print(f"  [I5c] EVAL_INPUT_DATA={recipe_eval} referenced in launcher — OK")

    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data path coherence gate for paper-grade RunPod launches."
    )
    parser.add_argument(
        "--profile",
        default="paper8b",
        choices=("paper8b", "budget30", "smoke"),
        help="Recipe profile to validate (default: paper8b)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 on any violation (for CI / smoke gate use)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-check detail")
    args = parser.parse_args()

    violations: list = []
    warnings: list = []

    recipe = parse_recipe(args.profile)
    if args.verbose:
        print(f"[check_data_paths] profile={args.profile} recipe_keys={len(recipe)}")
        for k in DATA_KEYS:
            print(f"  {k} = {recipe.get(k, '<not set>')}")

    # ------------------------------------------------------------------
    # Invariant 1: path existence
    # ------------------------------------------------------------------
    for key in DATA_KEYS:
        val = resolve_data_path(recipe, key)
        if not val:
            if args.verbose:
                print(f"  [I1] {key} not set in recipe — skipping existence check")
            continue
        path = REPO_ROOT / val
        if not path.exists():
            violations.append(f"[I1] path missing: {key}={val}")
        elif args.verbose:
            print(f"  [I1] {key}={val} exists — OK")

    # ------------------------------------------------------------------
    # Invariant 2: no .v2.jsonl for strict profiles
    # ------------------------------------------------------------------
    if args.profile in STRICT_PROFILES:
        for key in DATA_KEYS:
            val = resolve_data_path(recipe, key) or ""
            if ".v2." in val:
                violations.append(
                    f"[I2] strict profile '{args.profile}' uses {key}={val} (contains .v2. — must be v3)"
                )
            elif args.verbose and val:
                print(f"  [I2] {key}={val} — no .v2. — OK")

    # ------------------------------------------------------------------
    # Invariant 3: SFT data schema sniff
    # ------------------------------------------------------------------
    sft_data = resolve_data_path(recipe, "SFT_DATA")
    if sft_data:
        sft_path = REPO_ROOT / sft_data
        if sft_path.exists():
            try:
                with sft_path.open(encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if not first_line:
                    warnings.append(f"[I3] SFT data file is empty: {sft_data}")
                else:
                    first = json.loads(first_line)
                    # Accept any of these output-bearing keys
                    schema_ok_keys = ("output", "target_comment", "target", "completion", "chosen")
                    if not any(k in first for k in schema_ok_keys):
                        violations.append(
                            f"[I3] SFT schema: first row of {sft_data} has none of "
                            f"{schema_ok_keys} — found keys: {list(first.keys())[:8]}"
                        )
                    elif args.verbose:
                        found = [k for k in schema_ok_keys if k in first]
                        print(f"  [I3] SFT schema OK — key(s) found: {found}")
            except json.JSONDecodeError as exc:
                violations.append(f"[I3] SFT schema sniff failed (JSON parse): {exc}")
            except OSError as exc:
                warnings.append(f"[I3] SFT schema sniff I/O error: {exc}")
        elif args.verbose:
            print(f"  [I3] SFT file not found — skipped (caught by I1)")

    sft_eval_data = resolve_data_path(recipe, "SFT_EVAL_DATA")
    if sft_eval_data:
        eval_path = REPO_ROOT / sft_eval_data
        if eval_path.exists():
            try:
                with eval_path.open(encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if first_line:
                    first = json.loads(first_line)
                    schema_ok_keys = ("output", "target_comment", "target", "completion")
                    if not any(k in first for k in schema_ok_keys):
                        # eval schema mismatch is a warning (not always fatal)
                        warnings.append(
                            f"[I3-eval] SFT_EVAL_DATA first row of {sft_eval_data} has none of "
                            f"{schema_ok_keys} — found: {list(first.keys())[:8]}"
                        )
                    elif args.verbose:
                        found = [k for k in schema_ok_keys if k in first]
                        print(f"  [I3-eval] SFT_EVAL_DATA schema OK — key(s): {found}")
            except Exception as exc:
                warnings.append(f"[I3-eval] SFT_EVAL_DATA sniff error: {exc}")

    # ------------------------------------------------------------------
    # Invariant 4: SHA256 vs DATA_CARD.md
    # ------------------------------------------------------------------
    card_shas = parse_data_card_shas()
    if args.verbose:
        print(f"  [I4] DATA_CARD has {len(card_shas)} pinned SHA entries")

    for key in DATA_KEYS:
        val = resolve_data_path(recipe, key)
        if not val:
            continue
        path = REPO_ROOT / val
        if not path.exists():
            continue  # Already flagged by I1
        fname = Path(val).name
        expected = card_shas.get(fname) or card_shas.get(val)
        if not expected:
            if args.verbose:
                print(f"  [I4] {fname} — no pinned SHA in DATA_CARD (skip)")
            continue
        actual = sha256_of(path)
        if actual != expected:
            violations.append(
                f"[I4] SHA256 mismatch for {val}:\n"
                f"       actual  : {actual}\n"
                f"       expected: {expected}"
            )
        elif args.verbose:
            print(f"  [I4] {fname} SHA256 OK ({actual[:16]}...)")

    # ------------------------------------------------------------------
    # Invariant 5: recipe SFT_DATA <-> launcher coherence
    # ------------------------------------------------------------------
    coherence_warnings = check_launcher_coherence(recipe, args.verbose)
    warnings.extend(coherence_warnings)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_issues = len(violations) + len(warnings)
    print(
        f"[check_data_paths] profile={args.profile} strict={args.strict} "
        f"violations={len(violations)} warnings={len(warnings)}"
    )
    for v in violations:
        print(f"  VIOLATION: {v}")
    for w in warnings:
        print(f"  WARN: {w}")

    if not total_issues:
        print("  ALL INVARIANTS PASS")

    if violations:
        return 2 if args.strict else 1
    if warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
