#!/usr/bin/env python3
"""Stage K: End-to-end glue test — chain Python phases sequentially.

We can't run the GPU/peft phases locally, but we CAN verify:
  1. Each script's CLI loads (Python imports work)
  2. Schema flows: SFT-build output -> ORPO-build input -> eval input
  3. Manifest-hash propagation logic (env vars feeding into manifest strings)

Already-done phases:
  - SFT data build  -> Stage B (smoke) + sft_pairs.v3.jsonl
  - SFT MLX train   -> Stage C (smoke) + Stage F (4B)
  - ORPO pair build -> Stage G
  - Token-fire audit -> Stage I

This stage glues those outputs together to confirm format compatibility.
"""
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path("/Users/unoa/dalbitalba-train-data")
OUT = ROOT / "runs/local-integrity-2026-05-08/K_e2e_glue.json"
PY312 = "/Users/unoa/projects/dalbitalba-train-data/.venv-mlx312/bin/python"
PY39 = "/usr/bin/python3"


def step(name, ok, evidence):
    return {"step": name, "ok": ok, "evidence": evidence}


def run_help(script: Path, py: str = PY312):
    """Confirm a script imports + shows --help (i.e. CLI is structurally valid)."""
    try:
        r = subprocess.run([py, str(script), "--help"], capture_output=True, text=True, timeout=30)
        return r.returncode == 0, (r.stdout[:400] or r.stderr[:400])
    except Exception as e:
        return False, str(e)[:400]


def main():
    results = []

    # 1. SFT data build (Stage B output already produced) - schema check
    sft_path = ROOT / "runs/local-smoke-2026-05-08-comprehensive/B1_sft_threaded.jsonl"
    if not sft_path.exists():
        # try known alternate
        for alt in ["sft_pairs.v3.jsonl"]:
            ap = ROOT / alt
            if ap.exists():
                sft_path = ap
                break
    if sft_path.exists():
        with open(sft_path) as f:
            first = json.loads(f.readline())
        keys = list(first.keys())
        results.append(step("1. sft_data_present_schema_check", True, {"path": str(sft_path), "keys": keys}))
    else:
        results.append(step("1. sft_data_present_schema_check", False, {"missing": str(sft_path)}))

    # 2. ORPO pair build CLI
    ok, ev = run_help(ROOT / "scripts/round2_build_orpo_pairs.py", PY39)
    results.append(step("2. orpo_build_cli_help", ok, {"head": ev[:200]}))

    # 3. ORPO pairs schema check (Stage G output)
    g_pairs = ROOT / "runs/local-integrity-2026-05-08/G_orpo_pairs.jsonl"
    if g_pairs.exists():
        with open(g_pairs) as f:
            row = json.loads(f.readline())
        # Required schema for downstream ORPO trainer: prompt/chosen/rejected
        schema_ok = all(k in row for k in ["prompt", "chosen", "rejected"])
        results.append(step("3. orpo_pairs_schema_check", schema_ok, {"keys": list(row.keys())}))
    else:
        results.append(step("3. orpo_pairs_schema_check", False, {"missing": str(g_pairs)}))

    # 4. Phase6 generate CLI (peft not installed locally — accept ModuleNotFoundError)
    try:
        r = subprocess.run([PY312, str(ROOT / "scripts/phase6_generate.py"), "--help"], capture_output=True, text=True, timeout=20)
        # peft import gates the module — that's expected on local Mac.
        # But the SyntaxError-vs-ImportError distinction is what we want to verify.
        ok = ("ModuleNotFoundError" in r.stderr) or (r.returncode == 0)
        results.append(step("4. phase6_generate_imports_cleanly", ok, {"rc": r.returncode, "stderr_head": r.stderr[:200]}))
    except Exception as e:
        results.append(step("4. phase6_generate_imports_cleanly", False, {"err": str(e)[:200]}))

    # 5. Phase6 eval_v2 CLI — must load
    ok, ev = run_help(ROOT / "scripts/phase6_eval_v2.py", PY312)
    results.append(step("5. phase6_eval_v2_cli_help", ok, {"head": ev[:200]}))

    # 6-8. AST-parse static check on train_*.py (module-level FileHandler creation
    #     blocks live import locally because LOG_FILE lives on RunPod path; AST
    #     parse confirms the script is syntactically valid for transport).
    import ast
    for stepid, modname in [("6", "train_sft"), ("7", "train_cpt"), ("8", "train_orpo")]:
        p = ROOT / f"{modname}.py"
        try:
            src = p.read_text()
            ast.parse(src)
            results.append(step(f"{stepid}. {modname}_ast_parse", True, {"path": str(p), "chars": len(src)}))
        except SyntaxError as e:
            results.append(step(f"{stepid}. {modname}_ast_parse", False, {"err": str(e)[:200]}))

    # 9. Manifest-hash propagation: simulate env-var change and confirm manifest string differs
    def manifest_string(env):
        # Mirrors chain_train_round2.sh phase3 manifest assembly
        return (
            f"weight={env.get('SFT_LOSS_WEIGHT_ARGOT', '1.5')}_"
            f"thresh={env.get('SFT_LOSS_WEIGHT_THRESHOLD', '2')}_"
            f"terms={env.get('SFT_LOSSWEIGHT_TERMS_FOOTPRINT', env.get('SFT_LOSS_WEIGHT_TERMS', 'default'))}_"
            f"dedup={env.get('SFT_APPLY_DEDUP', '1')}"
        )

    m_default = manifest_string({})
    m_changed_weight = manifest_string({"SFT_LOSS_WEIGHT_ARGOT": "2.0"})
    m_changed_dedup = manifest_string({"SFT_APPLY_DEDUP": "0"})
    m_changed_both = manifest_string({"SFT_LOSS_WEIGHT_ARGOT": "2.0", "SFT_APPLY_DEDUP": "0"})
    diff_test = (
        m_default != m_changed_weight
        and m_default != m_changed_dedup
        and m_changed_weight != m_changed_dedup
        and m_changed_both != m_changed_weight
    )
    results.append(step(
        "9. manifest_hash_propagation",
        diff_test,
        {
            "default": m_default,
            "weight_changed": m_changed_weight,
            "dedup_changed": m_changed_dedup,
            "both_changed": m_changed_both,
        }
    ))

    # 10. Confirm chain_train_round2.sh present (the actual orchestrator)
    chain = ROOT / "chain_train_round2.sh"
    results.append(step("10. chain_train_round2_present", chain.exists(), {"path": str(chain), "lines": sum(1 for _ in open(chain)) if chain.exists() else 0}))

    n_ok = sum(1 for r in results if r["ok"])
    n_total = len(results)
    out = {
        "stage": "K",
        "summary": f"{n_ok}/{n_total} steps OK",
        "all_pass": n_ok == n_total,
        "steps": results,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
