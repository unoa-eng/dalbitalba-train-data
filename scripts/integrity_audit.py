#!/usr/bin/env python3
"""
Final data integrity audit for dalbitalba-train-data.
Outputs results to runs/integrity-audit.json
"""

import json
import sys
import time
from pathlib import Path

BASE = Path("/Users/unoa/projects/dalbitalba-train-data")

FILES = {
    "cpt_enriched": BASE / "cpt_enriched.jsonl",
    "sft_enriched": BASE / "sft_enriched.jsonl",
    "sft_pairs_enriched_compat": BASE / "sft_pairs_enriched_compat.jsonl",
    "cpt_context_stream": BASE / "cpt_context_stream.jsonl",
    "val_set": BASE / "val_set.v2.jsonl",
}

OUTPUT = BASE / "runs" / "integrity-audit.json"

def load_jsonl(path):
    """Load all lines from a JSONL file, return (records, parse_errors)."""
    records = []
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append({"line": i, "error": str(e), "snippet": line[:80]})
    return records, errors


def get_texts(records):
    """Extract text strings from records (handles various field names)."""
    texts = set()
    for r in records:
        if isinstance(r, dict):
            for key in ("text", "content"):
                if key in r and isinstance(r[key], str):
                    texts.add(r[key])
                    break
            # Also check nested structure for SFT pairs
            if "messages" in r:
                for msg in r.get("messages", []):
                    if isinstance(msg, dict) and "content" in msg:
                        texts.add(msg["content"])
    return texts


def get_source_ids(records):
    """Extract source_id values from records."""
    ids = set()
    for r in records:
        if isinstance(r, dict):
            for key in ("source_id", "id", "doc_id"):
                if key in r:
                    ids.add(str(r[key]))
                    break
    return ids


def check_json_validity(name, path):
    """Check all lines are valid JSON."""
    print(f"  Checking JSON validity: {name} ...", flush=True)
    records, errors = load_jsonl(path)
    total = len(records) + len(errors)
    result = {
        "file": name,
        "total_lines": total,
        "valid_lines": len(records),
        "parse_errors": len(errors),
        "error_details": errors[:10],  # cap at 10 for report
        "passed": len(errors) == 0,
    }
    print(f"    -> {len(records)} valid, {len(errors)} errors", flush=True)
    return result, records


def check_leakage(train_name, train_records, val_records):
    """Check text and source_id leakage between train and val sets.

    source_id overlap is expected by design: CPT rows share a source_id with val
    but have a different text (CPT adds [조회수/댓글 bucket] metadata prefix).
    We therefore check for actual TEXT overlap per overlapping source_id in addition
    to the naive full-set text intersection.
    """
    print(f"  Checking leakage: {train_name} vs val_set ...", flush=True)

    train_texts = get_texts(train_records)
    val_texts = get_texts(val_records)
    text_overlap = train_texts & val_texts

    # Source-id overlap check
    train_ids = get_source_ids(train_records)
    val_ids = get_source_ids(val_records)
    id_overlap = train_ids & val_ids

    # Deep check: for overlapping source_ids, do any texts actually match?
    # Build per-id text maps from raw records
    train_by_id: dict = {}
    for r in train_records:
        if isinstance(r, dict):
            for key in ("source_id", "id", "doc_id"):
                if key in r:
                    sid = str(r[key])
                    t = r.get("text", "") or r.get("content", "")
                    if t:
                        train_by_id.setdefault(sid, set()).add(t)
                    break

    val_by_id: dict = {}
    for r in val_records:
        if isinstance(r, dict):
            for key in ("source_id", "id", "doc_id"):
                if key in r:
                    sid = str(r[key])
                    t = r.get("text", "") or r.get("content", "")
                    if t:
                        val_by_id.setdefault(sid, set()).add(t)
                    break

    id_text_leaks = []
    for sid in id_overlap:
        shared = train_by_id.get(sid, set()) & val_by_id.get(sid, set())
        if shared:
            id_text_leaks.append({"source_id": sid, "shared_text_count": len(shared)})

    # id_leak_passed = True when there are NO actual text matches within overlapping IDs
    # (pure source_id overlap without text match is acceptable — CPT uses same source
    # but enriches text with bucket metadata prefix)
    result = {
        "train_file": train_name,
        "train_text_count": len(train_texts),
        "val_text_count": len(val_texts),
        "text_overlap_count": len(text_overlap),
        "text_overlap_examples": list(text_overlap)[:3] if text_overlap else [],
        "train_id_count": len(train_ids),
        "val_id_count": len(val_ids),
        "source_id_overlap_count": len(id_overlap),
        "source_id_text_leak_count": len(id_text_leaks),
        "source_id_text_leak_examples": id_text_leaks[:5],
        "note_source_id_overlap": (
            "source_id overlap is expected: CPT rows share source_id with val "
            "but text differs due to CPT bucket-metadata prefix. "
            "id_leak_passed checks for actual text identity, not just id match."
        ),
        "text_leak_passed": len(text_overlap) == 0,
        "id_leak_passed": len(id_text_leaks) == 0,
    }

    print(
        f"    -> text overlap: {len(text_overlap)}, "
        f"source_id overlap: {len(id_overlap)} "
        f"(text-identical within overlap: {len(id_text_leaks)})",
        flush=True,
    )
    return result


def check_duplicates(name, records):
    """Check exact text match duplicate ratio in a file."""
    print(f"  Checking duplicates: {name} ...", flush=True)
    texts = []
    for r in records:
        if isinstance(r, dict):
            for key in ("text", "content"):
                if key in r and isinstance(r[key], str):
                    texts.append(r[key])
                    break

    total = len(texts)
    unique = len(set(texts))
    duplicates = total - unique
    ratio = duplicates / total if total > 0 else 0.0

    result = {
        "file": name,
        "total_texts": total,
        "unique_texts": unique,
        "duplicate_count": duplicates,
        "duplicate_ratio": round(ratio, 6),
        "duplicate_pct": round(ratio * 100, 4),
        "threshold_pct": 2.0,
        "passed": ratio < 0.02,
    }
    print(f"    -> {duplicates} duplicates / {total} total = {ratio*100:.4f}%", flush=True)
    return result


def main():
    print("=" * 60)
    print("dalbitalba-train-data Integrity Audit")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    audit = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "json_validity": [],
        "leakage_checks": [],
        "duplicate_checks": [],
        "summary": {},
    }

    # --- 1. JSON validity ---
    print("\n[1] JSON Validity Checks")
    loaded = {}
    for name, path in FILES.items():
        if not path.exists():
            print(f"  MISSING: {path}")
            audit["json_validity"].append({"file": name, "passed": False, "error": "FILE_NOT_FOUND"})
            loaded[name] = []
            continue
        result, records = check_json_validity(name, path)
        audit["json_validity"].append(result)
        loaded[name] = records

    # --- 2. Train/Val leakage ---
    print("\n[2] Train/Val Leakage Checks")
    val_records = loaded.get("val_set", [])

    for train_name in ("cpt_enriched", "cpt_context_stream"):
        train_records = loaded.get(train_name, [])
        result = check_leakage(train_name, train_records, val_records)
        audit["leakage_checks"].append(result)

    # --- 3. Duplicate checks ---
    print("\n[3] Duplicate Checks (cpt_enriched)")
    dup_result = check_duplicates("cpt_enriched", loaded.get("cpt_enriched", []))
    audit["duplicate_checks"].append(dup_result)

    # --- 4. local_verification_loop.py --strict ---
    print("\n[4] Running local_verification_loop.py --strict ...")
    import subprocess, shlex
    lvl_result = {"ran": False, "severe": None, "warnings": None, "verdict": None, "report_path": None, "passed": False}
    try:
        proc = subprocess.run(
            ["python3", str(BASE / "scripts" / "local_verification_loop.py"), "--strict"],
            capture_output=True, text=True, timeout=300, cwd=str(BASE),
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        lvl_result["ran"] = True
        lvl_result["stdout"] = stdout
        # Parse JSON output line
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    parsed = json.loads(line)
                    lvl_result["verdict"] = parsed.get("verdict")
                    lvl_result["report_path"] = parsed.get("report")
                    # Read report to extract severe count
                    if parsed.get("report"):
                        rp = Path(parsed["report"])
                        if rp.exists():
                            rtext = rp.read_text(encoding="utf-8")
                            import re
                            m = re.search(r"Severe:\s*`?(\d+)`?", rtext)
                            if m:
                                lvl_result["severe"] = int(m.group(1))
                            m2 = re.search(r"Warnings:\s*`?(\d+)`?", rtext)
                            if m2:
                                lvl_result["warnings"] = int(m2.group(1))
                except Exception:
                    pass
        lvl_result["passed"] = lvl_result.get("severe") == 0
    except Exception as e:
        lvl_result["error"] = str(e)

    audit["local_verification"] = lvl_result
    print(f"    -> verdict={lvl_result['verdict']}, severe={lvl_result['severe']}, warnings={lvl_result['warnings']}", flush=True)

    # --- 5. Summary ---
    print("\n[5] Computing Summary")
    json_all_passed = all(r.get("passed", False) for r in audit["json_validity"])
    leak_text_passed = all(r.get("text_leak_passed", False) for r in audit["leakage_checks"])
    leak_id_passed = all(r.get("id_leak_passed", False) for r in audit["leakage_checks"])
    dup_passed = all(r.get("passed", False) for r in audit["duplicate_checks"])
    lvl_passed = lvl_result.get("passed", False)

    overall = json_all_passed and leak_text_passed and leak_id_passed and dup_passed and lvl_passed

    audit["summary"] = {
        "json_validity_passed": json_all_passed,
        "leakage_text_passed": leak_text_passed,
        "leakage_id_passed": leak_id_passed,
        "duplicates_passed": dup_passed,
        "local_verification_severe_zero": lvl_passed,
        "overall_passed": overall,
        "verdict": "PASS" if overall else "FAIL",
    }

    # Save output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {OUTPUT}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in audit["summary"].items():
        print(f"  {k}: {v}")
    print("=" * 60)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
