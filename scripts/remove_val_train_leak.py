#!/usr/bin/env python3
"""Remove val rows that are exact duplicates of CPT or SFT training rows.

v3 update:
- Paths updated to v3 files
- SFT leak check added: sft_thread_conditioned.jsonl output vs val target_comment
- sft_thread_conditioned.eval.jsonl also cleaned of SFT leaks
- Supports both cpt_corpus.v3.jsonl and cpt_enriched.jsonl (union of both)

v3.1-thread-holdout update:
- Added --enforce-thread-holdout flag (default: True)
- Removes val rows whose thread_key (or root_id) overlaps with any SFT root_id
- Writes backup val_set.v3.jsonl.pre-thread-holdout.bak before overwrite
- Saves thread_holdout_report.json to runs/
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

# v3 paths
CPT_PATHS = [
    ROOT / "cpt_corpus.v3.jsonl",
    ROOT / "cpt_enriched.jsonl",
]
SFT_PATH = ROOT / "sft_thread_conditioned.jsonl"
VAL_PATH = ROOT / "val_set.v3.jsonl"
BACKUP_PATH = ROOT / "val_set.v3.jsonl.pre-leak-removal.bak"
THREAD_HOLDOUT_BACKUP = ROOT / "val_set.v3.jsonl.pre-thread-holdout.bak"

SFT_EVAL_PATH = ROOT / "sft_thread_conditioned.eval.jsonl"
SFT_EVAL_BACKUP = ROOT / "sft_thread_conditioned.eval.jsonl.pre-leak-removal.bak"

RUNS_DIR = ROOT / "runs"
REPORT_PATH = RUNS_DIR / "leak_report.json"
THREAD_HOLDOUT_REPORT_PATH = RUNS_DIR / "thread_holdout_report.json"

# val_set.v3 uses target_comment as the text field
VAL_TEXT_FIELD = "target_comment"

# Minimum acceptable val rows for paper-grade evaluation
MIN_EVAL_ROWS = 100


def load_cpt_texts(paths):
    """Load all CPT text values from one or more JSONL files."""
    cpt_texts = set()
    for p in paths:
        if not p.exists():
            print(f"  [SKIP] CPT file not found: {p}")
            continue
        print(f"  Loading CPT: {p} ...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                val = row.get("text", "")
                cpt_texts.add(val.strip())
    return cpt_texts


def load_sft_outputs(path):
    """Load SFT output values (strip whitespace)."""
    outputs = set()
    if not path.exists():
        print(f"  [SKIP] SFT file not found: {path}")
        return outputs
    print(f"  Loading SFT outputs: {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            val = row.get("output") or row.get("target") or ""
            outputs.add(val.strip())
    return outputs


def filter_val(val_path, cpt_texts, sft_outputs, text_field):
    """Load val rows, partition into keep/leak sets, return (kept, cpt_leaked, sft_leaked)."""
    val_rows = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            val_rows.append(json.loads(line))
    print(f"  Val rows loaded: {len(val_rows):,}")

    kept = []
    cpt_leaked = []
    sft_leaked = []

    for row in val_rows:
        text = (row.get(text_field) or "").strip()
        if text in cpt_texts:
            cpt_leaked.append(row)
        elif text in sft_outputs:
            sft_leaked.append(row)
        else:
            kept.append(row)

    return val_rows, kept, cpt_leaked, sft_leaked


def load_sft_root_ids(path):
    """Load all root_id (or thread_key) values from SFT file as a set of strings."""
    root_ids = set()
    if not path.exists():
        print(f"  [SKIP] SFT file not found: {path}")
        return root_ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rid = row.get("root_id") or row.get("thread_key")
            if rid is not None:
                root_ids.add(str(rid))
    return root_ids


def enforce_thread_holdout(val_rows, sft_root_ids):
    """Remove val rows whose thread_key/root_id overlaps with any SFT root_id.

    Returns (kept, thread_leaked, overlap_ids).
    Val rows use 'thread_key'; SFT rows use 'root_id' — same semantic identifier.
    """
    kept = []
    thread_leaked = []
    overlap_ids = set()

    for row in val_rows:
        rid = row.get("thread_key") or row.get("root_id")
        rid_str = str(rid) if rid is not None else ""
        if rid_str and rid_str in sft_root_ids:
            thread_leaked.append(row)
            overlap_ids.add(rid_str)
        else:
            kept.append(row)

    return kept, thread_leaked, overlap_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove val leaks (text + thread-level) from val_set.v3.jsonl"
    )
    parser.add_argument(
        "--enforce-thread-holdout",
        dest="enforce_thread_holdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove val rows whose thread_key shares a root_id with any SFT training row (default: True)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Build CPT text set (union of all available v3 CPT files)
    print("=== Loading CPT corpus ===")
    cpt_texts = load_cpt_texts(CPT_PATHS)
    print(f"  CPT unique texts total: {len(cpt_texts):,}")

    # 2. Build SFT output set
    print("\n=== Loading SFT outputs ===")
    sft_outputs = load_sft_outputs(SFT_PATH)
    print(f"  SFT unique outputs: {len(sft_outputs):,}")

    # 3. Filter val_set.v3.jsonl (text-level leak)
    print(f"\n=== Filtering {VAL_PATH.name} (field='{VAL_TEXT_FIELD}') ===")
    val_rows, kept, cpt_leaked, sft_leaked = filter_val(
        VAL_PATH, cpt_texts, sft_outputs, VAL_TEXT_FIELD
    )

    total_leaked = len(cpt_leaked) + len(sft_leaked)
    print(f"\n=== Text-Leak Summary for {VAL_PATH.name} ===")
    print(f"  Original rows:        {len(val_rows):,}")
    print(f"  CPT leaked:           {len(cpt_leaked):,}")
    print(f"  SFT leaked:           {len(sft_leaked):,}")
    print(f"  Total removed:        {total_leaked:,}")
    print(f"  Kept after text-leak: {len(kept):,}")

    if cpt_leaked:
        print(f"\n--- CPT-leaked rows (first 3) ---")
        for row in cpt_leaked[:3]:
            preview = (row.get(VAL_TEXT_FIELD) or "")[:60].replace("\n", "\\n")
            print(f"  {preview}...")

    if sft_leaked:
        print(f"\n--- SFT-leaked rows (first 3) ---")
        for row in sft_leaked[:3]:
            preview = (row.get(VAL_TEXT_FIELD) or "")[:60].replace("\n", "\\n")
            print(f"  {preview}...")

    # 4. Thread-level holdout (ME-08 advisory)
    thread_kept = kept
    thread_leaked = []
    overlap_ids = set()
    sft_root_id_count = 0

    if args.enforce_thread_holdout:
        print(f"\n=== Thread-Level Holdout (--enforce-thread-holdout) ===")
        sft_root_ids = load_sft_root_ids(SFT_PATH)
        sft_root_id_count = len(sft_root_ids)
        print(f"  SFT unique root_ids: {sft_root_id_count:,}")

        thread_kept, thread_leaked, overlap_ids = enforce_thread_holdout(kept, sft_root_ids)

        print(f"  Val rows before thread-holdout: {len(kept):,}")
        print(f"  Overlapping root_ids found:     {len(overlap_ids):,}")
        print(f"  Val rows removed (thread):      {len(thread_leaked):,}")
        print(f"  Val rows kept after holdout:    {len(thread_kept):,}")
    else:
        print("\n[INFO] --no-enforce-thread-holdout: skipping thread-level check")

    # 5. Backup + write filtered val_set.v3
    #    Text-leak backup (pre-existing protocol)
    if not BACKUP_PATH.exists():
        print(f"\nBacking up {VAL_PATH.name} -> {BACKUP_PATH.name}")
        shutil.copy2(VAL_PATH, BACKUP_PATH)
    else:
        print(f"\n[INFO] Text-leak backup already exists: {BACKUP_PATH.name} (skipping)")

    #    Thread-holdout backup (new)
    if args.enforce_thread_holdout:
        if not THREAD_HOLDOUT_BACKUP.exists():
            print(f"Backing up {VAL_PATH.name} -> {THREAD_HOLDOUT_BACKUP.name}")
            shutil.copy2(VAL_PATH, THREAD_HOLDOUT_BACKUP)
        else:
            print(f"[INFO] Thread-holdout backup already exists: {THREAD_HOLDOUT_BACKUP.name} (skipping)")

    final_rows = thread_kept
    print(f"Writing final val set ({len(final_rows):,} rows) -> {VAL_PATH.name}")
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 6. Filter sft_thread_conditioned.eval.jsonl (SFT leak only – eval rows vs train SFT)
    print(f"\n=== Filtering {SFT_EVAL_PATH.name} ===")
    eval_rows_orig = []
    eval_kept = []
    eval_leaked = []

    if SFT_EVAL_PATH.exists():
        thread_leaked: list = []
        with open(SFT_EVAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                eval_rows_orig.append(row)
                out = (row.get("output") or row.get("target") or "").strip()
                rid = row.get("root_id") or row.get("thread_key")
                rid_str = str(rid) if rid else ""
                # Exact-text leak OR thread-level (root_id) overlap → drop
                if out in sft_outputs:
                    eval_leaked.append(row)
                elif args.enforce_thread_holdout and rid_str and rid_str in sft_root_ids:
                    thread_leaked.append(row)
                else:
                    eval_kept.append(row)

        print(f"  Original eval rows:   {len(eval_rows_orig):,}")
        print(f"  SFT exact-text leaked:{len(eval_leaked):,}")
        print(f"  Thread-level leaked:  {len(thread_leaked):,}")
        print(f"  Kept:                 {len(eval_kept):,}")
        eval_leaked = eval_leaked + thread_leaked  # merge for downstream backup/write

        if not SFT_EVAL_BACKUP.exists():
            print(f"Backing up {SFT_EVAL_PATH.name} -> {SFT_EVAL_BACKUP.name}")
            shutil.copy2(SFT_EVAL_PATH, SFT_EVAL_BACKUP)
        else:
            print(f"[INFO] Eval backup already exists (skipping overwrite)")

        if eval_leaked:
            print(f"Writing filtered eval set ({len(eval_kept):,} rows) -> {SFT_EVAL_PATH.name}")
            with open(SFT_EVAL_PATH, "w", encoding="utf-8") as f:
                for row in eval_kept:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            print("  No leaks in eval set. File unchanged.")
    else:
        print(f"  [SKIP] {SFT_EVAL_PATH} not found")

    # 7. N>=100 check
    n_final = len(final_rows)
    n_pass = n_final >= MIN_EVAL_ROWS
    print(f"\n=== N>=100 Check ===")
    print(f"  Final val rows: {n_final}")
    if n_pass:
        print(f"  PASS: {n_final} >= {MIN_EVAL_ROWS} — sufficient for paper-grade evaluation")
    else:
        print(f"  FAIL: {n_final} < {MIN_EVAL_ROWS} — CRITICAL: insufficient for paper-grade evaluation")
        print(f"  RECOMMENDATION: Consider moving some sft_thread_conditioned rows to holdout pool")
        warning_dir = ROOT / ".omc"
        warning_dir.mkdir(parents=True, exist_ok=True)
        warning_path = warning_dir / "thread_holdout_warning.md"
        with open(warning_path, "w", encoding="utf-8") as wf:
            wf.write(f"# CRITICAL: Thread Holdout N<100 Warning\n\n")
            wf.write(f"Generated: {datetime.now().isoformat()}\n\n")
            wf.write(f"After thread-level holdout enforcement, val_set.v3.jsonl has only **{n_final} rows** ")
            wf.write(f"(minimum required: {MIN_EVAL_ROWS}).\n\n")
            wf.write(f"## Stats\n")
            wf.write(f"- val_before_text_leak_removal: {len(val_rows)}\n")
            wf.write(f"- val_after_text_leak_removal: {len(kept)}\n")
            wf.write(f"- val_after_thread_holdout: {n_final}\n")
            wf.write(f"- thread_overlapping_root_ids: {len(overlap_ids)}\n\n")
            wf.write(f"## Recommendation\n")
            wf.write(f"Consider moving some rows from sft_thread_conditioned.jsonl into the holdout pool ")
            wf.write(f"to reach N>={MIN_EVAL_ROWS}. Do NOT execute this automatically — human review required.\n")
        print(f"  Warning written to: {warning_path}")

    # 8. Save audit reports
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Existing leak report
    report = {
        "timestamp": datetime.now().isoformat(),
        "version": "v3",
        "cpt_files": [str(p) for p in CPT_PATHS if p.exists()],
        "sft_file": str(SFT_PATH),
        "val_file": str(VAL_PATH),
        "val_text_field": VAL_TEXT_FIELD,
        "val_original_count": len(val_rows),
        "val_cpt_leak_count": len(cpt_leaked),
        "val_sft_leak_count": len(sft_leaked),
        "val_total_leak_count": total_leaked,
        "val_kept_count": len(kept),
        "eval_original_count": len(eval_rows_orig),
        "eval_sft_leak_count": len(eval_leaked),
        "eval_kept_count": len(eval_kept),
        "removed_val_rows": cpt_leaked + sft_leaked,
        "removed_eval_rows": eval_leaked,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nAudit report saved to {REPORT_PATH}")

    # Thread-holdout report (v3.1)
    thread_report = {
        "timestamp": datetime.now().isoformat(),
        "version": "v3.1-thread-holdout",
        "enforce_thread_holdout": args.enforce_thread_holdout,
        "val_before_count": len(val_rows),
        "val_after_text_leak": len(kept),
        "val_after_count": n_final,
        "removed_count": len(val_rows) - n_final,
        "text_leak_removed": total_leaked,
        "thread_holdout_removed": len(thread_leaked),
        "sft_root_count": sft_root_id_count,
        "overlapping_root_ids": sorted(overlap_ids),
        "overlapping_root_id_count": len(overlap_ids),
        "eval_kept_count": n_final,
        "n_pass": n_pass,
        "backup_path": str(THREAD_HOLDOUT_BACKUP),
    }
    with open(THREAD_HOLDOUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(thread_report, f, ensure_ascii=False, indent=2)
    print(f"Thread holdout report saved to {THREAD_HOLDOUT_REPORT_PATH}")

    print(f"\n=== Final Summary ===")
    print(f"  val_set.v3.jsonl:               {len(val_rows)} -> {n_final} ({len(val_rows) - n_final} removed total)")
    print(f"    of which text-leak:           {total_leaked}")
    print(f"    of which thread-holdout:      {len(thread_leaked)}")
    print(f"  sft_thread_conditioned.eval:    {len(eval_rows_orig)} -> {len(eval_kept)} ({len(eval_leaked)} removed)")
    print(f"  N>=100 check:                   {'PASS' if n_pass else 'FAIL'} ({n_final} rows)")


if __name__ == "__main__":
    main()
