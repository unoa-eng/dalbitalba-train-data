#!/usr/bin/env python3
"""Remove val rows that are exact duplicates of CPT or SFT training rows.

v3 update:
- Paths updated to v3 files
- SFT leak check added: sft_thread_conditioned.jsonl output vs val target_comment
- sft_thread_conditioned.eval.jsonl also cleaned of SFT leaks
- Supports both cpt_corpus.v3.jsonl and cpt_enriched.jsonl (union of both)
"""

import json
import shutil
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

SFT_EVAL_PATH = ROOT / "sft_thread_conditioned.eval.jsonl"
SFT_EVAL_BACKUP = ROOT / "sft_thread_conditioned.eval.jsonl.pre-leak-removal.bak"

RUNS_DIR = ROOT / "runs"
REPORT_PATH = RUNS_DIR / "leak_report.json"

# val_set.v3 uses target_comment as the text field
VAL_TEXT_FIELD = "target_comment"


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


def main():
    # 1. Build CPT text set (union of all available v3 CPT files)
    print("=== Loading CPT corpus ===")
    cpt_texts = load_cpt_texts(CPT_PATHS)
    print(f"  CPT unique texts total: {len(cpt_texts):,}")

    # 2. Build SFT output set
    print("\n=== Loading SFT outputs ===")
    sft_outputs = load_sft_outputs(SFT_PATH)
    print(f"  SFT unique outputs: {len(sft_outputs):,}")

    # 3. Filter val_set.v3.jsonl
    print(f"\n=== Filtering {VAL_PATH.name} (field='{VAL_TEXT_FIELD}') ===")
    val_rows, kept, cpt_leaked, sft_leaked = filter_val(
        VAL_PATH, cpt_texts, sft_outputs, VAL_TEXT_FIELD
    )

    total_leaked = len(cpt_leaked) + len(sft_leaked)
    print(f"\n=== Leak Summary for {VAL_PATH.name} ===")
    print(f"  Original rows:        {len(val_rows):,}")
    print(f"  CPT leaked:           {len(cpt_leaked):,}")
    print(f"  SFT leaked:           {len(sft_leaked):,}")
    print(f"  Total removed:        {total_leaked:,}")
    print(f"  Kept:                 {len(kept):,}")

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

    # 4. Backup + write filtered val_set.v3
    if not BACKUP_PATH.exists():
        print(f"\nBacking up {VAL_PATH.name} -> {BACKUP_PATH.name}")
        shutil.copy2(VAL_PATH, BACKUP_PATH)
    else:
        print(f"\n[INFO] Backup already exists: {BACKUP_PATH.name} (skipping overwrite)")

    print(f"Writing filtered val set ({len(kept):,} rows) -> {VAL_PATH.name}")
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 5. Filter sft_thread_conditioned.eval.jsonl (SFT leak only – eval rows vs train SFT)
    print(f"\n=== Filtering {SFT_EVAL_PATH.name} ===")
    eval_rows_orig = []
    eval_kept = []
    eval_leaked = []

    if SFT_EVAL_PATH.exists():
        with open(SFT_EVAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                eval_rows_orig.append(row)
                out = (row.get("output") or row.get("target") or "").strip()
                if out in sft_outputs:
                    eval_leaked.append(row)
                else:
                    eval_kept.append(row)

        print(f"  Original eval rows:   {len(eval_rows_orig):,}")
        print(f"  SFT leaked:           {len(eval_leaked):,}")
        print(f"  Kept:                 {len(eval_kept):,}")

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

    # 6. Save audit report
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
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

    print(f"\n=== Final Summary ===")
    print(f"  val_set.v3.jsonl:               {len(val_rows)} -> {len(kept)} ({total_leaked} removed)")
    print(f"  sft_thread_conditioned.eval:    {len(eval_rows_orig)} -> {len(eval_kept)} ({len(eval_leaked)} removed)")


if __name__ == "__main__":
    main()
