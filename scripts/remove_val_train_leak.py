#!/usr/bin/env python3
"""Remove val rows that are exact duplicates of CPT training rows."""

import json
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CPT_PATH = ROOT / "cpt_corpus.v2.jsonl"
VAL_PATH = ROOT / "val_set.v2.jsonl"
BACKUP_PATH = ROOT / "val_set.v2.jsonl.bak"
RUNS_DIR = ROOT / "runs"
REPORT_PATH = RUNS_DIR / "leak_report.json"


def main():
    # 1. Build set of all CPT text values
    print(f"Loading CPT corpus from {CPT_PATH} ...")
    cpt_texts = set()
    with open(CPT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cpt_texts.add(row["text"])
    print(f"  CPT unique texts: {len(cpt_texts):,}")

    # 2. Load val set
    print(f"Loading val set from {VAL_PATH} ...")
    val_rows = []
    with open(VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            val_rows.append(json.loads(line))
    print(f"  Val rows loaded: {len(val_rows):,}")

    # 3. Partition into keep / leak
    kept = []
    leaked = []
    for row in val_rows:
        if row["text"] in cpt_texts:
            leaked.append(row)
        else:
            kept.append(row)

    print(f"\n=== Leak Summary ===")
    print(f"  Removed (leaked): {len(leaked)}")
    print(f"  Kept:             {len(kept)}")

    if leaked:
        print(f"\n--- Removed rows (first 50 chars of text) ---")
        for i, row in enumerate(leaked):
            preview = row["text"][:50].replace("\n", "\\n")
            print(f"  [{i+1:3d}] kind={row.get('kind','?'):12s}  {preview}...")

    # 4. Backup original val set
    print(f"\nBacking up {VAL_PATH} -> {BACKUP_PATH}")
    shutil.copy2(VAL_PATH, BACKUP_PATH)

    # 5. Write filtered val set
    print(f"Writing filtered val set ({len(kept):,} rows) ...")
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 6. Save audit report
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "cpt_file": str(CPT_PATH),
        "val_file": str(VAL_PATH),
        "original_val_count": len(val_rows),
        "removed_count": len(leaked),
        "kept_count": len(kept),
        "removed_rows": leaked,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Audit report saved to {REPORT_PATH}")

    print(f"\nDone. val_set.v2.jsonl: {len(val_rows)} -> {len(kept)} rows ({len(leaked)} removed)")


if __name__ == "__main__":
    main()
