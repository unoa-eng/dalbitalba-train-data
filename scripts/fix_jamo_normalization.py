#!/usr/bin/env python3
"""
fix_jamo_normalization.py — Restore Compatibility Jamo from NFKC-damaged Choseong/Jungseong Jamo.

NFKC normalization converted standalone Compatibility Jamo (ㅋㅎㅠㅜㅡ etc., U+3131-3163)
into Choseong/Jungseong Jamo (ᄏ휴ᅮᅳ etc., U+1100-11FF). This script reverses that
transformation on all v2 dataset files.

Since composed Hangul syllables live in U+AC00-D7AF (not U+1100-11FF), a simple
character-by-character replacement of all U+1100-11FF characters is safe.

Usage:
    python scripts/fix_jamo_normalization.py [--dry-run]
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILES = [
    PROJECT_ROOT / "cpt_corpus.v2.jsonl",
    PROJECT_ROOT / "sft_pairs.v2.jsonl",
    PROJECT_ROOT / "val_set.v2.jsonl",
]
BACKUP_DIR = PROJECT_ROOT / "archive" / "dataset-backups"

# ── Mappings ────────────────────────────────────────────────────────────────

# Choseong (initial consonants) → Compatibility Jamo
CHOSEONG_TO_COMPAT = {
    '\u1100': 'ㄱ', '\u1101': 'ㄲ', '\u1102': 'ㄴ', '\u1103': 'ㄷ',
    '\u1104': 'ㄸ', '\u1105': 'ㄹ', '\u1106': 'ㅁ', '\u1107': 'ㅂ',
    '\u1108': 'ㅃ', '\u1109': 'ㅅ', '\u110A': 'ㅆ', '\u110B': 'ㅇ',
    '\u110C': 'ㅈ', '\u110D': 'ㅉ', '\u110E': 'ㅊ', '\u110F': 'ㅋ',
    '\u1110': 'ㅌ', '\u1111': 'ㅍ', '\u1112': 'ㅎ',
}

# Jungseong (vowels) → Compatibility Jamo
JUNGSEONG_TO_COMPAT = {
    '\u1161': 'ㅏ', '\u1162': 'ㅐ', '\u1163': 'ㅑ', '\u1164': 'ㅒ',
    '\u1165': 'ㅓ', '\u1166': 'ㅔ', '\u1167': 'ㅕ', '\u1168': 'ㅖ',
    '\u1169': 'ㅗ', '\u116A': 'ㅘ', '\u116B': 'ㅙ', '\u116C': 'ㅚ',
    '\u116D': 'ㅛ', '\u116E': 'ㅜ', '\u116F': 'ㅝ', '\u1170': 'ㅞ',
    '\u1171': 'ㅟ', '\u1172': 'ㅠ', '\u1173': 'ㅡ', '\u1174': 'ㅢ',
    '\u1175': 'ㅣ',
}

# Merge into a single lookup table
JAMO_FIX_MAP = {**CHOSEONG_TO_COMPAT, **JUNGSEONG_TO_COMPAT}

# Build translation table for str.translate (fast)
JAMO_TRANS = str.maketrans(JAMO_FIX_MAP)


def count_jamo(text: str) -> tuple[int, int]:
    """Return (choseong_jungseong_count, compat_jamo_count)."""
    cj = 0
    compat = 0
    for ch in text:
        cp = ord(ch)
        if 0x1100 <= cp <= 0x11FF:
            cj += 1
        elif 0x3131 <= cp <= 0x3163:
            compat += 1
    return cj, compat


def fix_file(path: Path, dry_run: bool) -> dict:
    """Fix jamo in one JSONL file. Returns stats dict.

    Uses binary read/write with b'\\n' splitting to preserve any embedded
    control characters inside JSON strings (splitlines() would incorrectly
    split on \\x0b, \\x0c, \\x1c-\\x1e, etc.).
    """
    before_cj = 0
    before_compat = 0
    after_cj = 0
    after_compat = 0
    lines_changed = 0
    total_lines = 0

    raw_bytes = path.read_bytes()
    # Split only on \n (not universal newlines) to preserve embedded control chars
    raw_lines = raw_bytes.split(b"\n")
    fixed_lines = []

    for raw_line in raw_lines:
        line = raw_line.decode("utf-8")
        if not line.strip():
            fixed_lines.append(raw_line)
            continue
        total_lines += 1

        cj, compat = count_jamo(line)
        before_cj += cj
        before_compat += compat

        fixed = line.translate(JAMO_TRANS)

        cj2, compat2 = count_jamo(fixed)
        after_cj += cj2
        after_compat += compat2

        if fixed != line:
            lines_changed += 1
        fixed_lines.append(fixed.encode("utf-8"))

    if not dry_run:
        path.write_bytes(b"\n".join(fixed_lines))

    return {
        "total_lines": total_lines,
        "lines_changed": lines_changed,
        "before_choseong_jungseong": before_cj,
        "before_compat_jamo": before_compat,
        "after_choseong_jungseong": after_cj,
        "after_compat_jamo": after_compat,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Restore Compatibility Jamo from NFKC-damaged Choseong/Jungseong"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts without modifying files")
    args = parser.parse_args()

    # Verify files exist
    for path in DATA_FILES:
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)

    # Sanity check: "ᄏᄏᄏ" → "ㅋㅋㅋ"
    test_input = "\u110F\u110F\u110F"
    test_output = test_input.translate(JAMO_TRANS)
    assert test_output == "ㅋㅋㅋ", f"Sanity check failed: got {test_output!r}"
    print(f"Sanity check: '{test_input}' → '{test_output}' ✓")

    # Backup
    if not args.dry_run:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = BACKUP_DIR / f"budget30-jamofix-{ts}"
        backup_path.mkdir(parents=True, exist_ok=True)
        for path in DATA_FILES:
            shutil.copy2(path, backup_path / path.name)
        print(f"Backed up to {backup_path}")
    else:
        print("[DRY RUN] — no files will be modified\n")

    # Process
    print("\n" + "=" * 65)
    print("JAMO NORMALIZATION FIX")
    print("=" * 65)

    total_before_cj = 0
    total_after_compat = 0

    for path in DATA_FILES:
        print(f"\n  {path.name}:")
        stats = fix_file(path, args.dry_run)

        total_before_cj += stats["before_choseong_jungseong"]
        total_after_compat += stats["after_compat_jamo"]

        print(f"    Total lines:              {stats['total_lines']:>8,}")
        print(f"    Lines changed:            {stats['lines_changed']:>8,}")
        print(f"    Before: Choseong/Jungseong: {stats['before_choseong_jungseong']:>8,}  Compat: {stats['before_compat_jamo']:>8,}")
        print(f"    After:  Choseong/Jungseong: {stats['after_choseong_jungseong']:>8,}  Compat: {stats['after_compat_jamo']:>8,}")

    print(f"\n  TOTAL:")
    print(f"    Choseong/Jungseong removed: {total_before_cj:>8,}")
    print(f"    Compatibility Jamo added:   {total_after_compat:>8,}")
    print("=" * 65)

    # Verify: no Choseong/Jungseong should remain (in mapped range)
    remaining = 0
    for path in DATA_FILES:
        text = path.read_text(encoding="utf-8") if not args.dry_run else ""
        for ch in text:
            if ch in JAMO_FIX_MAP:
                remaining += 1
    if not args.dry_run:
        if remaining == 0:
            print("\nVerification: No mapped Choseong/Jungseong characters remain. OK")
        else:
            print(f"\nWARNING: {remaining} mapped characters still remain!")

    print("\nDone.")


if __name__ == "__main__":
    main()
