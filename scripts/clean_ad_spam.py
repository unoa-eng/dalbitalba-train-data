#!/usr/bin/env python3
"""
clean_ad_spam.py — Remove recruiter/venue ad spam from training data.

Targets three contamination vectors the original SPAM_RE missed:
  1. Expanded kakao/ad patterns (카카오 ha1922, 카톡아이디 w21w, etc.)
  2. Cross-thread duplicate text (>=3 identical copies after whitespace norm)
  3. Rows with zero complete Hangul syllables (jamo-only / pure numbers)

Usage:
    python scripts/clean_ad_spam.py [--dry-run]
"""

import argparse
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── root paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILES = {
    "cpt": PROJECT_ROOT / "cpt_corpus.v2.jsonl",
    "sft": PROJECT_ROOT / "sft_pairs.v2.jsonl",
    "val": PROJECT_ROOT / "val_set.v2.jsonl",
}
BACKUP_DIR = PROJECT_ROOT / "archive" / "dataset-backups"

# ── 1. Expanded ad-spam patterns ────────────────────────────────────────────
AD_PATTERNS = [
    # kakao ID variations  (카카오톡아이디, 카카오 ID, 카톡아이디 …)
    r'카카오\s*(?:톡\s*)?(?:아이디|ID)?\s*[:\s]\s*[A-Za-z0-9_\-]{3,}',
    r'카톡\s*(?:아이디|ID)?\s*[:\s]\s*[A-Za-z0-9_\-]{3,}',
    # bare "카카오 ha1922" / "카톡 w21w"
    r'카카오\s+[a-zA-Z][a-zA-Z0-9]{2,}',
    r'카톡\s+[a-zA-Z][a-zA-Z0-9]{2,}',
    # Kakao ID after Kakao: pattern (catches "Kakao ID:uho3265")
    r'[Kk]akao\s*(?:ID)?\s*[:：]\s*[A-Za-z0-9_\-]{3,}',
    # PII placeholder + kakao in proximity (either order)
    r'\[전화번호\].*카[카톡]',
    r'카[카톡].*\[전화번호\]',
    # common recruiter phrases
    r'밀빵\s*확실',
    r'밀빵\s*가능',
    r'하루\s*평균\s*\d+\s*방',           # "하루평균 10방이상"
    r'문의\s*(주|하)\s*(세요|십시오|시면)',  # solicitation
    r'풀상주\s*풀케어',
    r'출근\s*문의',
    r'스타트톡\s*개수톡',
    r'[1-3]부\s*\d+인\s*\d+조',          # "1부~3부 2인1조"
    r'G팀\s*\d+방',                       # "G팀 10방이상"
    r'\d+방\s*팀사장',                    # "800-1000방 팀사장"
]
AD_RE = re.compile('|'.join(AD_PATTERNS), re.IGNORECASE)

# Complete Hangul syllable range
HANGUL_SYLLABLE_RE = re.compile(r'[\uac00-\ud7af]')

# ── helpers ─────────────────────────────────────────────────────────────────

def normalize_ws(text: str) -> str:
    """Collapse all whitespace to single spaces and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def has_hangul_syllable(text: str) -> bool:
    """Return True if text contains at least one complete Hangul syllable."""
    return bool(HANGUL_SYLLABLE_RE.search(text))


def extract_texts(row: dict, kind: str) -> list[str]:
    """Return a list of text fields to check, depending on file kind."""
    if kind == "sft":
        return [row.get("post", ""), row.get("comment", "")]
    else:
        return [row.get("text", "")]


def row_matches_ad(row: dict, kind: str) -> str | None:
    """Return the first matching pattern description, or None."""
    for t in extract_texts(row, kind):
        m = AD_RE.search(t)
        if m:
            return m.group()
    return None


def row_has_no_hangul(row: dict, kind: str) -> bool:
    """True if none of the text fields contain a complete Hangul syllable."""
    for t in extract_texts(row, kind):
        if has_hangul_syllable(t):
            return False
    return True


def build_text_key(row: dict, kind: str) -> str:
    """Build a normalized key for duplicate detection."""
    parts = extract_texts(row, kind)
    return normalize_ws(" ".join(parts))


# ── main pipeline ───────────────────────────────────────────────────────────

def process_file(path: Path, kind: str, dup_counts: Counter, dry_run: bool):
    """Process one JSONL file. Returns (kept, stats_dict)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total = len(rows)
    stats = {
        "total": total,
        "drop_ad_pattern": 0,
        "drop_cross_dup": 0,
        "drop_no_hangul": 0,
        "kept": 0,
        "ad_pattern_details": Counter(),
    }

    kept = []
    for row in rows:
        # 1. Ad pattern check
        ad_match = row_matches_ad(row, kind)
        if ad_match:
            stats["drop_ad_pattern"] += 1
            # Bucket by the first pattern token for reporting
            stats["ad_pattern_details"][ad_match[:30]] += 1
            continue

        # 2. Cross-thread duplicate check
        key = build_text_key(row, kind)
        if dup_counts[key] >= 3:
            stats["drop_cross_dup"] += 1
            continue

        # 3. Hangul syllable check
        if row_has_no_hangul(row, kind):
            stats["drop_no_hangul"] += 1
            continue

        kept.append(row)

    stats["kept"] = len(kept)

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return kept, stats


def main():
    parser = argparse.ArgumentParser(description="Clean ad spam from training data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts without modifying files")
    args = parser.parse_args()

    # Verify all files exist
    for name, path in DATA_FILES.items():
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)

    # ── backup ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = BACKUP_DIR / f"budget30-adclean-{ts}"
    if not args.dry_run:
        backup_path.mkdir(parents=True, exist_ok=True)
        for name, path in DATA_FILES.items():
            shutil.copy2(path, backup_path / path.name)
        print(f"Backed up to {backup_path}")
    else:
        print("[DRY RUN] — no files will be modified")

    # ── Pass 1: build global duplicate frequency map ────────────────────
    print("\nPass 1: Building cross-thread duplicate frequency map …")
    global_text_counts: Counter = Counter()
    for name, path in DATA_FILES.items():
        kind = name  # "cpt", "sft", "val"
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = build_text_key(row, kind)
                global_text_counts[key] += 1

    dup_keys = {k for k, v in global_text_counts.items() if v >= 3}
    print(f"  Unique texts: {len(global_text_counts):,}")
    print(f"  Texts appearing >=3 times: {len(dup_keys):,}")

    # ── Pass 2: filter each file ────────────────────────────────────────
    print("\nPass 2: Filtering …\n")
    all_stats = {}
    for name, path in DATA_FILES.items():
        kind = name
        print(f"Processing {path.name} …")
        _, stats = process_file(path, kind, global_text_counts, args.dry_run)
        all_stats[name] = stats

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CLEANING SUMMARY")
    print("=" * 65)
    total_before = 0
    total_after = 0
    total_ad = 0
    total_dup = 0
    total_hangul = 0

    for name in ("cpt", "sft", "val"):
        s = all_stats[name]
        total_before += s["total"]
        total_after += s["kept"]
        total_ad += s["drop_ad_pattern"]
        total_dup += s["drop_cross_dup"]
        total_hangul += s["drop_no_hangul"]

        removed = s["total"] - s["kept"]
        pct = (removed / s["total"] * 100) if s["total"] else 0
        print(f"\n  {DATA_FILES[name].name}:")
        print(f"    Before:           {s['total']:>8,}")
        print(f"    Ad pattern:       {s['drop_ad_pattern']:>8,}")
        print(f"    Cross-thread dup: {s['drop_cross_dup']:>8,}")
        print(f"    No Hangul:        {s['drop_no_hangul']:>8,}")
        print(f"    After:            {s['kept']:>8,}")
        print(f"    Removed:          {removed:>8,}  ({pct:.2f}%)")

    total_removed = total_before - total_after
    pct_total = (total_removed / total_before * 100) if total_before else 0
    print(f"\n  TOTAL:")
    print(f"    Before:           {total_before:>8,}")
    print(f"    Ad pattern:       {total_ad:>8,}")
    print(f"    Cross-thread dup: {total_dup:>8,}")
    print(f"    No Hangul:        {total_hangul:>8,}")
    print(f"    After:            {total_after:>8,}")
    print(f"    Removed:          {total_removed:>8,}  ({pct_total:.2f}%)")
    print("=" * 65)

    # Show top ad-pattern matches across all files
    print("\nTop ad-pattern match snippets (all files combined):")
    combined_ad = Counter()
    for s in all_stats.values():
        combined_ad.update(s["ad_pattern_details"])
    for snippet, cnt in combined_ad.most_common(20):
        print(f"  {cnt:>5}x  {snippet}")

    if not args.dry_run:
        print(f"\nBackup location: {backup_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
