#!/usr/bin/env python3
"""
Phase 1 — Data pipeline v2 for dalbitalba.

Replaces the old budget-cut-to-24k SFT path. Does NOT use any budget cap;
all 67k raw records are carried through filters, then length-stratified
oversample is applied so rare-but-load-bearing buckets (lg/xl/xxl and
digit/english-rich samples) are 2× weighted in the training mix.

Pipeline:
  1. Read raw JSON crawl
  2. PII scrub (NFKC normalize + immediate Compatibility Jamo restore so the
     pipeline emits a single, internally-consistent code-point convention
     for both phase1 output and downstream tokenization)
      - 주민등록번호 (RRN with checksum)    -> [주민번호]
      - 휴대전화/전화번호 (NFKC)            -> [전화번호]
      - 사업자등록번호                       -> [사업자번호]
      - 계좌번호 (bank proximity)            -> [계좌번호]
      - 이메일                               -> [이메일]
      - URL                                  -> [URL]
  3. Minor-proximity HARD filter (drops the whole record)
  4. Spam / ad / empty filter
  5. Thread-level near-duplicate removal (Jaccard >= 0.85 on char 4-grams
     within the same thread)
  6. Optional global MinHash near-dedup across all threads (catches
     cross-thread operator templates that thread-level dedup misses).
  7. Time-based 95/5 split  (by post createdAt if available, else deterministic hash)
  8. Length-bucket rebalance with 2x oversample of lg/xl/xxl buckets and
     digit/english-rich samples
  9. Format:
       - cpt_corpus.v2.jsonl         (raw continuation text)
       - sft_pairs.v2.jsonl          (reply-pair {"post","comment"} rows)
       - mix manifest records 80% raw / 20% pair
 10. Summary written to .planning/calibration/phase1_summary.json

No external deps. Stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------- #
#  PII patterns                                                    #
# ---------------------------------------------------------------- #

RRN_RE = re.compile(
    r"\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])-[1-8]\d{6}"
)
# Korean mobile + landline. Runs AFTER NFKC normalize.
PHONE_RE = re.compile(
    r"\b(?:\+?82[- ]?)?(?:0\d{1,2})[- .]?\d{3,4}[- .]?\d{4}\b"
)
BIZNUM_RE = re.compile(r"\b\d{3}-\d{2}-\d{5}\b")
# Very permissive account pattern restricted to bank-name proximity
BANK_NAMES = (
    r"(?:국민|신한|우리|하나|농협|기업|카카오뱅크|카뱅|케이뱅크|새마을|수협|"
    r"외환|SC제일|씨티|부산|대구|광주|전북|경남|우체국)"
)
ACCOUNT_CONTEXT_RE = re.compile(
    rf"{BANK_NAMES}[^\d]{{0,20}}\d{{3,6}}[-\s]\d{{2,6}}[-\s]\d{{2,8}}(?:[-\s]\d{{2,6}})?"
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\bhttps?://\S+", flags=re.IGNORECASE)


# ---------------------------------------------------------------- #
#  Compatibility Jamo restoration (paired with NFKC normalize)     #
# ---------------------------------------------------------------- #
# NFKC decomposes standalone Compatibility Jamo (U+3131-U+3163, e.g. ㅋ ㅎ ㅠ)
# into Choseong / Jungseong (U+1100-U+11FF, e.g. ᄏ ᅡ).  Qwen3's BPE has very
# different merge evidence for those two ranges, so a corpus that mixes them
# trains rare token IDs inconsistently. We invert the decomposition right
# after NFKC so every record uses a single consistent convention. Composed
# Hangul syllables (U+AC00-U+D7AF) are not in this map and are unaffected.
_CHOSEONG_TO_COMPAT = {
    "ᄀ": "ㄱ", "ᄁ": "ㄲ", "ᄂ": "ㄴ", "ᄃ": "ㄷ",
    "ᄄ": "ㄸ", "ᄅ": "ㄹ", "ᄆ": "ㅁ", "ᄇ": "ㅂ",
    "ᄈ": "ㅃ", "ᄉ": "ㅅ", "ᄊ": "ㅆ", "ᄋ": "ㅇ",
    "ᄌ": "ㅈ", "ᄍ": "ㅉ", "ᄎ": "ㅊ", "ᄏ": "ㅋ",
    "ᄐ": "ㅌ", "ᄑ": "ㅍ", "ᄒ": "ㅎ",
}
_JUNGSEONG_TO_COMPAT = {
    "ᅡ": "ㅏ", "ᅢ": "ㅐ", "ᅣ": "ㅑ", "ᅤ": "ㅒ",
    "ᅥ": "ㅓ", "ᅦ": "ㅔ", "ᅧ": "ㅕ", "ᅨ": "ㅖ",
    "ᅩ": "ㅗ", "ᅪ": "ㅘ", "ᅫ": "ㅙ", "ᅬ": "ㅚ",
    "ᅭ": "ㅛ", "ᅮ": "ㅜ", "ᅯ": "ㅝ", "ᅰ": "ㅞ",
    "ᅱ": "ㅟ", "ᅲ": "ㅠ", "ᅳ": "ㅡ", "ᅴ": "ㅢ",
    "ᅵ": "ㅣ",
}
_JAMO_RESTORE_TRANS = str.maketrans({**_CHOSEONG_TO_COMPAT, **_JUNGSEONG_TO_COMPAT})


def restore_compat_jamo(text: str) -> str:
    """Map U+1100-U+11FF Choseong/Jungseong back to Compatibility Jamo.

    Safe to call on any string: composed Hangul syllables (U+AC00-U+D7AF) are
    untouched because the map only covers U+1100-U+11FF.
    """
    return text.translate(_JAMO_RESTORE_TRANS)


def rrn_checksum_ok(rrn: str) -> bool:
    digits = re.sub(r"\D", "", rrn)
    if len(digits) != 13:
        return False
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    s = sum(int(digits[i]) * weights[i] for i in range(12))
    check = (11 - (s % 11)) % 10
    return check == int(digits[12])


def scrub_pii(text: str) -> tuple[str, Counter]:
    """Returns (scrubbed_text, scrub_counts).

    Atomic step: NFKC normalize → Compatibility Jamo restore. The two are
    inseparable for Korean community speech that is heavy in 초성 markers
    (ㅈㄴ, ㅇㅈ, ㄹㅇ, ㅋㅋ, ㅎㅎ); applying only NFKC silently degrades the
    corpus into Choseong/Jungseong code-points, which downstream BPE handles
    differently from the Compatibility Jamo block.
    """
    counts: Counter = Counter()
    text = unicodedata.normalize("NFKC", text)
    text = restore_compat_jamo(text)

    # RRN with checksum
    def rrn_sub(m: re.Match) -> str:
        if rrn_checksum_ok(m.group(0)):
            counts["rrn"] += 1
            return "[주민번호]"
        return m.group(0)

    text = RRN_RE.sub(rrn_sub, text)

    # Phone
    def phone_sub(m: re.Match) -> str:
        counts["phone"] += 1
        return "[전화번호]"

    text = PHONE_RE.sub(phone_sub, text)

    # Business registration
    def biz_sub(m: re.Match) -> str:
        counts["bizno"] += 1
        return "[사업자번호]"

    text = BIZNUM_RE.sub(biz_sub, text)

    # Bank account (context-gated)
    def acct_sub(m: re.Match) -> str:
        counts["account"] += 1
        return m.group(0).split()[0] + " [계좌번호]"

    text = ACCOUNT_CONTEXT_RE.sub(acct_sub, text)

    # Email
    def email_sub(m: re.Match) -> str:
        counts["email"] += 1
        return "[이메일]"

    text = EMAIL_RE.sub(email_sub, text)

    # URL
    def url_sub(m: re.Match) -> str:
        counts["url"] += 1
        return "[URL]"

    text = URL_RE.sub(url_sub, text)

    return text, counts


# ---------------------------------------------------------------- #
#  Minor-proximity hard filter                                     #
# ---------------------------------------------------------------- #

MINOR_TERMS = (
    "미성년자", "미성년", "청소년", "초등학생", "초딩",
    "중학생", "중딩", "고등학생", "고딩", "여중생", "남중생",
    "여고생", "남고생",
)
SEXUAL_TERMS = (
    "섹스", "성관계", "야동", "자위", "조건만남", "원나잇", "원나이트",
    "성매매", "오피", "풀살롱", "안마방", "대딸", "유사성행위",
    "오랄", "구강", "질내", "항문", "삽입", "자지", "보지",
)

MINOR_RE = re.compile("|".join(re.escape(t) for t in MINOR_TERMS))
SEXUAL_RE = re.compile("|".join(re.escape(t) for t in SEXUAL_TERMS))


def minor_proximity_block(text: str, window: int = 60) -> bool:
    """True if a minor-age term and a sexual term co-occur within `window`
    chars. Whole record must be dropped."""
    if not MINOR_RE.search(text):
        return False
    minor_positions = [m.start() for m in MINOR_RE.finditer(text)]
    sexual_positions = [m.start() for m in SEXUAL_RE.finditer(text)]
    if not sexual_positions:
        return False
    for mp in minor_positions:
        for sp in sexual_positions:
            if abs(mp - sp) <= window:
                return True
    return False


# ---------------------------------------------------------------- #
#  Spam / ad / near-dup                                            #
# ---------------------------------------------------------------- #

SPAM_PATTERNS = (
    r"텔레그램\s*[:：]?\s*@",
    r"카톡\s*[:：]?\s*[A-Za-z0-9_\-]{4,}",
    r"(광고|홍보|이벤트)\s*문의",
    r"(바로가기|입장하기|접속하기)\s*\[URL\]",
)
SPAM_RE = re.compile("|".join(SPAM_PATTERNS))


def char_ngrams(text: str, n: int = 4) -> set:
    stripped = re.sub(r"\s+", "", text)
    return {stripped[i : i + n] for i in range(max(0, len(stripped) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------- #
#  Raw reader                                                      #
# ---------------------------------------------------------------- #


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def iter_posts(raw_dir: str):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
    for fp in files:
        with open(fp, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        arr = obj if isinstance(obj, list) else (
            obj.get("posts") or obj.get("items") or []
        )
        if not isinstance(arr, list):
            continue
        for post in arr:
            if isinstance(post, dict):
                yield post, fp


# ---------------------------------------------------------------- #
#  Length buckets & oversample weights                             #
# ---------------------------------------------------------------- #

LENGTH_BUCKETS = [
    ("xs", 0, 19),
    ("sm", 20, 49),
    ("md", 50, 99),
    ("lg", 100, 199),
    ("xl", 200, 499),
    ("xxl", 500, 10**9),
]

TARGET_DIST = {
    "xs": 0.26,
    "sm": 0.35,
    "md": 0.18,
    "lg": 0.09,
    "xl": 0.11,
    "xxl": 0.01,
}


def length_bucket(n: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return name
    return "unk"


def is_digit_or_english_rich(text: str) -> bool:
    if not text:
        return False
    digits = sum(1 for c in text if c.isdigit())
    eng = sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    dr = digits / len(text)
    er = eng / len(text)
    return dr > 0.08 or er > 0.02


# ---------------------------------------------------------------- #
#  Main                                                            #
# ---------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="/mnt/c/Users/mapdr/Desktop/queenalba-crawler/crawled-data-v2",
    )
    parser.add_argument("--out-cpt", default="cpt_corpus.v2.jsonl")
    parser.add_argument("--out-sft", default="sft_pairs.v2.jsonl")
    parser.add_argument(
        "--summary", default=".planning/calibration/phase1_summary.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-chars", type=int, default=5, help="Drop records shorter than this"
    )
    parser.add_argument(
        "--dedup-jaccard",
        type=float,
        default=0.85,
        help="Jaccard threshold for near-duplicate comments within a thread",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Fraction for time-based val split"
    )
    parser.add_argument(
        "--minhash-dedup",
        action="store_true",
        default=True,
        help=(
            "Apply global MinHash near-dedup across threads after thread-level "
            "Jaccard dedup. Catches operator/ad templates copy-pasted across "
            "many threads (P0-1 hardening, see docs/PAPER_GRADE_ANALYSIS_*)."
        ),
    )
    parser.add_argument(
        "--no-minhash-dedup",
        dest="minhash_dedup",
        action="store_false",
        help="Disable the global MinHash dedup pass.",
    )
    parser.add_argument(
        "--minhash-bands",
        type=int,
        default=32,
        help="LSH bands (32 × 4 ≈ Jaccard 0.8 acceptance with 128 perms).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    posts = list(iter_posts(args.raw_dir))
    print(f"[phase1] raw posts: {len(posts):,}")

    # ----- collect records with PII scrub + minor filter + spam filter -----
    raw_records: list[dict] = []
    pii_totals: Counter = Counter()
    stats = Counter()

    for post, _ in posts:
        post_id = str(post.get("id") or post.get("post_id") or "")
        title = (post.get("title") or "").strip()
        body = (post.get("content") or "").strip()
        date_raw = (post.get("date") or post.get("createdAt") or "").strip()
        try:
            date = datetime.fromisoformat(date_raw.replace("Z", "+00:00")) if date_raw else None
        except Exception:
            date = None
        # deterministic fallback
        t_key = post_id or str(
            stable_hash((title + "|" + body) + f":{args.seed}")
        )

        combined = (title + "\n" + body).strip()
        if combined:
            cleaned, pii = scrub_pii(combined)
            pii_totals.update(pii)
            if minor_proximity_block(cleaned):
                stats["drop_minor"] += 1
            elif len(cleaned) < args.min_chars:
                stats["drop_short"] += 1
            elif SPAM_RE.search(cleaned):
                stats["drop_spam"] += 1
            else:
                raw_records.append(
                    {
                        "text": cleaned,
                        "kind": "post",
                        "thread_key": t_key,
                        "post_title_clean": (
                            scrub_pii(title)[0] if title else ""
                        ),
                        "post_body_clean": scrub_pii(body)[0] if body else "",
                        "source_id": post_id,
                        "date": date.isoformat() if date else None,
                        "source_field": "post",
                    }
                )
                stats["keep_post"] += 1
        else:
            stats["drop_empty_post"] += 1

        for comment in post.get("comments") or []:
            if not isinstance(comment, dict):
                continue
            c_text = (comment.get("content") or "").strip()
            if not c_text:
                stats["drop_empty_comment"] += 1
                continue
            cleaned_c, pii_c = scrub_pii(c_text)
            pii_totals.update(pii_c)
            if minor_proximity_block(cleaned_c):
                stats["drop_minor"] += 1
                continue
            if len(cleaned_c) < args.min_chars:
                stats["drop_short"] += 1
                continue
            if SPAM_RE.search(cleaned_c):
                stats["drop_spam"] += 1
                continue
            raw_records.append(
                {
                    "text": cleaned_c,
                    "kind": "comment",
                    "thread_key": t_key,
                    "post_title_clean": (
                        scrub_pii(title)[0] if title else ""
                    ),
                    "post_body_clean": scrub_pii(body)[0] if body else "",
                    "source_id": str(comment.get("id") or post_id),
                    "date": (
                        date.isoformat() if date else None
                    ),  # inherit post date as best-effort ordering key
                    "source_field": "comment",
                }
            )
            stats["keep_comment"] += 1

    print(f"[phase1] after PII+minor+spam: {len(raw_records):,} records")
    print(f"[phase1] drop stats: {dict(stats)}")
    print(f"[phase1] pii scrubbed: {dict(pii_totals)}")

    # ----- Thread-level near-duplicate removal (comments only) -----
    by_thread: dict[str, list[dict]] = defaultdict(list)
    for r in raw_records:
        by_thread[r["thread_key"]].append(r)

    deduped: list[dict] = []
    dedup_dropped = 0
    for t_key, recs in by_thread.items():
        kept_ngrams: list[set] = []
        for r in recs:
            if r["kind"] == "post":
                deduped.append(r)
                kept_ngrams.append(char_ngrams(r["text"]))
                continue
            ng = char_ngrams(r["text"])
            dup = False
            for prev in kept_ngrams:
                if jaccard(ng, prev) >= args.dedup_jaccard:
                    dup = True
                    break
            if dup:
                dedup_dropped += 1
            else:
                deduped.append(r)
                kept_ngrams.append(ng)
    print(
        f"[phase1] after dedup(Jaccard>={args.dedup_jaccard}): "
        f"{len(deduped):,} (dropped {dedup_dropped:,} near-dup comments)"
    )

    # ----- Global MinHash dedup across threads (P0-1) -----
    minhash_stats: dict | None = None
    if args.minhash_dedup:
        try:
            from dedup_minhash import dedup_records as _minhash_dedup
        except ImportError:
            # phase1 sometimes runs from repo root, sometimes from scripts/.
            # Add scripts/ to sys.path explicitly to be robust.
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
            from dedup_minhash import dedup_records as _minhash_dedup
        before = len(deduped)
        deduped, minhash_stats = _minhash_dedup(
            deduped,
            field="text",
            num_perm=128,
            bands=args.minhash_bands,
            shingle_n=5,
            seed=args.seed,
        )
        print(
            f"[phase1] after MinHash global dedup: {len(deduped):,} "
            f"(dropped {before - len(deduped):,} cross-thread duplicates; "
            f"biggest cluster={minhash_stats['biggest_cluster']}, "
            f"clusters≥5={minhash_stats['clusters_5plus']})"
        )

    # ----- Time-based 95/5 split -----
    dated = [r for r in deduped if r["date"]]
    undated = [r for r in deduped if not r["date"]]
    dated.sort(key=lambda r: r["date"])
    split_at = int(len(dated) * (1 - args.val_ratio))
    train_records = dated[:split_at]
    val_records = dated[split_at:]
    # undated records: deterministic hash split 95/5
    for r in undated:
        h = stable_hash(r.get("source_id", "") + f":{args.seed}")
        if (h % 1000) < int(args.val_ratio * 1000):
            val_records.append(r)
        else:
            train_records.append(r)
    print(
        f"[phase1] split: train={len(train_records):,} val={len(val_records):,}"
    )

    # ----- Length-bucket rebalance with 2x oversample of lg/xl/xxl + rich -----
    def sample_weight(r: dict) -> float:
        b = length_bucket(len(r["text"]))
        w = 1.0
        if b in {"lg", "xl", "xxl"}:
            w *= 2.0
        if is_digit_or_english_rich(r["text"]):
            w *= 2.0
        return w

    # Expand training set by weighted duplication (deterministic)
    expanded: list[dict] = []
    for r in train_records:
        w = sample_weight(r)
        # deterministic whole-part + prob part
        whole = int(w)
        frac = w - whole
        for i in range(whole):
            expanded.append(r)
        if frac > 0:
            # deterministic: use stable hash of source_id for tie-break
            h = stable_hash(str(r.get("source_id", "")) + f":frac:{args.seed}")
            if (h % 1000) / 1000 < frac:
                expanded.append(r)

    rng.shuffle(expanded)

    # measure bucket distribution
    bucket_counts: Counter = Counter()
    for r in expanded:
        bucket_counts[length_bucket(len(r["text"]))] += 1
    bucket_share = {
        k: bucket_counts[k] / max(1, len(expanded)) for k, _, _ in LENGTH_BUCKETS
    }

    print(
        f"[phase1] expanded training size: {len(expanded):,} "
        f"(raw weight = 1.0 baseline)"
    )
    print(f"[phase1] bucket share: {bucket_share}")
    print(f"[phase1] target dist  : {TARGET_DIST}")

    # ----- Write cpt_corpus.v2 -----
    cpt_out = Path(args.out_cpt)
    cpt_count = 0
    with cpt_out.open("w", encoding="utf-8") as h:
        for r in expanded:
            row = {
                "text": r["text"],
                "kind": r["kind"],
                "source_id": r.get("source_id"),
                "source_field": r.get("source_field"),
                "length_bucket": length_bucket(len(r["text"])),
            }
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
            cpt_count += 1
    print(f"[phase1] wrote {cpt_out} ({cpt_count:,} rows)")

    # ----- Write sft_pairs.v2  (post, comment) reply pairs -----
    # Use train_records (non-expanded) to build canonical pairs.
    # The 80/20 mix is handled at training time by the loader; here we keep
    # both files available.
    pair_rows = []
    posts_by_thread: dict[str, dict] = {}
    for r in train_records:
        if r["kind"] == "post":
            posts_by_thread[r["thread_key"]] = r

    for r in train_records:
        if r["kind"] != "comment":
            continue
        post = posts_by_thread.get(r["thread_key"])
        if not post:
            # synthesize minimal prompt from title/body if present
            post_text = r.get("post_title_clean") or r.get("post_body_clean") or ""
        else:
            post_text = (post["text"]).strip()
        if not post_text:
            continue
        pair_rows.append(
            {
                "post": post_text,
                "comment": r["text"],
                "thread_key": r["thread_key"],
                "source_id": r.get("source_id"),
                "length_bucket": length_bucket(len(r["text"])),
            }
        )

    # apply same oversample weight to pairs (on comment text)
    expanded_pairs: list[dict] = []
    for pr in pair_rows:
        # weight on comment length bucket
        fake = {"text": pr["comment"], "source_id": pr.get("source_id")}
        w = sample_weight(fake)
        whole = int(w)
        frac = w - whole
        for _ in range(whole):
            expanded_pairs.append(pr)
        if frac > 0:
            h = stable_hash(
                str(pr.get("source_id", "")) + f":pairfrac:{args.seed}"
            )
            if (h % 1000) / 1000 < frac:
                expanded_pairs.append(pr)
    rng.shuffle(expanded_pairs)

    sft_out = Path(args.out_sft)
    with sft_out.open("w", encoding="utf-8") as h:
        for pr in expanded_pairs:
            h.write(json.dumps(pr, ensure_ascii=False) + "\n")
    print(f"[phase1] wrote {sft_out} ({len(expanded_pairs):,} rows)")

    # ----- Write val set -----
    val_out = Path("val_set.v2.jsonl")
    with val_out.open("w", encoding="utf-8") as h:
        for r in val_records:
            row = {
                "text": r["text"],
                "kind": r["kind"],
                "source_id": r.get("source_id"),
                "source_field": r.get("source_field"),
                "length_bucket": length_bucket(len(r["text"])),
                "date": r.get("date"),
            }
            h.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[phase1] wrote {val_out} ({len(val_records):,} rows)")

    # ----- Summary -----
    summary = {
        "input": {
            "raw_dir": args.raw_dir,
            "raw_posts": len(posts),
        },
        "pii_scrubbed": dict(pii_totals),
        "filter_stats": dict(stats),
        "dedup": {
            "threshold": args.dedup_jaccard,
            "dropped_near_dup": dedup_dropped,
            "minhash": minhash_stats,
        },
        "split": {
            "train_records": len(train_records),
            "val_records": len(val_records),
            "val_ratio": args.val_ratio,
        },
        "bucket_share_expanded_cpt": bucket_share,
        "bucket_target": TARGET_DIST,
        "outputs": {
            "cpt_corpus.v2.jsonl": cpt_count,
            "sft_pairs.v2.jsonl": len(expanded_pairs),
            "val_set.v2.jsonl": len(val_records),
        },
        "mix_policy": {"raw_continuation": 0.8, "reply_pair": 0.2},
        "seed": args.seed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    sum_path = Path(args.summary)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    with sum_path.open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)
    print(f"[phase1] wrote {sum_path}")


if __name__ == "__main__":
    main()
