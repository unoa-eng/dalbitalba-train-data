#!/usr/bin/env python3
"""
Independent verification of cycle11 live-inplace-regen pilot.
Checks: Integrity, Verbatim Overlap, Banned Tokens, Encoding, Register Stats.
"""

import json
import re
import sys
from collections import defaultdict

SRC_PATH = "/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen/pilot_in/sample.json"
REGEN_PATH = "/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen/pilot_out/pilot_regen.json"

# ─── helpers ────────────────────────────────────────────────────────────────

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def eojul_tokens(text):
    """Split on whitespace → list of tokens (어절)."""
    return text.split() if text else []

def longest_common_run(a_tokens, b_tokens):
    """Longest contiguous run of identical tokens between two lists."""
    if not a_tokens or not b_tokens:
        return 0
    # Build suffix table O(n*m) – fine for small texts
    n, m = len(a_tokens), len(b_tokens)
    best = 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a_tokens[i-1] == b_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > best:
                    best = dp[i][j]
    return best

def all_text_parts(post, include_title=True):
    """Yield (label, text) for title, body, and each comment."""
    if include_title:
        yield "title", post.get("title", "")
    yield "body", post.get("body", "")
    for c in post.get("comments", []):
        yield f"comment[{c['id'][:8]}]", c.get("body", "")

# ─── Check 1: INTEGRITY ──────────────────────────────────────────────────────

def check_integrity(src_posts, regen_posts):
    print("\n=== CHECK 1: INTEGRITY ===")
    failures = []

    src_by_id  = {p["id"]: p for p in src_posts}
    regen_by_id = {p["id"]: p for p in regen_posts}

    # Post-level id sets
    src_ids   = set(src_by_id)
    regen_ids = set(regen_by_id)

    missing_in_regen = src_ids - regen_ids
    extra_in_regen   = regen_ids - src_ids

    if missing_in_regen:
        failures.append(f"Posts missing in regen: {missing_in_regen}")
    if extra_in_regen:
        failures.append(f"Extra posts in regen not in source: {extra_in_regen}")

    for pid in src_ids & regen_ids:
        sp = src_by_id[pid]
        rp = regen_by_id[pid]

        src_comment_count = sp.get("comment_count", len(sp.get("comments", [])))
        regen_comment_count = len(rp.get("comments", []))
        if src_comment_count != regen_comment_count:
            failures.append(
                f"Post {pid[:8]}: comment count mismatch "
                f"src={src_comment_count} regen={regen_comment_count}"
            )

        src_cids = {c["id"]: c for c in sp.get("comments", [])}
        regen_cids = {c["id"]: c for c in rp.get("comments", [])}

        missing_comments = set(src_cids) - set(regen_cids)
        extra_comments   = set(regen_cids) - set(src_cids)
        if missing_comments:
            failures.append(f"Post {pid[:8]}: comments missing in regen: {[x[:8] for x in missing_comments]}")
        if extra_comments:
            failures.append(f"Post {pid[:8]}: extra comments in regen: {[x[:8] for x in extra_comments]}")

        for cid in set(src_cids) & set(regen_cids):
            src_pid = src_cids[cid].get("parent_id")
            regen_pid = regen_cids[cid].get("parent_id")
            if src_pid != regen_pid:
                failures.append(
                    f"Post {pid[:8]} comment {cid[:8]}: parent_id mismatch "
                    f"src={src_pid} regen={regen_pid}"
                )

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"PASS — {len(src_ids)} posts, all comment ids + parent_ids match")

    return len(failures) == 0

# ─── Check 2: VERBATIM OVERLAP ───────────────────────────────────────────────

def check_verbatim(src_posts, regen_posts):
    print("\n=== CHECK 2: VERBATIM OVERLAP (어절 runs) ===")
    FLAG_THRESHOLD = 5

    src_by_id   = {p["id"]: p for p in src_posts}
    regen_by_id = {p["id"]: p for p in regen_posts}

    all_runs = []
    flags = []

    for pid in src_by_id:
        if pid not in regen_by_id:
            continue
        sp = src_by_id[pid]
        rp = regen_by_id[pid]

        # title
        for field in ("title",):
            s_tok = eojul_tokens(sp.get(field, ""))
            r_tok = eojul_tokens(rp.get(field, ""))
            run = longest_common_run(s_tok, r_tok)
            all_runs.append(run)
            if run >= FLAG_THRESHOLD:
                flags.append((pid[:8], field, run, sp.get(field,""), rp.get(field,"")))

        # body
        for field in ("body",):
            s_tok = eojul_tokens(sp.get(field, ""))
            r_tok = eojul_tokens(rp.get(field, ""))
            run = longest_common_run(s_tok, r_tok)
            all_runs.append(run)
            if run >= FLAG_THRESHOLD:
                flags.append((pid[:8], field, run, sp.get(field,"")[:80], rp.get(field,"")[:80]))

        # comments (match by id)
        src_cmap   = {c["id"]: c for c in sp.get("comments", [])}
        regen_cmap = {c["id"]: c for c in rp.get("comments", [])}
        for cid in src_cmap:
            if cid not in regen_cmap:
                continue
            s_tok = eojul_tokens(src_cmap[cid].get("body",""))
            r_tok = eojul_tokens(regen_cmap[cid].get("body",""))
            run = longest_common_run(s_tok, r_tok)
            all_runs.append(run)
            if run >= FLAG_THRESHOLD:
                flags.append((
                    pid[:8], f"comment[{cid[:8]}]", run,
                    src_cmap[cid].get("body","")[:80],
                    regen_cmap[cid].get("body","")[:80]
                ))

    max_run  = max(all_runs) if all_runs else 0
    mean_run = sum(all_runs) / len(all_runs) if all_runs else 0

    print(f"  Max run observed : {max_run} 어절")
    print(f"  Mean run         : {mean_run:.2f} 어절")
    print(f"  Fields checked   : {len(all_runs)}")

    if flags:
        print(f"FAIL — {len(flags)} field(s) with run >= {FLAG_THRESHOLD}:")
        for pid8, field, run, s_snip, r_snip in flags:
            print(f"  post={pid8} field={field} run={run}")
            print(f"    SRC  : {s_snip!r}")
            print(f"    REGEN: {r_snip!r}")
    else:
        print(f"PASS — no runs >= {FLAG_THRESHOLD} 어절")

    return len(flags) == 0

# ─── Check 3: BANNED TOKENS ──────────────────────────────────────────────────

# Patterns — we use word-ish boundaries for ambiguous ones.
# For Korean we can't use \b directly; use lookahead/lookbehind for non-word chars.

KR_WORD_BOUND = r'(?<![가-힣\w]){}(?![가-힣\w])'

BANNED_PATTERNS = [
    # Venue/brand — substring match OK for clear names
    (re.compile(r'세이렌'), "세이렌"),
    (re.compile(r'유앤미'), "유앤미"),
    (re.compile(r'도파민'), "도파민"),
    # 도파 as standalone token (not part of 도파민)
    (re.compile(KR_WORD_BOUND.format(r'도파')), "도파(standalone)"),
    (re.compile(r'보스턴'), "보스턴"),
    (re.compile(r'엘리트'), "엘리트"),
    (re.compile(r'퍼펙트'), "퍼펙트"),
    (re.compile(r'사라(?=[^하-힣]|$)'), "사라(venue)"),  # avoid 사라지다 etc.
    (re.compile(r'달토'), "달토"),
    (re.compile(r'정점'), "정점"),
    (re.compile(r'썸데이'), "썸데이"),
    (re.compile(r'하퍼'), "하퍼"),
    (re.compile(r'루나'), "루나"),
    (re.compile(r'퀸알바'), "퀸알바"),
    # 퀸 as standalone
    (re.compile(KR_WORD_BOUND.format(r'퀸')), "퀸(standalone)"),
    (re.compile(r'queen', re.IGNORECASE), "queen"),
    # Person remnants
    (re.compile(r'우지호'), "우지호"),
    # 혁 as standalone token
    (re.compile(KR_WORD_BOUND.format(r'혁')), "혁(standalone)"),
    (re.compile(r'\[실명\]'), "[실명]"),
    # Contact/ad
    (re.compile(r'카톡'), "카톡"),
    (re.compile(r'카카오'), "카카오"),
    (re.compile(r'텔레'), "텔레"),
    (re.compile(r'오픈채팅'), "오픈채팅"),
    (re.compile(r'명함'), "명함"),
    (re.compile(r'면접비'), "면접비"),
    (re.compile(r'01[016-9]\d{6,8}'), "phone number"),
    (re.compile(r'http'), "http"),
]

def check_banned_tokens(regen_posts):
    print("\n=== CHECK 3: BANNED TOKENS ===")
    hits = []

    for post in regen_posts:
        pid = post["id"]
        # Collect (label, text) pairs
        texts = [("title", post.get("title", "")), ("body", post.get("body", ""))]
        for c in post.get("comments", []):
            texts.append((f"comment[{c['id'][:8]}]", c.get("body", "")))

        for label, text in texts:
            if not text:
                continue
            for pattern, name in BANNED_PATTERNS:
                m = pattern.search(text)
                if m:
                    start = max(0, m.start() - 15)
                    end   = min(len(text), m.end() + 15)
                    snippet = text[start:end]
                    hits.append((pid[:8], label, name, snippet))

    if hits:
        print(f"FAIL — {len(hits)} banned token hit(s):")
        for pid8, label, name, snippet in hits:
            print(f"  post={pid8} [{label}] token={name!r}  snippet=…{snippet!r}…")
    else:
        print("PASS — zero banned tokens found")

    return len(hits) == 0

# ─── Check 4: ENCODING ───────────────────────────────────────────────────────

OLD_HANGUL_RANGES = [
    (0x1100, 0x11FF),   # Hangul Jamo (첫가끝)
    (0xA960, 0xA97F),   # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),   # Hangul Jamo Extended-B
]
# Note: compatibility jamo U+3130–U+318F (ㅋ, ㅠ, etc.) are ALLOWED.

def check_encoding(regen_posts):
    print("\n=== CHECK 4: ENCODING (old Hangul jamo) ===")
    hits = []

    for post in regen_posts:
        pid = post["id"]
        texts = [("title", post.get("title", "")), ("body", post.get("body", ""))]
        for c in post.get("comments", []):
            texts.append((f"comment[{c['id'][:8]}]", c.get("body", "")))

        for label, text in texts:
            if not text:
                continue
            for i, ch in enumerate(text):
                cp = ord(ch)
                for lo, hi in OLD_HANGUL_RANGES:
                    if lo <= cp <= hi:
                        ctx_start = max(0, i - 5)
                        ctx_end   = min(len(text), i + 6)
                        hits.append((pid[:8], label, hex(cp), ch, text[ctx_start:ctx_end]))
                        break

    if hits:
        print(f"FAIL — {len(hits)} old-Hangul jamo character(s):")
        for pid8, label, cphex, ch, ctx in hits:
            print(f"  post={pid8} [{label}] U+{cphex.upper().replace('0X','')} '{ch}' context={ctx!r}")
    else:
        print("PASS — no old-Hangul jamo (U+1100-11FF / A960-D7FF) found")

    return len(hits) == 0

# ─── Check 5: REGISTER STATS ─────────────────────────────────────────────────

JONDAEMAL_RE = re.compile(
    r'(요|니다|세요|네요|아요|어요|죠)\s*$'
)

def register_stats(posts, label):
    comment_bodies = []
    title_lens = []
    body_lens  = []
    jondae = 0
    banmal = 0

    for post in posts:
        title_lens.append(len(post.get("title", "")))
        body_lens.append(len(post.get("body", "")))
        for c in post.get("comments", []):
            body = c.get("body", "")
            comment_bodies.append(body)
            if JONDAEMAL_RE.search(body.strip()):
                jondae += 1
            else:
                banmal += 1

    total = len(comment_bodies)
    mean_comment = sum(len(b) for b in comment_bodies) / total if total else 0
    mean_title   = sum(title_lens) / len(title_lens) if title_lens else 0
    mean_body    = sum(body_lens) / len(body_lens) if body_lens else 0
    jondae_ratio = jondae / total if total else 0

    return {
        "label": label,
        "total_comments": total,
        "mean_title_len":   round(mean_title, 1),
        "mean_body_len":    round(mean_body, 1),
        "mean_comment_len": round(mean_comment, 1),
        "jondae_ratio":     round(jondae_ratio, 3),
        "banmal_count":     banmal,
        "jondae_count":     jondae,
    }

def check_register(src_posts, regen_posts):
    print("\n=== CHECK 5: REGISTER STATS (informational) ===")
    src_stats   = register_stats(src_posts,   "SOURCE")
    regen_stats = register_stats(regen_posts, "REGEN")

    for k in ("total_comments", "mean_title_len", "mean_body_len", "mean_comment_len", "jondae_ratio"):
        sv = src_stats[k]
        rv = regen_stats[k]
        print(f"  {k:22s}  SRC={sv}  REGEN={rv}")

    # Flag if jondae ratio differs by > 0.15 or mean comment len differs > 50%
    warnings = []
    if abs(src_stats["jondae_ratio"] - regen_stats["jondae_ratio"]) > 0.15:
        warnings.append(
            f"존댓말 ratio drift: src={src_stats['jondae_ratio']:.3f} regen={regen_stats['jondae_ratio']:.3f}"
        )
    if src_stats["mean_comment_len"] > 0:
        ratio = regen_stats["mean_comment_len"] / src_stats["mean_comment_len"]
        if ratio < 0.5 or ratio > 2.0:
            warnings.append(
                f"Mean comment length drifted >2x: src={src_stats['mean_comment_len']:.1f} regen={regen_stats['mean_comment_len']:.1f}"
            )
    if warnings:
        print("  WARN (informational):")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  OK — register within acceptable bounds")

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    src_posts   = load(SRC_PATH)
    regen_posts = load(REGEN_PATH)

    print(f"Loaded SOURCE: {len(src_posts)} posts")
    print(f"Loaded REGEN : {len(regen_posts)} posts")

    r1 = check_integrity(src_posts, regen_posts)
    r2 = check_verbatim(src_posts, regen_posts)
    r3 = check_banned_tokens(regen_posts)
    r4 = check_encoding(regen_posts)
    check_register(src_posts, regen_posts)

    print("\n" + "="*60)
    print("OVERALL VERDICT")
    print("="*60)
    gates = [
        ("1 INTEGRITY",       r1),
        ("2 VERBATIM OVERLAP", r2),
        ("3 BANNED TOKENS",    r3),
        ("4 ENCODING",         r4),
    ]
    all_pass = True
    for name, result in gates:
        status = "PASS" if result else "FAIL"
        print(f"  Check {name:22s}: {status}")
        if not result:
            all_pass = False

    print()
    if all_pass:
        print("VERDICT: SAFE TO SCALE — all hard gates pass.")
        print("  No banned tokens, no verbatim leakage >= 5 어절,")
        print("  structural integrity intact, encoding clean.")
    else:
        failed = [name for name, r in gates if not r]
        print(f"VERDICT: NOT SAFE — fix required before scaling.")
        print(f"  Failed gates: {', '.join(failed)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
