#!/usr/bin/env python3
"""
dedup_minhash.py — Global MinHash near-dedup for dalbitalba training data.

Replaces the thread-internal Jaccard ≥ 0.85 dedup that only dropped 19 rows
on a 67k corpus while local_verification_loop.py still warned dup_rate ≈ 0.40.
That signals operator/ad templates being copy-pasted across many threads.
Per-thread dedup cannot catch those by construction.

Algorithm: classical MinHash + LSH banding, stdlib only.
- Tokens: char 5-grams over whitespace-collapsed lowercased text
- Permutations: 128
- LSH bands: 32 bands × 4 rows  (Jaccard ≈ 0.8 acceptance)
- Per bucket: keep first record, drop subsequent

Tunable knobs via CLI but defaults follow FineWeb (arXiv:2406.17557) guidance:
"big-cluster dedup gives the largest perplexity gain; do not over-prune
small clusters". We only collapse a cluster down to one survivor; we do not
delete clusters wholesale.

Usage:
    python scripts/dedup_minhash.py --in cpt_corpus.v2.jsonl \
        --field text --out cpt_corpus.v3.jsonl

    python scripts/dedup_minhash.py --in sft_pairs.v2.jsonl \
        --field comment --out sft_pairs.v3.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

WS_RE = re.compile(r"\s+")
_LARGE_PRIME = (1 << 61) - 1
_HASH_MASK = (1 << 64) - 1


def _seed_pairs(num_perm: int, seed: int = 42) -> list[tuple[int, int]]:
    """Deterministic (a, b) coefficients for h_i(x) = (a*x + b) mod prime."""
    rng = _ParkMillerRng(seed)
    out = []
    for _ in range(num_perm):
        a = (rng.next() & _HASH_MASK) | 1
        b = rng.next() & _HASH_MASK
        out.append((a, b))
    return out


class _ParkMillerRng:
    """Minimal deterministic LCG so we don't depend on random.Random ordering
    across Python versions."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & _HASH_MASK or 1

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & _HASH_MASK
        return self.state


def _shingles(text: str, n: int = 5) -> list[str]:
    s = WS_RE.sub(" ", text.strip().lower())
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _hash_token(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


def minhash_signature(text: str, perms: list[tuple[int, int]], n: int = 5) -> tuple[int, ...]:
    shingles = _shingles(text, n=n)
    if not shingles:
        return tuple([0] * len(perms))
    base = [_hash_token(sh) for sh in shingles]
    sig = []
    for a, b in perms:
        m = _LARGE_PRIME
        for h in base:
            v = ((a * h + b) % _LARGE_PRIME) & _HASH_MASK
            if v < m:
                m = v
        sig.append(m)
    return tuple(sig)


def banded_keys(sig: tuple[int, ...], bands: int) -> list[bytes]:
    rows = len(sig) // bands
    out = []
    for b in range(bands):
        chunk = sig[b * rows : (b + 1) * rows]
        out.append(hashlib.blake2b(repr(chunk).encode(), digest_size=12).digest())
    return out


def dedup_records(
    records: list[dict],
    field: str,
    num_perm: int = 128,
    bands: int = 32,
    shingle_n: int = 5,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """Returns (kept_records, stats).

    `field` is the JSONL key whose text is hashed. For SFT pairs you typically
    want "comment"; for CPT corpus you want "text".
    """
    perms = _seed_pairs(num_perm, seed=seed)
    bucket_to_idx: dict[bytes, int] = {}
    keep_mask = [True] * len(records)
    cluster_size: defaultdict[int, int] = defaultdict(int)

    for idx, rec in enumerate(records):
        text = rec.get(field) or ""
        if not isinstance(text, str) or not text:
            keep_mask[idx] = False
            continue
        sig = minhash_signature(text, perms=perms, n=shingle_n)
        keys = banded_keys(sig, bands=bands)
        survivor_idx = None
        for k in keys:
            if k in bucket_to_idx:
                survivor_idx = bucket_to_idx[k]
                break
        if survivor_idx is not None:
            keep_mask[idx] = False
            cluster_size[survivor_idx] += 1
        else:
            for k in keys:
                bucket_to_idx[k] = idx
            cluster_size[idx] += 1

    kept = [rec for rec, keep in zip(records, keep_mask) if keep]
    big_clusters = sorted(
        (cluster_size[i] for i in cluster_size if cluster_size[i] >= 5), reverse=True
    )
    stats = {
        "input_rows": len(records),
        "kept_rows": len(kept),
        "dropped_rows": len(records) - len(kept),
        "drop_rate": (len(records) - len(kept)) / max(1, len(records)),
        "field": field,
        "num_perm": num_perm,
        "bands": bands,
        "shingle_n": shingle_n,
        "seed": seed,
        "biggest_cluster": max(cluster_size.values()) if cluster_size else 0,
        "clusters_5plus": len(big_clusters),
        "top10_cluster_sizes": big_clusters[:10],
    }
    return kept, stats


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Global MinHash near-dedup (stdlib).")
    p.add_argument("--in", dest="inp", required=True, help="input JSONL")
    p.add_argument("--out", required=True, help="output JSONL (deduped)")
    p.add_argument("--field", default="text", help="JSONL key holding the text")
    p.add_argument("--num-perm", type=int, default=128)
    p.add_argument("--bands", type=int, default=32)
    p.add_argument("--shingle-n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--summary",
        default=None,
        help="optional path to write a JSON stats summary",
    )
    args = p.parse_args()

    rows = _read_jsonl(Path(args.inp))
    kept, stats = dedup_records(
        rows,
        field=args.field,
        num_perm=args.num_perm,
        bands=args.bands,
        shingle_n=args.shingle_n,
        seed=args.seed,
    )
    _write_jsonl(Path(args.out), kept)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(
        f"[dedup_minhash] {args.inp} field={args.field}: "
        f"input={stats['input_rows']:,} kept={stats['kept_rows']:,} "
        f"dropped={stats['dropped_rows']:,} ({stats['drop_rate']*100:.1f}%); "
        f"biggest cluster={stats['biggest_cluster']}, "
        f"clusters≥5={stats['clusters_5plus']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
