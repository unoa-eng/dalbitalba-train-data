#!/usr/bin/env python3
"""
Phase 2 — cycle9 ORPO adapter mass expansion.

Reads Phase 1 Claude-direct seeds (threads_v1 + threads_v3), assembles cycle9
schema prompts (POST-TITLE / POST-BODY / CONTEXT / REPLY-DEPTH / PERSONA / [REPLY]),
and runs MLX `generate` against the cycle7 base + cycle9 ORPO adapter to expand
toward target=63,274 rows. Resumable append-only JSONL.

Each output row schema:
  {
    "id": str,
    "seed_id": str,
    "topic": str,           # title
    "style": str,           # 존댓말 | 반말 | 혼용
    "mood": str,            # depressed | anxious | seeking_empathy | neutral
    "persona_id": str,
    "prompt": str,
    "completion": str,
    "boardName": "cb2_밤문화이야기",
    "generated_at": str (UTC ISO),
    "adapter": str,
    "sampling": {temp, top_p, max_tokens, min_new_tokens, seed},
  }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_V1 = ROOT / "runs" / "cycle10-phase1-claude-direct" / "threads_v1.jsonl"
DEFAULT_SEED_V3 = ROOT / "runs" / "cycle10-phase1-claude-direct" / "threads_v3.jsonl"
DEFAULT_BASE = ROOT / "runs" / "cycle7-mac-simul" / "qwen3-8b-mlx-4bit"
DEFAULT_ADAPTER = ROOT / "runs" / "cycle9" / "phase3-pref-targeted-artifact-v5-smoke200"

STYLES = ["존댓말", "혼용", "반말"]
MOODS = ["depressed", "anxious", "seeking_empathy", "neutral"]

# Match cycle9 generation_audit.py post-cleaning behaviour minimally — strip a
# trailing `[N]` index prefix the model often emits, plus any leaked control
# tokens that show up at the end of a completion.
INDEX_PREFIX_RE = re.compile(r"^\s*\[\d+(?:-\d+)*\]\s*")
TRAILING_CTRL_RE = re.compile(
    r"(?:\n+\s*\[(?:POST-TITLE|POST-BODY|CONTEXT|PERSONA|REPLY|REPLY-DEPTH)[^\]]*\].*)$",
    re.DOTALL,
)


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pick_style(rng: random.Random) -> str:
    # cycle9 train mix roughly: 존댓말 ~35%, 혼용 ~25%, 반말 ~40%
    r = rng.random()
    if r < 0.35:
        return "존댓말"
    if r < 0.60:
        return "혼용"
    return "반말"


def pick_mood(rng: random.Random) -> str:
    return rng.choice(MOODS)


def pick_persona(rng: random.Random) -> str:
    # cycle9 train uses p-001..p-040 range
    return f"p-{rng.randint(1, 40):03d}"


def build_prompt(title: str, body: str, persona: str, style: str, mood: str) -> str:
    instruction = f"[POST-TITLE] {title}\n[POST-BODY] {body}"
    input_block = (
        "[CONTEXT]\n(no parent)\n"
        "[REPLY-DEPTH=1]\n"
        f"[PERSONA: {persona} | {style} | {mood}]"
    )
    return f"{instruction}\n{input_block}\n[REPLY]\n"


def load_seeds(paths: list[Path]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"[WARN] seed missing: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(obj.get("id") or "").strip()
                if not sid or sid in seen_ids:
                    continue
                title = str(obj.get("title") or "").strip()
                body = str(obj.get("content") or "").strip()
                if not title or not body:
                    continue
                seen_ids.add(sid)
                rows.append({"id": sid, "title": title, "body": body})
    return rows


def already_done(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    n = 0
    with out_path.open("r", encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


def clean_completion(text: str) -> str:
    text = text.strip()
    text = INDEX_PREFIX_RE.sub("", text)
    text = TRAILING_CTRL_RE.sub("", text)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 cycle9 ORPO mass expansion")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--seed-v1", default=str(DEFAULT_SEED_V1))
    parser.add_argument("--seed-v3", default=str(DEFAULT_SEED_V3))
    parser.add_argument(
        "--out",
        default=str(
            ROOT / "runs" / "cycle10-phase1-claude-direct" / "phase2_cycle9_expand.jsonl"
        ),
    )
    parser.add_argument("--target", type=int, default=63274)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--smoke", action="store_true", help="smoke mode: small N, more logging")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = load_seeds([Path(args.seed_v1), Path(args.seed_v3)])
    if not seeds:
        print("[FATAL] no seeds loaded", file=sys.stderr)
        return 2
    print(f"[info] loaded {len(seeds)} unique seeds")

    done = already_done(out_path) if args.resume else 0
    if done >= args.target:
        print(f"[done] {done} rows already >= target {args.target}")
        return 0
    print(f"[info] resume from {done}; target {args.target}; out={out_path}")

    # MLX import (deferred so smoke testing CLI args is fast)
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_lm.utils import load
        from mlx_lm.generate import generate, make_sampler
    except Exception as exc:  # pragma: no cover
        print(f"[FATAL] mlx_lm not available: {exc}", file=sys.stderr)
        return 3

    print("[info] loading model + adapter …")
    t_load = time.perf_counter()
    model, tokenizer = load(
        args.base,
        adapter_path=args.adapter,
        tokenizer_config={"trust_remote_code": True},
    )
    print(f"[info] loaded in {time.perf_counter() - t_load:.1f}s")
    sampler = make_sampler(args.temp, args.top_p)

    rng = random.Random(args.seed + done)

    adapter_label = str(Path(args.adapter).resolve())
    sampling_meta = {
        "temp": args.temp,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "min_new_tokens": args.min_new_tokens,
        "seed": args.seed,
    }

    eos_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    target = args.target
    if args.smoke and target > 20:
        print(f"[smoke] capping target {target} -> 5")
        target = 5

    written = done
    started = time.perf_counter()

    # Open in append mode; one flush per row keeps things resumable on crash.
    with out_path.open("a", encoding="utf-8") as out_fh:
        idx = 0
        while written < target:
            seed_row = seeds[idx % len(seeds)]
            idx += 1

            style = pick_style(rng)
            mood = pick_mood(rng)
            persona = pick_persona(rng)
            prompt = build_prompt(
                seed_row["title"], seed_row["body"], persona, style, mood
            )

            # Encode with chat template? Cycle9 train used raw text; generation_audit
            # also feeds raw text. Match that.
            try:
                gen_text = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            except TypeError:
                # Older mlx_lm signature
                gen_text = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temp=args.temp,
                    verbose=False,
                )

            completion = clean_completion(gen_text)
            if args.smoke:
                print(f"--- sample {written + 1} ({style}/{mood}/{persona}) ---")
                print(f"TITLE: {seed_row['title']}")
                print(f"BODY:  {seed_row['body'][:100]}")
                print(f"COMP:  {completion!r}")

            row_id = f"p2-{written + 1:06d}"
            out_row = {
                "id": row_id,
                "seed_id": seed_row["id"],
                "topic": seed_row["title"],
                "style": style,
                "mood": mood,
                "persona_id": persona,
                "prompt": prompt,
                "completion": completion,
                "boardName": "cb2_밤문화이야기",
                "generated_at": utc_iso(),
                "adapter": adapter_label,
                "sampling": sampling_meta,
            }
            out_fh.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_fh.flush()
            written += 1

            if (written - done) % 25 == 0 and not args.smoke:
                elapsed = time.perf_counter() - started
                rate = (written - done) / max(elapsed, 1e-6)
                remaining = (target - written) / max(rate, 1e-6)
                print(
                    f"[progress] {written}/{target} "
                    f"({rate:.2f} rows/s, ETA {remaining/60:.1f} min)"
                )

    elapsed = time.perf_counter() - started
    print(
        f"[done] wrote {written - done} new rows in {elapsed:.1f}s "
        f"(total {written}/{target}) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
