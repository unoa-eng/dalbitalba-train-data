"""Stage 1 — Extend Qwen3 tokenizer with structural markers + domain vocab.

Supersedes scripts/extend_tokenizer_v3.py by also adding domain vocabulary
identified in runs/audit/vocab_candidates_top.json (top N by score).

Designed for V4 Recipe Stage 1: addresses 'BPE fragments domain words like
밀빵 → 밀+빵' problem identified in 2026-05-04 audit.

Outputs:
- <out_dir>/tokenizer.json        (extended tokenizer)
- <out_dir>/added_tokens_v4.json  (manifest of all added tokens)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


BASE_MODEL = "Qwen/Qwen3-8B-Base"
COMMENT_DEPTH_RE = re.compile(r"<\|comment depth=(\d+)\|>")
TOKEN_LIST_FILENAME = "token_list.json"

# Carried over from v3 (structural markers).
STATIC_SPECIAL_TOKENS = [
    "<|post|>",
    "<|/post|>",
    "<|/comment|>",
    "<|thread|>",
    "<|/thread|>",
]

DEFAULT_DEPTH_MARKERS = [f"<|comment depth={d}|>" for d in range(0, 6)]


def discover_extra_depth_markers(jsonl_path: Path) -> list[str]:
    """Scan structured CPT for comment depth tokens deeper than 5."""
    if not jsonl_path or not jsonl_path.exists():
        return []
    extra: set[int] = set()
    with jsonl_path.open() as f:
        for ln in f:
            for m in COMMENT_DEPTH_RE.finditer(ln):
                d = int(m.group(1))
                if d > 5:
                    extra.add(d)
    return [f"<|comment depth={d}|>" for d in sorted(extra)]


def load_domain_vocab(top_json: Path, top_n: int) -> list[str]:
    """Read runs/audit/vocab_candidates_top.json and return top-N terms."""
    if not top_json.exists():
        print(f"[WARN] {top_json} not found — skipping domain vocab")
        return []
    data = json.loads(top_json.read_text())
    cands = data.get("candidates", [])
    return [c["term"] for c in cands[:top_n] if c.get("term")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structured-jsonl",
        type=Path,
        default=Path("v3-data/cpt_structured_v3.jsonl"),
        help="for discovering deep comment markers",
    )
    parser.add_argument(
        "--vocab-candidates-top",
        type=Path,
        default=Path("runs/audit/vocab_candidates_top.json"),
    )
    parser.add_argument(
        "--top-n", type=int, default=300,
        help="how many ranked domain candidates to add as new tokens",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="output dir for extended tokenizer + manifest",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) structural markers
    structural = list(STATIC_SPECIAL_TOKENS) + list(DEFAULT_DEPTH_MARKERS)
    structural += discover_extra_depth_markers(args.structured_jsonl)

    # 2) domain vocab
    domain = load_domain_vocab(args.vocab_candidates_top, args.top_n)

    print(f"[extend_v4] structural_tokens={len(structural)}  domain_tokens={len(domain)}")

    # 3) load tokenizer + add tokens
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    except Exception as exc:
        print(f"[WARN] tokenizer load failed: {exc}; writing token_list.json only")
        all_tokens = structural + domain
        (args.out_dir / TOKEN_LIST_FILENAME).write_text(
            json.dumps(all_tokens, ensure_ascii=False, indent=2)
        )
        return 0

    # special tokens (structural markers): added as additional_special_tokens
    n_special = tok.add_special_tokens(
        {"additional_special_tokens": structural}, replace_additional_special_tokens=False
    )
    # domain vocab: regular new tokens (not special; they appear in normal text)
    n_regular = tok.add_tokens(domain)

    print(f"[extend_v4] added_special={n_special}  added_regular={n_regular}")
    print(f"[extend_v4] new vocab_size={len(tok)}")

    tok.save_pretrained(args.out_dir)

    manifest = {
        "base_model": BASE_MODEL,
        "added_special_tokens": structural,
        "added_domain_tokens": domain,
        "n_added_special": n_special,
        "n_added_regular": n_regular,
        "new_vocab_size": len(tok),
        "source_vocab_top_json": str(args.vocab_candidates_top),
        "top_n_used": args.top_n,
    }
    (args.out_dir / "added_tokens_v4.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
