#!/usr/bin/env python3
"""
Extend the Qwen3 tokenizer with v3 structured CPT special tokens.

The script always adds the required structural markers:
  - <|post|>, <|/post|>, <|thread|>, <|/thread|>, <|/comment|>
  - <|comment depth=0|> ... <|comment depth=5|>

If the structured CPT JSONL contains deeper nesting than depth 5, those extra
depth markers are added too. On tokenizer/model download failure, the script
still writes `token_list.json` so training can extend the tokenizer at runtime.
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
STATIC_SPECIAL_TOKENS = [
    "<|post|>",
    "<|/post|>",
    "<|/comment|>",
    "<|thread|>",
    "<|/thread|>",
]
REQUIRED_DEPTH_TOKENS = [f"<|comment depth={depth}|>" for depth in range(6)]


def dedupe_preserve_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend Qwen3 tokenizer for v3 CPT")
    parser.add_argument(
        "--model",
        "--base-model",
        dest="base_model",
        default=BASE_MODEL,
        help="Tokenizer source model or local tokenizer path",
    )
    parser.add_argument(
        "--structured-jsonl",
        default="v3-data/cpt_structured_v3.jsonl",
        help="Structured CPT JSONL used to discover comment depth tokens",
    )
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="output_dir",
        default="v3-data/tokenizer",
        help="Directory where the extended tokenizer will be saved",
    )
    return parser.parse_args()


def discover_comment_depth_tokens(structured_jsonl: Path) -> list[str]:
    if not structured_jsonl.exists():
        raise FileNotFoundError(
            f"structured CPT jsonl not found: {structured_jsonl}"
        )

    depths: set[int] = set()
    with structured_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            for match in COMMENT_DEPTH_RE.findall(text):
                depths.add(int(match))

    return [f"<|comment depth={depth}|>" for depth in sorted(depths)]


def build_special_tokens(structured_jsonl: Path) -> list[str]:
    depth_tokens = list(REQUIRED_DEPTH_TOKENS)
    if structured_jsonl.exists():
        depth_tokens.extend(discover_comment_depth_tokens(structured_jsonl))
    ordered_depth_tokens = dedupe_preserve_order(depth_tokens)
    return [
        STATIC_SPECIAL_TOKENS[0],
        STATIC_SPECIAL_TOKENS[1],
        *ordered_depth_tokens,
        STATIC_SPECIAL_TOKENS[2],
        STATIC_SPECIAL_TOKENS[3],
        STATIC_SPECIAL_TOKENS[4],
    ]


def write_token_list(output_dir: Path, model_name: str, special_tokens: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    token_list_path = output_dir / TOKEN_LIST_FILENAME
    payload = {
        "model": model_name,
        "additional_special_tokens": special_tokens,
    }
    token_list_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return token_list_path


def main() -> None:
    args = parse_args()
    structured_jsonl = Path(args.structured_jsonl)
    output_dir = Path(args.output_dir)
    special_tokens = build_special_tokens(structured_jsonl)
    token_list_path = write_token_list(output_dir, args.base_model, special_tokens)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as exc:
        print(f"[tokenizer] source        : {args.base_model}")
        print(f"[tokenizer] structured    : {structured_jsonl}")
        print(f"[tokenizer] output        : {output_dir}")
        print(f"[tokenizer] token list    : {token_list_path}")
        print(f"[tokenizer] fallback only : {exc}")
        return

    base_vocab_size = len(tokenizer)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens}
    )
    tokenizer.save_pretrained(output_dir)

    print(f"[tokenizer] source        : {args.base_model}")
    print(f"[tokenizer] structured    : {structured_jsonl}")
    print(f"[tokenizer] output        : {output_dir}")
    print(f"[tokenizer] token list    : {token_list_path}")
    print(f"[tokenizer] base size     : {base_vocab_size}")
    print(f"[tokenizer] extended size : {len(tokenizer)}")
    print(f"[tokenizer] added         : {added}")
    print("[tokenizer] special tokens:")
    for token in special_tokens:
        print(f"  - {token} -> {tokenizer.convert_tokens_to_ids(token)}")


if __name__ == "__main__":
    main()
