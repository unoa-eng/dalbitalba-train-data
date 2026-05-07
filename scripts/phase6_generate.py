#!/usr/bin/env python3

import json
import os
import re
import sys


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base")
SFT_ADAPTER_REPO = os.environ.get("SFT_ADAPTER_REPO", "").strip()
SFT_ADAPTER_SUBFOLDER = os.environ.get("SFT_ADAPTER_SUBFOLDER", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
INPUT_PATH = "/workspace/data/val_set.v2.jsonl"
OUTPUT_PATH = "/workspace/ai_generated.jsonl"
MAX_ROWS = int(os.environ.get("EVAL_MAX_ROWS", "500"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.1"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
MIN_P = float(os.environ.get("MIN_P", "0.05"))
KIND_ORDER = {"post", "comment"}
COMMENT_HINT_RE = re.compile(r"(댓글|답글|comment|reply)")
POST_HINT_RE = re.compile(r"(게시글|본문|post|원글)")
REPLY_TAG_RE = re.compile(r"^\[\d+(?:-\d+)*\]\s*")

# 격식체/AI 문체 필터 — 생성 후 재시도 트리거
FORMAL_FILTER_RE = re.compile(
    r'안녕하세요|감사합니다\.|하겠습니다|도움이\s*되|말씀하신|'
    r'문의\s*사항|참고하시기|좋은\s*하루|에\s*대해\s*알려',
    re.IGNORECASE
)


def normalize_kind(value: object) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw == "context_comment":
        return "comment"
    if raw in KIND_ORDER:
        return raw
    if raw in {"reply", "댓글", "답글"}:
        return "comment"
    if raw in {"본문", "원글"}:
        return "post"
    return None


def derive_kind(row: dict[str, object]) -> str:
    for key in ("kind", "pair_type", "label"):
        normalized = normalize_kind(row.get(key))
        if normalized:
            return normalized

    has_post = bool(str(row.get("post") or "").strip())
    has_comment = bool(str(row.get("comment") or "").strip())
    if has_comment and not has_post:
        return "comment"
    if has_post and not has_comment:
        return "post"

    prompt_blob = "\n".join(
        str(row.get(key) or "").strip() for key in ("instruction", "input", "prompt", "text")
    ).lower()
    if COMMENT_HINT_RE.search(prompt_blob):
        return "comment"
    if POST_HINT_RE.search(prompt_blob):
        return "post"

    text = str(row.get("text") or "").lstrip()
    if REPLY_TAG_RE.match(text):
        return "comment"
    return "post"


def build_prompt_seed(row: dict[str, object]) -> str:
    direct_text = str(row.get("text") or "").strip()
    if direct_text:
        return direct_text[:40]

    for key in ("prompt", "instruction", "input"):
        value = str(row.get(key) or "").strip()
        if value:
            return value[:40]
    return ""


def main() -> int:
    import peft
    import torch
    import transformers

    if not SFT_ADAPTER_REPO:
        print("[error] SFT_ADAPTER_REPO is required", file=sys.stderr)
        return 1

    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    adapter_kwargs = {"token": HF_TOKEN}
    if SFT_ADAPTER_SUBFOLDER:
        adapter_kwargs["subfolder"] = SFT_ADAPTER_SUBFOLDER
    try:
        model = peft.PeftModel.from_pretrained(model, SFT_ADAPTER_REPO, **adapter_kwargs)
    except Exception:
        if SFT_ADAPTER_SUBFOLDER:
            raise
        model = peft.PeftModel.from_pretrained(
            model,
            SFT_ADAPTER_REPO,
            subfolder="sft-lora",
            token=HF_TOKEN,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(INPUT_PATH, "r", encoding="utf-8") as src, open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as dst:
        for i, line in enumerate(src, start=1):
            if i > MAX_ROWS:
                break

            row = json.loads(line)
            prompt = build_prompt_seed(row)
            if not prompt:
                print(f"[warn] row {i} missing usable prompt seed; skipping", file=sys.stderr)
                continue
            kind = derive_kind(row)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                min_p=MIN_P,
                pad_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded

            # 격식체/AI 문체 재시도 (최대 2회)
            for _retry in range(2):
                if not FORMAL_FILTER_RE.search(continuation):
                    break
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=min(TEMPERATURE + 0.1, 1.5),
                    top_p=TOP_P,
                    min_p=MIN_P,
                    pad_token_id=tokenizer.eos_token_id,
                )
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                continuation = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded

            dst.write(
                json.dumps(
                    {"text": continuation, "seed": prompt, "kind": kind},
                    ensure_ascii=False,
                )
                + "\n"
            )

            if i % 50 == 0:
                print(f"[{i}/{MAX_ROWS}] generated", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
