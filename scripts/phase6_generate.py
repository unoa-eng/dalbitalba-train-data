#!/usr/bin/env python3

import json
import os
import re
import sys
from pathlib import Path


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base")
CPT_MERGED_REPO = os.environ.get("CPT_MERGED_REPO", "").strip()
CPT_MERGED_PATH = os.environ.get("CPT_MERGED_PATH", "").strip()
SFT_ADAPTER_REPO = os.environ.get("SFT_ADAPTER_REPO", "").strip()
SFT_ADAPTER_SUBFOLDER = os.environ.get("SFT_ADAPTER_SUBFOLDER", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
INPUT_PATH = "/workspace/data/val_set.v3.jsonl"
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


def path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def has_explicit_merged_source() -> bool:
    return bool(CPT_MERGED_REPO or CPT_MERGED_PATH)


def resolve_model_source() -> str:
    if path_exists(CPT_MERGED_PATH):
        return CPT_MERGED_PATH
    if CPT_MERGED_REPO:
        return CPT_MERGED_REPO
    return BASE_MODEL


def resolve_adapter_subfolders() -> list[str | None]:
    subfolders: list[str | None] = []
    if SFT_ADAPTER_SUBFOLDER:
        subfolders.append(SFT_ADAPTER_SUBFOLDER)
    if has_explicit_merged_source():
        subfolders.extend([None, "sft-lora"])
    else:
        subfolders.extend([None, "sft-lora", "cpt-lora"])
    return subfolders


def load_peft_adapter(model, peft, adapter_repo: str, adapter_kwargs: dict[str, object]):
    attempted: list[str | None] = []
    subfolders = resolve_adapter_subfolders()

    for subfolder in subfolders:
        if subfolder in attempted:
            continue
        attempted.append(subfolder)
        kwargs = dict(adapter_kwargs)
        if subfolder:
            kwargs["subfolder"] = subfolder
        try:
            return peft.PeftModel.from_pretrained(model, adapter_repo, **kwargs)
        except Exception as exc:
            label = "(root)" if subfolder is None else subfolder
            print(f"[adapter-attempt-fail] {adapter_repo} subfolder={label}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

    attempted_text = ", ".join("(root)" if value is None else value for value in attempted)
    if has_explicit_merged_source():
        print(
            "[warn] adapter load failed; falling back to merged model only "
            f"(tried: {attempted_text})",
            file=sys.stderr,
        )
        return model
    raise RuntimeError(
        f"failed to load adapter from {adapter_repo}; tried subfolders: {attempted_text}"
    )


def load_tokenizer(transformers, model_source: str):
    attempted: list[tuple[str, str | None]] = []
    candidates: list[tuple[str, str | None]] = []

    if has_explicit_merged_source():
        candidates.append((model_source, None))
    if SFT_ADAPTER_REPO:
        for subfolder in resolve_adapter_subfolders():
            candidates.append((SFT_ADAPTER_REPO, subfolder))
    candidates.append((BASE_MODEL, None))

    for source, subfolder in candidates:
        candidate = (source, subfolder)
        if not source or candidate in attempted:
            continue
        attempted.append(candidate)
        kwargs = {
            "token": HF_TOKEN,
            "trust_remote_code": True,
            "use_fast": True,
        }
        if subfolder:
            kwargs["subfolder"] = subfolder
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(source, **kwargs)
            label = source if not subfolder else f"{source}/{subfolder}"
            return tokenizer, label
        except Exception as exc:
            label = source if not subfolder else f"{source}/{subfolder}"
            print(f"[tokenizer-attempt-fail] {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

    attempted_text = ", ".join(
        source if not subfolder else f"{source}/{subfolder}"
        for source, subfolder in attempted
    )
    raise RuntimeError(
        "failed to load tokenizer from merged/adapter/base candidates: "
        f"{attempted_text}"
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

    model_source = resolve_model_source()
    if model_source == BASE_MODEL and not SFT_ADAPTER_REPO:
        print(
            "[error] need either SFT_ADAPTER_REPO or CPT_MERGED_REPO/CPT_MERGED_PATH",
            file=sys.stderr,
        )
        return 1
    print(f"[model] base source: {model_source}", flush=True)

    tokenizer, tokenizer_source = load_tokenizer(transformers, model_source)
    print(f"[tokenizer] source: {tokenizer_source}", flush=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    input_embedding_count = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != input_embedding_count:
        print(
            f"[model] resize_token_embeddings: {input_embedding_count} -> {len(tokenizer)}",
            flush=True,
        )
        model.resize_token_embeddings(len(tokenizer))
    adapter_kwargs = {"token": HF_TOKEN}
    if SFT_ADAPTER_REPO:
        model = load_peft_adapter(model, peft, SFT_ADAPTER_REPO, adapter_kwargs)

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
