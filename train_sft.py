#!/usr/bin/env python3
"""
train_sft.py — v3 — Supervised fine-tune on CPT-merged base with
reply-pair and raw-continuation mixed data.

Key recipe changes vs v2:
  - DoRA enabled by default (arXiv:2402.09353, +2-5% domain adaptation)
  - chatml prompt template (vLLM/llama.cpp/ollama + Qwen3 native)
  - Thread-aware v3 loader (post_title, post_body_excerpt, parent_comment,
    target_comment, thread_key, depth, task_type, length_bucket, source_id)
  - SFT_LR raised 5e-5 → 1e-4 (per arXiv:2602.04998 + TRAINING_RECIPE_VALIDATION.md)
  - Base is the CPT-merged fp16 model (NOT the stacking pattern)
  - No alpaca template — uses:
        raw:    "{text}<|endoftext|>"     (label = input_ids)
        chatml: <|im_start|>system...assistant\n{target}<|im_end|>
                (loss masked on everything before assistant response)
        raw-pair (v2 compat): "{post}\\n{comment}<eos>"
  - r=64 rsLoRA, lr=1e-4, 3 epochs, seq_len=2048
  - NEFTune off (style transfer hurts from embedding noise)
  - Full seed path

Env:
  BASE_MODEL            : /workspace/out/cpt-merged-fp16 (merged)
  SFT_RAW_JSONL         : /workspace/data/cpt_corpus.v2.jsonl
  SFT_PAIR_JSONL        : /workspace/data/sft_pairs.v2.jsonl     (v2 schema)
  SFT_PAIR_JSONL_V3     : /workspace/data/sft_pairs.v3.jsonl     (v3 schema, P1-A)
  SFT_VAL_JSONL         : /workspace/data/val_set.v2.jsonl
  SFT_OUTPUT_DIR        : /workspace/out/sft-lora
  SFT_CKPT_DIR          : /workspace/out/sft-ckpt
  SFT_HUB_MODEL_ID      : (optional) HF repo to push
  SFT_RAW_RATIO         : 0.8
  SFT_NUM_EPOCHS        : 3
  SFT_LR                : 1e-4
  SFT_MAX_SEQ_LEN       : 2048
  SFT_USE_DORA          : 1   (DoRA enabled by default)
  SFT_PROMPT_FORMAT     : chatml  ("raw" keeps legacy pair behavior)
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    print(f"[오류] 의존성 누락: {e}")
    print(
        "pip install transformers>=4.51.3 peft>=0.13.2 bitsandbytes>=0.49.2 "
        "trl>=0.12.1 datasets>=2.21 accelerate>=0.33"
    )
    sys.exit(1)

# ── Env ──────────────────────────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/out/cpt-merged-fp16")
SFT_RAW_JSONL = os.environ.get("SFT_RAW_JSONL", "/workspace/data/cpt_corpus.v2.jsonl")
SFT_PAIR_JSONL = os.environ.get("SFT_PAIR_JSONL", "/workspace/data/sft_pairs.v2.jsonl")
SFT_PAIR_JSONL_V3 = os.environ.get("SFT_PAIR_JSONL_V3", "").strip() or None
SFT_VAL_JSONL = os.environ.get("SFT_VAL_JSONL", "/workspace/data/val_set.v2.jsonl")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "/workspace/out/sft-lora")
CKPT_DIR = os.environ.get("SFT_CKPT_DIR", "/workspace/out/sft-ckpt")
LOG_FILE = os.environ.get("SFT_LOG_FILE", "/workspace/train_sft.log")
HUB_MODEL_ID = os.environ.get("SFT_HUB_MODEL_ID", "").strip() or None

SEED = int(os.environ.get("TRAIN_SEED", "42"))
MAX_SEQ_LEN = int(os.environ.get("SFT_MAX_SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "16"))
# P1-C: lr raised from 5e-5 → 1e-4 (arXiv:2602.04998 + TRAINING_RECIPE_VALIDATION.md)
LR = float(os.environ.get("SFT_LR", "1e-4"))
WARMUP_RATIO = float(os.environ.get("SFT_WARMUP_RATIO", "0.05"))
WEIGHT_DECAY = float(os.environ.get("SFT_WEIGHT_DECAY", "0.01"))
NUM_EPOCHS = int(os.environ.get("SFT_NUM_EPOCHS", "2"))
MAX_STEPS_OVERRIDE = int(os.environ.get("SFT_MAX_STEPS", "0") or "0")
SAVE_STEPS = int(os.environ.get("SFT_SAVE_STEPS", "50"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SFT_SAVE_TOTAL_LIMIT", "3"))
EVAL_STEPS = int(os.environ.get("SFT_EVAL_STEPS", "200"))
LOGGING_STEPS = int(os.environ.get("SFT_LOGGING_STEPS", "10"))
RAW_LIMIT_ROWS = int(os.environ.get("SFT_RAW_LIMIT_ROWS", "0") or "0")
PAIR_LIMIT_ROWS = int(os.environ.get("SFT_PAIR_LIMIT_ROWS", "0") or "0")
VAL_LIMIT_ROWS = int(os.environ.get("SFT_VAL_LIMIT_ROWS", "0") or "0")

RAW_RATIO = float(os.environ.get("SFT_RAW_RATIO", "0.8"))
PAIR_RATIO = 1.0 - RAW_RATIO

LORA_R = int(os.environ.get("SFT_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("SFT_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("SFT_LORA_DROPOUT", "0.0"))
USE_RSLORA = os.environ.get("SFT_USE_RSLORA", "1") == "1"
# P1-C: DoRA enabled by default (arXiv:2402.09353, +2-5% domain adaptation at <0.1% extra params)
USE_DORA = os.environ.get("SFT_USE_DORA", "1") == "1"
FLASH_ATTN = os.environ.get("SFT_FLASH_ATTN", "flash_attention_2")

# P1-C: chatml prompt format (default) or "raw" for backward compat
PROMPT_FORMAT = os.environ.get("SFT_PROMPT_FORMAT", "chatml").strip().lower()

# chatml system prompt (Korean nightlife community persona)
_CHATML_SYSTEM = (
    "당신은 한국 유흥업 종사자 커뮤니티의 일반 사용자입니다. "
    "짧고 솔직하며 초성·은어를 자연스럽게 씁니다."
)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        logger.warning(f"missing file (skip): {path}")
        return []
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    logger.info(f"  load {path} -> {len(rows):,}")
    return rows


# ── chatml prompt builder ──────────────────────────────────────────────

def _build_chatml_user_text(row: dict) -> str:
    """Build the user-turn content from a v3 schema row."""
    board = row.get("board", "")
    kind = row.get("kind", row.get("task_type", ""))
    post_title = row.get("post_title", "")
    post_body_excerpt = row.get("post_body_excerpt", "")
    parent_comment = row.get("parent_comment")
    depth = row.get("depth", 1)

    lines = []
    if board:
        lines.append(f"[게시판] {board}")
    if kind:
        lines.append(f"[유형] {kind}")
    if post_title:
        lines.append(f"[제목] {post_title}")
    if post_body_excerpt:
        lines.append(f"[원글] {post_body_excerpt}")
    if parent_comment:
        lines.append(f"[부모댓글] {parent_comment}")
    lines.append(f"[depth] {depth}")
    lines.append("위에 어울리는 답글을 1개 작성하세요.")
    return "\n".join(lines)


def build_chatml_example(row: dict, tokenizer, is_v3: bool = True) -> dict | None:
    """
    Build a chatml-formatted training example with completion-only loss masking.

    Loss mask: everything before <|im_start|>assistant\n is set to -100.
    Only target_comment + closing <|im_end|> tokens are trained.

    Tries tokenizer.apply_chat_template first (Qwen3 native).
    Falls back to manual string construction + tokenize.
    """
    if is_v3:
        target = row.get("target_comment", "").strip()
    else:
        target = row.get("comment", "").strip()

    if not target:
        return None

    # Build message list for apply_chat_template
    if is_v3:
        user_content = _build_chatml_user_text(row)
    else:
        post = row.get("post", "").strip()
        if not post:
            return None
        user_content = post

    messages = [
        {"role": "system", "content": _CHATML_SYSTEM},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": target},
    ]

    # Try apply_chat_template (Qwen3 has one); fall back to manual construction
    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        use_native = True
    except Exception:
        use_native = False
        full_text = None

    if use_native and full_text is not None:
        # Tokenize the full sequence
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        # Find the assistant response start to build the loss mask.
        # Build a prefix (everything up to and including "\n{target}") then
        # find the boundary by tokenizing the prompt-only portion.
        assistant_marker = "<|im_start|>assistant\n"
        prompt_end_idx = full_text.rfind(assistant_marker)
        if prompt_end_idx == -1:
            # Fallback: mask nothing (train on full sequence)
            labels = full_ids.copy()
        else:
            prompt_text = full_text[: prompt_end_idx + len(assistant_marker)]
            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]
            prefix_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prefix_len + full_ids[prefix_len:]
    else:
        # Manual chatml construction (no native chat template)
        prompt_text = (
            f"<|im_start|>system\n{_CHATML_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        response_text = f"{target}<|im_end|>"

        prompt_ids = tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        resp_ids = tokenizer(
            response_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        full_ids = (prompt_ids + resp_ids)[:MAX_SEQ_LEN]
        prefix_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prefix_len + full_ids[prefix_len:]

    attn = [1] * len(full_ids)
    return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}


# ── v2 pair builder (backward compat, raw format) ────────────────────

def build_pair_example_raw(row: dict, tokenizer) -> dict | None:
    """v2 schema raw pair: '{post}\\n{comment}<eos>' with loss mask on post."""
    post = row.get("post", "").strip()
    comment = row.get("comment", "").strip()
    if not post or not comment:
        return None
    eos = tokenizer.eos_token or ""
    prefix = post + "\n"
    response = comment + eos
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids = (prefix_ids + resp_ids)[:MAX_SEQ_LEN]
    attn = [1] * len(input_ids)
    labels = [-100] * min(len(prefix_ids), len(input_ids))
    labels += input_ids[len(labels):]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def build_mixed_dataset(tokenizer) -> tuple[Dataset, Dataset | None]:
    rng = random.Random(SEED)

    raw_rows = load_jsonl(SFT_RAW_JSONL)

    # P1-C: v3 loader — when SFT_PAIR_JSONL_V3 is set, use v3 schema (P1-A output).
    # Falls back to v2 SFT_PAIR_JSONL when only that is set.
    if SFT_PAIR_JSONL_V3:
        logger.info(f"v3 pair loader active: {SFT_PAIR_JSONL_V3}")
        pair_rows = load_jsonl(SFT_PAIR_JSONL_V3)
        use_v3 = True
    else:
        logger.info(f"v2 pair loader active (compat): {SFT_PAIR_JSONL}")
        pair_rows = load_jsonl(SFT_PAIR_JSONL)
        use_v3 = False

    val_rows = load_jsonl(SFT_VAL_JSONL)

    if RAW_LIMIT_ROWS > 0 and len(raw_rows) > RAW_LIMIT_ROWS:
        logger.info(f"SFT_RAW_LIMIT_ROWS 적용: {len(raw_rows):,} -> {RAW_LIMIT_ROWS:,}")
        raw_rows = raw_rows[:RAW_LIMIT_ROWS]
    if PAIR_LIMIT_ROWS > 0 and len(pair_rows) > PAIR_LIMIT_ROWS:
        logger.info(f"SFT_PAIR_LIMIT_ROWS 적용: {len(pair_rows):,} -> {PAIR_LIMIT_ROWS:,}")
        pair_rows = pair_rows[:PAIR_LIMIT_ROWS]
    if VAL_LIMIT_ROWS > 0 and len(val_rows) > VAL_LIMIT_ROWS:
        logger.info(f"SFT_VAL_LIMIT_ROWS 적용: {len(val_rows):,} -> {VAL_LIMIT_ROWS:,}")
        val_rows = val_rows[:VAL_LIMIT_ROWS]

    logger.info(
        f"mix policy: raw={RAW_RATIO:.2f} pair={PAIR_RATIO:.2f} "
        f"format={PROMPT_FORMAT} v3={use_v3}"
    )

    # compute pair target count so that pairs / (raw + pairs) == PAIR_RATIO
    if RAW_RATIO >= 1.0 or not pair_rows:
        pairs_take = 0
    elif RAW_RATIO <= 0.0:
        pairs_take = len(pair_rows)
        raw_rows = []
    else:
        raw_n = len(raw_rows)
        pairs_take = int(round(raw_n * PAIR_RATIO / max(RAW_RATIO, 1e-6)))
        pairs_take = min(pairs_take, len(pair_rows))
    logger.info(
        f"sampled sizes: raw={len(raw_rows):,} pair={pairs_take:,} / "
        f"{len(pair_rows):,} available"
    )
    rng.shuffle(pair_rows)
    pair_rows = pair_rows[:pairs_take]

    eos = tokenizer.eos_token or ""
    if not eos:
        raise RuntimeError("tokenizer has no EOS token")

    def build_raw_example(row: dict) -> dict:
        text = row.get("text", "").strip()
        full = text + eos
        tok = tokenizer(
            full, truncation=True, max_length=MAX_SEQ_LEN,
            padding=False, add_special_tokens=True,
        )
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": tok["input_ids"].copy(),
        }

    def build_pair_example(row: dict) -> dict | None:
        if PROMPT_FORMAT == "chatml":
            return build_chatml_example(row, tokenizer, is_v3=use_v3)
        else:
            # raw format: v2 compat
            if use_v3:
                # v3 row but raw format requested — synthesise v2-style fields
                post = (row.get("post_title", "") + "\n" + row.get("post_body_excerpt", "")).strip()
                comment = row.get("target_comment", "").strip()
                synthetic = {"post": post, "comment": comment}
                return build_pair_example_raw(synthetic, tokenizer)
            else:
                return build_pair_example_raw(row, tokenizer)

    merged: list[dict] = []
    for row in raw_rows:
        merged.append(build_raw_example(row))
    for row in pair_rows:
        ex = build_pair_example(row)
        if ex is not None:
            merged.append(ex)
    rng.shuffle(merged)

    logger.info(f"merged train size: {len(merged):,}")
    train_ds = Dataset.from_list(merged)

    eval_ds = None
    if val_rows:
        eval_examples = [build_raw_example(r) for r in val_rows if r.get("text")]
        eval_ds = Dataset.from_list(eval_examples)
        logger.info(f"val size: {len(eval_ds):,}")

    return train_ds, eval_ds


def main() -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("SFT v3 시작")
    logger.info(f"  base          : {BASE_MODEL}")
    logger.info(f"  raw_jsonl     : {SFT_RAW_JSONL}")
    logger.info(f"  pair_jsonl_v3 : {SFT_PAIR_JSONL_V3 or '(unset)'}")
    logger.info(f"  pair_jsonl    : {SFT_PAIR_JSONL}")
    logger.info(f"  val_jsonl     : {SFT_VAL_JSONL}")
    logger.info(f"  output        : {OUTPUT_DIR}")
    logger.info(f"  hub_model_id  : {HUB_MODEL_ID or '(disabled)'}")
    logger.info(f"  prompt_format : {PROMPT_FORMAT}")
    logger.info(
        f"  LoRA r={LORA_R} α={LORA_ALPHA} rsLoRA={USE_RSLORA} DoRA={USE_DORA}"
    )
    logger.info(
        f"  seq_len={MAX_SEQ_LEN} eff_batch={BATCH_SIZE*GRAD_ACCUM} "
        f"lr={LR} epochs={NUM_EPOCHS}"
    )
    logger.info("=" * 60)

    set_all_seeds(SEED)

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA 없음 — CPU 실행 비실용적")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    tok_path = BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    mk = dict(
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if FLASH_ATTN:
        mk["attn_implementation"] = FLASH_ATTN
    try:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **mk)
    except Exception as e:
        logger.warning(f"flash_attention_2 로드 실패, eager로 fallback: {e}")
        mk.pop("attn_implementation", None)
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **mk)

    base_model.config.use_cache = False

    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        use_rslora=USE_RSLORA,
        use_dora=USE_DORA,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_ds, eval_ds = build_mixed_dataset(tokenizer)

    total_steps = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    if MAX_STEPS_OVERRIDE > 0:
        logger.info(f"SFT_MAX_STEPS override 적용: {total_steps} -> {MAX_STEPS_OVERRIDE}")
        total_steps = MAX_STEPS_OVERRIDE
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    logger.info(f"total_steps={total_steps} warmup={warmup_steps}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    kwargs = dict(
        output_dir=CKPT_DIR,
        num_train_epochs=NUM_EPOCHS,
        max_steps=MAX_STEPS_OVERRIDE if MAX_STEPS_OVERRIDE > 0 else -1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_steps=LOGGING_STEPS,
        logging_dir=f"{CKPT_DIR}/logs",
        report_to=os.environ.get("TRAIN_REPORT_TO", "none"),
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        group_by_length=False,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        weight_decay=WEIGHT_DECAY,
        ddp_find_unused_parameters=False,
        seed=SEED,
        # data_seed omitted — see train_cpt.py for rationale
        # (transformers 4.51 requires accelerate>=1.1 for data_seed;
        # we pin accelerate==0.34.2 for trl 0.12.1 compat).
    )
    if eval_ds is not None:
        kwargs.update(
            dict(
                eval_strategy="steps",
                eval_steps=EVAL_STEPS,
                per_device_eval_batch_size=BATCH_SIZE,
            )
        )
    if HUB_MODEL_ID:
        kwargs.update(
            dict(
                push_to_hub=True,
                hub_model_id=HUB_MODEL_ID,
                hub_strategy="checkpoint",
                hub_private_repo=True,
            )
        )
    training_args = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # resume
    resume_from = None
    ckpt_path = Path(CKPT_DIR)
    existing = sorted(ckpt_path.glob("checkpoint-*"), key=os.path.getmtime)
    if existing:
        resume_from = str(existing[-1])
        logger.info(f"resume from {resume_from}")

    logger.info("SFT 학습 시작")
    trainer.train(resume_from_checkpoint=resume_from)

    logger.info(f"SFT adapter 저장 → {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    elapsed = (time.time() - start) / 3600
    logger.info(f"SFT 완료 ({elapsed:.2f}h) → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
