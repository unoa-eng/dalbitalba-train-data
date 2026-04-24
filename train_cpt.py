#!/usr/bin/env python3
"""
train_cpt.py — v2 — Continued pretraining on raw Korean community text.

Recipe (2026-04, fixed from 9-voice research consensus):
  - Base:    Qwen/Qwen3-8B-Base       (BBPE 151K, KMMLU 52.54, Apache-2.0)
  - LoRA:    r=64, alpha=64, use_rslora=True, target_modules="all-linear"
  - LR:      2e-4 cosine, warmup 3%, weight_decay 0.01
  - seq_len: 1024
  - batch:   1 x grad_accum 16 (eff 16)
  - bf16 + gradient_checkpointing (use_reentrant=False) + flash-attn 2
  - Data:    cpt_corpus.v2.jsonl (raw continuation text, no template)
  - Val:     val_set.v2.jsonl (5% held-out, time-split)
  - Seed:    full deterministic path
  - Hub:     push every 50 steps to last-checkpoint (pod-death resume)

No alpaca template. No adapter stacking — a fresh LoRA is trained on the
quantized base. After this finishes, scripts/merge_cpt_to_fp16.py folds
the adapter into a fp16 checkpoint that is then the base for SFT.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import subprocess
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
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    print(f"[오류] 의존성 누락: {e}")
    print(
        "pip install transformers>=4.51.3 peft>=0.13.2 bitsandbytes>=0.49.2 "
        "trl>=0.12.1 datasets>=2.21 accelerate>=0.33 flash-attn>=2.6.3"
    )
    sys.exit(1)

# ── 환경변수 ──────────────────────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base")
INPUT_JSONL = os.environ.get("INPUT_JSONL", "/workspace/data/cpt_corpus.v2.jsonl")
VAL_JSONL = os.environ.get("CPT_VAL_JSONL", "/workspace/data/val_set.v2.jsonl")
OUTPUT_DIR = os.environ.get("CPT_OUTPUT_DIR", "/workspace/out/cpt-lora")
CKPT_DIR = os.environ.get("CPT_CKPT_DIR", "/workspace/out/cpt-ckpt")
LOG_FILE = os.environ.get("CPT_LOG_FILE", "/workspace/train_cpt.log")
HUB_MODEL_ID = os.environ.get("CPT_HUB_MODEL_ID", "").strip() or None

SEED = int(os.environ.get("TRAIN_SEED", "42"))
MAX_SEQ_LEN = int(os.environ.get("CPT_MAX_SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("CPT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("CPT_GRAD_ACCUM", "16"))
LR = float(os.environ.get("CPT_LR", "2e-4"))
WARMUP_RATIO = float(os.environ.get("CPT_WARMUP_RATIO", "0.03"))
WEIGHT_DECAY = float(os.environ.get("CPT_WEIGHT_DECAY", "0.01"))
NUM_EPOCHS = int(os.environ.get("CPT_NUM_EPOCHS", "1"))
SAVE_STEPS = int(os.environ.get("CPT_SAVE_STEPS", "50"))
SAVE_TOTAL_LIMIT = int(os.environ.get("CPT_SAVE_TOTAL_LIMIT", "3"))
EVAL_STEPS = int(os.environ.get("CPT_EVAL_STEPS", "200"))
LOGGING_STEPS = int(os.environ.get("CPT_LOGGING_STEPS", "20"))

LORA_R = int(os.environ.get("CPT_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("CPT_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("CPT_LORA_DROPOUT", "0.0"))
USE_RSLORA = os.environ.get("CPT_USE_RSLORA", "1") == "1"
USE_DORA = os.environ.get("CPT_USE_DORA", "0") == "1"

FLASH_ATTN = os.environ.get("CPT_FLASH_ATTN", "flash_attention_2")

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


def load_corpus(path: str) -> Dataset:
    records: list[dict] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "").strip()
                if text:
                    records.append({"text": text})
            except json.JSONDecodeError:
                skipped += 1
    logger.info(f"load_corpus({path}) -> {len(records):,} rows (skipped {skipped})")
    return Dataset.from_list(records)


def tokenize_fn(examples: dict, tokenizer) -> dict:
    # No template. Raw continuation; each row is a standalone text.
    out = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    out["labels"] = [ids.copy() for ids in out["input_ids"]]
    return out


def main() -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("CPT v2 시작")
    logger.info(f"  base         : {BASE_MODEL}")
    logger.info(f"  input        : {INPUT_JSONL}")
    logger.info(f"  val          : {VAL_JSONL}")
    logger.info(f"  output       : {OUTPUT_DIR}")
    logger.info(f"  hub_model_id : {HUB_MODEL_ID or '(disabled)'}")
    logger.info(
        f"  LoRA r={LORA_R} α={LORA_ALPHA} rsLoRA={USE_RSLORA} DoRA={USE_DORA}"
    )
    logger.info(
        f"  seq_len={MAX_SEQ_LEN} eff_batch={BATCH_SIZE*GRAD_ACCUM} "
        f"lr={LR} epochs={NUM_EPOCHS}"
    )
    logger.info(f"  seed         : {SEED}")
    logger.info("=" * 60)

    set_all_seeds(SEED)

    if not torch.cuda.is_available():
        logger.warning("CUDA 불가. CPU는 비실용적. 진행은 시도됨.")
    else:
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"tokenizer vocab size: {tokenizer.vocab_size}")

    # ── 4-bit quantized base ─────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = dict(
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if FLASH_ATTN:
        model_kwargs["attn_implementation"] = FLASH_ATTN
    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    except Exception as e:
        logger.warning(f"flash_attention_2 로드 실패, eager로 fallback: {e}")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
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
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Data ────────────────────────────────────────────────────────
    train_raw = load_corpus(INPUT_JSONL)
    eval_raw = load_corpus(VAL_JSONL) if os.path.exists(VAL_JSONL) else None

    train_ds = train_raw.map(
        lambda ex: tokenize_fn(ex, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        desc="tokenize train",
    )
    eval_ds = None
    if eval_raw is not None:
        eval_ds = eval_raw.map(
            lambda ex: tokenize_fn(ex, tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=["text"],
            desc="tokenize val",
        )
    total_tokens = sum(len(x) for x in train_ds["input_ids"])
    logger.info(
        f"tokenize 완료: train {len(train_ds):,} rows / ~{total_tokens:,} tokens"
        + (f", val {len(eval_ds):,} rows" if eval_ds is not None else "")
    )

    steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    logger.info(
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
        f"warmup={warmup_steps}"
    )

    logger.info("[step] DataCollatorForLanguageModeling build")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )
    logger.info("[step] DataCollator OK")

    logger.info("[step] building training_args_kwargs")
    training_args_kwargs = dict(
        output_dir=CKPT_DIR,
        num_train_epochs=NUM_EPOCHS,
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
        group_by_length=True,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        weight_decay=WEIGHT_DECAY,
        ddp_find_unused_parameters=False,
        seed=SEED,
        data_seed=SEED,
    )
    if eval_ds is not None:
        training_args_kwargs.update(
            dict(
                eval_strategy="steps",
                eval_steps=EVAL_STEPS,
                per_device_eval_batch_size=BATCH_SIZE,
            )
        )
    if HUB_MODEL_ID:
        training_args_kwargs.update(
            dict(
                push_to_hub=True,
                hub_model_id=HUB_MODEL_ID,
                hub_strategy="checkpoint",
                hub_private_repo=True,
            )
        )
    logger.info(
        f"[step] instantiate TrainingArguments "
        f"(report_to={training_args_kwargs.get('report_to')}, "
        f"push_to_hub={training_args_kwargs.get('push_to_hub', False)}, "
        f"optim={training_args_kwargs.get('optim')})"
    )
    training_args = TrainingArguments(**training_args_kwargs)
    logger.info("[step] TrainingArguments OK")

    logger.info("[step] instantiate Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    logger.info("[step] Trainer OK")

    # resume
    resume_from = None
    ckpt_path = Path(CKPT_DIR)
    existing = sorted(ckpt_path.glob("checkpoint-*"), key=os.path.getmtime)
    if existing:
        resume_from = str(existing[-1])
        logger.info(f"resume from {resume_from}")

    logger.info("학습 시작 — trainer.train()")
    trainer.train(resume_from_checkpoint=resume_from)

    logger.info(f"LoRA adapter 저장 → {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    elapsed = (time.time() - start) / 3600
    logger.info(f"CPT 완료 ({elapsed:.2f}h) → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
