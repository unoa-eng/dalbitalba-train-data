#!/usr/bin/env python3
"""
train_sft.py — v2 — Supervised fine-tune on CPT-merged base with
reply-pair and raw-continuation mixed data.

Key recipe changes vs v1:
  - Base is the CPT-merged fp16 model (NOT the stacking pattern)
  - No alpaca template — uses:
        raw:  "{text}<|endoftext|>"     (label = input_ids)
        pair: "{post}\\n{comment}<|endoftext|>"   (label masked on post,
               trained on comment only)
  - Mix 80% raw continuation / 20% reply-pair at dataset build time
  - r=64 rsLoRA, lr=5e-5, 2 epochs, seq_len=1024
  - NEFTune off (style transfer hurts from embedding noise)
  - DoRA reserved for 2nd-pass ablation
  - Full seed path

Env:
  BASE_MODEL            : /workspace/out/cpt-merged-fp16 (merged)
  SFT_RAW_JSONL         : /workspace/data/cpt_corpus.v2.jsonl
  SFT_INPUT_JSONL       : /workspace/data/sft_pairs.v2.jsonl
  SFT_VAL_JSONL         : /workspace/data/val_set.v2.jsonl
  SFT_OUTPUT_DIR        : /workspace/out/sft-lora
  SFT_CKPT_DIR          : /workspace/out/sft-ckpt
  SFT_HUB_MODEL_ID      : (optional) HF repo to push
  SFT_RAW_RATIO         : 0.8
  SFT_NUM_EPOCHS        : 2
  SFT_LR                : 5e-5
  SFT_MAX_SEQ_LEN       : 1024
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
LEGACY_SFT_PAIR_JSONL = os.environ.get("SFT_PAIR_JSONL", "/workspace/data/sft_pairs.v2.jsonl")
SFT_INPUT_JSONL = os.environ.get("SFT_INPUT_JSONL", LEGACY_SFT_PAIR_JSONL)
SFT_VAL_JSONL = os.environ.get("SFT_VAL_JSONL", "/workspace/data/val_set.v3.jsonl")
OUTPUT_DIR = os.environ.get("SFT_OUTPUT_DIR", "/workspace/out/sft-lora")
CKPT_DIR = os.environ.get("SFT_CKPT_DIR", "/workspace/out/sft-ckpt")
LOG_FILE = os.environ.get("SFT_LOG_FILE", "/workspace/train_sft.log")
HUB_MODEL_ID = os.environ.get("SFT_HUB_MODEL_ID", "").strip() or None

SEED = int(os.environ.get("TRAIN_SEED", "42"))
MAX_SEQ_LEN = int(os.environ.get("SFT_MAX_SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("SFT_GRAD_ACCUM", "16"))
LR = float(os.environ.get("SFT_LR", "5e-5"))
WARMUP_RATIO = float(os.environ.get("SFT_WARMUP_RATIO", "0.05"))
WEIGHT_DECAY = float(os.environ.get("SFT_WEIGHT_DECAY", "0.01"))
NUM_EPOCHS = int(os.environ.get("SFT_NUM_EPOCHS", "2"))
MAX_STEPS_OVERRIDE = int(os.environ.get("SFT_MAX_STEPS", "0") or "0")
SAVE_STEPS = int(os.environ.get("SFT_SAVE_STEPS", "50"))
SAVE_TOTAL_LIMIT = int(os.environ.get("SFT_SAVE_TOTAL_LIMIT", "3"))
EVAL_STEPS = int(os.environ.get("SFT_EVAL_STEPS", "200"))
LOGGING_STEPS = int(os.environ.get("SFT_LOGGING_STEPS", "10"))
RAW_LIMIT_ROWS = int(os.environ.get("SFT_RAW_LIMIT_ROWS", "0") or "0")
INPUT_LIMIT_ROWS = int(
    os.environ.get("SFT_INPUT_LIMIT_ROWS", os.environ.get("SFT_PAIR_LIMIT_ROWS", "0")) or "0"
)
VAL_LIMIT_ROWS = int(os.environ.get("SFT_VAL_LIMIT_ROWS", "0") or "0")

RAW_RATIO = float(os.environ.get("SFT_RAW_RATIO", "0.8"))
PAIR_RATIO = 1.0 - RAW_RATIO
PERSONA_BOOST = float(os.environ.get("SFT_PERSONA_BOOST", "1.0"))
ARGOT_WEIGHT_FLOOR = float(os.environ.get("SFT_LOSS_WEIGHT_ARGOT", "1.5"))

LORA_R = int(os.environ.get("SFT_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("SFT_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("SFT_LORA_DROPOUT", "0.0"))
USE_RSLORA = os.environ.get("SFT_USE_RSLORA", "1") == "1"
USE_DORA = os.environ.get("SFT_USE_DORA", "0") == "1"
FLASH_ATTN = os.environ.get("SFT_FLASH_ATTN", "flash_attention_2")

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


def detect_supervised_row_format(row: dict) -> str:
    if "instruction" in row:
        return "instruction"
    if "post" in row and "comment" in row:
        return "pair"
    return "unknown"


def build_mixed_dataset(tokenizer) -> tuple[Dataset, Dataset | None]:
    rng = random.Random(SEED)

    raw_rows = load_jsonl(SFT_RAW_JSONL)
    supervised_rows = load_jsonl(SFT_INPUT_JSONL)
    val_rows = load_jsonl(SFT_VAL_JSONL)
    if RAW_LIMIT_ROWS > 0 and len(raw_rows) > RAW_LIMIT_ROWS:
        logger.info(f"SFT_RAW_LIMIT_ROWS 적용: {len(raw_rows):,} -> {RAW_LIMIT_ROWS:,}")
        raw_rows = raw_rows[:RAW_LIMIT_ROWS]
    if INPUT_LIMIT_ROWS > 0 and len(supervised_rows) > INPUT_LIMIT_ROWS:
        logger.info(f"SFT_INPUT_LIMIT_ROWS 적용: {len(supervised_rows):,} -> {INPUT_LIMIT_ROWS:,}")
        supervised_rows = supervised_rows[:INPUT_LIMIT_ROWS]
    if VAL_LIMIT_ROWS > 0 and len(val_rows) > VAL_LIMIT_ROWS:
        logger.info(f"SFT_VAL_LIMIT_ROWS 적용: {len(val_rows):,} -> {VAL_LIMIT_ROWS:,}")
        val_rows = val_rows[:VAL_LIMIT_ROWS]

    logger.info(
        f"mix policy: raw={RAW_RATIO:.2f} pair={PAIR_RATIO:.2f}"
    )
    if supervised_rows:
        format_counts: dict[str, int] = {}
        for row in supervised_rows:
            key = detect_supervised_row_format(row)
            format_counts[key] = format_counts.get(key, 0) + 1
        logger.info(f"supervised row formats: {format_counts}")

    # compute pair target count so that pairs / (raw + pairs) == PAIR_RATIO
    if RAW_RATIO >= 1.0 or not supervised_rows:
        supervised_take = 0
    elif RAW_RATIO <= 0.0:
        supervised_take = len(supervised_rows)
        raw_rows = []
    else:
        raw_n = len(raw_rows)
        supervised_take = int(round(raw_n * PAIR_RATIO / max(RAW_RATIO, 1e-6)))
        supervised_take = min(supervised_take, len(supervised_rows))
    logger.info(
        f"sampled sizes: raw={len(raw_rows):,} supervised={supervised_take:,} / "
        f"{len(supervised_rows):,} available"
    )
    rng.shuffle(supervised_rows)
    supervised_rows = supervised_rows[:supervised_take]

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
            "example_weight": 1.0,
        }

    def build_completion_example(prompt: str, completion: str, example_weight: float = 1.0) -> dict | None:
        prompt = prompt.strip()
        completion = completion.strip()
        if not prompt or not completion:
            return None  # type: ignore[return-value]
        prefix_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        response = completion + eos
        resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = (prefix_ids + resp_ids)[:MAX_SEQ_LEN]
        attn = [1] * len(input_ids)
        prompt_len = min(len(prefix_ids), len(input_ids))
        labels = [-100] * prompt_len
        labels += input_ids[prompt_len:]
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "example_weight": float(example_weight),
        }

    def build_instruction_example(row: dict) -> dict | None:
        instruction = row.get("instruction", "").strip()
        input_text = row.get("input", "").strip()
        prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
        completion = row.get("output", "").strip()
        return build_completion_example(prompt, completion, compute_example_weight(row))

    def build_pair_example(row: dict) -> dict | None:
        post = row.get("post", "").strip()
        comment = row.get("comment", "").strip()
        prompt = post + "\n"
        return build_completion_example(prompt, comment, compute_example_weight(row))

    def build_supervised_example(row: dict) -> dict | None:
        row_format = detect_supervised_row_format(row)
        if row_format == "instruction":
            return build_instruction_example(row)
        if row_format == "pair":
            return build_pair_example(row)
        return None

    def compute_example_weight(row: dict) -> float:
        weight = float(row.get("loss_weight", 1.0) or 1.0)
        if weight > 1.0:
            weight = max(weight, ARGOT_WEIGHT_FLOOR)
        if row.get("persona_id"):
            weight *= max(PERSONA_BOOST, 0.0)
        return max(weight, 0.0)

    merged: list[dict] = []
    for row in raw_rows:
        merged.append(build_raw_example(row))
    for row in supervised_rows:
        ex = build_supervised_example(row)
        if ex is not None:
            merged.append(ex)
    rng.shuffle(merged)

    if not merged:
        raise RuntimeError("no SFT training examples built from configured inputs")

    logger.info(f"merged train size: {len(merged):,}")
    weighted_examples = [row["example_weight"] for row in merged if row.get("example_weight", 1.0) != 1.0]
    if weighted_examples:
        logger.info(
            "example_weight overrides: %s rows, min=%.2f max=%.2f avg=%.2f",
            len(weighted_examples),
            min(weighted_examples),
            max(weighted_examples),
            sum(weighted_examples) / len(weighted_examples),
        )
    train_ds = Dataset.from_list(merged)

    eval_ds = None
    if val_rows:
        eval_examples: list[dict] = []
        for row in val_rows:
            if row.get("text"):
                eval_examples.append(build_raw_example(row))
                continue
            ex = build_supervised_example(row)
            if ex is not None:
                eval_examples.append(ex)
        eval_ds = Dataset.from_list(eval_examples)
        logger.info(f"val size: {len(eval_ds):,}")

    return train_ds, eval_ds


class ExampleWeightCollator:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        weights = [float(feature.pop("example_weight", 1.0)) for feature in features]
        batch = self.base_collator(features)
        batch["example_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        example_weight = inputs.pop("example_weight", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if labels is None or not hasattr(outputs, "logits"):
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(-100)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view_as(shift_labels)

        seq_token_counts = valid_mask.sum(dim=1).clamp(min=1)
        seq_loss = (token_losses * valid_mask).sum(dim=1) / seq_token_counts

        if example_weight is not None:
            example_weight = example_weight.to(seq_loss.device, dtype=seq_loss.dtype)
            loss = (seq_loss * example_weight).sum() / example_weight.sum().clamp(min=1e-8)
        else:
            loss = seq_loss.mean()

        return (loss, outputs) if return_outputs else loss


def main() -> None:
    start = time.time()
    logger.info("=" * 60)
    logger.info("SFT v2 시작")
    logger.info(f"  base          : {BASE_MODEL}")
    logger.info(f"  raw_jsonl     : {SFT_RAW_JSONL}")
    logger.info(f"  input_jsonl   : {SFT_INPUT_JSONL}")
    logger.info(f"  val_jsonl     : {SFT_VAL_JSONL}")
    logger.info(f"  output        : {OUTPUT_DIR}")
    logger.info(f"  hub_model_id  : {HUB_MODEL_ID or '(disabled)'}")
    logger.info(
        f"  LoRA r={LORA_R} α={LORA_ALPHA} rsLoRA={USE_RSLORA} DoRA={USE_DORA}"
    )
    logger.info(
        f"  seq_len={MAX_SEQ_LEN} eff_batch={BATCH_SIZE*GRAD_ACCUM} "
        f"lr={LR} epochs={NUM_EPOCHS}"
    )
    logger.info(
        f"  example_weight persona_boost={PERSONA_BOOST:.2f} argot_floor={ARGOT_WEIGHT_FLOOR:.2f}"
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

    data_collator = ExampleWeightCollator(
        DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
        )
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
        dataloader_num_workers=0,
        group_by_length=False,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
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

    trainer = WeightedLossTrainer(
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
