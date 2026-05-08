#!/usr/bin/env python3
"""
train_sft.py — unified — Supervised fine-tune on CPT-merged base with
reply-pair and raw-continuation mixed data.

This file is the consolidation of two PRs:
  - PR #7 (base, anchor): loss_weight oversampling + schema-branching
    build_supervised_example (instruction/output OR post/comment).
  - PR #5 (overlay): DoRA + chatml prompt format + thread-aware v3 loader
    + completion-only loss masking. All overlays are env-gated and OFF by
    default — defaults stay on PR #7 so existing recipes do not destabilise.

Recipe (defaults preserve PR #7 stability):
  - r=64 rsLoRA, lr=5e-5, 2 epochs, seq_len=1024
  - Base: CPT-merged fp16 (NOT the stacking pattern)
  - Format: "raw"   (post + comment, mask on post)   <-- default
  - Format: "chatml" (Qwen3 chat template, completion-only loss) <-- opt-in
  - DoRA: opt-in via SFT_USE_DORA=1
  - v3 loader: opt-in via SFT_PAIR_JSONL_V3=<path>
  - NEFTune off (style transfer hurts from embedding noise)
  - Full seed path

Env:
  BASE_MODEL            : /workspace/out/cpt-merged-fp16 (merged)
  SFT_RAW_JSONL         : /workspace/data/cpt_corpus.v2.jsonl
  SFT_PAIR_JSONL        : /workspace/data/sft_pairs.v2.jsonl   (v2 schema)
  SFT_PAIR_JSONL_V3     : (opt) /workspace/data/sft_pairs.v3.jsonl  (v3 schema)
  SFT_VAL_JSONL         : /workspace/data/val_set.v2.jsonl
  SFT_OUTPUT_DIR        : /workspace/out/sft-lora
  SFT_CKPT_DIR          : /workspace/out/sft-ckpt
  SFT_HUB_MODEL_ID      : (optional) HF repo to push
  SFT_RAW_RATIO         : 0.8
  SFT_NUM_EPOCHS        : 2
  SFT_LR                : 5e-5
  SFT_MAX_SEQ_LEN       : 1024
  SFT_USE_DORA          : 0   (opt-in)
  SFT_PROMPT_FORMAT     : raw  ("chatml" enables Qwen3 chat template path)
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
# Tokenizer path: explicit override > local tokenizer_v4 dir > BASE_MODEL.
# tokenizer_v4 contains the +210 domain tokens used end-to-end.
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH") or (
    "tokenizer_v4" if os.path.isdir("tokenizer_v4") else BASE_MODEL
)
SFT_RAW_JSONL = os.environ.get("SFT_RAW_JSONL", "/workspace/data/cpt_corpus.v2.jsonl")
SFT_PAIR_JSONL = os.environ.get("SFT_PAIR_JSONL", "/workspace/data/sft_pairs.v2.jsonl")
# from PR #5 — v3 loader opt-in (empty string disables)
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
# Default lr stays 5e-5 (PR #7). PR #5's 1e-4 is opt-in via SFT_LR=1e-4.
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
PAIR_LIMIT_ROWS = int(os.environ.get("SFT_PAIR_LIMIT_ROWS", "0") or "0")
VAL_LIMIT_ROWS = int(os.environ.get("SFT_VAL_LIMIT_ROWS", "0") or "0")

RAW_RATIO = float(os.environ.get("SFT_RAW_RATIO", "0.8"))
PAIR_RATIO = 1.0 - RAW_RATIO

LORA_R = int(os.environ.get("SFT_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("SFT_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("SFT_LORA_DROPOUT", "0.0"))
USE_RSLORA = os.environ.get("SFT_USE_RSLORA", "1") == "1"
# Default DoRA OFF (PR #7); PR #5's DoRA is opt-in via SFT_USE_DORA=1.
USE_DORA = os.environ.get("SFT_USE_DORA", "0") == "1"
FLASH_ATTN = os.environ.get("SFT_FLASH_ATTN", "flash_attention_2")

# Prompt format: "raw" (PR #7 default, post+comment) or "chatml" (PR #5 opt-in,
# Qwen3 chat template with completion-only loss masking).
PROMPT_FORMAT = os.environ.get("SFT_PROMPT_FORMAT", "raw").strip().lower()

# chatml system prompt (Korean nightlife community persona) — used only when
# PROMPT_FORMAT == "chatml".  Source: PR #5.
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


# ── chatml prompt builder (from PR #5) ────────────────────────────────

def _build_chatml_user_text(row: dict) -> str:
    """Build the user-turn content from a v3 schema row.

    Source: PR #5.
    """
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

    Loss mask: everything before the assistant turn is set to -100. Only the
    assistant content + closing <|im_end|> is trained.

    Uses tokenizer.apply_chat_template (Qwen3 native) exclusively. The Qwen3
    chat template emits a `<think></think>` block for assistant turns; the
    previously-present manual string fallback did NOT, producing inconsistent
    label structure across rows. Per 3-model audit HIGH: any tokenizer with
    chat capability has apply_chat_template — if it doesn't, the run must
    fail loudly rather than silently fall back.

    Source: PR #5 (fallback removed).
    """
    if is_v3:
        target = row.get("target_comment", "").strip()
    else:
        target = row.get("comment", "").strip()

    if not target:
        return None

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

    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "tokenizer lacks apply_chat_template — refusing to fall back to "
            "manual ChatML which is inconsistent with the Qwen3 native template "
            "(missing <think></think> block). Use tokenizer_v4 or any tokenizer "
            "exposing apply_chat_template."
        )

    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"tokenizer.apply_chat_template failed: {exc!r}. The manual "
            "fallback was removed (3-model audit HIGH) because it omitted the "
            "<think></think> block emitted by the native template, causing "
            "label-structure drift across rows."
        ) from exc

    if full_text is None:
        raise RuntimeError(
            "tokenizer.apply_chat_template returned None; cannot build example."
        )

    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]

    assistant_marker = "<|im_start|>assistant\n"
    prompt_end_idx = full_text.rfind(assistant_marker)
    if prompt_end_idx == -1:
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

    attn = [1] * len(full_ids)
    return {"input_ids": full_ids, "attention_mask": attn, "labels": labels}


def build_mixed_dataset(tokenizer) -> tuple[Dataset, Dataset | None]:
    rng = random.Random(SEED)

    raw_rows = load_jsonl(SFT_RAW_JSONL)

    # PR #5 v3 loader — when SFT_PAIR_JSONL_V3 is set, use v3 schema
    # (P1-A output: post_title / post_body_excerpt / parent_comment /
    # target_comment / depth / task_type). Otherwise fall back to v2.
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

    # compute supervised target count so that pairs / (raw + pairs) == PAIR_RATIO
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

    # PR #7 loss_weight oversampling — preserved. v3 rows may also carry a
    # loss_weight field; if present and > 1.0 the row is duplicated. v2 rows
    # work the same way.
    weighted_extra = [
        row for row in pair_rows
        if float(row.get("loss_weight", 1.0) or 1.0) > 1.0
    ]
    if weighted_extra:
        pair_rows.extend(weighted_extra)
        rng.shuffle(pair_rows)
        logger.info(
            f"loss_weight oversampling: +{len(weighted_extra):,} supervised rows"
        )

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

    def build_supervised_example(row: dict) -> dict | None:
        """PR #7's schema-branching builder — handles instruction/output OR
        post/comment. v3 rows also use this when PROMPT_FORMAT='raw' via the
        synthetic post/comment shim below.
        """
        if row.get("instruction") is not None or row.get("output") is not None:
            instruction = str(row.get("instruction") or "").strip()
            input_text = str(row.get("input") or "").strip()
            response_text = str(row.get("output") or "").strip()
            if not instruction or not response_text:
                return None
            prefix = instruction
            if input_text:
                prefix += "\n" + input_text
            prefix += "\n"
            response = response_text + eos
        else:
            post = str(row.get("post") or "").strip()
            comment = str(row.get("comment") or "").strip()
            if not post or not comment:
                return None
            prefix = post + "\n"
            response = comment + eos
        if not response.strip():
            return None
        prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
        resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = (prefix_ids + resp_ids)[:MAX_SEQ_LEN]
        attn = [1] * len(input_ids)
        labels = [-100] * min(len(prefix_ids), len(input_ids))
        labels += input_ids[len(labels):]
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    def build_pair_example(row: dict) -> dict | None:
        if PROMPT_FORMAT == "chatml":
            return build_chatml_example(row, tokenizer, is_v3=use_v3)
        # raw format
        if use_v3:
            # v3 row but raw format requested — synthesise v2-style fields
            post = (row.get("post_title", "") + "\n" + row.get("post_body_excerpt", "")).strip()
            comment = row.get("target_comment", "").strip()
            synthetic = {"post": post, "comment": comment}
            return build_supervised_example(synthetic)
        return build_supervised_example(row)

    merged: list[dict] = []
    for row in raw_rows:
        merged.append(build_raw_example(row))
    for row in pair_rows:
        ex = build_pair_example(row)
        if ex is not None:
            merged.append(ex)
    if pair_rows and not any(
        label != -100
        for ex in merged[len(raw_rows):]
        for label in ex.get("labels", [])
    ):
        raise RuntimeError("SFT supervised rows produced zero trainable response tokens")
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
    logger.info("SFT 시작")
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

    tok_path = TOKENIZER_PATH
    logger.info(f"loading tokenizer from: {tok_path}")
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
