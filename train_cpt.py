#!/usr/bin/env python3
"""
train_cpt.py — v2 — Continued pretraining on raw Korean community text.

Recipe (2026-04, fixed from 9-voice research consensus):
  - Base:    Qwen/Qwen3-8B-Base       (BBPE 151K, KMMLU 52.54, Apache-2.0)
  - LoRA:    r=64, alpha=64, use_rslora=True, target_modules="all-linear"
             + optional embed_tokens/lm_head when CPT_LORA_EMBED=1
  - LR:      1e-4 cosine, warmup 3%, weight_decay 0.01
  - seq_len: 1024
  - batch:   1 x grad_accum 16 (eff 16), epochs=2 by default
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
import re
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
TRAIN_CPT_JSONL = os.environ.get("TRAIN_CPT_JSONL", "").strip()
VAL_JSONL = os.environ.get("CPT_VAL_JSONL", "/workspace/data/val_set.v2.jsonl")
OUTPUT_DIR = os.environ.get("CPT_OUTPUT_DIR", "/workspace/out/cpt-lora")
CKPT_DIR = os.environ.get("CPT_CKPT_DIR", "/workspace/out/cpt-ckpt")
LOG_FILE = os.environ.get("CPT_LOG_FILE", "/workspace/train_cpt.log")
HUB_MODEL_ID = os.environ.get("CPT_HUB_MODEL_ID", "").strip() or None
CPT_TOKENIZER_DIR = os.environ.get("CPT_TOKENIZER_DIR", "").strip()
CPT_EXTEND_TOKENS = os.environ.get("CPT_EXTEND_TOKENS", "0") == "1"

SEED = int(os.environ.get("TRAIN_SEED", "42"))
MAX_SEQ_LEN = int(os.environ.get("CPT_MAX_SEQ_LEN", "1024"))
BATCH_SIZE = int(os.environ.get("CPT_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("CPT_GRAD_ACCUM", "16"))
LR = float(os.environ.get("CPT_LR", "1e-4"))
WARMUP_RATIO = float(os.environ.get("CPT_WARMUP_RATIO", "0.03"))
WEIGHT_DECAY = float(os.environ.get("CPT_WEIGHT_DECAY", "0.01"))
NUM_EPOCHS = int(os.environ.get("CPT_NUM_EPOCHS", "2"))
MAX_STEPS_OVERRIDE = int(os.environ.get("CPT_MAX_STEPS", "0") or "0")
SAVE_STEPS = int(os.environ.get("CPT_SAVE_STEPS", "50"))
SAVE_TOTAL_LIMIT = int(os.environ.get("CPT_SAVE_TOTAL_LIMIT", "3"))
EVAL_STEPS = int(os.environ.get("CPT_EVAL_STEPS", "200"))
LOGGING_STEPS = int(os.environ.get("CPT_LOGGING_STEPS", "20"))
LIMIT_ROWS = int(os.environ.get("CPT_LIMIT_ROWS", "0") or "0")
VAL_LIMIT_ROWS = int(os.environ.get("CPT_VAL_LIMIT_ROWS", "0") or "0")

LORA_R = int(os.environ.get("CPT_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("CPT_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("CPT_LORA_DROPOUT", "0.0"))
USE_RSLORA = os.environ.get("CPT_USE_RSLORA", "1") == "1"
USE_DORA = os.environ.get("CPT_USE_DORA", "0") == "1"
USE_LORA_EMBED = os.environ.get("CPT_LORA_EMBED", "0") == "1"

FLASH_ATTN = os.environ.get("CPT_FLASH_ATTN", "flash_attention_2")
STRUCTURED_TOKEN_RE = re.compile(
    r"<\|(?:post|/post|thread|/thread|/comment)\|>|<\|comment depth=\d+\|>"
)
TOKENIZER_MANIFEST_FILENAME = "token_list.json"
STRUCTURED_SPECIAL_TOKENS = [
    "<|post|>",
    "<|/post|>",
    "<|comment depth=0|>",
    "<|comment depth=1|>",
    "<|comment depth=2|>",
    "<|comment depth=3|>",
    "<|comment depth=4|>",
    "<|comment depth=5|>",
    "<|/comment|>",
    "<|thread|>",
    "<|/thread|>",
]

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


def resolve_lora_target_modules(model) -> str | list[str]:
    if not USE_LORA_EMBED:
        return "all-linear"

    linear_types: list[type] = [torch.nn.Linear]
    try:
        import bitsandbytes as bnb

        for name in ("Linear4bit", "Linear8bitLt"):
            cls = getattr(bnb.nn, name, None)
            if cls is not None:
                linear_types.append(cls)
    except Exception as exc:
        logger.warning(f"bitsandbytes linear class inspection 실패, torch.nn.Linear만 사용: {exc}")

    target_modules: set[str] = set()
    for module_name, module in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        if isinstance(module, tuple(linear_types)):
            target_modules.add(leaf_name)
        if leaf_name == "embed_tokens" and isinstance(module, torch.nn.Embedding):
            target_modules.add(leaf_name)

    # Keep the default all-linear path unless the caller explicitly asks for
    # extra adaptation on token embeddings / output head.
    target_modules.add("lm_head")
    resolved = sorted(target_modules)
    logger.info("CPT_LORA_EMBED=1 -> target_modules=%s", ",".join(resolved))
    return resolved


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


def resolve_train_jsonl() -> str:
    return TRAIN_CPT_JSONL or INPUT_JSONL


def dedupe_preserve_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def extract_text_candidate(raw_line: str) -> str:
    try:
        obj = json.loads(raw_line)
    except json.JSONDecodeError:
        return raw_line
    if isinstance(obj, dict):
        return str(obj.get("text", ""))
    return raw_line


def dataset_contains_structured_tokens(path: str, sample_size: int = 256) -> bool:
    if not path or not os.path.exists(path):
        return False

    checked = 0
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            text = extract_text_candidate(raw_line)
            if STRUCTURED_TOKEN_RE.search(text):
                logger.info(
                    "structured CPT markers detected in %s (sample row %d)",
                    path,
                    checked + 1,
                )
                return True
            checked += 1
            if checked >= sample_size:
                break
    return False


def tokenizer_bundle_exists(tokenizer_path: Path) -> bool:
    primary_bundle = (
        (tokenizer_path / "tokenizer.json").exists()
        or (tokenizer_path / "vocab.json").exists()
    )
    config_files = (
        (tokenizer_path / "tokenizer_config.json").exists()
        and (tokenizer_path / "special_tokens_map.json").exists()
    )
    return tokenizer_path.is_dir() and primary_bundle and config_files


def load_runtime_special_tokens() -> list[str]:
    tokens = list(STRUCTURED_SPECIAL_TOKENS)
    manifest_candidates: list[Path] = []
    if CPT_TOKENIZER_DIR:
        manifest_candidates.append(Path(CPT_TOKENIZER_DIR) / TOKENIZER_MANIFEST_FILENAME)
    manifest_candidates.append(Path("v3-data/tokenizer") / TOKENIZER_MANIFEST_FILENAME)

    for manifest_path in manifest_candidates:
        if not manifest_path.exists():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("tokenizer manifest parse 실패 (%s): %s", manifest_path, exc)
            continue

        extra_tokens: list[str] = []
        if isinstance(payload, dict):
            extra_tokens = [
                str(token)
                for token in payload.get("additional_special_tokens", [])
                if str(token).strip()
            ]
        elif isinstance(payload, list):
            extra_tokens = [str(token) for token in payload if str(token).strip()]

        if extra_tokens:
            logger.info(
                "runtime tokenizer manifest loaded: %s (%d tokens)",
                manifest_path,
                len(extra_tokens),
            )
            tokens.extend(extra_tokens)
            break

    return dedupe_preserve_order(tokens)


def resolve_tokenizer_source(train_jsonl: str) -> tuple[str, bool]:
    structured_tokens_present = dataset_contains_structured_tokens(train_jsonl)

    if CPT_TOKENIZER_DIR:
        tokenizer_path = Path(CPT_TOKENIZER_DIR)
        if tokenizer_bundle_exists(tokenizer_path):
            if structured_tokens_present:
                logger.info(
                    "structured CPT detected; loading tokenizer from CPT_TOKENIZER_DIR=%s",
                    CPT_TOKENIZER_DIR,
                )
            else:
                logger.info(
                    "CPT_TOKENIZER_DIR is set without structured-token detection; using %s",
                    CPT_TOKENIZER_DIR,
                )
            return CPT_TOKENIZER_DIR, structured_tokens_present

        if CPT_EXTEND_TOKENS:
            logger.warning(
                "CPT_TOKENIZER_DIR=%s is missing a tokenizer bundle; "
                "falling back to BASE_MODEL with runtime token extension",
                CPT_TOKENIZER_DIR,
            )
            return BASE_MODEL, structured_tokens_present

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"CPT_TOKENIZER_DIR does not exist: {CPT_TOKENIZER_DIR}"
            )
        raise RuntimeError(
            "CPT_TOKENIZER_DIR exists but does not contain a tokenizer bundle. "
            "Run scripts/extend_tokenizer_v3.py or set CPT_EXTEND_TOKENS=1."
        )

    if structured_tokens_present and not CPT_EXTEND_TOKENS:
        raise RuntimeError(
            "Structured CPT markers detected in TRAIN_CPT_JSONL/INPUT_JSONL but "
            "CPT_TOKENIZER_DIR is not set. Run scripts/extend_tokenizer_v3.py and "
            "export CPT_TOKENIZER_DIR, or set CPT_EXTEND_TOKENS=1 to extend the "
            "base tokenizer at runtime before training."
        )

    return BASE_MODEL, structured_tokens_present


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
    train_jsonl = resolve_train_jsonl()
    logger.info("=" * 60)
    logger.info("CPT v2 시작")
    logger.info(f"  base         : {BASE_MODEL}")
    logger.info(f"  input        : {train_jsonl}")
    logger.info(f"  val          : {VAL_JSONL}")
    logger.info(f"  output       : {OUTPUT_DIR}")
    logger.info(f"  hub_model_id : {HUB_MODEL_ID or '(disabled)'}")
    logger.info(f"  tokenizer_dir: {CPT_TOKENIZER_DIR or '(base tokenizer)'}")
    logger.info(f"  extend_tokens: {CPT_EXTEND_TOKENS}")
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
    tokenizer_source, structured_tokens_present = resolve_tokenizer_source(train_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True, use_fast=True
    )
    if CPT_EXTEND_TOKENS:
        runtime_special_tokens = load_runtime_special_tokens()
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": runtime_special_tokens}
        )
        logger.info(
            "runtime tokenizer extension enabled: requested=%s added=%s total_vocab=%s",
            len(runtime_special_tokens),
            added,
            len(tokenizer),
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(
        "tokenizer size: base_vocab=%s total_vocab=%s structured=%s source=%s",
        tokenizer.vocab_size,
        len(tokenizer),
        structured_tokens_present,
        tokenizer_source,
    )

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

    input_embedding_count = model.get_input_embeddings().num_embeddings
    if len(tokenizer) != input_embedding_count:
        logger.info(
            "resize_token_embeddings: %s -> %s",
            input_embedding_count,
            len(tokenizer),
        )
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    target_modules = resolve_lora_target_modules(model)
    logger.info(f"LoRA target_modules={target_modules}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        use_rslora=USE_RSLORA,
        use_dora=USE_DORA,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Data ────────────────────────────────────────────────────────
    train_raw = load_corpus(train_jsonl)
    eval_raw = load_corpus(VAL_JSONL) if os.path.exists(VAL_JSONL) else None
    if LIMIT_ROWS > 0 and len(train_raw) > LIMIT_ROWS:
        logger.info(f"CPT_LIMIT_ROWS 적용: {len(train_raw):,} -> {LIMIT_ROWS:,}")
        train_raw = train_raw.select(range(LIMIT_ROWS))
    if eval_raw is not None and VAL_LIMIT_ROWS > 0 and len(eval_raw) > VAL_LIMIT_ROWS:
        logger.info(f"CPT_VAL_LIMIT_ROWS 적용: {len(eval_raw):,} -> {VAL_LIMIT_ROWS:,}")
        eval_raw = eval_raw.select(range(VAL_LIMIT_ROWS))

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
    if MAX_STEPS_OVERRIDE > 0:
        logger.info(f"CPT_MAX_STEPS override 적용: {total_steps} -> {MAX_STEPS_OVERRIDE}")
        total_steps = MAX_STEPS_OVERRIDE
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
        group_by_length=True,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        weight_decay=WEIGHT_DECAY,
        ddp_find_unused_parameters=False,
        seed=SEED,
        # data_seed intentionally omitted — transformers 4.51 requires
        # accelerate>=1.1 for it, but our pinned accelerate is 0.34.2
        # (required by trl 0.12.1). When data_seed is unset, transformers
        # falls back to `seed` for the data sampler, so reproducibility
        # is preserved.
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
