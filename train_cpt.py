#!/usr/bin/env python3
"""
train_cpt.py — Solar 10.7B CPT (Continued Pretraining) with QLoRA.

왜 이 설정인가:
- Solar 10.7B: 한국어 성능 우수 + Llama-2 아키텍처 호환 → PEFT LoRA 바로 적용 가능
- QLoRA 4bit: A100 80GB 기준 10.7B full fine-tune 불가, 4bit NF4 양자화로 ~16GB VRAM 사용
- r=32, alpha=64: r=16보다 표현력 높지만 r=64보다 VRAM 절약 (실험적 sweet spot)
- max_seq_len=1024: 코퍼스 평균 문서 길이 ~120자 ≈ 60토큰, 패딩 낭비 최소화하면서 긴 문서 처리
- grad_accum=16: batch_size=1 × 16 = effective 16, OOM 방지
- 예상 학습 시간: ~18시간 @ A100 80GB ($1.19/hr → ~$21.42)
- 예상 비용: $21 (1 epoch, 2.28M tokens, seq_len=1024, eff_batch=16)

실행 위치: RunPod pod 내 /workspace/
  python train_cpt.py

의존성 (pod 시작 후 설치):
  pip install transformers>=4.40 peft>=0.10 bitsandbytes>=0.43 datasets accelerate tqdm
"""

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# ── 의존성 확인 ──────────────────────────────────────────────────────────────
try:
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
    print(f"[오류] 필수 패키지 미설치: {e}")
    print("다음 명령으로 설치하세요:")
    print("  pip install transformers>=4.40 peft>=0.10 bitsandbytes>=0.43 datasets accelerate tqdm")
    sys.exit(1)

# ── 설정 ──────────────────────────────────────────────────────────────────────
BASE_MODEL = os.environ.get("BASE_MODEL", "upstage/SOLAR-10.7B-v1.0")          # HuggingFace 모델 ID
INPUT_JSONL = os.environ.get("INPUT_JSONL", "/workspace/data/cpt_corpus.jsonl")  # Phase 1 산출물
OUTPUT_DIR = os.environ.get("CPT_OUTPUT_DIR", "/workspace/out/cpt-lora")            # LoRA adapter 저장
CKPT_DIR = os.environ.get("CPT_CKPT_DIR", "/workspace/out/cpt-ckpt")              # 중간 체크포인트
LOG_FILE = os.environ.get("CPT_LOG_FILE", "/workspace/train_cpt.log")

MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 16          # effective batch = 16
LR = float(os.environ.get("CPT_LR", "1e-4"))
WARMUP_RATIO = 0.03
NUM_EPOCHS = int(os.environ.get("CPT_NUM_EPOCHS", "1"))
SAVE_STEPS = 500
LOGGING_STEPS = 20

# LoRA 설정 — Solar / LLaMA-2 아키텍처 전체 attention + FFN 레이어 타겟
LORA_R = 32
LORA_ALPHA = 64          # alpha = 2r → 학습률 스케일 안정
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── 로깅 설정 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_corpus(path: str) -> Dataset:
    """
    cpt_corpus.jsonl 로드 → HuggingFace Dataset.
    각 줄: {"text": "...", "kind": "comment"|"post"}
    """
    records = []
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
    logger.info(f"코퍼스 로드: {len(records)}건 (스킵 {skipped}건) ← {path}")
    return Dataset.from_list(records)


def tokenize_fn(examples: dict, tokenizer) -> dict:
    """
    텍스트 토크나이즈 + max_seq_len 절단.
    labels = input_ids (언어 모델링 목적)
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main() -> None:
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("CPT 학습 시작")
    logger.info(f"  모델: {BASE_MODEL}")
    logger.info(f"  입력: {INPUT_JSONL}")
    logger.info(f"  출력: {OUTPUT_DIR}")
    logger.info(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    logger.info(f"  seq_len={MAX_SEQ_LEN}, eff_batch={BATCH_SIZE * GRAD_ACCUM}")
    logger.info(f"  예상 소요: ~18시간 / 예상 비용: ~$21 @ A100 80GB")
    logger.info("=" * 60)

    # CUDA 확인
    if not torch.cuda.is_available():
        logger.warning("CUDA를 사용할 수 없습니다. CPU 학습은 매우 느립니다.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({vram_gb:.1f}GB)")

    # ── 출력 디렉토리 생성 ────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── 토크나이저 로드 ───────────────────────────────────────────────────────
    logger.info("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )
    # Solar/LLaMA 계열: pad 토큰이 없으면 eos로 대체
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info(f"토크나이저 vocab size: {tokenizer.vocab_size}")

    # ── 모델 로드 (4bit QLoRA) ────────────────────────────────────────────────
    logger.info("모델 로드 중 (QLoRA 4bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NF4: 정규분포 가중치에 최적화된 양자화
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16: A100에서 성능/정밀도 균형
        bnb_4bit_use_double_quant=True,     # double quantization: VRAM 추가 절약
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False          # gradient checkpointing과 호환
    model.config.pretraining_tp = 1

    # ── kbit 학습 준비 ────────────────────────────────────────────────────────
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,    # VRAM 절약 (속도 ~20% 감소)
    )

    # ── LoRA 적용 ─────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 데이터 로드 및 토크나이즈 ─────────────────────────────────────────────
    logger.info("데이터 로드 및 토크나이즈 중...")
    raw_dataset = load_corpus(INPUT_JSONL)

    tokenized = raw_dataset.map(
        lambda examples: tokenize_fn(examples, tokenizer),
        batched=True,
        batch_size=1000,
        remove_columns=["text"],
        desc="토크나이즈",
    )

    # 전체 토큰 수 추정
    total_tokens = sum(len(x) for x in tokenized["input_ids"])
    logger.info(f"토크나이즈 완료: {len(tokenized)}건, ~{total_tokens:,} tokens")

    # ── 학습 steps 계산 ──────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(tokenized) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    logger.info(f"steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup={warmup_steps}")

    # ── 데이터 콜레이터 ───────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,          # CLM (인과 언어 모델링)
        pad_to_multiple_of=8,
    )

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        save_steps=SAVE_STEPS,
        save_total_limit=3,             # 체크포인트 최대 3개 보존
        logging_steps=LOGGING_STEPS,
        logging_dir=f"{CKPT_DIR}/logs",
        report_to="none",               # wandb 미사용 (없으면 오류 방지)
        fp16=False,
        bf16=True,                      # A100은 bfloat16 native 지원
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        group_by_length=True,           # 비슷한 길이 묶기 → 패딩 최소화
        remove_unused_columns=False,
        optim="paged_adamw_32bit",      # bitsandbytes paged optimizer: OOM 방지
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # ── 체크포인트 재개 지원 ──────────────────────────────────────────────────
    resume_from = None
    ckpt_path = Path(CKPT_DIR)
    existing_ckpts = sorted(ckpt_path.glob("checkpoint-*"), key=os.path.getmtime)
    if existing_ckpts:
        resume_from = str(existing_ckpts[-1])
        logger.info(f"체크포인트 재개: {resume_from}")

    # ── 학습 실행 ─────────────────────────────────────────────────────────────
    logger.info("학습 시작...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── LoRA adapter 저장 ─────────────────────────────────────────────────────
    logger.info(f"LoRA adapter 저장 → {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    elapsed = (time.time() - start_time) / 3600
    logger.info(f"CPT 학습 완료. 소요: {elapsed:.2f}시간")
    logger.info(f"결과물: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
