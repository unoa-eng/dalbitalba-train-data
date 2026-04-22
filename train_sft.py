#!/usr/bin/env python3
"""
train_sft.py — Solar 10.7B SFT (Supervised Fine-Tuning) + CAI mix.

왜 이 설정인가:
- CPT adapter 위에 SFT: 도메인 지식(CPT) → 대화 형식 학습(SFT) 순서가 표준 파이프라인
- alpaca-style 템플릿: Solar/LLaMA 계열의 검증된 instruction 포맷, 커스텀 토큰 불필요
- CAI mix (sft:cai = 2:1): 헌법적 AI 학습으로 안전성·일관성 강화, 비율은 Anthropic CAI 논문 기준
- 3 epoch: SFT는 데이터가 CPT보다 적어(15K) overfitting 전 3회가 최적 (경험칙)
- lr=5e-5: CPT(1e-4)보다 낮게 — 이미 학습된 도메인 지식 유지하면서 미세조정
- 예상 학습 시간: ~5시간 @ A100 80GB ($1.19/hr → ~$5.95)

실행 위치: RunPod pod 내 /workspace/
  python train_sft.py

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

try:
    import torch
    from datasets import Dataset, concatenate_datasets
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    print(f"[오류] 필수 패키지 미설치: {e}")
    print("  pip install transformers>=4.40 peft>=0.10 bitsandbytes>=0.43 datasets accelerate tqdm")
    sys.exit(1)

# ── 설정 ──────────────────────────────────────────────────────────────────────
BASE_MODEL = "upstage/SOLAR-10.7B-v1.0"
CPT_LORA_DIR = "/workspace/out/cpt-lora"        # Phase 2 산출물 (CPT adapter)
SFT_JSONL = "/workspace/data/sft_pairs_v2.jsonl"  # v2: post+comment 보강 pairs
CAI_JSONL = "/workspace/data/cai_pairs.jsonl"   # 5K CAI triples (없으면 SFT만 사용)
OUTPUT_DIR = "/workspace/out/sft-lora"
CKPT_DIR = "/workspace/out/sft-ckpt"
LOG_FILE = "/workspace/train_sft.log"

MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 5e-5
WARMUP_RATIO = 0.05
NUM_EPOCHS = 3
SAVE_STEPS = 200
LOGGING_STEPS = 10

# SFT:CAI 샘플 비율 (CAI가 존재할 경우)
CAI_RATIO = 0.33   # 전체의 33% ≈ sft:cai = 2:1

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Alpaca 채팅 템플릿 ────────────────────────────────────────────────────────
# Solar/LLaMA 계열 표준 instruction 포맷
ALPACA_PROMPT = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)
ALPACA_PROMPT_NO_INPUT = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)


def format_alpaca(record: dict) -> str:
    """
    sft_pairs / cai_pairs → alpaca 포맷 문자열.

    sft_pairs 예상 스키마:
      {"instruction": "...", "input": "...(선택)", "output": "..."}

    cai_pairs 예상 스키마 (CAI triple):
      {"prompt": "...", "initial": "...", "critique": "...", "revision": "..."}
      → revision을 최종 output으로 사용
    """
    # CAI triple 처리
    if "revision" in record:
        instruction = record.get("prompt", "")
        output = record.get("revision", "")
        inp = ""
    else:
        instruction = record.get("instruction", "")
        inp = record.get("input", "").strip()
        output = record.get("output", "")

    if inp:
        return ALPACA_PROMPT.format(instruction=instruction, input=inp, output=output)
    else:
        return ALPACA_PROMPT_NO_INPUT.format(instruction=instruction, output=output)


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


def load_jsonl(path: str) -> list[dict]:
    """JSONL 파일 로드 → list of dict."""
    records = []
    if not os.path.exists(path):
        logger.warning(f"파일 없음 (스킵): {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info(f"  로드: {len(records)}건 ← {path}")
    return records


def build_dataset(tokenizer) -> Dataset:
    """
    SFT + CAI 데이터를 합쳐 tokenized Dataset 반환.

    라벨 마스킹: instruction 부분은 -100으로 마스킹 (loss 미산정),
    response 부분만 loss 계산 → 진짜 SFT.
    """
    sft_records = load_jsonl(SFT_JSONL)
    cai_records = load_jsonl(CAI_JSONL)

    # CAI 샘플 수 결정 (SFT:CAI = 2:1 비율 유지)
    if cai_records:
        n_cai = min(len(cai_records), int(len(sft_records) * CAI_RATIO / (1 - CAI_RATIO)))
        cai_records = cai_records[:n_cai]
        logger.info(f"CAI 샘플 {n_cai}건 사용 (전체 {len(sft_records) + n_cai}건, 비율={CAI_RATIO:.0%})")

    all_records = sft_records + cai_records

    # alpaca 포맷 변환
    formatted = [format_alpaca(r) for r in all_records]

    def tokenize_with_label_mask(examples):
        """
        Instruction 부분 loss 마스킹:
        "### Response:\n" 이후만 label = input_id, 이전은 -100.
        """
        input_ids_list = []
        labels_list = []
        attention_masks = []

        RESPONSE_TOKEN = "### Response:\n"

        for text in examples["text"]:
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding=False,
                add_special_tokens=True,
            )
            ids = tokenized["input_ids"]
            attn = tokenized["attention_mask"]

            # Response 시작 위치 찾기 (토큰 수준)
            resp_ids = tokenizer.encode(RESPONSE_TOKEN, add_special_tokens=False)
            resp_len = len(resp_ids)
            response_start = -1
            for i in range(len(ids) - resp_len + 1):
                if ids[i : i + resp_len] == resp_ids:
                    response_start = i + resp_len
                    break

            # 라벨 마스킹
            labels = [-100] * len(ids)
            if response_start > 0:
                labels[response_start:] = ids[response_start:]
            else:
                # Response 토큰을 찾지 못한 경우 전체 학습 (안전 폴백)
                labels = ids.copy()

            input_ids_list.append(ids)
            labels_list.append(labels)
            attention_masks.append(attn)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_masks,
        }

    raw = Dataset.from_list([{"text": t} for t in formatted])
    tokenized = raw.map(
        tokenize_with_label_mask,
        batched=True,
        batch_size=500,
        remove_columns=["text"],
        desc="토크나이즈 (라벨 마스킹)",
    )
    return tokenized


def main() -> None:
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("SFT 학습 시작")
    logger.info(f"  base: {BASE_MODEL}")
    logger.info(f"  CPT adapter: {CPT_LORA_DIR}")
    logger.info(f"  SFT data: {SFT_JSONL}")
    logger.info(f"  CAI data: {CAI_JSONL}")
    logger.info(f"  출력: {OUTPUT_DIR}")
    logger.info(f"  epochs={NUM_EPOCHS}, lr={LR}, eff_batch={BATCH_SIZE * GRAD_ACCUM}")
    logger.info(f"  예상 소요: ~5시간 / 예상 비용: ~$6 @ A100 80GB")
    logger.info("=" * 60)

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA 없음 — CPU 실행 (매우 느림)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── 토크나이저 ────────────────────────────────────────────────────────────
    logger.info("토크나이저 로드...")
    # CPT adapter 디렉토리에 tokenizer가 저장돼 있으면 그것 사용
    tok_path = CPT_LORA_DIR if os.path.exists(os.path.join(CPT_LORA_DIR, "tokenizer_config.json")) else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 모델 로드 (4bit) ──────────────────────────────────────────────────────
    logger.info("base 모델 로드 (QLoRA 4bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False

    # ── CPT LoRA adapter 로드 ─────────────────────────────────────────────────
    if os.path.exists(CPT_LORA_DIR):
        logger.info(f"CPT adapter 로드: {CPT_LORA_DIR}")
        # CPT adapter를 기반으로 추가 LoRA 레이어 적용 (adapter stacking)
        # PeftModel로 CPT adapter 로드 후 추가 학습 가능 상태로 전환
        model = PeftModel.from_pretrained(
            base_model,
            CPT_LORA_DIR,
            is_trainable=False,   # CPT adapter는 freeze
        )
        # SFT용 새 LoRA adapter 추가
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        sft_lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model.add_adapter("sft", sft_lora_config)
        model.set_adapter("sft")
    else:
        logger.warning(f"CPT adapter 없음 ({CPT_LORA_DIR}). base 모델에서 직접 SFT 진행.")
        model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
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

    # ── 데이터셋 ──────────────────────────────────────────────────────────────
    logger.info("데이터 로드 및 전처리...")
    tokenized_dataset = build_dataset(tokenizer)
    logger.info(f"최종 학습 데이터: {len(tokenized_dataset)}건")

    # ── 학습 steps ────────────────────────────────────────────────────────────
    total_steps = math.ceil(len(tokenized_dataset) / (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    logger.info(f"total_steps={total_steps}, warmup={warmup_steps}")

    # ── DataCollator ──────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
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
        save_total_limit=3,
        logging_steps=LOGGING_STEPS,
        logging_dir=f"{CKPT_DIR}/logs",
        report_to="none",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        group_by_length=True,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 체크포인트 재개
    resume_from = None
    ckpt_path = Path(CKPT_DIR)
    existing_ckpts = sorted(ckpt_path.glob("checkpoint-*"), key=os.path.getmtime)
    if existing_ckpts:
        resume_from = str(existing_ckpts[-1])
        logger.info(f"체크포인트 재개: {resume_from}")

    # ── 학습 ──────────────────────────────────────────────────────────────────
    logger.info("SFT 학습 시작...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── SFT adapter 저장 ──────────────────────────────────────────────────────
    logger.info(f"SFT adapter 저장 → {OUTPUT_DIR}")
    # sft adapter만 저장 (CPT는 별도 보관)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    elapsed = (time.time() - start_time) / 3600
    logger.info(f"SFT 학습 완료. 소요: {elapsed:.2f}시간")
    logger.info(f"결과물: {OUTPUT_DIR}")
    logger.info("다음 단계: build_e5_embeddings.py 로 임베딩 인덱스 구축")


if __name__ == "__main__":
    main()
