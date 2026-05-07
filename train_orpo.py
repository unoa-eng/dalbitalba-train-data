#!/usr/bin/env python3
"""train_orpo.py — Round-2 Phase-4 ORPO trainer (TRL ORPOTrainer).

Reads orpo_pairs.jsonl ({prompt, chosen, rejected}) and runs single-stage
preference optimization on top of the SFT-merged checkpoint.

Env vars (mirror train_sft.py naming):
  BASE_MODEL          base model or merged checkpoint dir (default sft-merged-fp16)
  ORPO_DATA           jsonl path
  ORPO_NUM_EPOCHS     default 1
  ORPO_BETA           default 0.1
  ORPO_OUTPUT_DIR     default /workspace/out/orpo-lora
  ORPO_LORA_R         default 64
  ORPO_LORA_ALPHA     default 64
  ORPO_MAX_SEQ_LEN    default 2048
  TRAIN_REPORT_TO     default none

This is a thin TRL ORPOTrainer wrapper. If TRL is not installed the script
exits 0 with a clear note (Phase-4 deferred to next cycle).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    base_model = os.environ.get("BASE_MODEL", "/workspace/out/sft-merged-fp16")
    orpo_data = os.environ.get("ORPO_DATA", "/workspace/data/orpo_pairs.jsonl")
    out_dir = os.environ.get("ORPO_OUTPUT_DIR", "/workspace/out/orpo-lora")
    epochs = int(os.environ.get("ORPO_NUM_EPOCHS", "1"))
    beta = float(os.environ.get("ORPO_BETA", "0.1"))
    lora_r = int(os.environ.get("ORPO_LORA_R", "64"))
    lora_alpha = int(os.environ.get("ORPO_LORA_ALPHA", "64"))
    max_seq_len = int(os.environ.get("ORPO_MAX_SEQ_LEN", "2048"))
    report_to = os.environ.get("TRAIN_REPORT_TO", "none")
    seed = int(os.environ.get("TRAIN_SEED", "42"))

    if not Path(orpo_data).exists():
        print(f"[orpo] missing data: {orpo_data}; deferring Phase-4")
        return 0

    try:
        from datasets import Dataset  # type: ignore
        from peft import LoraConfig  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from trl import ORPOConfig, ORPOTrainer  # type: ignore
    except ImportError as exc:
        print(f"[orpo] required deps missing ({exc}); deferring Phase-4 to next cycle")
        return 0

    print(f"[orpo] base_model={base_model} data={orpo_data} epochs={epochs} beta={beta}")

    rows = []
    with open(orpo_data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("prompt") and obj.get("chosen") and obj.get("rejected"):
                rows.append({
                    "prompt": str(obj["prompt"]),
                    "chosen": str(obj["chosen"]),
                    "rejected": str(obj["rejected"]),
                })
    if not rows:
        print("[orpo] no valid pairs found; exiting")
        return 0
    print(f"[orpo] loaded {len(rows)} preference pairs")

    ds = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    config = ORPOConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        beta=beta,
        max_length=max_seq_len,
        max_prompt_length=max_seq_len // 2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=8e-6,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to=report_to,
        seed=seed,
        bf16=True,
        gradient_checkpointing=True,
    )
    peft = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules="all-linear",
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")

    trainer = ORPOTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=peft,
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[orpo] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
