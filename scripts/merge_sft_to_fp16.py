#!/usr/bin/env python3
"""scripts/merge_sft_to_fp16.py — Merge the SFT LoRA adapter into the
post-CPT-merged base model and save a fp16 checkpoint to be used as the
ORPO Phase-4 base.

Mirrors merge_cpt_to_fp16.py's PEFT 2025 pattern:
  base(fp16) -> load adapter -> merge_and_unload -> save fp16.

Env:
  BASE_MODEL        : the post-CPT merged fp16 dir (default cpt-merged-fp16)
  SFT_LORA_DIR      : /workspace/out/sft-lora
  SFT_MERGED_DIR    : /workspace/out/sft-merged-fp16
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    print(f"[error] missing deps: {e}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base_model = os.environ.get("BASE_MODEL", "/workspace/out/cpt-merged-fp16")
    base_model_revision = os.environ.get("BASE_MODEL_REVISION", "main")
    adapter_dir = os.environ.get("SFT_LORA_DIR", "/workspace/out/sft-lora")
    merged_dir = os.environ.get("SFT_MERGED_DIR", "/workspace/out/sft-merged-fp16")
    # Tokenizer path: explicit override > local tokenizer_v4 dir > base_model.
    # The merged dir MUST contain the extended tokenizer (not stock) so that
    # downstream ORPO/eval reads the +210-token vocab consistently.
    tokenizer_path = os.environ.get("TOKENIZER_PATH") or (
        "tokenizer_v4" if os.path.isdir("tokenizer_v4") else base_model
    )

    if not Path(adapter_dir).exists():
        print(f"[error] adapter dir not found: {adapter_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"[merge-sft] base       : {base_model}")
    print(f"[merge-sft] revision   : {base_model_revision}")
    print(f"[merge-sft] lora       : {adapter_dir}")
    print(f"[merge-sft] tokenizer  : {tokenizer_path}")
    print(f"[merge-sft] out        : {merged_dir}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=base_model_revision,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model = model.merge_and_unload()

    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)

    # Save EXTENDED tokenizer (not stock) so the merged checkpoint matches
    # what train_sft.py / phase6_generate.py read at TOKENIZER_PATH.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True
    )
    tokenizer.save_pretrained(merged_dir)
    print(f"[merge-sft] saved merged fp16 model -> {merged_dir}")


if __name__ == "__main__":
    main()
