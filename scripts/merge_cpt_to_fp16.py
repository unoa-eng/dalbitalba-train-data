#!/usr/bin/env python3
"""
scripts/merge_cpt_to_fp16.py — Merge the CPT LoRA adapter into the base
model and save a fp16 checkpoint that becomes the new base for SFT.

This eliminates the "adapter stacking" mismatch between training and
inference. Uses the PEFT 2025 consensus pattern:
  base(fp16) -> load adapter -> merge_and_unload -> save fp16.

Env:
  BASE_MODEL        : Qwen/Qwen3-8B-Base (default)
  CPT_LORA_DIR      : /workspace/out/cpt-lora
  CPT_MERGED_DIR    : /workspace/out/cpt-merged-fp16
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
    print(f"[오류] 의존성 누락: {e}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base")
    adapter_dir = os.environ.get("CPT_LORA_DIR", "/workspace/out/cpt-lora")
    merged_dir = os.environ.get("CPT_MERGED_DIR", "/workspace/out/cpt-merged-fp16")
    # Tokenizer path: explicit override > local tokenizer_v4 dir > base_model.
    # The merged dir MUST contain the extended tokenizer (not stock) so that
    # downstream SFT/eval reads the +210-token vocab consistently.
    tokenizer_path = os.environ.get("TOKENIZER_PATH") or (
        "tokenizer_v4" if os.path.isdir("tokenizer_v4") else base_model
    )

    if not Path(adapter_dir).exists():
        print(f"[error] adapter dir not found: {adapter_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"[merge] base       : {base_model}")
    print(f"[merge] lora       : {adapter_dir}")
    print(f"[merge] tokenizer  : {tokenizer_path}")
    print(f"[merge] out        : {merged_dir}")

    # Load base in fp16/bf16 (NOT 4-bit) so the merge is lossless.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("[merge] loaded base in", dtype)

    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model = model.merge_and_unload()
    print("[merge] merge_and_unload done")

    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)

    # Save EXTENDED tokenizer (not stock) so the merged checkpoint matches
    # what train_cpt.py / train_sft.py / phase6_generate.py read at TOKENIZER_PATH.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=True
    )
    tokenizer.save_pretrained(merged_dir)

    print(f"[merge] saved merged fp16 model → {merged_dir}")


if __name__ == "__main__":
    main()
