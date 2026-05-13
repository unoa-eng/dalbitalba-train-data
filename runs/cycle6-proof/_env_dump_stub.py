#!/usr/bin/env python3
"""Env-dump stub used in lieu of train_cpt.py to verify CPT_LR/CPT_WARMUP_RATIO
+ W&B env propagation from chain_train_round2.sh phase1/phase2 invocations."""
import os
import sys

LABEL = os.environ.get("STUB_LABEL", "stub")
KEYS = [
    "CPT_LR",
    "CPT_WARMUP_RATIO",
    "CPT_NUM_EPOCHS",
    "CPT_USE_DORA",
    "CPT_LORA_R",
    "CPT_LORA_ALPHA",
    "CPT_MAX_SEQ_LEN",
    "BASE_MODEL",
    "BASE_MODEL_REVISION",
    "INPUT_JSONL",
    "TRAIN_REPORT_TO",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_RUN_GROUP",
    "WANDB_NAME",
]
print("=== stub:%s ===" % LABEL)
for k in KEYS:
    v = os.environ.get(k, "<UNSET>")
    print("  ENV[%s] = %s" % (k, v))
sys.exit(0)
