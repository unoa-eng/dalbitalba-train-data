#!/usr/bin/env python3
"""KoBEST regression evaluator (H4 gate).

Lightweight zero-shot accuracy on KoBEST `boolq` (binary). Compares a
base model to an adapter-applied model and rejects when the accuracy drop
exceeds the H4 threshold (Δ ≤ -0.05 / 5pp).

Usage:
    python3 eval/eval_kobest.py --base-model Qwen/Qwen3-0.6B-Base \\
        --adapter-path /path/to/lora --max-samples 50 \\
        --output runs/eval/kobest.json

When `--adapter-path` is omitted, the base model is scored against itself
(sanity dry-run; Δ = 0 by construction → PASS). Honors `H4_MAX_DROP` env
(default 0.05) for the reject threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TASK = "boolq"
DATASET = "skt/kobest_v1"
PASS_LABEL = {0: "아니오", 1: "예"}


def build_prompt(row: dict) -> str:
    return (
        f"문서:\n{row['paragraph']}\n"
        f"질문: {row['question']}\n"
        f"답 (예 또는 아니오):"
    )


def parse_pred(text: str) -> int | None:
    text = text.strip().split("\n")[0].strip()
    if "예" in text and "아니" not in text:
        return 1
    if "아니" in text or "아닙" in text:
        return 0
    return None


def score_model(model_id: str, adapter_path: str | None, rows: list, device: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "mps" else torch.bfloat16
    ).to(device)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    correct, parsed, started = 0, 0, time.time()
    for row in rows:
        prompt = build_prompt(row)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        decoded = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = parse_pred(decoded)
        if pred is not None:
            parsed += 1
            if pred == row["label"]:
                correct += 1
    n = len(rows)
    return {
        "model_id": model_id,
        "adapter_path": adapter_path,
        "n": n,
        "parsed": parsed,
        "accuracy": correct / n if n else 0.0,
        "elapsed_s": round(time.time() - started, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--output", type=Path, default=Path("runs/eval/kobest.json"))
    args = ap.parse_args()

    max_drop = float(os.environ.get("H4_MAX_DROP", "0.05"))
    from datasets import load_dataset
    rows = list(load_dataset(DATASET, TASK, split=f"validation[:{args.max_samples}]"))

    base = score_model(args.base_model, None, rows, args.device)
    if args.adapter_path:
        adapter = score_model(args.base_model, args.adapter_path, rows, args.device)
    else:
        adapter = {**base, "adapter_path": "<dry-run: base scored twice>"}

    drop = adapter["accuracy"] - base["accuracy"]
    verdict = "FAIL_H4" if drop < -max_drop else "PASS"

    report = {
        "task": f"KoBEST/{TASK}",
        "max_drop_threshold": max_drop,
        "base": base,
        "adapter": adapter,
        "accuracy_drop": round(drop, 4),
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "drop": drop, "report": str(args.output)}, ensure_ascii=False))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
