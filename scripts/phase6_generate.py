#!/usr/bin/env python3

import json
import os
import sys

import peft
import torch
import transformers


BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B-Base")
SFT_ADAPTER_REPO = os.environ.get("SFT_ADAPTER_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
INPUT_PATH = "/workspace/data/val_set.v2.jsonl"
OUTPUT_PATH = "/workspace/ai_generated.jsonl"
MAX_ROWS = 500


def main() -> int:
    if not SFT_ADAPTER_REPO:
        print("[error] SFT_ADAPTER_REPO is required", file=sys.stderr)
        return 1

    model = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        token=HF_TOKEN,
    )
    model = peft.PeftModel.from_pretrained(
        model,
        SFT_ADAPTER_REPO,
        token=HF_TOKEN,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
    )

    with open(INPUT_PATH, "r", encoding="utf-8") as src, open(
        OUTPUT_PATH, "w", encoding="utf-8"
    ) as dst:
        for i, line in enumerate(src, start=1):
            if i > MAX_ROWS:
                break

            row = json.loads(line)
            text = str(row["text"])
            prompt = text[:40]

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=1.1,
                top_p=0.9,
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            continuation = decoded[len(prompt) :] if decoded.startswith(prompt) else decoded

            dst.write(
                json.dumps(
                    {"text": continuation, "seed": prompt},
                    ensure_ascii=False,
                )
                + "\n"
            )

            if i % 50 == 0:
                print(f"[{i}/500] generated", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
