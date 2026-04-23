#!/usr/bin/env python3
"""
Generate blind-eval AI samples from the base model plus an uploaded adapter repo.

Expected adapter layout in Hugging Face:
  <HF_ADAPTER_REPO>/cpt-lora/
  <HF_ADAPTER_REPO>/sft-lora/  # optional
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = os.environ.get("BASE_MODEL", "upstage/SOLAR-10.7B-v1.0")
HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
OUTPUT_FILE = REPO_ROOT / "ai_generated.jsonl"
NUM_TOPICS = int(os.environ.get("NUM_TOPICS", "100"))
SEED = int(os.environ.get("SAMPLE_SEED", "42"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))

_FORMAL_TAIL = re.compile(r"(니다|습니다|드립니다|보겠습니다)\s*[\.\!\?~…]*\s*$")


def topic_candidates() -> list[Path]:
    return [
        REPO_ROOT / "cai_seeds.jsonl",
        REPO_ROOT / "cai_pairs.filtered.jsonl",
    ]


def load_topics() -> list[str]:
    topics: list[str] = []
    for candidate in topic_candidates():
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    obj = {"topic": line}
                topic = (
                    obj.get("topic")
                    or obj.get("prompt")
                    or obj.get("seed")
                    or obj.get("text")
                    or str(obj)
                )
                topic = str(topic).strip()
                if topic:
                    topics.append(topic)
                if len(topics) >= NUM_TOPICS:
                    return topics
    if not topics:
        return [f"게시판 샘플 주제 {i}" for i in range(1, NUM_TOPICS + 1)]
    while len(topics) < NUM_TOPICS:
        topics.append(f"{topics[len(topics) % max(1, len(topics))]} 변형 {len(topics)}")
    return topics[:NUM_TOPICS]


def apply_guardrails(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines and _FORMAL_TAIL.search(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip() or text.strip()


def load_generator():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    if HF_ADAPTER_REPO:
        print(f"[model] adapter repo: {HF_ADAPTER_REPO}", flush=True)
        try:
            model = PeftModel.from_pretrained(
                model,
                HF_ADAPTER_REPO,
                subfolder="cpt-lora",
                token=HF_TOKEN,
            )
            model = model.merge_and_unload()
        except Exception as exc:
            print(f"[warn] failed to load cpt-lora: {exc}", file=sys.stderr)

        try:
            model = PeftModel.from_pretrained(
                model,
                HF_ADAPTER_REPO,
                subfolder="sft-lora",
                token=HF_TOKEN,
            )
        except Exception as exc:
            print(f"[warn] failed to load sft-lora: {exc}", file=sys.stderr)
    else:
        print("[warn] HF_ADAPTER_REPO unset; evaluating the base model only.", file=sys.stderr)

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )


def generate_post(generator, topic: str) -> str:
    prompt = f"[스타일 SNS 게시글]\n주제: {topic}\n\n게시글:"
    output = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id,
        return_full_text=False,
    )[0]["generated_text"].strip()
    return apply_guardrails(output)


def main() -> None:
    random.seed(SEED)
    topics = load_topics()
    generator = load_generator()

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for index, topic in enumerate(topics, start=1):
            try:
                text = generate_post(generator, topic)
            except Exception as exc:
                text = ""
                print(f"[warn] topic {index} failed: {exc}", file=sys.stderr)
            record = {
                "id": index,
                "topic": topic,
                "text": text,
                "truth": "AI",
                "adapter_repo": HF_ADAPTER_REPO or None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 10 == 0:
                print(f"[generate] {index}/{len(topics)} complete", flush=True)

    print(f"[done] wrote {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
