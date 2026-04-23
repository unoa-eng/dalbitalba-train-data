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
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = os.environ.get("BASE_MODEL", "upstage/SOLAR-10.7B-v1.0")
HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None
OUTPUT_FILE = REPO_ROOT / "ai_generated.jsonl"
SFT_PROMPT_FILE = REPO_ROOT / "sft_pairs_v2.jsonl"
NUM_TOPICS = int(os.environ.get("NUM_TOPICS", "100"))
SEED = int(os.environ.get("SAMPLE_SEED", "42"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))
POST_RATIO = float(os.environ.get("POST_RATIO", "0.5"))

_FORMAL_TAIL = re.compile(r"(습니다|입니다|드립니다|하겠습니다)\s*[\.\!\?~]*$")


def topic_candidates() -> list[Path]:
    return [
        REPO_ROOT / "cai_seeds.jsonl",
        REPO_ROOT / "cai_pairs.filtered.jsonl",
    ]


def load_topics(limit: int) -> list[str]:
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
                if len(topics) >= limit:
                    return topics
    if not topics:
        return [f"게시글 샘플 주제 {i}" for i in range(1, limit + 1)]
    while len(topics) < limit:
        topics.append(f"{topics[len(topics) % max(1, len(topics))]} 변형 {len(topics)}")
    return topics[:limit]


def load_sft_prompts() -> dict[str, list[dict]]:
    prompts: dict[str, list[dict]] = defaultdict(list)
    if not SFT_PROMPT_FILE.exists():
        return prompts

    with SFT_PROMPT_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            instruction = str(row.get("instruction") or "").strip()
            kind = str(row.get("pair_type") or "").strip()
            if not instruction or kind not in {"post", "comment"}:
                continue

            prompts[kind].append(
                {
                    "instruction": instruction,
                    "input": str(row.get("input") or "").strip(),
                    "kind": kind,
                    "source_id": row.get("source_id"),
                    "topic": instruction,
                }
            )
    return prompts


def build_generation_jobs() -> list[dict]:
    rng = random.Random(SEED)
    prompts = load_sft_prompts()
    post_prompts = prompts.get("post", [])
    comment_prompts = prompts.get("comment", [])

    rng.shuffle(post_prompts)
    rng.shuffle(comment_prompts)

    jobs: list[dict] = []
    if post_prompts or comment_prompts:
        desired_posts = 0
        desired_comments = 0

        if post_prompts:
            desired_posts = min(len(post_prompts), max(1, round(NUM_TOPICS * POST_RATIO)))
        if comment_prompts:
            desired_comments = min(len(comment_prompts), NUM_TOPICS - desired_posts)

        while desired_posts + desired_comments < NUM_TOPICS:
            if len(post_prompts) > desired_posts:
                desired_posts += 1
                continue
            if len(comment_prompts) > desired_comments:
                desired_comments += 1
                continue
            break

        jobs.extend(post_prompts[:desired_posts])
        jobs.extend(comment_prompts[:desired_comments])

    if len(jobs) < NUM_TOPICS:
        for topic in load_topics(NUM_TOPICS - len(jobs)):
            jobs.append(
                {
                    "instruction": f"이 주제로 짧은 커뮤니티 글을 써라: {topic}",
                    "input": "",
                    "kind": "post",
                    "source_id": None,
                    "topic": topic,
                }
            )

    rng.shuffle(jobs)
    return jobs[:NUM_TOPICS]


def apply_guardrails(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines and _FORMAL_TAIL.search(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip() or text.strip()


def format_generation_prompt(instruction: str, input_text: str = "") -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


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


def generate_response(generator, instruction: str, input_text: str = "") -> str:
    prompt = format_generation_prompt(instruction, input_text)
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
    jobs = build_generation_jobs()
    generator = load_generator()

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for index, job in enumerate(jobs, start=1):
            try:
                text = generate_response(generator, job["instruction"], job.get("input", ""))
            except Exception as exc:
                text = ""
                print(f"[warn] sample {index} failed: {exc}", file=sys.stderr)

            record = {
                "id": index,
                "topic": job.get("topic") or job["instruction"],
                "instruction": job["instruction"],
                "kind": job.get("kind", "post"),
                "text": text,
                "truth": "AI",
                "source_id": job.get("source_id"),
                "adapter_repo": HF_ADAPTER_REPO or None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index % 10 == 0:
                print(f"[generate] {index}/{len(jobs)} complete", flush=True)

    print(f"[done] wrote {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
