#!/usr/bin/env python3
"""Round-2 Step-4 mac-mini local smoke: Qwen3-0.6B baseline vs few-shot simulation.

Real SFT requires RunPod (out of scope this session). Instead we generate:
- baseline.jsonl: Qwen3-0.6B raw output on neutral prompts (cb2 style)
- post-sft.jsonl: same model, but with 8-shot in-context examples from cb2 raw
  (simulating what SFT *should* produce; tests phase6 metric direction).

Usage on mac-mini (Apple Silicon MPS):
  source ~/.venv-train/bin/activate
  cd ~/projects/dalbitalba-train-data
  python scripts/round2_macmini_smoke.py \\
      --out-dir runs/round2-local-smoke --n 100
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

random.seed(42)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device}...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    model.eval()
    print("loaded", flush=True)

    seeds_post = [
        "오늘 손놈 진상 ㅈㄴ 만났는데",
        "강남에서 일하는 언니들 솔직히",
        "TC 적게 받고 끝났는데 케어 어려워서",
        "쩜오 신입인데 첫 출근날",
        "텐카 손님 응대 어떻게 해야",
        "퍼블 분위기 좋은 가게 추천",
        "오늘 일하다가 너무 짜증나서",
        "수익 정리하다 보니까 이번달",
        "마담언니가 자꾸 ㅈㄴ 갈구는데",
        "초이스 받기 위해서 노력하는데",
    ]
    seeds_comment = [
        "ㄹㅇㅋㅋ",
        "ㅈㄴ공감 ㅠ",
        "ㅇㅈ 저도 그래요",
        "ㄱㄱ 좋아요",
        "ㅅㅂ 진짜 짜증나",
        "와 그건 너무한듯 ㄷㄷ",
        "쩜오에서 그렇게 해도 ㄱㅊ아요?",
        "텐카 아니라 셔츠룸 가세요",
        "케어 잘해주는 언니 따로 있어요",
        "TC 그 가게 쎈편이에요",
    ]

    few_shot = """다음은 한국 밤문화 커뮤니티 글의 예시들이다. 같은 톤·문체·길이로 새 글을 한 편 써라. 형식체·AI 자기 소개·존댓말 길게 늘이기 금지.

예1: 오늘 ㅈㄴ 짜증났어 손놈이 또 진상부려서 화남.. 마담언니가 와서 진정시켜줬는데 그래도 끝까지 케어해야됨 ㅠㅠ

예2: 강남 텐 언니들 솔직히 TC 잘 못 받음. 셔츠룸이 더 잘 받는 듯. 진짜 갯수 차이 너무 남

예3: ㅋㅋㅋ ㄹㅇ공감.. 우리 가게는 마담이 원래 그래서 진작 포기함

새 글:
"""

    def gen(prompt: str, max_new: int) -> str:
        inputs = tok(prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    n_post = max(1, args.n // 5)  # 20% posts, 80% comments
    n_comment = args.n - n_post

    baseline_rows = []
    print("Generating baseline...", flush=True)
    t0 = time.time()
    for i in range(n_post):
        seed = random.choice(seeds_post)
        out = gen(seed, args.max_new_tokens)
        text = (seed + " " + out).strip()
        baseline_rows.append({"id": f"baseline-p{i}", "kind": "post", "text": text})
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  baseline post {i+1}/{n_post} ({elapsed:.0f}s)", flush=True)
    for i in range(n_comment):
        seed = random.choice(seeds_comment)
        out = gen(seed, max(40, args.max_new_tokens // 2))
        text = (seed + " " + out).strip()
        baseline_rows.append({"id": f"baseline-c{i}", "kind": "comment", "text": text})
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  baseline comment {i+1}/{n_comment} ({elapsed:.0f}s)", flush=True)
    bl_elapsed = time.time() - t0

    sft_rows = []
    print("Generating few-shot (SFT-sim)...", flush=True)
    t1 = time.time()
    for i in range(n_post):
        seed = random.choice(seeds_post)
        prompt = few_shot + seed
        out = gen(prompt, args.max_new_tokens)
        text = (seed + " " + out).strip()
        sft_rows.append({"id": f"sft-p{i}", "kind": "post", "text": text})
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t1
            print(f"  sft post {i+1}/{n_post} ({elapsed:.0f}s)", flush=True)
    for i in range(n_comment):
        seed = random.choice(seeds_comment)
        prompt = few_shot + seed
        out = gen(prompt, max(40, args.max_new_tokens // 2))
        text = (seed + " " + out).strip()
        sft_rows.append({"id": f"sft-c{i}", "kind": "comment", "text": text})
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t1
            print(f"  sft comment {i+1}/{n_comment} ({elapsed:.0f}s)", flush=True)
    sft_elapsed = time.time() - t1

    with (args.out_dir / "baseline.jsonl").open("w", encoding="utf-8") as fh:
        for r in baseline_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (args.out_dir / "post-sft.jsonl").open("w", encoding="utf-8") as fh:
        for r in sft_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "model": args.model,
        "device": args.device,
        "n_baseline": len(baseline_rows),
        "n_sft": len(sft_rows),
        "baseline_elapsed_sec": bl_elapsed,
        "sft_elapsed_sec": sft_elapsed,
        "max_new_tokens": args.max_new_tokens,
        "note": "post-sft is few-shot simulation. Real SFT requires RunPod GPU pod (out of session scope).",
    }
    (args.out_dir / "macmini-smoke-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"DONE baseline={len(baseline_rows)} sft={len(sft_rows)} bl={bl_elapsed:.0f}s sft={sft_elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
