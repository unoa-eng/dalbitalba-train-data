#!/usr/bin/env python3
"""Round-2 Step 1 — OpenVINO GPU/NPU classifier (faster than Ollama).

Reuses round2_source_classifier.py rule axes; runs LLM axes via OpenVINO GenAI
on Lunar Lake Arc 140V iGPU (~46 tok/s on Phi-3-mini-4k-int4-ov).

Note: This script is invoked from Windows Python (not WSL venv) because OV-GenAI
runs natively on Windows. Drop into the WSL clone but execute from Windows side:

  python scripts/round2_classifier_ov.py --source-dir source_db_cache \\
      --out-dir runs/round2-source-analysis --device GPU \\
      --model-dir "C:\\Users\\mapdr\\.cache\\huggingface\\Phi-3-mini-4k-instruct-int4-ov" \\
      --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import openvino_genai as ov_genai  # type: ignore

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
import round2_source_classifier as rsc  # type: ignore

SYSTEM_PROMPT = """너는 한국어 밤문화 커뮤니티 텍스트 라벨러다. 입력을 7축으로 분류하고 JSON 한 객체만 출력한다. 7개 키 모두 반드시 포함."""

USER_PROMPT_TEMPLATE = """입력: [KIND={kind}] {text}

다음 7축으로 분류해 JSON 한 객체만 출력. 7키 모두 필수.

post_type ∈ ["질문","고민상담","정보공유","후기","잡담","자랑","불만","경고","구인광고","기타"]
emotion ∈ ["기쁨","슬픔","분노","두려움","짜증","공감","냉소","비판","정보전달","중립"]
topic_cluster ∈ ["손님관계","동료관계","수익TC","외모관리","업장환경","업계전반","사생활","건강심리","기타"]
social_role ∈ ["언니선배","동생후배","신입경력자","손님언급","실장사장언급","자기서술","관계없음"]
register ∈ ["반말","존댓말","혼용","느낌체"]
domain_argot ∈ ["풍부","보통","없음"]
promo_likeness ∈ ["전형광고","암시광고","인간화법"]

social_role 결정 가이드:
- "손님/손놈/진상" 언급 → 손님언급
- "마담/실장/사장/원장" 언급 → 실장사장언급
- "언니/선배/n년차" 언급 → 언니선배
- "신입/막내" 언급 → 동생후배 또는 신입경력자
- 짧은 댓글로 1인칭만 → 자기서술
- 위 어느 것도 아님 → 관계없음

domain_argot: TC/밀빵/쩜오/텐카/케어/갯수/마담/퍼블/초이스 등 cb2 은어 2개+ → 풍부, 1개 → 보통, 0개 → 없음
promo_likeness: 010-/카톡/문의/모집 → 전형광고, 가게이름만 → 암시광고, 일반 화법 → 인간화법

예시 입력: [KIND=comment] ㅈㄴ짜증나 손놈이 또 진상부려서 화남
예시 출력: {"post_type":"불만","emotion":"분노","topic_cluster":"손님관계","social_role":"손님언급","register":"반말","domain_argot":"보통","promo_likeness":"인간화법"}

위 형식 그대로 7키 JSON 한 줄 출력. 설명 금지."""


def build_chat_prompt(kind: str, text: str) -> str:
    user = USER_PROMPT_TEMPLATE.replace("{kind}", kind).replace("{text}", text[:900])
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def classify_one_ov(item: dict[str, Any], pipe, config) -> dict[str, Any]:
    text = item["text"]
    kind = item["kind"]
    rule = rsc.rule_axes(text, kind, item.get("_post_meta") or {})
    out = {
        "id": item["id"],
        "kind": kind,
        "post_id": item.get("post_id"),
        "depth": rsc.reply_depth(text or "") if kind == "comment" else 0,
        "text_len": len(text or ""),
        "raw_text_head": (text or "")[:200],
    }
    out.update(rule)

    prompt = build_chat_prompt(kind, text)
    raw = pipe.generate(prompt, config)
    raw_str = str(raw).strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in raw_str:
            raw_str = raw_str.split(marker)[0]
    llm_obj = rsc.parse_llm_response(raw_str)
    out["llm_valid_count"] = len(llm_obj)
    for axis in rsc.LLM_AXES:
        out[axis] = llm_obj.get(axis)
    if len(llm_obj) < len(rsc.LLM_AXES):
        out["_llm_raw"] = raw_str[:400]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="Directory of cb2_*.json batches (mutually exclusive with --items-jsonl)")
    ap.add_argument("--items-jsonl", type=Path, default=None,
                    help="Pre-built jsonl of items {id,kind,text,...} to classify")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path,
                    help=r"e.g. C:\Users\mapdr\.cache\huggingface\Phi-3-mini-4k-instruct-int4-ov")
    ap.add_argument("--device", default="GPU", choices=["NPU", "GPU", "CPU"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    args = ap.parse_args()

    print(f"Loading {args.model_dir} on {args.device}...", flush=True)
    pipe = ov_genai.LLMPipeline(str(args.model_dir), args.device)
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = False
    print(f"Loaded. Iterating items...", flush=True)

    if args.items_jsonl:
        items = [json.loads(l) for l in args.items_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif args.source_dir:
        items = list(rsc.iter_source(args.source_dir))
    else:
        print("ERROR: provide --source-dir or --items-jsonl", file=sys.stderr)
        return 2
    if args.limit > 0:
        items = items[: args.limit]
    total = len(items)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        try:
            row = classify_one_ov(it, pipe, config)
        except Exception as exc:
            row = {"id": it["id"], "kind": it["kind"], "_error": str(exc)[:300]}
        rows.append(row)
        if i % 25 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.01)
            eta_min = (total - i) / max(rate, 0.01) / 60
            print(f"  {i}/{total} rate={rate:.2f} it/s eta={eta_min:.1f}min", flush=True)
        if i % args.checkpoint_every == 0:
            rsc.write_partial(rows, args.out_dir)

    rsc.write_partial(rows, args.out_dir)

    valid = sum(1 for r in rows if r.get("llm_valid_count", 0) == len(rsc.LLM_AXES))
    cover_rate = valid / max(total, 1)
    elapsed = time.time() - t0
    stats = {
        "total": total,
        "rule_valid": sum(1 for r in rows if r.get("punct_q") is not None),
        "rule_cover_rate": sum(1 for r in rows if r.get("punct_q") is not None) / max(total, 1),
        "llm_valid": valid,
        "llm_cover_rate": cover_rate,
        "elapsed_sec": elapsed,
        "throughput_it_per_sec": total / max(elapsed, 0.01),
        "model_dir": str(args.model_dir),
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
    }
    (args.out_dir / "run-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"DONE total={total} llm_valid={valid} cover={cover_rate:.4f} elapsed={elapsed:.1f}s", flush=True)
    return 0 if cover_rate >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
