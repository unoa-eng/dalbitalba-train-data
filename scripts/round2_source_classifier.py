#!/usr/bin/env python3
"""Round-2 Step 1 — 12-axis source-DB classifier.

Hybrid pipeline:
- 4 rule-based axes (punct_marker, choseong_marker, thread_role, comm_norm)
- 7 LLM axes via Ollama localhost:11434 (post_type, emotion, topic_cluster,
  social_role, register, domain_argot, promo_likeness)
- 1 deferred axis (persona_signature) — populated in Step 2 from Obsidian.

Inputs:
  --source-dir <dir of cb2_*.json>      # raw crawled batches (109 files)
  --out-dir <runs/round2-source-analysis>
  --limit N                              # smoke test cap (None = full)
  --ollama-url <http://localhost:11434>
  --model qwen2.5:3b
  --concurrency 16

Outputs:
  <out-dir>/distributions.parquet       # one row per item (post, comment)
  <out-dir>/schema.json                 # already produced separately
  <out-dir>/run-stats.json              # cover_rate, per-axis valid counts, throughput
  <out-dir>/items-failed.jsonl          # items that failed all retries
  <out-dir>/summary.md                  # markdown roll-up vs RAW_CRAWL_TRAINING_DESIGN.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None
try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None

REPLY_PREFIX = re.compile(r"^\s*\[(\d+(?:-\d+)*)\]\s*")
CHOSEONG_PATTERNS = {
    "jn": re.compile(r"ㅈㄴ"),
    "oj": re.compile(r"ㅇㅈ"),
    "rieng": re.compile(r"ㄹㅇ"),
    "sb": re.compile(r"ㅅㅂ"),
    "gc": re.compile(r"ㄱㅊ"),
    "gg": re.compile(r"ㄱㄱ"),
    "nn": re.compile(r"ㄴㄴ"),
    "bs": re.compile(r"ㅂㅅ"),
    "dd": re.compile(r"ㄷㄷ"),
    "kpp": re.compile(r"ㄲㅃ"),
    "ois": re.compile(r"ㅊㅇㅅ"),
}
PUNCT_PATTERNS = {
    "q": re.compile(r"\?"),
    "ex": re.compile(r"!"),
    "ellipsis": re.compile(r"\.{2,}|…"),
    "kkk": re.compile(r"ㅋ{2,}|크{2,}"),
    "huhu": re.compile(r"ㅎ{2,}"),
    "tt": re.compile(r"ㅠ{1,}|ㅜ{1,}"),
}

LLM_PROMPT = """\
다음 한국어 게시물 또는 댓글을 7개 축으로 분류한다.
- post_type: [질문, 고민상담, 정보공유, 후기, 잡담, 자랑, 불만, 경고, 구인광고, 기타] (댓글이면 "기타")
- emotion: [기쁨, 슬픔, 분노, 두려움, 짜증, 공감, 냉소, 비판, 정보전달, 중립]
- topic_cluster: [손님관계, 동료관계, 수익TC, 외모관리, 업장환경, 업계전반, 사생활, 건강심리, 기타]
- social_role: [언니선배, 동생후배, 신입경력자, 손님언급, 실장사장언급, 자기서술, 관계없음]
- register: [반말, 존댓말, 혼용, 느낌체]
- domain_argot: [풍부, 보통, 없음]   # TC, 초이스, 밀빵, 쩜오, 텐, 케어, 갯수 등 업계 은어
- promo_likeness: [전형광고, 암시광고, 인간화법]

반드시 JSON 한 줄로만 출력. 설명 금지.
형식: {{"post_type":"X","emotion":"X","topic_cluster":"X","social_role":"X","register":"X","domain_argot":"X","promo_likeness":"X"}}

[KIND={kind}] {text}
"""

VALID_LABELS = {
    "post_type": {"질문", "고민상담", "정보공유", "후기", "잡담", "자랑", "불만", "경고", "구인광고", "기타"},
    "emotion": {"기쁨", "슬픔", "분노", "두려움", "짜증", "공감", "냉소", "비판", "정보전달", "중립"},
    "topic_cluster": {"손님관계", "동료관계", "수익TC", "외모관리", "업장환경", "업계전반", "사생활", "건강심리", "기타"},
    "social_role": {"언니선배", "동생후배", "신입경력자", "손님언급", "실장사장언급", "자기서술", "관계없음"},
    "register": {"반말", "존댓말", "혼용", "느낌체"},
    "domain_argot": {"풍부", "보통", "없음"},
    "promo_likeness": {"전형광고", "암시광고", "인간화법"},
}
LLM_AXES = list(VALID_LABELS.keys())


def reply_depth(text: str) -> int:
    m = REPLY_PREFIX.match(text or "")
    if not m:
        return 0
    return m.group(1).count("-")


def thread_role_label(depth: int) -> str:
    if depth == 0:
        return "루트댓글"
    if depth == 1:
        return "직답글"
    if depth == 2:
        return "체인답글_2"
    return "체인답글_3+"


def views_bucket(views: int) -> str:
    if views < 100:
        return "0-100"
    if views < 1000:
        return "100-1k"
    if views < 10000:
        return "1k-10k"
    return "10k+"


def comment_engagement_bucket(comment_count: int, views: int) -> str:
    if views == 0:
        return "저"
    ratio = comment_count / max(views, 1)
    if ratio < 0.005:
        return "저"
    if ratio < 0.02:
        return "보통"
    return "고"


def rule_axes(text: str, kind: str, item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["punct_q"] = len(PUNCT_PATTERNS["q"].findall(text or ""))
    out["punct_ex"] = len(PUNCT_PATTERNS["ex"].findall(text or ""))
    out["punct_ellipsis"] = len(PUNCT_PATTERNS["ellipsis"].findall(text or ""))
    out["punct_kkk"] = len(PUNCT_PATTERNS["kkk"].findall(text or ""))
    out["punct_huhu"] = len(PUNCT_PATTERNS["huhu"].findall(text or ""))
    out["punct_tt"] = len(PUNCT_PATTERNS["tt"].findall(text or ""))
    for k, p in CHOSEONG_PATTERNS.items():
        out[f"cs_{k}"] = len(p.findall(text or ""))
    if kind == "comment":
        out["thread_role"] = thread_role_label(reply_depth(text or ""))
    else:
        out["thread_role"] = None
    if kind == "post":
        views_v = int(item.get("views") or 0)
        try:
            ccount = int(item.get("commentCount") or item.get("comment_count") or 0)
        except (TypeError, ValueError):
            ccount = 0
        out["views_bucket"] = views_bucket(views_v)
        out["comment_engagement"] = comment_engagement_bucket(ccount, views_v)
    else:
        out["views_bucket"] = None
        out["comment_engagement"] = None
    return out


def _fuzzy_match_label(value: str, valid: set[str]) -> str | None:
    """Recover near-miss labels: 자서술 -> 자기서술, 인광고 -> 인간화법, etc."""
    if not isinstance(value, str):
        return None
    if value in valid:
        return value
    # exact substring match
    for v in valid:
        if value == v:
            return v
    # value is a substring of valid (truncation)
    for v in valid:
        if value and value in v and len(value) >= 2:
            return v
    # valid is a substring of value (extra chars)
    for v in valid:
        if v in value and len(v) >= 2:
            return v
    # remove common chars and compare
    norm = value.replace("자기", "자").replace("인간", "인").strip()
    for v in valid:
        nv = v.replace("자기", "자").replace("인간", "인").strip()
        if nv == norm:
            return v
    return None


def parse_llm_response(raw: str) -> dict[str, str]:
    if raw is None:
        return {}
    raw = raw.strip()
    start = raw.find("{")
    end = raw.find("}", start)  # first closing brace, not last (avoid trailing example)
    if start < 0 or end < 0 or end <= start:
        return {}
    candidate = raw[start:end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # try cleaning whitespace inside
        try:
            obj = json.loads(candidate.replace("\n", "").replace("  ", ""))
        except json.JSONDecodeError:
            return {}
    if not isinstance(obj, dict):
        return {}
    cleaned: dict[str, str] = {}
    for axis in LLM_AXES:
        v = obj.get(axis)
        matched = _fuzzy_match_label(v, VALID_LABELS[axis]) if v is not None else None
        if matched:
            cleaned[axis] = matched
    return cleaned


def call_ollama(url: str, model: str, prompt: str, timeout: int = 180) -> str | None:
    if requests is None:
        raise RuntimeError("requests not installed in venv")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 200, "top_p": 0.9},
    }
    try:
        r = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response")
    except Exception as exc:  # noqa: BLE001
        return None


def classify_one(item: dict[str, Any], url: str, model: str, retries: int = 2,
                 skip_llm: bool = False) -> dict[str, Any]:
    text = item["text"]
    kind = item["kind"]
    rule = rule_axes(text, kind, item.get("_post_meta") or {})
    out = {
        "id": item["id"],
        "kind": kind,
        "post_id": item.get("post_id"),
        "depth": reply_depth(text or "") if kind == "comment" else 0,
        "text_len": len(text or ""),
        "raw_text_head": (text or "")[:200],
    }
    out.update(rule)
    if skip_llm:
        out["llm_valid_count"] = 0
        for axis in LLM_AXES:
            out[axis] = None
        return out
    prompt = LLM_PROMPT.format(kind=kind, text=text[:1200])
    llm_obj: dict[str, str] = {}
    last_raw = ""
    for attempt in range(retries + 1):
        raw = call_ollama(url, model, prompt)
        last_raw = raw or ""
        llm_obj = parse_llm_response(raw or "")
        if len(llm_obj) == len(LLM_AXES):
            break
    out["llm_valid_count"] = len(llm_obj)
    for axis in LLM_AXES:
        out[axis] = llm_obj.get(axis)
    if len(llm_obj) < len(LLM_AXES):
        out["_llm_raw"] = last_raw[:400]
    return out


def iter_source(source_dir: Path) -> Iterable[dict[str, Any]]:
    files = sorted(source_dir.glob("cb2_*.json"))
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {fp.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            continue
        for post in data:
            if not isinstance(post, dict):
                continue
            post_id = str(post.get("id") or "")
            views = int(post.get("views") or 0)
            try:
                ccount = int(post.get("commentCount") or 0)
            except (TypeError, ValueError):
                ccount = 0
            title = (post.get("title") or "").strip()
            content = (post.get("content") or "").strip()
            joined = "\n".join([t for t in [title, content] if t])
            yield {
                "id": f"p{post_id}",
                "kind": "post",
                "post_id": post_id,
                "text": joined,
                "_post_meta": {"views": views, "commentCount": ccount},
            }
            for c in post.get("comments") or []:
                if not isinstance(c, dict):
                    continue
                ctext = (c.get("content") or "").strip()
                if not ctext:
                    continue
                m = REPLY_PREFIX.match(ctext)
                anchor = m.group(1) if m else "0"
                yield {
                    "id": f"c{post_id}-{anchor}",
                    "kind": "comment",
                    "post_id": post_id,
                    "text": ctext,
                    "_post_meta": {"views": views, "commentCount": ccount},
                }


def write_partial(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_parquet(out_dir / "distributions.parquet", index=False)
    # also write jsonl as fallback
    with (out_dir / "distributions.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--skip-llm", action="store_true",
                    help="Compute only rule-based axes; skip Ollama (fast full-corpus pass)")
    args = ap.parse_args()

    if not args.source_dir.is_dir():
        print(f"source dir not found: {args.source_dir}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = list(iter_source(args.source_dir))
    if args.limit > 0:
        items = items[: args.limit]
    total = len(items)
    print(f"items_total={total} concurrency={args.concurrency} model={args.model}")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(classify_one, it, args.ollama_url, args.model, 2, args.skip_llm): it for it in items}
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            if row.get("llm_valid_count", 0) < len(LLM_AXES):
                failed.append({"id": row["id"], "raw_head": row.get("raw_text_head"), "llm_raw": row.get("_llm_raw")})
            rows.append(row)
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / max(rate, 0.01)
                print(f"  {done}/{total}  rate={rate:.2f} it/s  eta={eta/60:.1f}min", flush=True)
            if done % args.checkpoint_every == 0:
                write_partial(rows, args.out_dir)

    write_partial(rows, args.out_dir)

    valid_count = sum(1 for r in rows if r.get("llm_valid_count", 0) == len(LLM_AXES))
    cover_rate = valid_count / max(total, 1)
    rule_cover = sum(1 for r in rows if r.get("punct_q") is not None) / max(total, 1)
    stats = {
        "total": total,
        "rule_valid": sum(1 for r in rows if r.get("punct_q") is not None),
        "rule_cover_rate": rule_cover,
        "llm_valid": valid_count,
        "llm_cover_rate": cover_rate if not args.skip_llm else None,
        "elapsed_sec": time.time() - t0,
        "throughput_it_per_sec": done / max(time.time() - t0, 0.01),
        "model": args.model,
        "concurrency": args.concurrency,
        "skip_llm": bool(args.skip_llm),
        "failed_count": len(failed),
    }
    (args.out_dir / "run-stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed:
        with (args.out_dir / "items-failed.jsonl").open("w", encoding="utf-8") as fh:
            for f in failed:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    if args.skip_llm:
        print(f"DONE total={total} rule_valid={stats['rule_valid']} elapsed={stats['elapsed_sec']:.1f}s (rule-only)")
        return 0 if rule_cover >= 0.99 else 1
    print(f"DONE total={total} llm_valid={valid_count} cover_rate={cover_rate:.4f} elapsed={stats['elapsed_sec']:.1f}s")
    return 0 if cover_rate >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
