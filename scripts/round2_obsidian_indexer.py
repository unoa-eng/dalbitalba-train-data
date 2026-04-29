#!/usr/bin/env python3
"""Round-2 Step 2 — Full Obsidian vault indexer + content extractor.

Inputs:
  --vault <dir>            # local rsync'd copy of ai-detection-research vault
  --out-dir <runs/round2-obsidian-synthesis>
  --vault-root-name <ai-detection-research>  # for relative path display

Outputs:
  INDEX.md                 # one row per md (path | category | title | wc | summary)
  index.parquet            # tabular form
  failure-modes.md         # extracted failure modes / limitations
  eval-protocols.md        # extracted evaluation protocols
  category-distribution.json
  by-category/*.md         # rolled-up snapshots per category for downstream review
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

CATEGORY_RULES = [
    ("persona", re.compile(r"(?i)구현/페르소나-설계|persona|페르소나")),
    ("pipeline", re.compile(r"(?i)구현/파이프라인|pipeline")),
    ("ontology", re.compile(r"(?i)구현/온톨로지|ontology")),
    ("implementation", re.compile(r"(?i)/구현(/|$)")),
    ("paper", re.compile(r"(?i)/논문(/|$)")),
    ("research_prior", re.compile(r"(?i)연구/선행연구|prior")),
    ("research_experiment", re.compile(r"(?i)연구/실험|experiment")),
    ("research_crawl", re.compile(r"(?i)연구/크롤분석|crawl")),
    ("research_general", re.compile(r"(?i)/연구(/|$)")),
    ("material", re.compile(r"(?i)/자료(/|$)")),
    ("meeting_log", re.compile(r"(?i)회의-및-로그|meeting|log")),
    ("inbox", re.compile(r"(?i)0-INBOX|inbox")),
    ("system", re.compile(r"(?i)_system|_templates|HOME|VAULT-MAP|START-HERE")),
]

FAILURE_KEYWORDS = re.compile(r"(?:실패|에러|오류|버그|한계|limitation|failure|fail|미흡|약점|risk|위험)", re.IGNORECASE)
EVAL_KEYWORDS = re.compile(r"(?:평가|메트릭|metric|protocol|rubric|벤치|benchmark|점수|score|judge|판정|블라인드|blind)", re.IGNORECASE)
PERSONA_HEADER = re.compile(r"^(?:#{1,4}\s+)?(?:#?\s*\d+\.?\s*)?([가-힣A-Za-z0-9\-\(\)\[\] ]{2,30})\s*(?:[\(\[]([^\)\]]+)[\)\]])?\s*$")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def categorize(rel_path: str) -> str:
    for label, pattern in CATEGORY_RULES:
        if pattern.search(rel_path):
            return label
    return "uncategorized"


def parse_frontmatter(text: str) -> dict[str, str]:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip().strip("'\"")
    return fm


def extract_title(rel_path: str, body: str) -> str:
    for line in body.splitlines():
        m = re.match(r"^#\s+(.+)$", line)
        if m:
            return m.group(1).strip()
    return Path(rel_path).stem


def file_summary(body: str, n_chars: int = 300) -> str:
    text = re.sub(r"\s+", " ", body.strip())
    return text[:n_chars]


def extract_failure_modes(body: str, rel_path: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if FAILURE_KEYWORDS.search(line):
            ctx_start = max(0, i - 1)
            ctx_end = min(len(lines), i + 3)
            ctx = "\n".join(lines[ctx_start:ctx_end]).strip()
            if len(ctx) > 30:
                out.append({"file": rel_path, "line": str(i + 1), "context": ctx[:500]})
    return out


def extract_eval_protocols(body: str, rel_path: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if EVAL_KEYWORDS.search(line):
            ctx_start = max(0, i - 1)
            ctx_end = min(len(lines), i + 4)
            ctx = "\n".join(lines[ctx_start:ctx_end]).strip()
            if len(ctx) > 40:
                out.append({"file": rel_path, "line": str(i + 1), "context": ctx[:500]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--vault-root-name", default="ai-detection-research")
    ap.add_argument("--max-failure-rows", type=int, default=400)
    ap.add_argument("--max-eval-rows", type=int, default=400)
    args = ap.parse_args()

    if not args.vault.is_dir():
        print(f"vault dir not found: {args.vault}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "by-category").mkdir(parents=True, exist_ok=True)

    md_files = sorted(args.vault.rglob("*.md"))
    print(f"md_files={len(md_files)}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    evals: list[dict[str, str]] = []
    cat_counts: dict[str, int] = {}
    cat_buckets: dict[str, list[str]] = {}

    for fp in md_files:
        rel = fp.relative_to(args.vault).as_posix()
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {rel}: {exc}", file=sys.stderr)
            continue
        body = text
        fm = parse_frontmatter(body)
        if fm:
            body = body[FRONTMATTER_RE.match(body).end():]  # type: ignore[union-attr]
        cat = categorize(rel)
        title = (fm.get("title") if fm else None) or extract_title(rel, body)
        summary = file_summary(body)
        wc = len(body.split())
        rows.append({
            "path": rel,
            "category": cat,
            "title": title,
            "word_count": wc,
            "char_count": len(body),
            "summary": summary,
            "frontmatter_keys": ",".join(sorted(fm.keys())),
            "fm_tags": fm.get("tags", "") if fm else "",
        })
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_buckets.setdefault(cat, []).append(f"- [{title}]({rel})")
        for f in extract_failure_modes(body, rel):
            failures.append(f)
            if len(failures) >= args.max_failure_rows * 4:
                break
        for e in extract_eval_protocols(body, rel):
            evals.append(e)
            if len(evals) >= args.max_eval_rows * 4:
                break

    # write parquet + jsonl + INDEX.md
    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_parquet(args.out_dir / "index.parquet", index=False)
    with (args.out_dir / "index.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (args.out_dir / "INDEX.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# Obsidian vault index — {args.vault_root_name}\n\n")
        fh.write(f"Total markdown files: **{len(rows)}**\n\n")
        fh.write("| path | category | title | wc |\n|---|---|---|---|\n")
        for r in rows:
            t = r["title"].replace("|", "\\|")
            p = r["path"].replace("|", "\\|")
            fh.write(f"| {p} | {r['category']} | {t} | {r['word_count']} |\n")

    (args.out_dir / "category-distribution.json").write_text(
        json.dumps({"by_category": dict(sorted(cat_counts.items(), key=lambda x: -x[1])), "total": len(rows)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Roll-ups per category
    for cat, items in cat_buckets.items():
        path = args.out_dir / "by-category" / f"{cat}.md"
        path.write_text(f"# Category: {cat}\n\nCount: {len(items)}\n\n" + "\n".join(items[:1000]),
                        encoding="utf-8")

    # Failure modes report (capped to top N most distinct)
    seen_ctx: set[str] = set()
    distinct_failures: list[dict[str, str]] = []
    for f in failures:
        key = f["context"][:80]
        if key in seen_ctx:
            continue
        seen_ctx.add(key)
        distinct_failures.append(f)
        if len(distinct_failures) >= args.max_failure_rows:
            break
    with (args.out_dir / "failure-modes.md").open("w", encoding="utf-8") as fh:
        fh.write("# Round-2 Step-2 — Failure Modes (Obsidian-extracted)\n\n")
        fh.write(f"Total distinct snippets retained: {len(distinct_failures)} (cap={args.max_failure_rows})\n\n")
        for i, f in enumerate(distinct_failures, 1):
            fh.write(f"## {i}. `{f['file']}` line {f['line']}\n\n```\n{f['context']}\n```\n\n")

    seen_ctx = set()
    distinct_evals: list[dict[str, str]] = []
    for e in evals:
        key = e["context"][:80]
        if key in seen_ctx:
            continue
        seen_ctx.add(key)
        distinct_evals.append(e)
        if len(distinct_evals) >= args.max_eval_rows:
            break
    with (args.out_dir / "eval-protocols.md").open("w", encoding="utf-8") as fh:
        fh.write("# Round-2 Step-2 — Evaluation Protocols (Obsidian-extracted)\n\n")
        fh.write(f"Total distinct snippets retained: {len(distinct_evals)} (cap={args.max_eval_rows})\n\n")
        for i, e in enumerate(distinct_evals, 1):
            fh.write(f"## {i}. `{e['file']}` line {e['line']}\n\n```\n{e['context']}\n```\n\n")

    print(f"DONE rows={len(rows)} categories={len(cat_counts)} failures={len(distinct_failures)} evals={len(distinct_evals)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
