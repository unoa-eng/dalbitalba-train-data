#!/usr/bin/env python3
"""Create a deterministic held-out TC-SFT eval split."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "sft_thread_conditioned.jsonl"
EVAL = ROOT / "sft_thread_conditioned.eval.jsonl"
REPORT = ROOT / "runs" / "round2-sft-eval-split.json"


def main() -> int:
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = []
    eval_rows = []
    for idx, row in enumerate(rows):
        if idx % 10 == 0:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    with SRC.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with EVAL.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {"source": str(SRC.name), "train_rows": len(train_rows), "eval_rows": len(eval_rows)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"train_rows": len(train_rows), "eval_rows": len(eval_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
