#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = [
    "bigram_jsd",
    "length_kl",
    "digit_density_delta",
    "english_density_delta",
    "mauve_score",
]


def format_threshold(spec: dict) -> str:
    op = spec.get("op", "?")
    value = spec.get("value")
    symbol = {"le": "<=", "ge": ">="}.get(op, op)
    return f"{symbol} {value}"


def metric_passes(value: object, spec: dict) -> bool:
    if not isinstance(value, (int, float)):
        return False
    op = spec.get("op")
    threshold = spec.get("value")
    if not isinstance(threshold, (int, float)):
        return False
    if op == "le":
        return value <= threshold
    if op == "ge":
        return value >= threshold
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", required=True, help="Run suffix for runs/eval-run-<arg>")
    args = parser.parse_args()

    run_dir = Path("runs") / f"eval-run-{args.eval_run}"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "CYCLE_REPORT.md"

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    thresholds = payload.get("thresholds", {})
    verdict = payload.get("gate", {}).get("verdict", "UNKNOWN")

    lines = [
        f"# Cycle Report: eval-run-{args.eval_run}",
        "",
        "| Metric | Value | Threshold | Status |",
        "| --- | ---: | --- | --- |",
    ]
    for name in METRICS:
        value = metrics.get(name)
        threshold = thresholds.get(name, {})
        status = "PASS" if metric_passes(value, threshold) else "FAIL"
        value_text = f"{value:.6f}" if isinstance(value, (int, float)) else "N/A"
        lines.append(
            f"| {name} | {value_text} | {format_threshold(threshold)} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Gate Verdict",
            "",
            f"`{verdict}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
