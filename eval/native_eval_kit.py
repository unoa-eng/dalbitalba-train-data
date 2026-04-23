#!/usr/bin/env python3
"""
Render a compact HTML or Markdown evaluation report from judge JSON outputs.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(path: Path) -> dict[int, dict]:
    samples: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                samples[row["id"]] = row
    return samples


def summarize(results_dir: Path) -> list[dict]:
    results: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        results.append(load_json(path))
    return results


def disagreement_rows(samples: dict[int, dict], results: list[dict]) -> list[dict]:
    judge_maps = {result["judge"]: {row["sample_id"]: row for row in result["predictions"]} for result in results}
    sample_ids = sorted(samples.keys())
    rows: list[dict] = []
    for sample_id in sample_ids:
        votes = [judge_maps[result["judge"]][sample_id]["prediction"] for result in results if result["judge"] != "consensus"]
        unique_votes = sorted(set(votes))
        if len(unique_votes) <= 1:
            continue
        rows.append(
            {
                "sample": samples[sample_id],
                "votes": votes,
            }
        )
    return rows[:10]


def render_markdown(samples: dict[int, dict], results: list[dict]) -> str:
    lines = ["# Blind Evaluation Report", ""]
    lines.append("## Accuracy")
    for result in results:
        lines.append(f"- `{result['judge']}`: {result['accuracy']:.2%} ({result['sample_count']} samples)")

    lines.extend(["", "## Disagreements"])
    for row in disagreement_rows(samples, results):
        sample = row["sample"]
        lines.append(f"- Sample `{sample['id']}` votes={row['votes']} truth={sample['truth']}")
        lines.append(f"  title={sample.get('title') or '(none)'}")
        excerpt = sample["text"][:240].replace("\n", " ")
        lines.append(f"  excerpt={excerpt}")

    return "\n".join(lines) + "\n"


def render_html(samples: dict[int, dict], results: list[dict]) -> str:
    accuracy_rows = "\n".join(
        f"<tr><td>{html.escape(result['judge'])}</td><td>{result['accuracy']:.2%}</td><td>{result['sample_count']}</td></tr>"
        for result in results
    )

    disagreement_html = []
    for row in disagreement_rows(samples, results):
        sample = row["sample"]
        disagreement_html.append(
            "<article>"
            f"<h3>Sample {sample['id']} | truth={html.escape(sample['truth'])} | votes={html.escape(str(row['votes']))}</h3>"
            f"<p><strong>title:</strong> {html.escape(sample.get('title') or '(none)')}</p>"
            f"<pre>{html.escape(sample['text'][:600])}</pre>"
            "</article>"
        )

    disagreements = "\n".join(disagreement_html) or "<p>No disagreements.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Blind Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.5; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    pre {{ background: #f6f8fa; padding: 12px; white-space: pre-wrap; }}
    article {{ border-top: 1px solid #ddd; padding-top: 16px; margin-top: 16px; }}
  </style>
</head>
<body>
  <h1>Blind Evaluation Report</h1>
  <table>
    <thead><tr><th>Judge</th><th>Accuracy</th><th>Samples</th></tr></thead>
    <tbody>
      {accuracy_rows}
    </tbody>
  </table>
  <section>
    <h2>Disagreements</h2>
    {disagreements}
  </section>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render blind eval report")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--format", choices=("html", "md"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    samples = load_samples(args.samples)
    results = summarize(args.results_dir)

    if args.format == "html":
        payload = render_html(samples, results)
    else:
        payload = render_markdown(samples, results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
