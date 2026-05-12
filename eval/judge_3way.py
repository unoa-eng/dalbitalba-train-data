#!/usr/bin/env python3
"""
Run three judges against a blind evaluation set and persist per-judge JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


LABEL_PATTERN = re.compile(r"\b(AI|HUMAN)\b", re.IGNORECASE)
FORMAL_PATTERN = re.compile(r"(습니다|입니다|하십시오|하도록|겠습니다)$")
SLANG_PATTERN = re.compile(r"(ㅋㅋ|ㅎㅎ|ㅠㅠ|ㅜㅜ|ㄷㄷ|;;|~{2,}|\.\.\.)")


def load_samples(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_prediction(raw: str) -> tuple[str, str]:
    try:
        data = json.loads(raw)
        label = str(data.get("label", "")).upper().strip()
        rationale = str(data.get("rationale", "")).strip()
        if label in {"AI", "HUMAN"}:
            return label, rationale
    except json.JSONDecodeError:
        pass

    match = LABEL_PATTERN.search(raw)
    label = match.group(1).upper() if match else "HUMAN"
    return label, raw.strip()


def heuristic_prediction(sample: dict) -> tuple[str, str]:
    text = sample.get("text", "").strip()
    score = 0
    if FORMAL_PATTERN.search(text):
      score += 2
    if SLANG_PATTERN.search(text):
      score -= 2
    if "\n" in text:
      score += 1
    if len(text) < 40:
      score -= 1
    label = "AI" if score >= 1 else "HUMAN"
    rationale = f"heuristic score={score} len={len(text)}"
    return label, rationale


def anthropic_judge(model: str, sample: dict) -> tuple[str, str]:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = (
        "You are judging whether a Korean community post was written by a human or by an AI assistant.\n"
        "Return strict JSON with keys label and rationale.\n"
        "Allowed labels: AI, HUMAN.\n\n"
        f"Title: {sample.get('title') or '(none)'}\n"
        f"Topic: {sample.get('topic') or '(unknown)'}\n"
        f"Text:\n{sample.get('text', '')}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    chunks: list[str] = []
    for item in response.content:
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
    return parse_prediction("\n".join(chunks))


def openai_judge(model: str, sample: dict) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        "Judge whether this Korean community post is HUMAN or AI generated.\n"
        "Return strict JSON with keys label and rationale.\n"
        "Allowed labels: AI, HUMAN.\n\n"
        f"Title: {sample.get('title') or '(none)'}\n"
        f"Topic: {sample.get('topic') or '(unknown)'}\n"
        f"Text:\n{sample.get('text', '')}"
    )
    response = client.responses.create(model=model, input=prompt, temperature=0)
    output_text = getattr(response, "output_text", None)
    if output_text:
        return parse_prediction(output_text)
    payload = json.dumps(response.model_dump(), ensure_ascii=False)
    return parse_prediction(payload)


def run_judge(samples: list[dict], name: str, runner, model: str, output_path: Path, mode: str) -> dict:
    predictions: list[dict] = []
    correct = 0

    for sample in samples:
        try:
            label, rationale = runner(model, sample) if mode != "heuristic" else heuristic_prediction(sample)
        except Exception as exc:
            label, rationale = heuristic_prediction(sample)
            rationale = f"fallback: {exc}; {rationale}"
        is_correct = label == sample.get("truth")
        if is_correct:
            correct += 1
        predictions.append(
            {
                "sample_id": sample["id"],
                "truth": sample.get("truth"),
                "prediction": label,
                "is_correct": is_correct,
                "rationale": rationale,
            }
        )

    result = {
        "judge": name,
        "mode": mode,
        "model": model,
        "sample_count": len(samples),
        "accuracy": round(correct / max(1, len(samples)), 4),
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def build_consensus(samples: list[dict], results: list[dict], output_path: Path) -> None:
    indexed = {result["judge"]: {row["sample_id"]: row for row in result["predictions"]} for result in results}
    rows: list[dict] = []
    correct = 0

    for sample in samples:
        votes = []
        for judge_result in results:
            row = indexed[judge_result["judge"]][sample["id"]]
            votes.append(row["prediction"])
        counts = Counter(votes)
        prediction, _ = counts.most_common(1)[0]
        is_correct = prediction == sample.get("truth")
        if is_correct:
            correct += 1
        rows.append(
            {
                "sample_id": sample["id"],
                "truth": sample.get("truth"),
                "prediction": prediction,
                "votes": votes,
                "is_correct": is_correct,
            }
        )

    payload = {
        "judge": "consensus",
        "sample_count": len(samples),
        "accuracy": round(correct / max(1, len(samples)), 4),
        "predictions": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three blind judges")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    samples = load_samples(args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # pin to reproducible dated alias — paper-grade requires fixed judge model
    anthropic_model_primary = os.environ.get(
        "PAPER_GRADE_JUDGE_CLAUDE_PRIMARY",
        os.environ.get(
            "ANTHROPIC_PRIMARY_MODEL",
            "claude-3-7-sonnet-20250219",  # verify alias is still served by API
        ),
    )
    # pin to reproducible dated alias — paper-grade requires fixed judge model
    anthropic_model_secondary = os.environ.get(
        "PAPER_GRADE_JUDGE_CLAUDE_SECONDARY",
        os.environ.get(
            "ANTHROPIC_SECONDARY_MODEL",
            "claude-3-5-haiku-20241022",
        ),
    )
    # pin to reproducible dated alias — paper-grade requires fixed judge model
    openai_model = os.environ.get(
        "PAPER_GRADE_JUDGE_GPT",
        os.environ.get(
            "OPENAI_MODEL",
            "gpt-4o-2024-11-20",  # conservative well-known dated alias; gpt-4.1-2025-04-14 unverified
        ),
    )

    results: list[dict] = []

    if os.environ.get("ANTHROPIC_API_KEY"):
        results.append(
            run_judge(
                samples,
                "claude_primary",
                anthropic_judge,
                anthropic_model_primary,
                args.output_dir / "claude_primary.json",
                "anthropic",
            )
        )
        results.append(
            run_judge(
                samples,
                "claude_secondary",
                anthropic_judge,
                anthropic_model_secondary,
                args.output_dir / "claude_secondary.json",
                "anthropic",
            )
        )
    else:
        results.append(
            run_judge(
                samples,
                "claude_primary",
                lambda _model, sample: heuristic_prediction(sample),
                "heuristic-primary",
                args.output_dir / "claude_primary.json",
                "heuristic",
            )
        )
        results.append(
            run_judge(
                samples,
                "claude_secondary",
                lambda _model, sample: heuristic_prediction(sample),
                "heuristic-secondary",
                args.output_dir / "claude_secondary.json",
                "heuristic",
            )
        )

    if os.environ.get("OPENAI_API_KEY"):
        results.append(
            run_judge(
                samples,
                "gpt",
                openai_judge,
                openai_model,
                args.output_dir / "gpt.json",
                "openai",
            )
        )
    else:
        results.append(
            run_judge(
                samples,
                "gpt",
                lambda _model, sample: heuristic_prediction(sample),
                "heuristic-openai",
                args.output_dir / "gpt.json",
                "heuristic",
            )
        )

    build_consensus(samples, results, args.output_dir / "consensus.json")
    print(f"[done] wrote judge outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
