#!/usr/bin/env python3
"""
Run three judges against a blind evaluation set and persist per-judge JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

try:
    from statsmodels.stats.proportion import proportion_confint
except ImportError:  # surfaced at runtime so callers see a clear message
    proportion_confint = None


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


def compute_h2_endpoint(
    samples: list[dict],
    consensus_path: Path,
    min_rows: int = 200,
) -> dict:
    """H2 primary endpoint: AI detection rate (recall on truth=AI) with Wilson 95% CI.

    H2 in TRAINING_DESIGN_V3.md §0:
      threshold:  AI 식별률 ≤ 40% AND Wilson upper < 50% AND N ≥ 200
      reject:     AI 식별률 > 50% OR Wilson upper ≥ 50%
    """
    if not consensus_path.exists():
        return {"verdict": "NO_CONSENSUS", "reason": f"{consensus_path} missing"}
    cons = json.loads(consensus_path.read_text(encoding="utf-8"))
    rows = cons.get("predictions", [])
    ai_rows = [r for r in rows if r.get("truth") == "AI"]
    n = len(ai_rows)
    correct = sum(1 for r in ai_rows if r.get("prediction") == "AI")
    rate = correct / n if n > 0 else None

    if n < min_rows:
        return {
            "verdict": "UNDERPOWERED",
            "ai_detection_rate": rate,
            "n_ai_samples": n,
            "min_rows_required": min_rows,
            "reason": f"truth=AI samples {n} < required {min_rows}",
        }

    if proportion_confint is None:
        return {
            "verdict": "MISSING_DEP",
            "reason": "statsmodels not installed; `.venv/bin/pip install statsmodels`",
        }

    wilson_low, wilson_high = proportion_confint(correct, n, alpha=0.05, method="wilson")

    if rate > 0.50 or wilson_high >= 0.50:
        verdict = "REJECT_H2"
    elif rate > 0.40 or wilson_high >= 0.50:
        verdict = "MARGINAL"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "ai_detection_rate": round(rate, 4),
        "wilson_ci_low": round(wilson_low, 4),
        "wilson_ci_high": round(wilson_high, 4),
        "n_ai_samples": n,
        "correct_ai_predictions": correct,
        "min_rows_required": min_rows,
        "thresholds": {"pass_max": 0.40, "reject_min": 0.50, "ci_alpha": 0.05},
    }


def compute_stratification(samples: list[dict], min_per_stratum: int = 20) -> dict:
    """Group by (kind × length_bucket × reply_depth) and flag under-covered strata."""
    counts: Counter = Counter()
    for s in samples:
        key = (
            s.get("kind", "?"),
            s.get("length_bucket", "?"),
            str(s.get("reply_depth", s.get("depth", "?"))),
        )
        counts[key] += 1
    under = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in counts.items() if v < min_per_stratum}
    return {
        "n_strata": len(counts),
        "min_per_stratum": min_per_stratum,
        "strata_under_count": len(under),
        "strata_under": under,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run three blind judges")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--min-rows",
        type=int,
        default=200,
        help="H2 primary endpoint requires N>=200 truth=AI samples (TRAINING_DESIGN_V3 §0)",
    )
    parser.add_argument("--strata-min", type=int, default=20)
    args = parser.parse_args()

    samples = load_samples(args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # pin to reproducible dated aliases — paper-grade requires fixed judge models.
    # Defaults bumped 2026-05-13 (cycle-7) — older claude-3-x aliases may be retired.
    anthropic_model_primary = os.environ.get(
        "PAPER_GRADE_JUDGE_CLAUDE_PRIMARY",
        os.environ.get("ANTHROPIC_PRIMARY_MODEL", "claude-sonnet-4-5-20250929"),
    )
    anthropic_model_secondary = os.environ.get(
        "PAPER_GRADE_JUDGE_CLAUDE_SECONDARY",
        os.environ.get("ANTHROPIC_SECONDARY_MODEL", "claude-haiku-4-5-20251001"),
    )
    openai_model = os.environ.get(
        "PAPER_GRADE_JUDGE_GPT",
        os.environ.get("OPENAI_MODEL", "gpt-4o-2024-11-20"),
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

    h2 = compute_h2_endpoint(samples, args.output_dir / "consensus.json", min_rows=args.min_rows)
    strata = compute_stratification(samples, min_per_stratum=args.strata_min)
    report = {"h2_primary_endpoint": h2, "stratification": strata}
    (args.output_dir / "h2_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"h2_verdict": h2.get("verdict"), "h2_report": str(args.output_dir / "h2_report.json")}, ensure_ascii=False))

    if h2.get("verdict") in {"REJECT_H2", "UNDERPOWERED", "MISSING_DEP", "NO_CONSENSUS"}:
        sys.exit(2)


if __name__ == "__main__":
    main()
