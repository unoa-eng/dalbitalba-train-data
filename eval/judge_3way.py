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

    anthropic_model_primary = os.environ.get("ANTHROPIC_PRIMARY_MODEL", "claude-opus-4-6")
    anthropic_model_secondary = os.environ.get("ANTHROPIC_SECONDARY_MODEL", "claude-sonnet-4-6")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-5.4")

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

    # ── Obsidian EVAL-PROTOCOL: FP/FN 분리 + kind 층화 + 편향 분석 ──
    report = compute_detailed_metrics(samples, results)
    report_path = args.output_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVAL REPORT (Obsidian Protocol)")
    print(f"{'='*60}")
    for judge_name, metrics in report["per_judge"].items():
        print(f"\n  {judge_name}:")
        print(f"    accuracy={metrics['accuracy']:.3f}  FP={metrics['fp_rate']:.3f}  FN={metrics['fn_rate']:.3f}")
        print(f"    |FP-FN|={metrics['fp_fn_gap']:.3f}  bias={metrics['bias']}")
        if metrics.get("per_kind"):
            for kind, km in metrics["per_kind"].items():
                print(f"      {kind}: acc={km['accuracy']:.3f} n={km['n']}")
    overall = report["overall"]
    print(f"\n  OVERALL: accuracy={overall['accuracy']:.3f} FP={overall['fp_rate']:.3f} FN={overall['fn_rate']:.3f}")
    print(f"  TARGET: accuracy ≤ 0.60, |FP-FN| < 0.15")
    print(f"  VERDICT: {overall['verdict']}")
    print(f"{'='*60}")

    print(f"[done] wrote judge outputs + eval_report to {args.output_dir}")


def compute_detailed_metrics(samples: list[dict], results: list[dict]) -> dict:
    """Obsidian EVAL-PROTOCOL 4축 분석"""
    from collections import defaultdict

    report = {"per_judge": {}, "overall": {}}

    all_preds = []
    for result in results:
        preds = result["predictions"]
        tp = fn = fp = tn = 0
        per_kind = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "n": 0})

        for pred in preds:
            truth = pred["truth"]
            predicted = pred["prediction"]
            sample = next((s for s in samples if s["id"] == pred["sample_id"]), {})
            kind = sample.get("kind", "unknown")

            per_kind[kind]["n"] += 1

            if truth == "AI" and predicted == "AI":
                tp += 1; per_kind[kind]["tp"] += 1
            elif truth == "AI" and predicted == "HUMAN":
                fn += 1; per_kind[kind]["fn"] += 1
            elif truth == "HUMAN" and predicted == "AI":
                fp += 1; per_kind[kind]["fp"] += 1
            elif truth == "HUMAN" and predicted == "HUMAN":
                tn += 1; per_kind[kind]["tn"] += 1

        total = tp + fn + fp + tn
        ai_total = tp + fn
        human_total = fp + tn
        accuracy = (tp + tn) / max(total, 1)
        fp_rate = fp / max(human_total, 1)  # HUMAN→AI 오판
        fn_rate = fn / max(ai_total, 1)     # AI→HUMAN 오판
        fp_fn_gap = abs(fp_rate - fn_rate)

        # Bias detection (Obsidian: Judge가 AI 판정을 기피하는 편향)
        ai_predictions = sum(1 for p in preds if p["prediction"] == "AI")
        human_predictions = sum(1 for p in preds if p["prediction"] == "HUMAN")
        if total > 0:
            ai_pred_ratio = ai_predictions / total
            bias = "neutral"
            if ai_pred_ratio < 0.3:
                bias = "AI판정_기피"
            elif ai_pred_ratio > 0.7:
                bias = "AI판정_과다"
        else:
            bias = "unknown"

        kind_metrics = {}
        for kind, km in per_kind.items():
            kn = km["n"]
            kacc = (km["tp"] + km["tn"]) / max(kn, 1)
            kind_metrics[kind] = {"accuracy": round(kacc, 3), "n": kn,
                                  "tp": km["tp"], "fn": km["fn"], "fp": km["fp"], "tn": km["tn"]}

        report["per_judge"][result["judge"]] = {
            "accuracy": round(accuracy, 3),
            "fp_rate": round(fp_rate, 3),
            "fn_rate": round(fn_rate, 3),
            "fp_fn_gap": round(fp_fn_gap, 3),
            "bias": bias,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "per_kind": kind_metrics,
        }
        all_preds.append({"tp": tp, "fn": fn, "fp": fp, "tn": tn})

    # Overall (consensus-like)
    total_tp = sum(p["tp"] for p in all_preds)
    total_fn = sum(p["fn"] for p in all_preds)
    total_fp = sum(p["fp"] for p in all_preds)
    total_tn = sum(p["tn"] for p in all_preds)
    total_all = total_tp + total_fn + total_fp + total_tn
    overall_acc = (total_tp + total_tn) / max(total_all, 1)
    overall_fp_rate = total_fp / max(total_fp + total_tn, 1)
    overall_fn_rate = total_fn / max(total_tp + total_fn, 1)

    verdict = "PASS" if overall_acc <= 0.60 and abs(overall_fp_rate - overall_fn_rate) < 0.15 else "FAIL"

    report["overall"] = {
        "accuracy": round(overall_acc, 3),
        "fp_rate": round(overall_fp_rate, 3),
        "fn_rate": round(overall_fn_rate, 3),
        "fp_fn_gap": round(abs(overall_fp_rate - overall_fn_rate), 3),
        "verdict": verdict,
        "target": "accuracy ≤ 0.60, |FP-FN| < 0.15",
    }

    return report


if __name__ == "__main__":
    main()
