#!/usr/bin/env python3
"""
style_classifier_eval.py

Standalone evaluator: loads a trained Korean style classifier and returns
the mean community-style probability (score) for a batch of texts.

Usage (CLI):
  python3 scripts/style_classifier_eval.py \\
      --model runs/style_classifier_<timestamp>/ \\
      --texts sample.jsonl \\
      [--out report.json]

Programmatic usage (imported from phase6_eval.py):
  from scripts.style_classifier_eval import StyleClassifierEval
  clf = StyleClassifierEval("runs/style_classifier_20260507T.../")
  auc = clf.auc(ai_texts, raw_texts)   # lower = more formal/indistinguishable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class StyleClassifierEval:
    """Thin wrapper around a saved klue/roberta-base binary classifier."""

    def __init__(self, model_dir: str):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"torch/transformers not installed: {exc}") from exc

        self._torch = torch
        self._F = torch.nn.functional
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts: list[str], max_length: int = 256, batch_size: int = 16) -> list[float]:
        """Return P(community) for each text."""
        probs: list[float] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with self._torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                p = self._F.softmax(logits, dim=-1)[:, 1].tolist()
            probs.extend(p)
        return probs

    def auc(self, community_texts: list[str], formal_texts: list[str]) -> float:
        """
        Compute ROC-AUC treating community_texts as label=1, formal_texts as label=0.
        AUC < 0.65 means the classifier cannot distinguish => good for eval gate.
        """
        from sklearn.metrics import roc_auc_score  # type: ignore

        texts = community_texts + formal_texts
        labels = [1] * len(community_texts) + [0] * len(formal_texts)
        probs = self.predict_proba(texts)
        return float(roc_auc_score(labels, probs))

    def smoke_test(self, community_samples: list[str], formal_samples: list[str]) -> dict:
        """
        Returns mean community score for each group.
        Pass: community_mean > 0.7, formal_mean < 0.3.
        """
        c_probs = self.predict_proba(community_samples)
        f_probs = self.predict_proba(formal_samples)
        c_mean = sum(c_probs) / len(c_probs) if c_probs else 0.0
        f_mean = sum(f_probs) / len(f_probs) if f_probs else 0.0
        return {
            "community_mean_score": round(c_mean, 4),
            "formal_mean_score": round(f_mean, 4),
            "community_pass": c_mean > 0.7,
            "formal_pass": f_mean < 0.3,
            "smoke_pass": c_mean > 0.7 and f_mean < 0.3,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Korean style classifier evaluation")
    parser.add_argument("--model", required=True, help="Path to saved model directory")
    parser.add_argument("--texts", help="JSONL file with {text:...} entries to score")
    parser.add_argument("--community", help="JSONL with community texts for AUC/smoke")
    parser.add_argument("--formal", help="JSONL with formal texts for AUC/smoke")
    parser.add_argument("--out", help="Write JSON report to this path")
    args = parser.parse_args()

    clf = StyleClassifierEval(args.model)

    report: dict = {}

    if args.texts:
        texts = []
        with open(args.texts, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    t = (obj.get("text") or "").strip()
                    if t:
                        texts.append(t)
                except json.JSONDecodeError:
                    continue
        probs = clf.predict_proba(texts)
        report["scores"] = [round(p, 4) for p in probs]
        report["mean_community_score"] = round(sum(probs) / len(probs), 4) if probs else None

    if args.community and args.formal:
        def _load(path: str) -> list[str]:
            out = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = (obj.get("text") or "").strip()
                        if t:
                            out.append(t)
                    except json.JSONDecodeError:
                        continue
            return out

        c_texts = _load(args.community)
        f_texts = _load(args.formal)
        report["auc"] = clf.auc(c_texts, f_texts)
        report["smoke"] = clf.smoke_test(c_texts[:10], f_texts[:10])

    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
