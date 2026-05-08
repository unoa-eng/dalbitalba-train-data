#!/usr/bin/env python3
"""Stage J: PR #6 style classifier v2 — diversified-negatives retrain.

The PR #6 classifier scored AUC ~0.99 because the negative class was
exclusively formal Korean (Wikipedia / KMMLU / bundled formal sentences).
The task became "informal vs formal" rather than "dalbi-forum-style vs
generic-Korean", which is trivially separable on character-level cues
(emoji, ㅋㅋ density, sentence-final ㅁ-cuts) and over-fires the gate
during inference.

This v2 mixes the negatives:
  - 33% wikimedia/wikipedia ko (formal expository)        — same as PR #6
  - 33% klue/klue ynat (Korean news headlines)            — formal but conversational
  - 17% klue/klue nli premises (declarative)              — short formal
  - 17% synthesized polite-forum 존댓말                    — sourced from
        sft_pairs.v3.jsonl rows that end in '습니다/입니다/세요/요.' to
        force the classifier to discriminate dalbi-slang vs polite-forum

Acceptance gate: 0.80 < test_AUC < 0.99.
Output: runs/local-integrity-2026-05-08/J_style_classifier_v2/
"""
import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/unoa/dalbitalba-train-data")
SEED = 42
TARGET_PER_CLASS = 5000
BASE_MODEL = "klue/roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
COMMUNITY_PATTERN = re.compile(r"ㅈㄴ|ㅇㅈ|ㄹㅇ|ㅋㅋ|ㅎㅎ|쩜오|텐카|초이스|손놈|업진|밀빵")
POLITE_ENDING = re.compile(r"(?:습니다|입니다|세요|니까|십시오|드립니다)[\.\?\!]?\s*$")


def load_positive(corpus: Path, n: int, rng: random.Random) -> list[str]:
    rows = []
    with open(corpus) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = (d.get("text") or "").strip()
            if len(t) >= 30 and COMMUNITY_PATTERN.search(t):
                rows.append(t)
    rng.shuffle(rows)
    return rows[:n]


def load_polite_forum(sft_path: Path, n: int, rng: random.Random) -> list[str]:
    """Synthesize polite-forum negatives from sft_pairs.v3.jsonl target_comment
    rows that end in formal endings."""
    rows = []
    with open(sft_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = (d.get("target_comment") or "").strip()
            if 15 <= len(t) <= 256 and POLITE_ENDING.search(t):
                # Avoid leak: skip if community markers present
                if not COMMUNITY_PATTERN.search(t):
                    rows.append(t)
    rng.shuffle(rows)
    return rows[:n]


def stream_dataset_texts(name: str, config: str | None, split: str, key: str, n: int, max_chars: int = 512) -> list[str]:
    from datasets import load_dataset
    args = (name,) if config is None else (name, config)
    ds = load_dataset(*args, split=split, streaming=True)
    out = []
    for it in ds:
        v = it.get(key)
        if isinstance(v, str):
            v = v.strip()
            if v:
                # take first paragraph
                para = v.split("\n\n")[0].strip()
                if len(para) >= 50:
                    out.append(para[:max_chars])
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "runs/local-integrity-2026-05-08/J_style_classifier_v2"))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg: str):
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    rng = random.Random(SEED)
    t0 = time.time()
    log("=== Stage J: classifier v2 retrain ===")

    # POSITIVES
    log("[1/5] Loading positives from cpt_corpus.v3.jsonl ...")
    positives = load_positive(ROOT / "cpt_corpus.v3.jsonl", args.target_per_class, rng)
    log(f"  positives: {len(positives)}")

    # NEGATIVES (4 sources)
    n_per_class = len(positives)
    n_wiki = int(n_per_class * 0.33)
    n_ynat = int(n_per_class * 0.33)
    n_nli = int(n_per_class * 0.17)
    n_polite = n_per_class - n_wiki - n_ynat - n_nli

    log(f"[2/5] Loading wikipedia ko (target {n_wiki}) ...")
    wiki = stream_dataset_texts("wikimedia/wikipedia", "20231101.ko", "train", "text", n_wiki)
    log(f"  wikipedia: {len(wiki)}")

    log(f"[3/5] Loading klue ynat (target {n_ynat}) ...")
    try:
        ynat = stream_dataset_texts("klue/klue", "ynat", "train", "title", n_ynat)
    except Exception as e:
        log(f"  ynat fallback: {e}")
        ynat = []
    log(f"  ynat: {len(ynat)}")

    log(f"[4a/5] Loading klue nli premises (target {n_nli}) ...")
    try:
        nli = stream_dataset_texts("klue/klue", "nli", "train", "premise", n_nli)
    except Exception as e:
        log(f"  nli fallback: {e}")
        nli = []
    log(f"  nli: {len(nli)}")

    log(f"[4b/5] Synthesizing polite-forum 존댓말 from sft_pairs.v3.jsonl (target {n_polite}) ...")
    polite = load_polite_forum(ROOT / "sft_pairs.v3.jsonl", n_polite, rng)
    log(f"  polite: {len(polite)}")

    negatives = wiki + ynat + nli + polite
    rng.shuffle(negatives)
    # Trim to match positives count
    target_neg = len(positives)
    if len(negatives) < target_neg:
        # top up with wikipedia
        more_wiki = stream_dataset_texts("wikimedia/wikipedia", "20231101.ko", "train", "text", target_neg - len(negatives) + 100)
        negatives += more_wiki
    negatives = negatives[:target_neg]
    log(f"  total negatives: {len(negatives)} (mix: wiki={len(wiki)} ynat={len(ynat)} nli={len(nli)} polite={len(polite)})")

    # Save splits manifest
    manifest = {
        "stage": "J",
        "purpose": "PR#6 classifier v2 with diversified negatives",
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "negatives_mix": {
            "wikipedia": len(wiki),
            "klue_ynat": len(ynat),
            "klue_nli": len(nli),
            "polite_forum_synth": len(polite),
        },
        "base_model": BASE_MODEL,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Train
    log("[5/5] Training klue/roberta-base ...")
    from sklearn.model_selection import train_test_split
    pos_pairs = [(t, 1) for t in positives]
    neg_pairs = [(t, 0) for t in negatives]
    all_data = pos_pairs + neg_pairs
    texts_x = [t for t, _ in all_data]
    labels_y = [l for _, l in all_data]
    tr_t, tmp_t, tr_l, tmp_l = train_test_split(texts_x, labels_y, test_size=0.2, stratify=labels_y, random_state=SEED)
    va_t, te_t, va_l, te_l = train_test_split(tmp_t, tmp_l, test_size=0.5, stratify=tmp_l, random_state=SEED)
    log(f"  splits: train={len(tr_t)} val={len(va_t)} test={len(te_t)}")

    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    device = torch.device("cpu")
    log(f"  loading tokenizer + model ({BASE_MODEL})")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.to(device)

    class StyleDataset(TorchDataset):
        def __init__(self, pairs):
            self.pairs = pairs
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, i):
            t, l = self.pairs[i]
            enc = tokenizer(t, max_length=args.max_length, truncation=True, padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(l, dtype=torch.long),
            }

    train_loader = DataLoader(StyleDataset(list(zip(tr_t, tr_l))), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StyleDataset(list(zip(va_t, va_l))), batch_size=BATCH_SIZE)
    test_loader = DataLoader(StyleDataset(list(zip(te_t, te_l))), batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    epoch_metrics = []
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        nb = 0
        ep_t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            optimizer.step()
            ep_loss += float(out.loss)
            nb += 1
            if nb % 50 == 0:
                log(f"    ep{ep} batch{nb}/{len(train_loader)} loss={ep_loss/nb:.4f}")
        # val
        model.eval()
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                pred = out.logits.argmax(dim=-1)
                val_correct += int((pred == batch["labels"].to(device)).sum())
                val_n += batch["labels"].size(0)
        val_acc = val_correct / max(val_n, 1)
        ep_dt = time.time() - ep_t0
        log(f"  ep{ep} done loss={ep_loss/nb:.4f} val_acc={val_acc:.4f} dt={ep_dt:.1f}s")
        epoch_metrics.append({"epoch": ep, "train_loss": ep_loss / nb, "val_acc": val_acc, "elapsed_sec": round(ep_dt, 1)})

    # Test set: collect probs for AUC
    log("  evaluating on test ...")
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].tolist())
    test_auc = float(roc_auc_score(all_labels, all_probs))
    test_acc = float(accuracy_score(all_labels, all_preds))
    cm = confusion_matrix(all_labels, all_preds).tolist()
    log(f"  test_auc={test_auc:.4f}  test_acc={test_acc:.4f}  cm={cm}")

    # Acceptance gate
    gate_pass = 0.80 <= test_auc < 0.99
    metrics = {
        "test_auc": test_auc,
        "test_acc": test_acc,
        "confusion_matrix": cm,
        "epochs": epoch_metrics,
        "gate_pass_target_0.80_lt_AUC_lt_0.99": gate_pass,
        "elapsed_sec": round(time.time() - t0, 1),
        "manifest": manifest,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save model + tokenizer
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    log(f"  saved model to {model_dir}")
    log(f"=== DONE: AUC={test_auc:.4f} gate_pass={gate_pass} elapsed={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
