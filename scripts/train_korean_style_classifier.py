#!/usr/bin/env python3
"""
train_korean_style_classifier.py

Binary Korean style classifier:
  label=1  community/informal speech  (from cpt_corpus.v2.jsonl)
  label=0  formal Korean              (Wikipedia KO or KMMLU or bundled fallback)

Output:
  runs/style_classifier_<UTC_timestamp>/  — saved model + tokenizer
  .planning/calibration/style_classifier_baseline.json — metrics

Usage:
  python3 scripts/train_korean_style_classifier.py \\
      --corpus cpt_corpus.v2.jsonl \\
      --out-dir runs/ \\
      --baseline .planning/calibration/style_classifier_baseline.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ─── POSITIVE CORPUS KEYWORDS ─────────────────────────────────────────────────
COMMUNITY_PATTERN = re.compile(r"ㅈㄴ|ㅇㅈ|ㄹㅇ|ㅋㅋ|ㅎㅎ|쩜오|텐카|초이스|손놈|업진|밀빵")

SEED = 42
# Production config — full klue/roberta-base classifier on 5K+5K, 3 epochs,
# max_length=256. Wall-clock estimate on a single L40S CPU host: 60-130
# minutes (no GPU available in this project's training pod budget for
# the classifier). A reduced PoC config (1000+1000, max_length=128,
# epochs=1) was attempted on 2026-05-07 to fit a battery-only dev
# window, but the trainer hung at the model-load → train-loop transition
# (single-thread CPU bottleneck on a stale klue/roberta-base load).
# Re-run on AC power with the values below — see
# docs/HANDOFF_P1B_20260507.md for the full reproduction recipe.
MAX_LENGTH = 256
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
BASE_MODEL = "klue/roberta-base"
MAX_PER_BUCKET = 1500
TARGET_POSITIVE = 5000
TARGET_NEGATIVE = 5000


# ─── BUNDLED FORMAL FALLBACK ──────────────────────────────────────────────────
BUNDLED_FORMAL = [
    "대한민국은 동아시아에 위치한 나라로, 한반도의 남쪽에 자리하고 있다.",
    "서울은 대한민국의 수도로서 정치, 경제, 문화의 중심지 역할을 담당한다.",
    "조선왕조는 1392년부터 1897년까지 약 500년간 한반도를 통치하였다.",
    "한국어는 알타이어족으로 분류되며, 고유한 문자 체계인 한글을 사용한다.",
    "한글은 조선 세종대왕이 1443년에 창제한 표음 문자로, 과학적 구조로 유명하다.",
    "불교는 삼국시대에 한반도에 전래되어 고려시대에 국교로 자리매김하였다.",
    "대한민국 헌법은 1948년에 제정되었으며, 현재까지 여러 차례 개정을 거쳤다.",
    "경제 성장률은 전년 대비 2.3% 상승하였으며, 수출 규모도 증가하였다.",
    "국립중앙박물관은 서울 용산구에 위치하며, 다양한 역사 유물을 소장하고 있다.",
    "환경부는 온실가스 감축 목표를 설정하고 친환경 정책을 추진하고 있다.",
    "과학기술정보통신부는 디지털 전환 가속화를 위한 종합 계획을 발표하였다.",
    "한국의 고등교육 진학률은 OECD 국가 중 상위권에 해당하는 것으로 나타났다.",
    "국제통화기금(IMF)은 한국의 경제 성장 전망을 긍정적으로 평가하였다.",
    "고려청자는 12세기경에 최고의 기술 수준에 도달하였으며, 세계적으로 인정받는다.",
    "한반도의 지형은 산악 지대가 전체 면적의 70% 이상을 차지하고 있다.",
    "행정안전부는 지방자치단체의 행정 효율화를 위한 지침을 배포하였다.",
    "국회는 법률의 제정 및 개정, 예산 심의 등의 권한을 헌법으로부터 부여받는다.",
    "문화재청은 역사적 가치가 높은 유산을 국가 지정 문화재로 등록하여 보호한다.",
    "한국의 의료 보험 체계는 전 국민을 대상으로 하는 단일 보험자 방식을 채택한다.",
    "서울대학교는 1946년에 설립된 대한민국의 대표적인 국립 종합 대학교이다.",
    "해양수산부는 해양 자원의 지속 가능한 이용을 위한 정책을 수립하고 있다.",
    "조선시대의 과거 제도는 관리 선발을 위한 국가 시험으로 유교적 소양을 중시하였다.",
    "기상청은 태풍 발생 시 즉각적인 경보 체계를 가동하여 국민 안전을 도모한다.",
    "정부는 저출생 고령화 문제 해결을 위한 종합 대책을 마련하여 시행 중이다.",
    "인공지능 기술의 발전은 산업 전반에 걸쳐 광범위한 변화를 가져오고 있다.",
    "한국은 반도체, 자동차, 조선 등 주요 제조업 분야에서 세계 시장을 선도하고 있다.",
    "민법상 계약은 청약과 승낙의 의사 표시가 합치되어야 성립하는 것으로 규정된다.",
    "대법원은 최종 심급으로서 법률 해석의 통일성을 유지하는 역할을 수행한다.",
    "국토교통부는 균형 있는 지역 개발을 위한 국가 공간 계획을 수립하고 있다.",
    "한국의 전통 건축은 자연환경과의 조화를 중시하는 독특한 미학적 특성을 지닌다.",
    "금융감독원은 금융 시장의 안정성과 투명성 확보를 위한 감독 기능을 수행한다.",
    "연구 결과에 따르면 규칙적인 신체 활동은 만성 질환 예방에 효과적인 것으로 나타났다.",
    "한국의 인터넷 보급률은 세계 최고 수준으로, 디지털 인프라가 잘 갖추어져 있다.",
    "문학 작품은 시대적 배경과 사회적 맥락을 이해하는 데 중요한 자료가 된다.",
    "통계청의 발표에 의하면 지난해 소비자물가지수는 전년 대비 3.2% 상승하였다.",
    "교육부는 학교 폭력 예방을 위한 교육 프로그램을 전국 초중고에 확대 적용한다.",
    "근대 이후 한국 사회는 급격한 산업화와 도시화 과정을 경험하였다.",
    "외교부는 국제 사회에서의 한국의 위상 제고를 위한 공공 외교를 강화하고 있다.",
    "국방부는 첨단 무기 체계 도입 계획을 발표하며 자주 국방 능력 강화를 천명하였다.",
    "한국의 전통 의학인 한의학은 음양오행 이론에 기반한 치료 체계를 갖추고 있다.",
    "여성가족부는 성별 임금 격차 해소를 위한 제도적 개선 방안을 추진하고 있다.",
    "생물다양성 보전을 위한 국제 협약 이행은 각국 정부의 중요한 의무 사항이다.",
    "산업통상자원부는 신재생에너지 확대를 위한 중장기 에너지 전환 계획을 수립하였다.",
    "한국어의 경어법은 상대방과의 관계에 따라 다양한 존댓말 체계를 활용한다.",
    "대한민국 임시정부는 1919년 중국 상하이에서 수립되어 독립운동을 이끌었다.",
    "심리학 연구에 따르면 사회적 지지는 스트레스 완화에 중요한 역할을 한다.",
    "고령화 사회에 대응하기 위해 노인 복지 서비스의 확충이 시급한 과제로 대두된다.",
    "한국의 현대 미술은 전통과 현대의 융합을 통해 독자적인 예술 세계를 구축하고 있다.",
    "보건복지부는 정신 건강 증진을 위한 지역사회 중심의 서비스 체계를 구축하고 있다.",
    "국가인권위원회는 차별 행위에 대한 조사와 권고를 통해 인권 보호 기능을 수행한다.",
]


def load_positive(corpus_path: str) -> list[str]:
    """Load community-speech samples from cpt_corpus.v2.jsonl."""
    print(f"[+] Loading positive corpus from {corpus_path}", flush=True)
    rows: dict[str, list[str]] = {}  # bucket -> texts
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "").strip()
            bucket = obj.get("length_bucket", "other")
            if len(text) >= 30 and COMMUNITY_PATTERN.search(text):
                rows.setdefault(bucket, []).append(text)

    rng = random.Random(SEED)
    collected: list[str] = []
    for bucket, texts in rows.items():
        rng.shuffle(texts)
        collected.extend(texts[: MAX_PER_BUCKET])

    rng.shuffle(collected)
    result = collected[:TARGET_POSITIVE]
    print(f"[+] Positive samples: {len(result)}", flush=True)
    return result


def load_negative_wikipedia() -> list[str] | None:
    """Try loading formal Korean from Wikipedia via HuggingFace datasets."""
    try:
        from datasets import load_dataset  # type: ignore
        print("[+] Trying wikimedia/wikipedia ko subset...", flush=True)
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ko",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        texts: list[str] = []
        for item in ds:
            text = (item.get("text") or "").strip()
            # take first paragraph (up to first double-newline)
            para = text.split("\n\n")[0].strip()
            if len(para) >= 50:
                texts.append(para[:512])
            if len(texts) >= TARGET_NEGATIVE:
                break
        if texts:
            print(f"[+] Wikipedia negative samples: {len(texts)}", flush=True)
            return texts
    except Exception as exc:
        print(f"[!] Wikipedia load failed: {exc}", flush=True)
    return None


def load_negative_kmmlu() -> list[str] | None:
    """Try loading formal Korean from KMMLU question stems."""
    try:
        from datasets import load_dataset  # type: ignore
        print("[+] Trying HAERAE-HUB/KMMLU...", flush=True)
        ds = load_dataset("HAERAE-HUB/KMMLU", "all", split="test", trust_remote_code=True)
        texts: list[str] = []
        for item in ds:
            q = (item.get("question") or "").strip()
            if len(q) >= 20:
                texts.append(q)
            if len(texts) >= TARGET_NEGATIVE:
                break
        if texts:
            print(f"[+] KMMLU negative samples: {len(texts)}", flush=True)
            return texts
    except Exception as exc:
        print(f"[!] KMMLU load failed: {exc}", flush=True)
    return None


def load_negative_bundled() -> list[str]:
    """Fallback: repeat bundled formal sentences to reach TARGET_NEGATIVE."""
    print("[!] Using bundled formal fallback corpus (no internet access)", flush=True)
    rng = random.Random(SEED)
    texts: list[str] = []
    base = BUNDLED_FORMAL[:]
    while len(texts) < TARGET_NEGATIVE:
        rng.shuffle(base)
        texts.extend(base)
    return texts[:TARGET_NEGATIVE]


def load_negative() -> tuple[list[str], str]:
    """Return (texts, source_label)."""
    neg = load_negative_wikipedia()
    if neg and len(neg) >= 100:
        return neg, "wikimedia/wikipedia (ko)"
    neg = load_negative_kmmlu()
    if neg and len(neg) >= 100:
        return neg, "HAERAE-HUB/KMMLU"
    return load_negative_bundled(), "bundled_fallback"


def build_splits(
    positives: list[str], negatives: list[str]
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    """80/10/10 stratified split."""
    from sklearn.model_selection import train_test_split  # type: ignore

    pos_labels = [(t, 1) for t in positives]
    neg_labels = [(t, 0) for t in negatives]
    all_data = pos_labels + neg_labels
    texts = [t for t, _ in all_data]
    labels = [l for _, l in all_data]

    tr_t, tmp_t, tr_l, tmp_l = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    va_t, te_t, va_l, te_l = train_test_split(
        tmp_t, tmp_l, test_size=0.5, stratify=tmp_l, random_state=SEED
    )
    train = list(zip(tr_t, tr_l))
    val = list(zip(va_t, va_l))
    test = list(zip(te_t, te_l))
    return train, val, test


class StyleDataset:
    def __init__(self, data: list[tuple[str, int]], tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        import torch  # type: ignore
        text, label = self.data[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train(
    train_data: list[tuple[str, int]],
    val_data: list[tuple[str, int]],
    out_dir: Path,
) -> tuple[object, object, list[float]]:
    """Train and return (model, tokenizer, val_probs)."""
    import torch  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    torch.manual_seed(SEED)
    device = torch.device("cpu")

    print(f"[+] Loading tokenizer: {BASE_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model.to(device)

    train_ds = StyleDataset(train_data, tokenizer, MAX_LENGTH)
    val_ds = StyleDataset(val_data, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  epoch {epoch}/{EPOCHS}  train_loss={avg_loss:.4f}", flush=True)

    # Validation probabilities
    model.eval()
    import torch.nn.functional as F  # type: ignore

    val_probs: list[float] = []
    val_true: list[int] = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].tolist()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)[:, 1].tolist()
            val_probs.extend(probs)
            val_true.extend(labels)

    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[+] Model saved to {out_dir}", flush=True)
    return model, tokenizer, val_probs, val_true


def evaluate_metrics(
    model, tokenizer, test_data: list[tuple[str, int]]
) -> tuple[float, float, list, list[float]]:
    """Returns (auc, accuracy, confusion_matrix_list, probs)."""
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    device = torch.device("cpu")
    test_ds = StyleDataset(test_data, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model.eval()
    all_probs: list[float] = []
    all_true: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].tolist()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)[:, 1].tolist()
            preds = [1 if p >= 0.5 else 0 for p in probs]
            all_probs.extend(probs)
            all_true.extend(labels)
            all_preds.extend(preds)

    auc = roc_auc_score(all_true, all_probs)
    acc = accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds).tolist()
    return auc, acc, cm, all_probs


def val_auc_from_probs(val_probs: list[float], val_true: list[int]) -> float:
    from sklearn.metrics import roc_auc_score  # type: ignore
    return roc_auc_score(val_true, val_probs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="cpt_corpus.v2.jsonl")
    parser.add_argument("--out-dir", default="runs/")
    parser.add_argument(
        "--baseline",
        default=".planning/calibration/style_classifier_baseline.json",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    corpus_path = repo_root / args.corpus
    out_base = repo_root / args.out_dir
    baseline_path = repo_root / args.baseline

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_base / f"style_classifier_{timestamp}"

    positives = load_positive(str(corpus_path))
    negatives, neg_source = load_negative()

    print(f"[+] neg_source={neg_source}  neg_count={len(negatives)}", flush=True)

    if not positives:
        print("[ERROR] No positive samples found — check corpus path and keywords.")
        return 1

    train_data, val_data, test_data = build_splits(positives, negatives)
    print(
        f"[+] Split: train={len(train_data)} val={len(val_data)} test={len(test_data)}",
        flush=True,
    )

    model, tokenizer, val_probs, val_true = train(train_data, val_data, run_dir)
    val_auc = val_auc_from_probs(val_probs, val_true)
    print(f"[+] val_AUC={val_auc:.4f}", flush=True)

    test_auc, test_acc, cm, _ = evaluate_metrics(model, tokenizer, test_data)
    print(f"[+] test_AUC={test_auc:.4f}  test_acc={test_acc:.4f}", flush=True)

    baseline = {
        "train_size": len(train_data),
        "val_auc": round(val_auc, 6),
        "test_auc": round(test_auc, 6),
        "test_accuracy": round(test_acc, 6),
        "confusion_matrix": cm,
        "base_model": BASE_MODEL,
        "hyperparams": {
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "max_per_bucket": MAX_PER_BUCKET,
        },
        "seed": SEED,
        "timestamp": timestamp,
        "neg_source": neg_source,
        "model_path": str(run_dir),
    }
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] Baseline written to {baseline_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
