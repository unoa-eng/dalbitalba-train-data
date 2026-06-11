#!/usr/bin/env python3
"""threads_v4 최종 텍스트 감사: 생성 댓글 vs 원천 댓글 AUC (winning recipe 방식).

HashingVectorizer char_wb(2,4) + LogisticRegression 5-fold CV, multi-seed.
원천: runs/cycle10/data-representative-v1-clean/threads_flat/train.jsonl
(타 사이트 모더레이션/차단 안내문 오염 row 필터 — src contamination 메모리 반영)
"""
import json
import random
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

REPO = Path("/Users/unoa/dalbitalba-train-data")
v4_path = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 800

INDEX_RE = re.compile(r"^\[\d+(?:-\d+)?\]\s*")
CONTAM_RE = re.compile(r"차단|블라인드|관리자에 의해|신고가 접수|이용이 제한|운영원칙")

gen = []
for line in v4_path.open():
    t = json.loads(line)
    for c in t.get("comments") or []:
        x = INDEX_RE.sub("", c["content"]).strip()
        if len(x) > 3:
            gen.append(x)

src = []
for line in (REPO / "runs/cycle10/data-representative-v1-clean/threads_flat/train.jsonl").open():
    try:
        d = json.loads(line)
    except Exception:
        continue
    x = (d.get("text") or "").strip()
    if len(x) > 3 and not CONTAM_RE.search(x):
        src.append(x)

print(f"gen pool {len(gen)} | src pool {len(src)} | sample N={N}")

def auc_eval(g, s):
    texts = g + s
    labels = np.array([1] * len(g) + [0] * len(s))
    vec = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2 ** 14, alternate_sign=False)
    X = vec.transform(texts)
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(clf, X, labels, cv=cv, scoring="roc_auc").mean()

results = []
for seed in [11, 27, 53, 101, 151, 173, 211]:
    random.seed(seed)
    g = random.sample(gen, min(N, len(gen)))
    s = random.sample(src, min(N, len(src)))
    a = auc_eval(g, s)
    results.append(a)
    print(f"  seed={seed:3d}: AUC={a:.3f}")

m = statistics.mean(results)
sd = statistics.stdev(results)
print(f"\n=== v4 comments vs source: AUC {m:.3f} ± {sd:.3f} (7 seeds, N={N}) ===")
print("기존 local-LLM ceiling: 0.579 | target: ≤0.55" )
print("PASS" if m <= 0.55 else ("MARGINAL(<=0.60)" if m <= 0.60 else "FAIL"))
