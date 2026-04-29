import json
import os
import random
import re
from collections import Counter
from pathlib import Path

def get_hangul_ratio(text):
    if not text: return 0
    hangul = re.findall(r'[가-힣]', text)
    return len(hangul) / len(text)

def get_stats(texts):
    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / len(lengths)
    
    words = []
    for t in texts:
        words.extend(re.findall(r'\w+', t))
    vocab = Counter(words).most_common(20)
    
    hangul_ratios = [get_hangul_ratio(t) for t in texts]
    avg_hangul = sum(hangul_ratios) / len(hangul_ratios)
    
    return {
        "avg_len": avg_len,
        "vocab": vocab,
        "avg_hangul": avg_hangul,
        "raw_lengths": sorted(lengths)
    }

source_dir = Path("/Users/unoa/Downloads/crawled-data-v2")
source_files = list(source_dir.glob("*.json"))
source_texts = []

print(f"Sampling from {len(source_files)} source files...")
# Take a few from each file until we get 100
random.shuffle(source_files)
for sf in source_files:
    if len(source_texts) >= 100: break
    try:
        with open(sf, 'r') as f:
            data = json.load(f)
            for item in data:
                content = item.get("content", "")
                if len(content) > 10:
                    source_texts.append(content)
                if len(source_texts) >= 100: break
    except Exception as e:
        continue

train_file = "cpt_corpus.v2.jsonl"
train_texts = []
with open(train_file, 'r') as f:
    lines = f.readlines()
    sampled_lines = random.sample(lines, 100)
    for sl in sampled_lines:
        train_texts.append(json.loads(sl)["text"])

source_stats = get_stats(source_texts)
train_stats = get_stats(train_texts)

print("\n--- STATISTICS COMPARISON ---")
print(f"{'Metric':<15} | {'Source':<15} | {'Training':<15}")
print("-" * 50)
print(f"{'Avg Length':<15} | {source_stats['avg_len']:<15.2f} | {train_stats['avg_len']:<15.2f}")
print(f"{'Avg Hangul %':<15} | {source_stats['avg_hangul']*100:<15.2f} | {train_stats['avg_hangul']*100:<15.2f}")

print("\n--- TOP VOCABULARY ---")
print(f"{'Source':<30} | {'Training':<30}")
print("-" * 65)
for i in range(10):
    s_v = f"{source_stats['vocab'][i][0]} ({source_stats['vocab'][i][1]})"
    t_v = f"{train_stats['vocab'][i][0]} ({train_stats['vocab'][i][1]})"
    print(f"{s_v:<30} | {t_v:<30}")

# Pattern identification
# Over-filtering: Source has patterns missing in training
# Contamination: Training has patterns missing in source (like ad markers or PII placeholders)

print("\n--- SAMPLE TEXTS (FIRST 3) ---")
print("SOURCE:")
for t in source_texts[:3]:
    print(f"- {t[:100]}...")
print("\nTRAINING:")
for t in train_texts[:3]:
    print(f"- {t[:100]}...")
