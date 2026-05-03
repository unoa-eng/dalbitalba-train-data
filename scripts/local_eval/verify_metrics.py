import json, random

# L3
with open("/tmp/metrics_n128.json") as f:
    m = json.load(f)["metrics"]
print("=== L3: METRIC SANITY ===")
checks = [
    ("bigram_jsd in [0,1]", 0 <= m["bigram_jsd"] <= 1, m["bigram_jsd"]),
    ("length_kl >= 0", m["length_kl"] >= 0, m["length_kl"]),
    ("digit_density_delta >= 0", m["digit_density_delta"] >= 0, m["digit_density_delta"]),
    ("english_density_delta >= 0", m["english_density_delta"] >= 0, m["english_density_delta"]),
    ("domain_keyword_alignment in [0,1]", 0 <= m["domain_keyword_alignment"] <= 1, m["domain_keyword_alignment"]),
    ("tone_distribution_match in [0,1]", 0 <= m["tone_distribution_match"] <= 1, m["tone_distribution_match"]),
    ("korean_retention_ppl >= 1", m["korean_retention_ppl"] >= 1, m["korean_retention_ppl"]),
    ("n_ai > 100", m["n_ai"] > 100, m["n_ai"]),
    ("n_raw == 412", m["n_raw"] == 412, m["n_raw"]),
]
for name, ok, val in checks:
    tag = "OK" if ok else "FAIL"
    print(f"  [{tag}] {name}: {val}")

# L2
print()
print("=== L2: 5 RANDOM SAMPLES (HUMAN READS) ===")
random.seed(42)
lines = open("/tmp/ai_gen.jsonl").read().strip().split("\n")
picks = random.sample(range(len(lines)), 5)
for i, idx in enumerate(picks, 1):
    d = json.loads(lines[idx])
    kind = d.get("kind", "?")
    seed = d.get("seed", "")
    text = d.get("text", "")
    print(f"--- 샘플 {i} (line {idx+1}, kind={kind}) ---")
    print("[seed]")
    print(seed[:300])
    print("[generated]")
    print(text[:400] if text else "(EMPTY)")
    print()
