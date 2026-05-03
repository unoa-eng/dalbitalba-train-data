import json
d = json.load(open("/tmp/metrics_n128.json"))
print("=== METRICS (N=123 valid / 128 sampled, raw N=412) ===")
m = d["metrics"]
for k, v in m.items():
    print(f"  {k}: {v}")
print()
print("=== domain_keyword per-term (gen vs raw) ===")
per = d["details"]["domain_keyword_alignment"]["per_term"]
for term, info in per.items():
    g = info["generated_ratio"]
    r = info["raw_ratio"]
    ratio = info["ratio"]
    tag = "PASS" if info["aligned"] else "fail"
    absent = " (source_absent)" if info.get("source_absent") else ""
    print(f"  [{tag}] {term}: gen={g:.4f} raw={r:.4f} ratio={ratio:.3f}{absent}")
print()
print("=== thresholds ===")
for k, v in d["thresholds"].items():
    op = v["op"]; val = v["value"]
    print(f"  {k}: {op} {val}")
