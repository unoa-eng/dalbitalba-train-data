#!/usr/bin/env python3
"""Stage I: Full-corpus token-fire audit.

Counts how many times each of the 240 added vocab tokens fires across the full
training corpora. Identifies dead tokens (zero firings).
"""
import json, sys, os, re, time
from collections import Counter, defaultdict

ROOT = "/Users/unoa/dalbitalba-train-data"
TOK_PATH = f"{ROOT}/tokenizer_v4/added_tokens_v4.json"
OUT = f"{ROOT}/runs/local-integrity-2026-05-08/I_token_fire_full.json"

CORPORA = [
    ("cpt_corpus.v3.jsonl", f"{ROOT}/cpt_corpus.v3.jsonl", ["text", "content", "completion"]),
    ("sft_pairs.v3.jsonl", f"{ROOT}/sft_pairs.v3.jsonl", ["prompt", "completion", "messages"]),
    ("sft_thread_conditioned.jsonl", f"{ROOT}/sft_thread_conditioned.jsonl", ["prompt", "completion", "messages"]),
    ("val_set.v3.jsonl", f"{ROOT}/val_set.v3.jsonl", ["text", "prompt", "completion", "messages"]),
]


def extract_text(row, fields):
    """Extract a single text blob from a row spanning many possible schemas."""
    parts = []
    for f in fields:
        v = row.get(f)
        if v is None:
            continue
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            # messages: list of {role, content}
            for m in v:
                if isinstance(m, dict) and "content" in m:
                    parts.append(str(m["content"]))
                elif isinstance(m, str):
                    parts.append(m)
        elif isinstance(v, dict):
            parts.append(json.dumps(v, ensure_ascii=False))
    # fallback: dump whole row if nothing found
    if not parts:
        parts.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(parts)


def main():
    with open(TOK_PATH) as f:
        meta = json.load(f)
    special = meta["added_special_tokens"]
    domain = meta["added_domain_tokens"]
    force_core = meta["force_include_core"]
    force_slang = meta["force_include_slang_jamo"]
    all_tokens = special + domain
    print(f"[INFO] {len(special)} special + {len(domain)} domain = {len(all_tokens)} total added tokens", flush=True)
    print(f"[INFO] {len(force_core)} force-include core, {len(force_slang)} force-include slang_jamo", flush=True)

    # Per-corpus firing counts
    per_corpus = {}
    global_fires = Counter()
    rows_containing = Counter()  # token -> #rows that contain it (any corpus)

    for name, path, fields in CORPORA:
        if not os.path.exists(path):
            print(f"[WARN] missing: {path}", flush=True)
            continue
        rows = 0
        local_fires = Counter()
        local_row_with = Counter()
        t0 = time.time()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                rows += 1
                blob = extract_text(row, fields)
                seen_in_row = set()
                for tok in all_tokens:
                    c = blob.count(tok)
                    if c:
                        local_fires[tok] += c
                        seen_in_row.add(tok)
                for tok in seen_in_row:
                    local_row_with[tok] += 1
                if rows % 5000 == 0:
                    print(f"[{name}] processed {rows} rows ({time.time()-t0:.1f}s)", flush=True)
        per_corpus[name] = {
            "rows": rows,
            "elapsed_sec": round(time.time() - t0, 2),
            "total_fire_events": int(sum(local_fires.values())),
            "tokens_fired_at_least_once": int(sum(1 for t in all_tokens if local_fires.get(t, 0) > 0)),
            "fire_rate": round(sum(1 for t in all_tokens if local_fires.get(t, 0) > 0) / len(all_tokens), 4),
        }
        for tok, c in local_fires.items():
            global_fires[tok] += c
        for tok, c in local_row_with.items():
            rows_containing[tok] += c
        print(f"[{name}] DONE rows={rows} unique-fired={per_corpus[name]['tokens_fired_at_least_once']}/{len(all_tokens)} total_fires={per_corpus[name]['total_fire_events']}", flush=True)

    fired = [t for t in all_tokens if global_fires.get(t, 0) > 0]
    dead = [t for t in all_tokens if global_fires.get(t, 0) == 0]

    # Force-include tokens that died
    dead_force_core = [t for t in force_core if global_fires.get(t, 0) == 0]
    dead_force_slang = [t for t in force_slang if global_fires.get(t, 0) == 0]

    # Top firing tokens
    top_fired = sorted(((t, global_fires[t]) for t in all_tokens if global_fires[t] > 0),
                       key=lambda x: -x[1])[:30]
    bottom_fired = sorted(((t, global_fires[t]) for t in all_tokens if global_fires[t] > 0),
                          key=lambda x: x[1])[:30]

    out = {
        "total_tokens_added": len(all_tokens),
        "n_special": len(special),
        "n_domain": len(domain),
        "fired_at_least_once": len(fired),
        "dead_token_count": len(dead),
        "fire_rate": round(len(fired) / len(all_tokens), 4),
        "fire_rate_threshold_passed": len(fired) / len(all_tokens) >= 0.80,
        "dead_tokens": dead,
        "dead_force_include_core": dead_force_core,
        "dead_force_include_slang_jamo": dead_force_slang,
        "critical_force_include_dead": bool(dead_force_core) or bool(dead_force_slang),
        "top_fired": [{"token": t, "count": c} for t, c in top_fired],
        "bottom_fired": [{"token": t, "count": c} for t, c in bottom_fired],
        "per_corpus": per_corpus,
        "rows_containing_top": [
            {"token": t, "rows": rows_containing[t]}
            for t, _ in sorted(rows_containing.items(), key=lambda x: -x[1])[:20]
        ],
    }

    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[DONE] wrote {OUT}", flush=True)
    print(f"  fire_rate={out['fire_rate']} fired={out['fired_at_least_once']}/{out['total_tokens_added']}", flush=True)
    print(f"  dead_force_core={len(dead_force_core)} dead_force_slang={len(dead_force_slang)}", flush=True)
    if out["critical_force_include_dead"]:
        print(f"  CRITICAL: {dead_force_core + dead_force_slang}", flush=True)


if __name__ == "__main__":
    main()
