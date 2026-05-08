#!/usr/bin/env python3
"""Stage G.2: ORPO chosen-vs-rejected statistical sanity check."""
import json, re, statistics, sys

ROOT = "/Users/unoa/dalbitalba-train-data"
PAIRS = f"{ROOT}/runs/local-integrity-2026-05-08/G_orpo_pairs.jsonl"
VAL = f"{ROOT}/val_set.v3.jsonl"
OUT = f"{ROOT}/runs/local-integrity-2026-05-08/G_validation.json"

# Domain keywords (from PROD use)
DOMAIN_KW = ["언니", "손님", "쩜오", "텐카", "호빠", "와리", "퍼블", "셔츠룸", "보도", "샵", "초보", "스폰", "탄단지"]
AI_MARKERS = [
    "도움이 되", "참고하시", "추천드", "안녕하세요", "감사합니다", "확인하시", "바랍니다",
    "다음과 같", "유의하시", "Disclaimer", "AI", "OpenAI", "ChatGPT",
]


def kw_density(text, kws):
    if not text:
        return 0.0
    n = sum(text.count(k) for k in kws)
    return n / max(len(text), 1) * 1000  # per 1000 chars


def has_marker(text, markers):
    return any(m in text for m in markers)


def main():
    pairs = []
    with open(PAIRS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))

    print(f"loaded {len(pairs)} pairs", flush=True)

    # Length distribution
    chosen_lens = [len(p["chosen"]) for p in pairs]
    rejected_lens = [len(p["rejected"]) for p in pairs]

    # Domain density
    chosen_dens = [kw_density(p["chosen"], DOMAIN_KW) for p in pairs]
    rejected_dens = [kw_density(p["rejected"], DOMAIN_KW) for p in pairs]

    # AI markers
    chosen_ai = sum(1 for p in pairs if has_marker(p["chosen"], AI_MARKERS))
    rejected_ai = sum(1 for p in pairs if has_marker(p["rejected"], AI_MARKERS))

    # Schema check
    has_prompt = sum(1 for p in pairs if p.get("prompt"))
    has_chosen = sum(1 for p in pairs if p.get("chosen"))
    has_rejected = sum(1 for p in pairs if p.get("rejected"))

    # Leak check vs val set
    val_completions = set()
    with open(VAL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            for k in ("text", "completion", "answer", "target_comment"):
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    val_completions.add(v.strip())
            msgs = row.get("messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") == "assistant":
                        c = m.get("content")
                        if isinstance(c, str) and c.strip():
                            val_completions.add(c.strip())

    rejected_leaks = sum(1 for p in pairs if p["rejected"].strip() in val_completions)
    chosen_leaks = sum(1 for p in pairs if p["chosen"].strip() in val_completions)

    # Identical chosen==rejected
    identical = sum(1 for p in pairs if p["chosen"].strip() == p["rejected"].strip())

    # Pair "kind" / source distribution
    kinds = {}
    for p in pairs:
        k = p.get("kind", "?")
        kinds[k] = kinds.get(k, 0) + 1

    out = {
        "n_pairs": len(pairs),
        "schema": {
            "has_prompt": has_prompt,
            "has_chosen": has_chosen,
            "has_rejected": has_rejected,
            "all_complete": (has_prompt == has_chosen == has_rejected == len(pairs)),
        },
        "chosen_len_chars": {
            "min": min(chosen_lens), "median": statistics.median(chosen_lens),
            "mean": round(statistics.mean(chosen_lens), 1), "max": max(chosen_lens),
        },
        "rejected_len_chars": {
            "min": min(rejected_lens), "median": statistics.median(rejected_lens),
            "mean": round(statistics.mean(rejected_lens), 1), "max": max(rejected_lens),
        },
        "domain_density_per_1k_chars": {
            "chosen_mean": round(statistics.mean(chosen_dens), 3),
            "rejected_mean": round(statistics.mean(rejected_dens), 3),
            "delta": round(statistics.mean(chosen_dens) - statistics.mean(rejected_dens), 3),
        },
        "ai_marker_presence": {
            "chosen_count": chosen_ai,
            "rejected_count": rejected_ai,
            "rejected_minus_chosen": rejected_ai - chosen_ai,
            "expectation": "rejected should contain MORE AI markers than chosen",
            "passes_expectation": rejected_ai > chosen_ai,
        },
        "val_leak_check": {
            "val_completions_indexed": len(val_completions),
            "chosen_leaks_into_val": chosen_leaks,
            "rejected_leaks_into_val": rejected_leaks,
            "expectation": "neither should leak; rejected MUST not match val verbatim",
            "passes": (chosen_leaks == 0 and rejected_leaks == 0),
        },
        "identical_pair_count": identical,
        "kind_distribution": kinds,
        "verdict": (
            "PASS" if (has_prompt == has_chosen == has_rejected == len(pairs)
                       and rejected_ai >= chosen_ai
                       and chosen_leaks == 0 and rejected_leaks == 0
                       and identical == 0)
            else "PARTIAL"
        ),
    }
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
