#!/usr/bin/env python3
"""
Stage L distinguishability scorer.
Reads L_persona_matrix.txt; extracts persona-keyed OUTPUT blocks;
computes pairwise BLEU-style char-n-gram overlap & token-Jaccard
between every persona pair (averaged across the 3 prompts).
Lower overlap = more distinct.
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "L_persona_matrix.txt"
OUT = ROOT / "L_distinguishability.json"

if not SRC.exists():
    print(f"[fatal] {SRC} not found", file=sys.stderr)
    sys.exit(2)

text = SRC.read_text(encoding="utf-8", errors="replace")

# Split on "===== persona=NAME prompt=..."
blocks = re.split(r"^=====\s*persona=", text, flags=re.M)
generations = []  # list of (persona, prompt, output_text)
for blk in blocks[1:]:
    # First line: NAME prompt=PROMPTPREFIX... =====
    m = re.match(r"([^\s]+)\s+prompt=(.+?)\s*=====", blk)
    if not m:
        continue
    pname = m.group(1)
    pprompt = m.group(2)[:30]
    rest = blk[m.end():]
    # OUTPUT: ... up to next blank line / next block boundary
    om = re.search(r"^OUTPUT:\s*\n(.*?)(?=\n\n|\Z)", rest, flags=re.M | re.S)
    if not om:
        continue
    output = om.group(1).strip()
    # Strip mlx_lm.generate header lines if any leaked through
    output_lines = [
        ln for ln in output.splitlines()
        if not ln.startswith("==========")
        and not ln.startswith("Prompt:")
        and not ln.startswith("Generation:")
        and not ln.startswith("Peak memory")
        and not ln.startswith("Tokens per second")
        and not ln.startswith("Generation took")
        and not ln.startswith("Loading")
        and not ln.startswith("Fetching")
    ]
    output_clean = "\n".join(output_lines).strip()
    generations.append({"persona": pname, "prompt_prefix": pprompt, "output": output_clean})

# Group by persona
by_persona = {}
for g in generations:
    by_persona.setdefault(g["persona"], []).append(g["output"])


def char_ngrams(s, n=3):
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))


def cosine_overlap(c1, c2):
    if not c1 or not c2:
        return 0.0
    dot = sum(c1[k] * c2.get(k, 0) for k in c1)
    n1 = sum(v * v for v in c1.values()) ** 0.5
    n2 = sum(v * v for v in c2.values()) ** 0.5
    return dot / (n1 * n2) if n1 and n2 else 0.0


def jaccard_chars(s1, s2):
    a, b = set(s1), set(s2)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


personas = list(by_persona.keys())
pair_scores = []
for i, p1 in enumerate(personas):
    for p2 in personas[i+1:]:
        # Concatenate all 3 outputs of each persona
        s1 = "  ".join(by_persona[p1])
        s2 = "  ".join(by_persona[p2])
        if not s1 or not s2:
            pair_scores.append({"a": p1, "b": p2, "char3_cos": None, "char_jaccard": None, "skip": True})
            continue
        c1 = char_ngrams(s1, 3)
        c2 = char_ngrams(s2, 3)
        cos = cosine_overlap(c1, c2)
        jac = jaccard_chars(s1, s2)
        pair_scores.append({
            "a": p1,
            "b": p2,
            "char3_cosine_overlap": round(cos, 4),
            "char_set_jaccard": round(jac, 4),
            "len_a": len(s1),
            "len_b": len(s2),
        })

cos_vals = [p["char3_cosine_overlap"] for p in pair_scores if p.get("char3_cosine_overlap") is not None]
jac_vals = [p["char_set_jaccard"] for p in pair_scores if p.get("char_set_jaccard") is not None]
mean_cos = round(sum(cos_vals) / len(cos_vals), 4) if cos_vals else None
mean_jac = round(sum(jac_vals) / len(jac_vals), 4) if jac_vals else None

# Verdict: lower = more distinct. Heuristic: >0.7 cosine = collapsed; <0.4 = clearly distinct
if mean_cos is None:
    verdict = "INSUFFICIENT_DATA"
elif mean_cos < 0.4:
    verdict = "DISTINCT"
elif mean_cos < 0.7:
    verdict = "PARTIAL_DISTINCT"
else:
    verdict = "COLLAPSED"

result = {
    "stage": "L",
    "purpose": "Persona generation distinguishability — char-3gram cosine + char-set Jaccard between persona pairs",
    "n_personas": len(personas),
    "personas": personas,
    "n_generations_per_persona": {p: len(by_persona[p]) for p in personas},
    "n_pair_comparisons": len(pair_scores),
    "pairwise": pair_scores,
    "mean_char3_cosine_overlap": mean_cos,
    "mean_char_set_jaccard": mean_jac,
    "verdict": verdict,
    "interpretation": "Lower overlap = more distinct persona generation. Cosine >0.7 suggests persona conditioning failed (collapsed onto a generic style). 0.4-0.7 = partial differentiation. <0.4 = clear stylistic separation.",
    "caveat": "0.6B + 500-iter smoke adapter is a low-capacity persona vector; this is not a full persona-30 evaluation. Production persona evaluation runs on RunPod 8B with full Phase-3 SFT + ORPO."
}

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(f"Wrote {OUT}")
print(json.dumps({"mean_char3_cosine_overlap": mean_cos, "mean_char_set_jaccard": mean_jac, "verdict": verdict}, ensure_ascii=False))
