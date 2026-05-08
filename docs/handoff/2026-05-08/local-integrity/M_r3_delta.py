#!/usr/bin/env python3
"""Stage M: Quantify the delta from the R3 fix stack.

Two counter-factuals on real data:
  M.2a: train_sft.py multiplicity logic — pre-R3 used `int(round(lw))` which
        collapses 1.5 -> 2 AND 2.0 -> 2 (banker's rounding bug). Post-R3
        uses floor+Bernoulli(frac), preserving E[mult(w)] = w.
  M.2b: Show the expected expansion delta on real sft_pairs.v3.jsonl +
        sft_thread_conditioned.jsonl loss_weight distributions.

(M.1 — re-running round2_build_tc_sft.py — is blocked locally because the
required cpt_context_stream + source_db_cache live on the RunPod side. We
document the blocker and quantify M.2 only.)
"""
import json, math, random, statistics

ROOT = "/Users/unoa/dalbitalba-train-data"
OUT = f"{ROOT}/runs/local-integrity-2026-05-08/M_r3_delta.json"
SOURCES = [
    f"{ROOT}/sft_pairs.v3.jsonl",
    f"{ROOT}/sft_thread_conditioned.jsonl",
]


def pre_r3_multiplicity(lw):
    """Old: max(1, int(round(lw))) — Python banker's rounding."""
    return max(1, int(round(lw)))


def post_r3_multiplicity(lw, rng):
    """New: floor(lw) + Bernoulli(frac). E[mult] = lw."""
    lw = max(1.0, min(float(lw), 10.0))
    floor = max(1, int(math.floor(lw)))
    frac = lw - math.floor(lw)
    return floor + (1 if rng.random() < frac else 0)


def expand(rows, multiplicity_fn, rng=None):
    expanded = 0
    for r in rows:
        lw = r.get("loss_weight", 1.0)
        try:
            lw = float(lw)
        except Exception:
            lw = 1.0
        if rng is None:
            expanded += multiplicity_fn(lw)
        else:
            expanded += multiplicity_fn(lw, rng)
    return expanded


def analyze(path, name):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    n = len(rows)

    # loss_weight distribution
    lw_dist = {}
    for r in rows:
        lw = round(float(r.get("loss_weight", 1.0)), 3)
        lw_dist[lw] = lw_dist.get(lw, 0) + 1

    # Pre-R3 expansion: deterministic
    pre_r3 = expand(rows, pre_r3_multiplicity)

    # Post-R3 expansion: stochastic; average over 50 seeds for stable estimate
    post_r3_runs = []
    for seed in range(50):
        rng = random.Random(seed)
        post_r3_runs.append(expand(rows, post_r3_multiplicity, rng))
    post_r3_mean = statistics.mean(post_r3_runs)
    post_r3_std = statistics.stdev(post_r3_runs)

    # Theoretical post-R3 expectation: sum of lw values
    theoretical = sum(min(max(1.0, float(r.get("loss_weight", 1.0))), 10.0) for r in rows)

    return {
        "name": name,
        "rows_in": n,
        "loss_weight_distribution": lw_dist,
        "pre_r3_expanded": pre_r3,
        "pre_r3_multiplier": round(pre_r3 / n, 4),
        "post_r3_expanded_mean": round(post_r3_mean, 1),
        "post_r3_expanded_std": round(post_r3_std, 2),
        "post_r3_multiplier": round(post_r3_mean / n, 4),
        "theoretical_E_expanded": round(theoretical, 1),
        "delta_post_minus_pre": round(post_r3_mean - pre_r3, 1),
        "delta_pct": round(100.0 * (post_r3_mean - pre_r3) / pre_r3, 2),
        "verdict_pre_r3_collapse": (
            "Pre-R3 banker's rounding produces SAME multiplicity for lw=1.5 (-> 2) and lw=2.0 (-> 2). "
            f"Pre-R3 expanded {n}->{pre_r3} (multiplier {pre_r3 / n:.3f}). "
            f"Post-R3 expanded {n}->{post_r3_mean:.0f} (multiplier {post_r3_mean / n:.3f}). "
            f"Net effect of R3 fix: {post_r3_mean - pre_r3:+.0f} rows ({100.0 * (post_r3_mean - pre_r3) / pre_r3:+.2f}%)."
        ),
    }


def main():
    sources = []
    for p in SOURCES:
        try:
            sources.append(analyze(p, p.rsplit("/", 1)[-1]))
        except Exception as e:
            sources.append({"name": p, "error": str(e)})

    # Demonstrate the bug on a synthetic 50/50 lw=1.5 vs lw=2.0 set
    bug_demo = []
    rng = random.Random(0)
    n = 1000
    synthetic = [{"loss_weight": 1.5} for _ in range(n // 2)] + [{"loss_weight": 2.0} for _ in range(n // 2)]
    pre = expand(synthetic, pre_r3_multiplicity)
    post_runs = []
    for s in range(50):
        r = random.Random(s)
        post_runs.append(expand(synthetic, post_r3_multiplicity, r))
    post_mean = statistics.mean(post_runs)
    bug_demo = {
        "synthetic_50pct_lw1.5_50pct_lw2.0": {
            "n_input": n,
            "pre_r3_expanded": pre,
            "pre_r3_multiplier": pre / n,
            "post_r3_expanded_mean": round(post_mean, 1),
            "post_r3_multiplier": round(post_mean / n, 4),
            "theoretical_E": (1.5 + 2.0) / 2 * n,
            "key_finding": (
                "Pre-R3 collapses both lw=1.5 and lw=2.0 to multiplicity=2 — "
                "the SFT_LOSS_WEIGHT_ARGOT escalation from 1.5 to 2.0 had ZERO "
                "effect under banker's rounding. Post-R3 correctly shows "
                "1.75x vs 2.00x expansion."
            ),
        }
    }

    note_m1 = (
        "M.1 (rebuild SFT data with vs without R3 env vars + dedup toggle) is "
        "blocked locally: round2_build_tc_sft.py requires cpt_context_stream.jsonl "
        "and source_db_cache/ which are not present in this clone. The R3 "
        "manifest-hash check in chain_train_round2.sh already enforces rebuild "
        "on env-var change in the production pipeline; local repro deferred to "
        "RunPod where these inputs live."
    )

    out = {
        "stage": "M",
        "purpose": "Quantify R3 fix delta vs pre-R3 banker's-rounding bug",
        "M1_blocked_locally": True,
        "M1_blocker": note_m1,
        "M2_per_corpus": sources,
        "M2_synthetic_bug_demo": bug_demo,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
