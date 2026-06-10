"""Convert Phase 1 v1+v3 seeds into the prompt-pool JSONL format that
scripts/cycle9_generation_audit.py expects under --suite valid-prefix.

Each output row:
  {prompt, completion (target), root_id, persona_id, source}

The audit script will append "\\n" to prompt and run cycle9 ORPO generation
with proper sampling (rep_penalty, anonymous_marker_bias, stop_repeat,
min_new_tokens, ban_control_tokens — all things our previous standalone smoke
forgot).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path("/Users/unoa/dalbitalba-train-data")
P1_DIR = ROOT / "runs/cycle10-phase1-claude-direct"
OUT = P1_DIR / "phase2_prompt_pool.jsonl"

STYLES = ["존댓말", "혼용", "반말"]
STYLE_W = [0.45, 0.35, 0.20]
MOODS = [
    "seeking_empathy", "anxious", "industry_info", "anger",
    "resignation", "positive", "melancholy", "self_doubt", "neutral",
]

def topic_for(text: str, rng: random.Random) -> str:
    if any(k in text for k in ["매출", "팁", "티오", "100", "200", "돈"]):
        return "money"
    if any(k in text for k in ["ᄉᄇ", "ᄆᄎ", "ᄌᄅ", "진상", "개진상"]):
        return "anger_irritation"
    if any(k in text for k in ["우울", "ᅮ", "비번", "본가", "혼자"]):
        return "melancholy_fatigue"
    if any(k in text for k in ["다들", "어떻게", "조언", "비슷"]):
        return "empathy_seeking"
    if any(k in text for k in ["하퍼", "쩜오", "텐카", "텐프로", "헤메"]):
        return "industry_jargon"
    return rng.choice(["empathy_seeking", "industry_jargon", "money", "self_esteem", "positive"])


def build_prompt(title: str, body: str, persona_id: str, style: str, mood: str) -> str:
    title = title.strip()[:80]
    body = body.strip()[:300]
    return (
        f"[POST-TITLE] {title}\n"
        f"[POST-BODY] {body}\n"
        f"[CONTEXT]\n"
        f"(no parent)\n"
        f"[REPLY-DEPTH=1]\n"
        f"[PERSONA: {persona_id} | {style} | {mood}]\n"
        f"[REPLY]"
    )


def main():
    seeds = []
    seen = set()
    for fname in ["threads_v1.jsonl", "threads_v3.jsonl"]:
        p = P1_DIR / fname
        for line in p.open():
            t = json.loads(line)
            key = (t.get("title", ""), t.get("content", ""))
            if key in seen:
                continue
            seen.add(key)
            seeds.append(t)
    print(f"[load] {len(seeds)} unique seeds", file=sys.stderr)

    rng = random.Random(2026)
    rng.shuffle(seeds)

    target = 63274
    rows = []
    persona_counter = 0
    while len(rows) < target:
        seed = seeds[len(rows) % len(seeds)]
        persona_counter += 1
        style = rng.choices(STYLES, weights=STYLE_W)[0]
        mood = rng.choice(MOODS)
        persona_id = f"p-{persona_counter:03d}"
        prompt = build_prompt(seed.get("title", ""), seed.get("content", ""), persona_id, style, mood)
        target_text = ""
        comments = seed.get("comments") or []
        if comments:
            target_text = comments[0].get("content", "")[:200]
        rows.append({
            "prompt": prompt,
            "completion": target_text or "[1] 비슷",
            "root_id": str(seed.get("id", "")),
            "persona_id": persona_id,
            "style": style,
            "mood": mood,
            "topic": topic_for(prompt, rng),
        })

    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(rows)} prompts to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
