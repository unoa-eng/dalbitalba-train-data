#!/usr/bin/env python3
"""Re-generate the per-post requeue (gate failures) + structural bad batches via Codex, gate, and PATCH.

Reads requeue_posts.json (per-post failures) and requeue.json (bad batch names).
Collects those posts' SOURCE from batch_in, chunks (25/post), runs Codex with a STRICTER prompt,
re-gates each post, PATCHes the passers, and rewrites requeue_posts.json with the residual still-failing.
Idempotent; safe to re-run until residual is empty.
"""
import json, re, subprocess, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from load_batches import hard_hit, venue_hit, has_jamo, verbatim_fail, patch  # noqa

BASE = Path("/Users/unoa/dalbitalba-train-data/runs/cycle11-live-inplace-regen")
IN = BASE / "batch_in"; RQI = BASE / "requeue_in"; RQO = BASE / "requeue_out"
RQI.mkdir(exist_ok=True); RQO.mkdir(exist_ok=True)
RECIPE = (BASE / "scripts/recipe.md").read_text()
STRICT = RECIPE + """

[추가 강제 — 재생성 패스]
- 식별정보가 없어 보여도 **모든 title/body/comment의 표현을 반드시 바꿔라**(원문과 동일/거의동일 금지, 특히 짧은 제목도 단어를 갈아끼워라).
- comment의 `id`와 `parent_id`는 원본 그대로 **절대 변경/삭제/추가 금지**. 댓글 개수 정확히 유지.
- 업소·인물·브랜드(도파민/도파/세이렌/유앤미/보스턴/엘리트/퍼펙트/사라있네/달토/정점/썸데이/하퍼/루나/퀸 등) 흔적을 완전히 제거·일반화."""

def src_index():
    idx = {}
    for f in glob.glob(str(IN / "batch_*.json")):
        for p in json.load(open(f)): idx[p["id"]] = p
    return idx

def targets():
    ids = set()
    if (BASE / "requeue_posts.json").exists():
        for rp in json.loads((BASE / "requeue_posts.json").read_text()): ids.add(rp["id"])
    if (BASE / "requeue.json").exists():
        for n in json.loads((BASE / "requeue.json").read_text()):
            f = IN / f"{n}.json"
            if f.exists():
                for p in json.load(open(f)): ids.add(p["id"])
    return ids

def gate_post(s, r):
    e = []
    sc = {c["id"]: c for c in s["comments"]}; rc = {c["id"]: c for c in r.get("comments", [])}
    if len(sc) != len(rc): e.append("cmt count")
    blob = " ".join([s["title"], s["body"]] + [c["body"] for c in s["comments"]])
    fields = [("title", s["title"], r.get("title", "")), ("body", s["body"], r.get("body", ""))]
    for cid, c in sc.items():
        rcm = rc.get(cid)
        if not rcm: e.append("missing cmt"); continue
        if (c.get("parent_id") or None) != (rcm.get("parent_id") or None): e.append("parent")
        fields.append(("cmt", c["body"], rcm.get("body", "")))
    for nm, a, b in fields:
        if hard_hit(b) or venue_hit(b, blob) or has_jamo(b) or verbatim_fail(a, b): e.append(nm)
    return e

def codex_gen(chunk_path, out_path):
    src = chunk_path.read_text()
    prompt = STRICT + f"\n\n아래 입력 JSON 배열의 모든 post를 재작성하라. 출력은 유효한 JSON 배열 하나만(코드펜스/설명 금지).\n형식: [{{\"id\",\"title\",\"body\",\"comments\":[{{\"id\",\"parent_id\",\"body\"}}]}}]\n\n입력:\n{src}"
    raw = subprocess.run(["codex", "exec", "--skip-git-repo-check",
                          "--dangerously-bypass-approvals-and-sandbox", prompt],
                         capture_output=True, text=True, timeout=900).stdout
    raw = raw.replace("```json", "").replace("```", "")
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i: return None
    try:
        data = json.loads(raw[i:j+1]); out_path.write_text(json.dumps(data, ensure_ascii=False)); return data
    except Exception:
        return None

def main():
    idx = src_index(); ids = sorted(targets())
    print(f"requeue targets: {len(ids)} posts")
    if not ids: print("nothing to do"); return
    chunks = [ids[i:i+25] for i in range(0, len(ids), 25)]
    residual, patched = [], 0
    for ci, ch in enumerate(chunks):
        cin = RQI / f"rq_{ci:04d}.json"; cout = RQO / f"rq_{ci:04d}.json"
        cin.write_text(json.dumps([idx[i] for i in ch if i in idx], ensure_ascii=False))
        data = codex_gen(cin, cout)
        if not data: print(f"rq_{ci:04d}: codex/parse FAIL"); residual += [{"id": i, "reason": ["regen-fail"]} for i in ch]; continue
        rmap = {p["id"]: p for p in data}
        for pid in ch:
            s, r = idx.get(pid), rmap.get(pid)
            if not s or not r: residual.append({"id": pid, "reason": ["no-output"]}); continue
            e = gate_post(s, r)
            if e: residual.append({"id": pid, "reason": e[:3]}); continue
            patch("community_posts", pid, {"title": r["title"], "body": r["body"]}); patched += 1
            for c in r.get("comments", []): patch("community_comments", c["id"], {"body": c["body"]})
        print(f"rq_{ci:04d}: done ({ci+1}/{len(chunks)}) patched so far {patched}")
    (BASE / "requeue_posts.json").write_text(json.dumps(residual, ensure_ascii=False))
    (BASE / "requeue.json").write_text("[]")
    print(f"REQUEUE DONE: patched {patched} posts | residual still-failing {len(residual)}")

if __name__ == "__main__":
    main()
