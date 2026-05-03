import json, re, sys
from pathlib import Path

# 광고 의심 패턴
AD_PATTERNS = {
    "학원": re.compile(r"(엘리트|동훈팀|성해|전원합격|학년도|내신|모의고사|수능|입시|특목고|영재학교|과외)"),
    "광고문구": re.compile(r"(\d+%\s*전원|등록\s*문의|상담\s*문의|예약\s*문의|이벤트\s*진행|할인\s*이벤트|문의\s*환영)"),
    "전화번호": re.compile(r"010[\s\-.]?\d{4}[\s\-.]?\d{4}"),
    "부동산": re.compile(r"(매매|월세|전세|분양|매물|평수|평형|입주|아파트|오피스텔|원룸)"),
    "약관/공지": re.compile(r"(이용약관|개인정보|운영진|운영자|공지사항|회원가입)"),
    "쇼핑/판매": re.compile(r"(원\s*\.|구매|결제|배송|택배|상품|할인|쿠폰|적립)"),
    "주식/금융": re.compile(r"(주식|코인|상승률|하락률|매수|매도|거래소|주가)"),
}

def audit(path: str, max_lines: int = None) -> dict:
    p = Path(path)
    if not p.exists():
        return None
    counts = {k: 0 for k in AD_PATTERNS}
    examples = {k: [] for k in AD_PATTERNS}
    total = 0
    with p.open() as f:
        for ln in f:
            if max_lines and total >= max_lines:
                break
            total += 1
            try:
                d = json.loads(ln)
                # text field varies: "text" or "messages"[0]["content"] or just whole
                text = d.get("text", "")
                if not text and isinstance(d.get("messages"), list):
                    text = " ".join(m.get("content", "") for m in d["messages"] if isinstance(m, dict))
                if not text:
                    text = json.dumps(d, ensure_ascii=False)
            except Exception:
                text = ln
            for k, pat in AD_PATTERNS.items():
                m = pat.search(text)
                if m:
                    counts[k] += 1
                    if len(examples[k]) < 2:
                        snippet = text[max(0, m.start()-30):min(len(text), m.end()+50)]
                        examples[k].append(snippet.replace("\n", " "))
    return {"path": path, "total": total, "counts": counts, "examples": examples}

files = [
    ("/Users/unoa/projects/dalbitalba-train-data/val_set.v3.jsonl", None),
    ("/Users/unoa/projects/dalbitalba-train-data/cpt_corpus.v3.jsonl", 50000),
    ("/Users/unoa/projects/dalbitalba-train-data/v3-data/cpt_structured_v3.jsonl", 50000),
    ("/Users/unoa/projects/dalbitalba-train-data/v3-data/sft_5task_v3.jsonl", 50000),
    ("/Users/unoa/projects/dalbitalba-train-data/sft_thread_conditioned.jsonl", 50000),
    ("/tmp/ai_gen.jsonl", None),
]

results = []
for path, lim in files:
    r = audit(path, lim)
    if r:
        results.append(r)
        print(f"\n=== {Path(path).name} (lines audited: {r['total']}) ===")
        for k, c in r["counts"].items():
            pct = 100.0 * c / r["total"] if r["total"] else 0
            print(f"  {k:14s}: {c:5d} ({pct:5.2f}%)")

print("\n=== 대표 예시 (val_set vs cpt_corpus) ===")
for r in results:
    if "val_set" in r["path"] or "cpt_corpus" in r["path"]:
        name = Path(r["path"]).name
        print(f"\n[{name}]")
        for k in ("학원", "쇼핑/판매", "부동산"):
            for ex in r["examples"][k][:1]:
                print(f"  ({k}) ...{ex[:140]}...")
