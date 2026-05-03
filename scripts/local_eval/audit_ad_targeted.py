import json, re
from pathlib import Path

# 호빠 도메인과 명확히 구분되는 학원/이질 광고 마커
HARD_AD = re.compile(r"(전원합격|내신\s*\+?\s*모의고사|학년도|특목고|영재학교|수능\s*\d|등록금|모집공고|학원|입시\s*상담|동훈팀)")
# 약관/공지 공식어 (호빠 톤이 아닌 것)
FORMAL = re.compile(r"(이용약관|개인정보\s*처리|운영진\s*안내|가입\s*절차|회원가입\s*시)")
# 대량 판매(10만원 이상 정가 광고)
BULK_SALE = re.compile(r"\d{2,3}만원[\s,~]\d{2,3}만원|VAT\s*별도|카드\s*결제|현금가|최저가\s*판매")

files = [
    ("val_set", "/Users/unoa/projects/dalbitalba-train-data/val_set.v3.jsonl"),
    ("cpt_corpus", "/Users/unoa/projects/dalbitalba-train-data/cpt_corpus.v3.jsonl"),
    ("cpt_structured", "/Users/unoa/projects/dalbitalba-train-data/v3-data/cpt_structured_v3.jsonl"),
    ("sft_5task", "/Users/unoa/projects/dalbitalba-train-data/v3-data/sft_5task_v3.jsonl"),
    ("sft_thread", "/Users/unoa/projects/dalbitalba-train-data/sft_thread_conditioned.jsonl"),
    ("ai_generated", "/tmp/ai_gen.jsonl"),
]

print(f"{'name':18s} {'lines':>8s} {'학원광고':>10s} {'공식어':>10s} {'대량판매':>10s}")
print("-" * 65)

samples = {}
for name, path in files:
    p = Path(path)
    if not p.exists(): continue
    n = h = f = b = 0
    samples[name] = []
    with p.open() as fh:
        for ln in fh:
            n += 1
            if n > 50000 and "ai" not in name and "val" not in name: break
            try:
                d = json.loads(ln)
                t = d.get("text", "") or json.dumps(d, ensure_ascii=False)
            except:
                t = ln
            mh = HARD_AD.search(t)
            mf = FORMAL.search(t)
            mb = BULK_SALE.search(t)
            if mh: 
                h += 1
                if len(samples[name]) < 2:
                    samples[name].append(("학원광고", t[max(0,mh.start()-30):mh.end()+80].replace("\n", " ")))
            if mf: f += 1
            if mb: b += 1
    audited = min(n, 50000) if "ai" not in name and "val" not in name else n
    pct_h = 100*h/audited if audited else 0
    pct_f = 100*f/audited if audited else 0
    pct_b = 100*b/audited if audited else 0
    print(f"{name:18s} {audited:>8d} {h:>5d}({pct_h:4.2f}%) {f:>5d}({pct_f:4.2f}%) {b:>5d}({pct_b:4.2f}%)")

print("\n=== 학원광고 실제 매칭 예시 ===")
for name, exs in samples.items():
    if exs:
        print(f"\n[{name}]")
        for tag, snippet in exs[:2]:
            print(f"  ...{snippet[:160]}...")
