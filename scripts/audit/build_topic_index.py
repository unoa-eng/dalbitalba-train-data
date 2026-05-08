"""Stage 0 — Heuristic topic labeling for cpt_corpus.

Assigns one or more topic labels per doc using keyword matching.
Limit: this is heuristic; consider a learned classifier for better quality.
"""
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "cpt_corpus.v3.jsonl"
OUT = REPO / "runs/audit/topic_labels.jsonl"
SUMMARY = REPO / "runs/audit/topic_summary.json"

TOPIC_PATTERNS = {
    "출근": re.compile(r"(출근|첫날|첫 출근|언니 처음|새내기)"),
    "팁": re.compile(r"(팁|tip|TC|티씨|초이스|쩜오|밀빵)"),
    "마감": re.compile(r"(마감|새벽|퇴근|끝나|마지막)"),
    "단골": re.compile(r"(단골|손님|고정|진상|블랙)"),
    "조회": re.compile(r"(조회|선택|호명|콜|불려)"),
    "이직": re.compile(r"(이직|옮기|그만|관둘|새 가게|다른 가게)"),
    "케어": re.compile(r"(케어|관리|매니지|싸인|콜링|애프터|와리)"),
    "돈": re.compile(r"(돈|월수|소득|수입|벌이|매출|일당|페이|입금)"),
    "가족": re.compile(r"(엄마|아빠|부모|아들|딸|남편|와이프|자식)"),
    "연애": re.compile(r"(남자친구|여자친구|연애|썸|결혼|이혼|짝사랑)"),
    "심리": re.compile(r"(우울|힘들|불안|공황|상담|치료|병원)"),
    "건강": re.compile(r"(피부|다이어트|살|수술|시술|병원|약)"),
    "의상": re.compile(r"(드레스|원피스|구두|메이크|화장|머리)"),
    "후기": re.compile(r"(후기|썰|썰푼|이야기|경험|에피소드)"),
    "조언": re.compile(r"(조언|도움|추천|안되|망|위험|주의)"),
    "유머": re.compile(r"(웃|개웃|레알|진짜|미치|개꿀|꿀)"),
    "갈등": re.compile(r"(싸움|시비|차단|손절|배신|뒤통수)"),
    "가게_종류": re.compile(r"(텐카|쩜오|풀살롱|싸롱|텐|호빠|호스트|룸)"),
    "장소": re.compile(r"(강남|역삼|선릉|논현|서울|부산|대구|광주)"),
    "기타": re.compile(r".*"),  # fallback
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    n = 0
    with OUT.open("w") as out_f:
        for ln in SRC.open():
            d = json.loads(ln)
            n += 1
            text = d.get("text", "")
            labels = []
            for topic, pat in TOPIC_PATTERNS.items():
                if topic == "기타":
                    continue
                if pat.search(text):
                    labels.append(topic)
                    summary[topic] += 1
            if not labels:
                labels = ["기타"]
                summary["기타"] += 1
            out_f.write(json.dumps({
                "source_id": d.get("source_id"),
                "kind": d.get("kind"),
                "topics": labels,
            }, ensure_ascii=False) + "\n")
    sm = {
        "total_docs": n,
        "topic_doc_count": dict(summary),
        "topic_doc_ratio": {k: v/n for k, v in summary.items()},
    }
    SUMMARY.write_text(json.dumps(sm, ensure_ascii=False, indent=2))
    print(f"[topic_index] {n} docs, {len(summary)} topics labeled")
    for t, c in summary.most_common(10):
        print(f"  {t}: {c} ({c/n:.1%})")


if __name__ == "__main__":
    main()
