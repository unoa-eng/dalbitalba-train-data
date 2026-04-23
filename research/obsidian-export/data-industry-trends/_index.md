# 요즘 경기/시세 — 샘플 인덱스

**클러스터**: `industry-trends`
**총 샘플**: 56개 (바이럴 6개)

| 길이 | 개수 |
|------|------|
| short (≤80자) | 23 |
| medium (81~200자) | 22 |
| long (>200자) | 11 |

## 관련 슬랭
`요즘`, `언니들`, `텐카`, `하퍼`, `초이스`, `다들`, `원래`, `ㅈㄱㄴ`

```dataview
TABLE views AS "조회", comment_count AS "댓글", length_bucket AS "길이"
FROM "."
WHERE topic_cluster = "industry-trends"
SORT views DESC
```
