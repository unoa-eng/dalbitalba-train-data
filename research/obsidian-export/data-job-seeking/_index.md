# 구직/이직 — 샘플 인덱스

**클러스터**: `job-seeking`
**총 샘플**: 55개 (바이럴 5개)

| 길이 | 개수 |
|------|------|
| short (≤80자) | 22 |
| medium (81~200자) | 22 |
| long (>200자) | 11 |

## 관련 슬랭
`언니들`, `요즘`, `가게`, `출근`, `있나요`, `어때요`

```dataview
TABLE views AS "조회", comment_count AS "댓글", length_bucket AS "길이"
FROM "."
WHERE topic_cluster = "job-seeking"
SORT views DESC
```
