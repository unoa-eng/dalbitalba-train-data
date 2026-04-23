# 가게 후기 — 샘플 인덱스

**클러스터**: `venue-review`
**총 샘플**: 50개 (바이럴 3개)

| 길이 | 개수 |
|------|------|
| short (≤80자) | 22 |
| medium (81~200자) | 22 |
| long (>200자) | 6 |

## 관련 슬랭
`가게`, `언니들은`, `요즘`, `원래`, `다들`, `어때요`

```dataview
TABLE views AS "조회", comment_count AS "댓글", length_bucket AS "길이"
FROM "."
WHERE topic_cluster = "venue-review"
SORT views DESC
```
