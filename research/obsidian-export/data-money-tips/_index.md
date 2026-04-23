# 적금/재테크/세금 — 샘플 인덱스

**클러스터**: `money-tips`
**총 샘플**: 55개 (바이럴 5개)

| 길이 | 개수 |
|------|------|
| short (≤80자) | 22 |
| medium (81~200자) | 22 |
| long (>200자) | 11 |

## 관련 슬랭
`언니들`, `지금`, `요즘`, `원래`, `무조건`, `있나요`

```dataview
TABLE views AS "조회", comment_count AS "댓글", length_bucket AS "길이"
FROM "."
WHERE topic_cluster = "money-tips"
SORT views DESC
```
