# 기타 — 샘플 인덱스

**클러스터**: `기타`
**총 샘플**: 57개 (바이럴 7개)

| 길이 | 개수 |
|------|------|
| short (≤80자) | 24 |
| medium (81~200자) | 22 |
| long (>200자) | 11 |

## 관련 슬랭


```dataview
TABLE views AS "조회", comment_count AS "댓글", length_bucket AS "길이"
FROM "."
WHERE topic_cluster = "기타"
SORT views DESC
```
