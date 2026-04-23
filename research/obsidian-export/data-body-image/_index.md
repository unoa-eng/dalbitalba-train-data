# 성형/시술/다이어트 — 샘플 인덱스

**클러스터**: `body-image`
**총 샘플**: 55개 (바이럴 5개)

| 길이 | 개수 |
|------|------|
| short (≤80자) | 22 |
| medium (81~200자) | 22 |
| long (>200자) | 11 |

## 관련 슬랭
`사이즈`, `얼굴`, `언니들`, `많이`, `요즘`, `무조건`

```dataview
TABLE views AS "조회", comment_count AS "댓글", length_bucket AS "길이"
FROM "."
WHERE topic_cluster = "body-image"
SORT views DESC
```
