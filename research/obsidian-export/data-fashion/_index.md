# 옷/명품 — 샘플 인덱스

**클러스터**: `fashion`
**총 샘플**: 56개 (바이럴 6개)

| 길이 | 개수 |
|------|------|
| short (≤80자) | 22 |
| medium (81~200자) | 22 |
| long (>200자) | 12 |

## 관련 슬랭
`언니들`, `많이`, `사이즈`, `요즘`, `그냥`, `도파민`

```dataview
TABLE views AS "조회", comment_count AS "댓글", length_bucket AS "길이"
FROM "."
WHERE topic_cluster = "fashion"
SORT views DESC
```
