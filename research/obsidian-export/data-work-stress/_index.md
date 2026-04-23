# 손님/매출/컨디션 푸념 — 샘플 인덱스

**클러스터**: `work-stress`
**총 샘플**: 56개 (바이럴 6개)

| 길이 | 개수 |
|------|------|
| short (≤80자) | 23 |
| medium (81~200자) | 22 |
| long (>200자) | 11 |

## 관련 슬랭
`손님`, `손님이`, `존나`, `ㅈㄴ`, `ㅠㅠ`, `맨날`, `그냥`, `ㅅㅂ`, `없고`

```dataview
TABLE views AS "조회", comment_count AS "댓글", length_bucket AS "길이"
FROM "."
WHERE topic_cluster = "work-stress"
SORT views DESC
```
