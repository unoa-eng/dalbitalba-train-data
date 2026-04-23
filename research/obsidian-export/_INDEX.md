# Corpus Sample Index

```dataview
TABLE
  topic_label AS "주제",
  length_bucket AS "길이",
  views AS "조회",
  comment_count AS "댓글",
  is_viral AS "viral"
FROM "."
WHERE topic_cluster
SORT views DESC
LIMIT 50
```
