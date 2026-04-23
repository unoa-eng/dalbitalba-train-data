#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_WEIGHTS = {
    "reply_comment": 0.30,
    "root_comment": 0.40,
    "post_from_title": 0.20,
    "post_continue": 0.10,
}

LENGTH_BUCKETS = [
    ("xs", 0, 40),
    ("sm", 40, 80),
    ("md", 80, 160),
    ("lg", 160, 10**9),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="thread-aware SFT 데이터셋을 비용 예산에 맞게 샘플링")
    parser.add_argument("--input", required=True, help="원본 SFT JSONL 경로")
    parser.add_argument("--output", required=True, help="샘플링된 SFT JSONL 경로")
    parser.add_argument("--summary", required=True, help="샘플링 요약 JSON 경로")
    parser.add_argument("--target-size", type=int, default=24000, help="목표 샘플 수")
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=4,
        help="동일 source_id 에서 허용할 최대 샘플 수",
    )
    return parser.parse_args()


def stable_rank(record: dict) -> str:
    basis = "|".join(
        [
            str(record.get("task_type") or ""),
            str(record.get("source_id") or ""),
            str(record.get("comment_key") or ""),
            str(record.get("parent_comment_key") or ""),
            str(record.get("output") or ""),
        ]
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def length_bucket(text: str) -> str:
    size = len((text or "").strip())
    for name, start, end in LENGTH_BUCKETS:
        if start <= size < end:
            return name
    return "lg"


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (
                record.get("task_type"),
                record.get("source_id"),
                record.get("comment_key"),
                record.get("output"),
            )
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def allocate_targets(counts: Counter[str], target_size: int) -> dict[str, int]:
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0

    for task_type, weight in DEFAULT_WEIGHTS.items():
        available = counts.get(task_type, 0)
        raw_target = target_size * weight
        quota = min(available, int(raw_target))
        quotas[task_type] = quota
        assigned += quota
        remainders.append((raw_target - int(raw_target), task_type))

    remaining = min(target_size, sum(counts.values())) - assigned
    for _, task_type in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        available = counts.get(task_type, 0) - quotas.get(task_type, 0)
        if available <= 0:
            continue
        quotas[task_type] += 1
        remaining -= 1

    if remaining > 0:
        for task_type, available in counts.most_common():
            if remaining <= 0:
                break
            spare = available - quotas.get(task_type, 0)
            if spare <= 0:
                continue
            take = min(spare, remaining)
            quotas[task_type] = quotas.get(task_type, 0) + take
            remaining -= take

    return quotas


def allocate_bucket_targets(grouped: dict[str, list[dict]], quota: int) -> dict[str, int]:
    bucket_counts = Counter({bucket: len(records) for bucket, records in grouped.items()})
    if quota <= 0:
        return {bucket: 0 for bucket in bucket_counts}

    bucket_targets: dict[str, int] = {}
    assigned = 0
    remainders: list[tuple[float, str]] = []
    total = sum(bucket_counts.values())
    for bucket, count in bucket_counts.items():
        raw_target = quota * count / total
        value = min(count, int(raw_target))
        bucket_targets[bucket] = value
        assigned += value
        remainders.append((raw_target - int(raw_target), bucket))

    remaining = quota - assigned
    for _, bucket in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        spare = bucket_counts[bucket] - bucket_targets[bucket]
        if spare <= 0:
            continue
        bucket_targets[bucket] += 1
        remaining -= 1

    return bucket_targets


def select_records(records: list[dict], quota: int, max_per_source: int) -> list[dict]:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_bucket[length_bucket(record.get("output", ""))].append(record)

    bucket_targets = allocate_bucket_targets(by_bucket, quota)
    selected: list[dict] = []
    per_source: Counter[str] = Counter()

    for bucket_name in sorted(by_bucket):
        candidates = sorted(by_bucket[bucket_name], key=stable_rank)
        target = bucket_targets.get(bucket_name, 0)
        for record in candidates:
            if len(selected) >= quota or target <= 0:
                break
            source_id = str(record.get("source_id") or "")
            if source_id and per_source[source_id] >= max_per_source:
                continue
            selected.append(record)
            per_source[source_id] += 1
            target -= 1

    if len(selected) < quota:
        chosen = {stable_rank(record) for record in selected}
        leftovers = sorted(records, key=stable_rank)
        for record in leftovers:
            if len(selected) >= quota:
                break
            rank = stable_rank(record)
            if rank in chosen:
                continue
            source_id = str(record.get("source_id") or "")
            if source_id and per_source[source_id] >= max_per_source:
                continue
            selected.append(record)
            per_source[source_id] += 1
            chosen.add(rank)

    return selected


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    records = load_records(input_path)
    counts = Counter(str(record.get("task_type") or "(none)") for record in records)
    quotas = allocate_targets(counts, args.target_size)

    selected: list[dict] = []
    for task_type in sorted(counts):
        group = [record for record in records if str(record.get("task_type") or "(none)") == task_type]
        quota = quotas.get(task_type, 0)
        selected.extend(select_records(group, quota, args.max_per_source))

    selected = sorted(selected, key=stable_rank)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    selected_counts = Counter(str(record.get("task_type") or "(none)") for record in selected)
    selected_bucket_counts = Counter(length_bucket(record.get("output", "")) for record in selected)
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "summary": {
            "input_records": len(records),
            "output_records": len(selected),
            "target_size": args.target_size,
            "max_per_source": args.max_per_source,
            "input_task_type_counts": dict(counts),
            "quota_by_task_type": dict(quotas),
            "output_task_type_counts": dict(selected_counts),
            "output_length_bucket_counts": dict(selected_bucket_counts),
        },
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
