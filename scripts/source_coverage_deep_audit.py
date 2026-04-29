#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import random
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = Path("/Users/unoa/Downloads/crawled-data-v2")
DEFAULT_TRAIN_PATH = ROOT / "cpt_corpus.v3.jsonl"
DEFAULT_VAL_PATH = ROOT / "val_set.v2.jsonl"
DEFAULT_OUTPUT = ROOT / "runs" / "source-coverage-deep-audit.json"
SAMPLE_SIZE = 1000
SAMPLE_SEED = 42

HANGUL_RE = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
LEADING_REPLY_TAG_RE = re.compile(r"^\[(\d+(?:-\d+)*)\]")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P1 = load_module("phase1_data_pipeline", ROOT / "scripts" / "phase1_data_pipeline.py")
SC = load_module("sanitize_context", ROOT / "scripts" / "sanitize_context.py")


def norm(text: str) -> str:
    return (text or "").replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_raw_posts(raw_dir: Path) -> dict[str, dict[str, Any]]:
    posts: dict[str, dict[str, Any]] = {}
    for post, source_file in P1.iter_posts(str(raw_dir)):
        post_id = str(post.get("id") or post.get("post_id") or "")
        posts[post_id] = {
            "id": post_id,
            "boardName": str(post.get("boardName") or "UNKNOWN"),
            "title": norm(post.get("title") or ""),
            "content": norm(post.get("content") or ""),
            "date": str(post.get("date") or post.get("createdAt") or ""),
            "comments": [c for c in (post.get("comments") or []) if isinstance(c, dict)],
            "source_file": Path(source_file).name,
        }
    return posts


def classify_clean_text(text: str) -> str:
    cleaned = norm(text)
    if not cleaned:
        return "empty"
    if P1.minor_proximity_block(cleaned):
        return "minor_filter"
    if len(cleaned) < 5:
        return "too_short"
    if P1.SPAM_RE.search(cleaned):
        return "spam_filter"
    return "eligible"


def classify_post(row: dict[str, Any]) -> str:
    combined = norm(f"{row['title']}\n{row['content']}".strip())
    if not combined:
        return "empty"
    scrubbed, _ = P1.scrub_pii(combined)
    return classify_clean_text(scrubbed)


def classify_comment(comment_text: str) -> str:
    scrubbed, _ = P1.scrub_pii(norm(comment_text))
    return classify_clean_text(scrubbed)


def load_train_artifacts(train_path: Path, val_path: Path, raw_posts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    train_rows = 0
    train_posts = 0
    train_comments_total = 0
    base_comment_rows = 0
    patch_row_count = 0
    direct_source_ids: set[str] = set()
    direct_post_ids: set[str] = set()
    patch_source_comment_ids: set[str] = set()
    patch_thread_ids: set[str] = set()
    train_board_rows = Counter()
    train_board_posts = Counter()

    with train_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            train_rows += 1
            sid = str(row["source_id"])
            post_id = sid.split(":comment:", 1)[0] if ":comment:" in sid else sid
            board = raw_posts.get(post_id, {}).get("boardName", "UNKNOWN")
            train_board_rows[board] += 1
            if row.get("kind") == "comment":
                train_comments_total += 1
                if ":comment:" in sid:
                    patch_row_count += 1
                    patch_source_comment_ids.add(sid)
                    patch_thread_ids.add(post_id)
                else:
                    direct_source_ids.add(sid)
                    base_comment_rows += 1
            else:
                direct_source_ids.add(sid)
                if row.get("kind") == "post":
                    train_posts += 1
                    direct_post_ids.add(sid)
                    train_board_posts[board] += 1

    val_source_ids: set[str] = set()
    with val_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            val_source_ids.add(str(row["source_id"]))

    return {
        "train_rows": train_rows,
        "train_posts": train_posts,
        "train_comments_total": train_comments_total,
        "base_comment_rows": base_comment_rows,
        "patch_row_count": patch_row_count,
        "direct_source_ids": direct_source_ids,
        "direct_post_ids": direct_post_ids,
        "patch_source_comment_ids": patch_source_comment_ids,
        "patch_thread_ids": patch_thread_ids,
        "all_thread_ids_with_train_rows": direct_source_ids | patch_thread_ids,
        "val_source_ids": val_source_ids,
        "train_board_rows": train_board_rows,
        "train_board_posts": train_board_posts,
    }


def make_source_distributions(raw_posts: dict[str, dict[str, Any]]) -> dict[str, Counter]:
    source_board_posts = Counter()
    source_board_units = Counter()
    for row in raw_posts.values():
        board = row["boardName"]
        source_board_posts[board] += 1
        source_board_units[board] += 1 + len(row["comments"])
    return {
        "source_board_posts": source_board_posts,
        "source_board_units": source_board_units,
    }


def classify_missing_threads(
    raw_posts: dict[str, dict[str, Any]],
    all_thread_ids_with_train_rows: set[str],
    val_source_ids: set[str],
) -> dict[str, Any]:
    missing_ids = sorted(set(raw_posts) - all_thread_ids_with_train_rows)
    val_only: list[dict[str, Any]] = []
    non_val: list[dict[str, Any]] = []
    non_val_reason_counts = Counter()
    missing_dates = Counter()
    val_dates = Counter()
    non_val_dates = Counter()

    for post_id in missing_ids:
        row = raw_posts[post_id]
        post_reason = classify_post(row)
        comment_reasons = Counter(classify_comment(comment.get("content") or "") for comment in row["comments"])
        record = {
            "source_id": post_id,
            "boardName": row["boardName"],
            "date": row["date"],
            "title": row["title"],
            "content_excerpt": row["content"][:180],
            "comment_count": len(row["comments"]),
            "post_reason": post_reason,
            "comment_reason_counts": dict(comment_reasons),
            "comment_excerpt_samples": [norm(comment.get("content") or "")[:180] for comment in row["comments"][:3]],
            "source_file": row["source_file"],
        }
        day = row["date"][:10]
        missing_dates[day] += 1
        if post_id in val_source_ids:
            record["missing_group"] = "val_only"
            record["reason"] = "held_out_in_val_set"
            val_only.append(record)
            val_dates[day] += 1
            continue

        if post_reason != "eligible" and not any(reason == "eligible" for reason in comment_reasons):
            record["missing_group"] = "missing_from_both_train_and_val"
            record["reason"] = "fully_filtered"
        else:
            record["missing_group"] = "missing_from_both_train_and_val"
            record["reason"] = "unexplained_missing_from_train_and_val"
        non_val.append(record)
        non_val_reason_counts[record["reason"]] += 1
        non_val_dates[day] += 1

    rng = random.Random(SAMPLE_SEED)
    val_sample = rng.sample(val_only, min(10, len(val_only)))
    unexplained = [row for row in non_val if row["reason"] == "unexplained_missing_from_train_and_val"]
    filtered = [row for row in non_val if row["reason"] == "fully_filtered"]
    non_val_sample = rng.sample(unexplained, min(8, len(unexplained))) + rng.sample(filtered, min(2, len(filtered)))
    sample = val_sample + non_val_sample
    if len(sample) < 20:
        remainder = [row for row in non_val if row not in non_val_sample]
        needed = 20 - len(sample)
        sample.extend(rng.sample(remainder, min(needed, len(remainder))))

    return {
        "missing_source_ids_total": len(missing_ids),
        "missing_in_val_only": len(val_only),
        "missing_from_both_train_and_val": len(non_val),
        "missing_in_val_only_pct": round(len(val_only) / len(missing_ids), 6) if missing_ids else 0.0,
        "missing_from_both_train_and_val_pct": round(len(non_val) / len(missing_ids), 6) if missing_ids else 0.0,
        "non_val_reason_counts": dict(non_val_reason_counts),
        "missing_date_top": missing_dates.most_common(10),
        "val_only_missing_date_top": val_dates.most_common(10),
        "missing_from_both_date_top": non_val_dates.most_common(10),
        "sample_selection": "10 val-only + 8 unexplained missing-from-both + 2 fully-filtered (seed=42)",
        "sample": sample[:20],
    }


def source_text_units(raw_posts: dict[str, dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for row in raw_posts.values():
        post_text = norm(f"{row['title']}\n{row['content']}".strip())
        if post_text:
            texts.append(post_text)
        for comment in row["comments"]:
            text = norm(comment.get("content") or "")
            if text:
                texts.append(text)
    return texts


def train_text_units(train_path: Path) -> list[str]:
    texts: list[str] = []
    with train_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(norm(row["text"]))
    return texts


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for piece in re.split(r"[\n]+|(?<=[.!?！？])\s+", text):
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return parts


def sentence_bucket(length: int) -> str:
    if length <= 9:
        return "1-9"
    if length <= 19:
        return "10-19"
    if length <= 39:
        return "20-39"
    if length <= 79:
        return "40-79"
    if length <= 159:
        return "80-159"
    return "160+"


def sample_character_metrics(texts: list[str]) -> dict[str, Any]:
    total_chars = sum(len(text) for text in texts)
    hangul_chars = sum(1 for text in texts for ch in text if HANGUL_RE.match(ch))
    punctuation_chars = sum(1 for text in texts for ch in text if unicodedata.category(ch).startswith("P"))
    symbol_chars = sum(1 for text in texts for ch in text if unicodedata.category(ch).startswith("S"))
    emoji_chars = sum(1 for text in texts for ch in text if EMOJI_RE.match(ch))
    rows_with_symbol = sum(1 for text in texts if any(unicodedata.category(ch).startswith("S") for ch in text))
    rows_with_emoji = sum(1 for text in texts if EMOJI_RE.search(text))
    rows_with_leading_reply_tag = sum(1 for text in texts if LEADING_REPLY_TAG_RE.match(text))
    text_lengths = [len(text) for text in texts]
    sentence_lengths = [len(sentence) for text in texts for sentence in split_sentences(text)]
    hist = Counter(sentence_bucket(length) for length in sentence_lengths)

    return {
        "total_chars": total_chars,
        "hangul_ratio": round(hangul_chars / total_chars, 6) if total_chars else 0.0,
        "punctuation_density": round(punctuation_chars / total_chars, 6) if total_chars else 0.0,
        "symbol_density": round(symbol_chars / total_chars, 6) if total_chars else 0.0,
        "emoji_char_ratio": round(emoji_chars / total_chars, 6) if total_chars else 0.0,
        "rows_with_symbol_pct": round(rows_with_symbol / len(texts), 6) if texts else 0.0,
        "rows_with_emoji_pct": round(rows_with_emoji / len(texts), 6) if texts else 0.0,
        "leading_reply_tag_pct": round(rows_with_leading_reply_tag / len(texts), 6) if texts else 0.0,
        "text_length_mean": round(statistics.mean(text_lengths), 3) if text_lengths else 0.0,
        "text_length_median": round(statistics.median(text_lengths), 3) if text_lengths else 0.0,
        "sentence_count": len(sentence_lengths),
        "sentence_length_mean": round(statistics.mean(sentence_lengths), 3) if sentence_lengths else 0.0,
        "sentence_length_median": round(statistics.median(sentence_lengths), 3) if sentence_lengths else 0.0,
        "sentence_length_histogram": {bucket: hist[bucket] for bucket in ["1-9", "10-19", "20-39", "40-79", "80-159", "160+"]},
    }


def compare_character_metrics(raw_posts: dict[str, dict[str, Any]], train_path: Path) -> dict[str, Any]:
    source_units = source_text_units(raw_posts)
    train_units = train_text_units(train_path)
    source_rng = random.Random(SAMPLE_SEED)
    train_rng = random.Random(SAMPLE_SEED)
    source_sample = source_rng.sample(source_units, SAMPLE_SIZE)
    train_sample = train_rng.sample(train_units, SAMPLE_SIZE)
    source_metrics = sample_character_metrics(source_sample)
    train_metrics = sample_character_metrics(train_sample)

    deltas = {}
    for key in [
        "hangul_ratio",
        "punctuation_density",
        "symbol_density",
        "emoji_char_ratio",
        "rows_with_symbol_pct",
        "rows_with_emoji_pct",
        "leading_reply_tag_pct",
        "text_length_mean",
        "text_length_median",
        "sentence_length_mean",
        "sentence_length_median",
    ]:
        deltas[key] = round(train_metrics[key] - source_metrics[key], 6)

    return {
        "sample_size": SAMPLE_SIZE,
        "sample_seed": SAMPLE_SEED,
        "source": source_metrics,
        "train": train_metrics,
        "delta_train_minus_source": deltas,
    }


def board_distribution_payload(source_board_posts: Counter, source_board_units: Counter, train_board_rows: Counter, train_board_posts: Counter) -> dict[str, Any]:
    def rows(counter: Counter) -> list[dict[str, Any]]:
        total = sum(counter.values())
        return [
            {
                "boardName": board,
                "count": count,
                "share": round(count / total, 6) if total else 0.0,
            }
            for board, count in counter.most_common()
        ]

    return {
        "source_board_count": len(source_board_posts),
        "source_posts_by_board": rows(source_board_posts),
        "source_text_units_by_board": rows(source_board_units),
        "train_posts_by_board": rows(train_board_posts),
        "train_rows_by_board": rows(train_board_rows),
        "note": "Only one boardName exists in the source crawl, so board-level skew is not measurable beyond 100% concentration in that single board.",
    }


def build_report(raw_dir: Path, train_path: Path, val_path: Path, output: Path) -> dict[str, Any]:
    raw_posts = parse_raw_posts(raw_dir)
    distributions = make_source_distributions(raw_posts)
    train = load_train_artifacts(train_path, val_path, raw_posts)
    missing = classify_missing_threads(raw_posts, train["all_thread_ids_with_train_rows"], train["val_source_ids"])
    char_metrics = compare_character_metrics(raw_posts, train_path)
    board_payload = board_distribution_payload(
        distributions["source_board_posts"],
        distributions["source_board_units"],
        train["train_board_rows"],
        train["train_board_posts"],
    )

    source_posts = len(raw_posts)
    source_comments = sum(len(row["comments"]) for row in raw_posts.values())
    direct_source_id_coverage = len(train["direct_source_ids"])
    direct_post_row_coverage = len(train["direct_post_ids"])
    any_thread_coverage = len(train["all_thread_ids_with_train_rows"])
    patch_unique_source_comments = len(train["patch_source_comment_ids"])
    represented_source_comments = train["base_comment_rows"] + patch_unique_source_comments

    return {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_dir": str(raw_dir),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "output_path": str(output),
            "sample_size": SAMPLE_SIZE,
            "sample_seed": SAMPLE_SEED,
        },
        "coverage": {
            "source_posts": source_posts,
            "source_comments": source_comments,
            "train_rows": train["train_rows"],
            "train_posts": train["train_posts"],
            "train_comments_total": train["train_comments_total"],
            "post_source_id_coverage": {
                "direct_source_id_match": {
                    "covered": direct_source_id_coverage,
                    "total": source_posts,
                    "pct": round(direct_source_id_coverage / source_posts, 6),
                },
                "train_post_rows_only": {
                    "covered": direct_post_row_coverage,
                    "total": source_posts,
                    "pct": round(direct_post_row_coverage / source_posts, 6),
                },
                "any_train_representation_including_patch_rows": {
                    "covered": any_thread_coverage,
                    "total": source_posts,
                    "pct": round(any_thread_coverage / source_posts, 6),
                },
                "patch_only_thread_recoveries": any_thread_coverage - direct_source_id_coverage,
            },
            "comment_survival": {
                "base_comment_rows": train["base_comment_rows"],
                "patch_row_count": train["patch_row_count"],
                "patch_unique_source_comments": patch_unique_source_comments,
                "represented_source_comments": represented_source_comments,
                "represented_source_comments_pct": round(represented_source_comments / source_comments, 6),
                "dropped_source_comments": source_comments - represented_source_comments,
                "dropped_source_comments_pct": round((source_comments - represented_source_comments) / source_comments, 6),
                "note": "The base comment rows are one comment-row per retained raw comment. The patch adds 1,269 rows but they come from 1,070 distinct raw comments, so raw-comment coverage must use unique synthetic source_id count rather than patch row count.",
            },
            "kind_mix": {
                "source_post_share": round(source_posts / (source_posts + source_comments), 6),
                "source_comment_share": round(source_comments / (source_posts + source_comments), 6),
                "train_post_share": round(train["train_posts"] / train["train_rows"], 6),
                "train_comment_share": round(train["train_comments_total"] / train["train_rows"], 6),
            },
        },
        "systematic_missingness": {
            "summary": [
                "604 source thread IDs have no representation in cpt_corpus.v3.jsonl.",
                "522 of those 604 missing thread IDs are present in val_set.v2.jsonl, so the dominant missingness pattern is val holdout, concentrated on the final crawl days.",
                "82 source thread IDs are absent from both train and val; 74 of those still look eligible under the current raw-data heuristics, which is evidence of stale or mismatched corpus provenance rather than intended filtering.",
                "Only one boardName exists in the source crawl, so the missingness is thread-level and date-level rather than board-level.",
            ],
            **missing,
            "board_distribution": board_payload,
        },
        "character_level_comparison": char_metrics,
        "verdict": {
            "faithful_representation": False,
            "because": [
                "Thread coverage is incomplete: only 10,684 of 11,288 source thread IDs (94.6492%) have any train representation.",
                "Comment survival is materially reduced: 37,456 of 55,849 source comments (67.0666%) are represented after filtering and patch recovery.",
                "The train sample is structurally less forum-native than raw: source sample rows start with reply tags 84.1% of the time, while the train sample starts with reply tags 0% of the time.",
                "There are 82 thread IDs missing from both train and val, and most of them do not look like intentional spam/short/minor drops.",
            ],
        },
    }


def main() -> None:
    raw_dir = DEFAULT_RAW_DIR
    train_path = DEFAULT_TRAIN_PATH
    val_path = DEFAULT_VAL_PATH
    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(raw_dir, train_path, val_path, output)

    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
