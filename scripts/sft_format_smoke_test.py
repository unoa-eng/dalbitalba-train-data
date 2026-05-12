#!/usr/bin/env python3
"""
sft_format_smoke_test.py — P1-C format smoke test

Verifies train_sft.py's active SFT formatting paths produce correctly-masked
examples for raw continuation, legacy v2 supervised pairs, and thread-
conditioned v3 chatml rows, without loading any model weights.

Exit 0 = all assertions pass. Non-zero = at least one failure.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import sys
import tempfile
import types
from pathlib import Path

# ── Patch env before any import so train_sft doesn't try /workspace paths ────
os.environ.setdefault("SFT_LOG_FILE", "/tmp/sft_smoke_test.log")

# ── Import tokenizer runtime, with a torch stub fallback ─────────────────────
# transformers import_utils inspects torch availability. When torch is missing,
# provide a minimal stub so tokenizer-only smoke remains runnable on lightweight
# local environments.
try:  # pragma: no cover - depends on local environment
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - exercised on torch-less hosts
    _torch_stub = types.ModuleType("torch")
    _torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _torch_stub.__version__ = "0.0-smoke-stub"
    _torch_stub.Tensor = object
    _torch_stub.bfloat16 = "bfloat16"
    _torch_stub.manual_seed = lambda *args, **kwargs: None
    _torch_stub.use_deterministic_algorithms = lambda *args, **kwargs: None
    _torch_stub.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda *args, **kwargs: None,
        get_device_name=lambda *_args, **_kwargs: "smoke-stub-cuda",
    )
    _torch_stub.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)
    )
    sys.modules["torch"] = _torch_stub

from transformers import AutoTokenizer  # noqa: E402

# ── Stub heavy GPU-only / training-only packages ─────────────────────────────
# datasets
_ds_mod = types.ModuleType("datasets")
class _FakeDataset(list):
    @classmethod
    def from_list(cls, lst):
        return cls(lst)
_ds_mod.Dataset = _FakeDataset
sys.modules["datasets"] = _ds_mod

# peft
_peft_mod = types.ModuleType("peft")
class _TaskType:
    CAUSAL_LM = "CAUSAL_LM"
_peft_mod.TaskType = _TaskType
_peft_mod.LoraConfig = object
_peft_mod.get_peft_model = lambda m, c: m
_peft_mod.prepare_model_for_kbit_training = lambda m, **kw: m
sys.modules["peft"] = _peft_mod

# Replace transformers module in sys.modules with a thin wrapper that keeps
# AutoTokenizer real but stubs the training classes train_sft imports.
import transformers as _tf_real
_tf_stub = types.ModuleType("transformers")
_tf_stub.AutoTokenizer = AutoTokenizer
_tf_stub.AutoModelForCausalLM = object
_tf_stub.BitsAndBytesConfig = object
_tf_stub.DataCollatorForSeq2Seq = object
_tf_stub.Trainer = object
_tf_stub.TrainingArguments = object
sys.modules["transformers"] = _tf_stub

# ── Import train_sft from repo root ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_sft  # noqa: E402

# Restore real transformers so AutoTokenizer methods keep working
sys.modules["transformers"] = _tf_real


# ── Test harness ─────────────────────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    RESULTS.append((name, condition, detail))
    marker = "  [PASS]" if condition else "  [FAIL]"
    suffix = f" — {detail}" if detail else ""
    print(f"{marker} {name}{suffix}")
    return condition


V2_ROW = {
    "post": "오늘 강남 쪽 알바 구하는 중인데 추천 있나요?",
    "comment": "직통 DM 주면 소개해드림ㅋ",
    "thread_key": "thread_v2_001",
    "source_id": "dc_v2_001",
    "length_bucket": "short",
}

V3_ROW = {
    "instruction": "유흥업 종사자 커뮤니티 답글 스타일로 한 줄 답변을 작성하세요.",
    "input": (
        "[게시판] 유흥알바\n"
        "[유형] reply\n"
        "[제목] 강남 알바 질문\n"
        "[원글] 이번 주 강남 쪽 알바 구하는 중인데 어떤 데가 제일 나음?\n"
        "[부모댓글] 거기 대우 좋다더라\n"
        "[depth] 2\n"
        "위에 어울리는 답글을 1개 작성하세요."
    ),
    "output": "ㅇㅇ 거기 시급도 쏠쏠하고 분위기도 괜찮음",
    "persona_id": "p-011",
    "persona": "짧고 단호한 조언자",
    "board": "유흥알바",
    "post_title": "강남 알바 질문",
    "post_body_excerpt": "이번 주 강남 쪽 알바 구하는 중인데 어떤 데가 제일 나음?",
    "parent_comment": "거기 대우 좋다더라",
    "target_comment": "ㅇㅇ 거기 시급도 쏠쏠하고 분위기도 괜찮음",
    "thread_key": "thread_001",
    "depth": 2,
    "task_type": "reply",
    "length_bucket": "short",
    "source_id": "tc_001",
    "loss_weight": 1.0,
}

RAW_ROW = {
    "text": "오늘 출근했는데 생각보다 손님 텐션 괜찮아서 버틸만했음",
    "kind": "crawl_comment",
    "source_id": "raw_001",
    "source_field": "text",
    "length_bucket": "short",
}

VAL_ROW = {
    "text": "이번 주말 출근 괜찮냐는 질문에 답변 달린 예시 문장",
}


def set_train_sft_runtime(**overrides) -> None:
    for key, value in overrides.items():
        setattr(train_sft, key, value)
    train_sft.PAIR_RATIO = 1.0 - train_sft.RAW_RATIO


def write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [train_sft.json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset_case(
    tokenizer,
    tmpdir: Path,
    *,
    raw_rows: list[dict],
    pair_rows: list[dict],
    pair_v3_rows: list[dict] | None,
    prompt_format: str,
    raw_ratio: float,
) -> tuple[list, list | None]:
    raw_path = tmpdir / "raw.jsonl"
    pair_path = tmpdir / "pair.jsonl"
    pair_v3_path = tmpdir / "pair_v3.jsonl"
    val_path = tmpdir / "val.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(pair_path, pair_rows)
    write_jsonl(val_path, [VAL_ROW])
    if pair_v3_rows is not None:
        write_jsonl(pair_v3_path, pair_v3_rows)
        pair_v3_ref: str | None = str(pair_v3_path)
    else:
        pair_v3_ref = None

    set_train_sft_runtime(
        SFT_RAW_JSONL=str(raw_path),
        SFT_PAIR_JSONL=str(pair_path),
        SFT_PAIR_JSONL_V3=pair_v3_ref,
        SFT_VAL_JSONL=str(val_path),
        RAW_RATIO=raw_ratio,
        PROMPT_FORMAT=prompt_format,
        RAW_LIMIT_ROWS=0,
        PAIR_LIMIT_ROWS=0,
        VAL_LIMIT_ROWS=0,
    )
    return train_sft.build_mixed_dataset(tokenizer)


def check_masked_supervised_example(name: str, example: dict) -> None:
    check(f"{name}: example not None", example is not None)
    if not example:
        return
    check(f"{name}: prompt non-empty", len(example["input_ids"]) > 0, f"len={len(example['input_ids'])}")
    masked = [label for label in example["labels"] if label == -100]
    non_masked = [label for label in example["labels"] if label != -100]
    check(f"{name}: prompt tokens masked", len(masked) > 0, f"masked_count={len(masked)}")
    check(
        f"{name}: response tokens trainable",
        len(non_masked) > 0,
        f"non_masked_count={len(non_masked)}",
    )
    check(
        f"{name}: labels length == input_ids length",
        len(example["labels"]) == len(example["input_ids"]),
        f"{len(example['labels'])} == {len(example['input_ids'])}",
    )


def run_smoke_tests(tokenizer) -> None:
    with tempfile.TemporaryDirectory(prefix="sft-format-smoke-") as tmp:
        tmpdir = Path(tmp)

        # ── 1. raw continuation builder path ─────────────────────────────────
        print("\n[raw continuation builder]")
        raw_ds, raw_eval_ds = build_dataset_case(
            tokenizer,
            tmpdir,
            raw_rows=[RAW_ROW],
            pair_rows=[V2_ROW],
            pair_v3_rows=None,
            prompt_format="raw",
            raw_ratio=1.0,
        )
        raw_example = raw_ds[0] if raw_ds else None
        check("raw_continuation: example not None", raw_example is not None)
        if raw_example:
            check(
                "raw_continuation: labels exactly match input_ids",
                raw_example["labels"] == raw_example["input_ids"],
                f"len={len(raw_example['labels'])}",
            )
            check(
                "raw_continuation: attention length == input_ids length",
                len(raw_example["attention_mask"]) == len(raw_example["input_ids"]),
                f"{len(raw_example['attention_mask'])} == {len(raw_example['input_ids'])}",
            )
        check("raw_continuation: eval dataset built", raw_eval_ds is not None)
        if raw_eval_ds is not None:
            check("raw_continuation: eval dataset has rows", len(raw_eval_ds) == 1, f"len={len(raw_eval_ds)}")

        # ── 2. legacy v2 supervised raw-format builder path ──────────────────
        print("\n[v2 supervised raw-format builder]")
        v2_ds, _ = build_dataset_case(
            tokenizer,
            tmpdir,
            raw_rows=[RAW_ROW],
            pair_rows=[V2_ROW],
            pair_v3_rows=None,
            prompt_format="raw",
            raw_ratio=0.0,
        )
        v2_example = v2_ds[0] if v2_ds else None
        check_masked_supervised_example("v2_raw", v2_example)

        # ── 3. v3 thread-conditioned chatml builder path ─────────────────────
        print("\n[v3 thread-conditioned chatml builder]")
        v3_ds, _ = build_dataset_case(
            tokenizer,
            tmpdir,
            raw_rows=[RAW_ROW],
            pair_rows=[V2_ROW],
            pair_v3_rows=[V3_ROW],
            prompt_format="chatml",
            raw_ratio=0.0,
        )
        v3_example = v3_ds[0] if v3_ds else None
        check_masked_supervised_example("v3_chatml", v3_example)
        if v3_example:
            seq_len_chatml = len(v3_example["input_ids"])
            raw_text = V3_ROW["post_title"] + "\n" + V3_ROW["target_comment"]
            raw_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
            check(
                "v3_chatml: chatml seq_len > bare text seq_len",
                seq_len_chatml > len(raw_ids),
                f"chatml={seq_len_chatml} raw={len(raw_ids)}",
            )

        # ── 4. legacy v2 chatml builder path ─────────────────────────────────
        print("\n[v2 chatml builder]")
        v2_chatml = train_sft.build_chatml_example(V2_ROW, tokenizer, is_v3=False)
        check_masked_supervised_example("v2_chatml", v2_chatml)

        # ── 5. chatml special token round-trip ───────────────────────────────
        print("\n[chatml special token round-trip]")
        special_seq = (
            "<|im_start|>system\n테스트<|im_end|>\n"
            "<|im_start|>user\n안녕<|im_end|>\n"
            "<|im_start|>assistant\n안녕하세요<|im_end|>"
        )
        ids = tokenizer(special_seq, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        check(
            "round-trip: <|im_start|> preserved",
            "<|im_start|>" in decoded,
            f"decoded[:80]={decoded[:80]!r}",
        )
        check(
            "round-trip: <|im_end|> preserved",
            "<|im_end|>" in decoded,
        )

        # ── 6. v3 user text builder ──────────────────────────────────────────
        print("\n[v3 user text builder]")
        user_text = train_sft._build_chatml_user_text(V3_ROW)
        check("user_text: non-empty", len(user_text) > 0)
        check("user_text: contains post_title", V3_ROW["post_title"] in user_text)
        check("user_text: contains parent_comment", V3_ROW["parent_comment"] in user_text)
        check("user_text: contains depth marker", "[depth]" in user_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_tokenizer = "tokenizer_v4" if (Path(__file__).resolve().parents[1] / "tokenizer_v4").is_dir() else "Qwen/Qwen3-8B-Base"
    parser.add_argument("--tokenizer", default=default_tokenizer)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("sft_format_smoke_test.py — P1-C")
    print("=" * 60)

    print(f"\nLoading tokenizer: {args.tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=True,
    )
    print(f"  eos_token     : {tokenizer.eos_token!r}")
    print(f"  chat_template : {'yes' if tokenizer.chat_template else 'no'}")

    run_smoke_tests(tokenizer)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    total = len(RESULTS)
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if failed == 0:
        print("SUCCESS — all assertions passed")
        sys.exit(0)
    else:
        print("FAILURE — see [FAIL] lines above")
        sys.exit(1)


if __name__ == "__main__":
    main()
