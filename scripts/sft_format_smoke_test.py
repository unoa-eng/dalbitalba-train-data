#!/usr/bin/env python3
"""
sft_format_smoke_test.py — P1-C format smoke test

Verifies train_sft.py loaders produce correctly-masked examples for both
v2 and v3 schemas, without loading any model weights.

Exit 0 = all assertions pass. Non-zero = at least one failure.
"""

from __future__ import annotations

import os
import sys
import types

# ── Patch env before any import so train_sft doesn't try /workspace paths ────
os.environ.setdefault("SFT_LOG_FILE", "/tmp/sft_smoke_test.log")

# ── Import real packages first (torch, transformers) ─────────────────────────
# This must happen before any stubbing to avoid ImportError in transformers'
# internal import_utils which inspects torch.__spec__.
import torch  # noqa: F401 — real torch (cpu build is fine)
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


def run_smoke_tests(tokenizer) -> None:
    # ── Synthetic rows ────────────────────────────────────────────────────────

    V2_ROW = {
        "post": "오늘 강남 쪽 알바 구하는 중인데 추천 있나요?",
        "comment": "직통 DM 주면 소개해드림ㅋ",
    }

    V3_ROW = {
        "post_title": "강남 알바 질문",
        "post_body_excerpt": "이번 주 강남 쪽 알바 구하는 중인데 어떤 데가 제일 나음?",
        "parent_comment": "거기 대우 좋다더라",
        "target_comment": "ㅇㅇ 거기 시급도 쏠쏠하고 분위기도 괜찮음",
        "thread_key": "thread_001",
        "depth": 2,
        "task_type": "reply",
        "length_bucket": "short",
        "source_id": "dc_001",
        "board": "유흥알바",
    }

    # ── 1. v2 raw pair builder ────────────────────────────────────────────────
    print("\n[v2 raw pair builder]")
    ex_v2_raw = train_sft.build_pair_example_raw(V2_ROW, tokenizer)

    check("v2_raw: example not None", ex_v2_raw is not None)
    if ex_v2_raw:
        check(
            "v2_raw: prompt non-empty",
            len(ex_v2_raw["input_ids"]) > 0,
            f"len={len(ex_v2_raw['input_ids'])}",
        )
        non_masked = [l for l in ex_v2_raw["labels"] if l != -100]
        check(
            "v2_raw: response has non-(-100) labels",
            len(non_masked) > 0,
            f"non_masked_count={len(non_masked)}",
        )
        masked = [l for l in ex_v2_raw["labels"] if l == -100]
        check(
            "v2_raw: prompt tokens are masked (-100)",
            len(masked) > 0,
            f"masked_count={len(masked)}",
        )
        check(
            "v2_raw: labels length == input_ids length",
            len(ex_v2_raw["labels"]) == len(ex_v2_raw["input_ids"]),
            f"{len(ex_v2_raw['labels'])} == {len(ex_v2_raw['input_ids'])}",
        )

    # ── 2. v3 chatml builder ─────────────────────────────────────────────────
    print("\n[v3 chatml builder]")
    ex_v3_chatml = train_sft.build_chatml_example(V3_ROW, tokenizer, is_v3=True)

    check("v3_chatml: example not None", ex_v3_chatml is not None)
    if ex_v3_chatml:
        seq_len_chatml = len(ex_v3_chatml["input_ids"])
        check(
            "v3_chatml: prompt non-empty",
            seq_len_chatml > 0,
            f"seq_len={seq_len_chatml}",
        )
        non_masked_v3 = [l for l in ex_v3_chatml["labels"] if l != -100]
        check(
            "v3_chatml: response tokens have non-(-100) labels",
            len(non_masked_v3) > 0,
            f"non_masked_count={len(non_masked_v3)}",
        )
        masked_v3 = [l for l in ex_v3_chatml["labels"] if l == -100]
        check(
            "v3_chatml: prompt tokens have label==-100",
            len(masked_v3) > 0,
            f"masked_count={len(masked_v3)}",
        )
        check(
            "v3_chatml: labels length == input_ids length",
            len(ex_v3_chatml["labels"]) == len(ex_v3_chatml["input_ids"]),
            f"{len(ex_v3_chatml['labels'])} == {len(ex_v3_chatml['input_ids'])}",
        )

        # chatml seq_len > raw seq_len (chatml wraps with role markers)
        raw_text = V3_ROW["post_title"] + "\n" + V3_ROW["target_comment"]
        raw_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
        check(
            "v3_chatml: chatml seq_len > bare text seq_len",
            seq_len_chatml > len(raw_ids),
            f"chatml={seq_len_chatml} raw={len(raw_ids)}",
        )

    # ── 3. v2 chatml builder (is_v3=False) ──────────────────────────────────
    print("\n[v2 chatml builder]")
    ex_v2_chatml = train_sft.build_chatml_example(V2_ROW, tokenizer, is_v3=False)

    check("v2_chatml: example not None", ex_v2_chatml is not None)
    if ex_v2_chatml:
        check(
            "v2_chatml: prompt non-empty",
            len(ex_v2_chatml["input_ids"]) > 0,
        )
        non_masked_v2c = [l for l in ex_v2_chatml["labels"] if l != -100]
        check(
            "v2_chatml: response tokens have non-(-100) labels",
            len(non_masked_v2c) > 0,
            f"count={len(non_masked_v2c)}",
        )
        masked_v2c = [l for l in ex_v2_chatml["labels"] if l == -100]
        check(
            "v2_chatml: prompt tokens have label==-100",
            len(masked_v2c) > 0,
            f"count={len(masked_v2c)}",
        )

    # ── 4. chatml special token round-trip ───────────────────────────────────
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

    # ── 5. v3 user text builder ──────────────────────────────────────────────
    print("\n[v3 user text builder]")
    user_text = train_sft._build_chatml_user_text(V3_ROW)
    check("user_text: non-empty", len(user_text) > 0)
    check("user_text: contains post_title", V3_ROW["post_title"] in user_text)
    check("user_text: contains parent_comment", V3_ROW["parent_comment"] in user_text)
    check("user_text: contains depth marker", "[depth]" in user_text)


def main() -> None:
    print("=" * 60)
    print("sft_format_smoke_test.py — P1-C")
    print("=" * 60)

    print("\nLoading tokenizer: Qwen/Qwen3-8B-Base ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B-Base",
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
