#!/usr/bin/env python3
"""Round-2 Step 2 — Persona-30 extractor from Obsidian vault.

Targets:
- 구현/페르소나-설계/30인-요약.md  (the consolidated 30-persona summary)
- 구현/페르소나-설계/*.md           (any per-persona detail files)
- 구현/페르소나-설계/**/persona-*.md
- 0-INBOX/persona-*.md (drafts)

Output:
  persona-30-extracted.json  (list of persona records)
  persona-30-extracted.md    (human-readable digest)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PERSONA_DIR_HINTS = (
    "구현/페르소나-설계",
    "persona",
    "페르소나",
)

PERSONA_BLOCK_HEAD = re.compile(
    r"^(?:#{1,4}\s*)?(?:\*\*)?(\d{1,2})[\.\)]\s+([^\n]+)$",
    re.MULTILINE,
)
PERSONA_TABLE_ROW = re.compile(
    r"^\s*\|\s*(\d{1,2})\s*\|\s*([^|]+)\|([^|]+)\|([^|]+)\|([^|]*)\|"
)
PERSONA_PCODE_TABLE = re.compile(
    r"^\s*\|\s*p-(\d{2,3})\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]*)\|\s*([^|]*)\|"
)
PERSONA_PCODE_INLINE = re.compile(
    r"`?p-(\d{2,3})\s+([가-힣A-Za-z0-9_\-ㄱ-ㅎㅏ-ㅣ ]{1,20})`?"
)
NAME_HINT = re.compile(r"^[가-힣A-Za-z0-9_\-\(\) ]{2,30}$")
KEY_FIELDS = ["이름", "직업", "말투", "특징", "예시", "speech", "voice", "tone", "name", "role"]


def looks_like_persona_file(rel_path: str) -> bool:
    rel = rel_path.lower()
    return any(h.lower() in rel for h in PERSONA_DIR_HINTS) or "persona" in rel


def extract_personas_from_table(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        m = PERSONA_TABLE_ROW.match(line)
        if m:
            num = m.group(1).strip()
            cells = [c.strip() for c in m.groups()[1:]]
            out.append({
                "id": int(num),
                "name": cells[0],
                "role_or_age": cells[1],
                "speech_signature": cells[2],
                "example_or_note": cells[3] if len(cells) > 3 else "",
                "source": "table",
            })
        # p-NNN | nickname | tone | mood | particles | trait
        mp = PERSONA_PCODE_TABLE.match(line)
        if mp:
            num = int(mp.group(1))
            cells = [c.strip() for c in mp.groups()[1:]]
            out.append({
                "id": num,
                "name": cells[0],
                "tone": cells[1],
                "mood": cells[2],
                "particles": cells[3],
                "trait": cells[4] if len(cells) > 4 else "",
                "source": "p-table",
            })
    # Inline mentions like `p-013 ㅇㅇ`, `p-014 ㅇ` — pick up names not in table
    seen_ids = {p["id"] for p in out}
    for m in PERSONA_PCODE_INLINE.finditer(text):
        num = int(m.group(1))
        if num in seen_ids:
            continue
        nick = m.group(2).strip()
        if not nick or len(nick) > 25:
            continue
        out.append({
            "id": num,
            "name": nick,
            "source": "inline",
        })
        seen_ids.add(num)
    return out


def extract_personas_from_headings(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    matches = list(PERSONA_BLOCK_HEAD.finditer(text))
    for i, m in enumerate(matches):
        try:
            num = int(m.group(1))
        except ValueError:
            continue
        if num < 1 or num > 60:
            continue
        title = m.group(2).strip().rstrip("*")
        if not NAME_HINT.match(title.split("(")[0].strip()):
            # could still be persona header — keep
            pass
        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[block_start:block_end].strip()
        # try to pull "예시" or examples
        examples: list[str] = []
        for line in body.splitlines():
            ll = line.strip()
            if ll.startswith(">"):
                examples.append(ll.lstrip("> ").strip())
            elif ll.startswith("- ") and ('"' in ll or "「" in ll):
                examples.append(ll[2:].strip())
        out.append({
            "id": num,
            "name": title,
            "body_excerpt": body[:600],
            "examples": examples[:5],
            "source": "heading",
        })
    return out


def extract_kv_block(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*[-*]\s*\*?\*?([^:*]+)\*?\*?\s*[:：]\s*(.+)", line)
        if m:
            key = m.group(1).strip().lower()
            val = m.group(2).strip()
            for k in KEY_FIELDS:
                if k.lower() in key:
                    fields[k] = val
                    break
    return fields


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if not args.vault.is_dir():
        print(f"vault not found: {args.vault}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for fp in args.vault.rglob("*.md"):
        rel = fp.relative_to(args.vault).as_posix()
        if looks_like_persona_file(rel):
            candidates.append(fp)

    print(f"persona_candidate_files={len(candidates)}")

    aggregated: dict[int, dict[str, Any]] = {}
    free_form: list[dict[str, Any]] = []

    for fp in candidates:
        rel = fp.relative_to(args.vault).as_posix()
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {rel}: {exc}", file=sys.stderr)
            continue
        # 1) table-style rows
        for p in extract_personas_from_table(text):
            pid = p["id"]
            entry = aggregated.setdefault(pid, {"id": pid, "files": []})
            entry["files"].append(rel)
            for k, v in p.items():
                if k == "id" or v in (None, ""):
                    continue
                entry.setdefault(k, v)
        # 2) numbered headings
        for p in extract_personas_from_headings(text):
            pid = p["id"]
            entry = aggregated.setdefault(pid, {"id": pid, "files": []})
            if rel not in entry["files"]:
                entry["files"].append(rel)
            for k, v in p.items():
                if k == "id" or v in (None, ""):
                    continue
                entry.setdefault(k, v)
        # 3) per-file kv block (only for files that look like single-persona detail)
        if "/persona-" in rel.lower() or "/페르소나-" in rel:
            kv = extract_kv_block(text)
            if kv:
                free_form.append({"file": rel, "fields": kv, "head": text[:600]})

    persona_list = [aggregated[k] for k in sorted(aggregated.keys())]

    # Filter: must have name; signal is preferred but not strictly required for inline finds
    accepted = []
    for p in persona_list:
        has_name = bool(p.get("name"))
        has_signal = (p.get("speech_signature") or p.get("body_excerpt") or p.get("example_or_note")
                      or p.get("examples") or p.get("tone") or p.get("mood") or p.get("trait"))
        if has_name and (has_signal or p.get("source") == "inline") and 1 <= p["id"] <= 60:
            accepted.append(p)

    out_json = {
        "vault_root": str(args.vault),
        "candidate_files": [str(p.relative_to(args.vault).as_posix()) for p in candidates],
        "personas_aggregated_count": len(persona_list),
        "personas_accepted_count": len(accepted),
        "personas": accepted,
        "free_form_persona_files": free_form,
    }
    (args.out_dir / "persona-30-extracted.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (args.out_dir / "persona-30-extracted.md").open("w", encoding="utf-8") as fh:
        fh.write("# Round-2 Step-2 — Obsidian persona-30 extraction\n\n")
        fh.write(f"Total candidate files: {len(candidates)}\n")
        fh.write(f"Aggregated: {len(persona_list)}\n")
        fh.write(f"Accepted (with name + signal): {len(accepted)}\n\n")
        for p in accepted:
            name = p.get("name", "?")
            speech = p.get("speech_signature") or p.get("body_excerpt", "")[:200]
            fh.write(f"## {p['id']}. {name}\n\n")
            if "role_or_age" in p:
                fh.write(f"- **역할/나이**: {p['role_or_age']}\n")
            fh.write(f"- **말투/시그니처**: {speech}\n")
            ex = p.get("examples") or []
            if isinstance(ex, list) and ex:
                fh.write("- **예시**:\n")
                for e in ex[:3]:
                    fh.write(f"  - {e}\n")
            fh.write(f"- 출처: {', '.join(p.get('files', []))}\n\n")
        if free_form:
            fh.write("\n---\n## Free-form persona files (not numbered)\n\n")
            for ff in free_form[:30]:
                fh.write(f"- `{ff['file']}` — fields: {sorted(ff['fields'].keys())}\n")

    print(f"DONE candidates={len(candidates)} aggregated={len(persona_list)} accepted={len(accepted)} free_form={len(free_form)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
