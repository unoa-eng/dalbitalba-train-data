# Obsidian Scope Policy

## Decision

Obsidian data is in scope for round2, but it is not a direct raw-text training
source by default.

The accepted path is:

- Keep `research/obsidian-ref/` as human-readable reference notes.
- Keep `research/obsidian-export/` as auditable exported examples.
- Use `runs/round2-obsidian-synthesis/persona-30-extracted.json` as the
  curated metadata bridge for SFT persona conditioning and phase6 persona
  evaluation.
- Do not inject raw Obsidian markdown body text into prompts or targets unless a
  separate opt-in experiment documents provenance, sampling ratio, and eval
  impact.

## Current Coverage

- Reference notes: 18 markdown files under `research/obsidian-ref/`.
- Exported examples: 510 markdown files under `research/obsidian-export/`.
- Persona metadata: 30 accepted personas in
  `runs/round2-obsidian-synthesis/persona-30-extracted.json`.

## Guardrails

- Raw source DB and crawl data remain the primary training truth.
- Obsidian-derived persona metadata must preserve the same train/inference
  prompt schema.
- The local verifier fails if the Obsidian reference/export/persona scope is
  missing or materially incomplete.
- Cycle-1 synthesized personas are recorded as explicit metadata for 30-ID
  coverage; they must remain fully populated and auditable.

## Operational Check

Run:

```bash
python scripts/local_verification_loop.py --strict --profile paper8b
```

The report's `Obsidian Scope` section must show nonzero reference/export counts
and `Persona count: 30` before paid RunPod launch.
