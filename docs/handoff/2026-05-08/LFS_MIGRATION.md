# Git LFS Migration Guide (Optional, Future)

**Status**: NOT applied yet. Documented for when repo size becomes an operational issue.

## When to consider migrating

Current main has these tracked large files:
- `cpt_corpus.v3.jsonl` (~30MB)
- `sft_pairs.v3.jsonl` (~25MB)
- `sft_pairs.v2.jsonl`, `cpt_corpus.v2.jsonl` (~10-20MB each)
- `tokenizer_v4/tokenizer.json` (~11MB)
- `tokenizer_v4/merges.txt` (~3MB)

Total ≈ 100MB. GitHub allows up to 100MB per file (hard cap), 1GB per repo soft warning.

**Migrate to LFS when**: repo size exceeds 500MB OR a single file approaches 100MB.

## Migration commands (run when ready)

⚠ **WARNING**: LFS migration rewrites git history. Force-push required. Coordinate with all collaborators before running.

```bash
# 1. Install git-lfs (one-time per machine)
brew install git-lfs   # macOS
git lfs install

# 2. Migrate from current branch (main)
cd /Users/unoa/dalbitalba-train-data
git lfs migrate import \
  --include="*.jsonl,tokenizer_v4/tokenizer.json,tokenizer_v4/merges.txt,*.safetensors,*.bin"

# 3. Verify .gitattributes was created/updated
cat .gitattributes
# Expected:
# *.jsonl filter=lfs diff=lfs merge=lfs -text
# tokenizer_v4/tokenizer.json filter=lfs diff=lfs merge=lfs -text
# ...

# 4. Push (force, since history rewritten)
git push origin main --force-with-lease

# 5. Verify on GitHub: large files should now show "Stored with Git LFS"
gh repo view unoa-eng/dalbitalba-train-data --web
```

## Per-collaborator workflow after migration

Anyone with an existing clone must:

```bash
git fetch origin
git reset --hard origin/main
git lfs pull
```

Or re-clone fresh.

## Cost considerations

- **GitHub LFS free tier**: 1GB storage, 1GB/month bandwidth
- Above that: $5/month per 50GB pack
- For this repo's volume (~100MB), free tier is sufficient indefinitely

## Why this is deferred

1. Current repo size is well below GitHub's soft warning (1GB)
2. LFS migration requires force-push coordination
3. The benefit (download speed for fresh clones) is marginal at current scale
4. Adding `.gitattributes` LFS rules NOW would silently break `git add` for any future user who hasn't installed `git-lfs` — that's a worse UX than the slight bloat

**Recommendation**: keep current state. Re-evaluate when repo crosses 500MB.

---

Generated 2026-05-08 as part of Phase 5.9 final cosmetic cleanup pass.
