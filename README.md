# dalbitalba-train-data

Training, evaluation, RunPod, and research repository for the dalbitalba writing stack.

## Boundary

- `dalbitalba`: service repo for web, backend, Flutter, deployment, and production database code
- `dalbitalba-train-data`: datasets, RunPod launchers, blind-eval tooling, benchmark artifacts, and research archives

## Active layout

- root `*.jsonl`: curated corpora and seed datasets
- `train_*.py`, `chain_train.sh`: RunPod training entrypoints
- `scripts/launch_train_pod.py`: start a training pod
- `scripts/launch_eval_pod.py`: start an evaluation pod
- `scripts/generate_samples.py`: generate AI blind-test samples from the uploaded adapter repo
- `scripts/run_eval.sh`: end-to-end blind evaluation runner inside RunPod
- `eval/`: blind-set builder, three-judge runner, and report renderer
- `runs/`: Git-persisted outputs written back by RunPod jobs

## Imported artifacts

- `bench/`: local benchmark snapshots copied from the service repo
- `turing-test/`: manual blind quiz artifacts
- `research/obsidian-export/`: Obsidian-ready board and community research vault
- `archive/omni-audit/`: historical OMNI audit outputs
- `archive/runpod/`: local orchestration logs
- `archive/legacy-omni-audit/`: old scheduler and git-hook assets kept for reference

## RunPod flow

1. Launch training with `python scripts/launch_train_pod.py`.
2. Training uploads adapters to Hugging Face and pushes `runs/latest-train.json` plus `runs/train-run-*` branches back to GitHub.
3. Launch evaluation with `python scripts/launch_eval_pod.py`.
4. Evaluation generates samples, runs judges, renders reports, and pushes `runs/latest-eval.json` plus `runs/eval-run-*` branches back to GitHub.

## Bring-up on another environment

1. Clone this repository.
2. Copy `.env.local.example` to `.env.local`.
3. Fill the shared keys you may already have in `dalbitalba/apps/web/.env.local`:
   `RUNPOD_API_KEY`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
4. Add the train-repo-only keys:
   `HF_USERNAME`, `GITHUB_TOKEN`, and for local eval launches `HF_ADAPTER_REPO`.
5. Optionally set `NTFY_TOPIC`, `BASE_MODEL`, `GPU_TYPE`, and `CONTAINER_IMAGE`.
6. Run `python scripts/check_env.py --target train` or `python scripts/check_env.py --target eval`.
7. Launch with `python scripts/launch_train_pod.py` or `python scripts/launch_eval_pod.py`.

This means another machine can reconstruct the pipeline quickly, but resuming an old run
from the exact previous checkpoint still depends on whether the adapter or checkpoint was
already uploaded to Hugging Face or another persistent storage target.

## Service repo bridge

`unoa-eng/dalbitalba` can manually dispatch these workflows through its `Train Data Bridge`
workflow. The bridge keeps GPU and evaluation execution here while letting service-side
operators start a train or eval run from the app repository.

## Required GitHub configuration

Repository secrets used here:

- `RUNPOD_API_KEY`
- `HF_TOKEN`
- `HF_USERNAME` for training uploads
- `TRAIN_REPO_PUSH_TOKEN` for Git clone/push from inside RunPod
- `ANTHROPIC_API_KEY` for eval judging
- `OPENAI_API_KEY` for eval judging
- `NTFY_TOPIC` optional notification topic

Repository variables used here:

- `BASE_MODEL`
- `GPU_TYPE`
- `CONTAINER_IMAGE`

Additional service-repo secret:

- `dalbitalba` needs `TRAIN_REPO_DISPATCH_TOKEN` so its bridge workflow can dispatch
  these train-repo workflows.

If you are operating only through GitHub Actions, these values should be placed in
repository secrets and variables instead of a local `.env.local` file.

## Research note

`research/obsidian-export/` is preserved for manual analysis and board/community content review.
It is not production input and should not be fed into training without an explicit curation step.

## Source alignment check

The active training and evaluation jobs consume the curated snapshots in the repository root,
not the raw crawl directory directly. To verify that those snapshots still line up with the
current crawl source, run:

```bash
python scripts/validate_source_alignment.py --raw-dir "C:\Users\mapdr\Desktop\queenalba-crawler\crawled-data-v2"
```

This command compares:

- raw post bodies vs `cpt_corpus.jsonl`
- raw normalized comments vs `cpt_corpus.jsonl`
- raw post bodies vs `sft_pairs_v2.jsonl` post outputs
- raw reply chains vs `sft_pairs_v2.jsonl` comment instruction/output pairs

## Ethics & Legal

> 전체 데이터셋 datasheet는 [`docs/DATA_CARD.md`](docs/DATA_CARD.md) (NeurIPS *Datasheets for Datasets* 7 섹션 + 윤리·편향·SHA256 hash table)를 참조한다.

### Domain context
이 레포는 한국 성인 커뮤니티 "달빛알바"(공개 게시판 cb2_밤문화이야기)의 텍스트 데이터로 Qwen3-8B-Base를 도메인 적응시키는 연구 파이프라인이다. 학습 대상 텍스트는 사용자가 공개 게시판에 자발적으로 게시한 댓글/게시글이며, 광고/홍보성 콘텐츠는 is_promo_v2 필터로 제거되었다.

### 한국 법률 관계
- **성매매처벌법 제21조 (광고 알선 처벌)**: 이 프로젝트는 광고 알선 텍스트의 *생성* 또는 *알선*을 목적으로 하지 않는다. 광고 표현은 학습 데이터에서 적극 제거(`is_promo_v2`, `clean_ad_spam.py`)되었고, 모델은 도메인 화법(존댓말/은어/이모지 분포)의 학문적 분석을 위한 것이다.
- **개인정보보호법**: PII(전화/이메일/URL/주민번호/계좌) 정규식 필터 + B4 패치로 절단형/해외형 전화 패턴까지 검출하여 마스킹(`[REDACTED-PHONE]`). 모든 데이터셋 `phone_like=0` 검증 (paper8b-readiness summary).
- **저작권**: 게시판 텍스트는 공개 게시 상태이며 비상업적 연구 목적의 fair use 범주로 다룬다. 모델 산출물의 상업적 배포는 별도 라이선스 검토 필요.

### IRB 면제 근거
- 데이터: 공개 웹 게시판의 비식별 텍스트
- 식별 가능성: PII 필터 + 마스킹으로 제거
- 연구 목적: 학술 연구(도메인 적응 LLM 평가)
- 인간 대상 직접 실험 부재
- 따라서 IRB 심의 면제(`exempt`) 범주에 해당. 향후 학회 제출 시 면제 신청서를 작성하여 첨부 예정.

### 미성년자 보호
- 수집 단계 미성년 근접 콘텐츠 필터: `PAPER_GRADE_ANALYSIS_20260507.md §3.1`의 minor-proximity drop 1건 기록
- 모델 출력 시 미성년 관련 prompt에 대한 fallback 정책은 별도 시스템 프롬프트(서비스 적용 시)에서 처리

### 콘텐츠 정책 정합
- Anthropic Acceptable Use Policy: 본 프로젝트의 *연구 분석* 목적은 AUP 위반이 아니다. 다만 Claude judge가 도메인 콘텐츠를 human/AI 분류할 때 Anthropic models는 FN=1.00 편향이 관측됨(`docs/PAPER_GRADE_ANALYSIS_20260507.md §5.1`). 평가는 J classifier v2 + GPT 보완.
- OpenAI usage policy: GPT-4 judge 호출은 데이터 분석/평가 목적이며 콘텐츠 생성 목적이 아니다.

### Data lineage
- 수집: queenalba-crawler/crawled-data-v2 (Windows local), 2026-01~02
- 필터: is_promo_v2, AD_RE, dedup_minhash, PII regex(v2+ extended), val/train leak removal v3
- 데이터 카드 상세: [`docs/DATA_CARD.md`](docs/DATA_CARD.md)

### Reporting
PII / 윤리 / 콘텐츠 정책 위반 발견 시 mapdrawer2@gmail.com 또는 GitHub issue로 보고 바람.
