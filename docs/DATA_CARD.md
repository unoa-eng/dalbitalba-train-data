# DATA_CARD — dalbitalba-train-data v3.2

> NeurIPS *Datasheets for Datasets* (Gebru et al., 2021) 양식에 맞춰 작성된 단일 데이터셋 카드.
> 본 카드는 dalbitalba 한국어 성인 커뮤니티 라이팅 스택의 학습용 코퍼스(6종)를 통합 기술한다.
>
> - **Version**: v3.2 (cycle-3 critical fixes: eval thread holdout, meta sync)
> - **Date**: 2026-05-12
> - **Maintainer**: unoa-eng `<mapdrawer2@gmail.com>`
> - **Repo**: `dalbitalba-train-data`
> - **Related**: [`README.md`](../README.md) · [`TRAINING_DESIGN_V3.md`](../TRAINING_DESIGN_V3.md) · [`PAPER_GRADE_ANALYSIS_20260507.md`](./PAPER_GRADE_ANALYSIS_20260507.md)

---

## 0. 데이터셋 요약 (Summary Table)

| 파일명 (file) | 행수 (rows) | 단계 (stage) | 핵심 필드 (key fields) | 용도 (use) | SHA256 |
|---|---:|---|---|---|---|
| `cpt_enriched.jsonl` | 48,247 | CPT Phase 1 | `text`, `kind`, `source_id`, `view_bucket`, `comment_bucket` | Continued Pre-Training (구조 마커 enriched) | `68a60f9e5832346f4bee1b7f91c212690384c6fa46ebac58812f795a5127d7ed` |
| `cpt_corpus.v3.jsonl` | 41,576 | CPT Phase 2 | `text`, `kind`, `source_field`, `length_bucket` | CPT short-row style_signal 정책 적용본 | `87f6aa1b8219f5d928d0690d3bfaa174ae4121f6f9502e146b08a0ca543aa34a` |
| `sft_thread_conditioned.jsonl` | 10,245 | SFT (train) | `instruction`, `input`, `output`, `depth`, `root_id`, `parent_id`, `persona_id`, `loss_weight` | Thread-conditioned SFT (T2 69.7% / T3 30.3%) | `79a7c00fd5d558aabb351b704d7e108cfb0adaf1ce82f1a9c7c22e3e8be85ff5` |
| `sft_thread_conditioned.eval.jsonl` | 322 | SFT (held-out) | 위와 동일 | SFT 내부 평가 (held-out, cycle-3 eval thread holdout 후) | `02d2d623b28007aa88834444d47f54656d5e3477173258c077604669d2a7c6e0` |
| `orpo_pairs.jsonl` | 1,472 | ORPO/CAI | `prompt`, `chosen`, `rejected`, `reason` | CAI/ORPO 선호쌍 (현 사이클 `ORPO_EPOCHS=0`이라 미사용) | `fae20a8343cbb4ce6e0d0f591c5ba29b4cf77d26557f762a5cd54e9bfb47a211` |
| `val_set.v3.jsonl` | 119 | Validation | `post_title`, `post_body_excerpt`, `parent_comment`, `target_comment`, `thread_key`, `depth`, `task_type` | 외부 평가셋 (post-leak-removal + thread-holdout) | `1deffc8bc4cf4060df3d4fb2c43f976f982936521459ff8f72cb720fa17ff0ef` |

총 행수: **101,880** (CPT 89,823 + SFT 10,567 + ORPO 1,472 + Val 119).

---

## 1. Motivation — 동기

### 1.1 왜 만들어졌는가
본 데이터셋은 한국어 성인 커뮤니티(밤문화) 도메인에서 사람과 구분하기 어려운 자연스러운 댓글/스레드 생성기 학습을 위해 구축되었다. 일반 instruction-tuned LLM(예: Qwen3-Instruct, GPT-4o)은 한국어 모바일 구어체, 초성 축약(ㅋㅋㅋ, ㅇㅇ, ㄱㅅ), 야간 트래픽 특유의 톤, 그리고 도메인 슬랭에 대한 분포를 학습하지 못해 *AI 같다*는 휴리스틱에 쉽게 잡힌다. 이를 해결하기 위해 (a) raw crawl 기반 CPT 코퍼스로 도메인 어휘 분포를 주입하고, (b) 스레드 컨디셔닝 SFT로 부모-자식 댓글 관계와 페르소나를 학습하며, (c) ORPO/CAI 선호쌍으로 *과도하게 정중한/AI 같은 톤* 회피를 시도한다.

### 1.2 누가 만들었는가
- **주 작성자**: unoa-eng 1인 (mapdrawer2@gmail.com), 부속 dalbitalba 라이팅 스택 연구·서비스 운영자.
- **자동화 보조**: oh-my-claudecode (OMC) 멀티 에이전트 오케스트레이션 (Claude Opus 4.7, Codex, Gemini) — 코퍼스 정제·PII 스캐닝·leak 검출 파이프라인을 보조하였으나 의사결정/라벨링 기준은 작성자가 정의.
- **소속**: 비소속 개인 연구자(independent researcher).

### 1.3 자금 출처
외부 자금(grant)이 없는 self-funded 프로젝트이며, RunPod GPU 시간 비용은 운영자 개인 결제로 충당되었다.

### 1.4 기타 의견
본 데이터셋은 *연구·공정성 분석·튜링테스트형 사람-AI 식별 벤치* 목적으로 설계되었으며, 상업적 배포 또는 생산 시스템 배포를 위한 ToS 정합성은 확보되어 있지 않다.

---

## 2. Composition — 구성

### 2.1 인스턴스가 무엇을 표현하는가
모든 인스턴스는 한국 성인 커뮤니티 `cb2_밤문화이야기` (queenalba 사이트, 익명 게시판) 의 게시글 또는 댓글이다. 인스턴스 종류는 두 가지: (1) `post` — 제목+본문 결합 텍스트, (2) `comment` — 단일 댓글 또는 (부모 댓글 + 대상 댓글) 페어.

### 2.2 인스턴스 개수
- CPT Phase 1 (`cpt_enriched.jsonl`): 48,247 rows (post + comment 혼합, `[조회수:X|댓글:Y]` 구조 마커 prepend).
- CPT Phase 2 (`cpt_corpus.v3.jsonl`): 41,576 rows (short-row 비율 보정 후 `style_signal` 정책 적용본).
- SFT train (`sft_thread_conditioned.jsonl`): 10,245 페어 — T2 root reply 7,141건(69.7%) + T3 deep reply 3,104건(30.3%).
- SFT eval held-out (`sft_thread_conditioned.eval.jsonl`): 322 페어 (cycle-3 eval thread holdout 후; 원 1,139에서 817행 제거).
- ORPO/CAI pairs (`orpo_pairs.jsonl`): 1,472 chosen/rejected 쌍.
- External validation (`val_set.v3.jsonl`): 119 페어 (B3 leak 제거 + thread-holdout 후).

### 2.3 필드 (schema)
- **CPT**: `text`, `kind`(post/comment), `source_id`, `views`, `comment_count`, `view_bucket`, `comment_bucket`, `length_bucket`, `source_field`.
- **SFT**: `instruction`(post 컨텍스트), `input`(parent + depth + persona tag), `output`(target reply), `depth`(1/2+), `root_id`, `parent_id`, `persona_id`(p-001~p-NNN), `loss_weight`(0~1).
- **ORPO**: `prompt`, `chosen`, `rejected`, `kind`, `source_run_chosen`, `source_run_rejected`, `reason`.
- **Val**: `post_title`, `post_body_excerpt`, `parent_comment`, `target_comment`, `thread_key`, `depth`, `task_type`, `length_bucket`, `source_id`, `loss_weight`.

### 2.4 도메인 통계 (domain metrics, on training corpus)
- 초성 축약(ㅋ/ㅇ/ㄱ 등) 출현 비율: **30.3%**
- 웃음 토큰(ㅋㅋ, 하하 등): **13.4%**
- 울음/우는 표현(ㅠㅠ, ㅜㅜ): **8.3%**
- 물음표 사용: **26.5%**
- 댓글 평균 길이: **48자** (p50 = 30자, 모바일 한 손 입력 분포)

### 2.5 데이터셋이 모든 가능한 인스턴스를 포함하는가
**아니다.** 단일 커뮤니티의 2개월(2026-01-01 ~ 2026-02-28) 구간만 포함하며, 광고로 분류된 게시물 및 미성년 근접 텍스트는 제외되었다(§4 참조). 따라서 모집단(전체 한국어 성인 커뮤니티)의 표본일 뿐이다.

### 2.6 missing / noisy / 라벨
- `comment_bucket`, `view_bucket`은 휴리스틱 구간화(소수/보통/활발)로, 라벨 노이즈가 존재한다.
- `persona_id`는 작성자 ID anonymization 후 클러스터 기반 자동 부여(K=N 클러스터)이며 정답 라벨이 아니다.
- `loss_weight`는 길이/품질 휴리스틱에 기반한 weak label.

### 2.7 데이터 분할
- Train (CPT+SFT): `cpt_enriched.jsonl` + `cpt_corpus.v3.jsonl` + `sft_thread_conditioned.jsonl`.
- Held-out internal: `sft_thread_conditioned.eval.jsonl`.
- External validation: `val_set.v3.jsonl` — 원본 2,491개 중 CPT 누수(leak) 2,074개를 B3 패치로 제거 후 417개 → cycle-2 A10 thread-holdout 적용으로 **119개** 보존.
- SFT eval held-out: `sft_thread_conditioned.eval.jsonl` — 원 1,139개에서 cycle-3 eval thread holdout(SFT train과 root_id 겹치는 817행 제거)으로 **322개** 보존. 잔여 root_id 교집합 = 0 검증.

### 2.8 중복 제거
- MinHash LSH (perms=128, shingle=5) 적용.
- 정확 중복(exact dup): **0건** 확인.
- 근접 중복(near-dup, Jaccard ≥ 0.85): 사전 제거됨.

---

## 3. Collection Process — 수집 과정

### 3.1 어떻게 수집했는가
공개 웹 크롤링 — queenalba 사이트의 cb2 게시판에서 익명 게시글/댓글을 HTTP GET 기반 스크레이퍼로 수집. 로그인이 필요한 영역은 수집하지 않았으며, robots.txt 및 rate-limit(1~2 req/s)을 준수하였다.

### 3.2 수집 시점
**2026-01-01 ~ 2026-02-28** (약 2개월, 야간 시간대 트래픽 비중 우세). 원본 raw HTML은 `archive/raw-crawl/` (로컬 보관, 미배포)에 보존.

### 3.3 동의 (consent)
원 게시자(익명 사용자)로부터 명시적 사전 동의를 받지 않았다. 모든 콘텐츠는 사이트 정책상 익명 작성된 공개 텍스트로, ToS 상 공개 게시 콘텐츠의 합리적 사용(reasonable use) 범위에서 연구 목적으로 수집되었다. 다만 본인 동의 없는 게시 콘텐츠 사용에 따른 잠재 위험을 인지하며, 따라서 본 데이터셋은 **재배포 금지 / 연구 내부용**으로 한정한다(§6 참조).

### 3.4 수집자
unoa-eng 단독, 사이트와의 사전 협의는 없었음.

### 3.5 수집 도구
Python `httpx` + BeautifulSoup 기반 자체 스크레이퍼(미공개). User-Agent에 연구 목적·연락처 명시.

### 3.6 sampling
수집 구간 내 게시물은 시간순 전수 수집(census)이며 통계적 sampling은 적용하지 않았다. 단 §4 전처리에서 광고·미성년 근접 텍스트는 강제 제외되었다.

---

## 4. Preprocessing / Cleaning / Labeling — 전처리·정제·라벨링

### 4.1 PII 비식별 (de-identification)
다음 카테고리에 대해 정규식 + 휴리스틱 기반 마스킹을 수행하였다:

| 카테고리 | 처리 | 잔존 검출 (post B4 patch) |
|---|---|---|
| 전화번호 (`phone_like`) | `[REDACTED-PHONE]` 치환 | **0** |
| 이메일 | `[REDACTED-EMAIL]` | 0 |
| URL (외부 광고 링크) | `[REDACTED-URL]` 또는 row drop | 0 |
| 주민등록번호 (RRN) | `[REDACTED-RRN]` | 0 |
| 계좌번호 (account) | `[REDACTED-ACCT]` | 0 |

5종 데이터셋(`cpt_enriched`, `cpt_corpus.v3`, `sft_thread_conditioned`, `sft_thread_conditioned.eval`, `val_set.v3`) 모두에서 **`phone_like = 0`**을 B4 패치 이후 확인하였다.

### 4.2 광고 필터링 (`is_promo_v2`)
키워드 + URL 휴리스틱 + 길이 분포 기반 광고 분류기로 광고성 게시물을 제거. 단 본 필터는 **공격적(aggressive)** 으로 작동하여, 정상 댓글(예: 특정 업소·인물명 언급) 일부가 over-removal되는 trade-off가 존재한다(MEMORY: dalbitalba GAP diagnosis 참고).

### 4.3 미성년 근접 (minor-proximity)
미성년자 관련 표현 휴리스틱으로 스캔 후 검출된 **1건**을 수동 검토하여 제거.

### 4.4 누수 제거 (CPT ↔ val leak / SFT eval thread holdout)
- B3 패치(2026-05, cycle-2): val_set 원본 2,491개 중 CPT 코퍼스에 정확 일치하는 2,074개(83.3%)를 제거 → **417개** → cycle-2 A10 thread-holdout으로 **119개** 잔존. 외부 검증 점수 인플레이션 차단.
- cycle-3 C1 패치(2026-05-12): `sft_thread_conditioned.eval.jsonl` (SFT 내부 평가셋) 에서 SFT train(`sft_thread_conditioned.jsonl`) 과 root_id가 겹치는 817행 제거 → **322개** 잔존. H2(Turing pass-rate) thread leakage 차단. root_id 교집합 = 0 검증.

### 4.5 구조 마커 enrichment (CPT Phase 1)
`cpt_enriched.jsonl`은 각 row 앞에 `[조회수:X|댓글:Y]` 구조 마커를 prepend하여 인기도 시그널을 부여. Phase 2(`cpt_corpus.v3.jsonl`)는 짧은 댓글(<30자) 비율을 보정하고 `style_signal` 정책을 적용하여 톤 일관성을 강화.

### 4.6 raw data 보존
원본 잡음 포함 raw 텍스트는 `*.pre-pii-removal.bak`, `*.pre-leak-removal.bak`로 로컬 보존(미배포). 외부 공유 산출물에서는 raw 포함하지 않는다.

---

## 5. Uses — 사용

### 5.1 본 데이터셋이 이미 사용된 작업
- Qwen3-8B/14B 베이스 모델의 CPT → SFT → (옵션) ORPO 학습 (RunPod).
- 3-judge blind eval + 사람-AI 튜링테스트형 식별 평가.
- 도메인 톤 분포 분석(논문급 데이터 감사 — `PAPER_GRADE_ANALYSIS_20260507.md`).

### 5.2 권장 사용 (recommended)
- 한국어 모바일 구어체 / 익명 커뮤니티 톤 모델링 연구.
- 사람-AI 텍스트 식별 벤치마크의 *AI side* 생성기 학습.
- PII 마스킹·도메인 누수 제거 파이프라인의 재현 사례 연구.

### 5.3 권장하지 않는 사용 (not recommended)
- **상업적 챗봇/생성 서비스 배포** — 원 사이트 ToS 검증 미완.
- **개인 식별 시도** — 모든 PII는 마스킹되었으나, 좁은 시간·커뮤니티 범위 특성상 사회공학적 역추론 위험이 존재.
- **성적·유해 콘텐츠 생성을 목적으로 한 downstream fine-tune.**
- **타 도메인(뉴스, 기술 블로그, 공식 문서) 일반화** — 본 코퍼스는 야간 성인 커뮤니티 분포에 강하게 편향됨.
- **미성년 사용자 대상 시스템.**

### 5.4 향후 5-task 디자인과 실제 사용 격차
`TRAINING_DESIGN_V3.md`는 본래 5개의 SFT task(T1 post-write, T2 root reply, T3 deep reply, T4 thread-summary, T5 persona-conditioned write)를 설계했으나, **현 v3.1 사이클에서 실제 학습된 SFT는 T2(69.7%) + T3(30.3%) 두 task에만 집중**되어 있다. T1/T4/T5는 후속 v3.2 사이클에서 확장 예정이며, 현 데이터셋만으로 5-task 전체 일반화 성능을 평가해서는 안 된다.

---

## 6. Distribution — 배포

### 6.1 배포 대상
**비배포 (do-not-redistribute)**. 본 데이터셋은 `dalbitalba-train-data` 사설 저장소 내부에 한정 보관하며, 외부 공개·미러링·Hugging Face Hub 업로드를 수행하지 않는다. 학습된 어댑터(adapter)는 `HF_ADAPTER_REPO`에 private 업로드되며, 원본 코퍼스는 동행 배포되지 않는다.

### 6.2 라이선스
- **연구·내부용 한정 (research-only, non-commercial)**.
- 재배포(redistribution)는 원 사이트 ToS 검증 및 작성자 동의 절차가 필요하며, 현 시점에서는 **금지**.
- 코드(전처리·학습 스크립트)는 별도 OSS 라이선스(MIT) 적용 — 데이터와 분리.

### 6.3 IP / ToS
원 사이트(queenalba) ToS 및 한국 정보통신망법, 개인정보보호법, 저작권법에 따른 제3자 권리가 존재할 수 있다. 사용자는 본 데이터셋을 사용하기 전 본인의 사법 관할권에서의 합법성을 직접 확인해야 한다.

### 6.4 export control
해당 없음.

---

## 7. Maintenance — 유지보수

### 7.1 누가 유지·관리하는가
**unoa-eng `<mapdrawer2@gmail.com>`** 1인. 별도 maintainer 팀은 없다.

### 7.2 연락처
GitHub repo issue tracker (private) 또는 이메일 직접 연락.

### 7.3 erratum
- **v3 → v3.1** (2026-05-12): B3 leak removal patch (val 2,491→417), B4 PII residual patch (phone_like→0), audit close-out.
- **v3.1 → v3.2** (2026-05-12): cycle-3 C1 eval thread holdout (sft eval 1,139→322, intersect=0); val_set.v3 meta sync (rows 417→119, SHA 2589...→1def...); sft eval SHA 갱신 (2a23...→02d2...); prelaunch fail-closed on revision (C3).
- 이전 erratum은 `docs/BRANCH_MERGE_AUDIT_20260512.md`, `docs/PR3_VERIFICATION_REPORT_20260507.md` 참조.

### 7.4 업데이트 정책
주요 학습 사이클마다 마이너 버전 증가(v3.1 → v3.2). raw crawl 시간 구간 확장 시 메이저 버전 증가(v3 → v4). 본 카드의 SHA256 hash와 행수는 매 버전마다 재기록한다.

### 7.5 데이터 보존
원본 raw HTML과 `.pre-*` backup 파일은 maintainer 로컬에 무기한 보존하나, 외부 공유는 하지 않는다.

### 7.6 contribution
외부 contribution은 받지 않는다(데이터 ToS 리스크). 코드 contribution은 OSS 라이선스 범위 내에서 별도 채널로 받는다.

---

## 8. Known Biases & Limitations — 알려진 편향과 한계

본 카드는 솔직한 결함 공개를 우선한다:

1. **단일 커뮤니티 편향**: queenalba `cb2_밤문화이야기` 단일 보드만 포함. 타 커뮤니티(디시, 더쿠, 인스티즈 등) 일반화 불가.
2. **2개월 시간 범위 편향**: 2026-01 ~ 02 시즌 특수 어휘(연초/설/밸런타인 등)가 과대 표상될 수 있다.
3. **야간 트래픽 편향**: 본 보드 특성상 22:00~03:00 KST 작성 댓글이 과반. 주간 어휘 분포가 과소.
4. **성인·성별 도메인 편향**: 성인 업소·노동 환경 어휘가 일반 한국어 분포 대비 과대. 일반 채팅 톤 생성에 직접 활용 시 부적절 어휘 누출 위험.
5. **모바일·한글 입력 편향**: 댓글 평균 48자, p50 30자로 모바일 단타 입력 분포에 편중. 장문 글쓰기 능력은 학습되지 않음.
6. **광고 필터 over-removal**: `is_promo_v2`가 공격적으로 작동하여 정상 댓글(특히 특정 업소·인물명 포함) 일부가 누락. 결과적으로 일부 슬랭(예: TC·밀빵·케어 등)의 분포가 underrepresentation.
7. **5-task 미커버**: 실제 학습 데이터는 T2/T3에 집중. T1/T4/T5 일반화 성능은 평가 불가.
8. **persona_id 약라벨**: 자동 클러스터링 기반이며 정답 페르소나가 아니다.
9. **PII 잔존 위험**: 정규식 기반 마스킹의 한계로 비정형 PII(별명, 지명+상호 조합)는 잔존 가능.
10. **ORPO 미적용**: 현 사이클 `ORPO_EPOCHS=0` 정책으로 `orpo_pairs.jsonl`은 보관용이며 학습 영향 없음.

---

## 9. Ethics & Legal

본 섹션은 README.md에서 cross-reference 대상이다.

- **합법성**: 본 코퍼스는 공개 게시 텍스트에서 추출되었으나, 원 작성자의 명시적 동의 없음. 따라서 *연구 내부용*에 한정한다.
- **PII 보호**: phone/email/URL/RRN/account 5종은 자동 마스킹 + B4 패치로 잔존 0 검증. 그러나 비정형 식별자는 여전히 잔존 위험이 있어, 어떠한 형태의 재식별(reidentification) 시도도 금지한다.
- **유해 콘텐츠**: 미성년 근접 1건 제거, 광고 일괄 제거. 단 성인 도메인 특성상 성적·자조적·차별적 어휘가 잔존하며, 본 데이터셋 또는 파생 모델을 통한 유해 콘텐츠 생성에 대한 책임은 사용자에게 있다.
- **takedown**: 자신이 작성한 게시물의 본 데이터셋 포함을 거부하는 원작자는 maintainer에게 연락 시 즉시 제거 처리한다.
- **사회적 영향**: 본 데이터셋은 *사람-AI 식별 연구*를 보조하는 목적이며, 이를 사람을 사칭하거나 기만하는 시스템에 활용하는 것은 본 카드의 의도에 반한다.

---

## 10. 변경 이력 (Changelog)

| 버전 | 날짜 | 변경 사항 |
|---|---|---|
| v3 | 2026-04-28 | 초안 — paper-grade 학습 설계 확정, MLX 로컬 실험 완료 |
| v3.1 | 2026-05-12 | B3 leak removal patch (val 2491→417), B4 PII residual zero, 본 DATA_CARD.md 작성 |
| v3.2 | 2026-05-12 | cycle-3 C1: sft eval thread holdout (1139→322, intersect=0); C2: val meta sync (419→119, SHA 갱신); C3: prelaunch fail-closed on BASE_MODEL_REVISION for paper8b/budget30 |

---

*문서 종료. 본 카드의 정확성에 대한 질문 또는 erratum 보고는 maintainer 연락처로 송부 바람.*
