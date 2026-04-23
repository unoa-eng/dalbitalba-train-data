# 학습 고도화 오토파일럿 운영 기준

## 목표

최종 목표는 `C:\Users\mapdr\Desktop\queenalba-crawler\crawled-data-v2` 의 실제 글/댓글 분포와
학습 모델 생성물이 쉽게 구분되지 않을 정도로 간극을 줄이는 것이다.

단, "완벽"을 선언할 때는 감이 아니라 아래 근거가 필요하다.

- blind eval 에서 사람/AI 구분 난도가 높아질 것
- raw 대비 길이, 문장부호, 초성, 은어, 감정톤, 댓글 구조 분포가 가까워질 것
- 광고성/운영자성 노이즈를 사람 말투로 오인해 학습하지 않을 것

## 데이터 우선순위

1. 1차 원천: `C:\Users\mapdr\Desktop\queenalba-crawler\crawled-data-v2`
2. 2차 보조: `research/obsidian-export`
3. 3차 산출: `cpt_corpus.jsonl`, `sft_pairs_v2.jsonl`, `.state/thread_aware/*`

원천 raw 가 진실원본이다. Obsidian 은 검토/태깅/샘플 해설용 보조 레이어다.
기존 snapshot 데이터셋은 학습 입력이지만, raw 와 멀어지면 언제든 재생성 대상이다.

## 현재 핵심 원칙

- 게시글만 닮으면 안 되고 댓글 문맥과 reply 흐름까지 닮아야 한다.
- 단순 은어 사전이 아니라 초성, 감정 기호, 짧은 턴, 질문/반문, 생략 부호까지 같이 맞춰야 한다.
- 운영자 홍보 댓글, 전화번호, 오픈카톡, 반복 광고 템플릿은 최대한 제거하거나 문맥용으로만 제한한다.
- eval 은 post/comment 를 분리하지 않고 섞어버리면 왜곡되므로 kind 기준으로 층화한다.
- "개선"이라고 보고할 때는 항상 이전 실행 대비 근거를 같이 남긴다.
- GPU 는 비싸므로 raw 분석, 데이터 재생성, 정합성 검증, 프롬프트/샘플링 설계는 가능한 한 로컬에서 먼저 끝낸다.
- RunPod 는 "지금 GPU 를 켜야만 다음 증거를 얻을 수 있을 때만" 켠다.

## 오토파일럿 반복 루프

1. raw crawl 상태를 프로파일링한다.
   - `scripts/profile_raw_crawl.py`
   - 초성, 은어, 감정 표현, 길이 분포, 댓글 깊이, 광고 오염도를 확인한다.

2. thread-aware 데이터셋을 재생성한다.
   - `scripts/build_thread_aware_datasets.py`
   - root 댓글, reply 댓글, 게시글 맥락이 모두 살아 있는지 확인한다.

3. 기존 canonical 데이터셋보다 새 데이터셋이 목표에 더 맞으면 학습 입력 승격을 진행한다.
   - 필요 시 `cpt_corpus.jsonl`, `sft_pairs_v2.jsonl` 를 새 산출물 기준으로 교체하거나 branch 에 반영한다.
   - 변경 이유와 기대 효과를 진행 보고에 남긴다.

4. RunPod 에서 학습을 실행한다.
   - 현재 branch 를 기준으로 pod 를 띄운다.
   - 실패 pod 는 방치하지 말고 원인을 추적해 수정 후 재실행한다.

5. 학습 산출물을 Hugging Face 에 내구성 있게 올린다.
   - adapter repo id 를 확인한다.
   - 필요 시 `.env.local` 의 `HF_ADAPTER_REPO` 를 갱신한다.

6. stratified blind eval 을 실행한다.
   - `scripts/generate_samples.py`
   - `eval/make_eval_samples.py`
   - `eval/judge_3way.py`
   - `eval/native_eval_kit.py`

7. 생성물과 raw 분포를 비교한다.
   - 게시글/댓글 비율
   - 길이 분포
   - 질문/감탄/말줄임표 비율
   - 초성/은어/감정톤 빈도
   - reply 구조 유사성
   - judge 오판/불일치 샘플

8. 차이가 크면 원인을 분류하고 바로 다음 반복으로 넘긴다.
   - 데이터 필터 문제
   - 프롬프트/seed 문제
   - eval 샘플링 문제
   - 학습 파라미터 문제
   - 광고/중복 오염 문제

## RunPod 비용 통제 기준

### GPU 시작 전 게이트

아래가 모두 맞을 때만 새 pod 를 띄운다.

- 로컬 raw 분석과 데이터셋 생성이 끝났다.
- 기존 실행 대비 무엇이 달라졌는지 설명할 수 있다.
- 직전 실패 원인을 수정했다.
- 이미 살아 있는 pod 가 없다.
- 이번 실행이 목표 수렴에 실제로 기여한다.

### local-first 원칙

가능한 작업은 먼저 로컬에서 처리한다.

- raw 프로파일링
- 정합성 검증
- 데이터 필터링/재생성
- 샘플 몇 건의 포맷 검토
- env 검증
- GitHub/HF 상태 확인

GPU 는 아래처럼 진짜 필요할 때만 사용한다.

- 새 데이터셋으로 CPT/SFT 재학습
- adapter 를 얹은 생성 샘플 대량 생산
- blind eval 용 생성 단계

### 자동 중지 조건

아래 중 하나면 pod 를 바로 중지하거나 재시작 판단을 한다.

- 컨테이너 restart loop
- 로그 무출력 상태가 비정상적으로 길다.
- 잘못 설계된 eval/train 으로 판명됐다.
- 결과 업로드가 끝났고 더 돌릴 이유가 없다.
- 다음 단계가 GPU 없이 가능한 단계로 넘어갔다.

### 목표 달성 후 정책

아래가 충족되면 RunPod 는 더 이상 켜두지 않는다.

- 최신 adapter 와 eval 결과가 Hugging Face / GitHub 에 내구성 있게 남아 있다.
- blind eval 과 raw 분포 비교에서 현재 목표 수준에 도달했다.
- 다음 반복을 열어도 기대 이득보다 비용이 큰 상태다.

목표 달성 후에는 모든 pod 를 `EXITED` 상태로 유지하고, 이후에는 로컬 분석과 결과 정리만 계속한다.

## 승인 없이 계속 진행하는 범위

아래는 사용자 추가 승인 없이 계속 진행한다.

- raw 분석
- 데이터셋 재생성
- train/eval pod 재실행
- 실패 pod 중지 및 재시도
- Hugging Face repo 반영
- GitHub branch/pointer 업데이트
- 결과 요약과 리스크 보고

아래는 외부 하드 블로커가 생길 때만 사용자에게 바로 보고한다.

- 인증 만료
- quota/결제 문제
- 외부 서비스 장애
- 회복 불가능한 데이터 손실 징후

## 보고 원칙

- 진행 보고는 상태 변화가 있을 때마다 남긴다.
- "잘 되고 있다" 대신 무엇이 끝났고 무엇이 병목인지 구체적으로 적는다.
- 성능 향상 주장은 항상 수치 또는 샘플 근거와 같이 적는다.
