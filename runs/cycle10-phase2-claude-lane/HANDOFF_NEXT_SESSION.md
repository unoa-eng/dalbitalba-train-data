# HANDOFF — Phase 2 Claude-lane (2026-06-11 15:00 KST 기준)

세션 ee99a9ef 종료 시점 스냅샷. 이 문서만으로 새 세션이 이어갈 수 있다.
모든 서브에이전트/백그라운드 작업은 정지된 상태. 이 디렉토리가 진실의 원천
(/tmp 사본은 휘발 — 항상 여기 기준으로 작업하고 산출물도 여기로 백업할 것).

## 본질 목표 (확정된 user 결정들)
1. 24년 Q4 시작, exponential 성장한 사이트로 보이는 시뮬레이션 코퍼스
   (63k 1:1 재생산 아님 — 2026-06-09 redirect).
2. **초기 세팅(배포 전 backfill 코퍼스 전체)은 Claude가 직접 작성. 로컬
   LLM(boost-4)은 배포 후 운영(일 6-7글 incremental)용** — 2026-06-11 user 재확인.
   boost-4 로컬 full-run은 중단됨 (chunk 0 500개만 /tmp/dalbit_phase2_full에 남음, 폐기 가능).
3. 원천 크롤 양식 엄수 + 댓글-게시글 대응 + 조회수/추천/타이밍 메타데이터 고도화.
4. 묻지 말고 끝까지 (goal 명령 사용 중이었음).

## 베이스 데이터
- seed: `runs/cycle10-phase1-claude-direct/threads_v3.jsonl` (3,204 threads, 10,008 comments)
- 원천 분포 참조: `cpt_enriched.jsonl` (views/comment_count), 
  `runs/cycle10/data-representative-v1-clean/threads_flat/train.jsonl` (텍스트 — 차단문구/광고 필터 필요)

## 완료된 것
- **early 댓글 555개** (2024-10~2025-03): `bulk_out/../dalbit_phase2_early_merged.jsonl`
  (repo 루트의 `dalbit_phase2_early_merged.jsonl`, uniqueness 100%)
- **bulk 댓글 parts 00–23 완료**: `bulk_out/out00..23.jsonl` = 4,728개
  (입력: `bulk_parts/partNN.jsonl`, 키 = root_id + comment_index)
- **게시글 rework parts 00–03 완료**: `post_out/out00..03.jsonl` = 750/2,246
  (입력: `post_parts/partNN.jsonl`; need_title/need_body 플래그)
- **메타데이터 고도화 스크립트 검증 완료**: `scripts/dalbit_metadata_refine.py`
  (views 원천분포+성장ramp, likes 0-3 중심, 댓글타이밍 lognormal, crawledAt 6/11 00:30 스윕;
  위반 0/0/0 확인됨)
- **정규화기**: `scripts/dalbit_gen_normalize.py` — 호환자모→U+1100 변환(필수!),
  ᄏ/ᅲ run 원천분포 리샘플, 개행 주입(~37%), 띄어쓰기 붕괴(space_ratio 0.16 목표)
- 조립: `scripts/dalbit_assemble_v4.py`, 감사: `scripts/dalbit_v4_audit.py`

## 핵심 진단 결과 (2026-06-11 AUC 감사)
- raw Claude 댓글 vs 원천: AUC 0.984 → 정규화 후 0.966. 원인 분해:
  - 인코딩(원천은 ㅋㅋ/ㅠ가 전부 옛한글자모 U+1100, 호환자모 0%) → 정규화기로 해결
  - **존댓말율 gen 63% vs 원천 28%** ← 최대 잔여 갭. parts 00–23은 구 prompt(존댓말 과다)
  - ᄏ run: 원천은 2~6+ 분산 (gen은 91%가 딱 2개) → 정규화기 리샘플로 해결
  - 원천 풀에 구인광고/보도방 홍보글 존재 (우리 코퍼스 0개) — **사이트에 광고글 stream
    추가 여부는 user 미결정, 물어볼 가치 있음**
  - 주제 협소(시즌/한약/스토킹 템플릿 반복)는 corpus 설계 한계 — 부분적으로 post rework가 완화
- 주의: AUC-vs-원천은 "다른 사이트" 시뮬 특성상 주제어휘만으로도 높게 나옴.
  실질 게이트는 (a) 아티팩트 위생(인코딩/형식/중복), (b) HUMAN-EVAL 스타일 blind 판별
  (boost-4 기준 gate ≤55%, user 언급 ≤15% Turing도 있음 — 기준 재확인 필요).

## 남은 작업 (순서대로)
1. **bulk 댓글 parts 24–47 작성** (24 parts × ~197 = 4,725개). 반드시 **prompt v2** 사용:
   반말/단답 70%+, 존댓말 ≤30%, ㅋ 2~8 연속 다양, 구두점 ?!..~ 자유, "걍" 사용,
   중앙값 ~30자 짧게. (v2 전문은 git log에서 part24 launch 프롬프트 참조하거나 아래 요약 재구성)
   - 병렬 8 agents, 완료 시 보고: 개수/평균길이/존댓말율/중복
2. **게시글 rework parts 04–11** (~1,496개). 규칙: need_title→새 고유 제목(날짜-계절 정합),
   need_body→주제/핵심질문 유지하며 표현만 재작성, false→원본 그대로.
3. **register 재조정 패스**: parts 00–23 산출물(4,728)의 존댓말 비율을 63%→~28%로.
   존댓말 댓글 중 ~55%를 반말/단답으로 재작성 (agents, root_id+comment_index 보존).
4. **깨진글 댓글 재생성**: `scripts/dalbit_garbled_ids.json` 238 threads —
   post rework로 본문이 바뀌므로 그 글들의 댓글은 새 본문 기준으로 재작성 (딴지 댓글 orphan 방지).
5. **정규화**: 모든 댓글에 `dalbit_gen_normalize.py` 적용 (early 포함, post 본문/제목도 자모 변환 필요
   — normalize()를 게시글에도 적용하되 개행주입/space붕괴 파라미터는 본문용으로 약하게).
6. **조립**: `dalbit_assemble_v4.py /tmp/v4_pre.jsonl` → `dalbit_metadata_refine.py v4_pre v4_final`
   (assemble은 /tmp 경로의 out파일을 읽음 — repo 백업본으로 경로 수정 필요!)
7. **최종 감사**: dalbit_v4_audit.py (AUC), uniqueness(제목/본문/댓글 >95%), 타이밍 위반 0,
   commentCount==len(comments), 형식 diff (threads_v3 스키마: 문자열 숫자, author 비회원,
   댓글 "[N] " prefix — assemble이 재번호함).
   + 크로스파트 near-dup 오프닝 점검 ("변호사 명의 내용증명은"×8 류 — 발견 시 해당 댓글만 재작성)
8. **push**: threads_v4.jsonl + 감사 리포트 → `runs/cycle10-phase2-claude-lane/` 커밋.

## 환경 주의사항
- 같은 repo에서 **다른 claude 세션이 살아있는지 먼저 확인** (`ps aux | grep claude`,
  한달 묵은 세션이 같은 작업을 병렬로 밀다 충돌한 사고 있었음 — 6/10).
  63k expansion lane (phase2_prompt_pool/production_driver/autonomous_full_run)은 폐기 — 실행 금지.
- 메모리 16GB: MLX/대형 작업 전 free 확인. colima(`colima start`로 복구 가능)와
  Claude.app은 이 세션이 메모리 확보 위해 정지시킴.
- AUC 감사는 `.venv/bin/python` 사용 (시스템 python에 numpy 없음).
- 디스크: /tmp 산출물은 ENOSPC 사고 이력 — 단계마다 repo로 백업.

## 메모리 파일
`~/.claude/projects/-Users-unoa/memory/project_dalbitalba_phase2_inflight.md` 가 요약 보유.
이 핸드오프가 더 상세하므로 새 세션은 이 파일 우선.
