# Training Recipe Validation

기준 시각: 2026-04-24 (KST)

## 목표

- raw crawl 원천 데이터와 생성물의 간극을 줄인다.
- 게시글/댓글 문맥, 짧은 턴, reply 구조, 감정톤, 초성/은어 분포를 최대한 보존한다.
- RunPod 비용을 줄이되, 학습 품질을 해치지 않는 범위에서만 절감한다.

## 공식 자료 근거

1. Hugging Face Transformers bitsandbytes 문서
   - 4bit 학습에서는 `NF4`, `bf16`, `double quantization`이 권장된다.
   - 학습 시에는 `device_map="auto"`를 훈련 기본값처럼 쓰지 않는 편이 맞다.
   - 출처: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes

2. Hugging Face TRL SFTTrainer 문서
   - adapter 학습은 보통 더 높은 learning rate, 대략 `1e-4`를 쓴다.
   - completion-only loss와 packing은 짧은 샘플이 많은 SFT에서 특히 중요하다.
   - 출처: https://huggingface.co/docs/trl/main/sft_trainer
   - 출처: https://huggingface.co/docs/trl/reducing_memory_usage

3. Hugging Face PEFT LoRA 문서
   - QLoRA 스타일에서는 attention/MLP 선형층 전체를 대상으로 잡는 전략이 일반적이다.
   - 현재 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`는 타당한 범주다.
   - 출처: https://huggingface.co/docs/peft/main/en/developer_guides/lora

4. NVIDIA NeMo QLoRA 가이드
   - QLoRA는 LoRA보다 느리지만 메모리 절감이 크다.
   - QLoRA 논문 계열 레시피로 `1e-4` 전후 LR, 작은 batch, 전 선형층 LoRA를 권한다.
   - 출처: https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/qlora.html

5. RunPod 공식 문서
   - LLM fine-tuning은 원칙적으로 A100/H100급이 권장된다.
   - 다만 48GB 계열도 QLoRA 조건에서는 후보가 될 수 있다.
   - 출처: https://docs.runpod.io/pods/choose-a-pod
   - 출처: https://docs.runpod.io/references/gpu-types

6. NVIDIA 공식 GPU 스펙
   - A100 PCIe 80GB는 80GB HBM2e와 높은 메모리 대역폭을 제공한다.
   - L40S는 48GB 메모리와 높은 Tensor 성능을 제공한다.
   - 출처: https://www.nvidia.com/en-us/data-center/a100/
   - 출처: https://www.nvidia.com/en-us/data-center/l40s/

## 현재 데이터 분포 점검

### CPT

- `cpt_corpus.jsonl`: 53,341건
- `kind`: post 10,813 / comment 42,528
- 문자 길이 p50: 26
- 문자 길이 p90: 87
- 문자 길이 p95: 126
- 문자 길이 p99: 280
- 최대 길이: 1,467

### SFT

- `sft_pairs_v2.jsonl`: 24,000건
- `pair_type`: comment 16,800 / post 7,200
- `task_type`: root_comment 9,600 / reply_comment 7,200 / post_from_title 4,800 / post_continue 2,400
- total char p50: 133
- total char p90: 306
- total char p95: 363
- total char p99: 521
- total char max: 1,527
- output char p50: 29
- output char p90: 103
- output char p95: 154
- output char p99: 302

### CAI 혼합

- `cai_pairs.filtered.jsonl`: 1,743건
- 코드상 `CAI_RATIO=0.33`이지만 실제 파일 수 제한 때문에 현재 canonical SFT 대비 실효 비중은 약 6.8% 수준이다.
- 따라서 현재 학습은 raw-domain 스타일이 주축이고, CAI는 보조 안정화 수준으로만 작동한다.

## 검증 결론

### 유지

- `MAX_SEQ_LEN=1024`
  - 현재 데이터 p99 기준으로 과도하게 짧지 않고, 긴 꼬리도 어느 정도 수용한다.
  - 더 크게 늘리면 비용 증가 대비 실익이 제한적이다.

- `BATCH_SIZE=1`, `GRAD_ACCUM=16`
  - QLoRA 단일 GPU에서 무난한 출발점이다.
  - 공식 자료의 작은 batch 권고와도 맞다.

- `CPT_NUM_EPOCHS=1`
  - continued pretraining은 과적합보다 도메인 적응이 우선이고, 현재 corpus 규모에서 1 epoch가 비용 대비 적절하다.

- `SFT_NUM_EPOCHS=2`
  - 24k 규모, 짧은 샘플 중심 분포에서 3 epoch는 스타일 과적합 위험이 더 크다.
  - 2 epoch가 더 보수적이고 목표 친화적이다.

### 변경

- `device_map="auto"` 제거
  - Hugging Face 문서상 훈련 기본값으로 권장되지 않는다.
  - 단일 GPU 학습에서는 명시하지 않는 편이 더 안전하다.

- `SFT_LR=1e-4`
  - 현재 `5e-5`는 adapter 학습 기준으로 다소 보수적이다.
  - 공식 자료를 종합하면 `1e-4`가 더 타당하다.
  - 이 값은 자료 기반 추론이며, 실제 loss/eval이 악화되면 다음 반복에서 낮춘다.

## GPU 전략

1. 1순위: `NVIDIA A100 80GB PCIe`
   - RunPod 공식 권장과 메모리 대역폭 측면에서 가장 학습 친화적이다.

2. 2순위: `NVIDIA L40S`
   - 48GB로 QLoRA 10.7B에는 메모리상 충분하다.
   - 다만 학습 전용 관점에서는 A100보다 보수적으로 본다.

3. 3순위: `NVIDIA RTX 6000 Ada Generation`
   - 48GB 대안 카드.

## 다음 반복 실행 조건

- 로컬 문법/환경 검증 통과
- train launcher가 fallback GPU 목록을 자동 처리
- pod 준비 상태를 1분 단위로 감시
- 5분 동안 `uptimeSeconds=0` / `pod not ready`가 유지되면 즉시 중지 후 재시도
