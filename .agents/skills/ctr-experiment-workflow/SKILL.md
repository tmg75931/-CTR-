---
name: ctr-experiment-workflow
description: >-
  토스광고 CTR 예측 모델의 피처 추가, 모델 학습, 50:50 WLL 확률 보정, 검증 실험을
  체계적으로 수행하기 위한 표준 워크플로우 스킬입니다.
  사용자가 새로운 CTR 피처 개발, 모델 실험 설계, 리더보드 점수(AP + WLL) 개선,
  베이스라인 구축을 요청할 때 활성화하세요.
  단순 문서 수정이나 하네스 설정 변경에는 사용하지 마세요.
---

# CTR 모델 실험 및 검증 워크플로우

새로운 피처를 추가하거나 모델·후처리를 변경해 리더보드 점수(`AP 50% + WLL 50%`)를 개선할 때 아래 4단계 절차를 따릅니다.

## 1. 가설 수립 및 담당 서브에이전트 배정 (Hypothesis & Dispatch)
- 포착하려는 CTR 신호(지면 `inventory_id`, 광고셋 `l_feat_14`, 유저 시퀀스 `seq`, 정보영역 `feat_a~e`, 과거 인기도 `history_a`)를 정의합니다.
- `.agents/rules/orchestration-dispatch.md`의 소유권 매트릭스에 따라 적합한 담당 에이전트(`data-cv-architect`, `tabular-gbdt-engineer`, `seq-deepctr-engineer`, `ensemble-calibrator`)에게 작업을 위임합니다. (정형 트랙과 `seq` 트랙이 모두 필요하면 병렬 실행합니다.)

## 2. Colab-First 구현 및 표준 산출물 규격화 (Config & Implementation)
- 실험 파라미터(피처셋, 검증 분할, 샘플링 비율, 하이퍼파라미터, 시드)는 `configs/*.yaml`로 분리합니다.
- Colab 노트북(`notebooks/*.ipynb`)은 모든 코드 셀 뒤에 결과 해석용 마크다운 셀을 배치하며, 검증된 코드는 본선 코드 검증을 위해 `src/toss_ctr/` 모듈로 승격합니다.
- 모든 단일 모델은 앙상블 호환을 위해 표준 OOF(`row_id, clicked, raw_logit, pred_prob`) 및 Test(`ID, raw_logit, pred_prob`) 예측 파일을 저장합니다.

## 3. 독립 누수 감사 및 다각도 평가 (Audit & Evaluate)
- 구현 직후 `data-leakage-auditor` 서브에이전트로 시간축/폴드 누수 및 `50:50 WLL` 보정 수식을 감사하여 `CRITICAL` 결함이 0건인지 확인합니다.
- 동일한 Local Validation 위에서 `src/toss_ctr/evaluation/metrics.py`로 아래 지표를 분리해 비교합니다:
  1. **AP (Average Precision, 50%)**: 순수 순위 변별력 향상 여부 판단
  2. **WLL (50:50 Weighted LogLoss, 50%)**: 클래스 균형 가중치(`pos:neg = 50:50`), Prior Shift, Calibration 적정성 판단
  3. **Combined Score**: `0.5 * AP + 0.5 * (1 / (1 + WLL))`

## 4. 결과 기록 및 제출 무결성 검증 (Log & Verify)
- 실험 설정, Validation 지표(`AP`, `WLL`, `Score`), Public LB 점수를 기록하고 제출 파일(`1,527,298`행, `ID` 순서 일치, 결측 0건, `(0, 1)` 확률 범위) 무결성을 확인합니다.
