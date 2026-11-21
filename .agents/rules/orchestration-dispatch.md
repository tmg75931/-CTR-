---
trigger: always_on
---
# 오케스트레이터 업무 배분 및 멀티에이전트 협업 규칙

메인 코디네이터(Orchestrator)는 컨텍스트 비대화를 막고 병렬 개발 효율을 높이기 위해 직접 장문의 구현 코드를 작성하는 대신, 아래 **5대 전문 서브에이전트**에게 명확한 파일 경계(Single-Writer Ownership)로 업무를 배분합니다.

## 1. 서브에이전트 R&R 및 독점 파일 소유권 (Write Scope)

동시 실행 시 파일 덮어쓰기 충돌을 방지하기 위해 각 에이전트는 지정된 소유 경로만 수정합니다.

| 서브에이전트 (`TypeName`) | 핵심 담당 업무 | 독점 쓰기 경로 (Write Ownership) |
| :--- | :--- | :--- |
| `data-cv-architect` | EDA 요약 해석, PyArrow/Polars/DuckDB 경량화 캐싱, Local CV 폴드 확정 | `src/toss_ctr/data/**`, `notebooks/01_*.ipynb`, `notebooks/02_*.ipynb`, `configs/data_*.yaml`, `configs/cv_*.yaml`, `tests/unit/test_data_*.py` |
| `tabular-gbdt-engineer` | 117개 정형 파생 피처(교호작용·OOF 타겟 인코딩·인기도), LightGBM/CatBoost/XGBoost | `src/toss_ctr/features/`(`seq_*` 제외), `src/toss_ctr/models/gbdt_*.py`, `notebooks/03_*.ipynb`, `notebooks/04_*.ipynb`, `configs/gbdt_*.yaml`, `tests/unit/test_features_*.py`, `tests/unit/test_gbdt_*.py` |
| `seq-deepctr-engineer` | `seq` 시퀀스 파싱·토크나이징·요약 피처, Colab GPU용 PyTorch DeepCTR(DIN/GRU/Transformer+DCN) | `src/toss_ctr/features/seq_*.py`, `src/toss_ctr/models/deep_*.py`, `src/toss_ctr/models/nn_*.py`, `notebooks/05_*.ipynb`, `notebooks/06_*.ipynb`, `configs/deep_*.yaml`, `tests/unit/test_seq_*.py`, `tests/unit/test_deep_*.py` |
| `ensemble-calibrator` | GBDT+DeepCTR OOF Log-odds 블렌딩, `50:50 WLL` 확률 보정, 추론 CLI 및 제출 무결성 검증 | `src/toss_ctr/models/calibration.py`, `src/toss_ctr/inference/**`, `scripts/**`, `notebooks/07_*.ipynb`, `configs/ensemble_*.yaml`, `tests/unit/test_calibration_*.py`, `tests/integration/**` |
| `data-leakage-auditor` | 시간축/폴드 누수, 트랙 간 폴드 일치, `50:50 WLL` 수식, 시드 재현성 독립 감사 | **읽기 전용 (`Write 금지`)** |

- **공통 기준 보호**: `src/toss_ctr/evaluation/metrics.py`는 모든 모델의 공통 잣대이므로 코디네이터의 명시적 지시 없이 하위 에이전트가 수정할 수 없습니다.

## 2. 단계별 실행 게이트 (Orchestration Workflow Gates)

1. **Gate 1 — 데이터 계약 및 CV 확정 (직렬 선행)**:
   - `eda_summary.json` 등 프로파일링 결과 확보 후 `data-cv-architect`를 먼저 단독 실행해 **경량화 스키마 및 공통 `fold_id` 분할 규격(Data Contract)**을 확정합니다. (스키마 미확정 상태에서 하위 모델 에이전트를 성급히 병렬 실행하지 않습니다.)
2. **Gate 2 — 정형 GBDT & 시퀀스 DeepCTR 트랙 개발 (병렬 팬아웃)**:
   - Data Contract가 확정되면 `tabular-gbdt-engineer`와 `seq-deepctr-engineer`를 **동시에 병렬 호출(`invoke_subagent`)**하여 독립 개발합니다.
   - 두 트랙은 반드시 동일한 폴드 기준의 표준 예측 파일(`oof_<name>.parquet`: `row_id, clicked, raw_logit, pred_prob` / `test_<name>.parquet`: `ID, raw_logit, pred_prob`)을 출력하도록 지시합니다.
3. **Gate 3 — 독립 누수 및 재현성 감사 (필수 품질 게이트)**:
   - 각 트랙의 피처·학습·노트북 코드가 작성되면 `data-leakage-auditor`를 호출하여 감사를 수행합니다. `CRITICAL` 이슈가 발견되면 해당 담당 에이전트에게 즉시 수정을 재지시합니다.
4. **Gate 4 — 이종 블렌딩 & `50:50 WLL` 보정 (통합 마감)**:
   - 감사 통과된 다중 모델 OOF/Test 예측이 준비되면 `ensemble-calibrator`를 호출해 Log-odds 가중치 최적화, `50:50 WLL` Calibration, 최종 제출 검증 및 본선 서빙 스크립트를 완성합니다.

## 3. 서브에이전트 호출 브리프(Prompt) 필수 포함 항목
코디네이터가 `invoke_subagent`로 작업을 지시할 때는 반드시 다음 4가지를 명시합니다:
1. **참조 입력**: 읽어야 할 문서(`GEMINI.md`, `data/README.md`, `eda_summary.json`, 기존 설정 파일 등)
2. **허용 쓰기 범위**: 생성·수정할 정확한 파일 경로 목록
3. **핵심 제약**: Colab-First(로컬 대용량 데이터 로드 금지), 브라우저 팝업 금지, `AP`+`50:50 WLL` 분리 평가
4. **완료 기준**: 반환해야 할 요약 내용(구현 로직 요약, 테스트 통과 여부, 사용자 Colab 실행 안내 포인트)
