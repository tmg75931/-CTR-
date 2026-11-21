# 토스광고 클릭 예측(CTR) 모델 개발

토스 앱 내 광고 지면의 외부 디스플레이 광고 노출·클릭 로그를 기반으로 사용자 클릭 확률(CTR)을 예측하는 머신러닝·딥러닝 통합 파이프라인입니다.

- **평가 지표**: `Combined Score = 0.5 × AP + 0.5 × (1 / (1 + WLL))`
  - **AP (Average Precision, 50%)**: 순위 변별력 평가
  - **WLL (50:50 Weighted LogLoss, 50%)**: `clicked=0`과 `clicked=1`의 클래스 기여를 `50:50`으로 맞춘 확률 보정 평가

---

## 핵심 실측 결과

> Google Colab T4 GPU(VRAM 15 GB, RAM 12.67 GB) 환경에서 공통 5-Fold Stratified CV(`1,225,074`행 OOF / `day_of_week='7'` `172,178`행 OOT) 기준으로 측정한 단일 모델 실측 결과입니다.

| 순위 | 모델명 | 입력 피처 | OOF AP ↑ | OOF 50:50 WLL ↓ | OOF Score ↑ | OOT Score ↑ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **`xgb_fe_tuned`** (CUDA `hist`) | **244개** | **0.415821** | **0.594157** | **0.521556** | **0.510586** |
| 2 | `lgbm_baseline` (`prior_shift`) | 117개 | 0.409295 | 0.598260 | 0.517488 | 0.506630 |
| 3 | `lgbm_fe_tuned` (`balanced_weight`) | 244개 | 0.408997 | 0.599194 | 0.517156 | 0.505721 |
| 4 | `catboost_fe_tuned` (GPU) | 244개 | 0.407804 | 0.596783 | 0.517032 | 0.504438 |

### 주요 성과 및 엔지니어링 포인트
- **데이터 다이어트**: 원천 용량의 84.4%(`7,081.49 MB`)를 차지하는 `seq` 컬럼을 분리하고 수치형 다운캐스팅을 적용해 정형 데이터 용량을 **`9.385 GB → 1.400 GB` (`85.09%` 절감)**로 축소했습니다.
- **시간축 분포 차이(OOT) 검증**: 원천 Test(`1,527,298`행)가 100% 단일 요일(`day_of_week = '7'`, `TVD = 0.8574`)임을 전수 EDA에서 확인하고, 5-Fold Stratified CV와 요일 7 OOT 슬라이스 지표를 병행 추적했습니다.
- **50:50 확률 보정 검증**: 베이스라인 기준 무보정(`Score 0.475292`, `WLL 0.847438`) 대비 Log-odds `prior_shift(0.5)` 보정만으로 **`Score 0.517488` (`+0.042196`)** 개선을 확인했습니다.
- **12.67 GB RAM OOM 해결**: `TabularFeaturePipeline`에 `FoldFittedEncoders`를 도입하고 `GBDTOOFTrainer`가 학습 완료 후 학습 프레임을 해제한 뒤 Test 데이터를 25만 행 청크로 변환·추론하도록 리팩터링해 GPU 학습 OOM을 해소했습니다.
- **누수 차단 및 검증**: 모든 동적 인코딩(빈도·조건부 통계·내부 서브폴드 OOF 타겟 인코딩)과 시퀀스 어휘집 구축을 학습 폴드 내부로 완전 격리했습니다(`ruff` 통과, `pytest 39 passed`, 독립 누수 감사 `CRITICAL 0건`).

---

## 데이터셋 요약

| 구분 | 항목 | 실측 수치 |
| :--- | :--- | :--- |
| 원천 Train | 행 수 / 컬럼 수 | `10,704,179`행 / `119`개 (정형 117 + `seq` + `clicked`) |
| 원천 Test | 행 수 / 컬럼 수 | `1,527,298`행 / `119`개 (`day_of_week = '7'` 100%) |
| 타겟 분포 | 클릭(1) / 미클릭(0) | `204,179`건 / `10,500,000`건 (원천 CTR `1.9075%`, `1 : 51.43`) |
| 모델링 Train | 1:5 층화 학습 세트 | `1,225,074`행 (양성 100% 전수 + 요일별 층화 음성 1:5, CTR `16.67%`) |
| 시퀀스(`seq`) | 평균 토큰 길이 | Train `520.49` (최대 `10,560`) / Test `542.16` (최대 `15,752`) |

> 원천 데이터(`train.parquet`, `test.parquet`, `sample_submission.csv`)는 대회 보안 및 용량 정책에 따라 저장소에 포함하지 않았습니다. 컬럼 명세는 [data/README.md](data/README.md)를 참고하세요.

---

## 저장소 구조

```text
├── configs/                  # 단계별 실험 설정 (YAML)
├── data/                     # 데이터 명세서 및 로컬 데이터 경로 (.gitignore 제외)
├── notebooks/                # 01~07 단계별 분석·학습·추론 노트북
├── outputs/submissions/      # 제출 CSV 출력 경로 (.gitignore 제외)
├── reports/                  # 실측 리포트(00_PROJECT_REPORT.md, JSON 요약, 블로그 원고)
├── scripts/                  # 재현 검증용 CLI 엔트리포인트
├── src/toss_ctr/             # 핵심 파이프라인 패키지
│   ├── data/                 # 스키마 계약, 스트리밍 다이어트, 5-Fold/OOT 분할
│   ├── evaluation/           # 공식 AP, 50:50 WLL, Prior Shift 산식
│   ├── features/             # 교호작용 해시, 행·이력 통계, OOF 타겟 인코딩, seq 요약
│   ├── models/               # 다중 GBDT 학습기, PyTorch Target-Attention+DCN-v2, 확률 보정기
│   └── inference/            # OOF Log-odds 블렌더, 제출 무결성 검증, 실시간 서빙 레이어
└── tests/                    # 단위 및 종단간 통합 테스트
```

---

## 단계별 노트북 현황

| 단계 | 노트북 | 상태 | 설명 |
| :---: | :--- | :---: | :--- |
| 1 | [01_eda_profiling.ipynb](notebooks/01_eda_profiling.ipynb) | 실행 완료 (5/5) | 1,070만 행 전수 스트리밍 EDA 및 결측·지면·요일 분포 분석 |
| 2 | [02_data_diet_and_cv.ipynb](notebooks/02_data_diet_and_cv.ipynb) | 실행 완료 (5/5) | `seq` 분리·다운캐스팅(85% 절감) 및 공통 5-Fold/OOT 분할 확정 |
| 3 | [03_gbdt_baseline.ipynb](notebooks/03_gbdt_baseline.ipynb) | 실행 완료 (6/6) | LightGBM 베이스라인 및 `prior_shift` vs `scale_pos_weight` 비교 |
| 4 | [04_tabular_fe_and_gbdt.ipynb](notebooks/04_tabular_fe_and_gbdt.ipynb) | 실행 완료 (8/8) | 244개 파생 피처 및 LightGBM·CatBoost·XGBoost 5-Fold 비교 |
| 5 | [05_seq_feature_extraction.ipynb](notebooks/05_seq_feature_extraction.ipynb) | 실행 완료 (4/4) | `seq` 스트리밍 요약 피처 11종 추출 및 길이 구간별 CTR 분석 |
| 6 | [06_deepctr_seq_dcn_gpu.ipynb](notebooks/06_deepctr_seq_dcn_gpu.ipynb) | 구현 완료 (미실행) | PyTorch Target-Attention + DCN-v2 5-Fold GPU 학습 (자원 한계로 보류) |
| 7 | [07_oof_ensemble_calibration_submit.ipynb](notebooks/07_oof_ensemble_calibration_submit.ipynb) | 구현 완료 (미실행) | 이종 모델 OOF Log-odds 블렌딩, Affine 보정 및 최종 제출 검증 |

---

## 설치 및 실행 방법

### 1. 개발 환경 설치 및 테스트
```bash
pip install -e ".[dev]"
ruff check src scripts tests
pytest -q
```

### 2. 재현 CLI 파이프라인
```bash
# 1) 데이터 경량화 및 공통 5-Fold / OOT 분할 생성
python scripts/run_diet_and_cv.py --config configs/data_diet.yaml --cv-config configs/cv_strategy.yaml

# 2) 시퀀스 요약 피처 스트리밍 추출
python scripts/extract_seq_features.py --config configs/deep_seq_dcn.yaml

# 3a) 정형 FE 기반 다중 GBDT (LightGBM / CatBoost / XGBoost) 5-Fold 학습
python scripts/train_gbdt.py --config configs/gbdt_fe_tuned.yaml --model all

# 3b) PyTorch DeepCTR (Target-Attention + DCN-v2) 5-Fold GPU 학습
python scripts/train_deepctr.py --config configs/deep_seq_dcn.yaml

# 4) OOF Log-odds 블렌딩, 50:50 WLL 확률 보정 및 제출 검증
python scripts/blend_and_submit.py --config configs/ensemble_wll50.yaml
python scripts/verify_submission.py --config configs/ensemble_wll50.yaml
```

---

## 상세 리포트
- 프로젝트 종합 실측 리포트: [reports/00_PROJECT_REPORT.md](reports/00_PROJECT_REPORT.md)
- 기술 블로그 정리 원고: [reports/notion_ctr_post.md](reports/notion_ctr_post.md)
