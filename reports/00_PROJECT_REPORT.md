# 토스광고 클릭 예측(CTR) 프로젝트 리포트 및 실측 결과

> **진행 현황 요약**: Google Colab(T4 GPU)에서 **`01`~`05` 노트북 5종**의 실제 실행 및 출력물 회수를 완료했으며, Colab 컴퓨팅 자원 한계로 **`06`(PyTorch DeepCTR)** 및 **`07`(이종 앙상블·최종 보정)** 단계 전 시점에서 GPU 세션을 안전 종료(`colab stop`)하고 현재까지의 실측 결과를 최종 정리했습니다.

---

## 1. 실제 데이터 전수 EDA & 경량화·시간축 CV 실측 결과 (`01`, `02` 완료)

### ① 데이터셋 규모 및 타겟 불균형 (`eda_summary.json`)
- **Train 행 수**: `10,704,179`행 (`119`개 컬럼) / **Test 행 수**: `1,527,298`행 (`119`개 컬럼)
- **타겟(`clicked`) 분포**: 클릭(`1`) **`204,179`건** / 미클릭(`0`) **`10,500,000`건** (**전체 CTR = `1.9075%`**, `pos:neg = 1 : 51.43`)
- **피처 그룹(117개) 실제 구성**: `explicit` 6개(`gender`, `age_group`, `inventory_id`, `day_of_week`, `hour`, `l_feat_14`), `l_feat` 26개, `feat_e` 10개, `feat_d` 6개, `feat_c` 8개, `feat_b` 6개, `feat_a` 18개, `history_a` 7개, 추가 발견 `history_b_1~30` 30개 (`float` 112개 + `string` 5개)
- **주요 결측 & 지면 편차**:
  - `gender`, `age_group`, `l_feat_2`, `l_feat_8`, `l_feat_18~24`, `feat_e_1/2/4`가 동일한 결측률 **`0.1608%`**를 공유하며(`feat_e_3`은 `10.1414%`), 동시 결측 패턴 피처의 유효성을 확인했습니다.
  - 주요 `inventory_id`별 CTR은 `2`번 지면(`0.8787%`) vs `37`번 지면(`2.9917%`) vs `41`번 지면(`3.6165%`) vs `21`번 지면(`5.7634%`)으로 지면 간 4~6배 이상 격차가 존재합니다.

### ② 용량 다이어트 & 시간축 CV 검증 (`cv_summary.json`)
- **`seq` 분리 및 다운캐스팅 절감**:
  - 원천 합계 **`9.385 GB`** 중 `seq` 컬럼 단독 압축 용량이 **`7,081.49 MB` (84.4%)** 점유
  - `seq` 분리 + 다운캐스팅 후 전체 Diet **`5.946 GB`** (`36.64%` 절감), **정형 Diet(Tabular Only) 합계 `1.400 GB` (`85.09%` 절감)** 달성
- **시간축 구조 (`Out-of-Time Shifted`, `TVD = 0.8574`)**:
  - `train`: `day_of_week` `'1'~'7'` 균등 분포 / `test`: **`day_of_week = '7'` 단일 요일 100% (`1,527,298`건)** → `oot_day = '7'` (`is_oot_val = True`) 마킹 완료
- **모델링 세트 구성 (Colab T4 12.67 GB RAM 최적화)**:
  - **Train**: 양성(`clicked=1`) **`204,179`건 100% 전수** + 음성 요일별 층화 1:5 추출 `1,020,895`건 = 총 **`1,225,074`행** (`CTR = 16.6667%`, 5-Fold Stratified CV, OOT 슬라이스 `172,178`건)
  - **Test**: 원천 **`1,527,298`행 100% 전수**

---

## 2. 시퀀스(`seq`) 요약 피처 추출 실측 결과 (`05` 완료)

- **처리 규모**: Train `1,225,074`행 (`163.6s`, `17.66 MB`) / Test `1,527,298`행 (`213.3s`, `19.56 MB`), 구분자 `','` 자동 감지, 결측·빈 시퀀스 `0`건 (`reports/seq_summary_stats.json`)
- **토큰 길이 분포**: Train 평균 `520.49` (중앙값 `426`, p90 `1,116`, 최대 `10,560`) / Test 평균 `542.16` (중앙값 `453`, p90 `1,135`, 최대 `15,752`)
- **시퀀스 길이 구간별 CTR (1:5 학습 세트 기준)**:
  - `1~10` 토큰: **`29.72%`** (`31,901`건, `2.60%`)
  - `11~50` 토큰: **`24.79%`** (`86,066`건, `7.03%`)
  - `51~100` 토큰: **`20.36%`** (`86,248`건, `7.04%`)
  - `100+` 토큰: **`15.26%`** (`1,020,859`건, `83.33%`)

---

## 3. 모델별 5-Fold OOF / OOT 실측 성능 지표 (`03`, `04` 완료)

> **평가 산식**: `Combined Score = 0.5 × AP + 0.5 × (1 / (1 + WLL))` (여기서 `WLL`은 `clicked` 0/1 기여 `50:50` 가중 LogLoss)

### ① 최종 4개 GBDT 모델 비교 (OOF `1,225,074`행 / OOT `172,178`행)

| 순위 | 모델명 | 입력 피처 | OOF AP ↑ | OOF 50:50 WLL ↓ | `pos_logloss` | `neg_logloss` | **OOF Score ↑** | **OOT Score ↑** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`xgb_fe_tuned`** (`04`, CUDA `hist`) | **244개** | **0.415821** | **0.594157** | 0.620715 | 0.567599 | **0.521556** | **0.510586** |
| 2 | `lgbm_baseline` (`03`, `prior_shift`) | 117개 | 0.409295 | 0.598260 | 0.606429 | 0.590090 | 0.517488 | 0.506630 |
| 3 | `lgbm_fe_tuned` (`04`, `balanced_weight`) | 244개 | 0.408997 | 0.599194 | 0.633800 | 0.564587 | 0.517156 | 0.505721 |
| 4 | `catboost_fe_tuned` (`04`, GPU) | 244개 | 0.407804 | 0.596783 | 0.603826 | 0.589739 | 0.517032 | 0.504438 |

- **최고 단일 모델 성과**: **`xgb_fe_tuned`**가 베이스라인(`0.517488`) 대비 OOF Score **`+0.004068`** (`0.521556`), OOT Score **`+0.003956`** (`0.510586`), OOF AP **`+0.006526`**, 50:50 WLL **`-0.004103`** 동반 개선을 달성했습니다.
- **확률 보정 전략 비교 (`03` 베이스라인 실험)**:
  - `raw_unweighted + prior_shift(0.5)`: `AP = 0.409295`, `WLL = 0.598260`, **`Score = 0.517488`**
  - `balanced_weight (scale_pos_weight)`: `AP = 0.406046`, `WLL = 0.600189`, `Score = 0.515486`
  - `raw_unweighted (무보정)`: `AP = 0.409295`, `WLL = 0.847438`, `Score = 0.475292` → **50:50 보정만으로 Score `+0.042196` 개선** 실증.

### ② 파생 피처군별 중요도 및 모델 다양성 (`04` 실측)
- **피처군별 Gain 기여도 (LightGBM FE 기준)**:
  - `Raw Tabular`: `45.04%` / `Row/History Stats`: **`31.39%`** / `OOF Target Enc`: **`21.02%`** / `Fold Freq/Cond`: `1.58%` / `Seq Summary`: `0.52%` / `Interaction Key`: `0.46%`
- **OOF `raw_logit` Pearson 상관계수 (앙상블 다양성)**:
  - `lgbm_fe_tuned` ↔ `catboost_fe_tuned`: `0.9641`
  - `lgbm_fe_tuned` ↔ `xgb_fe_tuned`: `0.9647`
  - `catboost_fe_tuned` ↔ `xgb_fe_tuned`: `0.9810`

---

## 4. 노트북 실행 현황 및 확보된 산출물

| 단계 | 노트북 | 상태 | 비고 / 산출물 |
| :---: | :--- | :---: | :--- |
| 1 | `notebooks/01_eda_profiling.ipynb` | **완료 (5/5 셀 출력 회수)** | `reports/eda_summary.json` (에러 0건) |
| 2 | `notebooks/02_data_diet_and_cv.ipynb` | **완료 (5/5 셀 출력 회수)** | `reports/cv_summary.json` (에러 0건) |
| 3 | `notebooks/03_gbdt_baseline.ipynb` | **완료 (6/6 셀 출력 회수)** | `outputs/submissions/submission_lgbm_baseline.csv` (`1,527,298`행, 에러 0건) |
| 4 | `notebooks/04_tabular_fe_and_gbdt.ipynb` | **완료 (8/8 셀 출력 회수)** | GBDT 3종 OOF/OOT 비교 표·시각화·다양성 행렬 (에러 0건) |
| 5 | `notebooks/05_seq_feature_extraction.ipynb` | **완료 (4/4 셀 출력 회수)** | `reports/seq_summary_stats.json` (에러 0건) |
| 6 | `notebooks/06_deepctr_seq_dcn_gpu.ipynb` | **미실행 (Colab 자원 종료)** | 코드·테스트 구현 완료, 추후 GPU 확보 시 즉시 실행 가능 |
| 7 | `notebooks/07_oof_ensemble_calibration_submit.ipynb` | **미실행 (Colab 자원 종료)** | 코드·테스트 구현 완료, 추후 `06` 완료 또는 GBDT 4종으로 즉시 블렌딩 가능 |

---

## 5. 로컬 코드 품질 및 단위·통합 검증 현황

| 점검 항목 | 상태 | 상세 결과 |
| :--- | :---: | :--- |
| **패키지 린트 (`ruff`)** | **통과** | `ruff check src scripts tests` — `All checks passed!` |
| **단위·통합 테스트 (`pytest`)** | **통과** | `39 passed, 2 skipped` *(로컬 `.venv` PyTorch 미설치 2건 제외 전원 통과)* |
| **독립 데이터 누수 감사** | **통과** | `data-leakage-auditor` 감사 `CRITICAL` 결함 **0건** |
| **Colab 리소스 정리** | **완료** | `colab stop -s toss-ctr` 종료 완료 |
