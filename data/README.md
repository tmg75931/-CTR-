# Dataset Info

> 대용량 원천 데이터(`*.parquet`, `*.csv`)는 Google Colab / Google Drive 환경에서 적재 및 연산하며, 로컬 PC(`data/raw/`, `data/processed/`)에는 보관하지 않습니다.

## 1. 파일 개요

| 파일명 | 행 수 (샘플 수) | 컬럼 수 | 주요 특징 |
| :--- | :---: | :---: | :--- |
| `train.parquet` | 10,704,179 | 119 | 학습 데이터 (`clicked` 타겟 컬럼 포함) |
| `test.parquet` | 1,527,298 | 119 | 평가/추론 데이터 (`ID` 식별자 컬럼 포함) |
| `sample_submission.csv` | 1,527,298 | 2 | 제출 양식 (`ID`, `clicked`) |

---

## 2. `train.parquet` 컬럼 명세

- **총 10,704,179개 샘플**, **총 119개 컬럼** (`clicked` Target 컬럼 포함)

| 컬럼 / 그룹명 | 설명 |
| :--- | :--- |
| `gender` | 성별 |
| `age_group` | 연령 그룹 |
| `inventory_id` | 지면 ID |
| `day_of_week` | 주번호 |
| `hour` | 시간 |
| `seq` | 유저 서버 로그 시퀀스 |
| `l_feat_*` | 속성 정보 피처 (`l_feat_14`는 Ads set) |
| `feat_e_*` | 정보영역 e 피처 |
| `feat_d_*` | 정보영역 d 피처 |
| `feat_c_*` | 정보영역 c 피처 |
| `feat_b_*` | 정보영역 b 피처 |
| `feat_a_*` | 정보영역 a 피처 |
| `history_a_*` | 과거 인기도 피처 |
| `clicked` | 클릭 여부 (Label / Target) |

---

## 3. `test.parquet` 컬럼 명세

- **총 1,527,298개 샘플**, **총 119개 컬럼** (`ID` 식별자 컬럼 포함)

| 컬럼 / 그룹명 | 설명 |
| :--- | :--- |
| `ID` | 샘플 식별자 |
| `gender` | 성별 |
| `age_group` | 연령 그룹 |
| `inventory_id` | 지면 ID |
| `day_of_week` | 주번호 |
| `hour` | 시간 |
| `seq` | 유저 서버 로그 시퀀스 |
| `l_feat_*` | 속성 정보 피처 (`l_feat_14`는 Ads set) |
| `feat_e_*` | 정보영역 e 피처 |
| `feat_d_*` | 정보영역 d 피처 |
| `feat_c_*` | 정보영역 c 피처 |
| `feat_b_*` | 정보영역 b 피처 |
| `feat_a_*` | 정보영역 a 피처 |
| `history_a_*` | 과거 인기도 피처 |

---

## 4. `sample_submission.csv` (제출 양식)

| 컬럼명 | 설명 |
| :--- | :--- |
| `ID` | 샘플 식별자 |
| `clicked` | 광고를 클릭할 확률 (`0 ~ 1`) |
