---
trigger: glob
globs: "**/*.py, **/*.ipynb"
---
# CTR 모델링 및 Python 파이프라인 규칙

1. **대용량 메모리 안전성 (OOM 방지)**:
   - `train.parquet`(~8.2GB) 등 대용량 파일 처리 시 `pandas.read_parquet()` 전체 적재를 피하고, `pyarrow.parquet`, `polars.scan_parquet`, `duckdb`로 필요한 컬럼만 선택·청크 처리합니다.
   - 수치형·정수형 피처는 정보 손실 범위를 확인한 뒤 `float32`, `int32`/`int16`으로 다운캐스팅하고, 가변 길이 문자열인 `seq` 컬럼은 정형 테이블과 분리 처리합니다.
2. **시간축 정합성 & 타겟 누수 방지**:
   - `day_of_week`, `hour` 등 시간 구조를 반영해 검증 세트를 구성하며, 타겟 인코딩·클릭률 통계 피처는 반드시 Out-of-Fold 또는 과거 시점 데이터만으로 계산합니다.
3. **공식 평가지표(`AP` 50% + `50:50 WLL` 50%) 최적화 및 확률 보정**:
   - `WLL`은 `clicked=0`과 `clicked=1`의 기여도를 **50:50**으로 맞춘 가중 LogLoss입니다.
   - 불균형 원천 분포 그대로 학습한 모델의 원시 CTR 확률(`~1~3%`)을 보정 없이 제출하면 양성 클래스(`clicked=1`) 손실이 크게 벌어나 `WLL`이 급격히 악화될 수 있습니다.
   - 학습 시 `class_weight='balanced'`(`scale_pos_weight = N_0 / N_1`) 적용 여부와, Negative Downsampling(`1:k`) 적용 시 **50:50 Balanced Prior 기준 Log-odds 보정(`logit(p) + log(k)`)** vs 원시 확률의 Validation `WLL`을 반드시 비교·검증합니다. (단조 변환은 `AP`를 변화시키지 않으므로 `AP`와 `WLL`을 분리해 추적합니다.)
4. **코드 품질 및 본선 코드 검증 대비**:
   - 모든 함수에는 타입 힌트와 독스트링을 작성하고, 시드 고정 및 `configs/` 기반 설정 주입으로 100% 재현 가능하게 유지합니다.
