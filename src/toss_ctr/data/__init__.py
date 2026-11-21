"""데이터 로딩, 스키마 검증, 경량화(Diet) 및 시간축 기반 분할 모듈."""

from toss_ctr.data.diet import build_diet_artifacts
from toss_ctr.data.schema import (
    DEFAULT_CONTRACT,
    DatasetContract,
    collect_parquet_column_stats,
    discover_feature_groups,
    downcast_arrow_table,
    downcast_dataframe,
    extract_eda_feature_map,
    infer_safe_dtypes,
    load_eda_summary,
)
from toss_ctr.data.splits import build_cv_folds, load_folds, validate_folds_contract

__all__ = [
    "DEFAULT_CONTRACT",
    "DatasetContract",
    "build_cv_folds",
    "build_diet_artifacts",
    "collect_parquet_column_stats",
    "discover_feature_groups",
    "downcast_arrow_table",
    "downcast_dataframe",
    "extract_eda_feature_map",
    "infer_safe_dtypes",
    "load_eda_summary",
    "load_folds",
    "validate_folds_contract",
]
