"""시간축 적응형(Out-of-Time 감지) 및 계층형 Local CV 분할 모듈."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import KFold, StratifiedKFold

from toss_ctr.data.schema import DEFAULT_CONTRACT, DatasetContract

_WEEKDAY_ORDER: dict[str, int] = {
    "mon": 1,
    "monday": 1,
    "tue": 2,
    "tuesday": 2,
    "wed": 3,
    "wednesday": 3,
    "thu": 4,
    "thursday": 4,
    "fri": 5,
    "friday": 5,
    "sat": 6,
    "saturday": 6,
    "sun": 7,
    "sunday": 7,
}


def _read_parquet_columns_stream(
    parquet_path: str | Path,
    columns: Sequence[str],
    batch_size: int = 262_144,
) -> dict[str, np.ndarray]:
    """Parquet에서 지정된 소수 컬럼만 배치 스트리밍으로 읽어 NumPy 배열로 반환합니다."""
    pf = pq.ParquetFile(parquet_path)
    missing = [c for c in columns if c not in pf.schema_arrow.names]
    if missing:
        raise ValueError(f"파일({parquet_path})에 필요한 컬럼이 없습니다: {missing}")

    chunks: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns)):
        for idx, col in enumerate(columns):
            arr = batch.column(idx).to_numpy(zero_copy_only=False)
            chunks[col].append(arr)

    result: dict[str, np.ndarray] = {}
    for col in columns:
        if not chunks[col]:
            arrow_type = pf.schema_arrow.field(col).type
            result[col] = pa.array([], type=arrow_type).to_numpy(zero_copy_only=False)
        else:
            result[col] = np.concatenate(chunks[col], axis=0)
    return result


def _day_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, np.integer, float, np.floating)):
        return (0, float(value))
    text = str(value).strip()
    lower = text.lower()
    if lower in _WEEKDAY_ORDER:
        return (0, float(_WEEKDAY_ORDER[lower]))
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def _normalize_day_list(arr: np.ndarray) -> list[Any]:
    series = pd.Series(arr).dropna()
    return series.tolist()


def _compute_day_distribution(
    train_days_raw: np.ndarray,
    test_days_raw: np.ndarray,
) -> tuple[list[dict[str, Any]], float, list[Any], Any]:
    train_s = pd.Series(train_days_raw).dropna()
    test_s = pd.Series(test_days_raw).dropna()

    train_counts = train_s.value_counts().to_dict()
    test_counts = test_s.value_counts().to_dict()
    train_total = max(len(train_s), 1)
    test_total = max(len(test_s), 1)

    all_days = sorted(set(train_counts.keys()) | set(test_counts.keys()), key=_day_sort_key)
    train_day_set = set(train_counts.keys())
    test_day_set = set(test_counts.keys())
    unseen_test_days = sorted(test_day_set - train_day_set, key=_day_sort_key)

    max_train_day = max(train_day_set, key=_day_sort_key) if train_day_set else None

    comparison: list[dict[str, Any]] = []
    tvd = 0.0
    for day in all_days:
        tr_cnt = int(train_counts.get(day, 0))
        te_cnt = int(test_counts.get(day, 0))
        tr_share = tr_cnt / train_total
        te_share = te_cnt / test_total
        tvd += 0.5 * abs(tr_share - te_share)
        comparison.append(
            {
                "day_of_week": day.item() if isinstance(day, np.generic) else day,
                "train_count": tr_cnt,
                "test_count": te_cnt,
                "train_share": round(tr_share, 6),
                "test_share": round(te_share, 6),
            }
        )

    unseen_clean = [d.item() if isinstance(d, np.generic) else d for d in unseen_test_days]
    max_day_clean = max_train_day.item() if isinstance(max_train_day, np.generic) else max_train_day
    return comparison, float(tvd), unseen_clean, max_day_clean


def build_cv_folds(
    train_path: str | Path,
    test_path: str | Path,
    output_path: str | Path | None = None,
    n_splits: int = 5,
    seed: int = 42,
    oot_mode: str = "auto",
    tvd_threshold: float = 0.15,
    mark_last_day_if_in_time: bool = True,
    batch_size: int = 262_144,
    compression: str = "zstd",
    report_path: str | Path | None = None,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """`train`과 `test`의 시간축 분포를 비교해 OOT 마킹 및 결정론적 StratifiedKFold 분할을 생성합니다.

    Args:
        train_path: 원천 `train.parquet` 또는 경량화된 `train_tabular_diet.parquet` 경로.
        test_path: 원천 `test.parquet` 또는 경량화된 `test_tabular_diet.parquet` 경로.
        output_path: 저장할 `folds.parquet` 경로 (`None`이면 디스크 저장 생략).
        n_splits: StratifiedKFold 분할 수 (기본 5).
        seed: 결정론적 셔플용 난수 시드 (기본 42).
        oot_mode: `"auto"`(분포 편차/미등장 요일 감지 또는 마지막 요일 마킹),
                  `"last_day"`(항상 train 최대 요일 마킹), `"none"`(모두 False).
        tvd_threshold: `train` vs `test` 요일 분포 Total Variation Distance 임계치.
        mark_last_day_if_in_time: `auto` 모드에서 분포가 유사해도 마지막 요일을 OOT 보조 검증으로 마킹할지 여부.
        batch_size: PyArrow 스트리밍 배치 크기.
        compression: 출력 Parquet 압축 코덱.
        report_path: 요약 통계 JSON 저장 경로 (`None`이면 생략).
        contract: 데이터 계약 명세.

    Returns:
        `(folds_df, summary_dict)` 튜플. `folds_df` 스키마는 `["row_id", "fold_id", "is_oot_val"]`.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits는 2 이상이어야 합니다: {n_splits}")
    if oot_mode not in ("auto", "last_day", "none"):
        raise ValueError(f"oot_mode는 'auto', 'last_day', 'none' 중 하나여야 합니다: {oot_mode!r}")

    train_cols = _read_parquet_columns_stream(
        train_path,
        columns=[contract.time_col, contract.target_col],
        batch_size=batch_size,
    )
    test_cols = _read_parquet_columns_stream(
        test_path,
        columns=[contract.time_col],
        batch_size=batch_size,
    )

    train_dow = train_cols[contract.time_col]
    clicked_raw = train_cols[contract.target_col]
    if pd.Series(clicked_raw).isna().any():
        raise ValueError(f"타겟 '{contract.target_col}' 컬럼에 결측치가 포함되어 있습니다.")
    train_y = np.asarray(clicked_raw, dtype=np.int64)
    test_dow = test_cols[contract.time_col]
    n_rows = int(len(train_y))

    if n_rows < n_splits:
        raise ValueError(f"학습 행 수({n_rows})가 n_splits({n_splits})보다 작습니다.")

    dow_compare, tvd, unseen_test_days, max_train_day = _compute_day_distribution(
        train_dow, test_dow
    )
    unique_train_days = pd.Series(train_dow).dropna().nunique()

    is_temporal_shift = bool(unseen_test_days) or (tvd >= tvd_threshold)
    if unseen_test_days:
        temporal_regime = "out_of_time_unseen_days"
    elif tvd >= tvd_threshold:
        temporal_regime = "out_of_time_shifted"
    else:
        temporal_regime = "in_time_similar"

    should_mark_oot = False
    if unique_train_days >= 2 and max_train_day is not None:
        if oot_mode == "last_day":
            should_mark_oot = True
        elif oot_mode == "auto":
            should_mark_oot = is_temporal_shift or mark_last_day_if_in_time

    if should_mark_oot:
        is_oot_val = (pd.Series(train_dow) == max_train_day).fillna(False).to_numpy(dtype=bool)
    else:
        is_oot_val = np.zeros(n_rows, dtype=bool)

    # (day_of_week, clicked) 조합 기반 벡터화 스트래티피케이션 키 구성
    dow_codes, _ = pd.factorize(pd.Series(train_dow), use_na_sentinel=True)
    strat_key = (dow_codes.astype(np.int64) + 1) * 2 + train_y

    uniq_keys, key_counts = np.unique(strat_key, return_counts=True)
    rare_keys = uniq_keys[key_counts < n_splits]
    fallback_used = "none"

    if len(rare_keys) > 0:
        # 희소 (day_of_week, clicked) 조합은 clicked 단독 키로 안전 병합
        rare_mask = np.isin(strat_key, rare_keys)
        strat_key = strat_key.copy()
        strat_key[rare_mask] = -(train_y[rare_mask] + 1)
        fallback_used = "merged_rare_to_clicked"

        uniq_after, counts_after = np.unique(strat_key, return_counts=True)
        still_rare = uniq_after[counts_after < n_splits]
        if len(still_rare) > 0:
            # 극단적으로 전체 클래스 표본 자체가 n_splits 미만인 경우 clicked 단독 재시도 후 KFold 폴백
            _, y_counts = np.unique(train_y, return_counts=True)
            if np.all(y_counts >= n_splits) and len(y_counts) >= 2:
                strat_key = train_y
                fallback_used = "clicked_only"
            else:
                fallback_used = "plain_kfold"

    fold_id = np.empty(n_rows, dtype=np.int8)
    dummy_x = np.zeros(n_rows, dtype=np.int8)
    if fallback_used == "plain_kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(dummy_x)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(dummy_x, strat_key)

    for fold_idx, (_, val_idx) in enumerate(split_iter):
        fold_id[val_idx] = fold_idx

    row_id = np.arange(n_rows, dtype=np.int64)
    folds_table = pa.table(
        {
            contract.row_id_col: pa.array(row_id, type=pa.int64()),
            contract.fold_id_col: pa.array(fold_id, type=pa.int8()),
            contract.oot_col: pa.array(is_oot_val, type=pa.bool_()),
        }
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(folds_table, out_p, compression=compression)

    folds_df = folds_table.to_pandas()
    validate_folds_contract(folds_df, n_rows=n_rows, n_splits=n_splits, contract=contract)

    fold_summary: list[dict[str, Any]] = []
    for f_idx in range(n_splits):
        mask = fold_id == f_idx
        f_cnt = int(np.sum(mask))
        f_pos = int(np.sum(train_y[mask] == 1))
        f_oot = int(np.sum(is_oot_val[mask]))
        fold_summary.append(
            {
                "fold_id": f_idx,
                "rows": f_cnt,
                "pos_rows": f_pos,
                "ctr": round(f_pos / f_cnt, 6) if f_cnt > 0 else 0.0,
                "oot_rows": f_oot,
            }
        )

    oot_rows = int(np.sum(is_oot_val))
    summary: dict[str, Any] = {
        "n_rows": n_rows,
        "test_rows": int(len(test_dow)),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "overall_ctr": round(float(np.mean(train_y)), 6),
        "temporal_regime": temporal_regime,
        "tvd": round(tvd, 6),
        "tvd_threshold": float(tvd_threshold),
        "unseen_test_days": unseen_test_days,
        "oot_mode": oot_mode,
        "oot_marked": should_mark_oot,
        "oot_day": max_train_day if should_mark_oot else None,
        "oot_rows": oot_rows,
        "oot_ratio": round(oot_rows / n_rows, 6),
        "stratification_fallback": fallback_used,
        "day_of_week_compare": dow_compare,
        "fold_summary": fold_summary,
        "output_path": str(output_path) if output_path is not None else None,
    }

    if report_path is not None:
        rep_p = Path(report_path)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        with rep_p.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)

    return folds_df, summary


def validate_folds_contract(
    folds_df: pd.DataFrame | pa.Table | str | Path,
    n_rows: int | None = None,
    n_splits: int = 5,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """`folds.parquet`의 스키마, `row_id` 연속성, 결측 없음, `fold_id` 범위를 검증합니다."""
    if isinstance(folds_df, (str, Path)):
        df = pq.read_table(folds_df).to_pandas()
    elif isinstance(folds_df, pa.Table):
        df = folds_df.to_pandas()
    elif isinstance(folds_df, pd.DataFrame):
        df = folds_df
    else:
        raise TypeError(f"지원하지 않는 folds 입력 타입입니다: {type(folds_df).__name__}")

    required_cols = [contract.row_id_col, contract.fold_id_col, contract.oot_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"folds 계약 필수 컬럼이 누락되었습니다: {missing_cols}")

    actual_rows = len(df)
    if n_rows is not None and actual_rows != n_rows:
        raise ValueError(f"folds 행 수가 일치하지 않습니다: expected={n_rows}, actual={actual_rows}")

    for col in required_cols:
        if df[col].isna().any():
            raise ValueError(f"folds 컬럼 '{col}'에 결측치가 포함되어 있습니다.")

    row_id_s = df[contract.row_id_col]
    if not pd.api.types.is_integer_dtype(row_id_s.dtype):
        raise ValueError(f"'{contract.row_id_col}' 컬럼은 정수형이어야 합니다: {row_id_s.dtype}")
    expected_row_ids = np.arange(actual_rows, dtype=np.int64)
    if not np.array_equal(row_id_s.to_numpy(dtype=np.int64), expected_row_ids):
        raise ValueError(
            f"'{contract.row_id_col}'는 0부터 {actual_rows - 1}까지 오름차순 연속이어야 합니다."
        )

    fold_id_s = df[contract.fold_id_col]
    if not pd.api.types.is_integer_dtype(fold_id_s.dtype):
        raise ValueError(f"'{contract.fold_id_col}' 컬럼은 정수형이어야 합니다: {fold_id_s.dtype}")
    fold_vals = fold_id_s.to_numpy(dtype=np.int64)
    if actual_rows > 0:
        min_fold, max_fold = int(fold_vals.min()), int(fold_vals.max())
        if min_fold < 0 or max_fold >= n_splits:
            raise ValueError(
                f"'{contract.fold_id_col}' 범위 위반: 허용 [0, {n_splits - 1}], 실제 [{min_fold}, {max_fold}]"
            )
        if actual_rows >= n_splits:
            present_folds = set(np.unique(fold_vals).tolist())
            expected_folds = set(range(n_splits))
            if present_folds != expected_folds:
                missing_folds = sorted(expected_folds - present_folds)
                raise ValueError(f"할당되지 않은 폴드가 존재합니다: {missing_folds}")

    oot_s = df[contract.oot_col]
    if not pd.api.types.is_bool_dtype(oot_s.dtype):
        raise ValueError(f"'{contract.oot_col}' 컬럼은 bool 타입이어야 합니다: {oot_s.dtype}")

    return {
        "valid": True,
        "n_rows": int(actual_rows),
        "n_splits": int(n_splits),
        "oot_rows": int(oot_s.sum()),
    }


def load_folds(
    folds_path: str | Path,
    validate: bool = True,
    n_rows: int | None = None,
    n_splits: int = 5,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> pd.DataFrame:
    """`folds.parquet`을 로드하고 선택적으로 데이터 계약을 즉시 검증합니다."""
    df = pq.read_table(folds_path).to_pandas()
    if validate:
        validate_folds_contract(df, n_rows=n_rows, n_splits=n_splits, contract=contract)
    return df
