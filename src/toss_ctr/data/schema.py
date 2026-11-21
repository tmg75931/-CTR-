"""토스광고 CTR 데이터 계약(Data Contract), 피처 그룹 분류 및 안전 다운캐스팅 모듈."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass(frozen=True)
class DatasetContract:
    """트랙 간(GBDT/DeepCTR/Ensemble) 공유하는 표준 데이터 계약 명세."""

    target_col: str = "clicked"
    id_col: str = "ID"
    seq_col: str = "seq"
    row_id_col: str = "row_id"
    fold_id_col: str = "fold_id"
    oot_col: str = "is_oot_val"
    time_col: str = "day_of_week"
    hour_col: str = "hour"
    inventory_col: str = "inventory_id"
    ad_set_col: str = "l_feat_14"

    explicit_cols: tuple[str, ...] = (
        "gender",
        "age_group",
        "inventory_id",
        "day_of_week",
        "hour",
        "l_feat_14",
    )
    prefix_groups: tuple[tuple[str, str], ...] = (
        ("l_feat_", "l_feat"),
        ("feat_e_", "feat_e"),
        ("feat_d_", "feat_d"),
        ("feat_c_", "feat_c"),
        ("feat_b_", "feat_b"),
        ("feat_a_", "feat_a"),
        ("history_a_", "history_a"),
    )

    train_tabular_filename: str = "train_tabular_diet.parquet"
    test_tabular_filename: str = "test_tabular_diet.parquet"
    train_seq_filename: str = "train_seq.parquet"
    test_seq_filename: str = "test_seq.parquet"
    folds_filename: str = "folds.parquet"

    oof_required_cols: tuple[str, ...] = ("row_id", "clicked", "raw_logit", "pred_prob")
    test_pred_required_cols: tuple[str, ...] = ("ID", "raw_logit", "pred_prob")

    @property
    def reserved_cols(self) -> tuple[str, ...]:
        """피처 그룹 분류에서 제외되는 메타/식별/타겟/시퀀스 컬럼 목록."""
        return (
            self.target_col,
            self.id_col,
            self.seq_col,
            self.row_id_col,
            self.fold_id_col,
            self.oot_col,
        )


DEFAULT_CONTRACT = DatasetContract()

_INT_LIMITS: tuple[tuple[pa.DataType, int, int], ...] = (
    (pa.int8(), -128, 127),
    (pa.int16(), -32_768, 32_767),
    (pa.int32(), -2_147_483_648, 2_147_483_647),
    (pa.int64(), -9_223_372_036_854_775_808, 9_223_372_036_854_775_807),
)

_FLOAT32_MAX: float = float(np.finfo(np.float32).max)

_DTYPE_STR_TO_ARROW: dict[str, pa.DataType] = {
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
    "float16": pa.float16(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "double": pa.float64(),
    "bool": pa.bool_(),
    "bool_": pa.bool_(),
    "string": pa.string(),
    "large_string": pa.large_string(),
}


def resolve_arrow_dtype(dtype: pa.DataType | str) -> pa.DataType:
    """문자열 또는 PyArrow DataType을 표준 PyArrow DataType으로 변환합니다."""
    if isinstance(dtype, pa.DataType):
        return dtype
    normalized = str(dtype).strip().lower()
    if normalized in _DTYPE_STR_TO_ARROW:
        return _DTYPE_STR_TO_ARROW[normalized]
    raise ValueError(f"지원하지 않는 PyArrow dtype 표기입니다: {dtype!r}")


def load_eda_summary(path: str | Path | None) -> dict[str, Any] | None:
    """경량 프로파일링 요약 파일(`eda_summary.json`)을 로드합니다.

    파일 경로가 `None`이거나 존재하지 않으면 `None`을 반환합니다.
    """
    if path is None:
        return None
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"eda_summary.json 최상위 구조는 dict여야 합니다: {summary_path}")
    return payload


def extract_eda_feature_map(eda_summary: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """`eda_summary` 내 `features` 리스트를 컬럼명 키 딕셔너리로 정규화합니다."""
    if not eda_summary:
        return {}
    features = eda_summary.get("features", [])
    if not isinstance(features, Sequence):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in features:
        if isinstance(item, Mapping) and "column" in item:
            result[str(item["column"])] = dict(item)
    return result


def discover_feature_groups(
    columns: Sequence[str],
    eda_summary: Mapping[str, Any] | None = None,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> dict[str, list[str]]:
    """컬럼 목록에서 명시 피처 및 익명 프리픽스 그룹을 동적으로 상호배타 분류합니다.

    Args:
        columns: Parquet 스키마 또는 DataFrame의 전체 컬럼 목록.
        eda_summary: 선택적 EDA 요약 딕셔너리(존재 시 컬럼 메타데이터 참조 가능).
        contract: 데이터 계약 명세.

    Returns:
        `explicit`, `l_feat`, `feat_e`, `feat_d`, `feat_c`, `feat_b`, `feat_a`,
        `history_a`, `other` 키를 가진 그룹별 컬럼명 리스트 딕셔너리.
    """
    _ = extract_eda_feature_map(eda_summary)
    reserved = set(contract.reserved_cols)
    explicit_set = set(contract.explicit_cols)

    groups: dict[str, list[str]] = {"explicit": []}
    for _, group_name in contract.prefix_groups:
        groups[group_name] = []
    groups["other"] = []

    seen: set[str] = set()
    for col in columns:
        if col in seen or col in reserved:
            continue
        seen.add(col)

        if col in explicit_set:
            groups["explicit"].append(col)
            continue

        matched_group: str | None = None
        for prefix, group_name in contract.prefix_groups:
            if col.startswith(prefix):
                matched_group = group_name
                break

        if matched_group is not None:
            groups[matched_group].append(col)
        else:
            groups["other"].append(col)

    return groups


def _is_numeric_arrow_type(dtype: pa.DataType) -> bool:
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype)


def _pick_smallest_signed_int(min_val: int | float, max_val: int | float) -> pa.DataType:
    for arrow_type, low, high in _INT_LIMITS:
        if low <= min_val and max_val <= high:
            return arrow_type
    return pa.int64()


def _arrow_int_bitwidth(dtype: pa.DataType) -> int:
    if pa.types.is_int8(dtype) or pa.types.is_uint8(dtype):
        return 8
    if pa.types.is_int16(dtype) or pa.types.is_uint16(dtype):
        return 16
    if pa.types.is_int32(dtype) or pa.types.is_uint32(dtype):
        return 32
    return 64


def collect_parquet_column_stats(
    parquet_paths: Sequence[str | Path],
    columns: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Parquet Footer Row-Group 메타데이터만 읽어 컬럼별 전역 min/max 통계를 집계합니다.

    데이터 본문을 메모리에 올리지 않으므로 대용량 파일에서도 OOM 없이 즉시 실행됩니다.
    """
    target_cols = set(columns) if columns is not None else None
    acc: dict[str, dict[str, Any]] = {}

    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        meta = pf.metadata
        for rg_idx in range(meta.num_row_groups):
            rg = meta.row_group(rg_idx)
            for col_idx in range(rg.num_columns):
                col_meta = rg.column(col_idx)
                name = col_meta.path_in_schema
                if target_cols is not None and name not in target_cols:
                    continue
                if schema.get_field_index(name) == -1:
                    continue
                field_type = schema.field(name).type
                if not _is_numeric_arrow_type(field_type):
                    continue

                entry = acc.setdefault(
                    name,
                    {
                        "min": None,
                        "max": None,
                        "null_count": 0,
                        "complete": True,
                    },
                )
                stats = col_meta.statistics
                if stats is None or not stats.has_min_max:
                    entry["complete"] = False
                    continue

                if stats.null_count is not None:
                    entry["null_count"] += int(stats.null_count)

                rg_min = stats.min
                rg_max = stats.max
                if rg_min is None or rg_max is None:
                    continue
                if isinstance(rg_min, (int, float)) and isinstance(rg_max, (int, float)):
                    if np.isnan(rg_min) or np.isnan(rg_max):
                        entry["complete"] = False
                        continue
                    entry["min"] = rg_min if entry["min"] is None else min(entry["min"], rg_min)
                    entry["max"] = rg_max if entry["max"] is None else max(entry["max"], rg_max)
                else:
                    entry["complete"] = False

    # 통계가 불완전한 컬럼은 잘못된 축소를 막기 위해 min/max를 비활성화합니다.
    cleaned: dict[str, dict[str, Any]] = {}
    for name, info in acc.items():
        if info["complete"] and info["min"] is not None and info["max"] is not None:
            cleaned[name] = {
                "min": info["min"],
                "max": info["max"],
                "null_count": info["null_count"],
            }
    return cleaned


def infer_safe_dtypes(
    schema: pa.Schema,
    eda_summary: Mapping[str, Any] | None = None,
    column_stats: Mapping[str, Mapping[str, Any]] | None = None,
    contract: DatasetContract = DEFAULT_CONTRACT,
    downcast_float64_to_float32: bool = True,
    downcast_int_without_stats: bool = False,
) -> dict[str, pa.DataType]:
    """Parquet 스키마와 통계 정보를 바탕으로 안전한 다운캐스팅 PyArrow 타입을 결정합니다.

    - `row_id`는 `int64`, `clicked`는 `int8`, `fold_id`는 `int8`, `is_oot_val`은 `bool`로 고정합니다.
    - `ID` 및 `seq`는 원본 타입을 그대로 보존합니다.
    - 정수형 컬럼은 `min`/`max` 통계가 확인될 때만 `int8`/`int16`/`int32`로 안전 축소하며,
      통계가 없으면 `downcast_int_without_stats=False` 기준 원본 타입을 유지해 오버플로우를 차단합니다.
    - `float64` 컬럼은 `float32` 최대 표현 범위 내일 때 `float32`로 축소합니다.
    """
    eda_map = extract_eda_feature_map(eda_summary)
    dtype_map: dict[str, pa.DataType] = {}

    for field in schema:
        name = field.name
        ftype = field.type

        if name == contract.row_id_col:
            dtype_map[name] = pa.int64()
            continue
        if name == contract.target_col or name == contract.fold_id_col:
            dtype_map[name] = pa.int8()
            continue
        if name == contract.oot_col:
            dtype_map[name] = pa.bool_()
            continue
        if name in (contract.id_col, contract.seq_col):
            dtype_map[name] = ftype
            continue

        stat_entry = (column_stats or {}).get(name) or eda_map.get(name) or {}
        min_val = stat_entry.get("min")
        max_val = stat_entry.get("max")
        has_bounds = min_val is not None and max_val is not None

        if pa.types.is_integer(ftype):
            if has_bounds:
                candidate = _pick_smallest_signed_int(min_val, max_val)
                # 이미 더 작은 signed/unsigned 타입이면 불필요하게 키우지 않습니다.
                if _arrow_int_bitwidth(candidate) >= _arrow_int_bitwidth(ftype):
                    dtype_map[name] = ftype
                else:
                    dtype_map[name] = candidate
            elif downcast_int_without_stats and _arrow_int_bitwidth(ftype) > 32:
                dtype_map[name] = pa.int32()
            else:
                dtype_map[name] = ftype
            continue

        if pa.types.is_floating(ftype):
            if pa.types.is_float64(ftype) and downcast_float64_to_float32:
                if has_bounds and (
                    abs(float(min_val)) > _FLOAT32_MAX or abs(float(max_val)) > _FLOAT32_MAX
                ):
                    dtype_map[name] = pa.float64()
                else:
                    dtype_map[name] = pa.float32()
            else:
                dtype_map[name] = ftype
            continue

        dtype_map[name] = ftype

    return dtype_map


def _column_min_max(arr: pa.ChunkedArray | pa.Array) -> tuple[float | int | None, float | int | None]:
    if len(arr) == 0 or arr.null_count == len(arr):
        return None, None
    res = pc.min_max(arr)
    if res is None:
        return None, None
    min_scalar = res["min"].as_py()
    max_scalar = res["max"].as_py()
    return min_scalar, max_scalar


def _safe_cast_arrow_column(
    col_arr: pa.ChunkedArray,
    target_type: pa.DataType,
) -> tuple[pa.ChunkedArray | pa.Array, pa.DataType]:
    src_type = col_arr.type
    if src_type == target_type:
        return col_arr, src_type

    # 정수 목표 타입인데 원본이 float인 경우: NaN/NULL 또는 소수점 존재 시 float32로 안전 유지
    if pa.types.is_integer(target_type) and pa.types.is_floating(src_type):
        has_null = col_arr.null_count > 0
        has_nan = bool(pc.any(pc.is_nan(col_arr)).as_py()) if len(col_arr) > 0 else False
        if has_null or has_nan:
            fallback_type = pa.float32() if pa.types.is_float64(src_type) else src_type
            return pc.cast(col_arr, fallback_type, safe=False), fallback_type

        if len(col_arr) > 0:
            is_whole = pc.all(pc.equal(col_arr, pc.trunc(col_arr))).as_py()
            if not is_whole:
                fallback_type = pa.float32() if pa.types.is_float64(src_type) else src_type
                return pc.cast(col_arr, fallback_type, safe=False), fallback_type

    # 정수 범위 검증: 목표 정수 타입으로 표현 불가능하면 수용 가능한 상위 정수 타입으로 자동 상향
    if pa.types.is_integer(target_type) and (
        pa.types.is_integer(src_type) or pa.types.is_floating(src_type)
    ):
        c_min, c_max = _column_min_max(col_arr)
        effective_type = target_type
        if c_min is not None and c_max is not None:
            needed = _pick_smallest_signed_int(c_min, c_max)
            if _arrow_int_bitwidth(needed) > _arrow_int_bitwidth(target_type):
                effective_type = needed
        if src_type == effective_type:
            return col_arr, src_type
        return pc.cast(col_arr, effective_type, safe=True), effective_type

    # float64 -> float32: 정밀도 손실 허용(safe=False)하되 표현 범위 초과 시 float64 유지
    if pa.types.is_float32(target_type) and pa.types.is_float64(src_type):
        c_min, c_max = _column_min_max(col_arr)
        if c_min is not None and c_max is not None:
            if abs(float(c_min)) > _FLOAT32_MAX or abs(float(c_max)) > _FLOAT32_MAX:
                return col_arr, src_type
        return pc.cast(col_arr, pa.float32(), safe=False), pa.float32()

    return pc.cast(col_arr, target_type, safe=True), target_type


def downcast_arrow_table(
    table: pa.Table,
    dtype_map: Mapping[str, pa.DataType | str] | None = None,
    contract: DatasetContract = DEFAULT_CONTRACT,
    downcast_float64_to_float32: bool = True,
) -> pa.Table:
    """PyArrow 테이블의 수치/정수 컬럼을 값·NULL 손실 없이 안전하게 다운캐스팅합니다.

    `dtype_map`이 생략되면 테이블 내 실제 최솟값/최댓값을 측정해 최적 정수·실수 타입을 자동 산출합니다.
    정수 범위 초과 또는 NaN/소수점 포함 시 상위 안전 타입(`int32`/`int64`/`float32`)을 유지합니다.
    """
    if dtype_map is None:
        inferred_stats: dict[str, dict[str, Any]] = {}
        for field in table.schema:
            if _is_numeric_arrow_type(field.type):
                c_min, c_max = _column_min_max(table.column(field.name))
                if c_min is not None and c_max is not None and not (
                    isinstance(c_min, float) and (np.isnan(c_min) or np.isnan(c_max))
                ):
                    inferred_stats[field.name] = {"min": c_min, "max": c_max}
        resolved_map = infer_safe_dtypes(
            table.schema,
            column_stats=inferred_stats,
            contract=contract,
            downcast_float64_to_float32=downcast_float64_to_float32,
        )
    else:
        resolved_map = {k: resolve_arrow_dtype(v) for k, v in dtype_map.items()}

    new_columns: list[pa.ChunkedArray | pa.Array] = []
    new_fields: list[pa.Field] = []
    for field in table.schema:
        col_arr = table.column(field.name)
        target_type = resolved_map.get(field.name, field.type)
        cast_arr, actual_type = _safe_cast_arrow_column(col_arr, target_type)
        new_columns.append(cast_arr)
        new_fields.append(pa.field(field.name, actual_type, nullable=field.nullable))

    return pa.Table.from_arrays(new_columns, schema=pa.schema(new_fields))


def downcast_dataframe(
    df: pd.DataFrame,
    dtype_map: Mapping[str, pa.DataType | str] | None = None,
    contract: DatasetContract = DEFAULT_CONTRACT,
    downcast_float64_to_float32: bool = True,
) -> pd.DataFrame:
    """Pandas DataFrame(3.x 호환)을 값·결측치 손실 없이 안전하게 다운캐스팅합니다.

    NaN이 포함된 부동소수점 컬럼은 정수형으로 변환하지 않고 `float32`로 유지하며,
    nullable 정수형(`<NA>` 포함)은 nullable 하위 정수형을 유지합니다.
    """
    resolved_map = (
        {k: resolve_arrow_dtype(v) for k, v in dtype_map.items()} if dtype_map is not None else {}
    )
    out_cols: dict[str, pd.Series | np.ndarray] = {}

    for col in df.columns:
        series = df[col]
        if col in (contract.id_col, contract.seq_col):
            out_cols[col] = series
            continue

        if col == contract.row_id_col and pd.api.types.is_integer_dtype(series.dtype):
            if not series.isna().any():
                out_cols[col] = series.astype(np.int64)
                continue

        if col in (contract.target_col, contract.fold_id_col) and pd.api.types.is_integer_dtype(
            series.dtype
        ):
            if not series.isna().any():
                out_cols[col] = series.astype(np.int8)
                continue

        if pd.api.types.is_bool_dtype(series.dtype):
            out_cols[col] = series
            continue

        if pd.api.types.is_integer_dtype(series.dtype):
            has_na = bool(series.isna().any())
            if has_na:
                valid = series.dropna()
                if valid.empty:
                    out_cols[col] = series
                    continue
                c_min, c_max = int(valid.min()), int(valid.max())
                needed = _pick_smallest_signed_int(c_min, c_max)
                target_pa = resolved_map.get(col, needed)
                if pa.types.is_integer(target_pa) and _arrow_int_bitwidth(needed) > _arrow_int_bitwidth(
                    target_pa
                ):
                    target_pa = needed
                nullable_map = {
                    pa.int8(): "Int8",
                    pa.int16(): "Int16",
                    pa.int32(): "Int32",
                    pa.int64(): "Int64",
                }
                out_cols[col] = series.astype(nullable_map.get(target_pa, "Int64"))
            else:
                if series.empty:
                    out_cols[col] = series
                    continue
                c_min, c_max = int(series.min()), int(series.max())
                needed = _pick_smallest_signed_int(c_min, c_max)
                target_pa = resolved_map.get(col, needed)
                if not pa.types.is_integer(target_pa) or _arrow_int_bitwidth(
                    needed
                ) > _arrow_int_bitwidth(target_pa):
                    target_pa = needed
                np_map = {
                    pa.int8(): np.int8,
                    pa.int16(): np.int16,
                    pa.int32(): np.int32,
                    pa.int64(): np.int64,
                }
                out_cols[col] = series.astype(np_map.get(target_pa, np.int64))
            continue

        if pd.api.types.is_float_dtype(series.dtype):
            if downcast_float64_to_float32 and str(series.dtype) in ("float64", "Float64"):
                valid = series.dropna()
                if valid.empty or (
                    abs(float(valid.min())) <= _FLOAT32_MAX
                    and abs(float(valid.max())) <= _FLOAT32_MAX
                ):
                    target_dtype = "Float32" if str(series.dtype) == "Float64" else np.float32
                    out_cols[col] = series.astype(target_dtype)
                    continue
            out_cols[col] = series
            continue

        out_cols[col] = series

    return pd.DataFrame(out_cols, index=df.index)
