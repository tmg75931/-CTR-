"""CTR 예측용 교호작용 키 생성 및 폴드 안전 빈도·조건부 통계 인코더 모듈."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

CORE_INTERACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("inventory_id", "l_feat_14"),
    ("inventory_id", "hour"),
    ("age_group", "gender"),
    ("day_of_week", "hour"),
)

EXTENDED_INTERACTION_PAIRS: tuple[tuple[str, str], ...] = (
    *CORE_INTERACTION_PAIRS,
    ("inventory_id", "age_group"),
    ("inventory_id", "gender"),
    ("l_feat_14", "hour"),
)

_NA_HASH_SENTINEL = np.uint64(0x9E3779B97F4A7C15)
_MASK_64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_MASK_63 = np.uint64(0x7FFFFFFFFFFFFFFF)
_MIX_MUL_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_MUL_2 = np.uint64(0x94D049BB133111EB)
_FLOAT_INT_LIMIT = float(2**53)


def _normalize_for_mapping(series: pd.Series) -> pd.Series:
    """Pandas 3.x 호환 매핑을 위해 Categorical 컬럼을 원시 object 표현으로 정규화합니다."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.astype(object)
    return series


def _hash_series_uint64(series: pd.Series) -> np.ndarray:
    """결측치와 정수/실수/문자열 dtype 차이에 안전한 결정론적 64비트 해시를 생성합니다."""
    n = len(series)
    if n == 0:
        return np.empty(0, dtype=np.uint64)

    s = _normalize_for_mapping(series)
    na_mask = s.isna().to_numpy(dtype=bool, copy=False)

    if pd.api.types.is_bool_dtype(s.dtype):
        filled = np.zeros(n, dtype=np.int8)
        valid_mask = ~na_mask
        if np.any(valid_mask):
            filled[valid_mask] = s.iloc[valid_mask].to_numpy(dtype=np.int8)
        hashed = pd.util.hash_array(filled, categorize=False).astype(np.uint64, copy=False)
    elif pd.api.types.is_integer_dtype(s.dtype):
        filled = np.zeros(n, dtype=np.int64)
        valid_mask = ~na_mask
        if np.any(valid_mask):
            filled[valid_mask] = s.iloc[valid_mask].to_numpy(dtype=np.int64)
        hashed = pd.util.hash_array(filled, categorize=False).astype(np.uint64, copy=False)
    elif pd.api.types.is_float_dtype(s.dtype):
        arr_f = s.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        arr_f = np.where(arr_f == 0.0, 0.0, arr_f)
        valid_mask = ~na_mask
        # 원소 단위로 정수 표현 가능 값(1.0 == 1)과 소수 값(2.5)을 분리 해싱하여 혼합 float에서도 해시 일관성 보장
        int_mask = valid_mask & np.isfinite(arr_f) & (np.abs(arr_f) <= _FLOAT_INT_LIMIT)
        if np.any(int_mask):
            int_mask = int_mask & (arr_f == np.trunc(arr_f))
        float_only_mask = valid_mask & (~int_mask)

        hashed = np.empty(n, dtype=np.uint64)
        if np.any(int_mask):
            hashed[int_mask] = pd.util.hash_array(
                arr_f[int_mask].astype(np.int64), categorize=False
            ).astype(np.uint64, copy=False)
        if np.any(float_only_mask):
            hashed[float_only_mask] = pd.util.hash_array(
                arr_f[float_only_mask], categorize=False
            ).astype(np.uint64, copy=False)
    else:
        obj_arr = np.full(n, "__MISSING__", dtype=object)
        valid_mask = ~na_mask
        if np.any(valid_mask):
            obj_arr[valid_mask] = s.iloc[valid_mask].astype(str).to_numpy(dtype=object)
        hashed = pd.util.hash_array(obj_arr, categorize=False).astype(np.uint64, copy=False)

    if np.any(na_mask):
        hashed = hashed.copy()
        hashed[na_mask] = _NA_HASH_SENTINEL
    return hashed


def _mix_uint64_hashes(h_list: Sequence[np.ndarray]) -> np.ndarray:
    """여러 64비트 해시 배열을 순서 의존적으로 결합해 비음수 int64 키를 생성합니다."""
    acc = h_list[0].copy()
    for h in h_list[1:]:
        acc = (acc ^ (h + _NA_HASH_SENTINEL + ((acc << np.uint64(6)) & _MASK_64) + (acc >> np.uint64(2)))) & _MASK_64
        acc = (acc ^ (acc >> np.uint64(30))) * _MIX_MUL_1
        acc = (acc ^ (acc >> np.uint64(27))) * _MIX_MUL_2
        acc = acc ^ (acc >> np.uint64(31))
    return (acc & _MASK_63).astype(np.int64)


def make_interaction_column_name(
    cols: Sequence[str],
    prefix: str = "ix_",
    sep: str = "__",
) -> str:
    """교호작용 피처의 표준 컬럼명을 생성합니다."""
    return f"{prefix}{sep.join(cols)}"


def build_interaction_features(
    df: pd.DataFrame,
    interaction_pairs: Sequence[Sequence[str]] = CORE_INTERACTION_PAIRS,
    output_format: Literal["hash_int64", "string"] = "hash_int64",
    prefix: str = "ix_",
    sep: str = "__",
    keep_original: bool = False,
) -> pd.DataFrame:
    """존재하는 컬럼만 안전하게 결합해 교호작용 키 피처를 생성합니다.

    타 행이나 외부 통계를 전혀 참조하지 않는 순수 행 단위 연산입니다.

    Args:
        df: 입력 DataFrame.
        interaction_pairs: 결합할 컬럼 쌍(또는 튜플) 목록.
        output_format: `"hash_int64"`(메모리 효율적 63비트 비음수 정수 해시) 또는 `"string"`.
        prefix: 생성 컬럼 접두사.
        sep: 컬럼명 결합 구분자.
        keep_original: `True`면 원본 컬럼에 결합 피처를 추가한 새 DataFrame 반환,
            `False`면 새로 생성된 교호작용 컬럼만 반환.
    """
    if output_format not in ("hash_int64", "string"):
        raise ValueError(f"지원하지 않는 output_format입니다: {output_format!r}")

    available_cols = set(df.columns)
    new_cols: dict[str, pd.Series | np.ndarray] = {}
    hash_cache: dict[str, np.ndarray] = {}
    str_cache: dict[str, pd.Series] = {}

    for raw_pair in interaction_pairs:
        pair = tuple(raw_pair)
        if len(pair) < 2:
            continue
        if any(col not in available_cols for col in pair):
            continue

        out_name = make_interaction_column_name(pair, prefix=prefix, sep=sep)
        if output_format == "hash_int64":
            pair_hashes: list[np.ndarray] = []
            for col in pair:
                if col not in hash_cache:
                    hash_cache[col] = _hash_series_uint64(df[col])
                pair_hashes.append(hash_cache[col])
            new_cols[out_name] = _mix_uint64_hashes(pair_hashes)
        else:
            for col in pair:
                if col not in str_cache:
                    s_norm = _normalize_for_mapping(df[col])
                    str_cache[col] = s_norm.astype("string").fillna("__MISSING__")
            combined = str_cache[pair[0]]
            for col in pair[1:]:
                combined = combined + sep + str_cache[col]
            new_cols[out_name] = combined

    out_df = pd.DataFrame(new_cols, index=df.index)
    if keep_original:
        return pd.concat([df.copy(), out_df], axis=1)
    return out_df


class FoldFrequencyEncoder:
    """학습 폴드에서만 빈도를 집계해 검증/테스트 데이터 사전 포함 누수를 차단하는 인코더."""

    def __init__(
        self,
        columns: Sequence[str],
        suffix: str = "_freq",
        normalize: bool = False,
        add_log1p: bool = False,
        keep_original: bool = False,
        ignore_missing: bool = True,
    ) -> None:
        self.columns = tuple(columns)
        self.suffix = suffix
        self.normalize = normalize
        self.add_log1p = add_log1p
        self.keep_original = keep_original
        self.ignore_missing = ignore_missing

        self.fitted_columns_: list[str] = []
        self.freq_maps_: dict[str, pd.Series] = {}
        self.missing_counts_: dict[str, int] = {}
        self.n_train_rows_: int = 0
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame, y: Any = None) -> FoldFrequencyEncoder:
        """오직 `train_df` 내부 행만으로 범주 빈도를 집계합니다."""
        _ = y
        self.fitted_columns_ = []
        self.freq_maps_ = {}
        self.missing_counts_ = {}
        self.n_train_rows_ = int(len(train_df))

        for col in self.columns:
            if col not in train_df.columns:
                if self.ignore_missing:
                    continue
                raise KeyError(f"FoldFrequencyEncoder 대상 컬럼이 없습니다: {col!r}")

            s = _normalize_for_mapping(train_df[col])
            na_mask = s.isna()
            missing_cnt = int(na_mask.sum())
            valid_s = s.loc[~na_mask] if missing_cnt > 0 else s
            vc = valid_s.value_counts(dropna=True)

            self.fitted_columns_.append(col)
            self.freq_maps_[col] = vc
            self.missing_counts_[col] = missing_cnt

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 시 집계된 빈도 사전으로 매핑하며, 미등장 범주는 `0`으로 처리합니다."""
        if not self.is_fitted_:
            raise RuntimeError("transform 호출 전에 fit()을 먼저 실행해야 합니다.")

        count_dtype: Any = np.int32 if self.n_train_rows_ <= 2_000_000_000 else np.int64
        out_cols: dict[str, np.ndarray] = {}
        denom = float(max(self.n_train_rows_, 1))

        for col in self.fitted_columns_:
            if col not in df.columns:
                if self.ignore_missing:
                    continue
                raise KeyError(f"transform 입력에 컬럼이 없습니다: {col!r}")

            s = _normalize_for_mapping(df[col])
            na_mask = s.isna().to_numpy(dtype=bool, copy=False)
            freq_map = self.freq_maps_[col]

            if len(freq_map) == 0:
                counts = np.zeros(len(df), dtype=np.float64)
            else:
                mapped = s.map(freq_map)
                counts = mapped.to_numpy(dtype=np.float64, na_value=0.0, copy=True)

            if np.any(na_mask):
                counts[na_mask] = float(self.missing_counts_[col])

            base_name = f"{col}{self.suffix}"
            if self.normalize:
                out_cols[base_name] = (counts / denom).astype(np.float32)
            else:
                out_cols[base_name] = counts.astype(count_dtype)

            if self.add_log1p:
                out_cols[f"{base_name}_log1p"] = np.log1p(counts).astype(np.float32)

        out_df = pd.DataFrame(out_cols, index=df.index)
        if self.keep_original:
            return pd.concat([df.copy(), out_df], axis=1)
        return out_df

    def fit_transform(self, train_df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """`train_df`에 fit 후 동일 데이터를 안전하게 변환합니다."""
        return self.fit(train_df, y=y).transform(train_df)

    def get_feature_names_out(self) -> list[str]:
        """생성되는 출력 피처 이름 목록을 반환합니다."""
        names: list[str] = []
        for col in self.fitted_columns_:
            base_name = f"{col}{self.suffix}"
            names.append(base_name)
            if self.add_log1p:
                names.append(f"{base_name}_log1p")
        return names


class FoldConditionalStatsEncoder:
    """학습 폴드에서만 고유 파트너 수(`nunique`)와 조건부 점유율(`P(B|A)`)을 집계하는 인코더."""

    def __init__(
        self,
        pair_specs: Sequence[tuple[str, str]] = (
            ("inventory_id", "l_feat_14"),
            ("l_feat_14", "inventory_id"),
            ("inventory_id", "hour"),
        ),
        keep_original: bool = False,
        ignore_missing: bool = True,
    ) -> None:
        self.pair_specs = tuple(pair_specs)
        self.keep_original = keep_original
        self.ignore_missing = ignore_missing

        self.fitted_specs_: list[tuple[str, str]] = []
        self.nunique_maps_: dict[tuple[str, str], pd.Series] = {}
        self.share_maps_: dict[tuple[str, str], pd.Series] = {}
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame, y: Any = None) -> FoldConditionalStatsEncoder:
        """오직 `train_df` 내부 행만으로 고유 파트너 수 및 조건부 노출 비율을 집계합니다."""
        _ = y
        self.fitted_specs_ = []
        self.nunique_maps_ = {}
        self.share_maps_ = {}

        for key_col, partner_col in self.pair_specs:
            if key_col not in train_df.columns or partner_col not in train_df.columns:
                if self.ignore_missing:
                    continue
                raise KeyError(
                    f"FoldConditionalStatsEncoder 컬럼 누락: ({key_col!r}, {partner_col!r})"
                )

            k_s = _normalize_for_mapping(train_df[key_col])
            p_s = _normalize_for_mapping(train_df[partner_col])
            valid_mask = (~k_s.isna()) & (~p_s.isna())

            if not bool(valid_mask.any()):
                nunique_s = pd.Series(dtype=np.int32)
                share_s = pd.Series(dtype=np.float32)
            else:
                k_valid = k_s.loc[valid_mask]
                p_valid = p_s.loc[valid_mask]
                pair_keys = _mix_uint64_hashes(
                    [_hash_series_uint64(k_valid), _hash_series_uint64(p_valid)]
                )
                pair_df = pd.DataFrame(
                    {"k": k_valid.to_numpy(), "pk": pair_keys},
                )
                key_totals = pair_df.groupby("k", sort=False).size().astype(np.float64)
                pair_agg = pair_df.groupby("pk", sort=False).agg(
                    k=("k", "first"),
                    cnt=("pk", "size"),
                )
                nunique_s = pair_agg.groupby("k", sort=False).size().astype(np.int32)
                share_vals = pair_agg["cnt"].astype(np.float64) / pair_agg["k"].map(key_totals)
                share_s = share_vals.astype(np.float32)

            spec = (key_col, partner_col)
            self.fitted_specs_.append(spec)
            self.nunique_maps_[spec] = nunique_s
            self.share_maps_[spec] = share_s

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습된 고유 파트너 수 및 조건부 노출 비율을 매핑합니다(미등장 조합은 `0`)."""
        if not self.is_fitted_:
            raise RuntimeError("transform 호출 전에 fit()을 먼저 실행해야 합니다.")

        out_cols: dict[str, np.ndarray] = {}

        for spec in self.fitted_specs_:
            key_col, partner_col = spec
            if key_col not in df.columns or partner_col not in df.columns:
                if self.ignore_missing:
                    continue
                raise KeyError(
                    f"transform 입력에 컬럼이 없습니다: ({key_col!r}, {partner_col!r})"
                )

            k_s = _normalize_for_mapping(df[key_col])
            p_s = _normalize_for_mapping(df[partner_col])

            nunique_map = self.nunique_maps_[spec]
            if len(nunique_map) == 0:
                nunique_arr = np.zeros(len(df), dtype=np.int32)
            else:
                nunique_arr = k_s.map(nunique_map).to_numpy(
                    dtype=np.int32, na_value=0, copy=True
                )
            out_cols[f"{key_col}__nunique_{partner_col}"] = nunique_arr

            share_map = self.share_maps_[spec]
            share_arr = np.zeros(len(df), dtype=np.float32)
            if len(share_map) > 0:
                valid_mask = (~k_s.isna()) & (~p_s.isna())
                if bool(valid_mask.any()):
                    pair_keys = _mix_uint64_hashes(
                        [
                            _hash_series_uint64(k_s.loc[valid_mask]),
                            _hash_series_uint64(p_s.loc[valid_mask]),
                        ]
                    )
                    mapped = pd.Series(pair_keys).map(share_map).to_numpy(
                        dtype=np.float32, na_value=0.0, copy=True
                    )
                    share_arr[valid_mask.to_numpy(dtype=bool, copy=False)] = mapped
            out_cols[f"{key_col}_{partner_col}__share_of_{key_col}"] = share_arr

        out_df = pd.DataFrame(out_cols, index=df.index)
        if self.keep_original:
            return pd.concat([df.copy(), out_df], axis=1)
        return out_df

    def fit_transform(self, train_df: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """`train_df`에 fit 후 동일 데이터를 안전하게 변환합니다."""
        return self.fit(train_df, y=y).transform(train_df)

    def get_feature_names_out(self) -> list[str]:
        """생성되는 출력 피처 이름 목록을 반환합니다."""
        names: list[str] = []
        for key_col, partner_col in self.fitted_specs_:
            names.append(f"{key_col}__nunique_{partner_col}")
            names.append(f"{key_col}_{partner_col}__share_of_{key_col}")
        return names
