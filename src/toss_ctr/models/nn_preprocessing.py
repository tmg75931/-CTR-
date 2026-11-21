"""PyTorch 비의존 순수 NumPy/Pandas 기반 DeepCTR 학습 폴드 전용 전처리기 모듈."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

CAT_MISSING_ID: int = 0
CAT_UNK_ID: int = 1


def signed_log1p(x: ArrayLike) -> np.ndarray:
    """부호를 보존하는 로그 변환 `sign(x) * log1p(|x|)`를 계산합니다.

    큰 스케일 편차와 음수 값을 동시에 가진 정형 피처의 분포를 안정화합니다.
    """
    arr = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if np.any(finite):
        vals = arr[finite]
        out[finite] = np.sign(vals) * np.log1p(np.abs(vals))
    return out


def _normalize_categorical_series(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """범주형 Series를 일관된 문자열 키와 결측 마스크(`True`=결측)로 정규화합니다.

    `1`(int)과 `1.0`(float)이 train/test 간 서로 다른 dtype으로 들어와도 동일한 키 `"1"`로 매핑합니다.
    """
    n_rows = len(series)
    if n_rows == 0:
        return pd.Series([], dtype="object", index=series.index), np.zeros(0, dtype=bool)

    na_mask = series.isna().to_numpy(dtype=bool)

    if pd.api.types.is_bool_dtype(series.dtype):
        keys = series.astype("object").where(~na_mask, "").astype(str)
        return keys, na_mask

    if pd.api.types.is_integer_dtype(series.dtype):
        int_arr = series.fillna(0).to_numpy(dtype=np.int64, copy=False)
        keys = pd.Series(int_arr, index=series.index).astype(str)
        return keys, na_mask

    if pd.api.types.is_float_dtype(series.dtype):
        arr = series.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        finite = np.isfinite(arr)
        missing_mask = na_mask | (~finite)
        if not np.any(finite):
            return pd.Series([""] * n_rows, index=series.index, dtype="object"), missing_mask

        finite_vals = arr[finite]
        # 모든 유효 float 값이 안전한 int64 표현 범위 내 정수면 일괄 정수 문자열로 변환
        is_all_whole = bool(
            np.all((finite_vals == np.trunc(finite_vals)) & (np.abs(finite_vals) <= 9e15))
        )
        if is_all_whole:
            int_arr = np.zeros(n_rows, dtype=np.int64)
            int_arr[finite] = finite_vals.astype(np.int64)
            keys = pd.Series(int_arr, index=series.index).astype(str)
            return keys, missing_mask

        out_list: list[str] = [""] * n_rows
        for idx in np.flatnonzero(finite):
            v = float(arr[idx])
            if v.is_integer() and abs(v) <= 9e15:
                out_list[idx] = str(int(v))
            else:
                out_list[idx] = str(v)
        return pd.Series(out_list, index=series.index, dtype="object"), missing_mask

    # 문자열 / 카테고리 / 오브젝트 타입
    if isinstance(series.dtype, pd.CategoricalDtype):
        str_series = series.astype("object")
    else:
        str_series = series

    keys = str_series.fillna("").astype(str).str.strip()
    empty_mask = (keys == "").to_numpy(dtype=bool)
    return keys, (na_mask | empty_mask)


class FoldCategoricalIndexer:
    """학습 폴드 데이터만으로 범주 어휘를 구축하는 누수 방지 범주형 인덱서.

    - `0`: `MISSING / PAD` (결측치 및 빈 문자열)
    - `1`: `UNK` (`min_freq` 미만 희귀 값 및 검증/테스트 전용 미등장 카테고리)
    - `2+`: 학습 폴드에서 `min_freq` 이상 관측된 유효 카테고리
    """

    MISSING_ID: int = CAT_MISSING_ID
    UNK_ID: int = CAT_UNK_ID

    def __init__(
        self,
        cat_cols: Sequence[str] | None = None,
        min_freq: int = 5,
        max_cardinality: int | None = 100_000,
    ) -> None:
        if min_freq < 1:
            raise ValueError(f"min_freq는 1 이상이어야 합니다: {min_freq}")
        if max_cardinality is not None and max_cardinality < 2:
            raise ValueError(f"max_cardinality는 2 이상이어야 합니다: {max_cardinality}")

        self.cat_cols: list[str] | None = list(cat_cols) if cat_cols is not None else None
        self.min_freq = int(min_freq)
        self.max_cardinality = int(max_cardinality) if max_cardinality is not None else None

        self.cat_cols_: list[str] = []
        self.mappings_: dict[str, dict[str, int]] = {}
        self.vocab_sizes_: dict[str, int] = {}
        self._counters: dict[str, Counter[str]] = {}
        self.is_fitted_: bool = False

    def reset(self) -> None:
        """누적 카운터와 학습된 매핑을 초기화합니다."""
        self.cat_cols_ = []
        self.mappings_ = {}
        self.vocab_sizes_ = {}
        self._counters = {}
        self.is_fitted_ = False

    def partial_fit(
        self,
        df: pd.DataFrame,
        cat_cols: Sequence[str] | None = None,
    ) -> FoldCategoricalIndexer:
        """청크 단위 DataFrame에서 학습 폴드 범주 빈도를 누적합니다."""
        target_cols = list(cat_cols) if cat_cols is not None else self.cat_cols
        if target_cols is None:
            raise ValueError("cat_cols를 지정해야 합니다.")

        missing_cols = [c for c in target_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame에 범주형 컬럼이 없습니다: {missing_cols}")

        if not self.cat_cols_:
            self.cat_cols_ = list(target_cols)
            self._counters = {c: Counter() for c in self.cat_cols_}

        for col in self.cat_cols_:
            keys, na_mask = _normalize_categorical_series(df[col])
            valid_keys = keys.loc[~na_mask]
            if not valid_keys.empty:
                vc = valid_keys.value_counts(sort=False)
                self._counters[col].update(
                    {str(k): int(v) for k, v in zip(vc.index.tolist(), vc.to_numpy().tolist())}
                )

        return self

    def finalize_fit(self) -> FoldCategoricalIndexer:
        """누적된 빈도를 바탕으로 최종 `category -> int32` 매핑을 확정합니다."""
        self.mappings_ = {}
        self.vocab_sizes_ = {}

        for col in self.cat_cols_:
            counter = self._counters.get(col, Counter())
            eligible = [(k, freq) for k, freq in counter.items() if freq >= self.min_freq]
            eligible.sort(key=lambda item: (-item[1], item[0]))

            if self.max_cardinality is not None:
                eligible = eligible[: max(self.max_cardinality - 2, 0)]

            col_map = {k: idx + 2 for idx, (k, _) in enumerate(eligible)}
            self.mappings_[col] = col_map
            self.vocab_sizes_[col] = len(col_map) + 2

        self.is_fitted_ = True
        return self

    def fit(
        self,
        train_df: pd.DataFrame,
        cat_cols: Sequence[str] | None = None,
    ) -> FoldCategoricalIndexer:
        """오직 학습 폴드 DataFrame만으로 범주 어휘를 구축합니다."""
        self.reset()
        resolved_cols = list(cat_cols) if cat_cols is not None else self.cat_cols
        if resolved_cols is None:
            resolved_cols = [
                c
                for c in train_df.columns
                if isinstance(train_df[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(train_df[c].dtype)
                or pd.api.types.is_object_dtype(train_df[c].dtype)
            ]
        self.partial_fit(train_df, cat_cols=resolved_cols)
        return self.finalize_fit()

    @property
    def vocab_sizes(self) -> dict[str, int]:
        """컬럼별 임베딩 크기(`MISSING(0)`, `UNK(1)` 포함)를 반환합니다."""
        if not self.is_fitted_:
            raise RuntimeError("FoldCategoricalIndexer.fit()을 먼저 호출해야 합니다.")
        return dict(self.vocab_sizes_)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """범주형 컬럼들을 `(N, num_cat_cols)` `int32` 인덱스 행렬로 변환합니다."""
        if not self.is_fitted_:
            raise RuntimeError("FoldCategoricalIndexer.fit()을 먼저 호출해야 합니다.")

        n_rows = len(df)
        n_cols = len(self.cat_cols_)
        if n_cols == 0:
            return np.zeros((n_rows, 0), dtype=np.int32)

        missing_cols = [c for c in self.cat_cols_ if c not in df.columns]
        if missing_cols:
            raise ValueError(f"transform 대상 DataFrame에 컬럼이 없습니다: {missing_cols}")

        out = np.empty((n_rows, n_cols), dtype=np.int32)
        for col_idx, col in enumerate(self.cat_cols_):
            keys, na_mask = _normalize_categorical_series(df[col])
            col_map = self.mappings_[col]
            if not col_map:
                col_ids = np.where(na_mask, CAT_MISSING_ID, CAT_UNK_ID).astype(np.int32)
            else:
                mapped = keys.map(col_map)
                col_ids = mapped.fillna(CAT_UNK_ID).to_numpy(dtype=np.int32, copy=True)
                if np.any(na_mask):
                    col_ids[na_mask] = CAT_MISSING_ID
            out[:, col_idx] = col_ids

        return out

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        cat_cols: Sequence[str] | None = None,
    ) -> np.ndarray:
        """학습 폴드에 `fit`한 직후 동일 데이터에 대해 인덱싱 변환을 수행합니다."""
        self.fit(train_df, cat_cols=cat_cols)
        return self.transform(train_df)

    def to_dict(self) -> dict[str, Any]:
        """인덱서 상태를 직렬화 가능한 딕셔너리로 변환합니다."""
        return {
            "cat_cols": self.cat_cols_,
            "min_freq": self.min_freq,
            "max_cardinality": self.max_cardinality,
            "mappings": self.mappings_,
            "vocab_sizes": self.vocab_sizes_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FoldCategoricalIndexer:
        """직렬화된 딕셔너리에서 `FoldCategoricalIndexer`를 복원합니다."""
        obj = cls(
            cat_cols=list(payload["cat_cols"]),
            min_freq=int(payload["min_freq"]),
            max_cardinality=payload.get("max_cardinality"),
        )
        obj.cat_cols_ = list(payload["cat_cols"])
        obj.mappings_ = {str(k): {str(ck): int(cv) for ck, cv in v.items()} for k, v in payload["mappings"].items()}
        obj.vocab_sizes_ = {str(k): int(v) for k, v in payload["vocab_sizes"].items()}
        obj.is_fitted_ = True
        return obj


class FoldNumericPreprocessor:
    """학습 폴드 통계만으로 정규화하고 결측 마스크를 생성하는 수치형 전처리기.

    파이프라인 순서:
    1. 비유한 값(`NaN`, `+inf`, `-inf`) 탐지 → 결측 지시 마스크(`float32`, `1.0` = 결측, `0.0` = 유효)
    2. 유효 관측치에만 선택적 `signed_log1p(x) = sign(x) * log1p(|x|)` 적용
    3. 학습 폴드 유효 관측치만의 `mean`, `std` (zero-std 시 `1.0` 방어)로 표준화
    4. 결측·비유한 위치는 `0.0`으로 안전 대치 (`float32` 반환)
    """

    def __init__(
        self,
        num_cols: Sequence[str] | None = None,
        apply_signed_log1p: bool = True,
        eps: float = 1e-6,
        clip_value: float | None = None,
    ) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps는 0보다 커야 합니다: {eps}")
        if clip_value is not None and clip_value <= 0.0:
            raise ValueError(f"clip_value는 양수여야 합니다: {clip_value}")

        self.num_cols: list[str] | None = list(num_cols) if num_cols is not None else None
        self.apply_signed_log1p = bool(apply_signed_log1p)
        self.eps = float(eps)
        self.clip_value = float(clip_value) if clip_value is not None else None

        self.num_cols_: list[str] = []
        self.mean_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.std_: np.ndarray = np.ones(0, dtype=np.float64)
        self._count: np.ndarray = np.zeros(0, dtype=np.int64)
        self._m2: np.ndarray = np.zeros(0, dtype=np.float64)
        self.is_fitted_: bool = False

    def reset(self) -> None:
        """누적 통계와 학습된 평균/표준편차를 초기화합니다."""
        self.num_cols_ = []
        self.mean_ = np.zeros(0, dtype=np.float64)
        self.std_ = np.ones(0, dtype=np.float64)
        self._count = np.zeros(0, dtype=np.int64)
        self._m2 = np.zeros(0, dtype=np.float64)
        self.is_fitted_ = False

    def _extract_raw_matrix(self, df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
        n_rows = len(df)
        n_cols = len(cols)
        if n_cols == 0:
            return np.zeros((n_rows, 0), dtype=np.float64)

        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame에 수치형 컬럼이 없습니다: {missing_cols}")

        mat = np.empty((n_rows, n_cols), dtype=np.float64)
        for idx, col in enumerate(cols):
            mat[:, idx] = df[col].to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        return mat

    def partial_fit(
        self,
        df: pd.DataFrame,
        num_cols: Sequence[str] | None = None,
    ) -> FoldNumericPreprocessor:
        """Chan의 병렬 분산 병합 공식으로 청크 단위 유효 관측치 통계를 누적합니다."""
        target_cols = list(num_cols) if num_cols is not None else self.num_cols
        if target_cols is None:
            raise ValueError("num_cols를 지정해야 합니다.")

        if not self.num_cols_:
            self.num_cols_ = list(target_cols)
            d = len(self.num_cols_)
            self.mean_ = np.zeros(d, dtype=np.float64)
            self.std_ = np.ones(d, dtype=np.float64)
            self._count = np.zeros(d, dtype=np.int64)
            self._m2 = np.zeros(d, dtype=np.float64)

        if not self.num_cols_:
            return self

        raw = self._extract_raw_matrix(df, self.num_cols_)
        for col_idx in range(len(self.num_cols_)):
            col_vals = raw[:, col_idx]
            valid = col_vals[np.isfinite(col_vals)]
            n_b = int(valid.size)
            if n_b == 0:
                continue

            if self.apply_signed_log1p:
                valid = np.sign(valid) * np.log1p(np.abs(valid))

            mean_b = float(np.mean(valid))
            m2_b = float(np.sum((valid - mean_b) ** 2))

            n_a = int(self._count[col_idx])
            if n_a == 0:
                self._count[col_idx] = n_b
                self.mean_[col_idx] = mean_b
                self._m2[col_idx] = m2_b
            else:
                n_total = n_a + n_b
                delta = mean_b - float(self.mean_[col_idx])
                self.mean_[col_idx] = float(self.mean_[col_idx]) + delta * (n_b / n_total)
                self._m2[col_idx] = float(self._m2[col_idx]) + m2_b + (delta ** 2) * (n_a * n_b / n_total)
                self._count[col_idx] = n_total

        return self

    def finalize_fit(self) -> FoldNumericPreprocessor:
        """누적된 유효 관측치 통계를 확정하고 zero-std 방어를 적용합니다."""
        d = len(self.num_cols_)
        std = np.ones(d, dtype=np.float64)
        for col_idx in range(d):
            cnt = int(self._count[col_idx])
            if cnt > 0:
                var = max(float(self._m2[col_idx]) / cnt, 0.0)
                s = float(np.sqrt(var))
                std[col_idx] = s if (np.isfinite(s) and s >= self.eps) else 1.0
            else:
                self.mean_[col_idx] = 0.0
                std[col_idx] = 1.0

        self.std_ = std
        self.is_fitted_ = True
        return self

    def fit(
        self,
        train_df: pd.DataFrame,
        num_cols: Sequence[str] | None = None,
    ) -> FoldNumericPreprocessor:
        """오직 학습 폴드 DataFrame만으로 평균/표준편차를 계산합니다."""
        self.reset()
        resolved_cols = list(num_cols) if num_cols is not None else self.num_cols
        if resolved_cols is None:
            resolved_cols = [
                c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c].dtype)
            ]
        self.partial_fit(train_df, num_cols=resolved_cols)
        return self.finalize_fit()

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """수치형 컬럼을 `(scaled_values, missing_mask)` `float32` 튜플로 변환합니다.

        - `scaled_values`: `(N, D_num)`, 결측·무한대 위치는 `0.0`
        - `missing_mask`: `(N, D_num)`, 결측·무한대 위치는 `1.0`, 유효 위치는 `0.0`
        """
        if not self.is_fitted_:
            raise RuntimeError("FoldNumericPreprocessor.fit()을 먼저 호출해야 합니다.")

        n_rows = len(df)
        d = len(self.num_cols_)
        if d == 0:
            return (
                np.zeros((n_rows, 0), dtype=np.float32),
                np.zeros((n_rows, 0), dtype=np.float32),
            )

        raw = self._extract_raw_matrix(df, self.num_cols_)
        finite_mask = np.isfinite(raw)
        missing_mask = (~finite_mask).astype(np.float32)

        transformed = np.zeros((n_rows, d), dtype=np.float64)
        if np.any(finite_mask):
            vals = raw[finite_mask]
            if self.apply_signed_log1p:
                vals = np.sign(vals) * np.log1p(np.abs(vals))
            transformed[finite_mask] = vals

        scaled = (transformed - self.mean_[None, :]) / self.std_[None, :]
        if self.clip_value is not None:
            scaled = np.clip(scaled, -self.clip_value, self.clip_value)

        if np.any(~finite_mask):
            scaled[~finite_mask] = 0.0

        return scaled.astype(np.float32), missing_mask

    def transform_combined(self, df: pd.DataFrame) -> np.ndarray:
        """표준화된 수치 피처와 결측 지시 마스크를 수평 결합(`2 * D_num`)해 반환합니다."""
        scaled, missing_mask = self.transform(df)
        return np.concatenate([scaled, missing_mask], axis=1)

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        num_cols: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """학습 폴드에 `fit`한 직후 동일 데이터에 대해 변환을 수행합니다."""
        self.fit(train_df, num_cols=num_cols)
        return self.transform(train_df)

    def to_dict(self) -> dict[str, Any]:
        """전처리기 통계를 직렬화 가능한 딕셔너리로 반환합니다."""
        return {
            "num_cols": self.num_cols_,
            "apply_signed_log1p": self.apply_signed_log1p,
            "eps": self.eps,
            "clip_value": self.clip_value,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FoldNumericPreprocessor:
        """직렬화된 딕셔너리에서 `FoldNumericPreprocessor`를 복원합니다."""
        obj = cls(
            num_cols=list(payload["num_cols"]),
            apply_signed_log1p=bool(payload["apply_signed_log1p"]),
            eps=float(payload["eps"]),
            clip_value=payload.get("clip_value"),
        )
        obj.num_cols_ = list(payload["num_cols"])
        obj.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        obj.std_ = np.asarray(payload["std"], dtype=np.float64)
        obj.is_fitted_ = True
        return obj
