"""누수 제로 Out-of-Fold M-estimate 타겟 인코딩 모듈."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from toss_ctr.features.interactions import _normalize_for_mapping


def _extract_fold_array(folds: pd.DataFrame | ArrayLike, n_rows: int) -> np.ndarray:
    """다양한 형식의 폴드 입력을 검증된 1D int64 배열로 변환합니다."""
    if isinstance(folds, pd.DataFrame):
        if "fold_id" not in folds.columns:
            raise KeyError("folds DataFrame에 'fold_id' 컬럼이 존재해야 합니다.")
        arr_raw = folds["fold_id"].to_numpy()
    elif isinstance(folds, pd.Series):
        arr_raw = folds.to_numpy()
    else:
        arr_raw = np.asarray(folds)

    if arr_raw.ndim != 1 or len(arr_raw) != n_rows:
        raise ValueError(
            f"folds 길이가 학습 데이터와 일치하지 않습니다: expected={n_rows}, actual={arr_raw.shape}"
        )
    if pd.isna(arr_raw).any():
        raise ValueError("folds에 결측치가 포함되어 있습니다.")
    return arr_raw.astype(np.int64, copy=False)


def _extract_target_array(y_train: ArrayLike, n_rows: int) -> np.ndarray:
    """타겟 배열을 검증된 1D 이진(0/1) int8 배열로 변환합니다."""
    if isinstance(y_train, pd.Series):
        arr_raw = y_train.to_numpy()
    else:
        arr_raw = np.asarray(y_train)

    if arr_raw.ndim != 1 or len(arr_raw) != n_rows:
        raise ValueError(
            f"y_train 길이가 학습 데이터와 일치하지 않습니다: expected={n_rows}, actual={arr_raw.shape}"
        )
    if pd.isna(arr_raw).any():
        raise ValueError("y_train에 결측치가 포함되어 있습니다.")
    y_int = arr_raw.astype(np.int64, copy=False)
    unique_vals = np.unique(y_int)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"y_train은 이진(0/1) 타겟이어야 합니다: {unique_vals.tolist()}")
    return y_int.astype(np.int8, copy=False)


class OutOfFoldTargetEncoder:
    """자기 행과 동일 폴드 행을 완전히 배제하는 M-estimate Smoothed CTR 인코더.

    각 폴드 `fold_id == k`에 대해 오직 `fold_id != k` 행들의 클릭 합(`sum_y`),
    노출 수(`cnt`), 그리고 Out-of-Fold 전역 사전 확률(`oof_prior`)만으로
    `(sum_y + smoothing * oof_prior) / (cnt + smoothing)`을 계산합니다.
    """

    def __init__(
        self,
        columns: Sequence[str],
        smoothing: float = 20.0,
        suffix: str = "_te",
        handle_missing: Literal["encode", "prior"] = "encode",
        keep_original: bool = False,
        ignore_missing: bool = True,
    ) -> None:
        if smoothing <= 0.0:
            raise ValueError(f"smoothing은 0보다 커야 합니다: {smoothing}")
        if handle_missing not in ("encode", "prior"):
            raise ValueError(
                f"handle_missing은 'encode' 또는 'prior'여야 합니다: {handle_missing!r}"
            )

        self.columns = tuple(columns)
        self.smoothing = float(smoothing)
        self.suffix = suffix
        self.handle_missing = handle_missing
        self.keep_original = keep_original
        self.ignore_missing = ignore_missing

        self.fitted_columns_: list[str] = []
        self.encoding_maps_: dict[str, pd.Series] = {}
        self.missing_values_: dict[str, np.float32] = {}
        self.global_prior_: float = 0.0
        self.oof_priors_: dict[int, float] = {}
        self.is_fitted_: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: ArrayLike) -> OutOfFoldTargetEncoder:
        """전체 학습 데이터 통계로 추론용 매핑 사전만 적합합니다(OOF 계산 없이 `transform` 전용)."""
        n_rows = len(X_train)
        if n_rows == 0:
            raise ValueError("빈 DataFrame으로 OutOfFoldTargetEncoder를 학습할 수 없습니다.")
        y_arr = _extract_target_array(y_train, n_rows)
        pos_mask = y_arr == 1
        total_y_sum = int(pos_mask.sum())
        self.global_prior_ = float(total_y_sum / n_rows)
        self.oof_priors_ = {}
        self.fitted_columns_ = []
        self.encoding_maps_ = {}
        self.missing_values_ = {}

        for col in self.columns:
            if col not in X_train.columns:
                if self.ignore_missing:
                    continue
                raise KeyError(f"OutOfFoldTargetEncoder 대상 컬럼이 없습니다: {col!r}")

            s = _normalize_for_mapping(X_train[col])
            codes, uniques = pd.factorize(s, use_na_sentinel=True)
            codes_shifted = codes.astype(np.int64, copy=False) + 1
            n_buckets = len(uniques) + 1

            total_cnt = np.bincount(codes_shifted, minlength=n_buckets)
            total_sum = np.bincount(codes_shifted[pos_mask], minlength=n_buckets)
            full_table = (total_sum + self.smoothing * self.global_prior_) / (
                total_cnt + self.smoothing
            )

            self.fitted_columns_.append(col)
            self.encoding_maps_[col] = pd.Series(
                full_table[1:].astype(np.float32), index=uniques
            )
            if self.handle_missing == "prior" or total_cnt[0] == 0:
                self.missing_values_[col] = np.float32(self.global_prior_)
            else:
                self.missing_values_[col] = np.float32(full_table[0])

        self.is_fitted_ = True
        return self

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: ArrayLike,
        folds: pd.DataFrame | ArrayLike,
    ) -> pd.DataFrame:
        """폴드 기반 Out-of-Fold 인코딩을 산출하고 전체 학습 기반 추론 사전을 함께 저장합니다."""
        n_rows = len(X_train)
        if n_rows == 0:
            raise ValueError("빈 DataFrame으로 OutOfFoldTargetEncoder를 학습할 수 없습니다.")

        y_arr = _extract_target_array(y_train, n_rows)
        folds_arr = _extract_fold_array(folds, n_rows)

        unique_folds = [int(f) for f in np.unique(folds_arr)]
        if len(unique_folds) < 2:
            raise ValueError(
                f"Out-of-Fold 타겟 인코딩을 위해서는 최소 2개 이상의 폴드가 필요합니다: {unique_folds}"
            )

        pos_mask = y_arr == 1
        total_y_sum = int(pos_mask.sum())
        self.global_prior_ = float(total_y_sum / n_rows)

        fold_masks: dict[int, np.ndarray] = {}
        fold_pos_masks: dict[int, np.ndarray] = {}
        oof_priors: dict[int, float] = {}
        for k in unique_folds:
            m_k = folds_arr == k
            n_k = int(m_k.sum())
            oof_n = n_rows - n_k
            if oof_n <= 0:
                raise ValueError(f"폴드 {k}를 제외한 학습 행이 0건입니다.")
            y_k = int(y_arr[m_k].sum())
            oof_priors[k] = float((total_y_sum - y_k) / oof_n)
            fold_masks[k] = m_k
            fold_pos_masks[k] = m_k & pos_mask

        self.oof_priors_ = oof_priors
        self.fitted_columns_ = []
        self.encoding_maps_ = {}
        self.missing_values_ = {}

        out_cols: dict[str, np.ndarray] = {}
        smoothing = self.smoothing

        for col in self.columns:
            if col not in X_train.columns:
                if self.ignore_missing:
                    continue
                raise KeyError(f"OutOfFoldTargetEncoder 대상 컬럼이 없습니다: {col!r}")

            s = _normalize_for_mapping(X_train[col])
            codes, uniques = pd.factorize(s, use_na_sentinel=True)
            codes_shifted = codes.astype(np.int64, copy=False) + 1
            n_buckets = len(uniques) + 1

            total_cnt = np.bincount(codes_shifted, minlength=n_buckets)
            total_sum = np.bincount(codes_shifted[pos_mask], minlength=n_buckets)

            oof_encoded = np.empty(n_rows, dtype=np.float32)
            for k in unique_folds:
                m_k = fold_masks[k]
                m_k_pos = fold_pos_masks[k]
                oof_prior_k = oof_priors[k]

                fold_cnt = np.bincount(codes_shifted[m_k], minlength=n_buckets)
                fold_sum = np.bincount(codes_shifted[m_k_pos], minlength=n_buckets)

                oof_cnt = total_cnt - fold_cnt
                oof_sum = total_sum - fold_sum
                table_k = (oof_sum + smoothing * oof_prior_k) / (oof_cnt + smoothing)
                if self.handle_missing == "prior":
                    table_k[0] = oof_prior_k

                oof_encoded[m_k] = table_k[codes_shifted[m_k]].astype(np.float32)

            full_table = (total_sum + smoothing * self.global_prior_) / (total_cnt + smoothing)

            self.fitted_columns_.append(col)
            self.encoding_maps_[col] = pd.Series(
                full_table[1:].astype(np.float32), index=uniques
            )
            if self.handle_missing == "prior" or total_cnt[0] == 0:
                self.missing_values_[col] = np.float32(self.global_prior_)
            else:
                self.missing_values_[col] = np.float32(full_table[0])

            out_cols[f"{col}{self.suffix}"] = oof_encoded

        self.is_fitted_ = True
        out_df = pd.DataFrame(out_cols, index=X_train.index)
        if self.keep_original:
            return pd.concat([X_train.copy(), out_df], axis=1)
        return out_df

    def transform(self, X_eval: pd.DataFrame) -> pd.DataFrame:
        """전체 학습 통계 기반의 매핑 사전으로 검증/테스트 데이터를 변환합니다(`float32`)."""
        if not self.is_fitted_:
            raise RuntimeError("transform 호출 전에 fit() 또는 fit_transform()을 실행해야 합니다.")

        prior_f32 = np.float32(self.global_prior_)
        out_cols: dict[str, np.ndarray] = {}

        for col in self.fitted_columns_:
            if col not in X_eval.columns:
                if self.ignore_missing:
                    continue
                raise KeyError(f"transform 입력에 컬럼이 없습니다: {col!r}")

            s = _normalize_for_mapping(X_eval[col])
            na_mask = s.isna().to_numpy(dtype=bool, copy=False)
            enc_map = self.encoding_maps_[col]

            if len(enc_map) == 0:
                arr = np.full(len(X_eval), prior_f32, dtype=np.float32)
            else:
                mapped = s.map(enc_map)
                arr = mapped.to_numpy(dtype=np.float32, na_value=prior_f32, copy=True)

            if np.any(na_mask):
                arr[na_mask] = self.missing_values_[col]

            out_cols[f"{col}{self.suffix}"] = arr.astype(np.float32, copy=False)

        out_df = pd.DataFrame(out_cols, index=X_eval.index)
        if self.keep_original:
            return pd.concat([X_eval.copy(), out_df], axis=1)
        return out_df

    def get_feature_names_out(self) -> list[str]:
        """생성되는 타겟 인코딩 컬럼명 목록을 반환합니다."""
        return [f"{col}{self.suffix}" for col in self.fitted_columns_]
