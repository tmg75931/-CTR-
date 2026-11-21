"""정형 CTR 파생 피처(교호작용·행 통계·seq 요약 조인·OOF 인코딩) 통합 파이프라인 모듈."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold, StratifiedKFold

from toss_ctr.data.schema import DEFAULT_CONTRACT, DatasetContract
from toss_ctr.features.history_agg import build_row_statistics
from toss_ctr.features.interactions import (
    CORE_INTERACTION_PAIRS,
    EXTENDED_INTERACTION_PAIRS,
    FoldConditionalStatsEncoder,
    FoldFrequencyEncoder,
    build_interaction_features,
    make_interaction_column_name,
)
from toss_ctr.features.target_encoding import (
    OutOfFoldTargetEncoder,
    _extract_fold_array,
    _extract_target_array,
)

DEFAULT_FREQ_COLS: tuple[str, ...] = (
    "inventory_id",
    "l_feat_14",
    "age_group",
    "gender",
    "hour",
    "day_of_week",
)

DEFAULT_TE_COLS: tuple[str, ...] = (
    "inventory_id",
    "l_feat_14",
    "age_group",
    "gender",
    "hour",
)

DEFAULT_CONDITIONAL_SPECS: tuple[tuple[str, str], ...] = (
    ("inventory_id", "l_feat_14"),
    ("l_feat_14", "inventory_id"),
    ("inventory_id", "hour"),
)


@dataclass
class FoldFittedEncoders:
    """단일 폴드 학습 행(`folds != fold_id`)에서만 적합된 동적 인코더 컨테이너."""

    base_cols: list[str]
    freq_enc: FoldFrequencyEncoder | None = None
    cond_enc: FoldConditionalStatsEncoder | None = None
    te_enc: OutOfFoldTargetEncoder | None = None

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """정적 피처 DataFrame(또는 슬라이스 청크)에 폴드 인코더를 행 순서 보존 변환합니다."""
        base = df[self.base_cols]
        parts: list[pd.DataFrame] = [base]
        if self.freq_enc is not None:
            parts.append(self.freq_enc.transform(base))
        if self.cond_enc is not None:
            cond_df = self.cond_enc.transform(base)
            if not cond_df.empty:
                parts.append(cond_df)
        if self.te_enc is not None:
            parts.append(self.te_enc.transform(base))
        return pd.concat(parts, axis=1)


@dataclass
class FoldFeatureBundle:
    """단일 CV 폴드 학습/검증/추론을 위한 누수 차단 피처 번들."""

    fold_id: int
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_valid: pd.DataFrame
    y_valid: np.ndarray
    X_test: pd.DataFrame | None
    feature_names: list[str]
    categorical_features: list[str]
    train_row_ids: np.ndarray
    valid_row_ids: np.ndarray
    encoders: FoldFittedEncoders | None = None


def join_seq_summary(
    df: pd.DataFrame,
    seq_summary: pd.DataFrame | str | Path | None,
    key_col: str,
    prefix: str = "seq_",
    ignore_missing_path: bool = True,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> pd.DataFrame:
    """`seq` 요약 피처 테이블을 `row_id` 또는 `ID` 기준으로 순서/행수 보존 Left Join합니다."""
    if seq_summary is None:
        return df

    if isinstance(seq_summary, (str, Path)):
        summary_path = Path(seq_summary)
        if not summary_path.exists():
            if ignore_missing_path:
                return df
            raise FileNotFoundError(f"seq 요약 파일을 찾을 수 없습니다: {summary_path}")
        summary_df = pq.read_table(summary_path).to_pandas()
    elif isinstance(seq_summary, pd.DataFrame):
        summary_df = seq_summary
    else:
        raise TypeError(f"지원하지 않는 seq_summary 타입입니다: {type(seq_summary).__name__}")

    if key_col not in df.columns:
        raise KeyError(f"대상 DataFrame에 조인 키 '{key_col}' 컬럼이 없습니다.")
    if key_col not in summary_df.columns:
        raise KeyError(f"seq 요약 테이블에 조인 키 '{key_col}' 컬럼이 없습니다.")

    if summary_df[key_col].isna().any():
        raise ValueError(f"seq 요약 테이블의 조인 키 '{key_col}'에 결측치가 존재합니다.")
    if summary_df[key_col].duplicated().any():
        raise ValueError(
            f"seq 요약 테이블의 조인 키 '{key_col}'에 중복이 있어 행 수가 변할 수 있습니다."
        )

    reserved = set(contract.reserved_cols)
    existing_cols = set(df.columns)
    feature_cols = [c for c in summary_df.columns if c != key_col and c not in reserved]
    if not feature_cols:
        return df

    rename_map: dict[str, str] = {}
    for col in feature_cols:
        target_name = col if col.startswith(prefix) else f"{prefix}{col}"
        if target_name in existing_cols:
            target_name = f"{prefix}summary_{col}"
        rename_map[col] = target_name

    indexed_summary = summary_df[[key_col, *feature_cols]].rename(columns=rename_map).set_index(
        key_col
    )
    aligned = indexed_summary.reindex(df[key_col].to_numpy())
    aligned.index = df.index

    # 메모리 절감을 위해 float64 요약 피처를 float32로 축소
    cast_cols: dict[str, pd.Series] = {}
    for col in aligned.columns:
        s = aligned[col]
        if pd.api.types.is_float_dtype(s.dtype) and str(s.dtype) == "float64":
            cast_cols[col] = s.astype(np.float32)
        else:
            cast_cols[col] = s
    aligned_cast = pd.DataFrame(cast_cols, index=df.index)
    return pd.concat([df, aligned_cast], axis=1)


def _build_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """`hour` 컬럼 존재 시 타 행과 무관한 주기(sin/cos) 피처를 생성합니다."""
    if "hour" not in df.columns or not pd.api.types.is_numeric_dtype(df["hour"].dtype):
        return pd.DataFrame(index=df.index)

    hour_vals = df["hour"].to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
    radians = (2.0 * np.pi / 24.0) * hour_vals
    return pd.DataFrame(
        {
            "hour_sin": np.sin(radians).astype(np.float32),
            "hour_cos": np.cos(radians).astype(np.float32),
        },
        index=df.index,
    )


def _build_sub_folds(y_tr: np.ndarray, sub_folds: np.ndarray, seed: int) -> np.ndarray:
    """외부 폴드 제외 학습 데이터에서 최소 2개 이상의 서브 폴드를 보장합니다."""
    if len(np.unique(sub_folds)) >= 2:
        return sub_folds

    n_rows = len(y_tr)
    _, class_counts = np.unique(y_tr, return_counts=True)
    if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
        n_inner = min(3, int(class_counts.min()))
        splitter = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(n_rows, dtype=np.int8), y_tr)
    else:
        n_inner = min(3, n_rows)
        splitter = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(n_rows, dtype=np.int8))

    inner = np.empty(n_rows, dtype=np.int64)
    for idx, (_, val_idx) in enumerate(split_iter):
        inner[val_idx] = idx
    return inner


class TabularFeaturePipeline:
    """행 단위 파생, `seq` 조인, 폴드 안전 빈도 및 OOF 타겟 인코딩을 통합하는 파이프라인."""

    def __init__(
        self,
        interaction_pairs: Sequence[Sequence[str]] = EXTENDED_INTERACTION_PAIRS,
        interaction_format: Literal["hash_int64", "string"] = "hash_int64",
        add_row_stats: bool = True,
        add_cyclical_hour: bool = True,
        freq_encode_cols: Sequence[str] = DEFAULT_FREQ_COLS,
        include_interactions_in_freq: bool = True,
        add_freq_log1p: bool = True,
        conditional_pair_specs: Sequence[tuple[str, str]] = DEFAULT_CONDITIONAL_SPECS,
        target_encode_cols: Sequence[str] = DEFAULT_TE_COLS,
        include_interactions_in_te: bool = True,
        te_smoothing: float = 20.0,
        te_handle_missing: Literal["encode", "prior"] = "encode",
        seed: int = 42,
        contract: DatasetContract = DEFAULT_CONTRACT,
    ) -> None:
        self.interaction_pairs = tuple(tuple(p) for p in interaction_pairs)
        self.interaction_format = interaction_format
        self.add_row_stats = add_row_stats
        self.add_cyclical_hour = add_cyclical_hour
        self.freq_encode_cols = tuple(freq_encode_cols)
        self.include_interactions_in_freq = include_interactions_in_freq
        self.add_freq_log1p = add_freq_log1p
        self.conditional_pair_specs = tuple(conditional_pair_specs)
        self.target_encode_cols = tuple(target_encode_cols)
        self.include_interactions_in_te = include_interactions_in_te
        self.te_smoothing = float(te_smoothing)
        self.te_handle_missing = te_handle_missing
        self.seed = int(seed)
        self.contract = contract

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        contract: DatasetContract = DEFAULT_CONTRACT,
    ) -> TabularFeaturePipeline:
        """YAML 설정 딕셔너리에서 파이프라인 인스턴스를 생성합니다."""
        fe_cfg = cfg.get("features", cfg)
        pairs = fe_cfg.get("interaction_pairs", CORE_INTERACTION_PAIRS)
        cond_specs = [tuple(x) for x in fe_cfg.get("conditional_pair_specs", DEFAULT_CONDITIONAL_SPECS)]
        return cls(
            interaction_pairs=pairs,
            interaction_format=fe_cfg.get("interaction_format", "hash_int64"),
            add_row_stats=bool(fe_cfg.get("add_row_stats", True)),
            add_cyclical_hour=bool(fe_cfg.get("add_cyclical_hour", True)),
            freq_encode_cols=fe_cfg.get("freq_encode_cols", DEFAULT_FREQ_COLS),
            include_interactions_in_freq=bool(fe_cfg.get("include_interactions_in_freq", True)),
            add_freq_log1p=bool(fe_cfg.get("add_freq_log1p", True)),
            conditional_pair_specs=cond_specs,
            target_encode_cols=fe_cfg.get("target_encode_cols", DEFAULT_TE_COLS),
            include_interactions_in_te=bool(fe_cfg.get("include_interactions_in_te", True)),
            te_smoothing=float(fe_cfg.get("te_smoothing", 20.0)),
            te_handle_missing=fe_cfg.get("te_handle_missing", "encode"),
            seed=int(cfg.get("seed", 42)),
            contract=contract,
        )

    def _resolve_interaction_col_names(self, available_cols: set[str]) -> list[str]:
        names: list[str] = []
        for pair in self.interaction_pairs:
            if len(pair) >= 2 and all(c in available_cols for c in pair):
                ix_name = make_interaction_column_name(pair)
                if ix_name in available_cols:
                    names.append(ix_name)
        return names

    def _dedup_cols(self, cols: Sequence[str], available_cols: set[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for c in cols:
            if c in available_cols and c not in seen:
                seen.add(c)
                result.append(c)
        return result

    def prepare_static_features(
        self,
        df: pd.DataFrame,
        seq_summary: pd.DataFrame | str | Path | None = None,
        key_col: str | None = None,
    ) -> pd.DataFrame:
        """타 행을 참조하지 않는 정적 피처(`seq` 요약 조인 + 교호작용 + 행 통계)를 생성합니다."""
        work = df
        if seq_summary is not None:
            resolved_key = key_col
            if resolved_key is None:
                resolved_key = (
                    self.contract.row_id_col
                    if self.contract.row_id_col in work.columns
                    else self.contract.id_col
                )
            work = join_seq_summary(
                work,
                seq_summary=seq_summary,
                key_col=resolved_key,
                contract=self.contract,
            )

        parts: list[pd.DataFrame] = [work]
        if self.interaction_pairs:
            ix_df = build_interaction_features(
                work,
                interaction_pairs=self.interaction_pairs,
                output_format=self.interaction_format,
                keep_original=False,
            )
            if not ix_df.empty:
                new_ix_cols = [c for c in ix_df.columns if c not in work.columns]
                if new_ix_cols:
                    parts.append(ix_df[new_ix_cols])

        if self.add_row_stats:
            stats_df = build_row_statistics(
                work,
                contract=self.contract,
                keep_original=False,
            )
            if not stats_df.empty:
                new_stat_cols = [c for c in stats_df.columns if c not in work.columns]
                if new_stat_cols:
                    parts.append(stats_df[new_stat_cols])

        if self.add_cyclical_hour:
            cyc_df = _build_cyclical_time_features(work)
            if not cyc_df.empty:
                new_cyc_cols = [c for c in cyc_df.columns if c not in work.columns]
                if new_cyc_cols:
                    parts.append(cyc_df[new_cyc_cols])

        if len(parts) == 1:
            return work.copy()
        return pd.concat(parts, axis=1)

    def _extract_feature_columns(self, df: pd.DataFrame) -> list[str]:
        reserved = set(self.contract.reserved_cols)
        return [c for c in df.columns if c not in reserved]

    def _infer_categorical_features(
        self,
        df: pd.DataFrame,
        feature_names: Sequence[str],
    ) -> list[str]:
        cat_cols: list[str] = []
        for col in feature_names:
            dtype = df[col].dtype
            if (
                isinstance(dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(dtype)
                or pd.api.types.is_object_dtype(dtype)
            ):
                cat_cols.append(col)
        return cat_cols

    def fit_transform_fold(
        self,
        train_df: pd.DataFrame,
        y_train: ArrayLike,
        folds: pd.DataFrame | ArrayLike,
        fold_id: int,
        test_df: pd.DataFrame | None = None,
        train_seq_summary: pd.DataFrame | str | Path | None = None,
        test_seq_summary: pd.DataFrame | str | Path | None = None,
        assume_static_prepared: bool = False,
    ) -> FoldFeatureBundle:
        """지정된 `fold_id`를 완전히 격리한 채 학습/검증/테스트 피처 번들을 생성합니다.

        - `FoldFrequencyEncoder`와 `FoldConditionalStatsEncoder`는 오직 `folds != fold_id` 행만 학습합니다.
        - `OutOfFoldTargetEncoder`는 `folds != fold_id` 내부의 서브 폴드로 OOF 인코딩을 수행하고,
          검증(`folds == fold_id`) 및 테스트 데이터에는 `folds != fold_id`에서 산출된 통계만 매핑합니다.
        """
        n_rows = len(train_df)
        y_arr = _extract_target_array(y_train, n_rows)
        folds_arr = _extract_fold_array(folds, n_rows)

        va_mask = folds_arr == fold_id
        tr_mask = ~va_mask
        if not np.any(tr_mask) or not np.any(va_mask):
            raise ValueError(f"fold_id={fold_id}에 대한 학습 또는 검증 행이 비어 있습니다.")

        if assume_static_prepared:
            static_train = train_df
            static_test = test_df
        else:
            static_train = self.prepare_static_features(
                train_df,
                seq_summary=train_seq_summary,
                key_col=self.contract.row_id_col,
            )
            static_test = (
                self.prepare_static_features(
                    test_df,
                    seq_summary=test_seq_summary,
                    key_col=self.contract.id_col,
                )
                if test_df is not None
                else None
            )

        if self.contract.row_id_col in static_train.columns:
            all_row_ids = static_train[self.contract.row_id_col].to_numpy(dtype=np.int64)
        else:
            all_row_ids = np.arange(n_rows, dtype=np.int64)

        base_cols = self._extract_feature_columns(static_train)
        X_tr_base = static_train.loc[tr_mask, base_cols]
        X_va_base = static_train.loc[va_mask, base_cols]
        X_te_base = static_test[base_cols] if static_test is not None else None

        y_tr = y_arr[tr_mask]
        y_va = y_arr[va_mask]
        sub_folds = _build_sub_folds(y_tr, folds_arr[tr_mask], seed=self.seed + int(fold_id))

        available_set = set(base_cols)
        ix_cols = self._resolve_interaction_col_names(available_set)

        freq_targets = list(self.freq_encode_cols)
        if self.include_interactions_in_freq:
            freq_targets.extend(ix_cols)
        freq_targets = self._dedup_cols(freq_targets, available_set)

        te_targets = list(self.target_encode_cols)
        if self.include_interactions_in_te:
            te_targets.extend(ix_cols)
        te_targets = self._dedup_cols(te_targets, available_set)

        tr_parts: list[pd.DataFrame] = [X_tr_base]
        va_parts: list[pd.DataFrame] = [X_va_base]
        te_parts: list[pd.DataFrame] = [X_te_base] if X_te_base is not None else []

        fitted_freq_enc: FoldFrequencyEncoder | None = None
        if freq_targets:
            fitted_freq_enc = FoldFrequencyEncoder(
                columns=freq_targets,
                add_log1p=self.add_freq_log1p,
                keep_original=False,
                ignore_missing=True,
            )
            tr_parts.append(fitted_freq_enc.fit_transform(X_tr_base))
            va_parts.append(fitted_freq_enc.transform(X_va_base))
            if X_te_base is not None:
                te_parts.append(fitted_freq_enc.transform(X_te_base))

        fitted_cond_enc: FoldConditionalStatsEncoder | None = None
        if self.conditional_pair_specs:
            cond_enc = FoldConditionalStatsEncoder(
                pair_specs=self.conditional_pair_specs,
                keep_original=False,
                ignore_missing=True,
            )
            tr_cond = cond_enc.fit_transform(X_tr_base)
            if not tr_cond.empty:
                fitted_cond_enc = cond_enc
                tr_parts.append(tr_cond)
                va_parts.append(cond_enc.transform(X_va_base))
                if X_te_base is not None:
                    te_parts.append(cond_enc.transform(X_te_base))

        fitted_te_enc: OutOfFoldTargetEncoder | None = None
        if te_targets:
            fitted_te_enc = OutOfFoldTargetEncoder(
                columns=te_targets,
                smoothing=self.te_smoothing,
                handle_missing=self.te_handle_missing,
                keep_original=False,
                ignore_missing=True,
            )
            tr_parts.append(fitted_te_enc.fit_transform(X_tr_base, y_tr, sub_folds))
            va_parts.append(fitted_te_enc.transform(X_va_base))
            if X_te_base is not None:
                te_parts.append(fitted_te_enc.transform(X_te_base))

        fold_encoders = FoldFittedEncoders(
            base_cols=list(base_cols),
            freq_enc=fitted_freq_enc,
            cond_enc=fitted_cond_enc,
            te_enc=fitted_te_enc,
        )

        X_tr_final = pd.concat(tr_parts, axis=1)
        del tr_parts, X_tr_base
        X_va_final = pd.concat(va_parts, axis=1)
        del va_parts, X_va_base
        X_te_final = pd.concat(te_parts, axis=1) if X_te_base is not None else None
        del te_parts, X_te_base

        feature_names = list(X_tr_final.columns)
        cat_cols = self._infer_categorical_features(X_tr_final, feature_names)

        return FoldFeatureBundle(
            fold_id=int(fold_id),
            X_train=X_tr_final,
            y_train=y_tr,
            X_valid=X_va_final,
            y_valid=y_va,
            X_test=X_te_final,
            feature_names=feature_names,
            categorical_features=cat_cols,
            train_row_ids=all_row_ids[tr_mask],
            valid_row_ids=all_row_ids[va_mask],
            encoders=fold_encoders,
        )

    def run_oof_fe(
        self,
        train_df: pd.DataFrame,
        y_train: ArrayLike,
        folds: pd.DataFrame | ArrayLike,
        test_df: pd.DataFrame | None = None,
        train_seq_summary: pd.DataFrame | str | Path | None = None,
        test_seq_summary: pd.DataFrame | str | Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
        """전체 학습 세트에 대해 누수 없는 Out-of-Fold 피처 행렬과 테스트 피처 행렬을 생성합니다."""
        n_rows = len(train_df)
        y_arr = _extract_target_array(y_train, n_rows)
        folds_arr = _extract_fold_array(folds, n_rows)
        unique_folds = [int(f) for f in np.unique(folds_arr)]
        if len(unique_folds) < 2:
            raise ValueError("run_oof_fe를 위해서는 최소 2개 이상의 폴드가 필요합니다.")

        static_train = self.prepare_static_features(
            train_df,
            seq_summary=train_seq_summary,
            key_col=self.contract.row_id_col,
        )
        static_test = (
            self.prepare_static_features(
                test_df,
                seq_summary=test_seq_summary,
                key_col=self.contract.id_col,
            )
            if test_df is not None
            else None
        )

        base_cols = self._extract_feature_columns(static_train)
        available_set = set(base_cols)
        ix_cols = self._resolve_interaction_col_names(available_set)

        freq_targets = list(self.freq_encode_cols)
        if self.include_interactions_in_freq:
            freq_targets.extend(ix_cols)
        freq_targets = self._dedup_cols(freq_targets, available_set)

        te_targets = list(self.target_encode_cols)
        if self.include_interactions_in_te:
            te_targets.extend(ix_cols)
        te_targets = self._dedup_cols(te_targets, available_set)

        train_parts: list[pd.DataFrame] = [static_train[base_cols]]
        test_parts: list[pd.DataFrame] = [static_test[base_cols]] if static_test is not None else []

        # 빈도 인코더도 동일 폴드 배제 OOF 방식으로 계산해 검증 행의 빈도 누수 차단
        if freq_targets:
            oof_freq_df: pd.DataFrame | None = None
            for k in unique_folds:
                tr_m = folds_arr != k
                va_m = folds_arr == k
                enc = FoldFrequencyEncoder(
                    columns=freq_targets,
                    add_log1p=self.add_freq_log1p,
                    keep_original=False,
                    ignore_missing=True,
                )
                enc.fit(static_train.loc[tr_m, base_cols])
                chunk = enc.transform(static_train.loc[va_m, base_cols])
                if oof_freq_df is None:
                    oof_freq_df = pd.DataFrame(
                        {col: np.empty(n_rows, dtype=chunk[col].dtype) for col in chunk.columns},
                        index=static_train.index,
                    )
                va_pos = np.flatnonzero(va_m)
                for col in chunk.columns:
                    oof_freq_df.iloc[va_pos, oof_freq_df.columns.get_loc(col)] = chunk[
                        col
                    ].to_numpy()
            if oof_freq_df is not None:
                train_parts.append(oof_freq_df)

            if static_test is not None:
                full_freq_enc = FoldFrequencyEncoder(
                    columns=freq_targets,
                    add_log1p=self.add_freq_log1p,
                    keep_original=False,
                    ignore_missing=True,
                )
                full_freq_enc.fit(static_train[base_cols])
                test_parts.append(full_freq_enc.transform(static_test[base_cols]))

        if self.conditional_pair_specs:
            oof_cond_df: pd.DataFrame | None = None
            for k in unique_folds:
                tr_m = folds_arr != k
                va_m = folds_arr == k
                cond_enc = FoldConditionalStatsEncoder(
                    pair_specs=self.conditional_pair_specs,
                    keep_original=False,
                    ignore_missing=True,
                )
                cond_enc.fit(static_train.loc[tr_m, base_cols])
                chunk = cond_enc.transform(static_train.loc[va_m, base_cols])
                if chunk.empty:
                    continue
                if oof_cond_df is None:
                    oof_cond_df = pd.DataFrame(
                        {col: np.empty(n_rows, dtype=chunk[col].dtype) for col in chunk.columns},
                        index=static_train.index,
                    )
                va_pos = np.flatnonzero(va_m)
                for col in chunk.columns:
                    oof_cond_df.iloc[va_pos, oof_cond_df.columns.get_loc(col)] = chunk[
                        col
                    ].to_numpy()
            if oof_cond_df is not None:
                train_parts.append(oof_cond_df)

                if static_test is not None:
                    full_cond_enc = FoldConditionalStatsEncoder(
                        pair_specs=self.conditional_pair_specs,
                        keep_original=False,
                        ignore_missing=True,
                    )
                    full_cond_enc.fit(static_train[base_cols])
                    test_parts.append(full_cond_enc.transform(static_test[base_cols]))

        if te_targets:
            oof_te_enc = OutOfFoldTargetEncoder(
                columns=te_targets,
                smoothing=self.te_smoothing,
                handle_missing=self.te_handle_missing,
                keep_original=False,
                ignore_missing=True,
            )
            train_parts.append(oof_te_enc.fit_transform(static_train[base_cols], y_arr, folds_arr))
            if static_test is not None:
                test_parts.append(oof_te_enc.transform(static_test[base_cols]))

        X_train_fe = pd.concat(train_parts, axis=1)
        X_test_fe = pd.concat(test_parts, axis=1) if static_test is not None else None
        feature_names = list(X_train_fe.columns)
        return X_train_fe, X_test_fe, feature_names
