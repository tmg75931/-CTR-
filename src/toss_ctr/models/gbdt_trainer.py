"""다중 백엔드(LightGBM/CatBoost/XGBoost/Sklearn HGBT) OOF GBDT 학습·추론 모듈."""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from toss_ctr.data.schema import DEFAULT_CONTRACT, DatasetContract
from toss_ctr.evaluation.metrics import (
    EvalResult,
    clip_probs,
    evaluate_predictions,
    shift_prob_prior,
)
from toss_ctr.features.tabular_pipeline import FoldFittedEncoders, TabularFeaturePipeline
from toss_ctr.features.target_encoding import _extract_target_array

SUPPORTED_BACKENDS: tuple[str, ...] = ("lightgbm", "catboost", "xgboost", "sklearn_hgbt")
WeightingMode = Literal["balanced_weight", "prior_shift", "none"]
_TEST_INFER_CHUNK_SIZE: int = 250_000


@dataclass
class GBDTTrainResult:
    """5-Fold OOF GBDT 학습·평가 및 표준 예측 아티팩트 결과 컨테이너."""

    model_name: str
    backend: str
    oof_df: pd.DataFrame
    test_df: pd.DataFrame | None
    overall_metrics: EvalResult
    fold_metrics: list[dict[str, Any]]
    oot_metrics: EvalResult | None
    feature_importance: pd.DataFrame
    unshifted_metrics: EvalResult | None
    saved_paths: dict[str, str]


def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
    p = clip_probs(probs)
    return np.log(p / (1.0 - p))


def _logits_to_probs(logits: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(logits, dtype=np.float64), -35.0, 35.0)
    return clip_probs(1.0 / (1.0 + np.exp(-z)))


def _align_folds_frame(
    train_df: pd.DataFrame,
    folds: pd.DataFrame,
    contract: DatasetContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """`train_df`와 `folds`를 `row_id` 기준으로 정확히 정렬하고 검증합니다."""
    n_rows = len(train_df)
    if contract.fold_id_col not in folds.columns:
        raise KeyError(f"folds 입력에 '{contract.fold_id_col}' 컬럼이 없습니다.")

    if contract.row_id_col in train_df.columns and contract.row_id_col in folds.columns:
        train_rids = train_df[contract.row_id_col].to_numpy(dtype=np.int64)
        folds_indexed = folds.set_index(contract.row_id_col)
        if not np.array_equal(train_rids, folds[contract.row_id_col].to_numpy(dtype=np.int64)):
            folds_aligned = folds_indexed.reindex(train_rids)
        else:
            folds_aligned = folds_indexed
        if folds_aligned[contract.fold_id_col].isna().any():
            raise ValueError("train_df의 일부 row_id에 해당하는 fold_id가 folds에 존재하지 않습니다.")
        fold_arr = folds_aligned[contract.fold_id_col].to_numpy(dtype=np.int8)
        oot_arr = (
            folds_aligned[contract.oot_col].to_numpy(dtype=bool)
            if contract.oot_col in folds_aligned.columns
            else None
        )
        row_ids = train_rids
    else:
        if len(folds) != n_rows:
            raise ValueError(
                f"train_df({n_rows})와 folds({len(folds)})의 행 수가 일치하지 않습니다."
            )
        fold_arr = folds[contract.fold_id_col].to_numpy(dtype=np.int8)
        oot_arr = (
            folds[contract.oot_col].to_numpy(dtype=bool)
            if contract.oot_col in folds.columns
            else None
        )
        row_ids = np.arange(n_rows, dtype=np.int64)

    return row_ids, fold_arr, oot_arr


@dataclass
class _FoldFrameEncoder:
    """학습 폴드에서 확정된 범주 매핑·dtype 규칙으로 검증/테스트 청크를 변환하는 인코더."""

    backend: str
    feature_cols: list[str]
    cat_set: set[str]
    effective_cats: list[str]
    code_maps: dict[str, tuple[pd.Index, pd.Series]]
    int_cat_cols: set[str]

    def _transform_series(self, col: str, s: pd.Series) -> Any:
        if col in self.cat_set:
            if self.backend in ("lightgbm", "xgboost"):
                uniques, code_map = self.code_maps[col]
                s_raw = s.astype(object) if isinstance(s.dtype, pd.CategoricalDtype) else s
                codes = s_raw.map(code_map).to_numpy(dtype=np.float64, na_value=-1.0, copy=True)
                return pd.Categorical.from_codes(codes.astype(np.int32), categories=uniques)
            if self.backend == "catboost":
                if col in self.int_cat_cols:
                    return s.to_numpy(dtype=np.int64, na_value=-1, copy=False)
                return s.astype("string").fillna("__MISSING__").astype(str)
            _, code_map = self.code_maps[col]
            s_raw = s.astype(object) if isinstance(s.dtype, pd.CategoricalDtype) else s
            return s_raw.map(code_map).to_numpy(dtype=np.float32, na_value=-1.0, copy=True)

        if pd.api.types.is_bool_dtype(s.dtype):
            return s.to_numpy(dtype=np.int8, na_value=0, copy=False)
        if pd.api.types.is_extension_array_dtype(s.dtype):
            return s.to_numpy(dtype=np.float32, na_value=np.nan, copy=False)
        return s

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 폴드 스키마에 맞춰 임의 DataFrame 또는 청크를 변환합니다."""
        cols_dict = {col: self._transform_series(col, df[col]) for col in self.feature_cols}
        return pd.DataFrame(cols_dict, index=df.index)


def _fit_prepare_train_val(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    categorical_features: Sequence[str],
    backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], _FoldFrameEncoder]:
    """학습/검증 프레임만 먼저 변환하고 테스트 청크 재사용을 위한 `_FoldFrameEncoder`를 반환합니다."""
    cat_set = set(categorical_features)
    for col in X_tr.columns:
        dtype = X_tr[col].dtype
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
        ):
            cat_set.add(col)

    feature_cols = list(X_tr.columns)
    ordered_cats = [c for c in feature_cols if c in cat_set]
    effective_cats = [] if backend == "sklearn_hgbt" else ordered_cats

    tr_cols: dict[str, Any] = {}
    code_maps: dict[str, tuple[pd.Index, pd.Series]] = {}
    int_cat_cols: set[str] = set()

    for col in feature_cols:
        s_tr = X_tr[col]
        if col in cat_set:
            if backend in ("lightgbm", "xgboost"):
                s_tr_raw = s_tr.astype(object) if isinstance(s_tr.dtype, pd.CategoricalDtype) else s_tr
                codes_tr, uniques = pd.factorize(s_tr_raw, use_na_sentinel=True)
                tr_cols[col] = pd.Categorical.from_codes(
                    codes_tr.astype(np.int32, copy=False), categories=uniques
                )
                code_maps[col] = (uniques, pd.Series(np.arange(len(uniques), dtype=np.int32), index=uniques))
            elif backend == "catboost":
                if pd.api.types.is_integer_dtype(s_tr.dtype) and not pd.api.types.is_bool_dtype(
                    s_tr.dtype
                ):
                    int_cat_cols.add(col)
                    tr_cols[col] = s_tr.to_numpy(dtype=np.int64, na_value=-1, copy=False)
                else:
                    tr_cols[col] = s_tr.astype("string").fillna("__MISSING__").astype(str)
            else:
                # sklearn_hgbt: 255 초과 카디널리티 제약 회피를 위해 학습 폴드 기준 정수 코드화
                s_tr_raw = s_tr.astype(object) if isinstance(s_tr.dtype, pd.CategoricalDtype) else s_tr
                codes_tr, uniques = pd.factorize(s_tr_raw, use_na_sentinel=True)
                tr_cols[col] = codes_tr.astype(np.float32)
                code_maps[col] = (uniques, pd.Series(np.arange(len(uniques), dtype=np.int32), index=uniques))
        else:
            if pd.api.types.is_bool_dtype(s_tr.dtype):
                tr_cols[col] = s_tr.to_numpy(dtype=np.int8, na_value=0, copy=False)
            elif pd.api.types.is_extension_array_dtype(s_tr.dtype):
                tr_cols[col] = s_tr.to_numpy(dtype=np.float32, na_value=np.nan, copy=False)
            else:
                tr_cols[col] = s_tr

    encoder = _FoldFrameEncoder(
        backend=backend,
        feature_cols=feature_cols,
        cat_set=cat_set,
        effective_cats=effective_cats,
        code_maps=code_maps,
        int_cat_cols=int_cat_cols,
    )
    out_tr = pd.DataFrame(tr_cols, index=X_tr.index)
    out_va = encoder.transform(X_va)
    return out_tr, out_va, effective_cats, encoder


def _prepare_fold_frames(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame | None,
    categorical_features: Sequence[str],
    backend: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[str]]:
    """백엔드 규격에 맞게 범주형/확장형 컬럼을 학습 폴드 기준으로 안전하게 변환합니다."""
    out_tr, out_va, effective_cats, encoder = _fit_prepare_train_val(
        X_tr,
        X_va,
        categorical_features=categorical_features,
        backend=backend,
    )
    out_te = encoder.transform(X_te) if X_te is not None else None
    return out_tr, out_va, out_te, effective_cats


def _predict_proba_chunked(
    model: Any,
    X_df: pd.DataFrame,
    chunk_size: int = _TEST_INFER_CHUNK_SIZE,
) -> np.ndarray:
    """대용량 추론 시 메모리 스파이크를 막기 위해 청크 단위로 양성 확률을 예측합니다."""
    n = len(X_df)
    if n <= chunk_size:
        return np.asarray(model.predict_proba(X_df)[:, 1], dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = model.predict_proba(X_df.iloc[start:end])[:, 1]
    return out


def _predict_model_chunk(
    model: Any,
    fit_chunk: pd.DataFrame,
    backend: str,
    effective_cats: list[str],
) -> np.ndarray:
    """변환 완료된 단일 추론 청크에서 양성 확률을 예측합니다."""
    if backend == "catboost":
        from catboost import Pool

        te_pool = Pool(fit_chunk, cat_features=effective_cats or None)
        preds = np.asarray(model.predict_proba(te_pool)[:, 1], dtype=np.float64)
        del te_pool
        return preds
    return np.asarray(model.predict_proba(fit_chunk)[:, 1], dtype=np.float64)


def _fit_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: list[str],
    params: Mapping[str, Any],
    scale_pos_weight: float | None,
    va_sample_weight: np.ndarray | None,
    early_stopping_rounds: int | None,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, int | None]:
    import lightgbm as lgb

    lgb_params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "n_estimators": 2000,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "min_child_samples": 100,
        "n_jobs": -1,
        "verbose": -1,
    }
    lgb_params.update(params)
    lgb_params.setdefault("random_state", seed)
    if scale_pos_weight is not None and "scale_pos_weight" not in params:
        lgb_params["scale_pos_weight"] = float(scale_pos_weight)

    model = lgb.LGBMClassifier(**lgb_params)
    callbacks = []
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    fit_kwargs: dict[str, Any] = {
        "eval_set": [(X_va, y_va)],
        "callbacks": callbacks,
    }
    if va_sample_weight is not None:
        fit_kwargs["eval_sample_weight"] = [va_sample_weight]
    if cat_cols:
        fit_kwargs["categorical_feature"] = cat_cols

    model.fit(X_tr, y_tr, **fit_kwargs)
    va_pred = _predict_proba_chunked(model, X_va)

    if hasattr(model, "booster_"):
        importances = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=np.float64)
    else:
        importances = np.asarray(model.feature_importances_, dtype=np.float64)
    best_iter = getattr(model, "best_iteration_", None)
    return model, va_pred, importances, int(best_iter) if best_iter else None


def _fit_catboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: list[str],
    params: Mapping[str, Any],
    scale_pos_weight: float | None,
    va_sample_weight: np.ndarray | None,
    early_stopping_rounds: int | None,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, int | None]:
    from catboost import CatBoostClassifier, Pool

    cb_params: dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": seed,
        "verbose": 0,
    }
    cb_params.update(params)
    if scale_pos_weight is not None and "scale_pos_weight" not in params:
        cb_params["scale_pos_weight"] = float(scale_pos_weight)

    tr_pool = Pool(X_tr, label=y_tr, cat_features=cat_cols or None)
    va_pool = Pool(X_va, label=y_va, weight=va_sample_weight, cat_features=cat_cols or None)

    model = CatBoostClassifier(**cb_params)
    fit_kwargs: dict[str, Any] = {"eval_set": va_pool, "use_best_model": True}
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    model.fit(tr_pool, **fit_kwargs)

    va_pred = np.asarray(model.predict_proba(va_pool)[:, 1], dtype=np.float64)
    importances = np.asarray(model.get_feature_importance(tr_pool), dtype=np.float64)
    best_iter = model.get_best_iteration()
    del tr_pool, va_pool
    gc.collect()
    return model, va_pred, importances, int(best_iter) if best_iter is not None else None


def _fit_xgboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: list[str],
    params: Mapping[str, Any],
    scale_pos_weight: float | None,
    va_sample_weight: np.ndarray | None,
    early_stopping_rounds: int | None,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, int | None]:
    import xgboost as xgb

    xgb_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "enable_categorical": bool(cat_cols),
        "n_jobs": -1,
        "random_state": seed,
    }
    xgb_params.update(params)
    if scale_pos_weight is not None and "scale_pos_weight" not in params:
        xgb_params["scale_pos_weight"] = float(scale_pos_weight)
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        xgb_params.setdefault("early_stopping_rounds", int(early_stopping_rounds))

    model = xgb.XGBClassifier(**xgb_params)
    fit_kwargs: dict[str, Any] = {"eval_set": [(X_va, y_va)], "verbose": False}
    if va_sample_weight is not None:
        fit_kwargs["sample_weight_eval_set"] = [va_sample_weight]
    model.fit(X_tr, y_tr, **fit_kwargs)

    booster = model.get_booster()
    gain_map = booster.get_score(importance_type="gain")
    importances = np.array(
        [float(gain_map.get(col, 0.0)) for col in X_tr.columns], dtype=np.float64
    )
    best_iter = getattr(model, "best_iteration", None)

    # CUDA hist 학습 후 예측 시 device='cpu'로 전환해 device 불일치 DMatrix fallback 및 복사 메모리 제거
    if str(xgb_params.get("device", "")).startswith("cuda"):
        booster.set_param({"device": "cpu"})
        model.set_params(device="cpu")

    va_pred = _predict_proba_chunked(model, X_va)
    return model, va_pred, importances, int(best_iter) if best_iter is not None else None


def _fit_sklearn_hgbt(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    cat_cols: list[str],
    params: Mapping[str, Any],
    scale_pos_weight: float | None,
    va_sample_weight: np.ndarray | None,
    early_stopping_rounds: int | None,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, int | None]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    _ = (y_va, cat_cols, va_sample_weight, early_stopping_rounds)
    hgbt_params: dict[str, Any] = {
        "max_iter": 100,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "early_stopping": False,
        "random_state": seed,
    }
    hgbt_params.update(params)

    model = HistGradientBoostingClassifier(**hgbt_params)
    sample_weight = None
    if scale_pos_weight is not None:
        sample_weight = np.where(y_tr == 1, float(scale_pos_weight), 1.0)

    model.fit(X_tr, y_tr, sample_weight=sample_weight)
    va_pred = _predict_proba_chunked(model, X_va)
    importances = np.zeros(X_tr.shape[1], dtype=np.float64)
    n_iter = getattr(model, "n_iter_", None)
    return model, va_pred, importances, int(n_iter) if n_iter is not None else None


class GBDTOOFTrainer:
    """공통 `folds.parquet` 계약 기반 다중 백엔드 GBDT OOF 학습·추론기."""

    def __init__(
        self,
        backend: Literal["lightgbm", "catboost", "xgboost", "sklearn_hgbt"] = "lightgbm",
        model_name: str = "gbdt_model",
        params: Mapping[str, Any] | None = None,
        categorical_features: Sequence[str] | None = None,
        weighting_mode: WeightingMode | None = None,
        use_50_50_weight: bool = True,
        apply_prior_shift: bool = False,
        target_pos_rate: float = 0.5,
        early_stopping_rounds: int | None = 100,
        test_agg: Literal["logit_mean", "prob_mean"] = "logit_mean",
        reuse_cached_artifacts: bool = False,
        seed: int = 42,
        contract: DatasetContract = DEFAULT_CONTRACT,
    ) -> None:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"지원하지 않는 GBDT 백엔드입니다: {backend!r}. 허용 목록: {SUPPORTED_BACKENDS}"
            )
        if test_agg not in ("logit_mean", "prob_mean"):
            raise ValueError(f"test_agg는 'logit_mean' 또는 'prob_mean'이어야 합니다: {test_agg!r}")

        if weighting_mode is not None:
            if weighting_mode not in ("balanced_weight", "prior_shift", "none"):
                raise ValueError(f"지원하지 않는 weighting_mode입니다: {weighting_mode!r}")
            resolved_mode: WeightingMode = weighting_mode
        elif use_50_50_weight:
            resolved_mode = "balanced_weight"
        elif apply_prior_shift:
            resolved_mode = "prior_shift"
        else:
            resolved_mode = "none"

        self.backend = backend
        self.model_name = model_name
        self.params = dict(params or {})
        self.categorical_features = tuple(categorical_features or ())
        self.weighting_mode: WeightingMode = resolved_mode
        self.target_pos_rate = float(target_pos_rate)
        self.early_stopping_rounds = early_stopping_rounds
        self.test_agg = test_agg
        self.reuse_cached_artifacts = bool(reuse_cached_artifacts)
        self.seed = int(seed)
        self.contract = contract

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        model_key: str | None = None,
        contract: DatasetContract = DEFAULT_CONTRACT,
    ) -> GBDTOOFTrainer:
        """YAML 설정 딕셔너리에서 `GBDTOOFTrainer` 인스턴스를 생성합니다."""
        if model_key is not None and "models" in cfg and model_key in cfg["models"]:
            m_cfg = cfg["models"][model_key]
        else:
            m_cfg = cfg.get("model", cfg)

        return cls(
            backend=m_cfg.get("backend", cfg.get("backend", "lightgbm")),
            model_name=m_cfg.get("model_name", cfg.get("model_name", model_key or "gbdt_model")),
            params=m_cfg.get("params", {}),
            categorical_features=m_cfg.get("categorical_features", cfg.get("categorical_features")),
            weighting_mode=m_cfg.get("weighting_mode", cfg.get("weighting_mode")),
            use_50_50_weight=bool(m_cfg.get("use_50_50_weight", cfg.get("use_50_50_weight", True))),
            apply_prior_shift=bool(
                m_cfg.get("apply_prior_shift", cfg.get("apply_prior_shift", False))
            ),
            target_pos_rate=float(m_cfg.get("target_pos_rate", cfg.get("target_pos_rate", 0.5))),
            early_stopping_rounds=m_cfg.get(
                "early_stopping_rounds", cfg.get("early_stopping_rounds", 100)
            ),
            test_agg=m_cfg.get("test_agg", cfg.get("test_agg", "logit_mean")),
            reuse_cached_artifacts=bool(
                m_cfg.get("reuse_cached_artifacts", cfg.get("reuse_cached_artifacts", False))
            ),
            seed=int(m_cfg.get("seed", cfg.get("seed", 42))),
            contract=contract,
        )

    def _dispatch_backend(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
        cat_cols: list[str],
        scale_pos_weight: float | None,
        va_sample_weight: np.ndarray | None,
        fold_seed: int,
    ) -> tuple[Any, np.ndarray, np.ndarray, int | None]:
        if self.backend == "lightgbm":
            return _fit_lightgbm(
                X_tr,
                y_tr,
                X_va,
                y_va,
                cat_cols,
                self.params,
                scale_pos_weight,
                va_sample_weight,
                self.early_stopping_rounds,
                fold_seed,
            )
        if self.backend == "catboost":
            return _fit_catboost(
                X_tr,
                y_tr,
                X_va,
                y_va,
                cat_cols,
                self.params,
                scale_pos_weight,
                va_sample_weight,
                self.early_stopping_rounds,
                fold_seed,
            )
        if self.backend == "xgboost":
            return _fit_xgboost(
                X_tr,
                y_tr,
                X_va,
                y_va,
                cat_cols,
                self.params,
                scale_pos_weight,
                va_sample_weight,
                self.early_stopping_rounds,
                fold_seed,
            )
        return _fit_sklearn_hgbt(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cat_cols,
            self.params,
            scale_pos_weight,
            va_sample_weight,
            self.early_stopping_rounds,
            fold_seed,
        )

    def save_artifacts(
        self,
        result: GBDTTrainResult,
        output_dir: str | Path,
        compression: str = "zstd",
    ) -> dict[str, str]:
        """표준 OOF/Test 예측 Parquet 및 지표/중요도 요약 파일을 디스크에 저장합니다."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        oof_path = out_dir / f"oof_{result.model_name}.parquet"
        oof_table = pa.Table.from_pandas(result.oof_df, preserve_index=False)
        pq.write_table(oof_table, oof_path, compression=compression)

        saved: dict[str, str] = {"oof": str(oof_path)}

        if result.test_df is not None:
            test_path = out_dir / f"test_{result.model_name}.parquet"
            test_table = pa.Table.from_pandas(result.test_df, preserve_index=False)
            pq.write_table(test_table, test_path, compression=compression)
            saved["test"] = str(test_path)

        fi_path = out_dir / f"{result.model_name}_feature_importance.csv"
        result.feature_importance.to_csv(fi_path, index=False)
        saved["feature_importance"] = str(fi_path)

        metrics_path = out_dir / f"{result.model_name}_metrics.json"
        saved["metrics"] = str(metrics_path)
        payload: dict[str, Any] = {
            "model_name": result.model_name,
            "backend": result.backend,
            "weighting_mode": self.weighting_mode,
            "seed": self.seed,
            "overall_oof": result.overall_metrics.to_dict(),
            "oot_metrics": result.oot_metrics.to_dict() if result.oot_metrics is not None else None,
            "unshifted_metrics": (
                result.unshifted_metrics.to_dict()
                if result.unshifted_metrics is not None
                else None
            ),
            "fold_metrics": result.fold_metrics,
            "top_features": result.feature_importance.head(30).to_dict(orient="records"),
            "saved_paths": saved,
        }
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

        result.saved_paths = saved
        return saved

    def _try_load_cached_result(
        self,
        output_dir: str | Path,
        row_ids: np.ndarray,
        fold_arr: np.ndarray,
        y_all: np.ndarray,
        oot_arr: np.ndarray | None,
        test_df: pd.DataFrame | None,
    ) -> GBDTTrainResult | None:
        """이미 검증 완료된 OOF/Test 아티팩트가 존재할 경우 재학습 없이 로드해 반환합니다."""
        out_dir = Path(output_dir)
        oof_path = out_dir / f"oof_{self.model_name}.parquet"
        fi_path = out_dir / f"{self.model_name}_feature_importance.csv"
        metrics_path = out_dir / f"{self.model_name}_metrics.json"
        test_path = out_dir / f"test_{self.model_name}.parquet"

        required_paths = [oof_path, fi_path, metrics_path]
        if test_df is not None:
            required_paths.append(test_path)
        if not all(p.exists() for p in required_paths):
            return None

        try:
            oof_df = pq.read_table(oof_path).to_pandas()
            req_oof_cols = [
                self.contract.row_id_col,
                self.contract.fold_id_col,
                self.contract.target_col,
                "raw_logit",
                "pred_prob",
            ]
            if (
                len(oof_df) != len(row_ids)
                or not all(col in oof_df.columns for col in req_oof_cols)
                or bool(oof_df[req_oof_cols].isna().any().any())
            ):
                return None

            order = np.argsort(row_ids, kind="mergesort")
            if not (
                np.array_equal(
                    oof_df[self.contract.row_id_col].to_numpy(dtype=np.int64),
                    row_ids[order],
                )
                and np.array_equal(
                    oof_df[self.contract.fold_id_col].to_numpy(dtype=np.int8),
                    fold_arr[order],
                )
                and np.array_equal(
                    oof_df[self.contract.target_col].to_numpy(dtype=np.int8),
                    y_all[order],
                )
            ):
                return None

            out_test_df: pd.DataFrame | None = None
            saved: dict[str, str] = {"oof": str(oof_path)}
            if test_df is not None:
                out_test_df = pq.read_table(test_path).to_pandas()
                req_test_cols = [self.contract.id_col, "raw_logit", "pred_prob"]
                if (
                    len(out_test_df) != len(test_df)
                    or not all(col in out_test_df.columns for col in req_test_cols)
                    or bool(out_test_df[req_test_cols].isna().any().any())
                    or not np.array_equal(
                        out_test_df[self.contract.id_col].to_numpy(dtype=object),
                        test_df[self.contract.id_col].to_numpy(dtype=object),
                    )
                ):
                    return None
                saved["test"] = str(test_path)

            fi_df = pd.read_csv(fi_path)
            saved["feature_importance"] = str(fi_path)
            saved["metrics"] = str(metrics_path)
            with metrics_path.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception:
            return None

        y_ordered = y_all[order]
        probs_ordered = clip_probs(oof_df["pred_prob"].to_numpy(dtype=np.float64))
        overall_metrics = evaluate_predictions(y_ordered, probs_ordered)

        oot_metrics: EvalResult | None = None
        if oot_arr is not None:
            oot_ordered = oot_arr[order]
            if np.any(oot_ordered):
                y_oot = y_ordered[oot_ordered]
                if np.any(y_oot == 1) and np.any(y_oot == 0):
                    oot_metrics = evaluate_predictions(y_oot, probs_ordered[oot_ordered])

        unshifted_raw = payload.get("unshifted_metrics")
        unshifted_metrics = (
            EvalResult(
                ap=float(unshifted_raw["ap"]),
                wll=float(unshifted_raw["wll"]),
                score=float(unshifted_raw["score"]),
                pos_logloss=float(unshifted_raw["pos_logloss"]),
                neg_logloss=float(unshifted_raw["neg_logloss"]),
            )
            if isinstance(unshifted_raw, Mapping)
            else None
        )

        print(f"[CACHE] Reusing verified 5-Fold artifacts for {self.model_name} from {out_dir}")
        return GBDTTrainResult(
            model_name=self.model_name,
            backend=str(payload.get("backend", self.backend)),
            oof_df=oof_df,
            test_df=out_test_df,
            overall_metrics=overall_metrics,
            fold_metrics=list(payload.get("fold_metrics", [])),
            oot_metrics=oot_metrics,
            feature_importance=fi_df,
            unshifted_metrics=unshifted_metrics,
            saved_paths=saved,
        )

    def fit_predict(
        self,
        train_df: pd.DataFrame,
        folds: pd.DataFrame,
        test_df: pd.DataFrame | None = None,
        feature_cols: Sequence[str] | None = None,
        categorical_features: Sequence[str] | None = None,
        pipeline: TabularFeaturePipeline | None = None,
        train_seq_summary: pd.DataFrame | str | Path | None = None,
        test_seq_summary: pd.DataFrame | str | Path | None = None,
        assume_static_prepared: bool = False,
        output_dir: str | Path | None = None,
        reuse_cached_artifacts: bool | None = None,
    ) -> GBDTTrainResult:
        """표준 5-Fold 교차검증으로 GBDT를 학습하고 OOF/Test 예측을 산출합니다."""
        n_rows = len(train_df)
        if self.contract.target_col not in train_df.columns:
            raise KeyError(f"train_df에 타겟 컬럼 '{self.contract.target_col}'이 없습니다.")

        y_all = _extract_target_array(train_df[self.contract.target_col], n_rows)
        row_ids, fold_arr, oot_arr = _align_folds_frame(train_df, folds, self.contract)
        unique_folds = sorted(int(f) for f in np.unique(fold_arr))

        if test_df is not None and self.contract.id_col not in test_df.columns:
            raise KeyError(f"test_df에 식별자 컬럼 '{self.contract.id_col}'이 없습니다.")

        use_cache = (
            self.reuse_cached_artifacts
            if reuse_cached_artifacts is None
            else bool(reuse_cached_artifacts)
        )
        if use_cache and output_dir is not None:
            cached_res = self._try_load_cached_result(
                output_dir=output_dir,
                row_ids=row_ids,
                fold_arr=fold_arr,
                y_all=y_all,
                oot_arr=oot_arr,
                test_df=test_df,
            )
            if cached_res is not None:
                return cached_res

        if pipeline is not None:
            if assume_static_prepared:
                static_train = train_df
                static_test = test_df
            else:
                static_train = pipeline.prepare_static_features(
                    train_df,
                    seq_summary=train_seq_summary,
                    key_col=self.contract.row_id_col,
                )
                static_test = (
                    pipeline.prepare_static_features(
                        test_df,
                        seq_summary=test_seq_summary,
                        key_col=self.contract.id_col,
                    )
                    if test_df is not None
                    else None
                )
            base_feature_cols = None
        else:
            reserved = set(self.contract.reserved_cols)
            if feature_cols is not None:
                base_feature_cols = [c for c in feature_cols if c not in reserved]
            else:
                base_feature_cols = [c for c in train_df.columns if c not in reserved]
            static_train = train_df
            static_test = test_df

        oof_probs = np.empty(n_rows, dtype=np.float64)
        oof_raw_unshifted = (
            np.empty(n_rows, dtype=np.float64) if self.weighting_mode == "prior_shift" else None
        )
        n_test = len(test_df) if test_df is not None else 0
        test_acc = np.zeros(n_test, dtype=np.float64) if test_df is not None else None

        fold_metrics: list[dict[str, Any]] = []
        importance_records: dict[str, list[float]] = {}

        for fold_id in unique_folds:
            va_mask = fold_arr == fold_id
            tr_mask = ~va_mask
            fold_encoders: FoldFittedEncoders | None = None

            if pipeline is not None:
                # 학습 시에는 Test 피처 프레임(1.5 GB+)을 생성하지 않고 폴드 인코더만 유지
                bundle = pipeline.fit_transform_fold(
                    static_train,
                    y_all,
                    fold_arr,
                    fold_id=fold_id,
                    test_df=None,
                    assume_static_prepared=True,
                )
                X_tr_raw = bundle.X_train
                y_tr = bundle.y_train
                X_va_raw = bundle.X_valid
                y_va = bundle.y_valid
                fold_encoders = bundle.encoders
                cat_list = list(
                    dict.fromkeys(
                        [
                            *self.categorical_features,
                            *(categorical_features or ()),
                            *bundle.categorical_features,
                        ]
                    )
                )
                del bundle
            else:
                assert base_feature_cols is not None
                X_tr_raw = static_train.loc[tr_mask, base_feature_cols]
                y_tr = y_all[tr_mask]
                X_va_raw = static_train.loc[va_mask, base_feature_cols]
                y_va = y_all[va_mask]
                cat_list = list(
                    dict.fromkeys([*self.categorical_features, *(categorical_features or ())])
                )
            cat_list = [c for c in cat_list if c in X_tr_raw.columns]

            X_tr_fit, X_va_fit, effective_cats, frame_enc = _fit_prepare_train_val(
                X_tr_raw,
                X_va_raw,
                categorical_features=cat_list,
                backend=self.backend,
            )
            del X_tr_raw, X_va_raw
            gc.collect()

            n_tr_pos = int((y_tr == 1).sum())
            n_tr_neg = int((y_tr == 0).sum())
            if n_tr_pos == 0 or n_tr_neg == 0:
                raise ValueError(f"Fold {fold_id} 학습 세트에 양/음성 클래스가 모두 필요합니다.")

            if self.weighting_mode == "balanced_weight":
                spw: float | None = float(n_tr_neg / n_tr_pos)
                n_va_pos = max(int((y_va == 1).sum()), 1)
                n_va_neg = max(int((y_va == 0).sum()), 1)
                va_weight: np.ndarray | None = np.where(
                    y_va == 1, float(n_va_neg / n_va_pos), 1.0
                ).astype(np.float64)
            else:
                spw = None
                va_weight = None

            model, va_pred_raw, fold_imp, best_iter = self._dispatch_backend(
                X_tr_fit,
                y_tr,
                X_va_fit,
                y_va,
                effective_cats,
                scale_pos_weight=spw,
                va_sample_weight=va_weight,
                fold_seed=self.seed + fold_id,
            )

            for fname, imp_val in zip(X_tr_fit.columns, fold_imp):
                importance_records.setdefault(fname, []).append(float(imp_val))

            # Test 청크 변환·추론 전 학습/검증 프레임 메모리 즉시 반환
            del X_tr_fit, X_va_fit
            gc.collect()

            te_pred_raw: np.ndarray | None = None
            if static_test is not None and test_acc is not None:
                te_pred_raw = np.empty(n_test, dtype=np.float64)
                for start in range(0, n_test, _TEST_INFER_CHUNK_SIZE):
                    end = min(start + _TEST_INFER_CHUNK_SIZE, n_test)
                    test_slice = static_test.iloc[start:end]
                    if fold_encoders is not None:
                        raw_chunk = fold_encoders.transform_df(test_slice)
                    else:
                        assert base_feature_cols is not None
                        raw_chunk = test_slice[base_feature_cols]
                    fit_chunk = frame_enc.transform(raw_chunk)
                    del raw_chunk
                    te_pred_raw[start:end] = _predict_model_chunk(
                        model=model,
                        fit_chunk=fit_chunk,
                        backend=self.backend,
                        effective_cats=effective_cats,
                    )
                    del fit_chunk

            va_pred_raw = clip_probs(va_pred_raw)
            if self.weighting_mode == "prior_shift":
                train_pos_rate = float(n_tr_pos / len(y_tr))
                va_pred = clip_probs(
                    shift_prob_prior(
                        va_pred_raw,
                        train_pos_rate=train_pos_rate,
                        target_pos_rate=self.target_pos_rate,
                    )
                )
                assert oof_raw_unshifted is not None
                oof_raw_unshifted[va_mask] = va_pred_raw
            else:
                va_pred = va_pred_raw

            oof_probs[va_mask] = va_pred

            if test_acc is not None and te_pred_raw is not None:
                te_pred_clipped = clip_probs(te_pred_raw)
                del te_pred_raw
                if self.weighting_mode == "prior_shift":
                    train_pos_rate = float(n_tr_pos / len(y_tr))
                    te_pred_clipped = clip_probs(
                        shift_prob_prior(
                            te_pred_clipped,
                            train_pos_rate=train_pos_rate,
                            target_pos_rate=self.target_pos_rate,
                        )
                    )
                if self.test_agg == "logit_mean":
                    test_acc += _probs_to_logits(te_pred_clipped)
                else:
                    test_acc += te_pred_clipped

            f_eval = evaluate_predictions(y_va, va_pred)
            f_dict = f_eval.to_dict()
            f_dict.update(
                {
                    "fold_id": int(fold_id),
                    "n_train": int(len(y_tr)),
                    "n_valid": int(len(y_va)),
                    "best_iteration": best_iter,
                }
            )
            fold_metrics.append(f_dict)

            del model, fold_encoders, frame_enc
            gc.collect()

        oof_probs_clipped = clip_probs(oof_probs)
        oof_logits = _probs_to_logits(oof_probs_clipped)
        overall_metrics = evaluate_predictions(y_all, oof_probs_clipped)

        unshifted_metrics = (
            evaluate_predictions(y_all, clip_probs(oof_raw_unshifted))
            if oof_raw_unshifted is not None
            else None
        )

        oot_metrics: EvalResult | None = None
        if oot_arr is not None and np.any(oot_arr):
            y_oot = y_all[oot_arr]
            if np.any(y_oot == 1) and np.any(y_oot == 0):
                oot_metrics = evaluate_predictions(y_oot, oof_probs_clipped[oot_arr])

        oof_df = pd.DataFrame(
            {
                self.contract.row_id_col: row_ids.astype(np.int64),
                self.contract.fold_id_col: fold_arr.astype(np.int8),
                self.contract.target_col: y_all.astype(np.int8),
                "raw_logit": oof_logits.astype(np.float64),
                "pred_prob": oof_probs_clipped.astype(np.float64),
            }
        )
        oof_df = oof_df.sort_values(self.contract.row_id_col, kind="mergesort").reset_index(
            drop=True
        )

        out_test_df: pd.DataFrame | None = None
        if test_df is not None and test_acc is not None:
            n_folds_used = float(len(unique_folds))
            if self.test_agg == "logit_mean":
                mean_logits = test_acc / n_folds_used
                test_probs = _logits_to_probs(mean_logits)
                test_logits = _probs_to_logits(test_probs)
            else:
                test_probs = clip_probs(test_acc / n_folds_used)
                test_logits = _probs_to_logits(test_probs)

            out_test_df = pd.DataFrame(
                {
                    self.contract.id_col: test_df[self.contract.id_col].to_numpy(),
                    "raw_logit": test_logits.astype(np.float64),
                    "pred_prob": test_probs.astype(np.float64),
                }
            )

        fi_rows = [
            {
                "feature": fname,
                "importance_mean": float(np.mean(vals)),
                "importance_std": float(np.std(vals)),
            }
            for fname, vals in importance_records.items()
        ]
        fi_df = (
            pd.DataFrame(fi_rows)
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
            if fi_rows
            else pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        )

        result = GBDTTrainResult(
            model_name=self.model_name,
            backend=self.backend,
            oof_df=oof_df,
            test_df=out_test_df,
            overall_metrics=overall_metrics,
            fold_metrics=fold_metrics,
            oot_metrics=oot_metrics,
            feature_importance=fi_df,
            unshifted_metrics=unshifted_metrics,
            saved_paths={},
        )

        if output_dir is not None:
            self.save_artifacts(result, output_dir=output_dir)

        return result
