"""이종 모델(GBDT + DeepCTR) OOF/Test 정렬 검증 및 Log-odds 공간 CV 블렌딩 모듈."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from toss_ctr.data.schema import DEFAULT_CONTRACT, DatasetContract
from toss_ctr.evaluation.metrics import (
    EvalResult,
    clip_probs,
    evaluate_predictions,
    weighted_logloss_50_50,
)
from toss_ctr.models.calibration import logit_to_prob, prob_to_logit

BlendObjective = Literal["combined_score", "wll"]
BlendSpace = Literal["logit", "prob"]
TestAggMode = Literal["logit_mean", "prob_mean"]


def _id_arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """정수/문자열 ID 배열이 Pandas 3.x dtype 차이와 무관하게 100% 동일한지 검사합니다."""
    if len(left) != len(right):
        return False
    if left.dtype == right.dtype and np.array_equal(left, right):
        return True
    left_is_num = np.issubdtype(left.dtype, np.number)
    right_is_num = np.issubdtype(right.dtype, np.number)
    if left_is_num != right_is_num:
        return False
    return bool(np.array_equal(np.asarray(left, dtype=object), np.asarray(right, dtype=object)))


def assert_oof_alignment(
    oof_dfs: Mapping[str, pd.DataFrame],
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """모든 모델 OOF DataFrame의 `row_id`, `fold_id`, `clicked` 및 정렬 일관성을 검증합니다.

    검증 항목:
    1. 필수 컬럼(`row_id`, `fold_id`, `clicked`, `raw_logit`, `pred_prob`) 존재 및 결측/비유한 값 0건
    2. `row_id` 오름차순 유일 정렬 (`row_id[i] < row_id[i+1]`)
    3. 모든 모델 간 행 수, `row_id`, `fold_id`, `clicked` 100% 일치
    """
    if not oof_dfs:
        raise ValueError("oof_dfs가 비어 있습니다. 최소 1개 이상의 모델 OOF DataFrame이 필요합니다.")

    required_cols = (
        contract.row_id_col,
        contract.fold_id_col,
        contract.target_col,
        "raw_logit",
        "pred_prob",
    )

    ref_name: str | None = None
    ref_row_ids: np.ndarray | None = None
    ref_fold_ids: np.ndarray | None = None
    ref_clicked: np.ndarray | None = None

    for model_name, df in oof_dfs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"[{model_name}] OOF 입력은 pd.DataFrame이어야 합니다.")

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"[{model_name}] OOF 필수 컬럼이 누락되었습니다: {missing}")

        if len(df) == 0:
            raise ValueError(f"[{model_name}] OOF DataFrame이 비어 있습니다.")

        for col in required_cols:
            if df[col].isna().any():
                raise ValueError(f"[{model_name}] OOF 컬럼 '{col}'에 결측치가 존재합니다.")

        raw_logits = df["raw_logit"].to_numpy(dtype=np.float64, copy=False)
        pred_probs = df["pred_prob"].to_numpy(dtype=np.float64, copy=False)
        if not np.all(np.isfinite(raw_logits)):
            raise ValueError(f"[{model_name}] OOF 'raw_logit'에 NaN 또는 Inf가 포함되어 있습니다.")
        if not np.all(np.isfinite(pred_probs)) or np.any(pred_probs < 0.0) or np.any(pred_probs > 1.0):
            raise ValueError(f"[{model_name}] OOF 'pred_prob'는 [0, 1] 구간의 유한 값이어야 합니다.")

        row_ids = df[contract.row_id_col].to_numpy(dtype=np.int64, copy=False)
        fold_ids = df[contract.fold_id_col].to_numpy(dtype=np.int64, copy=False)
        clicked = df[contract.target_col].to_numpy(dtype=np.int8, copy=False)

        if len(row_ids) > 1 and not bool(np.all(row_ids[:-1] < row_ids[1:])):
            raise ValueError(
                f"[{model_name}] OOF '{contract.row_id_col}'가 중복 없는 엄격한 오름차순이 아닙니다."
            )

        if ref_name is None:
            ref_name = model_name
            ref_row_ids = row_ids
            ref_fold_ids = fold_ids
            ref_clicked = clicked
        else:
            assert ref_row_ids is not None and ref_fold_ids is not None and ref_clicked is not None
            if len(row_ids) != len(ref_row_ids):
                raise ValueError(
                    f"모델 '{model_name}'({len(row_ids)}행)과 기준 '{ref_name}'({len(ref_row_ids)}행)의 "
                    "OOF 행 수가 일치하지 않습니다."
                )
            if not np.array_equal(row_ids, ref_row_ids):
                raise ValueError(
                    f"모델 '{model_name}'과 '{ref_name}' 간 OOF '{contract.row_id_col}' 순서가 일치하지 않습니다."
                )
            if not np.array_equal(fold_ids, ref_fold_ids):
                raise ValueError(
                    f"모델 '{model_name}'과 '{ref_name}' 간 OOF '{contract.fold_id_col}' 분할이 일치하지 않습니다."
                )
            if not np.array_equal(clicked, ref_clicked):
                raise ValueError(
                    f"모델 '{model_name}'과 '{ref_name}' 간 OOF '{contract.target_col}' 타겟 값이 일치하지 않습니다."
                )

    assert ref_row_ids is not None and ref_fold_ids is not None and ref_clicked is not None
    return ref_row_ids, ref_fold_ids, ref_clicked


def assert_test_alignment(
    test_dfs: Mapping[str, pd.DataFrame],
    reference_ids: ArrayLike | None = None,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> np.ndarray:
    """모든 모델 Test 예측 DataFrame의 `ID` 순서 및 무결성을 검증합니다."""
    if not test_dfs:
        raise ValueError("test_dfs가 비어 있습니다.")

    id_col = contract.id_col
    required_cols = (id_col, "raw_logit", "pred_prob")
    ref_name: str | None = None
    ref_ids: np.ndarray | None = (
        np.asarray(reference_ids) if reference_ids is not None else None
    )

    for model_name, df in test_dfs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"[{model_name}] Test 입력은 pd.DataFrame이어야 합니다.")

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"[{model_name}] Test 예측 필수 컬럼이 누락되었습니다: {missing}")

        if len(df) == 0:
            raise ValueError(f"[{model_name}] Test 예측 DataFrame이 비어 있습니다.")

        for col in required_cols:
            if df[col].isna().any():
                raise ValueError(f"[{model_name}] Test 예측 컬럼 '{col}'에 결측치가 존재합니다.")

        raw_logits = df["raw_logit"].to_numpy(dtype=np.float64, copy=False)
        pred_probs = df["pred_prob"].to_numpy(dtype=np.float64, copy=False)
        if not np.all(np.isfinite(raw_logits)):
            raise ValueError(f"[{model_name}] Test 'raw_logit'에 NaN 또는 Inf가 포함되어 있습니다.")
        if not np.all(np.isfinite(pred_probs)) or np.any(pred_probs < 0.0) or np.any(pred_probs > 1.0):
            raise ValueError(f"[{model_name}] Test 'pred_prob'는 [0, 1] 구간의 유한 값이어야 합니다.")

        ids = df[id_col].to_numpy()
        if ref_ids is None:
            ref_name = model_name
            ref_ids = ids
        else:
            src_label = ref_name if ref_name is not None else "reference_ids"
            if len(ids) != len(ref_ids):
                raise ValueError(
                    f"모델 '{model_name}'({len(ids)}행)과 '{src_label}'({len(ref_ids)}행)의 Test 행 수가 일치하지 않습니다."
                )
            if not _id_arrays_equal(ids, ref_ids):
                raise ValueError(
                    f"모델 '{model_name}'과 '{src_label}' 간 Test '{id_col}' 순서가 100% 일치하지 않습니다."
                )
            if ref_name is None:
                ref_name = model_name

    assert ref_ids is not None
    return ref_ids


def load_prediction_artifacts(
    preds_dir: str | Path,
    model_names: Sequence[str] | None = None,
    require_test: bool = False,
    skip_missing: bool = False,
    reference_test_ids: ArrayLike | None = None,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """`preds_dir`에서 `oof_<name>.parquet` 및 `test_<name>.parquet`을 로드하고 정합성을 검증합니다."""
    base_dir = Path(preds_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"예측 아티팩트 디렉터리를 찾을 수 없습니다: {base_dir}")

    if model_names is None:
        discovered = sorted(base_dir.glob("oof_*.parquet"))
        candidates = [p.stem[len("oof_") :] for p in discovered]
    else:
        candidates = list(model_names)

    if not candidates:
        raise ValueError(f"로드할 OOF 모델 후보가 없습니다: {base_dir}")

    oof_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}

    for name in candidates:
        oof_path = base_dir / f"oof_{name}.parquet"
        test_path = base_dir / f"test_{name}.parquet"

        if not oof_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"OOF 예측 파일을 찾을 수 없습니다: {oof_path}")

        if require_test and not test_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Test 예측 파일을 찾을 수 없습니다: {test_path}")

        oof_dfs[name] = pq.read_table(oof_path).to_pandas()
        if test_path.exists():
            test_dfs[name] = pq.read_table(test_path).to_pandas()

    if not oof_dfs:
        raise ValueError(
            f"조건(require_test={require_test})을 만족하는 예측 파일을 찾지 못했습니다: {candidates}"
        )

    # 일부 모델만 Test 파일이 있는 불일치 방지
    if test_dfs and set(test_dfs.keys()) != set(oof_dfs.keys()):
        if require_test:
            missing_t = sorted(set(oof_dfs.keys()) - set(test_dfs.keys()))
            raise ValueError(f"일부 모델의 Test 예측 파일이 누락되었습니다: {missing_t}")
        test_dfs = {}

    assert_oof_alignment(oof_dfs, contract=contract)
    if test_dfs:
        assert_test_alignment(test_dfs, reference_ids=reference_test_ids, contract=contract)

    return oof_dfs, test_dfs


@dataclass
class BlendResult:
    """OOF Log-odds 앙상블 블렌딩 결과 컨테이너."""

    model_names: list[str]
    weights: dict[str, float]
    fold_weights: list[dict[str, Any]]
    oof_df: pd.DataFrame
    test_df: pd.DataFrame | None
    oof_metrics: EvalResult
    fold_metrics: list[dict[str, Any]]
    single_model_metrics: dict[str, dict[str, float]]
    correlation_matrix: pd.DataFrame


class OOFLogitBlender:
    """`fold_id` 기반 교차 검증으로 비음수 Log-odds 가중치(`w >= 0, sum(w) = 1`)를 최적화하는 블렌더.

    각 폴드 `k`에 대해 오직 `fold_id != k` 행에서만 `scipy.optimize.minimize`로 가중치 `w^{(k)}`를
    최적화한 뒤 `fold_id == k` 검증 행을 결합하므로 블렌딩 가중치 과적합 없는 엄격한 OOF 점수를 산출합니다.
    """

    def __init__(
        self,
        objective: BlendObjective = "combined_score",
        blend_space: BlendSpace = "logit",
        opt_max_samples: int | None = 500_000,
        n_restarts: int = 24,
        max_iter: int = 120,
        test_agg: TestAggMode = "logit_mean",
        seed: int = 42,
        contract: DatasetContract = DEFAULT_CONTRACT,
    ) -> None:
        if objective not in ("combined_score", "wll"):
            raise ValueError(f"objective는 'combined_score' 또는 'wll'이어야 합니다: {objective!r}")
        if blend_space not in ("logit", "prob"):
            raise ValueError(f"blend_space는 'logit' 또는 'prob'이어야 합니다: {blend_space!r}")
        if test_agg not in ("logit_mean", "prob_mean"):
            raise ValueError(f"test_agg는 'logit_mean' 또는 'prob_mean'이어야 합니다: {test_agg!r}")

        self.objective: BlendObjective = objective
        self.blend_space: BlendSpace = blend_space
        self.opt_max_samples = int(opt_max_samples) if opt_max_samples is not None else None
        self.n_restarts = int(n_restarts)
        self.max_iter = int(max_iter)
        self.test_agg: TestAggMode = test_agg
        self.seed = int(seed)
        self.contract = contract

        self.model_names_: list[str] = []
        self.weights_: dict[str, float] = {}
        self.weight_vector_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.fold_weights_: list[dict[str, Any]] = []
        self.fold_weight_matrix_: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.is_fitted_: bool = False

    def _build_feature_matrix(self, dfs: Mapping[str, pd.DataFrame], model_names: Sequence[str]) -> np.ndarray:
        cols: list[np.ndarray] = []
        for name in model_names:
            if self.blend_space == "logit":
                vals = dfs[name]["raw_logit"].to_numpy(dtype=np.float64, copy=False)
            else:
                vals = clip_probs(dfs[name]["pred_prob"].to_numpy(dtype=np.float64, copy=False))
            cols.append(vals)
        return np.column_stack(cols)

    def _stratified_subsample(
        self,
        mat: np.ndarray,
        y: np.ndarray,
        fold_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_rows = len(y)
        if self.opt_max_samples is None or n_rows <= self.opt_max_samples:
            return mat, y

        rng = np.random.default_rng(fold_seed)
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        n_pos_take = max(1, int(round(self.opt_max_samples * (len(pos_idx) / n_rows))))
        n_pos_take = min(n_pos_take, len(pos_idx))
        n_neg_take = max(1, min(self.opt_max_samples - n_pos_take, len(neg_idx)))

        sel_pos = rng.choice(pos_idx, size=n_pos_take, replace=False)
        sel_neg = rng.choice(neg_idx, size=n_neg_take, replace=False)
        sel = np.sort(np.concatenate([sel_pos, sel_neg]))
        return mat[sel], y[sel]

    @staticmethod
    def _normalize_weights(raw_w: np.ndarray) -> np.ndarray:
        w = np.maximum(np.asarray(raw_w, dtype=np.float64), 0.0)
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 1.0e-12:
            return np.full(len(w), 1.0 / float(len(w)), dtype=np.float64)
        return w / total

    def _optimize_single_fold(
        self,
        mat_tr: np.ndarray,
        y_tr: np.ndarray,
        fold_id: int,
    ) -> np.ndarray:
        n_models = mat_tr.shape[1]
        if n_models == 1:
            return np.array([1.0], dtype=np.float64)

        mat_opt, y_opt = self._stratified_subsample(
            mat_tr, y_tr, fold_seed=self.seed + int(fold_id) * 101
        )

        def _eval_loss(raw_w: np.ndarray) -> float:
            w = self._normalize_weights(raw_w)
            combined = mat_opt @ w
            probs = (
                logit_to_prob(combined)
                if self.blend_space == "logit"
                else clip_probs(combined)
            )
            if self.objective == "wll":
                wll, _, _ = weighted_logloss_50_50(y_opt, probs)
                return float(wll)
            eval_res = evaluate_predictions(y_opt, probs)
            return float(-eval_res.score)

        # 단일 모델 꼭짓점, 균등 가중치, Dirichlet 다중 시작점 중 최고 초기점 탐색
        candidates: list[np.ndarray] = [np.full(n_models, 1.0 / n_models, dtype=np.float64)]
        for idx in range(n_models):
            corner = np.zeros(n_models, dtype=np.float64)
            corner[idx] = 1.0
            candidates.append(corner)

        if self.n_restarts > 0:
            rng = np.random.default_rng(self.seed + int(fold_id) * 37)
            dir_samples = rng.dirichlet(np.ones(n_models, dtype=np.float64), size=self.n_restarts)
            candidates.extend(dir_samples)

        best_x = min(candidates, key=_eval_loss)
        best_val = _eval_loss(best_x)

        if self.objective == "wll":
            res = minimize(
                _eval_loss,
                x0=best_x,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * n_models,
                constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
                options={"maxiter": self.max_iter, "ftol": 1e-10},
            )
        else:
            res = minimize(
                _eval_loss,
                x0=best_x,
                method="Powell",
                bounds=[(0.0, 1.0)] * n_models,
                options={"maxiter": self.max_iter, "ftol": 1e-8, "xtol": 1e-6},
            )

        opt_w = self._normalize_weights(res.x)
        opt_val = _eval_loss(opt_w)
        if opt_val <= best_val:
            return opt_w
        return self._normalize_weights(best_x)

    def fit_blend(
        self,
        oof_dfs: Mapping[str, pd.DataFrame],
        test_dfs: Mapping[str, pd.DataFrame] | None = None,
        reference_test_ids: ArrayLike | None = None,
    ) -> BlendResult:
        """OOF 가중치 최적화·교차 검증 및 선택적 Test 예측 블렌딩을 수행합니다."""
        row_ids, fold_ids, y_all = assert_oof_alignment(oof_dfs, contract=self.contract)
        model_names = list(oof_dfs.keys())
        n_rows = len(row_ids)
        n_models = len(model_names)

        if test_dfs is not None:
            if set(test_dfs.keys()) != set(model_names):
                raise ValueError(
                    f"oof_dfs({model_names})와 test_dfs({list(test_dfs.keys())})의 모델 구성이 다릅니다."
                )
            test_ids = assert_test_alignment(
                test_dfs, reference_ids=reference_test_ids, contract=self.contract
            )
        else:
            test_ids = None

        unique_folds = sorted(int(f) for f in np.unique(fold_ids))
        if len(unique_folds) < 2:
            raise ValueError(f"CV 블렌딩을 위해서는 최소 2개 이상의 폴드가 필요합니다: {unique_folds}")

        mat_oof = self._build_feature_matrix(oof_dfs, model_names)

        # 단일 모델별 벤치마크 및 예측 상관관계(Diversity) 집계
        single_metrics: dict[str, dict[str, float]] = {}
        logit_cols: dict[str, np.ndarray] = {}
        for idx, name in enumerate(model_names):
            m_probs = clip_probs(oof_dfs[name]["pred_prob"].to_numpy(dtype=np.float64, copy=False))
            single_metrics[name] = evaluate_predictions(y_all, m_probs).to_dict()
            logit_cols[name] = oof_dfs[name]["raw_logit"].to_numpy(dtype=np.float64, copy=False)
        corr_matrix = pd.DataFrame(logit_cols).corr(method="pearson")

        oof_blended_vals = np.empty(n_rows, dtype=np.float64)
        fold_weights_list: list[dict[str, Any]] = []
        fold_metrics_list: list[dict[str, Any]] = []
        fold_weight_mat = np.zeros((len(unique_folds), n_models), dtype=np.float64)

        for f_idx, f_id in enumerate(unique_folds):
            va_mask = fold_ids == f_id
            tr_mask = ~va_mask
            if not np.any(tr_mask) or not np.any(va_mask):
                raise ValueError(f"폴드 {f_id}의 학습 또는 검증 행이 비어 있습니다.")

            w_fold = self._optimize_single_fold(mat_oof[tr_mask], y_all[tr_mask], fold_id=f_id)
            fold_weight_mat[f_idx] = w_fold

            va_combined = mat_oof[va_mask] @ w_fold
            oof_blended_vals[va_mask] = va_combined
            va_probs = (
                logit_to_prob(va_combined)
                if self.blend_space == "logit"
                else clip_probs(va_combined)
            )
            f_eval = evaluate_predictions(y_all[va_mask], va_probs)

            w_dict = {name: float(w_fold[i]) for i, name in enumerate(model_names)}
            fold_weights_list.append({"fold_id": f_id, **w_dict})
            fold_metrics_list.append(
                {
                    "fold_id": f_id,
                    "n_valid": int(np.sum(va_mask)),
                    **f_eval.to_dict(),
                }
            )

        if self.blend_space == "logit":
            oof_probs = logit_to_prob(oof_blended_vals)
            oof_logits = oof_blended_vals
        else:
            oof_probs = clip_probs(oof_blended_vals)
            oof_logits = prob_to_logit(oof_probs)

        oof_metrics = evaluate_predictions(y_all, oof_probs)
        mean_w = self._normalize_weights(np.mean(fold_weight_mat, axis=0))

        self.model_names_ = model_names
        self.weight_vector_ = mean_w
        self.weights_ = {name: float(mean_w[i]) for i, name in enumerate(model_names)}
        self.fold_weights_ = fold_weights_list
        self.fold_weight_matrix_ = fold_weight_mat
        self.is_fitted_ = True

        oof_df = pd.DataFrame(
            {
                self.contract.row_id_col: row_ids.astype(np.int64),
                self.contract.fold_id_col: fold_ids.astype(np.int8),
                self.contract.target_col: y_all.astype(np.int8),
                "raw_logit": oof_logits.astype(np.float64),
                "pred_prob": oof_probs.astype(np.float64),
            }
        )

        out_test_df: pd.DataFrame | None = None
        if test_dfs is not None and test_ids is not None:
            out_test_df = self.transform_test(test_dfs, reference_ids=test_ids)

        return BlendResult(
            model_names=model_names,
            weights=dict(self.weights_),
            fold_weights=fold_weights_list,
            oof_df=oof_df,
            test_df=out_test_df,
            oof_metrics=oof_metrics,
            fold_metrics=fold_metrics_list,
            single_model_metrics=single_metrics,
            correlation_matrix=corr_matrix,
        )

    def transform_test(
        self,
        test_dfs: Mapping[str, pd.DataFrame],
        reference_ids: ArrayLike | None = None,
    ) -> pd.DataFrame:
        """학습된 폴드 가중치로 Test 예측(`ID, raw_logit, pred_prob`)을 블렌딩합니다."""
        if not self.is_fitted_:
            raise RuntimeError("transform_test 호출 전에 fit_blend()를 먼저 실행해야 합니다.")
        if set(test_dfs.keys()) != set(self.model_names_):
            raise ValueError(
                f"학습 시 모델({self.model_names_})과 test_dfs({list(test_dfs.keys())})가 일치하지 않습니다."
            )

        test_ids = assert_test_alignment(
            test_dfs, reference_ids=reference_ids, contract=self.contract
        )
        mat_test = self._build_feature_matrix(test_dfs, self.model_names_)

        if self.test_agg == "logit_mean" and self.blend_space == "logit":
            test_logits = mat_test @ self.weight_vector_
            test_probs = logit_to_prob(test_logits)
        else:
            n_folds = float(self.fold_weight_matrix_.shape[0])
            prob_acc = np.zeros(len(test_ids), dtype=np.float64)
            for k in range(self.fold_weight_matrix_.shape[0]):
                combined_k = mat_test @ self.fold_weight_matrix_[k]
                p_k = (
                    logit_to_prob(combined_k)
                    if self.blend_space == "logit"
                    else clip_probs(combined_k)
                )
                prob_acc += p_k
            test_probs = clip_probs(prob_acc / n_folds)
            test_logits = prob_to_logit(test_probs)

        return pd.DataFrame(
            {
                self.contract.id_col: test_ids,
                "raw_logit": test_logits.astype(np.float64),
                "pred_prob": test_probs.astype(np.float64),
            }
        )
