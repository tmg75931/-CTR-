"""50:50 WLL 정밀 확률 보정(Affine Logit, Temperature Scaling, Group Shrinkage, Out-of-Fold) 모듈."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from toss_ctr.evaluation.metrics import EPSILON, clip_probs, weighted_logloss_50_50

InputSpace = Literal["logit", "prob"]
TestAggMode = Literal["logit_mean", "prob_mean"]

DEFAULT_CLIP_LOGIT: float = 35.0


def prob_to_logit(y_pred: ArrayLike, eps: float = EPSILON) -> np.ndarray:
    """예측 확률을 안전한 클리핑 후 Log-odds(Logit) 공간으로 변환합니다."""
    p = clip_probs(y_pred, eps=eps)
    if not np.all(np.isfinite(p)):
        raise ValueError("prob_to_logit 입력에 NaN 또는 Inf가 포함되어 있습니다.")
    return np.log(p / (1.0 - p)).astype(np.float64, copy=False)


def logit_to_prob(
    logits: ArrayLike,
    clip_logit: float = DEFAULT_CLIP_LOGIT,
    eps: float = EPSILON,
) -> np.ndarray:
    """Log-odds(Logit) 값을 수치 오버플로우 없이 확률 `(eps, 1 - eps)`로 변환합니다."""
    arr = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("logit_to_prob 입력에 NaN 또는 Inf가 포함되어 있습니다.")
    z = np.clip(arr, -float(clip_logit), float(clip_logit))
    probs = 1.0 / (1.0 + np.exp(-z))
    return clip_probs(probs, eps=eps)


def _coerce_1d_logits(
    inputs: ArrayLike,
    input_space: InputSpace = "logit",
    eps: float = EPSILON,
) -> np.ndarray:
    """입력을 검증된 1차원 `float64` 로짓 배열로 정규화합니다."""
    arr = np.asarray(inputs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"1차원 입력 배열이 필요합니다. 실제 shape={arr.shape}")
    if len(arr) == 0:
        raise ValueError("빈 입력 배열은 처리할 수 없습니다.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("보정 입력 배열에 NaN 또는 Inf가 포함되어 있습니다.")
    if input_space == "prob":
        return prob_to_logit(arr, eps=eps)
    if input_space == "logit":
        return arr
    raise ValueError(f"지원하지 않는 input_space입니다: {input_space!r}")


def _extract_binary_targets(y_true: ArrayLike, n_rows: int) -> np.ndarray:
    """이진 타겟(`0`/`1`) 1D `int8` 배열을 검증·추출합니다."""
    arr = np.asarray(y_true)
    if arr.ndim != 1 or len(arr) != n_rows:
        raise ValueError(
            f"y_true 길이가 입력과 일치하지 않습니다: expected={n_rows}, actual={arr.shape}"
        )
    if pd.isna(arr).any():
        raise ValueError("y_true에 결측치가 포함되어 있습니다.")
    y_int = arr.astype(np.int8, copy=False)
    unique_vals = np.unique(y_int)
    if not np.array_equal(unique_vals, np.array([0, 1], dtype=np.int8)):
        raise ValueError(f"y_true에는 양성(1)과 음성(0) 클래스가 모두 포함되어야 합니다: {unique_vals.tolist()}")
    return y_int


def _extract_fold_ids(folds: pd.DataFrame | ArrayLike, n_rows: int) -> np.ndarray:
    """폴드 입력(`DataFrame` 또는 1D 배열)을 검증된 `int64` 배열로 변환합니다."""
    if isinstance(folds, pd.DataFrame):
        if "fold_id" not in folds.columns:
            raise KeyError("folds DataFrame에 'fold_id' 컬럼이 없습니다.")
        raw = folds["fold_id"].to_numpy()
    elif isinstance(folds, pd.Series):
        raw = folds.to_numpy()
    else:
        raw = np.asarray(folds)

    if raw.ndim != 1 or len(raw) != n_rows:
        raise ValueError(
            f"folds 길이가 입력과 일치하지 않습니다: expected={n_rows}, actual={raw.shape}"
        )
    if pd.isna(raw).any():
        raise ValueError("folds에 결측치가 포함되어 있습니다.")
    return raw.astype(np.int64, copy=False)


def _normalize_group_keys(groups: ArrayLike | pd.Series, n_rows: int) -> np.ndarray:
    """Pandas 3.x 호환성을 보장하며 그룹 키를 결정론적 문자열 배열로 정규화합니다."""
    if isinstance(groups, pd.Series):
        s = groups.astype(object) if isinstance(groups.dtype, pd.CategoricalDtype) else groups
    else:
        arr = np.asarray(groups)
        if arr.ndim != 1:
            raise ValueError(f"groups는 1차원이어야 합니다. 실제 shape={arr.shape}")
        s = pd.Series(arr)

    if len(s) != n_rows:
        raise ValueError(f"groups 길이가 일치하지 않습니다: expected={n_rows}, actual={len(s)}")

    na_mask = s.isna().to_numpy(dtype=bool, copy=False)
    keys = np.empty(n_rows, dtype=object)

    if pd.api.types.is_integer_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype):
        valid_mask = ~na_mask
        if np.any(valid_mask):
            vals = s.loc[valid_mask].to_numpy(dtype=np.int64)
            keys[valid_mask] = [str(int(v)) for v in vals]
    elif pd.api.types.is_float_dtype(s.dtype):
        arr_f = s.to_numpy(dtype=np.float64, na_value=np.nan, copy=False)
        na_mask = ~np.isfinite(arr_f)
        for idx in np.flatnonzero(~na_mask):
            val = float(arr_f[idx])
            if val.is_integer() and abs(val) <= 9e15:
                keys[idx] = str(int(val))
            else:
                keys[idx] = str(val)
    else:
        valid_mask = ~na_mask
        if np.any(valid_mask):
            keys[valid_mask] = s.loc[valid_mask].astype(str).to_numpy(dtype=object)

    if np.any(na_mask):
        keys[na_mask] = "__MISSING__"
    return keys


class AffineLogitCalibrator:
    """`p_cal = sigmoid(a * logit + b)` (`a > 0`) 형태의 50:50 WLL 직접 최적화 보정기.

    `a > 0` 제약을 강제해 단조 증가 변환을 보장하므로 순위 기반 지표인 **`AP`를 100% 보존**하면서,
    원천 CTR 불균형이나 온도 스케일 편향으로 왜곡된 `50:50 WLL`을 정밀하게 보정합니다.
    """

    def __init__(
        self,
        l2_reg: float = 1.0e-6,
        min_scale: float = 1.0e-3,
        max_scale: float = 20.0,
        max_bias: float = 25.0,
        max_iter: int = 200,
        fit_intercept: bool = True,
        input_space: InputSpace = "logit",
        clip_logit: float = DEFAULT_CLIP_LOGIT,
        eps: float = EPSILON,
    ) -> None:
        if l2_reg < 0.0:
            raise ValueError(f"l2_reg는 0 이상이어야 합니다: {l2_reg}")
        if min_scale <= 0.0 or max_scale <= min_scale:
            raise ValueError(
                f"유효하지 않은 scale 범위입니다: min_scale={min_scale}, max_scale={max_scale}"
            )
        if max_bias <= 0.0:
            raise ValueError(f"max_bias는 양수여야 합니다: {max_bias}")

        self.l2_reg = float(l2_reg)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.max_iter = int(max_iter)
        self.fit_intercept = bool(fit_intercept)
        self.input_space: InputSpace = input_space
        self.clip_logit = float(clip_logit)
        self.eps = float(eps)

        self.scale_: float = 1.0
        self.bias_: float = 0.0
        self.fit_loss_: float = float("nan")
        self.n_samples_: int = 0
        self.is_fitted_: bool = False

    def fit(
        self,
        inputs: ArrayLike,
        y_true: ArrayLike,
        input_space: InputSpace | None = None,
    ) -> AffineLogitCalibrator:
        """`scipy.optimize.minimize`로 `weighted_logloss_50_50` + L2 정규화를 최소화합니다."""
        space = input_space or self.input_space
        logits = _coerce_1d_logits(inputs, input_space=space, eps=self.eps)
        y_arr = _extract_binary_targets(y_true, n_rows=len(logits))

        pos_mask = y_arr == 1
        n_pos = int(np.sum(pos_mask))
        n_neg = len(y_arr) - n_pos
        w_pos = 0.5 / float(n_pos)
        w_neg = 0.5 / float(n_neg)
        l2 = self.l2_reg
        clip_l = self.clip_logit
        eps = self.eps

        if self.fit_intercept:

            def _objective(params: np.ndarray) -> tuple[float, np.ndarray]:
                a = float(params[0])
                b = float(params[1])
                u = a * logits + b
                p_cal = logit_to_prob(u, clip_logit=clip_l, eps=eps)
                wll, _, _ = weighted_logloss_50_50(y_arr, p_cal, eps=eps)
                reg = 0.5 * l2 * ((a - 1.0) ** 2 + b**2)

                diff = np.where(pos_mask, w_pos * (p_cal - 1.0), w_neg * p_cal)
                grad_a = float(np.dot(diff, logits)) + l2 * (a - 1.0)
                grad_b = float(np.sum(diff)) + l2 * b
                return wll + reg, np.array([grad_a, grad_b], dtype=np.float64)

            mean_logit = float(np.mean(logits))
            init_b = float(np.clip(-mean_logit, -self.max_bias * 0.8, self.max_bias * 0.8))
            candidates = [
                np.array([1.0, 0.0], dtype=np.float64),
                np.array([1.0, init_b], dtype=np.float64),
            ]
            x0 = min(candidates, key=lambda c: _objective(c)[0])
            res = minimize(
                _objective,
                x0=x0,
                method="L-BFGS-B",
                jac=True,
                bounds=[(self.min_scale, self.max_scale), (-self.max_bias, self.max_bias)],
                options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-8},
            )
            self.scale_ = float(np.clip(res.x[0], self.min_scale, self.max_scale))
            self.bias_ = float(np.clip(res.x[1], -self.max_bias, self.max_bias))
            self.fit_loss_ = float(res.fun)
        else:

            def _objective_temp(params: np.ndarray) -> tuple[float, np.ndarray]:
                a = float(params[0])
                u = a * logits
                p_cal = logit_to_prob(u, clip_logit=clip_l, eps=eps)
                wll, _, _ = weighted_logloss_50_50(y_arr, p_cal, eps=eps)
                reg = 0.5 * l2 * ((a - 1.0) ** 2)

                diff = np.where(pos_mask, w_pos * (p_cal - 1.0), w_neg * p_cal)
                grad_a = float(np.dot(diff, logits)) + l2 * (a - 1.0)
                return wll + reg, np.array([grad_a], dtype=np.float64)

            res = minimize(
                _objective_temp,
                x0=np.array([1.0], dtype=np.float64),
                method="L-BFGS-B",
                jac=True,
                bounds=[(self.min_scale, self.max_scale)],
                options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-8},
            )
            self.scale_ = float(np.clip(res.x[0], self.min_scale, self.max_scale))
            self.bias_ = 0.0
            self.fit_loss_ = float(res.fun)

        self.n_samples_ = int(len(logits))
        self.is_fitted_ = True
        return self

    def transform_logits(
        self,
        inputs: ArrayLike,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """보정된 Log-odds(`a * logit + b`)를 반환합니다."""
        if not self.is_fitted_:
            raise RuntimeError("transform_logits 호출 전에 fit()을 먼저 실행해야 합니다.")
        space = input_space or self.input_space
        logits = _coerce_1d_logits(inputs, input_space=space, eps=self.eps)
        return (self.scale_ * logits + self.bias_).astype(np.float64, copy=False)

    def predict_proba(
        self,
        inputs: ArrayLike,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """보정된 확률 `sigmoid(a * logit + b)`을 반환합니다."""
        cal_logits = self.transform_logits(inputs, input_space=input_space)
        return logit_to_prob(cal_logits, clip_logit=self.clip_logit, eps=self.eps)

    def transform(
        self,
        inputs: ArrayLike,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """`predict_proba`의 별칭입니다."""
        return self.predict_proba(inputs, input_space=input_space)

    def get_params_dict(self) -> dict[str, Any]:
        """보정 파라미터를 직렬화 가능한 딕셔너리로 반환합니다."""
        return {
            "calibrator_type": "AffineLogitCalibrator",
            "scale": self.scale_,
            "bias": self.bias_,
            "fit_intercept": self.fit_intercept,
            "l2_reg": self.l2_reg,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "max_bias": self.max_bias,
            "input_space": self.input_space,
            "fit_loss": self.fit_loss_,
            "n_samples": self.n_samples_,
        }

    def to_dict(self) -> dict[str, Any]:
        """`get_params_dict`의 직렬화 별칭입니다."""
        return self.get_params_dict()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AffineLogitCalibrator:
        """직렬화된 딕셔너리에서 `AffineLogitCalibrator`를 복원합니다."""
        obj = cls(
            l2_reg=float(payload.get("l2_reg", 1.0e-6)),
            min_scale=float(payload.get("min_scale", 1.0e-3)),
            max_scale=float(payload.get("max_scale", 20.0)),
            max_bias=float(payload.get("max_bias", 25.0)),
            fit_intercept=bool(payload.get("fit_intercept", True)),
            input_space=payload.get("input_space", "logit"),
        )
        obj.scale_ = float(payload["scale"])
        obj.bias_ = float(payload["bias"])
        obj.fit_loss_ = float(payload.get("fit_loss", float("nan")))
        obj.n_samples_ = int(payload.get("n_samples", 0))
        obj.is_fitted_ = True
        return obj


class TemperatureScalingCalibrator(AffineLogitCalibrator):
    """단일 온도 계수 `T = 1 / a` (`p_cal = sigmoid(logit / T)`)만 학습하는 보정기."""

    def __init__(
        self,
        l2_reg: float = 1.0e-6,
        min_scale: float = 1.0e-3,
        max_scale: float = 20.0,
        max_iter: int = 200,
        input_space: InputSpace = "logit",
    ) -> None:
        super().__init__(
            l2_reg=l2_reg,
            min_scale=min_scale,
            max_scale=max_scale,
            max_iter=max_iter,
            fit_intercept=False,
            input_space=input_space,
        )

    @property
    def temperature_(self) -> float:
        """학습된 온도 파라미터 `T = 1 / scale_`를 반환합니다."""
        return float(1.0 / self.scale_)


class GroupBiasShrinkageCalibrator:
    """지면(`inventory_id`) 등 그룹별 residual logit offset을 Empirical Bayes Shrinkage로 보정하는 보정기.

    1. 선택적으로 전역 `AffineLogitCalibrator`를 먼저 적합합니다(`fit_global=True`).
    2. `50:50 WLL` 클래스 균형 가중치 하에서 각 그룹의 residual logit offset `raw_delta_g`를 추정한 뒤,
       Empirical Bayes shrinkage 계수 `n_g / (n_g + shrinkage)`를 곱해 희소 그룹의 과적합을 방지합니다.
    3. 미등장 그룹(`n_g == 0`)은 추가 offset `0.0`(전역 보정)으로 안전하게 폴백합니다.
    """

    def __init__(
        self,
        shrinkage: float = 500.0,
        max_offset: float = 3.0,
        fit_global: bool = True,
        l2_reg: float = 1.0e-6,
        min_scale: float = 1.0e-3,
        max_scale: float = 20.0,
        newton_steps: int = 5,
        input_space: InputSpace = "logit",
        clip_logit: float = DEFAULT_CLIP_LOGIT,
        eps: float = EPSILON,
    ) -> None:
        if shrinkage < 0.0:
            raise ValueError(f"shrinkage는 0 이상이어야 합니다: {shrinkage}")
        if max_offset <= 0.0:
            raise ValueError(f"max_offset은 양수여야 합니다: {max_offset}")

        self.shrinkage = float(shrinkage)
        self.max_offset = float(max_offset)
        self.fit_global = bool(fit_global)
        self.l2_reg = float(l2_reg)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.newton_steps = int(newton_steps)
        self.input_space: InputSpace = input_space
        self.clip_logit = float(clip_logit)
        self.eps = float(eps)

        self.global_calibrator_: AffineLogitCalibrator | None = None
        self.group_offsets_: dict[str, float] = {}
        self.group_counts_: dict[str, int] = {}
        self.scale_: float = 1.0
        self.bias_: float = 0.0
        self.is_fitted_: bool = False

    def fit(
        self,
        inputs: ArrayLike,
        y_true: ArrayLike,
        groups: ArrayLike | pd.Series,
        input_space: InputSpace | None = None,
    ) -> GroupBiasShrinkageCalibrator:
        """전역 보정 후 그룹별 residual logit offset을 Empirical Bayes Shrinkage로 적합합니다."""
        space = input_space or self.input_space
        raw_logits = _coerce_1d_logits(inputs, input_space=space, eps=self.eps)
        n_rows = len(raw_logits)
        y_arr = _extract_binary_targets(y_true, n_rows=n_rows)
        group_keys = _normalize_group_keys(groups, n_rows=n_rows)

        if self.fit_global:
            global_cal = AffineLogitCalibrator(
                l2_reg=self.l2_reg,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                fit_intercept=True,
                input_space="logit",
                clip_logit=self.clip_logit,
                eps=self.eps,
            ).fit(raw_logits, y_arr, input_space="logit")
            self.global_calibrator_ = global_cal
            self.scale_ = global_cal.scale_
            self.bias_ = global_cal.bias_
            base_logits = global_cal.transform_logits(raw_logits, input_space="logit")
        else:
            self.global_calibrator_ = None
            self.scale_ = 1.0
            self.bias_ = 0.0
            base_logits = raw_logits

        n_pos = int(np.sum(y_arr == 1))
        n_neg = n_rows - n_pos
        # 50:50 WLL과 정확히 비례하면서 평균 가중치가 1.0인 정규화 샘플 가중치
        w_pos = 0.5 * float(n_rows) / float(n_pos)
        w_neg = 0.5 * float(n_rows) / float(n_neg)
        sample_w = np.where(y_arr == 1, w_pos, w_neg).astype(np.float64)

        codes, uniques = pd.factorize(pd.Series(group_keys), sort=False)
        self.group_offsets_ = {}
        self.group_counts_ = {}

        for g_idx, g_key in enumerate(uniques):
            mask = codes == g_idx
            n_g = int(np.sum(mask))
            if n_g == 0:
                continue

            z_g = base_logits[mask]
            y_g = y_arr[mask].astype(np.float64)
            w_g = sample_w[mask]

            delta = 0.0
            for _ in range(self.newton_steps):
                p_g = logit_to_prob(z_g + delta, clip_logit=self.clip_logit, eps=self.eps)
                grad = float(np.sum(w_g * (p_g - y_g)))
                hess = float(np.sum(w_g * p_g * (1.0 - p_g))) + 1.0e-3
                step = np.clip(-grad / hess, -1.0, 1.0)
                delta = float(np.clip(delta + step, -self.max_offset, self.max_offset))
                if abs(step) < 1e-6:
                    break

            shrink_factor = float(n_g / (n_g + self.shrinkage)) if self.shrinkage > 0.0 else 1.0
            shrunk_delta = float(np.clip(shrink_factor * delta, -self.max_offset, self.max_offset))

            key_str = str(g_key)
            self.group_offsets_[key_str] = shrunk_delta
            self.group_counts_[key_str] = n_g

        self.is_fitted_ = True
        return self

    def transform_logits(
        self,
        inputs: ArrayLike,
        groups: ArrayLike | pd.Series,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """전역 보정 로짓에 그룹별 shrunk residual offset을 가산합니다."""
        if not self.is_fitted_:
            raise RuntimeError("transform_logits 호출 전에 fit()을 먼저 실행해야 합니다.")
        space = input_space or self.input_space
        raw_logits = _coerce_1d_logits(inputs, input_space=space, eps=self.eps)
        group_keys = _normalize_group_keys(groups, n_rows=len(raw_logits))

        if self.global_calibrator_ is not None:
            base_logits = self.global_calibrator_.transform_logits(raw_logits, input_space="logit")
        else:
            base_logits = raw_logits

        offsets = pd.Series(group_keys).map(self.group_offsets_).to_numpy(
            dtype=np.float64, na_value=0.0, copy=True
        )
        return (base_logits + offsets).astype(np.float64, copy=False)

    def predict_proba(
        self,
        inputs: ArrayLike,
        groups: ArrayLike | pd.Series,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """그룹별 Shrinkage 보정 후 확률을 반환합니다."""
        cal_logits = self.transform_logits(inputs, groups=groups, input_space=input_space)
        return logit_to_prob(cal_logits, clip_logit=self.clip_logit, eps=self.eps)

    def transform(
        self,
        inputs: ArrayLike,
        groups: ArrayLike | pd.Series,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """`predict_proba`의 별칭입니다."""
        return self.predict_proba(inputs, groups=groups, input_space=input_space)

    def get_params_dict(self) -> dict[str, Any]:
        """직렬화 가능한 그룹 보정 파라미터를 반환합니다."""
        return {
            "calibrator_type": "GroupBiasShrinkageCalibrator",
            "scale": self.scale_,
            "bias": self.bias_,
            "shrinkage": self.shrinkage,
            "max_offset": self.max_offset,
            "n_groups": len(self.group_offsets_),
            "group_offsets": dict(self.group_offsets_),
            "group_counts": dict(self.group_counts_),
        }


class IsotonicCalibrator:
    """`50:50 WLL` 균형 가중치를 반영한 비모수 Isotonic Regression 보정기(비교 검증용)."""

    def __init__(self, eps: float = EPSILON) -> None:
        self.eps = float(eps)
        self._iso: Any = None
        self.is_fitted_: bool = False

    def fit(
        self,
        inputs: ArrayLike,
        y_true: ArrayLike,
        input_space: InputSpace = "logit",
    ) -> IsotonicCalibrator:
        """`50:50` 클래스 균형 가중치로 Isotonic Regression을 적합합니다."""
        from sklearn.isotonic import IsotonicRegression

        probs = (
            logit_to_prob(inputs, eps=self.eps)
            if input_space == "logit"
            else clip_probs(inputs, eps=self.eps)
        )
        y_arr = _extract_binary_targets(y_true, n_rows=len(probs))
        n_pos = int(np.sum(y_arr == 1))
        n_neg = len(y_arr) - n_pos
        sample_weight = np.where(y_arr == 1, 0.5 / n_pos, 0.5 / n_neg).astype(np.float64)

        iso = IsotonicRegression(y_min=self.eps, y_max=1.0 - self.eps, out_of_bounds="clip")
        iso.fit(probs, y_arr, sample_weight=sample_weight)
        self._iso = iso
        self.is_fitted_ = True
        return self

    def predict_proba(
        self,
        inputs: ArrayLike,
        input_space: InputSpace = "logit",
    ) -> np.ndarray:
        """Isotonic 보정 확률을 반환합니다."""
        if not self.is_fitted_ or self._iso is None:
            raise RuntimeError("predict_proba 호출 전에 fit()을 먼저 실행해야 합니다.")
        probs = (
            logit_to_prob(inputs, eps=self.eps)
            if input_space == "logit"
            else clip_probs(inputs, eps=self.eps)
        )
        return clip_probs(self._iso.transform(probs), eps=self.eps)

    def transform_logits(
        self,
        inputs: ArrayLike,
        input_space: InputSpace = "logit",
    ) -> np.ndarray:
        """Isotonic 보정 확률의 Log-odds 표현을 반환합니다."""
        return prob_to_logit(self.predict_proba(inputs, input_space=input_space), eps=self.eps)


@dataclass
class OOFCalibrationResult:
    """Out-of-Fold 확률 보정 결과 컨테이너."""

    oof_logits: np.ndarray
    oof_probs: np.ndarray
    fold_params: list[dict[str, Any]]
    mean_scale: float | None
    mean_bias: float | None


class OutOfFoldCalibrator:
    """`fold_id` 기준 엄격한 Out-of-Fold 확률 보정 래퍼.

    각 폴드 `k`에 대해 `fold_id != k` 행으로만 기본 보정기를 적합하고 `fold_id == k`를 변환하여
    인샘플 과적합을 원천 차단합니다. 테스트/추론 시에는 모든 폴드 보정기의 평균 예측(기본 Logit 평균)
    또는 등가 평균 파라미터를 일관되게 적용합니다.
    """

    def __init__(
        self,
        calibrator_factory: Callable[[], Any] | None = None,
        *,
        l2_reg: float = 1.0e-6,
        min_scale: float = 1.0e-3,
        max_scale: float = 20.0,
        fit_intercept: bool = True,
        input_space: InputSpace = "logit",
        test_agg: TestAggMode = "logit_mean",
        clip_logit: float = DEFAULT_CLIP_LOGIT,
        eps: float = EPSILON,
    ) -> None:
        if test_agg not in ("logit_mean", "prob_mean"):
            raise ValueError(f"test_agg는 'logit_mean' 또는 'prob_mean'이어야 합니다: {test_agg!r}")

        self.calibrator_factory = calibrator_factory
        self.l2_reg = float(l2_reg)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.fit_intercept = bool(fit_intercept)
        self.input_space: InputSpace = input_space
        self.test_agg: TestAggMode = test_agg
        self.clip_logit = float(clip_logit)
        self.eps = float(eps)

        self.fold_calibrators_: dict[int, Any] = {}
        self.fold_params_: list[dict[str, Any]] = []
        self.oof_logits_: np.ndarray | None = None
        self.oof_probs_: np.ndarray | None = None
        self.mean_scale_: float | None = None
        self.mean_bias_: float | None = None
        self.is_fitted_: bool = False

    def _build_base_calibrator(self) -> Any:
        if self.calibrator_factory is not None:
            return self.calibrator_factory()
        return AffineLogitCalibrator(
            l2_reg=self.l2_reg,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            fit_intercept=self.fit_intercept,
            input_space="logit",
            clip_logit=self.clip_logit,
            eps=self.eps,
        )

    def _fit_single(
        self,
        calibrator: Any,
        logits: np.ndarray,
        y_arr: np.ndarray,
        groups: np.ndarray | None,
    ) -> Any:
        if groups is not None and isinstance(calibrator, GroupBiasShrinkageCalibrator):
            return calibrator.fit(logits, y_arr, groups=groups, input_space="logit")
        if groups is not None:
            try:
                return calibrator.fit(logits, y_arr, groups=groups, input_space="logit")
            except TypeError:
                pass
        return calibrator.fit(logits, y_arr, input_space="logit")

    def _transform_single_logits(
        self,
        calibrator: Any,
        logits: np.ndarray,
        groups: np.ndarray | None,
    ) -> np.ndarray:
        if groups is not None and isinstance(calibrator, GroupBiasShrinkageCalibrator):
            return calibrator.transform_logits(logits, groups=groups, input_space="logit")
        if groups is not None:
            try:
                return calibrator.transform_logits(logits, groups=groups, input_space="logit")
            except TypeError:
                pass
        if hasattr(calibrator, "transform_logits"):
            return calibrator.transform_logits(logits, input_space="logit")
        return prob_to_logit(calibrator.predict_proba(logits, input_space="logit"), eps=self.eps)

    def fit_transform_result(
        self,
        inputs: ArrayLike,
        y_true: ArrayLike,
        folds: pd.DataFrame | ArrayLike,
        groups: ArrayLike | pd.Series | None = None,
        input_space: InputSpace | None = None,
    ) -> OOFCalibrationResult:
        """폴드 격리 Out-of-Fold 보정을 적합하고 전체 결과 컨테이너를 반환합니다."""
        space = input_space or self.input_space
        raw_logits = _coerce_1d_logits(inputs, input_space=space, eps=self.eps)
        n_rows = len(raw_logits)
        y_arr = _extract_binary_targets(y_true, n_rows=n_rows)
        fold_arr = _extract_fold_ids(folds, n_rows=n_rows)
        norm_groups = _normalize_group_keys(groups, n_rows=n_rows) if groups is not None else None

        unique_folds = sorted(int(f) for f in np.unique(fold_arr))
        if len(unique_folds) < 2:
            raise ValueError(
                f"Out-of-Fold 보정을 위해서는 최소 2개 이상의 폴드가 필요합니다: {unique_folds}"
            )

        oof_logits = np.empty(n_rows, dtype=np.float64)
        self.fold_calibrators_ = {}
        self.fold_params_ = []
        scales: list[float] = []
        biases: list[float] = []

        for k in unique_folds:
            va_mask = fold_arr == k
            tr_mask = ~va_mask
            if not np.any(tr_mask) or not np.any(va_mask):
                raise ValueError(f"폴드 {k}의 학습 또는 검증 데이터가 비어 있습니다.")

            cal = self._build_base_calibrator()
            tr_groups = norm_groups[tr_mask] if norm_groups is not None else None
            va_groups = norm_groups[va_mask] if norm_groups is not None else None

            self._fit_single(cal, raw_logits[tr_mask], y_arr[tr_mask], tr_groups)
            oof_logits[va_mask] = self._transform_single_logits(cal, raw_logits[va_mask], va_groups)

            self.fold_calibrators_[k] = cal
            param_rec: dict[str, Any] = {
                "fold_id": k,
                "n_train": int(np.sum(tr_mask)),
                "n_valid": int(np.sum(va_mask)),
            }
            if hasattr(cal, "scale_") and hasattr(cal, "bias_"):
                s_val = float(cal.scale_)
                b_val = float(cal.bias_)
                param_rec["scale"] = s_val
                param_rec["bias"] = b_val
                scales.append(s_val)
                biases.append(b_val)
            self.fold_params_.append(param_rec)

        oof_probs = logit_to_prob(oof_logits, clip_logit=self.clip_logit, eps=self.eps)
        self.oof_logits_ = oof_logits
        self.oof_probs_ = oof_probs
        self.mean_scale_ = float(np.mean(scales)) if scales else None
        self.mean_bias_ = float(np.mean(biases)) if biases else None
        self.is_fitted_ = True

        return OOFCalibrationResult(
            oof_logits=oof_logits,
            oof_probs=oof_probs,
            fold_params=list(self.fold_params_),
            mean_scale=self.mean_scale_,
            mean_bias=self.mean_bias_,
        )

    def fit(
        self,
        inputs: ArrayLike,
        y_true: ArrayLike,
        folds: pd.DataFrame | ArrayLike,
        groups: ArrayLike | pd.Series | None = None,
        input_space: InputSpace | None = None,
    ) -> OutOfFoldCalibrator:
        """폴드 격리 Out-of-Fold 보정 파라미터를 학습합니다."""
        self.fit_transform_result(
            inputs=inputs,
            y_true=y_true,
            folds=folds,
            groups=groups,
            input_space=input_space,
        )
        return self

    def fit_transform(
        self,
        inputs: ArrayLike,
        y_true: ArrayLike,
        folds: pd.DataFrame | ArrayLike,
        groups: ArrayLike | pd.Series | None = None,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """Out-of-Fold 보정을 적합하고 보정된 OOF 확률을 반환합니다."""
        res = self.fit_transform_result(
            inputs=inputs,
            y_true=y_true,
            folds=folds,
            groups=groups,
            input_space=input_space,
        )
        return res.oof_probs

    def transform_logits(
        self,
        inputs: ArrayLike,
        groups: ArrayLike | pd.Series | None = None,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """학습된 모든 폴드 보정기를 평균 결합해 테스트/신규 데이터의 보정 로짓을 계산합니다."""
        if not self.is_fitted_ or not self.fold_calibrators_:
            raise RuntimeError("transform_logits 호출 전에 fit()을 먼저 실행해야 합니다.")

        space = input_space or self.input_space
        raw_logits = _coerce_1d_logits(inputs, input_space=space, eps=self.eps)
        norm_groups = _normalize_group_keys(groups, n_rows=len(raw_logits)) if groups is not None else None

        n_folds = float(len(self.fold_calibrators_))
        if self.test_agg == "logit_mean":
            acc = np.zeros(len(raw_logits), dtype=np.float64)
            for cal in self.fold_calibrators_.values():
                acc += self._transform_single_logits(cal, raw_logits, norm_groups)
            return acc / n_folds

        prob_acc = np.zeros(len(raw_logits), dtype=np.float64)
        for cal in self.fold_calibrators_.values():
            cal_logits = self._transform_single_logits(cal, raw_logits, norm_groups)
            prob_acc += logit_to_prob(cal_logits, clip_logit=self.clip_logit, eps=self.eps)
        return prob_to_logit(prob_acc / n_folds, eps=self.eps)

    def predict_proba(
        self,
        inputs: ArrayLike,
        groups: ArrayLike | pd.Series | None = None,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """학습된 폴드 보정기 평균을 적용해 보정된 테스트/신규 예측 확률을 반환합니다."""
        cal_logits = self.transform_logits(inputs, groups=groups, input_space=input_space)
        return logit_to_prob(cal_logits, clip_logit=self.clip_logit, eps=self.eps)

    def transform(
        self,
        inputs: ArrayLike,
        groups: ArrayLike | pd.Series | None = None,
        input_space: InputSpace | None = None,
    ) -> np.ndarray:
        """`predict_proba`의 별칭입니다."""
        return self.predict_proba(inputs, groups=groups, input_space=input_space)

    def get_inference_calibrator(self) -> AffineLogitCalibrator:
        """실시간 서빙(`RealtimeCTRPredictor`)용 단일 평균 `AffineLogitCalibrator`를 반환합니다."""
        if not self.is_fitted_:
            raise RuntimeError("get_inference_calibrator 호출 전에 fit()을 먼저 실행해야 합니다.")
        if self.mean_scale_ is None or self.mean_bias_ is None:
            raise ValueError(
                "현재 기본 보정기는 scale_/bias_ 속성을 제공하지 않아 단일 AffineLogitCalibrator로 병합할 수 없습니다."
            )
        return AffineLogitCalibrator.from_dict(
            {
                "scale": self.mean_scale_,
                "bias": self.mean_bias_,
                "l2_reg": self.l2_reg,
                "min_scale": self.min_scale,
                "max_scale": self.max_scale,
                "fit_intercept": self.fit_intercept,
                "input_space": "logit",
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """OOF 보정 요약 및 평균 추론 파라미터를 직렬화 가능한 딕셔너리로 반환합니다."""
        return {
            "calibrator_type": "OutOfFoldCalibrator",
            "test_agg": self.test_agg,
            "mean_scale": self.mean_scale_,
            "mean_bias": self.mean_bias_,
            "fold_params": list(self.fold_params_),
        }
