"""토스 NEXT ML Challenge 공식 평가 지표(AP, 50:50 WLL) 및 확률 보정 모듈."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score

EPSILON: float = 1e-15


@dataclass(frozen=True)
class EvalResult:
    """평가 지표 계산 결과 컨테이너."""

    ap: float
    wll: float
    score: float
    pos_logloss: float
    neg_logloss: float

    def to_dict(self) -> dict[str, float]:
        """직렬화 가능한 딕셔너리로 변환합니다."""
        return {
            "ap": self.ap,
            "wll": self.wll,
            "score": self.score,
            "pos_logloss": self.pos_logloss,
            "neg_logloss": self.neg_logloss,
        }


def clip_probs(y_pred: ArrayLike, eps: float = EPSILON) -> np.ndarray:
    """수치 안정성을 위해 예측 확률을 [eps, 1 - eps] 구간으로 클리핑합니다."""
    arr = np.asarray(y_pred, dtype=np.float64)
    return np.clip(arr, eps, 1.0 - eps)


def weighted_logloss_50_50(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    eps: float = EPSILON,
) -> tuple[float, float, float]:
    """clicked=0과 clicked=1의 클래스 기여를 50:50으로 맞춘 가중 LogLoss를 계산합니다.

    Returns:
        (wll, pos_logloss, neg_logloss) 튜플.
    """
    y = np.asarray(y_true, dtype=np.int8)
    p = clip_probs(y_pred, eps=eps)

    pos_mask = y == 1
    neg_mask = y == 0
    if not np.any(pos_mask) or not np.any(neg_mask):
        raise ValueError("y_true에는 양성(1)과 음성(0) 샘플이 모두 포함되어야 합니다.")

    pos_ll = float(-np.mean(np.log(p[pos_mask])))
    neg_ll = float(-np.mean(np.log(1.0 - p[neg_mask])))
    wll = 0.5 * pos_ll + 0.5 * neg_ll
    return wll, pos_ll, neg_ll


def evaluate_predictions(y_true: ArrayLike, y_pred: ArrayLike) -> EvalResult:
    """공식 리더보드 지표(AP 50% + WLL 50%)를 일괄 계산합니다.

    결합 점수 산식:
        Score = 0.5 * AP + 0.5 * (1 / (1 + WLL))
    """
    y = np.asarray(y_true, dtype=np.int8)
    p = clip_probs(y_pred)

    ap = float(average_precision_score(y, p))
    wll, pos_ll, neg_ll = weighted_logloss_50_50(y, p)
    score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))
    return EvalResult(
        ap=ap,
        wll=wll,
        score=score,
        pos_logloss=pos_ll,
        neg_logloss=neg_ll,
    )


def shift_prob_prior(
    y_pred: ArrayLike,
    train_pos_rate: float,
    target_pos_rate: float = 0.5,
    eps: float = EPSILON,
) -> np.ndarray:
    """예측 확률의 사전 분포(Prior)를 Log-odds 공간에서 목표 양성 비율로 변환합니다.

    예를 들어 원천 불균형 데이터(train_pos_rate = p0)에서 학습된 모델이나
    1:k Negative Downsampling(train_pos_rate = 1/(1+k))에서 학습된 모델의 확률을
    50:50 WLL 최적화를 위한 균형 사전 확률(target_pos_rate = 0.5) 또는
    실제 사전 확률로 단조 변환할 때 사용합니다. (단조 증가 변환이므로 AP는 불변)
    """
    if not (0.0 < train_pos_rate < 1.0) or not (0.0 < target_pos_rate < 1.0):
        raise ValueError("사전 확률(pos_rate)은 (0, 1) 개구간 내에 있어야 합니다.")

    p = clip_probs(y_pred, eps=eps)
    logit = np.log(p / (1.0 - p))
    train_logit = np.log(train_pos_rate / (1.0 - train_pos_rate))
    target_logit = np.log(target_pos_rate / (1.0 - target_pos_rate))

    shifted_logit = logit - train_logit + target_logit
    return 1.0 / (1.0 + np.exp(-np.clip(shifted_logit, -35.0, 35.0)))
