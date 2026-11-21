"""제출 파일 생성, sample_submission 정합성 검증, YAML 오버라이드 및 실시간 추론 서빙 모듈."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike

from toss_ctr.evaluation.metrics import EPSILON, clip_probs
from toss_ctr.inference.blender import _id_arrays_equal
from toss_ctr.models.calibration import (
    AffineLogitCalibrator,
    GroupBiasShrinkageCalibrator,
    OutOfFoldCalibrator,
    logit_to_prob,
)


def _parse_override_value(raw_val: str) -> Any:
    """CLI `--override key=val` 문자열 값을 경로 안전성을 유지하며 Python 타입으로 변환합니다."""
    stripped = raw_val.strip()
    lower = stripped.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none", "~"):
        return None

    # YAML 날짜 자동 변환이나 Windows/POSIX 경로 왜곡을 방지하기 위해 숫자/리스트/딕셔너리만 파싱
    if (stripped.startswith("[") and stripped.endswith("]")) or (
        stripped.startswith("{") and stripped.endswith("}")
    ):
        try:
            return yaml.safe_load(stripped)
        except yaml.YAMLError:
            return stripped

    try:
        if stripped.startswith(("+", "-")) and stripped[1:].isdigit():
            return int(stripped)
        if stripped.isdigit():
            return int(stripped)
    except ValueError:
        pass

    try:
        # float('nan') / float('inf') 문자열이 의도치 않게 숫자로 바뀌지 않도록 방어
        if lower not in ("nan", "inf", "+inf", "-inf", "infinity"):
            return float(stripped)
    except ValueError:
        pass

    return stripped


def load_yaml_with_overrides(
    config_path: str | Path,
    overrides: Sequence[str] | None = None,
) -> dict[str, Any]:
    """YAML 설정 파일을 로드하고 `a.b=val` 형식의 CLI 오버라이드를 중첩 딕셔너리에 적용합니다."""
    cfg_p = Path(config_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {cfg_p}")

    with cfg_p.open("r", encoding="utf-8") as fp:
        loaded = yaml.safe_load(fp) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML 루트 구조는 매핑(dict)이어야 합니다: {cfg_p}")

    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"--override 인자는 'key.path=value' 형식이어야 합니다: {item!r}")
        raw_key, raw_val = item.split("=", 1)
        keys = [k.strip() for k in raw_key.split(".") if k.strip()]
        if not keys:
            raise ValueError(f"유효하지 않은 override 키 경로입니다: {item!r}")

        cursor: dict[str, Any] = loaded
        for part in keys[:-1]:
            nxt = cursor.get(part)
            if nxt is None:
                nxt = {}
                cursor[part] = nxt
            elif not isinstance(nxt, dict):
                raise ValueError(
                    f"override 경로 충돌('{part}'는 딕셔너리가 아님): {raw_key!r}"
                )
            cursor = nxt
        cursor[keys[-1]] = _parse_override_value(raw_val)

    return loaded


def generate_submission(
    test_ids: ArrayLike,
    pred_probs: ArrayLike,
    output_path: str | Path | None = None,
    id_col: str = "ID",
    target_col: str = "clicked",
    eps: float = EPSILON,
) -> pd.DataFrame:
    """표준 제출 DataFrame(`ID`, `clicked`)을 생성하고 선택적으로 CSV 파일로 저장합니다."""
    ids_arr = np.asarray(test_ids)
    probs_raw = np.asarray(pred_probs, dtype=np.float64)

    if ids_arr.ndim != 1 or probs_raw.ndim != 1:
        raise ValueError(
            f"test_ids({ids_arr.shape})와 pred_probs({probs_raw.shape})는 1차원 배열이어야 합니다."
        )
    if len(ids_arr) != len(probs_raw):
        raise ValueError(
            f"test_ids({len(ids_arr)})와 pred_probs({len(probs_raw)})의 길이가 일치하지 않습니다."
        )
    if len(ids_arr) == 0:
        raise ValueError("빈 예측 배열로 제출 파일을 생성할 수 없습니다.")
    if pd.isna(ids_arr).any():
        raise ValueError("test_ids에 결측치가 포함되어 있습니다.")
    if not np.all(np.isfinite(probs_raw)):
        raise ValueError("pred_probs에 NaN 또는 Inf가 포함되어 있어 제출 파일을 생성할 수 없습니다.")

    probs_clipped = clip_probs(probs_raw, eps=eps)
    sub_df = pd.DataFrame({id_col: ids_arr, target_col: probs_clipped})

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sub_df.to_csv(out_p, index=False)

    return sub_df


def verify_submission_df(
    sub_df: pd.DataFrame,
    sample_sub_df: pd.DataFrame | None = None,
    expected_rows: int | None = None,
    id_col: str = "ID",
    target_col: str = "clicked",
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """제출 DataFrame의 스키마, 결측/비유한, 확률 범위, ID 순서, 분포 통계를 일괄 검증합니다."""
    errors: list[str] = []
    warnings: list[str] = []

    expected_cols = [id_col, target_col]
    actual_cols = list(sub_df.columns)
    if actual_cols != expected_cols:
        errors.append(f"컬럼 불일치: expected={expected_cols}, actual={actual_cols}")

    n_rows = int(len(sub_df))
    if n_rows == 0:
        errors.append("제출 DataFrame의 행 수가 0건입니다.")
    if expected_rows is not None and n_rows != int(expected_rows):
        errors.append(f"행 수 불일치: expected={expected_rows}, actual={n_rows}")

    if id_col in sub_df.columns:
        id_nulls = int(sub_df[id_col].isna().sum())
        if id_nulls > 0:
            errors.append(f"'{id_col}' 컬럼에 결측치 {id_nulls}건이 존재합니다.")
        if sample_sub_df is not None:
            if id_col not in sample_sub_df.columns:
                errors.append(f"sample_submission에 '{id_col}' 컬럼이 없습니다.")
            else:
                if len(sample_sub_df) != n_rows:
                    errors.append(
                        f"sample_submission 행 수({len(sample_sub_df)})와 제출 행 수({n_rows})가 다릅니다."
                    )
                elif not _id_arrays_equal(
                    sub_df[id_col].to_numpy(),
                    sample_sub_df[id_col].to_numpy(),
                ):
                    errors.append(f"'{id_col}' 순서 또는 값이 sample_submission과 100% 일치하지 않습니다.")

    stats: dict[str, Any] = {}
    if target_col in sub_df.columns and n_rows > 0:
        col_s = sub_df[target_col]
        if pd.api.types.is_bool_dtype(col_s.dtype) or not pd.api.types.is_numeric_dtype(col_s.dtype):
            errors.append(f"'{target_col}' 컬럼은 숫자형이어야 합니다: dtype={col_s.dtype}")
        else:
            vals = col_s.to_numpy(dtype=np.float64, na_value=np.nan)
            n_nan = int(np.isnan(vals).sum())
            n_inf = int(np.isinf(vals).sum())
            finite_mask = np.isfinite(vals)
            finite_vals = vals[finite_mask]

            if n_nan > 0:
                errors.append(f"'{target_col}'에 NaN {n_nan}건이 존재합니다.")
            if n_inf > 0:
                errors.append(f"'{target_col}'에 Inf {n_inf}건이 존재합니다.")

            if finite_vals.size > 0:
                out_of_range_mask = (finite_vals <= 0.0) | (finite_vals >= 1.0)
                n_out_of_range = int(np.sum(out_of_range_mask))
                if n_out_of_range > 0:
                    errors.append(
                        f"'{target_col}'에 (0, 1) 개구간 이탈 값 {n_out_of_range}건이 존재합니다 "
                        f"(min={float(np.min(finite_vals)):.6g}, max={float(np.max(finite_vals)):.6g})."
                    )

                quantiles = np.quantile(finite_vals, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
                mean_val = float(np.mean(finite_vals))
                std_val = float(np.std(finite_vals))
                stats = {
                    "n_nan": n_nan,
                    "n_inf": n_inf,
                    "n_out_of_range": n_out_of_range,
                    "min": float(np.min(finite_vals)),
                    "p01": float(quantiles[0]),
                    "p05": float(quantiles[1]),
                    "p25": float(quantiles[2]),
                    "p50": float(quantiles[3]),
                    "p75": float(quantiles[4]),
                    "p95": float(quantiles[5]),
                    "p99": float(quantiles[6]),
                    "max": float(np.max(finite_vals)),
                    "mean": mean_val,
                    "std": std_val,
                    "positive_rate_gt_05": float(np.mean(finite_vals > 0.5)),
                }

                if std_val < 1.0e-9:
                    warnings.append("예측 확률 분산이 거의 0입니다(상수 예측 의심).")
                if mean_val < 1.0e-4 or mean_val > 0.99:
                    warnings.append(f"평균 예측 확률({mean_val:.6f})이 비정상 극단 구간입니다.")
            else:
                stats = {"n_nan": n_nan, "n_inf": n_inf, "n_out_of_range": n_rows}

    report: dict[str, Any] = {
        "valid": len(errors) == 0,
        "rows": n_rows,
        "columns": actual_cols,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }

    if raise_on_error and errors:
        raise ValueError("제출 파일 무결성 검증 실패: " + " | ".join(errors))

    return report


def verify_submission_file(
    submission_path: str | Path,
    sample_path: str | Path | None = None,
    expected_rows: int | None = None,
    id_col: str = "ID",
    target_col: str = "clicked",
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """디스크에 저장된 제출 CSV/Parquet 파일과 `sample_submission.csv` 간 무결성을 검증합니다."""
    sub_p = Path(submission_path)
    if not sub_p.exists():
        raise FileNotFoundError(f"제출 파일을 찾을 수 없습니다: {sub_p}")

    if sub_p.suffix.lower() == ".parquet":
        sub_df = pd.read_parquet(sub_p)
    else:
        sub_df = pd.read_csv(sub_p, dtype={id_col: str})

    sample_df: pd.DataFrame | None = None
    if sample_path is not None:
        samp_p = Path(sample_path)
        if not samp_p.exists():
            raise FileNotFoundError(f"sample_submission 파일을 찾을 수 없습니다: {samp_p}")
        if samp_p.suffix.lower() == ".parquet":
            sample_df = pd.read_parquet(samp_p, columns=[id_col])
        else:
            sample_df = pd.read_csv(samp_p, usecols=[id_col], dtype={id_col: str})

    if sub_df.shape[1] >= 1 and id_col in sub_df.columns and sample_df is not None:
        # CSV/Parquet 혼합 비교 시 ID를 문자열로 정규화해 타입 불일치 오탐 방지
        sub_df = sub_df.copy()
        sub_df[id_col] = sub_df[id_col].astype(str)
        sample_df = sample_df.copy()
        sample_df[id_col] = sample_df[id_col].astype(str)

    report = verify_submission_df(
        sub_df=sub_df,
        sample_sub_df=sample_df,
        expected_rows=expected_rows,
        id_col=id_col,
        target_col=target_col,
        raise_on_error=raise_on_error,
    )
    report["submission_path"] = str(sub_p)
    report["sample_path"] = str(sample_path) if sample_path is not None else None
    return report


class OfflineFeatureStore:
    """추론 주기 분리 아키텍처를 위한 오프라인 배치 피처 스토어(O(1) 키 조회).

    배치/니어리얼타임 주기로 사전 집계된 지면(`inventory_id`), 광고셋(`l_feat_14`),
    유저 시퀀스 요약(`seq_*`) 또는 타겟/빈도 통계를 메모리에 적재해 실시간 요청에 결합합니다.
    """

    def __init__(self) -> None:
        self._tables: dict[str, dict[Any, dict[str, Any]]] = {}
        self._table_cols: dict[str, list[str]] = {}
        self._defaults: dict[str, dict[str, Any]] = {}

    def register_table(
        self,
        key_col: str,
        feature_df: pd.DataFrame,
        default_values: Mapping[str, Any] | None = None,
    ) -> OfflineFeatureStore:
        """오프라인 집계 DataFrame을 `key_col` 기준 O(1) 조회 테이블로 등록합니다."""
        if key_col not in feature_df.columns:
            raise KeyError(f"feature_df에 키 컬럼 '{key_col}'이 없습니다.")

        feat_cols = [c for c in feature_df.columns if c != key_col]
        dedup_df = feature_df.drop_duplicates(subset=[key_col], keep="last")

        lookup: dict[Any, dict[str, Any]] = {}
        keys = dedup_df[key_col].tolist()
        records = dedup_df[feat_cols].to_dict(orient="records")
        for key_val, rec in zip(keys, records):
            norm_key = str(key_val) if not pd.isna(key_val) else "__MISSING__"
            lookup[norm_key] = rec

        self._tables[key_col] = lookup
        self._table_cols[key_col] = feat_cols
        self._defaults[key_col] = dict(default_values or {})
        return self

    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """실시간 요청 DataFrame의 행 순서를 유지하면서 등록된 오프라인 피처를 결합합니다."""
        if not self._tables:
            return df.copy()

        work = df.copy()
        for key_col, lookup in self._tables.items():
            if key_col not in work.columns:
                continue
            feat_cols = self._table_cols[key_col]
            defaults = self._defaults[key_col]
            raw_keys = work[key_col].tolist()

            col_buffers: dict[str, list[Any]] = {c: [] for c in feat_cols}
            for raw_k in raw_keys:
                norm_k = str(raw_k) if not pd.isna(raw_k) else "__MISSING__"
                hit = lookup.get(norm_k)
                for col in feat_cols:
                    if hit is not None:
                        col_buffers[col].append(hit[col])
                    else:
                        col_buffers[col].append(defaults.get(col, np.nan))

            for col in feat_cols:
                work[col] = col_buffers[col]

        return work


class RealtimeCTRPredictor:
    """오프라인 집계 피처 스토어 + 온라인 컨텍스트 결합 + 이종 모델 Log-odds 블렌딩 + 50:50 WLL 보정 서빙기.

    `GEMINI.md`의 "최종 모델은 실시간 서빙이 가능해야 하지만, 추론 주기가 서로 다른 여러 개의 모델이
    존재하는 것도 가능" 요건에 맞춰 동기 모델 예측기(`model_predictors`)와
    비동기/배치 캐시 예측(`precomputed_logits`)을 동시에 결합할 수 있습니다.
    """

    def __init__(
        self,
        model_predictors: Mapping[str, Callable[[pd.DataFrame], ArrayLike]],
        weights: Mapping[str, float] | None = None,
        calibrator: AffineLogitCalibrator
        | OutOfFoldCalibrator
        | GroupBiasShrinkageCalibrator
        | None = None,
        offline_store: OfflineFeatureStore | None = None,
        online_feature_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        group_col: str | None = None,
        model_cadences: Mapping[str, str] | None = None,
        id_col: str = "ID",
    ) -> None:
        if not model_predictors and not weights:
            raise ValueError("최소 1개 이상의 model_predictors 또는 weights가 필요합니다.")

        self.model_predictors = dict(model_predictors)
        model_names = list(self.model_predictors.keys()) if self.model_predictors else list(weights.keys())  # type: ignore[union-attr]

        if weights is None:
            raw_w = {name: 1.0 / float(len(model_names)) for name in model_names}
        else:
            raw_w = {str(k): max(float(v), 0.0) for k, v in weights.items()}
            total_w = sum(raw_w.values())
            if total_w <= 1.0e-12:
                raise ValueError("블렌딩 가중치 합은 0보다 커야 합니다.")
            raw_w = {k: v / total_w for k, v in raw_w.items()}

        self.weights = raw_w
        self.model_names = list(raw_w.keys())
        self.calibrator = calibrator
        self.offline_store = offline_store
        self.online_feature_fn = online_feature_fn
        self.group_col = group_col
        self.model_cadences = dict(
            model_cadences
            or {name: "realtime_sync" for name in self.model_names}
        )
        self.id_col = id_col

    def _apply_calibrator(
        self,
        blended_logits: np.ndarray,
        features_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.calibrator is None:
            return blended_logits, logit_to_prob(blended_logits)

        groups_arg = (
            features_df[self.group_col]
            if self.group_col is not None and self.group_col in features_df.columns
            else None
        )
        if isinstance(self.calibrator, GroupBiasShrinkageCalibrator):
            if groups_arg is None:
                raise KeyError(
                    f"GroupBiasShrinkageCalibrator 적용을 위한 그룹 컬럼 '{self.group_col}'이 없습니다."
                )
            cal_logits = self.calibrator.transform_logits(
                blended_logits, groups=groups_arg, input_space="logit"
            )
            cal_probs = logit_to_prob(cal_logits)
            return cal_logits, cal_probs

        if isinstance(self.calibrator, OutOfFoldCalibrator):
            cal_logits = self.calibrator.transform_logits(
                blended_logits, groups=groups_arg, input_space="logit"
            )
            cal_probs = logit_to_prob(cal_logits)
            return cal_logits, cal_probs

        cal_logits = self.calibrator.transform_logits(blended_logits, input_space="logit")
        cal_probs = logit_to_prob(cal_logits)
        return cal_logits, cal_probs

    def predict_batch(
        self,
        request_df: pd.DataFrame,
        precomputed_logits: Mapping[str, ArrayLike] | None = None,
    ) -> pd.DataFrame:
        """실시간 미니배치 요청을 오프라인/온라인 피처 결합 후 블렌딩·보정 추론합니다."""
        if len(request_df) == 0:
            raise ValueError("빈 request_df는 추론할 수 없습니다.")

        work = (
            self.offline_store.enrich_dataframe(request_df)
            if self.offline_store is not None
            else request_df.copy()
        )
        if self.online_feature_fn is not None:
            work = self.online_feature_fn(work)

        n_rows = len(work)
        precomputed = dict(precomputed_logits or {})
        blended_logits = np.zeros(n_rows, dtype=np.float64)
        model_logit_cols: dict[str, np.ndarray] = {}

        for name in self.model_names:
            w = self.weights[name]
            if name in precomputed:
                m_logits = np.asarray(precomputed[name], dtype=np.float64)
            elif name in self.model_predictors:
                m_logits = np.asarray(self.model_predictors[name](work), dtype=np.float64)
            else:
                raise KeyError(
                    f"모델 '{name}'에 대한 model_predictors 또는 precomputed_logits가 없습니다."
                )

            if m_logits.shape != (n_rows,):
                raise ValueError(
                    f"모델 '{name}' 출력 shape 오류: expected=({n_rows},), actual={m_logits.shape}"
                )
            if not np.all(np.isfinite(m_logits)):
                raise ValueError(f"모델 '{name}' 출력 로짓에 NaN/Inf가 포함되어 있습니다.")

            model_logit_cols[f"logit_{name}"] = m_logits
            blended_logits += w * m_logits

        cal_logits, cal_probs = self._apply_calibrator(blended_logits, work)

        out_data: dict[str, Any] = {}
        if self.id_col in request_df.columns:
            out_data[self.id_col] = request_df[self.id_col].to_numpy()
        out_data["raw_logit"] = blended_logits
        out_data["calibrated_logit"] = cal_logits
        out_data["clicked"] = cal_probs
        out_data.update(model_logit_cols)

        return pd.DataFrame(out_data, index=request_df.index)

    def predict_one(
        self,
        record: Mapping[str, Any],
        precomputed_logits: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """단건 실시간 광고 노출 요청의 클릭 확률과 모델별 로짓, 지연시간(ms)을 반환합니다."""
        t0 = time.perf_counter()
        req_df = pd.DataFrame([dict(record)])
        pre_map = (
            {k: np.array([float(v)], dtype=np.float64) for k, v in precomputed_logits.items()}
            if precomputed_logits is not None
            else None
        )
        res_df = self.predict_batch(req_df, precomputed_logits=pre_map)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        row = res_df.iloc[0]
        return {
            "clicked": float(row["clicked"]),
            "raw_logit": float(row["raw_logit"]),
            "calibrated_logit": float(row["calibrated_logit"]),
            "model_logits": {
                name: float(row[f"logit_{name}"]) for name in self.model_names
            },
            "model_cadences": dict(self.model_cadences),
            "latency_ms": round(latency_ms, 4),
        }
