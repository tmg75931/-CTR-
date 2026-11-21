"""이종 모델 OOF Log-odds 블렌딩, 50:50 WLL OOF 보정, 최종 제출 생성 및 무결성 검증 CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from toss_ctr.evaluation.metrics import evaluate_predictions  # noqa: E402
from toss_ctr.inference.blender import (  # noqa: E402
    OOFLogitBlender,
    _id_arrays_equal,
    load_prediction_artifacts,
)
from toss_ctr.inference.submission import (  # noqa: E402
    generate_submission,
    load_yaml_with_overrides,
    verify_submission_df,
)
from toss_ctr.models.calibration import (  # noqa: E402
    GroupBiasShrinkageCalibrator,
    OutOfFoldCalibrator,
)


def _flatten_overrides(raw_groups: list[list[str]] | None) -> list[str]:
    return [item for grp in (raw_groups or []) for item in grp]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="다중 모델 OOF Log-odds 블렌딩 + 50:50 WLL OOF 보정 + submission.csv 생성/검증"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "ensemble_wll50.yaml"),
        help="앙상블/보정 YAML 설정 파일 경로 (기본: configs/ensemble_wll50.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=None,
        help="YAML 설정 오버라이드 (예: --override paths.preds_dir=/tmp/preds submission.expected_rows=60)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """앙상블 블렌딩, OOF 확률 보정 및 제출 무결성 검증 파이프라인을 실행합니다."""
    args = parse_args(argv)
    cfg = load_yaml_with_overrides(args.config, _flatten_overrides(args.override))

    paths_cfg = cfg["paths"]
    models_cfg = cfg.get("models", {})
    blend_cfg = cfg.get("blender", {})
    cal_cfg = cfg.get("calibration", {})
    sub_cfg = cfg.get("submission", {})

    id_col = str(sub_cfg.get("id_col", "ID"))
    target_col = str(sub_cfg.get("target_col", "clicked"))

    require_test = bool(models_cfg.get("require_test_preds", True))
    auto_select = bool(models_cfg.get("auto_select_existing", True))
    candidates = models_cfg.get("candidates")
    preds_dir = Path(paths_cfg["preds_dir"])

    sample_path_raw = paths_cfg.get("sample_submission")
    sample_df: pd.DataFrame | None = None
    reference_ids: np.ndarray | None = None
    if sample_path_raw and Path(sample_path_raw).exists():
        sample_df = pd.read_csv(sample_path_raw, usecols=[id_col], dtype={id_col: str})
        # test_*.parquet가 정수형 ID를 사용하는 경우 sample ID 타입을 정합화
        probe_names = list(candidates) if candidates else [
            p.stem[len("test_") :] for p in sorted(preds_dir.glob("test_*.parquet"))
        ]
        for m_name in probe_names:
            test_p = preds_dir / f"test_{m_name}.parquet"
            if test_p.exists():
                pf_test = pq.ParquetFile(test_p)
                if id_col in pf_test.schema_arrow.names and pa.types.is_integer(
                    pf_test.schema_arrow.field(id_col).type
                ):
                    sample_df[id_col] = sample_df[id_col].astype(np.int64)
                break
        reference_ids = sample_df[id_col].to_numpy()

    oof_dfs, test_dfs = load_prediction_artifacts(
        preds_dir=preds_dir,
        model_names=candidates,
        require_test=require_test,
        skip_missing=auto_select,
        reference_test_ids=reference_ids,
    )

    blender = OOFLogitBlender(
        objective=str(blend_cfg.get("objective", "combined_score")),  # type: ignore[arg-type]
        blend_space=str(blend_cfg.get("blend_space", "logit")),  # type: ignore[arg-type]
        opt_max_samples=blend_cfg.get("opt_max_samples", 500_000),
        n_restarts=int(blend_cfg.get("n_restarts", 24)),
        max_iter=int(blend_cfg.get("max_iter", 120)),
        test_agg=str(blend_cfg.get("test_agg", "logit_mean")),  # type: ignore[arg-type]
        seed=int(cfg.get("seed", 42)),
    )
    blend_res = blender.fit_blend(
        oof_dfs=oof_dfs,
        test_dfs=test_dfs if test_dfs else None,
        reference_test_ids=reference_ids,
    )

    y_all = blend_res.oof_df["clicked"].to_numpy(dtype=np.int8)
    folds_all = blend_res.oof_df["fold_id"].to_numpy(dtype=np.int64)
    blended_oof_logits = blend_res.oof_df["raw_logit"].to_numpy(dtype=np.float64)
    pre_cal_metrics = blend_res.oof_metrics

    # 기본 50:50 WLL OOF Affine Logit Calibration (a > 0, AP 100% 보존)
    affine_oof_cal = OutOfFoldCalibrator(
        l2_reg=float(cal_cfg.get("l2_reg", 1.0e-6)),
        min_scale=float(cal_cfg.get("min_scale", 1.0e-3)),
        max_scale=float(cal_cfg.get("max_scale", 20.0)),
        fit_intercept=bool(cal_cfg.get("fit_intercept", True)),
        test_agg=str(cal_cfg.get("test_agg", "logit_mean")),  # type: ignore[arg-type]
    )
    affine_oof_probs = affine_oof_cal.fit_transform(
        blended_oof_logits, y_all, folds_all, input_space="logit"
    )
    affine_metrics = evaluate_predictions(y_all, affine_oof_probs)

    active_calibrator: OutOfFoldCalibrator = affine_oof_cal
    final_oof_probs = affine_oof_probs
    final_metrics = affine_metrics
    selected_cal_method = "affine_logit"

    # 선택적 지면(inventory_id) Shrinkage 바이어스 보정 비교 및 조건부 채택
    gb_cfg = cal_cfg.get("group_bias", {})
    group_bias_eval: dict[str, Any] | None = None
    tr_groups: np.ndarray | None = None
    te_groups: np.ndarray | None = None
    group_col = str(gb_cfg.get("group_col", "inventory_id"))

    explicit_group_method = str(cal_cfg.get("method", "")) == "group_shrinkage"
    if bool(gb_cfg.get("enabled", False)) or explicit_group_method:
        tr_tab_p = paths_cfg.get("train_tabular")
        te_tab_p = paths_cfg.get("test_tabular")
        if not tr_tab_p or not Path(tr_tab_p).exists():
            if explicit_group_method:
                raise FileNotFoundError(
                    f"group_shrinkage 보정에는 train_tabular 파일이 필요합니다: {tr_tab_p}"
                )
        else:
            tr_schema_names = set(pq.ParquetFile(tr_tab_p).schema_arrow.names)
            if group_col not in tr_schema_names:
                raise KeyError(f"train_tabular에 그룹 컬럼 '{group_col}'이 없습니다.")
            read_tr_cols = ["row_id", group_col] if "row_id" in tr_schema_names else [group_col]
            tr_meta_df = pq.read_table(tr_tab_p, columns=read_tr_cols).to_pandas()
            if "row_id" in tr_meta_df.columns:
                tr_meta_df = (
                    tr_meta_df.set_index("row_id")
                    .reindex(blend_res.oof_df["row_id"].to_numpy())
                    .reset_index()
                )
            tr_groups = tr_meta_df[group_col].to_numpy()
            if te_tab_p and Path(te_tab_p).exists():
                te_schema_names = set(pq.ParquetFile(te_tab_p).schema_arrow.names)
                if id_col not in te_schema_names or group_col not in te_schema_names:
                    raise KeyError(
                        f"test_tabular({te_tab_p})에 '{id_col}' 및 '{group_col}' 컬럼이 모두 필요합니다."
                    )
                te_meta_df = pq.read_table(te_tab_p, columns=[id_col, group_col]).to_pandas()
                expected_te_ids = (
                    reference_ids
                    if reference_ids is not None
                    else (
                        blend_res.test_df[id_col].to_numpy()
                        if blend_res.test_df is not None
                        else None
                    )
                )
                if expected_te_ids is not None and not _id_arrays_equal(
                    te_meta_df[id_col].to_numpy(), expected_te_ids
                ):
                    raise ValueError(
                        f"test_tabular의 '{id_col}' 순서가 reference_ids / test 예측과 일치하지 않습니다."
                    )
                te_groups = te_meta_df[group_col].to_numpy()

            gb_cal = OutOfFoldCalibrator(
                calibrator_factory=lambda: GroupBiasShrinkageCalibrator(
                    shrinkage=float(gb_cfg.get("shrinkage", 500.0)),
                    max_offset=float(gb_cfg.get("max_offset", 3.0)),
                    l2_reg=float(cal_cfg.get("l2_reg", 1.0e-6)),
                    min_scale=float(cal_cfg.get("min_scale", 1.0e-3)),
                    max_scale=float(cal_cfg.get("max_scale", 20.0)),
                ),
                test_agg=str(cal_cfg.get("test_agg", "logit_mean")),  # type: ignore[arg-type]
            )
            gb_oof_probs = gb_cal.fit_transform(
                blended_oof_logits, y_all, folds_all, groups=tr_groups, input_space="logit"
            )
            gb_metrics = evaluate_predictions(y_all, gb_oof_probs)
            group_bias_eval = gb_metrics.to_dict()

            only_if_improves = bool(gb_cfg.get("only_accept_if_improves", True))
            if explicit_group_method or (not only_if_improves) or (
                gb_metrics.score > affine_metrics.score + 1e-6
            ):
                active_calibrator = gb_cal
                final_oof_probs = gb_oof_probs
                final_metrics = gb_metrics
                selected_cal_method = "group_shrinkage"

    submission_report: dict[str, Any] | None = None
    submission_path = paths_cfg.get("submission_path")

    if blend_res.test_df is not None and submission_path:
        test_logits = blend_res.test_df["raw_logit"].to_numpy(dtype=np.float64)
        test_ids = blend_res.test_df[id_col].to_numpy()

        if selected_cal_method == "group_shrinkage":
            if te_groups is None:
                raise ValueError(
                    "group_shrinkage 보정기가 선택되었으나 test_tabular에서 te_groups를 로드할 수 없어 "
                    "OOF 평가 기준과 일관된 Test 추론을 수행할 수 없습니다."
                )
            cal_test_probs = active_calibrator.predict_proba(
                test_logits, groups=te_groups, input_space="logit"
            )
        else:
            cal_test_probs = affine_oof_cal.predict_proba(test_logits, input_space="logit")

        sub_df = generate_submission(
            test_ids=test_ids,
            pred_probs=cal_test_probs,
            output_path=submission_path,
            id_col=id_col,
            target_col=target_col,
            eps=float(sub_cfg.get("eps", 1.0e-15)),
        )

        if sample_df is not None and not pd.api.types.is_numeric_dtype(sample_df[id_col].dtype):
            sub_check_df = sub_df.assign(**{id_col: sub_df[id_col].astype(str)})
        else:
            sub_check_df = sub_df

        exp_rows_cfg = sub_cfg.get("expected_rows")
        expected_rows = int(exp_rows_cfg) if exp_rows_cfg is not None else None
        submission_report = verify_submission_df(
            sub_df=sub_check_df,
            sample_sub_df=sample_df,
            expected_rows=expected_rows,
            id_col=id_col,
            target_col=target_col,
            raise_on_error=bool(sub_cfg.get("strict_verify", True)),
        )
        submission_report["submission_path"] = str(submission_path)

    calibrator_path = paths_cfg.get("calibrator_path")
    if calibrator_path:
        cal_p = Path(calibrator_path)
        cal_p.parent.mkdir(parents=True, exist_ok=True)
        cal_payload = {
            "weights": blend_res.weights,
            "selected_calibration_method": selected_cal_method,
            "oof_calibrator": active_calibrator.to_dict(),
            "inference_affine_calibrator": (
                affine_oof_cal.get_inference_calibrator().to_dict()
                if affine_oof_cal.mean_scale_ is not None
                else None
            ),
        }
        with cal_p.open("w", encoding="utf-8") as fp:
            json.dump(cal_payload, fp, ensure_ascii=False, indent=2)

    report_payload: dict[str, Any] = {
        "models": blend_res.model_names,
        "weights": blend_res.weights,
        "fold_weights": blend_res.fold_weights,
        "single_model_metrics": blend_res.single_model_metrics,
        "blended_pre_calibration": pre_cal_metrics.to_dict(),
        "calibrated_affine_oof": affine_metrics.to_dict(),
        "ap_preservation_abs_diff": abs(affine_metrics.ap - pre_cal_metrics.ap),
        "group_bias_oof": group_bias_eval,
        "selected_calibration_method": selected_cal_method,
        "final_oof_metrics": final_metrics.to_dict(),
        "oof_pred_mean": float(np.mean(final_oof_probs)),
        "submission_verification": submission_report,
    }

    report_path = paths_cfg.get("report_path")
    if report_path:
        rep_p = Path(report_path)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        with rep_p.open("w", encoding="utf-8") as fp:
            json.dump(report_payload, fp, ensure_ascii=False, indent=2)

    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
