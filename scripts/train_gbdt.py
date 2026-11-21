"""정형 피처 파이프라인 및 다중 백엔드(LightGBM/CatBoost/XGBoost/Sklearn HGBT) OOF 학습 CLI."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from toss_ctr.data import DEFAULT_CONTRACT, load_folds  # noqa: E402
from toss_ctr.features.tabular_pipeline import TabularFeaturePipeline  # noqa: E402
from toss_ctr.inference.submission import load_yaml_with_overrides  # noqa: E402
from toss_ctr.models.gbdt_trainer import GBDTOOFTrainer  # noqa: E402


def _flatten_overrides(raw_groups: list[list[str]] | None) -> list[str]:
    return [item for grp in (raw_groups or []) for item in grp]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="5-Fold GBDT(LightGBM/CatBoost/XGBoost/Sklearn HGBT) 학습 및 OOF/Test 예측 생성"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "gbdt_baseline.yaml"),
        help="GBDT YAML 설정 파일 경로 (gbdt_baseline.yaml 또는 gbdt_fe_tuned.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="다중 모델 설정(`models:` 블록) 중 학습할 모델 키 (미지정 시 전체 또는 단일 모델 실행)",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=None,
        help="YAML 설정 오버라이드 (예: --override backend=sklearn_hgbt params.max_iter=30)",
    )
    return parser.parse_args(argv)


def _apply_top_level_override_to_model(
    cfg: dict[str, Any],
    model_key: str,
    explicit_keys: set[str] | None = None,
) -> None:
    if "models" not in cfg or model_key not in cfg["models"]:
        return
    m_cfg = cfg["models"][model_key]
    keys_set = explicit_keys or set()
    for shared_key in (
        "backend",
        "model_name",
        "weighting_mode",
        "use_50_50_weight",
        "apply_prior_shift",
        "target_pos_rate",
        "early_stopping_rounds",
        "test_agg",
    ):
        if shared_key in cfg:
            m_cfg[shared_key] = cfg[shared_key]
    if "seed" in keys_set and f"models.{model_key}.seed" not in keys_set and "seed" in cfg:
        m_cfg["seed"] = cfg["seed"]
    if "params" in cfg and isinstance(cfg["params"], dict):
        m_cfg.setdefault("params", {}).update(cfg["params"])


def _sanitize_params_for_backend(cfg: dict[str, Any], model_key: str | None) -> None:
    """`backend=sklearn_hgbt` 오버라이드 시 타 백엔드 전용 하이퍼파라미터를 안전 제거합니다."""
    target_cfg = (
        cfg["models"][model_key]
        if model_key is not None and "models" in cfg and model_key in cfg["models"]
        else cfg.get("model", cfg)
    )
    backend = target_cfg.get("backend", cfg.get("backend", "lightgbm"))
    if backend != "sklearn_hgbt":
        return

    from sklearn.ensemble import HistGradientBoostingClassifier

    valid_keys = set(inspect.signature(HistGradientBoostingClassifier.__init__).parameters.keys()) - {
        "self"
    }
    raw_params = target_cfg.get("params")
    if isinstance(raw_params, dict):
        sanitized = {k: v for k, v in raw_params.items() if k in valid_keys}
        if "verbose" in sanitized and isinstance(sanitized["verbose"], (int, float)):
            sanitized["verbose"] = max(0, int(sanitized["verbose"]))
        target_cfg["params"] = sanitized


def main(argv: list[str] | None = None) -> int:
    """GBDT 5-Fold OOF 학습 및 표준 아티팩트 저장을 실행합니다."""
    args = parse_args(argv)
    flat_overrides = _flatten_overrides(args.override)
    explicit_keys = {ov.split("=", 1)[0].strip() for ov in flat_overrides if "=" in ov}
    cfg = load_yaml_with_overrides(args.config, flat_overrides)

    paths_cfg = cfg["paths"]
    train_df = pq.read_table(paths_cfg["train_path"]).to_pandas()
    test_path_raw = paths_cfg.get("test_path")
    test_df = (
        pq.read_table(test_path_raw).to_pandas()
        if test_path_raw and Path(test_path_raw).exists()
        else None
    )
    folds_df = load_folds(paths_cfg["folds_path"], validate=True, n_rows=len(train_df))

    pipeline: TabularFeaturePipeline | None = None
    seq_tr_path = paths_cfg.get("seq_summary_train")
    seq_te_path = paths_cfg.get("seq_summary_test")
    assume_static_prepared = False
    if "features" in cfg:
        pipeline = TabularFeaturePipeline.from_config(cfg)
        if seq_tr_path and not Path(seq_tr_path).exists():
            seq_tr_path = None
        if seq_te_path and not Path(seq_te_path).exists():
            seq_te_path = None
        # 다중 모델(--model all) 학습 시 행 단위 정적 피처/seq 조인을 1회만 생성해 중복 연산·메모리 절감
        train_df = pipeline.prepare_static_features(
            train_df,
            seq_summary=seq_tr_path,
            key_col=DEFAULT_CONTRACT.row_id_col,
        )
        if test_df is not None:
            test_df = pipeline.prepare_static_features(
                test_df,
                seq_summary=seq_te_path,
                key_col=DEFAULT_CONTRACT.id_col,
            )
        assume_static_prepared = True
        gc.collect()

    preds_dir = Path(paths_cfg["preds_dir"])

    if "models" in cfg and isinstance(cfg["models"], dict):
        if args.model is None or args.model == "all":
            model_keys: list[str | None] = list(cfg["models"].keys())
        else:
            if args.model not in cfg["models"]:
                raise KeyError(
                    f"모델 키 '{args.model}'를 설정에서 찾을 수 없습니다: {list(cfg['models'].keys())}"
                )
            model_keys = [args.model]
    else:
        model_keys = [None]

    summaries: list[dict[str, Any]] = []
    for m_key in model_keys:
        if m_key is not None:
            _apply_top_level_override_to_model(cfg, m_key, explicit_keys=explicit_keys)
        _sanitize_params_for_backend(cfg, m_key)

        trainer = GBDTOOFTrainer.from_config(cfg, model_key=m_key)
        res = trainer.fit_predict(
            train_df=train_df,
            folds=folds_df,
            test_df=test_df,
            pipeline=pipeline,
            train_seq_summary=None if assume_static_prepared else seq_tr_path,
            test_seq_summary=None if assume_static_prepared else seq_te_path,
            assume_static_prepared=assume_static_prepared,
            output_dir=preds_dir,
        )
        summaries.append(
            {
                "model_name": res.model_name,
                "backend": res.backend,
                "overall_oof": res.overall_metrics.to_dict(),
                "oot_metrics": res.oot_metrics.to_dict() if res.oot_metrics else None,
                "saved_paths": res.saved_paths,
            }
        )
        del res, trainer
        gc.collect()

    print(json.dumps({"models": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
