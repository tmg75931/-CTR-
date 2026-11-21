"""PyTorch Target-Attention + DCN-v2 DeepCTR 5-Fold 스트리밍 학습·추론 CLI 스크립트."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from toss_ctr.inference.submission import load_yaml_with_overrides  # noqa: E402
from toss_ctr.models.deep_ctr import DeepCTRConfig, train_deep_ctr_oof_from_parquet  # noqa: E402


def _flatten_overrides(raw_groups: list[list[str]] | None) -> list[str]:
    return [item for grp in (raw_groups or []) for item in grp]


def _build_deep_config_from_dict(raw: dict[str, Any]) -> DeepCTRConfig:
    seq_cfg = raw.get("seq", {})
    prep_cfg = raw.get("preprocessing", {})
    model_cfg = raw.get("model", {})
    train_cfg = raw.get("training", {})
    return DeepCTRConfig(
        delimiter=seq_cfg.get("delimiter", "auto"),
        max_seq_len=int(seq_cfg.get("max_seq_len", 100)),
        min_freq=int(seq_cfg.get("min_freq", 5)),
        max_vocab_size=int(seq_cfg.get("max_vocab_size", 100_000)),
        padding=str(seq_cfg.get("padding", "post")),
        truncating=str(seq_cfg.get("truncating", "pre")),
        cat_min_freq=int(prep_cfg.get("cat_min_freq", 5)),
        cat_max_cardinality=int(prep_cfg.get("cat_max_cardinality", 100_000)),
        apply_signed_log1p=bool(prep_cfg.get("apply_signed_log1p", True)),
        numeric_clip_value=(
            float(prep_cfg["numeric_clip_value"])
            if prep_cfg.get("numeric_clip_value") is not None
            else None
        ),
        stream_batch_size=int(prep_cfg.get("stream_batch_size", 262_144)),
        seq_embed_dim=int(model_cfg.get("seq_embed_dim", 32)),
        cat_embed_dim=int(model_cfg.get("cat_embed_dim", 16)),
        query_cols=list(model_cfg.get("query_cols", ["inventory_id", "l_feat_14"])),
        att_hidden_dim=int(model_cfg.get("att_hidden_dim", 64)),
        num_proj_dim=int(model_cfg.get("num_proj_dim", 64)),
        cross_layers=int(model_cfg.get("cross_layers", 3)),
        cross_low_rank_dim=(
            int(model_cfg["cross_low_rank_dim"])
            if model_cfg.get("cross_low_rank_dim") is not None
            else None
        ),
        mlp_dims=[int(x) for x in model_cfg.get("mlp_dims", [256, 128, 64])],
        dropout=float(model_cfg.get("dropout", 0.15)),
        n_splits=int(train_cfg.get("n_splits", 5)),
        batch_size=int(train_cfg.get("batch_size", 2048)),
        eval_batch_size=int(train_cfg.get("eval_batch_size", 4096)),
        epochs=int(train_cfg.get("epochs", 8)),
        lr=float(train_cfg.get("lr", 1.0e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-5)),
        use_amp=bool(train_cfg.get("use_amp", True)),
        amp_dtype=str(train_cfg.get("amp_dtype", "float16")),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        early_stopping_patience=int(train_cfg.get("early_stopping_patience", 2)),
        monitor_metric=str(train_cfg.get("monitor_metric", "score")).lower(),
        seed=int(train_cfg.get("seed", 42)),
        use_pos_weight_50_50=bool(train_cfg.get("use_pos_weight_50_50", True)),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="PyTorch Target-Attention + DCN-v2 5-Fold DeepCTR 학습·추론"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "deep_seq_dcn.yaml"),
        help="DeepCTR YAML 설정 파일 경로 (기본: configs/deep_seq_dcn.yaml)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="실행할 특정 폴드 인덱스 (예: --folds 0 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch 연산 디바이스 (예: cuda, cpu)",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=None,
        help="YAML 설정 오버라이드 (예: --override training.epochs=2 training.use_amp=false)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """DeepCTR 스트리밍 Memmap OOF 학습·추론을 실행합니다."""
    args = parse_args(argv)
    raw_cfg = load_yaml_with_overrides(args.config, _flatten_overrides(args.override))
    deep_cfg = _build_deep_config_from_dict(raw_cfg)

    paths_cfg = raw_cfg["paths"]
    pred_dir = Path(paths_cfg["pred_dir"])
    oof_out = pred_dir / paths_cfg.get("oof_pred_filename", "oof_deep_seq_dcn.parquet")
    test_out = pred_dir / paths_cfg.get("test_pred_filename", "test_deep_seq_dcn.parquet")
    report_dir = Path(paths_cfg.get("report_dir", pred_dir))
    report_path = report_dir / "deep_seq_dcn_report.json"

    test_tab = paths_cfg.get("test_tabular")
    test_seq = paths_cfg.get("test_seq")
    has_test = bool(test_tab and test_seq and Path(test_tab).exists() and Path(test_seq).exists())

    summary = train_deep_ctr_oof_from_parquet(
        train_tabular_path=paths_cfg["train_tabular"],
        train_seq_path=paths_cfg["train_seq"],
        folds_path=paths_cfg["folds_path"],
        test_tabular_path=test_tab if has_test else None,
        test_seq_path=test_seq if has_test else None,
        config=deep_cfg,
        oof_output_path=oof_out,
        test_output_path=test_out if has_test else None,
        memmap_cache_dir=paths_cfg.get("memmap_cache_dir", "/content/tmp_deepctr_cache"),
        model_dir=paths_cfg.get("model_dir"),
        report_path=report_path,
        folds_to_run=args.folds,
        device_str=args.device,
        cleanup_cache=True,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
