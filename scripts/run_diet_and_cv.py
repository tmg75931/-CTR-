"""Data Diet(정형·시퀀스 스트리밍 분리) 및 시간축 적응형 Local CV 분할 CLI 스크립트."""

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

from toss_ctr.data import DEFAULT_CONTRACT, build_cv_folds, build_diet_artifacts  # noqa: E402
from toss_ctr.inference.submission import load_yaml_with_overrides  # noqa: E402


def _flatten_overrides(raw_groups: list[list[str]] | None) -> list[str]:
    return [item for grp in (raw_groups or []) for item in grp]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="원천 Parquet 스트리밍 경량화(Diet) 및 5-Fold Local CV 분할 생성기"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "data_diet.yaml"),
        help="Data Diet YAML 설정 파일 경로 (기본: configs/data_diet.yaml)",
    )
    parser.add_argument(
        "--cv-config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "cv_strategy.yaml"),
        help="CV 분할 YAML 설정 파일 경로 (기본: configs/cv_strategy.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=None,
        help="YAML 설정 오버라이드 (예: --override paths.raw_train=/tmp/train.parquet cv.n_splits=5)",
    )
    parser.add_argument("--skip-diet", action="store_true", help="Diet 생성을 건너뛰고 CV만 실행")
    parser.add_argument("--skip-cv", action="store_true", help="CV 생성을 건너뛰고 Diet만 실행")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Data Diet 및 CV 분할 파이프라인을 실행합니다."""
    args = parse_args(argv)
    flat_overrides = _flatten_overrides(args.override)

    diet_overrides: list[str] = []
    cv_overrides: list[str] = []
    explicit_cv_paths: set[str] = set()

    for ov in flat_overrides:
        key_part = ov.split("=", 1)[0].strip()
        if key_part.startswith("diet."):
            diet_overrides.append(ov[len("diet.") :])
        elif key_part.startswith("cv_cfg."):
            sub_ov = ov[len("cv_cfg.") :]
            cv_overrides.append(sub_ov)
            explicit_cv_paths.add(sub_ov.split("=", 1)[0].strip())
        elif key_part.startswith(("cv.", "oot.")):
            cv_overrides.append(ov)
        elif key_part.startswith("paths."):
            # output_dir 등 공통 경로가 변경되면 양쪽 모두에서 일관성 유지
            if key_part in ("paths.raw_train", "paths.raw_test", "paths.eda_summary", "paths.output_dir"):
                diet_overrides.append(ov)
            else:
                cv_overrides.append(ov)
                explicit_cv_paths.add(key_part)
        else:
            diet_overrides.append(ov)
            if key_part.startswith("streaming."):
                cv_overrides.append(ov)

    diet_cfg = load_yaml_with_overrides(args.config, diet_overrides)
    cv_cfg = load_yaml_with_overrides(args.cv_config, cv_overrides)

    out_dir = Path(diet_cfg["paths"]["output_dir"])
    artifacts_cfg = diet_cfg.get("artifacts", {})
    train_tab_name = artifacts_cfg.get("train_tabular", DEFAULT_CONTRACT.train_tabular_filename)
    test_tab_name = artifacts_cfg.get("test_tabular", DEFAULT_CONTRACT.test_tabular_filename)

    if "paths.train_path" not in explicit_cv_paths and (
        "paths.output_dir=" in " ".join(diet_overrides) or not Path(cv_cfg["paths"]["train_path"]).exists()
    ):
        cv_cfg["paths"]["train_path"] = str(out_dir / train_tab_name)
    if "paths.test_path" not in explicit_cv_paths and (
        "paths.output_dir=" in " ".join(diet_overrides) or not Path(cv_cfg["paths"]["test_path"]).exists()
    ):
        cv_cfg["paths"]["test_path"] = str(out_dir / test_tab_name)
    if "paths.output_path" not in explicit_cv_paths and "paths.output_dir=" in " ".join(diet_overrides):
        cv_cfg["paths"]["output_path"] = str(out_dir / DEFAULT_CONTRACT.folds_filename)
    if "paths.report_path" not in explicit_cv_paths and "paths.output_dir=" in " ".join(diet_overrides):
        cv_cfg["paths"]["report_path"] = str(out_dir / "cv_summary.json")

    result: dict[str, Any] = {}

    if not args.skip_diet:
        d_paths = diet_cfg["paths"]
        d_stream = diet_cfg.get("streaming", {})
        d_cast = diet_cfg.get("downcast", {})
        manifest = build_diet_artifacts(
            train_path=d_paths["raw_train"],
            test_path=d_paths["raw_test"],
            output_dir=out_dir,
            batch_size=int(d_stream.get("batch_size", 262_144)),
            compression=str(d_stream.get("compression", "zstd")),
            compression_level=d_stream.get("compression_level", 3),
            eda_summary_path=d_paths.get("eda_summary"),
            downcast_float32=bool(d_cast.get("downcast_float32", True)),
            downcast_integers=bool(d_cast.get("downcast_integers", True)),
            verify_row_counts=bool(d_cast.get("verify_row_counts", True)),
            save_manifest=bool(d_cast.get("save_manifest", True)),
        )
        result["diet_manifest"] = {
            "train_rows": manifest["train_rows"],
            "test_rows": manifest["test_rows"],
            "total_size_reduction_pct": manifest["total_size_reduction_pct"],
            "artifacts": manifest["artifacts"],
        }

    if not args.skip_cv:
        c_paths = cv_cfg["paths"]
        c_cv = cv_cfg.get("cv", {})
        c_oot = cv_cfg.get("oot", {})
        c_stream = cv_cfg.get("streaming", {})
        _, cv_summary = build_cv_folds(
            train_path=c_paths["train_path"],
            test_path=c_paths["test_path"],
            output_path=c_paths["output_path"],
            n_splits=int(c_cv.get("n_splits", 5)),
            seed=int(c_cv.get("seed", 42)),
            oot_mode=str(c_oot.get("mode", "auto")),
            tvd_threshold=float(c_oot.get("tvd_threshold", 0.15)),
            mark_last_day_if_in_time=bool(c_oot.get("mark_last_day_if_in_time", True)),
            batch_size=int(c_stream.get("batch_size", 262_144)),
            compression=str(c_stream.get("compression", "zstd")),
            report_path=c_paths.get("report_path"),
        )
        result["cv_summary"] = cv_summary

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
