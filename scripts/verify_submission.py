"""최종 제출 파일(`submission.csv`)과 `sample_submission.csv` 간 무결성 검증 CLI 스크립트."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from toss_ctr.inference.submission import (  # noqa: E402
    load_yaml_with_overrides,
    verify_submission_file,
)


def _flatten_overrides(raw_groups: list[list[str]] | None) -> list[str]:
    return [item for grp in (raw_groups or []) for item in grp]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="제출 파일 스키마, NaN/Inf, (0,1) 범위, ID 순서 및 분포 리포트 검증"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "ensemble_wll50.yaml"),
        help="기본 경로·행 수를 참조할 YAML 설정 파일 (기본: configs/ensemble_wll50.yaml)",
    )
    parser.add_argument(
        "--submission",
        type=str,
        default=None,
        help="검증할 제출 CSV/Parquet 경로 (미지정 시 config.paths.submission_path 사용)",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="비교 기준 sample_submission.csv 경로 (미지정 시 config.paths.sample_submission 사용)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="기대 행 수 (예: 1527298, 0 전달 시 행 수 고정 검사 생략)",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=None,
        help="YAML 설정 오버라이드 (예: --override submission.expected_rows=60)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """제출 무결성 검증을 실행하고 리포트를 출력합니다."""
    args = parse_args(argv)
    cfg: dict = {}
    if args.config and Path(args.config).exists():
        cfg = load_yaml_with_overrides(args.config, _flatten_overrides(args.override))

    paths_cfg = cfg.get("paths", {})
    sub_cfg = cfg.get("submission", {})

    submission_path = args.submission or paths_cfg.get("submission_path")
    if not submission_path:
        raise ValueError("--submission 인자 또는 config의 paths.submission_path가 필요합니다.")

    sample_path = args.sample
    if sample_path is None:
        cfg_sample = paths_cfg.get("sample_submission")
        if cfg_sample and Path(cfg_sample).exists():
            sample_path = cfg_sample

    if args.expected_rows is not None:
        expected_rows = None if args.expected_rows <= 0 else int(args.expected_rows)
    else:
        cfg_exp = sub_cfg.get("expected_rows")
        expected_rows = int(cfg_exp) if cfg_exp is not None else None

    report = verify_submission_file(
        submission_path=submission_path,
        sample_path=sample_path,
        expected_rows=expected_rows,
        id_col=str(sub_cfg.get("id_col", "ID")),
        target_col=str(sub_cfg.get("target_col", "clicked")),
        raise_on_error=False,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
