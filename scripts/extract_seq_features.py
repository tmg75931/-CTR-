"""대용량 `seq` Parquet에서 GBDT/정형 조인용 경량 요약 피처를 스트리밍 추출하는 CLI 스크립트."""

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

from toss_ctr.features.seq_features import extract_seq_summary_parquet  # noqa: E402
from toss_ctr.inference.submission import load_yaml_with_overrides  # noqa: E402


def _flatten_overrides(raw_groups: list[list[str]] | None) -> list[str]:
    return [item for grp in (raw_groups or []) for item in grp]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="train_seq.parquet / test_seq.parquet에서 경량 시퀀스 요약 피처 추출"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "deep_seq_dcn.yaml"),
        help="DeepCTR/Seq YAML 설정 파일 경로 (기본: configs/deep_seq_dcn.yaml)",
    )
    parser.add_argument(
        "--split",
        choices=["both", "train", "test"],
        default="both",
        help="추출 대상 분할 (기본: both)",
    )
    parser.add_argument(
        "--override",
        action="append",
        nargs="+",
        default=None,
        help="YAML 설정 오버라이드 (예: --override paths.train_seq=/tmp/train_seq.parquet)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """시퀀스 요약 피처 추출을 실행합니다."""
    args = parse_args(argv)
    cfg = load_yaml_with_overrides(args.config, _flatten_overrides(args.override))

    paths_cfg = cfg["paths"]
    seq_cfg = cfg.get("seq", {})
    delimiter = seq_cfg.get("delimiter", "auto")
    batch_size = int(seq_cfg.get("summary_batch_size", 131_072))
    compression = str(seq_cfg.get("compression", "zstd"))
    compression_level = seq_cfg.get("compression_level", 3)

    summary_out: dict[str, Any] = {}

    if args.split in ("both", "train"):
        summary_out["train"] = extract_seq_summary_parquet(
            seq_parquet_path=paths_cfg["train_seq"],
            output_path=paths_cfg["train_seq_summary"],
            id_col=None,
            delimiter=delimiter,
            batch_size=batch_size,
            compression=compression,
            compression_level=compression_level,
        )

    if args.split in ("both", "test"):
        test_seq = paths_cfg.get("test_seq")
        test_out = paths_cfg.get("test_seq_summary")
        if test_seq and test_out:
            summary_out["test"] = extract_seq_summary_parquet(
                seq_parquet_path=test_seq,
                output_path=test_out,
                id_col=None,
                delimiter=delimiter,
                batch_size=batch_size,
                compression=compression,
                compression_level=compression_level,
            )

    print(json.dumps(summary_out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
