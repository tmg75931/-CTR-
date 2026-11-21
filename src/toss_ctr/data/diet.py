"""대용량 Parquet 스트리밍 분리(Tabular Diet + Seq 분리) 파이프라인 모듈."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toss_ctr.data.schema import (
    DEFAULT_CONTRACT,
    DatasetContract,
    collect_parquet_column_stats,
    discover_feature_groups,
    downcast_arrow_table,
    infer_safe_dtypes,
    load_eda_summary,
)

_LEVEL_SUPPORTED_CODECS = frozenset({"zstd", "gzip", "brotli"})


def _normalize_compression_kwargs(
    compression: str,
    compression_level: int | None,
) -> dict[str, Any]:
    codec = compression.lower()
    kwargs: dict[str, Any] = {"compression": codec}
    if compression_level is not None and codec in _LEVEL_SUPPORTED_CODECS:
        kwargs["compression_level"] = int(compression_level)
    return kwargs


def _build_target_schema(
    ordered_cols: list[str],
    source_schema: pa.Schema,
    dtype_map: Mapping[str, pa.DataType],
    extra_fields: Mapping[str, pa.Field] | None = None,
) -> pa.Schema:
    fields: list[pa.Field] = []
    extra = extra_fields or {}
    for col in ordered_cols:
        if col in extra:
            fields.append(extra[col])
            continue
        src_field = source_schema.field(col)
        target_type = dtype_map.get(col, src_field.type)
        fields.append(pa.field(col, target_type, nullable=src_field.nullable))
    return pa.schema(fields)


def _stream_split_parquet(
    source_path: Path,
    tabular_out_path: Path,
    seq_out_path: Path,
    tabular_feature_cols: list[str],
    tabular_target_schema: pa.Schema,
    dtype_map: Mapping[str, pa.DataType],
    is_train: bool,
    batch_size: int,
    writer_kwargs: dict[str, Any],
    contract: DatasetContract,
) -> int:
    """원천 Parquet을 배치 스트리밍으로 읽어 정형 Diet 테이블과 seq 테이블로 동시 분리 기록합니다."""
    pf = pq.ParquetFile(source_path)
    has_seq = contract.seq_col in pf.schema_arrow.names
    if not has_seq:
        raise ValueError(f"원천 파일에 '{contract.seq_col}' 컬럼이 없습니다: {source_path}")

    if is_train:
        seq_key_field = pa.field(contract.row_id_col, pa.int64(), nullable=False)
        read_cols = [contract.target_col, *tabular_feature_cols, contract.seq_col]
    else:
        id_src_field = pf.schema_arrow.field(contract.id_col)
        seq_key_field = pa.field(contract.id_col, id_src_field.type, nullable=id_src_field.nullable)
        read_cols = [contract.id_col, *tabular_feature_cols, contract.seq_col]

    seq_src_field = pf.schema_arrow.field(contract.seq_col)
    seq_target_schema = pa.schema(
        [
            seq_key_field,
            pa.field(contract.seq_col, seq_src_field.type, nullable=seq_src_field.nullable),
        ]
    )

    row_offset = 0
    with (
        pq.ParquetWriter(tabular_out_path, tabular_target_schema, **writer_kwargs) as tab_writer,
        pq.ParquetWriter(seq_out_path, seq_target_schema, **writer_kwargs) as seq_writer,
    ):
        for record_batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
            n_batch = record_batch.num_rows
            if n_batch == 0:
                continue

            batch_table = pa.Table.from_batches([record_batch])
            seq_col_arr = batch_table.column(contract.seq_col)

            if is_train:
                row_id_np = np.arange(row_offset, row_offset + n_batch, dtype=np.int64)
                row_id_arr = pa.array(row_id_np, type=pa.int64())

                tab_arrays: list[pa.Array | pa.ChunkedArray] = [row_id_arr]
                tab_fields: list[pa.Field] = [
                    pa.field(contract.row_id_col, pa.int64(), nullable=False)
                ]
                for col in [contract.target_col, *tabular_feature_cols]:
                    src_f = pf.schema_arrow.field(col)
                    tab_arrays.append(batch_table.column(col))
                    tab_fields.append(pa.field(col, src_f.type, nullable=src_f.nullable))
                raw_tab_table = pa.Table.from_arrays(tab_arrays, schema=pa.schema(tab_fields))

                seq_table = pa.Table.from_arrays(
                    [row_id_arr, seq_col_arr],
                    schema=seq_target_schema,
                )
            else:
                id_col_arr = batch_table.column(contract.id_col)
                tab_arrays = [id_col_arr]
                id_f = pf.schema_arrow.field(contract.id_col)
                tab_fields = [pa.field(contract.id_col, id_f.type, nullable=id_f.nullable)]
                for col in tabular_feature_cols:
                    src_f = pf.schema_arrow.field(col)
                    tab_arrays.append(batch_table.column(col))
                    tab_fields.append(pa.field(col, src_f.type, nullable=src_f.nullable))
                raw_tab_table = pa.Table.from_arrays(tab_arrays, schema=pa.schema(tab_fields))

                seq_table = pa.Table.from_arrays(
                    [id_col_arr, seq_col_arr],
                    schema=seq_target_schema,
                )

            downcast_tab = downcast_arrow_table(
                raw_tab_table,
                dtype_map=dtype_map,
                contract=contract,
            )
            # 미리 고정된 target_schema와 nullable 플래그/순서를 일치시킵니다.
            aligned_tab = pa.Table.from_arrays(
                [downcast_tab.column(f.name) for f in tabular_target_schema],
                schema=tabular_target_schema,
            )

            tab_writer.write_table(aligned_tab)
            seq_writer.write_table(seq_table)
            row_offset += n_batch

        if row_offset == 0:
            tab_writer.write_table(tabular_target_schema.empty_table())
            seq_writer.write_table(seq_target_schema.empty_table())

    return row_offset


def build_diet_artifacts(
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 262_144,
    compression: str = "zstd",
    compression_level: int | None = 3,
    eda_summary_path: str | Path | None = None,
    downcast_float32: bool = True,
    downcast_integers: bool = True,
    contract: DatasetContract = DEFAULT_CONTRACT,
    verify_row_counts: bool = True,
    save_manifest: bool = True,
) -> dict[str, Any]:
    """원천 `train.parquet`과 `test.parquet`을 OOM 없이 스트리밍 분리·다운캐스팅합니다.

    생성 산출물:
    - `train_tabular_diet.parquet`: `row_id` (0-based `int64`), `clicked` (`int8`), 정형 다운캐스팅 피처
    - `test_tabular_diet.parquet`: `ID` (원본 순서/타입 보존), 정형 다운캐스팅 피처
    - `train_seq.parquet`: `row_id`, `seq`
    - `test_seq.parquet`: `ID`, `seq`
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size는 1 이상이어야 합니다: {batch_size}")

    train_src = Path(train_path)
    test_src = Path(test_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_pf = pq.ParquetFile(train_src)
    test_pf = pq.ParquetFile(test_src)

    train_schema = train_pf.schema_arrow
    test_schema = test_pf.schema_arrow

    if contract.target_col not in train_schema.names:
        raise ValueError(f"train 파일에 타겟 '{contract.target_col}' 컬럼이 없습니다.")
    if contract.id_col not in test_schema.names:
        raise ValueError(f"test 파일에 식별자 '{contract.id_col}' 컬럼이 없습니다.")

    excluded = {contract.target_col, contract.id_col, contract.seq_col, contract.row_id_col}
    train_features = [c for c in train_schema.names if c not in excluded]
    test_features = [c for c in test_schema.names if c not in excluded]
    shared_features = [c for c in train_features if c in set(test_features)]

    eda_summary = load_eda_summary(eda_summary_path)
    column_stats = (
        collect_parquet_column_stats([train_src, test_src], columns=shared_features)
        if downcast_integers or downcast_float32
        else {}
    )

    # 공통 피처 + train/test 전용 컬럼의 스키마를 병합하여 일관된 다운캐스팅 매핑을 생성합니다.
    merged_fields: list[pa.Field] = [
        pa.field(contract.row_id_col, pa.int64(), nullable=False),
        pa.field(contract.target_col, pa.int8(), nullable=False),
        test_schema.field(contract.id_col),
    ]
    seen_field_names = {f.name for f in merged_fields}
    for f in list(train_schema) + list(test_schema):
        if f.name in seen_field_names or f.name == contract.seq_col:
            continue
        merged_fields.append(f)
        seen_field_names.add(f.name)

    merged_schema = pa.schema(merged_fields)
    if downcast_integers:
        dtype_map = infer_safe_dtypes(
            merged_schema,
            eda_summary=eda_summary,
            column_stats=column_stats,
            contract=contract,
            downcast_float64_to_float32=downcast_float32,
            downcast_int_without_stats=False,
        )
    else:
        # 정수 다운캐스팅이 꺼져 있으면 정수는 원본 타입 유지, float64만 선택적으로 축소
        dtype_map = infer_safe_dtypes(
            merged_schema,
            eda_summary=None,
            column_stats={},
            contract=contract,
            downcast_float64_to_float32=downcast_float32,
            downcast_int_without_stats=False,
        )

    train_tabular_cols = [contract.row_id_col, contract.target_col, *train_features]
    test_tabular_cols = [contract.id_col, *test_features]

    train_tabular_schema = _build_target_schema(
        train_tabular_cols,
        train_schema,
        dtype_map,
        extra_fields={
            contract.row_id_col: pa.field(contract.row_id_col, pa.int64(), nullable=False),
            contract.target_col: pa.field(contract.target_col, pa.int8(), nullable=False),
        },
    )
    test_tabular_schema = _build_target_schema(
        test_tabular_cols,
        test_schema,
        dtype_map,
    )

    train_tab_path = out_dir / contract.train_tabular_filename
    test_tab_path = out_dir / contract.test_tabular_filename
    train_seq_path = out_dir / contract.train_seq_filename
    test_seq_path = out_dir / contract.test_seq_filename

    writer_kwargs = _normalize_compression_kwargs(compression, compression_level)

    written_train_rows = _stream_split_parquet(
        source_path=train_src,
        tabular_out_path=train_tab_path,
        seq_out_path=train_seq_path,
        tabular_feature_cols=train_features,
        tabular_target_schema=train_tabular_schema,
        dtype_map=dtype_map,
        is_train=True,
        batch_size=batch_size,
        writer_kwargs=writer_kwargs,
        contract=contract,
    )
    written_test_rows = _stream_split_parquet(
        source_path=test_src,
        tabular_out_path=test_tab_path,
        seq_out_path=test_seq_path,
        tabular_feature_cols=test_features,
        tabular_target_schema=test_tabular_schema,
        dtype_map=dtype_map,
        is_train=False,
        batch_size=batch_size,
        writer_kwargs=writer_kwargs,
        contract=contract,
    )

    if verify_row_counts:
        expected_train = train_pf.metadata.num_rows
        expected_test = test_pf.metadata.num_rows
        for p, expected in (
            (train_tab_path, expected_train),
            (train_seq_path, expected_train),
            (test_tab_path, expected_test),
            (test_seq_path, expected_test),
        ):
            actual = pq.ParquetFile(p).metadata.num_rows
            if actual != expected:
                raise ValueError(
                    f"산출물 행 수 불일치 ({p.name}): expected={expected}, actual={actual}"
                )

    raw_train_bytes = train_src.stat().st_size
    raw_test_bytes = test_src.stat().st_size
    raw_total_bytes = raw_train_bytes + raw_test_bytes

    out_files = {
        "train_tabular": train_tab_path,
        "test_tabular": test_tab_path,
        "train_seq": train_seq_path,
        "test_seq": test_seq_path,
    }
    artifact_stats: dict[str, dict[str, Any]] = {}
    diet_total_bytes = 0
    tabular_total_bytes = 0
    for key, p in out_files.items():
        size_b = p.stat().st_size
        pf_out = pq.ParquetFile(p)
        diet_total_bytes += size_b
        if "tabular" in key:
            tabular_total_bytes += size_b
        artifact_stats[key] = {
            "path": str(p),
            "rows": int(pf_out.metadata.num_rows),
            "columns": int(pf_out.metadata.num_columns),
            "size_bytes": int(size_b),
            "size_mb": round(size_b / (1024 ** 2), 3),
        }

    feature_groups = discover_feature_groups(train_features, eda_summary=eda_summary, contract=contract)

    manifest: dict[str, Any] = {
        "train_rows": int(written_train_rows),
        "test_rows": int(written_test_rows),
        "compression": writer_kwargs["compression"],
        "batch_size": int(batch_size),
        "raw_sizes_bytes": {
            "train": int(raw_train_bytes),
            "test": int(raw_test_bytes),
            "total": int(raw_total_bytes),
        },
        "diet_total_bytes": int(diet_total_bytes),
        "tabular_only_bytes": int(tabular_total_bytes),
        "total_size_reduction_pct": (
            round((1.0 - diet_total_bytes / raw_total_bytes) * 100.0, 2)
            if raw_total_bytes > 0
            else 0.0
        ),
        "tabular_size_reduction_pct": (
            round((1.0 - tabular_total_bytes / raw_total_bytes) * 100.0, 2)
            if raw_total_bytes > 0
            else 0.0
        ),
        "artifacts": artifact_stats,
        "feature_groups": feature_groups,
        "feature_group_counts": {k: len(v) for k, v in feature_groups.items()},
        "tabular_dtypes": {
            col: str(train_tabular_schema.field(col).type) for col in train_tabular_cols
        },
    }

    if save_manifest:
        manifest_path = out_dir / "diet_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fp:
            json.dump(manifest, fp, ensure_ascii=False, indent=2)
        manifest["manifest_path"] = str(manifest_path)

    return manifest
