"""데이터 스키마, 스트리밍 경량화(Diet), Local CV 분할 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from toss_ctr.data import (
    DEFAULT_CONTRACT,
    build_cv_folds,
    build_diet_artifacts,
    collect_parquet_column_stats,
    discover_feature_groups,
    downcast_arrow_table,
    downcast_dataframe,
    infer_safe_dtypes,
    load_eda_summary,
    load_folds,
    validate_folds_contract,
)


def _create_synthetic_parquets(
    tmp_path: Path,
    train_days: list[int] | None = None,
    test_days: list[int] | None = None,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """다중 Row Group을 가진 합성 train(120행)·test(40행) Parquet 파일을 생성합니다."""
    rng = np.random.default_rng(1234)
    n_train = 120
    n_test = 40

    if train_days is None:
        train_days = [1, 2, 3, 4, 5, 6]
    if test_days is None:
        test_days = [6, 7]

    def _build_common_cols(n: int, days: list[int], is_test: bool) -> dict[str, np.ndarray | list]:
        # float64 + NaN 포함 컬럼, 소범위 int64, test에서 범위가 더 큰 int64 포함
        feat_a = rng.normal(0.0, 1.0, size=n).astype(np.float64)
        feat_a[::13] = np.nan

        feat_b = rng.integers(0, 50, size=n, dtype=np.int64)
        # train은 int8 범위(0~100), test는 int16 범위(0~1000)로 만들어 통합 스키마 안전성 검증
        feat_c = (
            rng.integers(500, 1000, size=n, dtype=np.int64)
            if is_test
            else rng.integers(0, 100, size=n, dtype=np.int64)
        )
        feat_d = rng.uniform(-10.0, 10.0, size=n).astype(np.float64)
        feat_e = rng.integers(0, 2, size=n, dtype=np.int64)
        history_a = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
        l_feat_1 = rng.integers(0, 300, size=n, dtype=np.int64)
        l_feat_14 = rng.integers(1000, 1020, size=n, dtype=np.int64)

        seqs = [
            f"{i % 7 + 1},{(i * 3) % 19 + 10},{(i * 5) % 29 + 100}"
            if i % 11 != 0
            else ""
            for i in range(n)
        ]

        return {
            "gender": rng.integers(0, 3, size=n, dtype=np.int64),
            "age_group": rng.integers(1, 8, size=n, dtype=np.int64),
            "inventory_id": rng.integers(1, 15, size=n, dtype=np.int64),
            "day_of_week": np.array([days[i % len(days)] for i in range(n)], dtype=np.int64),
            "hour": rng.integers(0, 24, size=n, dtype=np.int64),
            "l_feat_14": l_feat_14,
            "l_feat_1": l_feat_1,
            "feat_a_1": feat_a,
            "feat_b_1": feat_b,
            "feat_c_1": feat_c,
            "feat_d_1": feat_d,
            "feat_e_1": feat_e,
            "history_a_1": history_a,
            "seq": seqs,
        }

    train_data = _build_common_cols(n_train, train_days, is_test=False)
    clicked = np.zeros(n_train, dtype=np.int64)
    clicked[::5] = 1  # 20% 양성 비율 (24개 양성)
    train_data["clicked"] = clicked
    train_df = pd.DataFrame(train_data)

    test_data = _build_common_cols(n_test, test_days, is_test=True)
    test_ids = [f"TEST_{i:05d}" for i in range(n_test)]
    test_df = pd.DataFrame({"ID": test_ids, **test_data})

    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"

    # 다중 Row Group 강제 생성 (train: 40행씩 3개, test: 15행씩 3개)
    pq.write_table(pa.Table.from_pandas(train_df, preserve_index=False), train_path, row_group_size=40)
    pq.write_table(pa.Table.from_pandas(test_df, preserve_index=False), test_path, row_group_size=15)

    return train_path, test_path, train_df, test_df


def test_discover_feature_groups_and_eda_loader(tmp_path: Path) -> None:
    """명시 컬럼과 익명 프리픽스 그룹 동적 분류 및 eda_summary 로더를 검증합니다."""
    train_path, _, train_df, _ = _create_synthetic_parquets(tmp_path)
    pf = pq.ParquetFile(train_path)

    eda_file = tmp_path / "eda_summary.json"
    eda_file.write_text(
        json.dumps(
            {
                "features": [
                    {"column": "feat_b_1", "min": 0, "max": 49, "null_rate": 0.0},
                    {"column": "feat_a_1", "min": -3.0, "max": 3.0, "null_rate": 0.08},
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded_eda = load_eda_summary(eda_file)
    assert loaded_eda is not None
    assert load_eda_summary(tmp_path / "missing.json") is None

    groups = discover_feature_groups(pf.schema_arrow.names, eda_summary=loaded_eda)
    assert set(groups["explicit"]) == {
        "gender",
        "age_group",
        "inventory_id",
        "day_of_week",
        "hour",
        "l_feat_14",
    }
    assert groups["l_feat"] == ["l_feat_1"]
    assert groups["feat_a"] == ["feat_a_1"]
    assert groups["feat_b"] == ["feat_b_1"]
    assert groups["feat_c"] == ["feat_c_1"]
    assert groups["feat_d"] == ["feat_d_1"]
    assert groups["feat_e"] == ["feat_e_1"]
    assert groups["history_a"] == ["history_a_1"]
    assert groups["other"] == []

    # 예약 컬럼(clicked, seq)은 피처 그룹에 포함되지 않아야 함
    all_grouped = [col for cols in groups.values() for col in cols]
    assert "clicked" not in all_grouped
    assert "seq" not in all_grouped
    assert len(all_grouped) == len(train_df.columns) - 2


def test_downcast_safety_arrow_and_pandas() -> None:
    """정수 범위 초과 방어, NaN 포함 float32 유지, 결측 보존을 검증합니다."""
    raw_table = pa.table(
        {
            "row_id": pa.array([0, 1, 2, 3], type=pa.int64()),
            "clicked": pa.array([0, 1, 0, 1], type=pa.int64()),
            "ID": pa.array(["A", "B", "C", "D"], type=pa.string()),
            "small_int": pa.array([1, 2, 3, 4], type=pa.int64()),
            "overflow_int": pa.array([10, 20, 40_000, 50], type=pa.int64()),
            "nan_float_intlike": pa.array([1.0, 2.0, np.nan, 4.0], type=pa.float64()),
            "null_float": pa.array([1.5, None, -2.5, 0.0], type=pa.float64()),
        }
    )

    # 의도적으로 overflow_int에 int8, nan_float_intlike에 int16 목표를 주입해 방어 로직을 테스트합니다.
    unsafe_map = {
        "row_id": "int64",
        "clicked": "int8",
        "small_int": "int8",
        "overflow_int": "int8",
        "nan_float_intlike": "int16",
        "null_float": "float32",
    }
    casted = downcast_arrow_table(raw_table, dtype_map=unsafe_map)
    assert casted.schema.field("row_id").type == pa.int64()
    assert casted.schema.field("clicked").type == pa.int8()
    assert casted.schema.field("ID").type == pa.string()
    assert casted.schema.field("small_int").type == pa.int8()
    # 40_000은 int8/int16을 초과하므로 int32로 자동 상향되어야 함
    assert casted.schema.field("overflow_int").type == pa.int32()
    # NaN이 포함된 float은 정수로 강제 변환되지 않고 float32로 유지되어야 함
    assert casted.schema.field("nan_float_intlike").type == pa.float32()
    assert casted.schema.field("null_float").type == pa.float32()
    assert casted.column("null_float").null_count == 1
    assert casted.column("overflow_int").to_pylist() == [10, 20, 40_000, 50]

    # Pandas DataFrame 다운캐스팅 (nullable Int64 & NaN 포함 float64)
    df = pd.DataFrame(
        {
            "row_id": np.array([0, 1, 2], dtype=np.int64),
            "clicked": np.array([0, 1, 0], dtype=np.int64),
            "ID": ["x", "y", "z"],
            "nullable_int": pd.Series([1, pd.NA, 10], dtype="Int64"),
            "float_with_nan": np.array([1.0, np.nan, 3.0], dtype=np.float64),
        }
    )
    casted_df = downcast_dataframe(df)
    assert casted_df["row_id"].dtype == np.int64
    assert casted_df["clicked"].dtype == np.int8
    assert str(casted_df["nullable_int"].dtype) == "Int8"
    assert casted_df["nullable_int"].isna().sum() == 1
    assert casted_df["float_with_nan"].dtype == np.float32
    assert np.isnan(casted_df["float_with_nan"].iloc[1])


def test_build_diet_artifacts_streaming_and_alignment(tmp_path: Path) -> None:
    """다중 Row-Group 스트리밍 분리, seq 분리, row_id/ID 보존, Train/Test dtype 일치를 검증합니다."""
    train_path, test_path, train_df, test_df = _create_synthetic_parquets(tmp_path)
    out_dir = tmp_path / "processed"

    # 배치 크기 35로 강제 분할하여 다중 배치 ParquetWriter 스트리밍 검증
    manifest = build_diet_artifacts(
        train_path=train_path,
        test_path=test_path,
        output_dir=out_dir,
        batch_size=35,
        compression="zstd",
        compression_level=3,
    )

    assert manifest["train_rows"] == 120
    assert manifest["test_rows"] == 40
    assert Path(manifest["manifest_path"]).exists()

    train_tab_pf = pq.ParquetFile(out_dir / DEFAULT_CONTRACT.train_tabular_filename)
    test_tab_pf = pq.ParquetFile(out_dir / DEFAULT_CONTRACT.test_tabular_filename)
    train_seq_pf = pq.ParquetFile(out_dir / DEFAULT_CONTRACT.train_seq_filename)
    test_seq_pf = pq.ParquetFile(out_dir / DEFAULT_CONTRACT.test_seq_filename)

    # 1) seq 분리 및 기본 컬럼 계약 확인
    assert "seq" not in train_tab_pf.schema_arrow.names
    assert "seq" not in test_tab_pf.schema_arrow.names
    assert train_seq_pf.schema_arrow.names == ["row_id", "seq"]
    assert test_seq_pf.schema_arrow.names == ["ID", "seq"]

    # 2) Train/Test 정형 테이블 스키마 및 공통 피처 dtype 일치 확인
    raw_test_pf = pq.ParquetFile(test_path)
    assert train_tab_pf.schema_arrow.field("row_id").type == pa.int64()
    assert train_tab_pf.schema_arrow.field("clicked").type == pa.int8()
    assert test_tab_pf.schema_arrow.field("ID").type == raw_test_pf.schema_arrow.field("ID").type

    shared_features = [
        c for c in train_tab_pf.schema_arrow.names if c not in ("row_id", "clicked")
    ]
    for col in shared_features:
        assert train_tab_pf.schema_arrow.field(col).type == test_tab_pf.schema_arrow.field(col).type

    # feat_c_1은 train(0~99)만 보면 int8이지만 test(500~999) 때문에 int16으로 통합되어야 함
    assert train_tab_pf.schema_arrow.field("feat_c_1").type == pa.int16()
    assert train_tab_pf.schema_arrow.field("feat_a_1").type == pa.float32()
    assert train_tab_pf.schema_arrow.field("feat_b_1").type == pa.int8()

    # 3) row_id 연속성, ID 순서 및 값/NULL 정합성 확인
    diet_train = pq.read_table(out_dir / DEFAULT_CONTRACT.train_tabular_filename).to_pandas()
    diet_test = pq.read_table(out_dir / DEFAULT_CONTRACT.test_tabular_filename).to_pandas()
    diet_train_seq = pq.read_table(out_dir / DEFAULT_CONTRACT.train_seq_filename).to_pandas()
    diet_test_seq = pq.read_table(out_dir / DEFAULT_CONTRACT.test_seq_filename).to_pandas()

    assert np.array_equal(diet_train["row_id"].to_numpy(), np.arange(120, dtype=np.int64))
    assert np.array_equal(diet_train_seq["row_id"].to_numpy(), np.arange(120, dtype=np.int64))
    assert diet_test["ID"].tolist() == test_df["ID"].tolist()
    assert diet_test_seq["ID"].tolist() == test_df["ID"].tolist()
    assert diet_train_seq["seq"].tolist() == train_df["seq"].tolist()
    assert diet_test_seq["seq"].tolist() == test_df["seq"].tolist()

    assert np.array_equal(diet_train["clicked"].to_numpy(), train_df["clicked"].to_numpy())
    assert np.array_equal(diet_test["feat_c_1"].to_numpy(), test_df["feat_c_1"].to_numpy())
    assert np.allclose(
        diet_train["feat_a_1"].to_numpy(),
        train_df["feat_a_1"].to_numpy(),
        rtol=1e-6,
        equal_nan=True,
    )

    # 통계 메타데이터 조회 함수 단독 검증
    stats = collect_parquet_column_stats([train_path, test_path], columns=["feat_c_1"])
    assert stats["feat_c_1"]["min"] < 100
    assert stats["feat_c_1"]["max"] >= 500
    inferred = infer_safe_dtypes(train_tab_pf.schema_arrow, column_stats=stats)
    assert inferred["feat_c_1"] == pa.int16()


def test_build_cv_folds_determinism_oot_and_contract(tmp_path: Path) -> None:
    """시드 재현성, OOT 마킹, 희소 그룹 fallback 및 folds 계약 검증을 테스트합니다."""
    train_path, test_path, train_df, _ = _create_synthetic_parquets(
        tmp_path,
        train_days=[1, 2, 3, 4, 5, 6],
        test_days=[6, 7],  # unseen day 7 포함 -> OOT 감지 및 train 최대 요일(6) 마킹
    )
    folds_path = tmp_path / "folds.parquet"
    report_path = tmp_path / "cv_summary.json"

    folds_1, summary_1 = build_cv_folds(
        train_path=train_path,
        test_path=test_path,
        output_path=folds_path,
        n_splits=5,
        seed=42,
        report_path=report_path,
    )
    folds_2, _ = build_cv_folds(
        train_path=train_path,
        test_path=test_path,
        output_path=None,
        n_splits=5,
        seed=42,
    )
    folds_diff_seed, _ = build_cv_folds(
        train_path=train_path,
        test_path=test_path,
        output_path=None,
        n_splits=5,
        seed=99,
    )

    # 1) 결정론성 검증
    assert np.array_equal(folds_1["fold_id"].to_numpy(), folds_2["fold_id"].to_numpy())
    assert np.array_equal(folds_1["is_oot_val"].to_numpy(), folds_2["is_oot_val"].to_numpy())
    assert not np.array_equal(folds_1["fold_id"].to_numpy(), folds_diff_seed["fold_id"].to_numpy())

    # 2) OOT 감지 및 마킹 정확성 검증
    assert summary_1["temporal_regime"] == "out_of_time_unseen_days"
    assert summary_1["unseen_test_days"] == [7]
    assert summary_1["oot_marked"] is True
    assert summary_1["oot_day"] == 6
    expected_oot = (train_df["day_of_week"] == 6).to_numpy()
    assert np.array_equal(folds_1["is_oot_val"].to_numpy(), expected_oot)

    # 3) 디스크 로드 및 validate_folds_contract 검증
    loaded = load_folds(folds_path, validate=True, n_rows=120, n_splits=5)
    assert list(loaded.columns) == ["row_id", "fold_id", "is_oot_val"]
    assert report_path.exists()

    # 4) 희소 (day_of_week, clicked) 1개 샘플 그룹 fallback 안전성 검증
    rare_df = train_df.copy()
    rare_df.loc[0, "day_of_week"] = 99  # 단 1행만 있는 요일
    rare_df.loc[0, "clicked"] = 1
    rare_train_path = tmp_path / "rare_train.parquet"
    pq.write_table(pa.Table.from_pandas(rare_df, preserve_index=False), rare_train_path)

    rare_folds, rare_summary = build_cv_folds(
        train_path=rare_train_path,
        test_path=test_path,
        n_splits=5,
        seed=42,
    )
    assert rare_summary["stratification_fallback"] == "merged_rare_to_clicked"
    assert validate_folds_contract(rare_folds, n_rows=120, n_splits=5)["valid"] is True

    # 5) 문자열 day_of_week("1"~"7") + in-time 분포 케이스 검증
    str_train_df = train_df.copy()
    str_train_df["day_of_week"] = [str((i % 7) + 1) for i in range(len(str_train_df))]
    str_test_df = str_train_df.iloc[:35].drop(columns=["clicked"]).copy()
    str_train_path = tmp_path / "str_train.parquet"
    str_test_path = tmp_path / "str_test.parquet"
    pq.write_table(pa.Table.from_pandas(str_train_df, preserve_index=False), str_train_path)
    pq.write_table(pa.Table.from_pandas(str_test_df, preserve_index=False), str_test_path)

    str_folds, str_summary = build_cv_folds(
        train_path=str_train_path,
        test_path=str_test_path,
        n_splits=5,
        seed=42,
    )
    assert str_summary["temporal_regime"] == "in_time_similar"
    assert str_summary["oot_day"] == "7"
    assert validate_folds_contract(str_folds, n_rows=120, n_splits=5)["valid"] is True

    # 6) validate_folds_contract 훼손 입력 실패 검증
    bad_missing = loaded.drop(columns=["is_oot_val"])
    with pytest.raises(ValueError):
        validate_folds_contract(bad_missing, n_rows=120, n_splits=5)

    bad_row_id = loaded.copy()
    bad_row_id.loc[5, "row_id"] = 999
    with pytest.raises(ValueError):
        validate_folds_contract(bad_row_id, n_rows=120, n_splits=5)

    bad_fold_range = loaded.copy()
    bad_fold_range.loc[0, "fold_id"] = 5
    with pytest.raises(ValueError):
        validate_folds_contract(bad_fold_range, n_rows=120, n_splits=5)

    bad_nan = loaded.copy()
    bad_nan["fold_id"] = bad_nan["fold_id"].astype("float32")
    bad_nan.loc[0, "fold_id"] = np.nan
    with pytest.raises(ValueError):
        validate_folds_contract(bad_nan, n_rows=120, n_splits=5)


def test_configs_and_notebooks_structure() -> None:
    """YAML 설정 파일과 Colab 노트북의 JSON 및 셀 배치 규칙을 검증합니다."""
    root = Path(__file__).resolve().parents[2]

    diet_cfg = yaml.safe_load((root / "configs" / "data_diet.yaml").read_text(encoding="utf-8"))
    cv_cfg = yaml.safe_load((root / "configs" / "cv_strategy.yaml").read_text(encoding="utf-8"))
    assert diet_cfg["streaming"]["batch_size"] > 0
    assert diet_cfg["streaming"]["compression"] == "zstd"
    assert cv_cfg["cv"]["n_splits"] == 5
    assert cv_cfg["cv"]["seed"] == 42

    for nb_name in ("01_eda_profiling.ipynb", "02_data_diet_and_cv.ipynb"):
        nb_data = json.loads((root / "notebooks" / nb_name).read_text(encoding="utf-8"))
        cells = nb_data["cells"]
        assert len(cells) >= 4
        # 모든 코드 셀 바로 뒤에는 결과 해석 마크다운 셀이 와야 함
        for idx, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                assert idx + 1 < len(cells), f"{nb_name} 마지막 코드 셀 뒤에 해석 셀이 없습니다."
                assert (
                    cells[idx + 1]["cell_type"] == "markdown"
                ), f"{nb_name} cell[{idx}] 코드 셀 바로 뒤에 마크다운 셀이 필요합니다."
        assert cells[-1]["cell_type"] == "markdown"
