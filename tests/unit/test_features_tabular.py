"""정형 CTR 피처 엔지니어링(교호작용·행 통계·타겟 인코딩·통합 파이프라인) 누수·무결성 단위 테스트."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.metrics import roc_auc_score

from toss_ctr.features.history_agg import build_row_statistics
from toss_ctr.features.interactions import (
    FoldConditionalStatsEncoder,
    FoldFrequencyEncoder,
    build_interaction_features,
)
from toss_ctr.features.tabular_pipeline import TabularFeaturePipeline, join_seq_summary
from toss_ctr.features.target_encoding import OutOfFoldTargetEncoder


def _make_synthetic_tabular_df(
    n_rows: int = 150,
    seed: int = 42,
    is_test: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray | list] = {
        "gender": rng.integers(0, 3, size=n_rows, dtype=np.int16),
        "age_group": rng.integers(1, 8, size=n_rows, dtype=np.int16),
        "inventory_id": rng.integers(10, 22, size=n_rows, dtype=np.int16),
        "day_of_week": (np.arange(n_rows) % 6 + 1).astype(np.int16),
        "hour": rng.integers(0, 24, size=n_rows, dtype=np.int16),
        "l_feat_14": rng.integers(100, 112, size=n_rows, dtype=np.int32),
        "feat_a_1": rng.normal(0.0, 1.0, size=n_rows).astype(np.float32),
        "feat_a_2": rng.normal(1.0, 0.5, size=n_rows).astype(np.float32),
        "feat_b_1": rng.uniform(0.0, 10.0, size=n_rows).astype(np.float32),
        "feat_c_1": rng.integers(0, 100, size=n_rows, dtype=np.int16),
        "feat_d_1": rng.normal(-2.0, 2.0, size=n_rows).astype(np.float32),
        "feat_e_1": rng.integers(0, 2, size=n_rows, dtype=np.int8),
        "history_a_1": rng.uniform(0.0, 1.0, size=n_rows).astype(np.float32),
        "history_a_2": rng.uniform(0.5, 2.0, size=n_rows).astype(np.float32),
    }
    # 일부 행에 결측(특히 0번 행 전체 feat_a 결측 포함) 주입
    data["feat_a_1"][0] = np.nan
    data["feat_a_2"][0] = np.nan
    data["feat_a_1"][::17] = np.nan
    data["feat_b_1"][::19] = np.nan
    data["history_a_2"][::23] = np.nan

    df = pd.DataFrame(data)
    if is_test:
        df.insert(0, "ID", [f"TEST_{i:05d}" for i in range(n_rows)])
    else:
        df.insert(0, "row_id", np.arange(n_rows, dtype=np.int64))
        df["clicked"] = (rng.random(n_rows) < 0.2).astype(np.int8)
    return df


def test_interactions_deterministic_and_missing_safe() -> None:
    """교호작용 키 생성이 결정론적이며 미존재 컬럼과 결측치를 안전하게 처리하는지 검증합니다."""
    train_df = _make_synthetic_tabular_df(60, seed=1)
    test_df = _make_synthetic_tabular_df(30, seed=2, is_test=True)

    # 첫 행 동일한 페어 및 결측 페어 강제 일치
    test_df.loc[0, ["inventory_id", "l_feat_14"]] = train_df.loc[0, ["inventory_id", "l_feat_14"]]
    train_nan = train_df.copy()
    train_nan["inventory_id"] = train_nan["inventory_id"].astype(np.float32)
    train_nan.loc[1, "inventory_id"] = np.nan
    test_nan = test_df.copy()
    test_nan["inventory_id"] = test_nan["inventory_id"].astype(np.float32)
    test_nan.loc[1, "inventory_id"] = np.nan

    ix_tr = build_interaction_features(train_nan)
    ix_te = build_interaction_features(test_nan)

    expected_cols = {
        "ix_inventory_id__l_feat_14",
        "ix_inventory_id__hour",
        "ix_age_group__gender",
        "ix_day_of_week__hour",
    }
    assert set(ix_tr.columns) == expected_cols
    assert set(ix_te.columns) == expected_cols
    assert (ix_tr.dtypes == np.int64).all()
    assert (ix_tr >= 0).all().all()
    assert ix_tr.loc[0, "ix_inventory_id__l_feat_14"] == ix_te.loc[0, "ix_inventory_id__l_feat_14"]

    # 문자열 출력 모드 및 일부 컬럼 누락 시 안전 건너뛰기 검증
    partial_df = train_df.drop(columns=["gender"])
    ix_str = build_interaction_features(partial_df, output_format="string")
    assert "ix_age_group__gender" not in ix_str.columns
    assert "ix_inventory_id__l_feat_14" in ix_str.columns
    assert pd.api.types.is_string_dtype(ix_str["ix_inventory_id__l_feat_14"].dtype)


def test_interactions_float_int_elementwise_hash_consistency() -> None:
    """혼합 float 컬럼([1.0, 2.5])에서도 정수형 float 원소 1.0의 해시가 [1, 2] 및 [1.0, 2.0]과 일치함을 검증합니다."""
    df_int = pd.DataFrame({"inventory_id": pd.Series([1, 2], dtype=np.int64), "hour": [10, 10]})
    df_float_clean = pd.DataFrame(
        {"inventory_id": pd.Series([1.0, 2.0], dtype=np.float64), "hour": [10.0, 10.0]}
    )
    df_float_mixed = pd.DataFrame(
        {"inventory_id": pd.Series([1.0, 2.5], dtype=np.float64), "hour": [10.0, 10.5]}
    )
    df_float_nullable = pd.DataFrame(
        {"inventory_id": pd.Series([1.0, pd.NA], dtype="Float32"), "hour": pd.Series([10, 10], dtype="Int16")}
    )

    h_int = build_interaction_features(df_int, interaction_pairs=[("inventory_id", "hour")])
    h_clean = build_interaction_features(df_float_clean, interaction_pairs=[("inventory_id", "hour")])
    h_mixed = build_interaction_features(df_float_mixed, interaction_pairs=[("inventory_id", "hour")])
    h_null = build_interaction_features(df_float_nullable, interaction_pairs=[("inventory_id", "hour")])

    val_int = h_int.loc[0, "ix_inventory_id__hour"]
    assert h_clean.loc[0, "ix_inventory_id__hour"] == val_int
    assert h_mixed.loc[0, "ix_inventory_id__hour"] == val_int
    assert h_null.loc[0, "ix_inventory_id__hour"] == val_int
    assert h_mixed.loc[1, "ix_inventory_id__hour"] != h_clean.loc[1, "ix_inventory_id__hour"]


def test_fold_frequency_encoder_zero_test_leakage() -> None:
    """FoldFrequencyEncoder가 테스트 데이터 빈도를 전혀 참조하지 않고 미등장 범주를 0으로 매핑하는지 검증합니다."""
    train_df = pd.DataFrame(
        {
            "cat": ["A", "A", "A", "B", "B", None],
            "num": [10, 10, 20, 30, 30, 30],
        }
    )
    test_small = pd.DataFrame(
        {
            "cat": ["A", "UNSEEN", None],
            "num": [10, 999, 30],
        }
    )
    test_skewed = pd.DataFrame(
        {
            "cat": ["A", "UNSEEN", None] + ["B"] * 50 + ["UNSEEN"] * 100,
            "num": [10, 999, 30] + [20] * 150,
        }
    )

    enc = FoldFrequencyEncoder(columns=["cat", "num"], add_log1p=True).fit(train_df)
    tr_out = enc.transform(train_df)
    te_small_out = enc.transform(test_small)
    te_skewed_out = enc.transform(test_skewed)

    assert tr_out["cat_freq"].tolist() == [3, 3, 3, 2, 2, 1]
    assert te_small_out["cat_freq"].tolist() == [3, 0, 1]
    assert te_small_out["num_freq"].tolist() == [2, 0, 3]
    assert te_small_out["cat_freq_log1p"].iloc[1] == pytest.approx(0.0)
    # test 분포가 크게 달라져도 처음 3행의 매핑 빈도는 train 기준 그대로 불변
    pd.testing.assert_frame_equal(te_small_out, te_skewed_out.iloc[:3].reset_index(drop=True))


def test_fold_conditional_stats_encoder() -> None:
    """FoldConditionalStatsEncoder의 고유 파트너 수 및 조건부 점유율 계산과 미등장 조합 0 폴백을 검증합니다."""
    train_df = pd.DataFrame(
        {
            "inventory_id": [1, 1, 1, 2, 2],
            "l_feat_14": [100, 100, 200, 300, 300],
        }
    )
    eval_df = pd.DataFrame(
        {
            "inventory_id": [1, 1, 999],
            "l_feat_14": [100, 999, 100],
        }
    )
    enc = FoldConditionalStatsEncoder(pair_specs=[("inventory_id", "l_feat_14")]).fit(train_df)
    out = enc.transform(eval_df)

    assert out["inventory_id__nunique_l_feat_14"].tolist() == [2, 2, 0]
    assert out["inventory_id_l_feat_14__share_of_inventory_id"].iloc[0] == pytest.approx(2.0 / 3.0)
    assert out["inventory_id_l_feat_14__share_of_inventory_id"].iloc[1] == pytest.approx(0.0)
    assert out["inventory_id_l_feat_14__share_of_inventory_id"].iloc[2] == pytest.approx(0.0)


def test_build_row_statistics_row_isolation_and_no_warnings() -> None:
    """build_row_statistics가 타 행에 전혀 영향받지 않고 전체-NaN 행에서도 경고 없이 동작하는지 검증합니다."""
    df_base = _make_synthetic_tabular_df(150, seed=7)
    df_modified = df_base.copy()
    # 0번 행 이외의 모든 행을 극단치와 NaN으로 변경해 교차 행 누수 여부 검사
    df_modified.iloc[1:, df_modified.columns.get_loc("feat_a_1")] = 1e6
    df_modified.iloc[1:, df_modified.columns.get_loc("history_a_1")] = -999.0
    df_modified.iloc[1:, df_modified.columns.get_loc("feat_b_1")] = np.nan

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stats_base = build_row_statistics(df_base)
        stats_mod = build_row_statistics(df_modified)

    runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warns, f"예기치 않은 RuntimeWarning 발생: {runtime_warns}"

    # 0번 행의 모든 feat_a_*가 NaN인 경우 검증
    assert stats_base.loc[0, "feat_a_null_cnt"] == 2
    assert stats_base.loc[0, "feat_a_all_null"] == 1
    assert np.isnan(stats_base.loc[0, "feat_a_row_mean"])
    assert np.isnan(stats_base.loc[0, "feat_a_row_std"])
    assert (stats_base.loc[0, "domain_all_null_mask"] & 1) == 1
    assert "history_a_row_mean" in stats_base.columns
    assert "history_a_1_minus_feat_a_1" in stats_base.columns
    assert "history_a_1_div_feat_a_1" in stats_base.columns

    # 타 행을 극단적으로 변조해도 0번 행 통계는 비트 단위까지 완전히 동일
    pd.testing.assert_series_equal(stats_base.iloc[0], stats_mod.iloc[0])


def test_oof_target_encoder_zero_leakage_unique_ids() -> None:
    """100% 고유 ID와 완전 무작위 이진 타겟에서 OutOfFoldTargetEncoder의 자기 타겟 누수가 0임을 검증합니다."""
    n_rows = 150
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=n_rows, dtype=np.int8)
    folds = np.arange(n_rows, dtype=np.int64) % 5

    # 행마다 100% 고유한 키 + 반복 범주 키 동시 테스트
    df = pd.DataFrame(
        {
            "unique_id": [f"UID_{i:04d}" for i in range(n_rows)],
            "group_cat": [f"G_{i % 6}" if i % 15 != 0 else None for i in range(n_rows)],
        }
    )

    te = OutOfFoldTargetEncoder(columns=["unique_id", "group_cat"], smoothing=20.0)
    oof_encoded = te.fit_transform(df, y, folds)

    assert list(oof_encoded.columns) == ["unique_id_te", "group_cat_te"]
    assert oof_encoded["unique_id_te"].dtype == np.float32
    assert oof_encoded["group_cat_te"].dtype == np.float32
    assert not oof_encoded.isna().any().any()

    # 100% 고유 ID는 모든 폴드에서 오직 (fold != k)의 평균(oof_prior_k)만 받아야 함
    for k in range(5):
        expected_prior_k = float(y[folds != k].mean())
        actual_k = oof_encoded.loc[folds == k, "unique_id_te"].to_numpy()
        np.testing.assert_allclose(actual_k, expected_prior_k, rtol=0.0, atol=1e-6)

    # 무작위 타겟 + 고유 ID 조합에서 과적합 AUC가 절대 0.55를 넘지 않음 (이론상 <= 0.50)
    leakage_auc = float(roc_auc_score(y, oof_encoded["unique_id_te"]))
    assert leakage_auc <= 0.55, f"OOF 타겟 인코딩 누수 의심: AUC={leakage_auc:.4f}"

    # 반복 범주(group_cat)의 수동 M-estimate 계산 일치 확인 (0번 행: fold=0, group_cat=None)
    mask_oof_0 = folds != 0
    mask_na_oof_0 = mask_oof_0 & df["group_cat"].isna().to_numpy()
    cnt_oof = int(mask_na_oof_0.sum())
    sum_oof = int(y[mask_na_oof_0].sum())
    prior_oof_0 = float(y[mask_oof_0].mean())
    expected_row0 = (sum_oof + 20.0 * prior_oof_0) / (cnt_oof + 20.0)
    assert oof_encoded.loc[0, "group_cat_te"] == pytest.approx(expected_row0, abs=1e-6)

    # transform(X_eval): 전체 학습 통계 + 미등장 범주 global_prior 폴백 검증
    eval_df = pd.DataFrame(
        {
            "unique_id": ["UID_0000", "UNSEEN_9999"],
            "group_cat": ["G_1", "UNSEEN_GRP"],
        }
    )
    eval_out = te.transform(eval_df)
    assert eval_out.dtypes.eq(np.float32).all()
    assert eval_out.loc[1, "unique_id_te"] == pytest.approx(float(y.mean()), abs=1e-6)
    assert eval_out.loc[1, "group_cat_te"] == pytest.approx(float(y.mean()), abs=1e-6)


def test_tabular_feature_pipeline_fold_isolation_and_seq_join(tmp_path: Path) -> None:
    """TabularFeaturePipeline의 seq 요약 조인, 폴드 격리 인코딩 및 OOF FE 일관성을 검증합니다."""
    train_df = _make_synthetic_tabular_df(150, seed=101)
    test_df = _make_synthetic_tabular_df(40, seed=202, is_test=True)
    folds_df = pd.DataFrame(
        {
            "row_id": train_df["row_id"],
            "fold_id": (np.arange(150, dtype=np.int8) % 5),
            "is_oot_val": np.zeros(150, dtype=bool),
        }
    )

    seq_tr = pd.DataFrame(
        {
            "row_id": train_df["row_id"],
            "len": np.arange(150, dtype=np.int32) % 20 + 1,
            "seq_nunique": np.arange(150, dtype=np.int32) % 7 + 1,
        }
    )
    seq_te = pd.DataFrame(
        {
            "ID": test_df["ID"],
            "len": np.arange(40, dtype=np.int32) % 20 + 1,
            "seq_nunique": np.arange(40, dtype=np.int32) % 7 + 1,
        }
    )
    seq_tr_path = tmp_path / "seq_summary_train.parquet"
    seq_te_path = tmp_path / "seq_summary_test.parquet"
    pq.write_table(pa.Table.from_pandas(seq_tr, preserve_index=False), seq_tr_path)
    pq.write_table(pa.Table.from_pandas(seq_te, preserve_index=False), seq_te_path)

    pipeline = TabularFeaturePipeline(seed=42)
    bundle = pipeline.fit_transform_fold(
        train_df=train_df,
        y_train=train_df["clicked"],
        folds=folds_df,
        fold_id=0,
        test_df=test_df,
        train_seq_summary=seq_tr_path,
        test_seq_summary=seq_te_path,
    )

    assert len(bundle.X_train) == 120
    assert len(bundle.X_valid) == 30
    assert bundle.X_test is not None and len(bundle.X_test) == 40
    assert set(bundle.train_row_ids).isdisjoint(set(bundle.valid_row_ids))
    assert "clicked" not in bundle.feature_names
    assert "row_id" not in bundle.feature_names
    assert "ID" not in bundle.feature_names
    assert "seq_len" in bundle.feature_names
    assert "seq_nunique" in bundle.feature_names
    assert "inventory_id_te" in bundle.feature_names
    assert "ix_inventory_id__l_feat_14_freq" in bundle.feature_names

    # run_oof_fe 출력 shape 및 피처 정렬 일치 검증
    X_oof, X_test_oof, oof_cols = pipeline.run_oof_fe(
        train_df=train_df,
        y_train=train_df["clicked"],
        folds=folds_df,
        test_df=test_df,
        train_seq_summary=seq_tr_path,
        test_seq_summary=seq_te_path,
    )
    assert len(X_oof) == 150
    assert X_test_oof is not None and len(X_test_oof) == 40
    assert list(X_oof.columns) == oof_cols
    assert list(X_test_oof.columns) == oof_cols

    # seq_summary 중복 키가 있으면 행 폭발 방지를 위해 ValueError 발생 확인
    dup_seq = pd.concat([seq_tr, seq_tr.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="중복"):
        join_seq_summary(train_df, dup_seq, key_col="row_id")
