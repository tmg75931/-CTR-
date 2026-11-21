"""DeepCTR 전처리, 설정, 표준 아티팩트 저장 및(Torch 가용 시) 모델 학습 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toss_ctr.models.deep_ctr import (
    TORCH_AVAILABLE,
    DeepCTRConfig,
    SeqDCNv2CTRModel,
    _validate_parquet_row_id_alignment,
    compute_pos_weight_50_50,
    infer_deep_feature_columns,
    load_deep_ctr_config,
    logits_to_probs,
    require_torch,
    save_deep_predictions,
    train_deep_ctr_oof,
    train_deep_ctr_oof_from_parquet,
    validate_deep_prediction_artifacts,
)
from toss_ctr.models.nn_preprocessing import (
    CAT_MISSING_ID,
    CAT_UNK_ID,
    FoldCategoricalIndexer,
    FoldNumericPreprocessor,
    signed_log1p,
)


def test_load_deep_seq_dcn_yaml_config() -> None:
    """`configs/deep_seq_dcn.yaml`이 명세된 하이퍼파라미터로 로드되는지 검증합니다."""
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "deep_seq_dcn.yaml"
    cfg = load_deep_ctr_config(cfg_path)

    assert cfg.max_seq_len == 100
    assert cfg.min_freq == 5
    assert cfg.max_vocab_size == 100_000
    assert cfg.seq_embed_dim == 32
    assert cfg.cat_embed_dim == 16
    assert cfg.cross_layers == 3
    assert cfg.mlp_dims == [256, 128, 64]
    assert cfg.batch_size == 2048
    assert cfg.epochs == 8
    assert cfg.lr == pytest.approx(1.0e-3)
    assert cfg.use_amp is True
    assert cfg.seed == 42
    assert cfg.use_pos_weight_50_50 is True


def test_lazy_torch_import_guard() -> None:
    """PyTorch 미설치 환경에서도 전처리/설정은 import되며 torch 호출 시 명확한 안내가 나옵니다."""
    assert isinstance(TORCH_AVAILABLE, bool)
    if not TORCH_AVAILABLE:
        with pytest.raises(ImportError, match="PyTorch"):
            require_torch()
        with pytest.raises(ImportError, match="PyTorch"):
            SeqDCNv2CTRModel(
                seq_vocab_size=10,
                cat_vocab_sizes=[5],
                num_numeric_features=2,
            )


def test_fold_categorical_indexer_no_leakage_and_consistency() -> None:
    """FoldCategoricalIndexer의 MISSING=0, UNK=1, min_freq 및 int/float 일관성을 검증합니다."""
    train_df = pd.DataFrame(
        {
            "inventory_id": [10, 10, 20, 20, 99, None],
            "gender": ["M", "M", "F", "F", "X", ""],
        }
    )
    val_df = pd.DataFrame(
        {
            "inventory_id": [10.0, 20.0, 99.0, 777.0, np.nan],
            "gender": ["M", "NEW_GENDER", "X", None, "  "],
        }
    )

    indexer = FoldCategoricalIndexer(cat_cols=["inventory_id", "gender"], min_freq=2).fit(train_df)
    tr_ids = indexer.transform(train_df)
    va_ids = indexer.transform(val_df)

    assert tr_ids.dtype == np.int32
    assert va_ids.dtype == np.int32
    assert tr_ids.shape == (len(train_df), 2)
    assert va_ids.shape == (len(val_df), 2)

    # 결측치 및 빈 문자열은 MISSING_ID(0)
    assert tr_ids[5, 0] == CAT_MISSING_ID
    assert tr_ids[5, 1] == CAT_MISSING_ID
    assert va_ids[4, 0] == CAT_MISSING_ID
    assert va_ids[3, 1] == CAT_MISSING_ID
    assert va_ids[4, 1] == CAT_MISSING_ID

    # min_freq=2 미만(99, X) 및 val 전용(777, NEW_GENDER)은 모두 UNK_ID(1)
    assert tr_ids[4, 0] == CAT_UNK_ID
    assert tr_ids[4, 1] == CAT_UNK_ID
    assert va_ids[2, 0] == CAT_UNK_ID
    assert va_ids[3, 0] == CAT_UNK_ID
    assert va_ids[1, 1] == CAT_UNK_ID
    assert va_ids[2, 1] == CAT_UNK_ID

    # int(10)과 float(10.0)이 동일한 ID로 매핑되어야 함
    assert va_ids[0, 0] == tr_ids[0, 0] >= 2
    assert va_ids[1, 0] == tr_ids[2, 0] >= 2
    assert va_ids[0, 1] == tr_ids[0, 1] >= 2

    # partial_fit + finalize_fit 및 직렬화 일관성 검증
    chunk_indexer = FoldCategoricalIndexer(cat_cols=["inventory_id", "gender"], min_freq=2)
    chunk_indexer.partial_fit(train_df.iloc[:3])
    chunk_indexer.partial_fit(train_df.iloc[3:])
    chunk_indexer.finalize_fit()
    assert np.array_equal(chunk_indexer.transform(val_df), va_ids)

    restored = FoldCategoricalIndexer.from_dict(indexer.to_dict())
    assert restored.vocab_sizes == indexer.vocab_sizes
    assert np.array_equal(restored.transform(val_df), va_ids)


def test_fold_numeric_preprocessor_signed_log1p_missing_and_zero_std() -> None:
    """FoldNumericPreprocessor의 signed_log1p, zero-std 방어, 무한대/결측 마스크, partial_fit 동등성을 검증합니다."""
    raw_arr = np.array([-9.0, 0.0, 9.0, np.nan, np.inf])
    sl = signed_log1p(raw_arr)
    assert sl[0] == pytest.approx(-np.log1p(9.0))
    assert sl[1] == pytest.approx(0.0)
    assert sl[2] == pytest.approx(np.log1p(9.0))

    train_df = pd.DataFrame(
        {
            "num_var": [0.0, 10.0, -10.0, np.nan, np.inf],
            "num_const": [5.0, 5.0, 5.0, 5.0, np.nan],
            "num_all_na": [np.nan, np.nan, -np.inf, np.inf, np.nan],
        }
    )
    val_df = pd.DataFrame(
        {
            "num_var": [1000.0, np.nan, -np.inf],
            "num_const": [5.0, 9.0, np.nan],
            "num_all_na": [1.0, np.nan, 0.0],
        }
    )

    prep = FoldNumericPreprocessor(
        num_cols=["num_var", "num_const", "num_all_na"],
        apply_signed_log1p=True,
    ).fit(train_df)

    # zero-std 및 전체 결측 컬럼의 std는 1.0으로 안전 방어되어야 함
    assert prep.std_[1] == pytest.approx(1.0)
    assert prep.std_[2] == pytest.approx(1.0)
    assert prep.mean_[2] == pytest.approx(0.0)

    tr_x, tr_mask = prep.transform(train_df)
    va_x, va_mask = prep.transform(val_df)

    assert tr_x.dtype == np.float32
    assert tr_mask.dtype == np.float32
    assert np.all(np.isfinite(tr_x))
    assert np.all(np.isfinite(va_x))

    # NaN / +inf / -inf 위치는 missing_mask=1.0, scaled=0.0
    assert tr_mask[3, 0] == 1.0 and tr_x[3, 0] == 0.0
    assert tr_mask[4, 0] == 1.0 and tr_x[4, 0] == 0.0
    assert va_mask[1, 0] == 1.0 and va_x[1, 0] == 0.0
    assert va_mask[2, 0] == 1.0 and va_x[2, 0] == 0.0
    assert va_mask[0, 0] == 0.0

    # partial_fit(청크 2+3행) 결과가 일괄 fit 결과와 일치해야 함
    chunk_prep = FoldNumericPreprocessor(
        num_cols=["num_var", "num_const", "num_all_na"],
        apply_signed_log1p=True,
    )
    chunk_prep.partial_fit(train_df.iloc[:2])
    chunk_prep.partial_fit(train_df.iloc[2:])
    chunk_prep.finalize_fit()
    c_va_x, c_va_mask = chunk_prep.transform(val_df)
    assert np.allclose(c_va_x, va_x, atol=1e-6)
    assert np.array_equal(c_va_mask, va_mask)


def test_feature_split_pos_weight_and_prediction_artifacts(tmp_path: Path) -> None:
    """피처 분리, 50:50 pos_weight 계산 및 OOF/Test 표준 Parquet 저장·검증을 확인합니다."""
    sample_df = pd.DataFrame(
        {
            "row_id": [0, 1],
            "clicked": [0, 1],
            "seq": ["1,2", "3,4"],
            "gender": [1, 2],
            "age_group": [3, 4],
            "inventory_id": [10, 20],
            "day_of_week": [1, 2],
            "hour": [12, 13],
            "l_feat_14": [100, 101],
            "feat_a_1": [0.1, 0.2],
            "history_a_1": [0.5, 0.6],
        }
    )
    cat_cols, num_cols = infer_deep_feature_columns(sample_df)
    assert set(cat_cols) == {"gender", "age_group", "inventory_id", "day_of_week", "hour", "l_feat_14"}
    assert num_cols == ["feat_a_1", "history_a_1"]

    y = np.array([0, 0, 0, 0, 1], dtype=np.int8)
    assert compute_pos_weight_50_50(y) == pytest.approx(4.0)
    probs = logits_to_probs(np.array([-100.0, 0.0, 100.0]))
    assert probs[1] == pytest.approx(0.5)
    assert np.all((probs > 0.0) & (probs < 1.0))

    # 셔플된 row_id도 save_deep_predictions가 0..N-1 오름차순으로 정렬해 저장해야 함
    oof_path = tmp_path / "oof_deep_seq_dcn.parquet"
    test_path = tmp_path / "test_deep_seq_dcn.parquet"
    save_deep_predictions(
        oof_path=oof_path,
        row_ids=np.array([2, 0, 1], dtype=np.int64),
        fold_ids=np.array([2, 0, 1], dtype=np.int8),
        clicked=np.array([1, 0, 0], dtype=np.int8),
        oof_raw_logits=np.array([1.2, -0.5, 0.1], dtype=np.float32),
        test_path=test_path,
        test_ids=np.array(["ID_B", "ID_A"]),
        test_raw_logits=np.array([-0.3, 0.7], dtype=np.float32),
    )

    check = validate_deep_prediction_artifacts(
        oof_path=oof_path,
        test_path=test_path,
        expected_n_train=3,
        expected_n_test=2,
        expected_test_ids=np.array(["ID_B", "ID_A"]),
    )
    assert check["valid"] is True
    assert check["oof_rows"] == 3
    assert check["test_rows"] == 2

    saved_oof = pq.read_table(oof_path).to_pandas()
    assert saved_oof["row_id"].tolist() == [0, 1, 2]
    assert saved_oof["clicked"].tolist() == [0, 0, 1]

    # 잘못된 Test ID 순서 입력 시 검증 예외 확인
    with pytest.raises(ValueError, match="ID 순서"):
        validate_deep_prediction_artifacts(
            test_path=test_path,
            expected_test_ids=np.array(["ID_A", "ID_B"]),
        )


def test_validate_parquet_row_id_alignment_guards(tmp_path: Path) -> None:
    """`row_id` 스트리밍 정렬 검증이 순서 불일치/컬럼 누락/결측을 정확히 차단하는지 확인합니다."""
    expected = np.arange(6, dtype=np.int64)

    ok_path = tmp_path / "ok_rowid.parquet"
    pq.write_table(
        pa.table({"row_id": pa.array(expected, type=pa.int64()), "val": pa.array(range(6))}),
        ok_path,
        row_group_size=2,
    )
    _validate_parquet_row_id_alignment(ok_path, expected, "row_id", batch_size=2, dataset_label="ok")

    shuffled_path = tmp_path / "shuffled_rowid.parquet"
    pq.write_table(
        pa.table({"row_id": pa.array([0, 2, 1, 3, 4, 5], type=pa.int64())}),
        shuffled_path,
        row_group_size=2,
    )
    with pytest.raises(ValueError, match="오름차순"):
        _validate_parquet_row_id_alignment(
            shuffled_path, expected, "row_id", batch_size=2, dataset_label="shuffled"
        )

    missing_col_path = tmp_path / "missing_col.parquet"
    pq.write_table(pa.table({"val": pa.array(range(6))}), missing_col_path)
    with pytest.raises(ValueError, match="row_id"):
        _validate_parquet_row_id_alignment(
            missing_col_path, expected, "row_id", batch_size=2, dataset_label="missing"
        )

    null_rowid_path = tmp_path / "null_rowid.parquet"
    pq.write_table(
        pa.table({"row_id": pa.array([0, 1, None, 3, 4, 5], type=pa.int64())}),
        null_rowid_path,
    )
    with pytest.raises(ValueError, match="결측치"):
        _validate_parquet_row_id_alignment(
            null_rowid_path, expected, "row_id", batch_size=2, dataset_label="null"
        )


def _build_synthetic_deep_frames(n_train: int = 80, n_test: int = 24) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)

    def _make_df(n: int, is_test: bool) -> pd.DataFrame:
        seqs = [
            "10,20,30,40" if i % 5 != 0 else ""
            for i in range(n)
        ]
        data = {
            "gender": rng.integers(1, 3, size=n),
            "age_group": rng.integers(1, 6, size=n),
            "inventory_id": rng.integers(100, 108, size=n),
            "day_of_week": (np.arange(n) % 4) + 1,
            "hour": rng.integers(0, 24, size=n),
            "l_feat_14": rng.integers(1000, 1005, size=n),
            "feat_a_1": rng.normal(0.0, 2.0, size=n).astype(np.float32),
            "history_a_1": rng.uniform(0.0, 1.0, size=n).astype(np.float32),
            "seq": seqs,
        }
        data["feat_a_1"][::9] = np.nan
        if is_test:
            return pd.DataFrame({"ID": [f"TE_{i:04d}" for i in range(n)], **data})
        y = np.zeros(n, dtype=np.int8)
        y[::4] = 1
        return pd.DataFrame({"row_id": np.arange(n, dtype=np.int64), "clicked": y, **data})

    train_df = _make_df(n_train, is_test=False)
    test_df = _make_df(n_test, is_test=True)
    folds_df = pd.DataFrame(
        {
            "row_id": np.arange(n_train, dtype=np.int64),
            "fold_id": (np.arange(n_train, dtype=np.int8) % 2),
            "is_oot_val": (train_df["day_of_week"].to_numpy() == 4),
        }
    )
    return train_df, folds_df, test_df


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch가 설치되어 있지 않습니다.")
def test_seq_dcn_model_forward_and_padding_invariance() -> None:
    """PyTorch 가용 환경에서 빈 시퀀스 안전성 및 PAD 마스킹 불변성을 검증합니다."""
    torch = require_torch()
    torch.manual_seed(0)
    model = SeqDCNv2CTRModel(
        seq_vocab_size=20,
        cat_vocab_sizes=[5, 8],
        num_numeric_features=3,
        query_cat_indices=[0, 1],
        seq_embed_dim=8,
        cat_embed_dim=4,
        att_hidden_dim=16,
        num_proj_dim=8,
        cross_layers=2,
        mlp_dims=[16, 8],
        dropout=0.0,
    )
    model.eval()

    seq_ids = torch.tensor(
        [
            [2, 5, 7, 0, 0],
            [0, 0, 0, 0, 0],  # 빈 시퀀스
            [3, 4, 6, 8, 9],
        ],
        dtype=torch.long,
    )
    seq_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [False, False, False, False, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    seq_lengths = torch.tensor([3, 0, 5], dtype=torch.long)
    cat_ids = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.long)
    num_x = torch.randn(3, 3, dtype=torch.float32)
    num_mask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    with torch.no_grad():
        out1 = model(seq_ids, seq_mask, seq_lengths, cat_ids, num_x, num_mask)
        assert out1.shape == (3,)
        assert torch.all(torch.isfinite(out1))

        # PAD 위치의 토큰 ID를 다른 값(15)으로 변조해도 seq_mask=False이므로 출력은 완전히 불변이어야 함
        seq_ids_mutated = seq_ids.clone()
        seq_ids_mutated[0, 3:] = 15
        seq_ids_mutated[1, :] = 15
        out2 = model(seq_ids_mutated, seq_mask, seq_lengths, cat_ids, num_x, num_mask)
        assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch가 설치되어 있지 않습니다.")
def test_train_deep_ctr_oof_and_parquet_streaming(tmp_path: Path) -> None:
    """PyTorch 가용 환경에서 인메모리 및 Parquet 스트리밍 2-Fold E2E 학습·추론을 검증합니다."""
    train_df, folds_df, test_df = _build_synthetic_deep_frames(n_train=64, n_test=20)
    cfg = DeepCTRConfig(
        max_seq_len=6,
        min_freq=1,
        max_vocab_size=100,
        cat_min_freq=1,
        seq_embed_dim=8,
        cat_embed_dim=4,
        att_hidden_dim=8,
        num_proj_dim=8,
        cross_layers=2,
        mlp_dims=[16, 8],
        dropout=0.0,
        n_splits=2,
        batch_size=16,
        eval_batch_size=32,
        epochs=2,
        use_amp=False,
        stream_batch_size=19,
    )

    oof_p1 = tmp_path / "oof_mem.parquet"
    test_p1 = tmp_path / "test_mem.parquet"
    res_mem = train_deep_ctr_oof(
        train_df=train_df,
        folds_df=folds_df,
        test_df=test_df,
        config=cfg,
        oof_output_path=oof_p1,
        test_output_path=test_p1,
        device_str="cpu",
    )
    assert "ap" in res_mem["oof_metrics"] and "wll" in res_mem["oof_metrics"]
    validate_deep_prediction_artifacts(
        oof_path=oof_p1,
        test_path=test_p1,
        expected_n_train=len(train_df),
        expected_n_test=len(test_df),
        expected_test_ids=test_df["ID"].to_numpy(),
    )

    # 서로 다른 Row-Group 크기를 가진 Parquet 파일에서도 정렬 스트리밍이 정상 동작해야 함
    tr_tab_p = tmp_path / "train_tab.parquet"
    tr_seq_p = tmp_path / "train_seq.parquet"
    te_tab_p = tmp_path / "test_tab.parquet"
    te_seq_p = tmp_path / "test_seq.parquet"
    folds_p = tmp_path / "folds.parquet"

    pq.write_table(
        pa.Table.from_pandas(train_df.drop(columns=["seq"]), preserve_index=False),
        tr_tab_p,
        row_group_size=15,
    )
    pq.write_table(
        pa.Table.from_pandas(train_df[["row_id", "seq"]], preserve_index=False),
        tr_seq_p,
        row_group_size=23,
    )
    pq.write_table(
        pa.Table.from_pandas(test_df.drop(columns=["seq"]), preserve_index=False),
        te_tab_p,
        row_group_size=7,
    )
    pq.write_table(
        pa.Table.from_pandas(test_df[["ID", "seq"]], preserve_index=False),
        te_seq_p,
        row_group_size=11,
    )
    pq.write_table(pa.Table.from_pandas(folds_df, preserve_index=False), folds_p)

    oof_p2 = tmp_path / "oof_pq.parquet"
    test_p2 = tmp_path / "test_pq.parquet"
    model_dir = tmp_path / "checkpoints"
    res_pq = train_deep_ctr_oof_from_parquet(
        train_tabular_path=tr_tab_p,
        train_seq_path=tr_seq_p,
        folds_path=folds_p,
        test_tabular_path=te_tab_p,
        test_seq_path=te_seq_p,
        config=cfg,
        oof_output_path=oof_p2,
        test_output_path=test_p2,
        memmap_cache_dir=tmp_path / "mmap_cache",
        model_dir=model_dir,
        device_str="cpu",
    )
    assert res_pq["all_folds_completed"] is True
    for f_id in range(2):
        assert (model_dir / f"model_fold_{f_id}.pt").exists()
        assert (model_dir / f"seq_vocab_fold_{f_id}.json").exists()
        assert (model_dir / f"cat_indexer_fold_{f_id}.json").exists()
        assert (model_dir / f"num_prep_fold_{f_id}.json").exists()

    # 특정 폴드(fold 1) 단독 재실행 시 전체 실행과 동일한 시드 재현성 확인
    res_single = train_deep_ctr_oof_from_parquet(
        train_tabular_path=tr_tab_p,
        train_seq_path=tr_seq_p,
        folds_path=folds_p,
        config=cfg,
        memmap_cache_dir=tmp_path / "mmap_cache_single",
        folds_to_run=[1],
        device_str="cpu",
    )
    assert res_single["fold_reports"][0]["best_val_metrics"] == res_pq["fold_reports"][1]["best_val_metrics"]

    validate_deep_prediction_artifacts(
        oof_path=oof_p2,
        test_path=test_p2,
        expected_n_train=len(train_df),
        expected_n_test=len(test_df),
        expected_test_ids=test_df["ID"].to_numpy(),
    )
