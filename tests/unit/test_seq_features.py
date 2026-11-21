"""시퀀스 파싱, 경량 요약 피처 스트리밍 추출 및 FoldSeqVocabulary 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toss_ctr.features.seq_features import (
    PAD_ID,
    SEQ_SUMMARY_FEATURE_COLS,
    UNK_ID,
    FoldSeqVocabulary,
    detect_seq_delimiter,
    extract_seq_summary_features,
    extract_seq_summary_parquet,
    parse_seq_string,
    stable_token_to_int32,
)


def test_detect_and_parse_seq_string_variants() -> None:
    """다양한 구분자(쉼표/공백/파이프/세미콜론)와 결측·빈문자열 파싱을 검증합니다."""
    assert detect_seq_delimiter(["10,20,30", "40,50", ""]) == ","
    assert detect_seq_delimiter(["10 20 30", "40 50", None]) == " "
    assert detect_seq_delimiter(["a|b|c", "d|e"]) == "|"
    assert detect_seq_delimiter(["a;b;c", "d;e"]) == ";"
    assert detect_seq_delimiter(["", None, np.nan]) is None

    assert parse_seq_string("10, 20, 30") == ["10", "20", "30"]
    assert parse_seq_string("10  20   30") == ["10", "20", "30"]
    assert parse_seq_string("a|b|c") == ["a", "b", "c"]
    assert parse_seq_string("x;y;z") == ["x", "y", "z"]
    assert parse_seq_string(None) == []
    assert parse_seq_string(np.nan) == []
    assert parse_seq_string(pd.NA) == []
    assert parse_seq_string("") == []
    assert parse_seq_string("   ") == []
    assert parse_seq_string(["t1", "", None, "t2"]) == ["t1", "t2"]


def test_stable_token_to_int32_is_deterministic_and_positive() -> None:
    """결정론적 해시 ID가 항상 양의 int32(>=1)이며 실행 간 일관됨을 검증합니다."""
    for tok in ["0", "1", "102938", "ev_click_main", "토큰"]:
        val = stable_token_to_int32(tok)
        assert isinstance(val, int)
        assert 1 <= val <= 2_147_483_647
        assert val == stable_token_to_int32(tok)


def test_extract_seq_summary_features_accuracy() -> None:
    """11종 시퀀스 요약 피처의 수학적 정확성과 ID 보존을 검증합니다."""
    df = pd.DataFrame(
        {
            "row_id": np.array([10, 11, 12, 13], dtype=np.int64),
            "seq": [
                "",
                None,
                "A",
                "A,B,A,C,D,D,E",
            ],
        }
    )
    out = extract_seq_summary_features(df, id_col="row_id", seq_col="seq")
    assert list(out.columns) == ["row_id", *SEQ_SUMMARY_FEATURE_COLS]
    assert np.array_equal(out["row_id"].to_numpy(), np.array([10, 11, 12, 13], dtype=np.int64))

    # 빈 시퀀스(0, 1행)
    for idx in (0, 1):
        assert out.loc[idx, "seq_is_empty"] == 1
        assert out.loc[idx, "seq_token_len"] == 0
        assert out.loc[idx, "seq_unique_cnt"] == 0
        assert out.loc[idx, "seq_unique_ratio"] == pytest.approx(0.0)
        assert out.loc[idx, "seq_last_1"] == 0
        assert out.loc[idx, "seq_last_2"] == 0
        assert out.loc[idx, "seq_last_3"] == 0
        assert out.loc[idx, "seq_first_1"] == 0
        assert out.loc[idx, "seq_top1_share"] == pytest.approx(0.0)
        assert out.loc[idx, "seq_recent5_unique_ratio"] == pytest.approx(0.0)

    # 단일 토큰 "A" (2행)
    assert out.loc[2, "seq_is_empty"] == 0
    assert out.loc[2, "seq_token_len"] == 1
    assert out.loc[2, "seq_unique_cnt"] == 1
    assert out.loc[2, "seq_unique_ratio"] == pytest.approx(1.0)
    assert out.loc[2, "seq_first_1"] == out.loc[2, "seq_last_1"] == stable_token_to_int32("A")
    assert out.loc[2, "seq_last_2"] == 0
    assert out.loc[2, "seq_last_3"] == 0

    # "A,B,A,C,D,D,E" -> 길이 7, 고유 {A,B,C,D,E}=5, 최빈값 A/D=2회, 최근5 [A,C,D,D,E] 고유=4
    assert out.loc[3, "seq_is_empty"] == 0
    assert out.loc[3, "seq_char_len"] == len("A,B,A,C,D,D,E")
    assert out.loc[3, "seq_token_len"] == 7
    assert out.loc[3, "seq_unique_cnt"] == 5
    assert out.loc[3, "seq_unique_ratio"] == pytest.approx(5 / 7, rel=1e-5)
    assert out.loc[3, "seq_first_1"] == stable_token_to_int32("A")
    assert out.loc[3, "seq_last_1"] == stable_token_to_int32("E")
    assert out.loc[3, "seq_last_2"] == stable_token_to_int32("D")
    assert out.loc[3, "seq_last_3"] == stable_token_to_int32("D")
    assert out.loc[3, "seq_top1_share"] == pytest.approx(2 / 7, rel=1e-5)
    assert out.loc[3, "seq_recent5_unique_ratio"] == pytest.approx(4 / 5, rel=1e-5)


def test_extract_seq_summary_parquet_streaming_preserves_order(tmp_path: Path) -> None:
    """다중 Row-Group Parquet 스트리밍 요약 추출 시 row_id 및 ID 순서 보존을 검증합니다."""
    n_train = 55
    train_df = pd.DataFrame(
        {
            "row_id": np.arange(n_train, dtype=np.int64),
            "seq": [f"{i%5},{(i+1)%7},{(i+2)%9}" if i % 6 != 0 else None for i in range(n_train)],
        }
    )
    train_seq_path = tmp_path / "train_seq.parquet"
    train_out_path = tmp_path / "train_seq_summary.parquet"
    pq.write_table(
        pa.Table.from_pandas(train_df, preserve_index=False),
        train_seq_path,
        row_group_size=20,
    )

    stats = extract_seq_summary_parquet(
        seq_parquet_path=train_seq_path,
        output_path=train_out_path,
        batch_size=17,
    )
    assert stats["rows"] == n_train
    assert stats["id_col"] == "row_id"
    assert stats["detected_delimiter"] == ","

    loaded_tr = pq.read_table(train_out_path).to_pandas()
    assert len(loaded_tr) == n_train
    assert np.array_equal(loaded_tr["row_id"].to_numpy(), np.arange(n_train, dtype=np.int64))
    assert not loaded_tr.isna().any().any()

    # Test ID 보존 검증
    test_ids = [f"TEST_{i:04d}" for i in range(30)]
    test_df = pd.DataFrame(
        {
            "ID": test_ids,
            "seq": [f"10 20 {i}" if i % 4 != 0 else "" for i in range(30)],
        }
    )
    test_seq_path = tmp_path / "test_seq.parquet"
    test_out_path = tmp_path / "test_seq_summary.parquet"
    pq.write_table(pa.Table.from_pandas(test_df, preserve_index=False), test_seq_path, row_group_size=11)

    t_stats = extract_seq_summary_parquet(test_seq_path, test_out_path, batch_size=13)
    assert t_stats["id_col"] == "ID"
    assert t_stats["detected_delimiter"] == " "
    loaded_te = pq.read_table(test_out_path).to_pandas()
    assert loaded_te["ID"].tolist() == test_ids
    assert not loaded_te.isna().any().any()


def test_fold_seq_vocabulary_tail_truncation_padding_and_no_leakage(tmp_path: Path) -> None:
    """FoldSeqVocabulary의 Tail Truncation, PAD=0 마스킹 및 미등장 토큰 UNK=1 누수 차단을 검증합니다."""
    train_seqs = [
        "t1,t2,t3,t4,t5",
        "t2,t3,t4,t5",
        "t2,t4,t5,rare_only_once",
    ]
    vocab = FoldSeqVocabulary(min_freq=2, max_vocab_size=100).fit(train_seqs)

    assert vocab.token_to_id["<PAD>"] == PAD_ID == 0
    assert vocab.token_to_id["<UNK>"] == UNK_ID == 1
    assert "t1" not in vocab.token_to_id  # 빈도 1 < min_freq=2
    assert "rare_only_once" not in vocab.token_to_id
    for frequent in ("t2", "t3", "t4", "t5"):
        assert vocab.token_to_id[frequent] >= 2

    # 검증/테스트 전용 신규 토큰 "unseen_future_tok"는 UNK=1로 매핑되며 어휘집에 추가되지 않음
    vocab_size_before = vocab.vocab_size
    val_seqs = [
        "t1,t2,t3,t4,t5",  # 길이 5 -> max_seq_len=3 Tail Truncation 시 [t3, t4, t5] 보존
        "unseen_future_tok,t2",
        "",
    ]
    padded_ids, seq_mask, seq_lengths = vocab.encode_batch(val_seqs, max_seq_len=3, padding="post")
    assert vocab.vocab_size == vocab_size_before

    assert padded_ids.dtype == np.int32
    assert seq_mask.dtype == np.bool_
    assert seq_lengths.dtype == np.int32
    assert padded_ids.shape == (3, 3)

    # 첫 행: 최근 3개 [t3, t4, t5] 보존
    assert padded_ids[0].tolist() == [
        vocab.token_to_id["t3"],
        vocab.token_to_id["t4"],
        vocab.token_to_id["t5"],
    ]
    assert seq_mask[0].tolist() == [True, True, True]
    assert seq_lengths[0] == 3

    # 둘째 행: 미등장 토큰 -> UNK(1), post padding -> 마지막은 PAD(0)
    assert padded_ids[1].tolist() == [UNK_ID, vocab.token_to_id["t2"], PAD_ID]
    assert seq_mask[1].tolist() == [True, True, False]
    assert seq_lengths[1] == 2

    # 셋째 행: 빈 시퀀스
    assert padded_ids[2].tolist() == [PAD_ID, PAD_ID, PAD_ID]
    assert seq_mask[2].tolist() == [False, False, False]
    assert seq_lengths[2] == 0

    # pre padding 검증
    pre_ids, pre_mask, pre_lens = vocab.encode_batch(["unseen_future_tok,t2"], max_seq_len=3, padding="pre")
    assert pre_ids[0].tolist() == [PAD_ID, UNK_ID, vocab.token_to_id["t2"]]
    assert pre_mask[0].tolist() == [False, True, True]
    assert pre_lens[0] == 2

    # 직렬화 라운드트립 및 max_vocab_size 제한 검증
    save_path = tmp_path / "vocab.json"
    vocab.save(save_path)
    reloaded = FoldSeqVocabulary.load(save_path)
    r_ids, r_mask, r_len = reloaded.encode_batch(val_seqs, max_seq_len=3)
    assert np.array_equal(padded_ids, r_ids)
    assert np.array_equal(seq_mask, r_mask)
    assert np.array_equal(seq_lengths, r_len)

    capped = FoldSeqVocabulary(min_freq=1, max_vocab_size=4).fit(train_seqs)
    assert capped.vocab_size == 4

    # fit_from_parquet_mask 누수 차단 검증 (val 행의 토큰은 어휘집에 절대 포함되지 않아야 함)
    pq_df = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "seq": ["tr_a,tr_b", "tr_a,tr_b", "val_only_x,val_only_y", "val_only_x"],
        }
    )
    pq_path = tmp_path / "fold_seq.parquet"
    pq.write_table(pa.Table.from_pandas(pq_df, preserve_index=False), pq_path, row_group_size=2)
    mask_vocab = FoldSeqVocabulary(min_freq=1, max_vocab_size=100).fit_from_parquet_mask(
        pq_path, train_mask=np.array([True, True, False, False]), batch_size=2
    )
    assert "tr_a" in mask_vocab.token_to_id
    assert "val_only_x" not in mask_vocab.token_to_id


def test_fit_from_parquet_mask_multibatch_delimiter_auto(tmp_path: Path) -> None:
    """첫 배치에 단일 토큰만 있어도 후속 배치에서 delimiter='auto'가 정상 감지됨을 검증합니다."""
    pq_df = pd.DataFrame(
        {
            "row_id": np.arange(6, dtype=np.int64),
            "seq": [
                "single_only",
                "",
                "p1|p2|p3",
                "p2|p4",
                "val_secret|p9",
                None,
            ],
        }
    )
    pq_path = tmp_path / "multibatch_seq.parquet"
    pq.write_table(pa.Table.from_pandas(pq_df, preserve_index=False), pq_path, row_group_size=2)

    vocab = FoldSeqVocabulary(min_freq=1, max_vocab_size=100, delimiter="auto").fit_from_parquet_mask(
        pq_path,
        train_mask=np.array([True, True, True, True, False, True]),
        batch_size=2,
    )
    assert vocab.resolved_delimiter_ == "|"
    for tok in ("single_only", "p1", "p2", "p3", "p4"):
        assert tok in vocab.token_to_id
    assert "val_secret" not in vocab.token_to_id

    # 공백 구분자 전용 다중 배치 케이스 검증
    ws_df = pd.DataFrame(
        {
            "row_id": np.arange(4, dtype=np.int64),
            "seq": ["solo", "", "w1 w2", "w2 w3"],
        }
    )
    ws_path = tmp_path / "multibatch_ws.parquet"
    pq.write_table(pa.Table.from_pandas(ws_df, preserve_index=False), ws_path, row_group_size=2)
    ws_vocab = FoldSeqVocabulary(min_freq=1, max_vocab_size=100, delimiter="auto").fit_from_parquet_mask(
        ws_path,
        train_mask=np.array([True, True, True, True]),
        batch_size=2,
    )
    assert ws_vocab.resolved_delimiter_ == " "
    assert {"solo", "w1", "w2", "w3"}.issubset(ws_vocab.token_to_id.keys())
