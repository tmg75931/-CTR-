"""다중 백엔드 GBDT OOF 학습기(`GBDTOOFTrainer`) 및 표준 아티팩트 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toss_ctr.data import build_cv_folds
from toss_ctr.evaluation.metrics import evaluate_predictions
from toss_ctr.features.tabular_pipeline import TabularFeaturePipeline
from toss_ctr.models.gbdt_trainer import GBDTOOFTrainer


def _create_synthetic_dataset(tmp_path: Path, n_train: int = 150, n_test: int = 40) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """합성 Diet Tabular 데이터 및 `folds.parquet` 계약을 생성합니다."""
    rng = np.random.default_rng(42)

    def _build_features(n: int) -> dict[str, np.ndarray | pd.Series]:
        inv = rng.integers(1, 8, size=n, dtype=np.int16)
        ad_set = rng.integers(100, 115, size=n, dtype=np.int16)
        hour = rng.integers(0, 24, size=n, dtype=np.int16)
        dow = (np.arange(n) % 6 + 1).astype(np.int16)
        age = rng.integers(1, 6, size=n, dtype=np.int16)
        gender = rng.integers(0, 3, size=n, dtype=np.int16)
        feat_a_1 = rng.normal(0.0, 1.0, size=n).astype(np.float32)
        feat_a_1[::11] = np.nan
        feat_b_1 = pd.Series(
            [int(v) if i % 9 != 0 else pd.NA for i, v in enumerate(rng.integers(0, 20, size=n))],
            dtype="Int16",
        )
        history_a_1 = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        return {
            "gender": gender,
            "age_group": age,
            "inventory_id": inv,
            "day_of_week": dow,
            "hour": hour,
            "l_feat_14": ad_set,
            "feat_a_1": feat_a_1,
            "feat_b_1": feat_b_1,
            "history_a_1": history_a_1,
        }

    tr_feats = _build_features(n_train)
    # 불균형(~15% 클릭률)이면서 피처 신호가 존재하는 합성 타겟 생성
    signal = (
        0.9 * (np.asarray(tr_feats["inventory_id"]) <= 3).astype(np.float64)
        + 0.8 * np.asarray(tr_feats["history_a_1"])
        + rng.normal(0.0, 0.35, size=n_train)
    )
    clicked = (signal > np.quantile(signal, 0.85)).astype(np.int8)

    train_df = pd.DataFrame({"row_id": np.arange(n_train, dtype=np.int64), "clicked": clicked, **tr_feats})
    # 비정렬 커스텀 ID로 test 순서 보존 여부 엄격 검증
    test_ids = [f"CUSTOM_Z_{n_test - i:03d}" if i % 2 == 0 else f"CUSTOM_A_{i:03d}" for i in range(n_test)]
    test_df = pd.DataFrame({"ID": test_ids, **_build_features(n_test)})

    train_path = tmp_path / "train_tabular_diet.parquet"
    test_path = tmp_path / "test_tabular_diet.parquet"
    folds_path = tmp_path / "folds.parquet"

    pq.write_table(pa.Table.from_pandas(train_df, preserve_index=False), train_path)
    pq.write_table(pa.Table.from_pandas(test_df, preserve_index=False), test_path)

    folds_df, _ = build_cv_folds(
        train_path=train_path,
        test_path=test_path,
        output_path=folds_path,
        n_splits=5,
        seed=42,
    )
    return train_df, test_df, folds_df


def test_gbdt_trainer_sklearn_hgbt_oof_and_test_contract(tmp_path: Path) -> None:
    """`sklearn_hgbt` 백엔드가 표준 OOF/Test Parquet 계약, 정렬, 평가지표, 재현성을 충족하는지 검증합니다."""
    train_df, test_df, folds_df = _create_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "preds"

    trainer = GBDTOOFTrainer(
        backend="sklearn_hgbt",
        model_name="hgbt_baseline",
        params={"max_iter": 25, "max_leaf_nodes": 15, "min_samples_leaf": 5},
        categorical_features=["gender", "age_group", "inventory_id"],
        weighting_mode="balanced_weight",
        seed=42,
    )

    res1 = trainer.fit_predict(
        train_df=train_df,
        folds=folds_df,
        test_df=test_df,
        output_dir=out_dir,
    )
    res2 = trainer.fit_predict(
        train_df=train_df,
        folds=folds_df,
        test_df=test_df,
    )

    # 동일 시드 결정론적 재현성 검증
    np.testing.assert_allclose(res1.oof_df["pred_prob"], res2.oof_df["pred_prob"], rtol=0.0, atol=1e-12)
    assert res1.test_df is not None and res2.test_df is not None
    np.testing.assert_allclose(res1.test_df["pred_prob"], res2.test_df["pred_prob"], rtol=0.0, atol=1e-12)

    # OOF 파일 스키마·정렬·무결성 검증
    oof_path = out_dir / "oof_hgbt_baseline.parquet"
    test_path = out_dir / "test_hgbt_baseline.parquet"
    metrics_path = out_dir / "hgbt_baseline_metrics.json"
    assert oof_path.exists()
    assert test_path.exists()
    assert metrics_path.exists()

    oof_loaded = pq.read_table(oof_path).to_pandas()
    assert list(oof_loaded.columns) == ["row_id", "fold_id", "clicked", "raw_logit", "pred_prob"]
    assert np.array_equal(oof_loaded["row_id"].to_numpy(), np.arange(len(train_df), dtype=np.int64))
    assert not oof_loaded.isna().any().any()
    assert ((oof_loaded["pred_prob"] > 0.0) & (oof_loaded["pred_prob"] < 1.0)).all()
    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-np.clip(oof_loaded["raw_logit"].to_numpy(), -35.0, 35.0))),
        oof_loaded["pred_prob"].to_numpy(),
        atol=1e-7,
    )

    # Test 파일 스키마 및 ID 순서 100% 보존 검증
    test_loaded = pq.read_table(test_path).to_pandas()
    assert list(test_loaded.columns) == ["ID", "raw_logit", "pred_prob"]
    assert test_loaded["ID"].tolist() == test_df["ID"].tolist()
    assert not test_loaded.isna().any().any()
    assert ((test_loaded["pred_prob"] > 0.0) & (test_loaded["pred_prob"] < 1.0)).all()

    # evaluate_predictions 일관성 검증
    direct_eval = evaluate_predictions(train_df["clicked"].to_numpy(), oof_loaded["pred_prob"].to_numpy())
    assert res1.overall_metrics.ap == pytest.approx(direct_eval.ap, abs=1e-12)
    assert res1.overall_metrics.wll == pytest.approx(direct_eval.wll, abs=1e-12)
    assert res1.overall_metrics.score == pytest.approx(direct_eval.score, abs=1e-12)
    assert len(res1.fold_metrics) == 5

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["overall_oof"]["score"] == pytest.approx(direct_eval.score, abs=1e-12)


def test_gbdt_trainer_prior_shift_preserves_ap_and_improves_wll(tmp_path: Path) -> None:
    """`prior_shift` 모드가 순위 지표(AP)를 완벽히 보존하면서 불균형 원시 확률 대비 50:50 WLL을 개선함을 검증합니다."""
    train_df, test_df, folds_df = _create_synthetic_dataset(tmp_path)

    trainer_shift = GBDTOOFTrainer(
        backend="sklearn_hgbt",
        model_name="hgbt_prior_shift",
        params={"max_iter": 25, "max_leaf_nodes": 15, "min_samples_leaf": 5},
        weighting_mode="prior_shift",
        target_pos_rate=0.5,
        seed=42,
    )
    res_shift = trainer_shift.fit_predict(train_df=train_df, folds=folds_df, test_df=test_df)

    assert res_shift.unshifted_metrics is not None
    # 단조 변환이므로 AP는 거의 완전히 동일(폴드별 약간 다른 train_pos_rate 차이를 고려해 근사 일치 확인)
    assert res_shift.overall_metrics.ap == pytest.approx(res_shift.unshifted_metrics.ap, abs=0.02)
    # 15% 불균형 데이터에서 50:50 Prior Shift 적용 시 WLL이 원시 확률보다 명확히 감소
    assert res_shift.overall_metrics.wll < res_shift.unshifted_metrics.wll


def test_gbdt_trainer_with_tabular_pipeline_and_from_config(tmp_path: Path) -> None:
    """`TabularFeaturePipeline`과 `GBDTOOFTrainer.from_config` 연동 종단간 학습·추론을 검증합니다."""
    train_df, test_df, folds_df = _create_synthetic_dataset(tmp_path)
    # 문자열 범주 컬럼 추가해 백엔드 범주 인코딩 호환성도 동시 확인
    train_df["str_cat"] = [f"S_{i % 4}" if i % 10 != 0 else None for i in range(len(train_df))]
    test_df["str_cat"] = [f"S_{i % 4}" if i % 7 != 0 else "UNSEEN" for i in range(len(test_df))]

    cfg = {
        "seed": 42,
        "features": {
            "interaction_pairs": [["inventory_id", "l_feat_14"], ["age_group", "gender"]],
            "interaction_format": "hash_int64",
            "add_row_stats": True,
            "te_smoothing": 15.0,
        },
        "models": {
            "hgbt_fe": {
                "backend": "sklearn_hgbt",
                "model_name": "hgbt_fe",
                "use_50_50_weight": True,
                "params": {"max_iter": 20, "min_samples_leaf": 5},
                "categorical_features": ["str_cat", "gender"],
            }
        },
    }

    pipeline = TabularFeaturePipeline.from_config(cfg)
    trainer = GBDTOOFTrainer.from_config(cfg, model_key="hgbt_fe")
    res = trainer.fit_predict(
        train_df=train_df,
        folds=folds_df,
        test_df=test_df,
        pipeline=pipeline,
        output_dir=tmp_path / "fe_preds",
    )

    assert len(res.oof_df) == len(train_df)
    assert res.test_df is not None and len(res.test_df) == len(test_df)
    assert "ix_inventory_id__l_feat_14" in res.feature_importance["feature"].tolist()
    assert "inventory_id_te" in res.feature_importance["feature"].tolist()
    assert res.oot_metrics is not None


def test_gbdt_trainer_reuse_cached_artifacts(tmp_path: Path) -> None:
    """`reuse_cached_artifacts=True` 시 검증된 기존 OOF/Test/지표 아티팩트를 재학습 없이 로드함을 검증합니다."""
    train_df, test_df, folds_df = _create_synthetic_dataset(tmp_path)
    out_dir = tmp_path / "cached_preds"

    trainer_first = GBDTOOFTrainer(
        backend="sklearn_hgbt",
        model_name="hgbt_cached",
        params={"max_iter": 15, "min_samples_leaf": 5},
        weighting_mode="balanced_weight",
        seed=42,
    )
    res1 = trainer_first.fit_predict(
        train_df=train_df,
        folds=folds_df,
        test_df=test_df,
        output_dir=out_dir,
    )

    cfg_cached = {
        "models": {
            "hgbt_cached": {
                "backend": "sklearn_hgbt",
                "model_name": "hgbt_cached",
                "reuse_cached_artifacts": True,
                "seed": 42,
            }
        }
    }
    trainer_cached = GBDTOOFTrainer.from_config(cfg_cached, model_key="hgbt_cached")
    assert trainer_cached.reuse_cached_artifacts is True

    # 백엔드가 절대 호출되지 않도록 _dispatch_backend를 실패 함수로 치환
    def _must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("캐시 재사용 시 _dispatch_backend가 호출되어서는 안 됩니다.")

    trainer_cached._dispatch_backend = _must_not_run  # type: ignore[method-assign]
    res2 = trainer_cached.fit_predict(
        train_df=train_df,
        folds=folds_df,
        test_df=test_df,
        output_dir=out_dir,
    )

    pd.testing.assert_frame_equal(res1.oof_df, res2.oof_df)
    assert res1.test_df is not None and res2.test_df is not None
    pd.testing.assert_frame_equal(res1.test_df, res2.test_df)
    assert res2.overall_metrics.ap == pytest.approx(res1.overall_metrics.ap, abs=1e-12)
    assert res2.overall_metrics.wll == pytest.approx(res1.overall_metrics.wll, abs=1e-12)
    assert res2.overall_metrics.score == pytest.approx(res1.overall_metrics.score, abs=1e-12)
    assert res2.fold_metrics == res1.fold_metrics


def test_gbdt_trainer_chunked_test_inference_consistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """테스트 청크 크기를 작게 분할해도 일괄 변환 결과와 100% 동일한 예측을 산출함을 검증합니다."""
    import toss_ctr.models.gbdt_trainer as gbdt_mod

    train_df, test_df, folds_df = _create_synthetic_dataset(tmp_path)
    cfg = {
        "seed": 42,
        "features": {
            "interaction_pairs": [["inventory_id", "l_feat_14"], ["age_group", "gender"]],
            "interaction_format": "hash_int64",
            "add_row_stats": True,
            "te_smoothing": 15.0,
        },
        "models": {
            "hgbt_chunk": {
                "backend": "sklearn_hgbt",
                "model_name": "hgbt_chunk",
                "use_50_50_weight": True,
                "params": {"max_iter": 20, "min_samples_leaf": 5},
                "categorical_features": ["inventory_id", "gender"],
            }
        },
    }
    pipeline = TabularFeaturePipeline.from_config(cfg)
    trainer = GBDTOOFTrainer.from_config(cfg, model_key="hgbt_chunk")

    res_full = trainer.fit_predict(
        train_df=train_df,
        folds=folds_df,
        test_df=test_df,
        pipeline=pipeline,
    )

    monkeypatch.setattr(gbdt_mod, "_TEST_INFER_CHUNK_SIZE", 13)
    res_chunked = trainer.fit_predict(
        train_df=train_df,
        folds=folds_df,
        test_df=test_df,
        pipeline=pipeline,
    )

    pd.testing.assert_frame_equal(res_full.oof_df, res_chunked.oof_df)
    assert res_full.test_df is not None and res_chunked.test_df is not None
    assert res_chunked.test_df["ID"].tolist() == test_df["ID"].tolist()
    np.testing.assert_allclose(
        res_full.test_df["pred_prob"],
        res_chunked.test_df["pred_prob"],
        rtol=0.0,
        atol=1e-12,
    )


def test_gbdt_trainer_invalid_backend_raises() -> None:
    """지원하지 않는 백엔드명 입력 시 명확한 ValueError가 발생함을 검증합니다."""
    with pytest.raises(ValueError, match="지원하지 않는 GBDT 백엔드"):
        GBDTOOFTrainer(backend="invalid_backend")  # type: ignore[arg-type]
