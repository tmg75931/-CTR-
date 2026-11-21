"""50:50 WLL 확률 보정기, OOF 블렌더, 제출 검증기 및 Torch 비의존 import 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import toss_ctr.features as feat_pkg
import toss_ctr.inference as inf_pkg
import toss_ctr.models as model_pkg
from toss_ctr.evaluation.metrics import evaluate_predictions
from toss_ctr.inference.blender import (
    OOFLogitBlender,
    assert_oof_alignment,
    assert_test_alignment,
    load_prediction_artifacts,
)
from toss_ctr.inference.submission import generate_submission, verify_submission_df
from toss_ctr.models.calibration import (
    AffineLogitCalibrator,
    GroupBiasShrinkageCalibrator,
    OutOfFoldCalibrator,
    TemperatureScalingCalibrator,
    logit_to_prob,
    prob_to_logit,
)


def _make_biased_2pct_ctr_data(
    n_rows: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """원천 ~2% CTR 분포로 편향되었지만 순위 변별력이 있는 합성 데이터를 생성합니다."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 1.2, size=n_rows)
    true_logit = -4.2 + signal
    true_prob = 1.0 / (1.0 + np.exp(-true_logit))
    y_true = rng.binomial(1, true_prob).astype(np.int8)

    # 모델 예측은 실제 ~2% CTR 스케일 근처에 있으나 온도 스케일이 과도하게 압축된 상태 가정
    raw_logit = -3.9 + 0.55 * signal + rng.normal(0.0, 0.25, size=n_rows)
    raw_prob = logit_to_prob(raw_logit)
    folds = (np.arange(n_rows) % 5).astype(np.int8)
    return y_true, raw_logit, raw_prob, folds


def test_package_public_exports_without_torch() -> None:
    """`torch`/GBDT 라이브러리 미설치 환경에서도 패키지 public export가 정상 임포트됩니다."""
    assert hasattr(feat_pkg, "TabularFeaturePipeline")
    assert hasattr(feat_pkg, "FoldSeqVocabulary")
    assert hasattr(model_pkg, "AffineLogitCalibrator")
    assert hasattr(model_pkg, "GBDTOOFTrainer")
    assert hasattr(model_pkg, "SeqDCNv2CTRModel")
    assert hasattr(inf_pkg, "OOFLogitBlender")
    assert hasattr(inf_pkg, "RealtimeCTRPredictor")


def test_affine_logit_calibrator_preserves_ap_and_improves_wll() -> None:
    """`AffineLogitCalibrator`가 `a > 0` 단조 변환으로 `AP`를 100% 보존하고 `50:50 WLL`을 대폭 개선합니다."""
    y_true, raw_logit, raw_prob, _ = _make_biased_2pct_ctr_data(n_rows=2500, seed=101)
    eval_before = evaluate_predictions(y_true, raw_prob)

    cal = AffineLogitCalibrator(l2_reg=1.0e-6).fit(raw_logit, y_true, input_space="logit")
    cal_prob = cal.predict_proba(raw_logit, input_space="logit")
    eval_after = evaluate_predictions(y_true, cal_prob)

    assert cal.scale_ > 0.0
    assert cal.bias_ > 1.0  # 2% CTR -> 50:50 사전 확률(+3.8 부근) 시프트 확인
    assert eval_after.ap == pytest.approx(eval_before.ap, rel=0.0, abs=1e-12)
    assert eval_after.wll < eval_before.wll - 0.5
    assert eval_after.score > eval_before.score

    # 확률 입력 모드 및 JSON 직렬화 왕복 일관성 검증
    cal_prob_from_p = AffineLogitCalibrator(l2_reg=1.0e-6).fit(
        raw_prob, y_true, input_space="prob"
    )
    np.testing.assert_allclose(
        cal_prob_from_p.predict_proba(raw_prob, input_space="prob"),
        cal_prob,
        atol=1e-4,
    )
    restored = AffineLogitCalibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(restored.predict_proba(raw_logit), cal_prob, atol=1e-12)


def test_temperature_scaling_preserves_ap() -> None:
    """`TemperatureScalingCalibrator`(`b = 0`)도 양수 기울기를 유지해 `AP`를 100% 보존합니다."""
    rng = np.random.default_rng(7)
    logits = rng.normal(0.0, 2.5, size=1000)
    probs_true = logit_to_prob(0.4 * logits)
    y_true = rng.binomial(1, probs_true).astype(np.int8)

    eval_before = evaluate_predictions(y_true, logit_to_prob(logits))
    t_cal = TemperatureScalingCalibrator().fit(logits, y_true)
    cal_probs = t_cal.predict_proba(logits)
    eval_after = evaluate_predictions(y_true, cal_probs)

    assert t_cal.scale_ > 0.0
    assert t_cal.bias_ == 0.0
    assert t_cal.temperature_ > 1.0
    assert eval_after.ap == pytest.approx(eval_before.ap, rel=0.0, abs=1e-12)
    assert eval_after.wll <= eval_before.wll


def test_out_of_fold_calibrator_strict_oof_and_test_consistency() -> None:
    """`OutOfFoldCalibrator`가 인샘플 누수 없이 폴드별 보정 및 테스트 평균 보정을 수행합니다."""
    y_true, raw_logit, raw_prob, folds = _make_biased_2pct_ctr_data(n_rows=2000, seed=202)
    eval_before = evaluate_predictions(y_true, raw_prob)

    oof_cal = OutOfFoldCalibrator(l2_reg=1.0e-6)
    oof_probs = oof_cal.fit_transform(raw_logit, y_true, folds, input_space="logit")
    eval_oof = evaluate_predictions(y_true, oof_probs)

    # Fold 0 검증 행의 예측이 오직 fold != 0만으로 학습한 보정기 출력과 100% 일치하는지 확인
    m_tr0 = folds != 0
    m_va0 = folds == 0
    manual_cal0 = AffineLogitCalibrator(l2_reg=1.0e-6).fit(raw_logit[m_tr0], y_true[m_tr0])
    np.testing.assert_allclose(oof_probs[m_va0], manual_cal0.predict_proba(raw_logit[m_va0]), atol=1e-12)

    # OOF 전체 AP 유사/보존 및 WLL 대폭 개선
    assert eval_oof.wll < eval_before.wll - 0.5
    assert eval_oof.score > eval_before.score

    # 테스트 추론 시 폴드별 logit 평균과 평균 파라미터 단일 보정기 일치 확인
    test_logits = np.linspace(-5.0, 1.0, 50)
    test_probs_oof_avg = oof_cal.predict_proba(test_logits)
    inf_cal = oof_cal.get_inference_calibrator()
    test_probs_single = inf_cal.predict_proba(test_logits)
    np.testing.assert_allclose(test_probs_oof_avg, test_probs_single, atol=1e-12)
    assert np.all(np.diff(test_probs_oof_avg) > 0)  # 테스트에서도 엄격한 단조 증가


def test_group_bias_shrinkage_calibrator_and_unseen_fallback() -> None:
    """`GroupBiasShrinkageCalibrator`의 Empirical Bayes 축소 및 미등장 그룹 폴백을 검증합니다."""
    rng = np.random.default_rng(303)
    groups = np.array(["INV_POS"] * 400 + ["INV_NEG"] * 400 + ["INV_RARE"] * 5)
    n = len(groups)
    base_logit = rng.normal(0.0, 0.8, size=n)
    # INV_POS는 실제 양성률이 훨씬 높고 INV_NEG는 낮으며 INV_RARE는 표본 5건
    group_shift = np.where(groups == "INV_POS", 1.5, np.where(groups == "INV_NEG", -1.5, 2.0))
    y_true = rng.binomial(1, logit_to_prob(base_logit + group_shift)).astype(np.int8)

    cal = GroupBiasShrinkageCalibrator(shrinkage=200.0, max_offset=3.0).fit(
        base_logit, y_true, groups=groups
    )
    assert cal.group_offsets_["INV_POS"] > 0.3
    assert cal.group_offsets_["INV_NEG"] < -0.3
    # 표본 5건 희소 그룹은 shrinkage 5 / (5 + 200) 로 강하게 0으로 축소됨
    assert abs(cal.group_offsets_["INV_RARE"]) < 0.2

    # 미등장 그룹("INV_UNSEEN")은 추가 offset 없이 전역 보정기 출력과 정확히 일치
    test_logits = np.array([-1.0, 0.0, 1.0])
    unseen_probs = cal.predict_proba(test_logits, groups=np.array(["INV_UNSEEN"] * 3))
    assert cal.global_calibrator_ is not None
    expected_global = cal.global_calibrator_.predict_proba(test_logits)
    np.testing.assert_allclose(unseen_probs, expected_global, atol=1e-12)

    # OutOfFoldCalibrator 내부에서도 groups 전달 정상 동작 확인
    folds = np.arange(n) % 5
    oof_gb = OutOfFoldCalibrator(
        calibrator_factory=lambda: GroupBiasShrinkageCalibrator(shrinkage=100.0)
    )
    oof_probs = oof_gb.fit_transform(base_logit, y_true, folds, groups=groups)
    assert oof_probs.shape == (n,)
    assert np.all((oof_probs > 0.0) & (oof_probs < 1.0))


def test_oof_alignment_and_logit_blender(tmp_path: Path) -> None:
    """`assert_oof_alignment`의 불일치 탐지 및 `OOFLogitBlender` CV 블렌딩을 검증합니다."""
    rng = np.random.default_rng(55)
    n = 500
    row_id = np.arange(n, dtype=np.int64)
    fold_id = (row_id % 5).astype(np.int8)
    f1 = rng.normal(0.0, 1.0, size=n)
    f2 = rng.normal(0.0, 1.0, size=n)
    y_true = rng.binomial(1, logit_to_prob(1.2 * f1 + 1.2 * f2)).astype(np.int8)

    # 모델 A는 f1만, 모델 B는 f2만 관측해 상호 보완적인 상황 구성
    l_a = 1.5 * f1 + rng.normal(0.0, 0.2, size=n)
    l_b = 1.5 * f2 + rng.normal(0.0, 0.2, size=n)

    oof_a = pd.DataFrame(
        {
            "row_id": row_id,
            "fold_id": fold_id,
            "clicked": y_true,
            "raw_logit": l_a,
            "pred_prob": logit_to_prob(l_a),
        }
    )
    oof_b = pd.DataFrame(
        {
            "row_id": row_id,
            "fold_id": fold_id,
            "clicked": y_true,
            "raw_logit": l_b,
            "pred_prob": logit_to_prob(l_b),
        }
    )

    # 1) 정상 정렬 통과
    assert_oof_alignment({"m_a": oof_a, "m_b": oof_b})

    # 2) row_id 역순 / fold_id 불일치 / clicked 불일치 탐지
    bad_order = oof_b.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="오름차순"):
        assert_oof_alignment({"m_a": oof_a, "m_b": bad_order})

    bad_fold = oof_b.copy()
    bad_fold.loc[0, "fold_id"] = 4
    with pytest.raises(ValueError, match="fold_id"):
        assert_oof_alignment({"m_a": oof_a, "m_b": bad_fold})

    bad_y = oof_b.copy()
    bad_y.loc[0, "clicked"] = 1 - int(bad_y.loc[0, "clicked"])
    with pytest.raises(ValueError, match="clicked"):
        assert_oof_alignment({"m_a": oof_a, "m_b": bad_y})

    # 3) Test 정렬 검증 및 load_prediction_artifacts 테스트
    test_ids = np.array([f"ID_{i:03d}" for i in range(30)])
    test_a = pd.DataFrame(
        {"ID": test_ids, "raw_logit": np.zeros(30), "pred_prob": np.full(30, 0.5)}
    )
    test_b = pd.DataFrame(
        {"ID": test_ids[::-1], "raw_logit": np.zeros(30), "pred_prob": np.full(30, 0.5)}
    )
    with pytest.raises(ValueError, match="ID"):
        assert_test_alignment({"m_a": test_a, "m_b": test_b})

    oof_a.to_parquet(tmp_path / "oof_m_a.parquet", index=False)
    oof_b.to_parquet(tmp_path / "oof_m_b.parquet", index=False)
    test_a.to_parquet(tmp_path / "test_m_a.parquet", index=False)
    test_a.to_parquet(tmp_path / "test_m_b.parquet", index=False)
    loaded_oof, loaded_test = load_prediction_artifacts(
        tmp_path, model_names=["m_a", "m_b", "nonexistent"], skip_missing=True, require_test=True
    )
    assert list(loaded_oof.keys()) == ["m_a", "m_b"]

    # 4) OOFLogitBlender 최적화 검증 (combined_score & wll)
    blender = OOFLogitBlender(objective="combined_score", seed=42)
    res = blender.fit_blend(loaded_oof, loaded_test)
    assert res.weights["m_a"] >= 0.0 and res.weights["m_b"] >= 0.0
    assert sum(res.weights.values()) == pytest.approx(1.0, abs=1e-9)
    best_single = max(m["score"] for m in res.single_model_metrics.values())
    assert res.oof_metrics.score > best_single
    assert res.test_df is not None and len(res.test_df) == 30

    blender_wll = OOFLogitBlender(objective="wll", seed=42)
    res_wll = blender_wll.fit_blend(loaded_oof)
    assert sum(res_wll.weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert res_wll.oof_metrics.wll < min(m["wll"] for m in res_wll.single_model_metrics.values())

    # logit round-trip sanity
    np.testing.assert_allclose(prob_to_logit(logit_to_prob(l_a)), l_a, atol=1e-6)


def test_verify_submission_df_valid_and_invalid_cases(tmp_path: Path) -> None:
    """`generate_submission` 및 `verify_submission_df`의 통과/오류 탐지 로직을 검증합니다."""
    ids = np.array(["T_10", "T_02", "T_07"])
    probs = np.array([0.12, 0.55, 0.89])
    sample_df = pd.DataFrame({"ID": ids, "clicked": [0, 0, 0]})

    csv_path = tmp_path / "submission.csv"
    sub_df = generate_submission(ids, probs, output_path=csv_path)
    assert csv_path.exists()

    ok_report = verify_submission_df(sub_df, sample_sub_df=sample_df, expected_rows=3)
    assert ok_report["valid"] is True
    assert ok_report["stats"]["n_nan"] == 0
    assert ok_report["stats"]["n_inf"] == 0
    assert ok_report["stats"]["p50"] == pytest.approx(0.55)

    # 결측치, 범위 이탈(0.0 / 1.0), ID 불일치, 컬럼 불일치 탐지
    nan_df = sub_df.copy()
    nan_df.loc[1, "clicked"] = np.nan
    assert verify_submission_df(nan_df)["valid"] is False

    inf_df = sub_df.copy()
    inf_df.loc[0, "clicked"] = np.inf
    assert verify_submission_df(inf_df)["valid"] is False

    zero_df = sub_df.copy()
    zero_df.loc[0, "clicked"] = 0.0
    assert verify_submission_df(zero_df)["valid"] is False

    bad_id_df = sub_df.iloc[::-1].reset_index(drop=True)
    rep_id = verify_submission_df(bad_id_df, sample_sub_df=sample_df)
    assert rep_id["valid"] is False
    assert any("ID" in err for err in rep_id["errors"])

    bad_col_df = sub_df.rename(columns={"clicked": "pred"})
    with pytest.raises(ValueError, match="컬럼"):
        verify_submission_df(bad_col_df, raise_on_error=True)
