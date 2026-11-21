"""Diet -> CV -> Seq 요약 -> 이종 모델 학습 -> OOF 블렌딩 -> WLL 보정 -> 제출/실시간 서빙 종단간 통합 테스트."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from toss_ctr.data import build_cv_folds, build_diet_artifacts
from toss_ctr.evaluation.metrics import clip_probs, evaluate_predictions
from toss_ctr.features import (
    TabularFeaturePipeline,
    build_interaction_features,
    extract_seq_summary_features,
    extract_seq_summary_parquet,
)
from toss_ctr.inference import (
    OfflineFeatureStore,
    OOFLogitBlender,
    RealtimeCTRPredictor,
    assert_oof_alignment,
    assert_test_alignment,
    generate_submission,
    load_prediction_artifacts,
    verify_submission_df,
)
from toss_ctr.models import GBDTOOFTrainer, OutOfFoldCalibrator, logit_to_prob, prob_to_logit
from toss_ctr.models.deep_ctr import save_deep_predictions


def _build_synthetic_raw_dataset(
    raw_dir: Path,
    n_train: int = 200,
    n_test: int = 60,
    seed: int = 42,
) -> tuple[Path, Path, Path, pd.DataFrame]:
    """200행 합성 Train + 60행 합성 Test(`train.parquet`, `test.parquet`, `sample_submission.csv`)를 생성합니다."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vocab_tokens = [f"item_{i}" for i in range(1, 25)]

    def _make_seq_list(n: int, signal_arr: np.ndarray) -> list[str | None]:
        seqs: list[str | None] = []
        for idx in range(n):
            if idx % 15 == 0:
                seqs.append(None)
                continue
            length = int(rng.integers(2, 12))
            if signal_arr[idx] > 0.3:
                toks = rng.choice(vocab_tokens[:8], size=length, replace=True).tolist()
            else:
                toks = rng.choice(vocab_tokens[8:], size=length, replace=True).tolist()
            seqs.append(",".join(toks))
        return seqs

    def _make_features(n: int, days: list[int]) -> tuple[dict[str, object], np.ndarray]:
        inv = rng.integers(1, 6, size=n, dtype=np.int64)
        ad_set = rng.integers(10, 22, size=n, dtype=np.int64)
        hour = rng.integers(0, 24, size=n, dtype=np.int64)
        dow = rng.choice(days, size=n).astype(np.int64)
        age = rng.integers(1, 6, size=n, dtype=np.int64)
        gender = rng.integers(0, 3, size=n, dtype=np.int64)
        feat_a_1 = rng.normal(0.0, 1.0, size=n).astype(np.float64)
        feat_a_1[::17] = np.nan
        feat_b_1 = rng.normal(0.0, 1.0, size=n).astype(np.float64)
        history_a_1 = rng.uniform(0.0, 1.0, size=n).astype(np.float64)

        latent = (
            1.6 * (inv <= 2).astype(np.float64)
            + 1.8 * (history_a_1 - 0.5)
            + 1.2 * np.nan_to_num(feat_a_1, nan=0.0)
            + 0.8 * feat_b_1
        )
        feats: dict[str, object] = {
            "gender": gender,
            "age_group": age,
            "inventory_id": inv,
            "day_of_week": dow,
            "hour": hour,
            "l_feat_14": ad_set,
            "feat_a_1": feat_a_1,
            "feat_b_1": feat_b_1,
            "history_a_1": history_a_1,
            "seq": _make_seq_list(n, latent),
        }
        return feats, latent

    tr_feats, tr_latent = _make_features(n_train, [1, 2, 3, 4, 5])
    # 불균형(~15-20% CTR)이면서 분명한 신호가 담긴 클릭 라벨 생성
    tr_logit = -2.0 + tr_latent + rng.normal(0.0, 0.25, size=n_train)
    clicked = rng.binomial(1, logit_to_prob(tr_logit)).astype(np.int64)
    train_df = pd.DataFrame({"clicked": clicked, **tr_feats})

    te_feats, _ = _make_features(n_test, [5, 6])
    test_ids = [f"TEST_{n_test - i:04d}" if i % 2 == 0 else f"TEST_A_{i:04d}" for i in range(n_test)]
    test_df = pd.DataFrame({"ID": test_ids, **te_feats})
    sample_sub_df = pd.DataFrame({"ID": test_ids, "clicked": np.zeros(n_test, dtype=np.float64)})

    train_path = raw_dir / "train.parquet"
    test_path = raw_dir / "test.parquet"
    sample_path = raw_dir / "sample_submission.csv"

    pq.write_table(pa.Table.from_pandas(train_df, preserve_index=False), train_path)
    pq.write_table(pa.Table.from_pandas(test_df, preserve_index=False), test_path)
    sample_sub_df.to_csv(sample_path, index=False)

    return train_path, test_path, sample_path, sample_sub_df


def _train_logreg_oof_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    pipeline: TabularFeaturePipeline,
    train_seq_summary: Path,
    test_seq_summary: Path,
    preds_dir: Path,
    model_name: str = "logreg_seq_linear",
) -> None:
    """선형 로지스틱 회귀 기반 2호 이종 모델을 5-Fold OOF 학습하고 표준 아티팩트로 저장합니다."""
    row_ids = folds_df["row_id"].to_numpy(dtype=np.int64)
    fold_ids = folds_df["fold_id"].to_numpy(dtype=np.int8)
    y_all = train_df["clicked"].to_numpy(dtype=np.int8)

    static_tr = pipeline.prepare_static_features(train_df, seq_summary=train_seq_summary, key_col="row_id")
    static_te = pipeline.prepare_static_features(test_df, seq_summary=test_seq_summary, key_col="ID")

    oof_logits = np.zeros(len(train_df), dtype=np.float64)
    test_logits_sum = np.zeros(len(test_df), dtype=np.float64)
    unique_folds = sorted(int(f) for f in np.unique(fold_ids))

    for f_id in unique_folds:
        bundle = pipeline.fit_transform_fold(
            static_tr,
            y_all,
            fold_ids,
            fold_id=f_id,
            test_df=static_te,
            assume_static_prepared=True,
        )
        num_cols = [
            c
            for c in bundle.feature_names
            if pd.api.types.is_numeric_dtype(bundle.X_train[c].dtype)
        ]
        X_tr = bundle.X_train[num_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        X_va = bundle.X_valid[num_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        assert bundle.X_test is not None
        X_te = bundle.X_test[num_cols].to_numpy(dtype=np.float64, na_value=np.nan)

        clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=42)),
            ]
        )
        clf.fit(X_tr, bundle.y_train)

        va_mask = fold_ids == f_id
        va_probs = clip_probs(clf.predict_proba(X_va)[:, 1])
        te_probs = clip_probs(clf.predict_proba(X_te)[:, 1])
        oof_logits[va_mask] = prob_to_logit(va_probs)
        test_logits_sum += prob_to_logit(te_probs)

    test_logits = test_logits_sum / float(len(unique_folds))
    save_deep_predictions(
        oof_path=preds_dir / f"oof_{model_name}.parquet",
        row_ids=row_ids,
        fold_ids=fold_ids,
        clicked=y_all,
        oof_raw_logits=oof_logits,
        oof_pred_probs=logit_to_prob(oof_logits),
        test_path=preds_dir / f"test_{model_name}.parquet",
        test_ids=test_df["ID"].to_numpy(),
        test_raw_logits=test_logits,
        test_pred_probs=logit_to_prob(test_logits),
    )


def test_full_end_to_end_pipeline_and_realtime_serving(tmp_path: Path) -> None:
    """200행 Train / 60행 Test 합성 데이터로 전체 CTR 파이프라인 및 실시간 서빙을 종단간 검증합니다."""
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    preds_dir = proc_dir / "predictions"

    raw_train_p, raw_test_p, _, sample_sub_df = _build_synthetic_raw_dataset(
        raw_dir, n_train=200, n_test=60, seed=42
    )

    # 1. Data Diet + Adaptive CV Splits
    manifest = build_diet_artifacts(
        train_path=raw_train_p,
        test_path=raw_test_p,
        output_dir=proc_dir,
        batch_size=64,
    )
    assert manifest["train_rows"] == 200
    assert manifest["test_rows"] == 60

    train_tab_p = proc_dir / "train_tabular_diet.parquet"
    test_tab_p = proc_dir / "test_tabular_diet.parquet"
    train_seq_p = proc_dir / "train_seq.parquet"
    test_seq_p = proc_dir / "test_seq.parquet"
    folds_p = proc_dir / "folds.parquet"

    folds_df, _ = build_cv_folds(
        train_path=train_tab_p,
        test_path=test_tab_p,
        output_path=folds_p,
        n_splits=5,
        seed=42,
    )

    # 2. Seq 요약 피처 추출 + TabularFeaturePipeline
    tr_seq_sum_p = proc_dir / "train_seq_summary.parquet"
    te_seq_sum_p = proc_dir / "test_seq_summary.parquet"
    extract_seq_summary_parquet(train_seq_p, tr_seq_sum_p, batch_size=64)
    extract_seq_summary_parquet(test_seq_p, te_seq_sum_p, batch_size=64)

    train_tab_df = pq.read_table(train_tab_p).to_pandas()
    test_tab_df = pq.read_table(test_tab_p).to_pandas()

    pipeline = TabularFeaturePipeline(
        freq_encode_cols=("inventory_id", "l_feat_14", "seq_last_1"),
        target_encode_cols=("inventory_id", "l_feat_14", "seq_last_1"),
        te_smoothing=10.0,
        seed=42,
    )

    # 3. 이종 모델 2종 학습 및 표준 예측 아티팩트 생성
    gbdt_trainer = GBDTOOFTrainer(
        backend="sklearn_hgbt",
        model_name="hgbt_fe_balanced",
        params={"max_iter": 30, "max_leaf_nodes": 15, "min_samples_leaf": 5},
        categorical_features=["gender", "age_group", "inventory_id", "l_feat_14"],
        weighting_mode="balanced_weight",
        seed=42,
    )
    gbdt_res = gbdt_trainer.fit_predict(
        train_df=train_tab_df,
        folds=folds_df,
        test_df=test_tab_df,
        pipeline=pipeline,
        train_seq_summary=tr_seq_sum_p,
        test_seq_summary=te_seq_sum_p,
        output_dir=preds_dir,
    )
    assert gbdt_res.overall_metrics.ap > 0.2

    _train_logreg_oof_model(
        train_df=train_tab_df,
        test_df=test_tab_df,
        folds_df=folds_df,
        pipeline=pipeline,
        train_seq_summary=tr_seq_sum_p,
        test_seq_summary=te_seq_sum_p,
        preds_dir=preds_dir,
        model_name="logreg_seq_linear",
    )

    # 4. 정합성 검증 -> OOF Log-odds 블렌딩 -> OOF 50:50 WLL 보정
    oof_dfs, test_dfs = load_prediction_artifacts(
        preds_dir=preds_dir,
        model_names=["hgbt_fe_balanced", "logreg_seq_linear"],
        require_test=True,
    )
    _, _, y_all = assert_oof_alignment(oof_dfs)
    assert_test_alignment(test_dfs, reference_ids=sample_sub_df["ID"].to_numpy())

    blender = OOFLogitBlender(objective="combined_score", n_restarts=12, seed=42)
    blend_res = blender.fit_blend(oof_dfs=oof_dfs, test_dfs=test_dfs)
    assert sum(blend_res.weights.values()) == pytest.approx(1.0, abs=1e-9)

    blended_oof_logits = blend_res.oof_df["raw_logit"].to_numpy(dtype=np.float64)
    oof_cal = OutOfFoldCalibrator(l2_reg=1.0e-6)
    cal_oof_probs = oof_cal.fit_transform(
        blended_oof_logits, y_all, folds_df, input_space="logit"
    )
    cal_eval = evaluate_predictions(y_all, cal_oof_probs)

    # 폴드 내 단조 변환 AP 100% 보존, 평균 추론 보정기 AP 100% 보존 & 정상 지표 산출 검증
    fold_ids_arr = folds_df["fold_id"].to_numpy()
    for f_k in sorted(np.unique(fold_ids_arr).tolist()):
        m_k = fold_ids_arr == f_k
        ap_pre_k = evaluate_predictions(y_all[m_k], blend_res.oof_df["pred_prob"].to_numpy()[m_k]).ap
        ap_post_k = evaluate_predictions(y_all[m_k], cal_oof_probs[m_k]).ap
        assert ap_post_k == pytest.approx(ap_pre_k, rel=0.0, abs=1e-12)

    inf_probs = oof_cal.get_inference_calibrator().predict_proba(blended_oof_logits, input_space="logit")
    inf_eval = evaluate_predictions(y_all, inf_probs)
    assert inf_eval.ap == pytest.approx(blend_res.oof_metrics.ap, rel=0.0, abs=1e-12)
    assert cal_eval.ap == pytest.approx(blend_res.oof_metrics.ap, abs=0.02)
    assert np.isfinite(cal_eval.wll) and cal_eval.wll > 0.0
    assert cal_eval.wll <= blend_res.oof_metrics.wll + 0.05

    # 5. 최종 제출 파일 생성 및 무결성 검증
    assert blend_res.test_df is not None
    test_blended_logits = blend_res.test_df["raw_logit"].to_numpy(dtype=np.float64)
    final_test_probs = oof_cal.predict_proba(test_blended_logits, input_space="logit")

    sub_path = tmp_path / "submission_final.csv"
    sub_df = generate_submission(
        test_ids=blend_res.test_df["ID"].to_numpy(),
        pred_probs=final_test_probs,
        output_path=sub_path,
    )
    verif_report = verify_submission_df(
        sub_df=sub_df,
        sample_sub_df=sample_sub_df,
        expected_rows=60,
        raise_on_error=True,
    )
    assert verif_report["valid"] is True
    assert verif_report["rows"] == 60

    # 6. RealtimeCTRPredictor 단건/미니배치 실시간 추론 스모크 테스트
    inv_offline_stats = (
        train_tab_df.groupby("inventory_id", as_index=False)["clicked"]
        .mean()
        .rename(columns={"clicked": "inv_offline_ctr"})
    )
    offline_store = OfflineFeatureStore().register_table(
        key_col="inventory_id",
        feature_df=inv_offline_stats,
        default_values={"inv_offline_ctr": float(train_tab_df["clicked"].mean())},
    )

    def _online_context_fn(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if "seq" in work.columns:
            seq_sum = extract_seq_summary_features(work, id_col=None)
            work = pd.concat([work, seq_sum], axis=1)
        ix_df = build_interaction_features(work, [("inventory_id", "l_feat_14")])
        work = pd.concat([work, ix_df], axis=1)
        work["hour_sin"] = np.sin(2.0 * np.pi * work["hour"].to_numpy(dtype=np.float64) / 24.0)
        return work

    def _fast_realtime_scorer(df: pd.DataFrame) -> np.ndarray:
        return (
            0.8 * df["inv_offline_ctr"].fillna(0.15).to_numpy(dtype=np.float64)
            + 0.3 * df["history_a_1"].fillna(0.5).to_numpy(dtype=np.float64)
            + 0.1 * df["hour_sin"].fillna(0.0).to_numpy(dtype=np.float64)
        )

    def _async_seq_scorer(df: pd.DataFrame) -> np.ndarray:
        tok_len = df.get("seq_token_len", pd.Series(np.zeros(len(df)))).to_numpy(dtype=np.float64)
        return 0.05 * tok_len - 0.2

    rt_predictor = RealtimeCTRPredictor(
        model_predictors={
            "hgbt_fe_balanced": _fast_realtime_scorer,
            "logreg_seq_linear": _async_seq_scorer,
        },
        weights=blend_res.weights,
        calibrator=oof_cal.get_inference_calibrator(),
        offline_store=offline_store,
        online_feature_fn=_online_context_fn,
        model_cadences={
            "hgbt_fe_balanced": "realtime_sync",
            "logreg_seq_linear": "nearline_async",
        },
    )

    mini_batch_df = pq.read_table(raw_test_p).to_pandas().iloc[:5].copy()
    batch_out = rt_predictor.predict_batch(mini_batch_df)
    assert len(batch_out) == 5
    assert list(batch_out["ID"]) == list(mini_batch_df["ID"])
    assert np.all((batch_out["clicked"] > 0.0) & (batch_out["clicked"] < 1.0))

    single_record = mini_batch_df.iloc[0].to_dict()
    one_out = rt_predictor.predict_one(single_record)
    assert 0.0 < one_out["clicked"] < 1.0
    assert one_out["clicked"] == pytest.approx(float(batch_out.iloc[0]["clicked"]), abs=1e-12)
    assert one_out["latency_ms"] >= 0.0


def test_cli_scripts_end_to_end_reproducibility(tmp_path: Path) -> None:
    """재현 CLI 스크립트(`run_diet_and_cv`, `extract_seq`, `train_gbdt`, `blend_and_submit`, `verify`)를 실제 실행 검증합니다."""
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    preds_dir = proc_dir / "predictions"
    sub_path = tmp_path / "outputs" / "sub.csv"
    rep_path = tmp_path / "reports" / "ensemble.json"
    cal_path = tmp_path / "artifacts" / "calibrator.json"

    raw_train_p, raw_test_p, sample_p, _ = _build_synthetic_raw_dataset(
        raw_dir, n_train=150, n_test=40, seed=99
    )

    def _run_script(script_name: str, extra_args: list[str]) -> None:
        cmd = [sys.executable, str(repo_root / "scripts" / script_name), *extra_args]
        res = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"{script_name} 실패:\nSTDOUT:{res.stdout}\nSTDERR:{res.stderr}"

    _run_script(
        "run_diet_and_cv.py",
        [
            "--config",
            str(repo_root / "configs" / "data_diet.yaml"),
            "--cv-config",
            str(repo_root / "configs" / "cv_strategy.yaml"),
            "--override",
            f"paths.raw_train={raw_train_p}",
            f"paths.raw_test={raw_test_p}",
            f"paths.output_dir={proc_dir}",
            "streaming.batch_size=64",
            "cv.n_splits=5",
        ],
    )

    _run_script(
        "extract_seq_features.py",
        [
            "--config",
            str(repo_root / "configs" / "deep_seq_dcn.yaml"),
            "--override",
            f"paths.train_seq={proc_dir / 'train_seq.parquet'}",
            f"paths.test_seq={proc_dir / 'test_seq.parquet'}",
            f"paths.train_seq_summary={proc_dir / 'train_seq_summary.parquet'}",
            f"paths.test_seq_summary={proc_dir / 'test_seq_summary.parquet'}",
            "seq.summary_batch_size=64",
        ],
    )

    # 모델 1: gbdt_baseline -> sklearn_hgbt 오버라이드 실행
    _run_script(
        "train_gbdt.py",
        [
            "--config",
            str(repo_root / "configs" / "gbdt_baseline.yaml"),
            "--override",
            f"paths.train_path={proc_dir / 'train_tabular_diet.parquet'}",
            f"paths.test_path={proc_dir / 'test_tabular_diet.parquet'}",
            f"paths.folds_path={proc_dir / 'folds.parquet'}",
            f"paths.preds_dir={preds_dir}",
            "backend=sklearn_hgbt",
            "model_name=hgbt_cli_base",
            "params.max_iter=15",
            "params.min_samples_leaf=5",
        ],
    )

    # 모델 2: gbdt_fe_tuned (lgbm_fe_tuned 키) -> sklearn_hgbt + seq_summary 조인 오버라이드 실행
    _run_script(
        "train_gbdt.py",
        [
            "--config",
            str(repo_root / "configs" / "gbdt_fe_tuned.yaml"),
            "--model",
            "lgbm_fe_tuned",
            "--override",
            f"paths.train_path={proc_dir / 'train_tabular_diet.parquet'}",
            f"paths.test_path={proc_dir / 'test_tabular_diet.parquet'}",
            f"paths.folds_path={proc_dir / 'folds.parquet'}",
            f"paths.seq_summary_train={proc_dir / 'train_seq_summary.parquet'}",
            f"paths.seq_summary_test={proc_dir / 'test_seq_summary.parquet'}",
            f"paths.preds_dir={preds_dir}",
            "backend=sklearn_hgbt",
            "model_name=hgbt_cli_fe",
            "params.max_iter=20",
            "params.min_samples_leaf=5",
        ],
    )

    _run_script(
        "blend_and_submit.py",
        [
            "--config",
            str(repo_root / "configs" / "ensemble_wll50.yaml"),
            "--override",
            f"paths.preds_dir={preds_dir}",
            f"paths.sample_submission={sample_p}",
            f"paths.train_tabular={proc_dir / 'train_tabular_diet.parquet'}",
            f"paths.test_tabular={proc_dir / 'test_tabular_diet.parquet'}",
            f"paths.submission_path={sub_path}",
            f"paths.report_path={rep_path}",
            f"paths.calibrator_path={cal_path}",
            'models.candidates=["hgbt_cli_base","hgbt_cli_fe"]',
            "submission.expected_rows=40",
            "blender.n_restarts=6",
        ],
    )

    _run_script(
        "verify_submission.py",
        [
            "--config",
            str(repo_root / "configs" / "ensemble_wll50.yaml"),
            "--submission",
            str(sub_path),
            "--sample",
            str(sample_p),
            "--expected-rows",
            "40",
        ],
    )
    assert sub_path.exists()
    assert rep_path.exists()
    assert cal_path.exists()

    # 감사 권고 검증 1: train_gbdt.py 최상위 --override seed=... 전파 확인
    _run_script(
        "train_gbdt.py",
        [
            "--config",
            str(repo_root / "configs" / "gbdt_fe_tuned.yaml"),
            "--model",
            "lgbm_fe_tuned",
            "--override",
            f"paths.train_path={proc_dir / 'train_tabular_diet.parquet'}",
            f"paths.test_path={proc_dir / 'test_tabular_diet.parquet'}",
            f"paths.folds_path={proc_dir / 'folds.parquet'}",
            f"paths.preds_dir={preds_dir}",
            "backend=sklearn_hgbt",
            "model_name=hgbt_seed_check",
            "seed=777",
            "params.max_iter=10",
            "params.min_samples_leaf=5",
        ],
    )
    import json as _json

    seed_metrics = _json.loads((preds_dir / "hgbt_seed_check_metrics.json").read_text(encoding="utf-8"))
    assert seed_metrics["seed"] == 777

    # 감사 권고 검증 2: group_shrinkage 선택 시 test_tabular 누락이면 조용한 폴백 없이 명시적 실패 확인
    fail_res = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "blend_and_submit.py"),
            "--config",
            str(repo_root / "configs" / "ensemble_wll50.yaml"),
            "--override",
            f"paths.preds_dir={preds_dir}",
            f"paths.sample_submission={sample_p}",
            f"paths.train_tabular={proc_dir / 'train_tabular_diet.parquet'}",
            f"paths.test_tabular={tmp_path / 'missing_test_tabular.parquet'}",
            f"paths.submission_path={sub_path}",
            'models.candidates=["hgbt_cli_base","hgbt_cli_fe"]',
            "calibration.method=group_shrinkage",
            "submission.expected_rows=40",
            "blender.n_restarts=4",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert fail_res.returncode != 0
    assert "te_groups" in fail_res.stderr
