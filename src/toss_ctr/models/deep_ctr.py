"""PyTorch 지연 임포트 기반 Target-Attention + DCN-v2 DeepCTR 학습 및 추론 모듈."""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from numpy.typing import ArrayLike

from toss_ctr.data.schema import DEFAULT_CONTRACT, DatasetContract
from toss_ctr.data.splits import validate_folds_contract
from toss_ctr.evaluation.metrics import clip_probs, evaluate_predictions
from toss_ctr.features.seq_features import FoldSeqVocabulary
from toss_ctr.models.nn_preprocessing import FoldCategoricalIndexer, FoldNumericPreprocessor

TORCH_AVAILABLE: bool = importlib.util.find_spec("torch") is not None

_TORCH_CLASSES_CACHE: dict[str, Any] | None = None

OOF_PRED_COLUMNS: tuple[str, ...] = ("row_id", "fold_id", "clicked", "raw_logit", "pred_prob")
TEST_PRED_COLUMNS: tuple[str, ...] = ("ID", "raw_logit", "pred_prob")


def require_torch() -> Any:
    """실제 학습/모델 생성 시점에 PyTorch를 지연 임포트해 반환합니다."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch가 설치되어 있지 않습니다. "
            "DeepCTR 모델 생성 및 학습/추론은 Google Colab GPU 환경 또는 `pip install torch` 환경에서 실행하세요."
        )
    import torch

    return torch


def set_all_seeds(seed: int) -> None:
    """Python, NumPy 및(설치 시) PyTorch 난수 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_pos_weight_50_50(y_train: ArrayLike) -> float:
    """`50:50 WLL` 손실 정렬을 위한 `BCEWithLogitsLoss` 양성 가중치(`N_0 / N_1`)를 계산합니다."""
    y = np.asarray(y_train, dtype=np.int8)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"양성({n_pos}) 및 음성({n_neg}) 샘플이 모두 1건 이상 있어야 합니다.")
    return float(n_neg / n_pos)


def logits_to_probs(raw_logit: ArrayLike) -> np.ndarray:
    """로짓(`raw_logit`)을 수치 안정적으로 확률 `(0, 1)`로 변환합니다."""
    arr = np.asarray(raw_logit, dtype=np.float64)
    clipped = np.clip(arr, -35.0, 35.0)
    probs = 1.0 / (1.0 + np.exp(-clipped))
    return clip_probs(probs)


@dataclass
class DeepCTRConfig:
    """Target-Attention + DCN-v2 파이프라인 전체 설정."""

    # seq 설정
    delimiter: str | None = "auto"
    max_seq_len: int = 100
    min_freq: int = 5
    max_vocab_size: int = 100_000
    padding: str = "post"
    truncating: str = "pre"

    # preprocessing 설정
    cat_min_freq: int = 5
    cat_max_cardinality: int = 100_000
    apply_signed_log1p: bool = True
    numeric_clip_value: float | None = 10.0
    stream_batch_size: int = 262_144

    # model 설정
    seq_embed_dim: int = 32
    cat_embed_dim: int = 16
    query_cols: list[str] = field(default_factory=lambda: ["inventory_id", "l_feat_14"])
    att_hidden_dim: int = 64
    num_proj_dim: int = 64
    cross_layers: int = 3
    cross_low_rank_dim: int | None = None
    mlp_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.15

    # training 설정
    n_splits: int = 5
    batch_size: int = 2048
    eval_batch_size: int = 4096
    epochs: int = 8
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    use_amp: bool = True
    amp_dtype: str = "float16"
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 2
    monitor_metric: str = "score"
    seed: int = 42
    use_pos_weight_50_50: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> DeepCTRConfig:
        """`configs/deep_seq_dcn.yaml` 파일에서 설정을 로드합니다."""
        with Path(yaml_path).open("r", encoding="utf-8") as fp:
            raw = yaml.safe_load(fp) or {}

        seq_cfg = raw.get("seq", {})
        prep_cfg = raw.get("preprocessing", {})
        model_cfg = raw.get("model", {})
        train_cfg = raw.get("training", {})

        return cls(
            delimiter=seq_cfg.get("delimiter", "auto"),
            max_seq_len=int(seq_cfg.get("max_seq_len", 100)),
            min_freq=int(seq_cfg.get("min_freq", 5)),
            max_vocab_size=int(seq_cfg.get("max_vocab_size", 100_000)),
            padding=str(seq_cfg.get("padding", "post")),
            truncating=str(seq_cfg.get("truncating", "pre")),
            cat_min_freq=int(prep_cfg.get("cat_min_freq", 5)),
            cat_max_cardinality=int(prep_cfg.get("cat_max_cardinality", 100_000)),
            apply_signed_log1p=bool(prep_cfg.get("apply_signed_log1p", True)),
            numeric_clip_value=(
                float(prep_cfg["numeric_clip_value"])
                if prep_cfg.get("numeric_clip_value") is not None
                else None
            ),
            stream_batch_size=int(prep_cfg.get("stream_batch_size", 262_144)),
            seq_embed_dim=int(model_cfg.get("seq_embed_dim", 32)),
            cat_embed_dim=int(model_cfg.get("cat_embed_dim", 16)),
            query_cols=list(model_cfg.get("query_cols", ["inventory_id", "l_feat_14"])),
            att_hidden_dim=int(model_cfg.get("att_hidden_dim", 64)),
            num_proj_dim=int(model_cfg.get("num_proj_dim", 64)),
            cross_layers=int(model_cfg.get("cross_layers", 3)),
            cross_low_rank_dim=(
                int(model_cfg["cross_low_rank_dim"])
                if model_cfg.get("cross_low_rank_dim") is not None
                else None
            ),
            mlp_dims=[int(x) for x in model_cfg.get("mlp_dims", [256, 128, 64])],
            dropout=float(model_cfg.get("dropout", 0.15)),
            n_splits=int(train_cfg.get("n_splits", 5)),
            batch_size=int(train_cfg.get("batch_size", 2048)),
            eval_batch_size=int(train_cfg.get("eval_batch_size", 4096)),
            epochs=int(train_cfg.get("epochs", 8)),
            lr=float(train_cfg.get("lr", 1.0e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1.0e-5)),
            use_amp=bool(train_cfg.get("use_amp", True)),
            amp_dtype=str(train_cfg.get("amp_dtype", "float16")),
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
            early_stopping_patience=int(train_cfg.get("early_stopping_patience", 2)),
            monitor_metric=str(train_cfg.get("monitor_metric", "score")).lower(),
            seed=int(train_cfg.get("seed", 42)),
            use_pos_weight_50_50=bool(train_cfg.get("use_pos_weight_50_50", True)),
        )


def load_deep_ctr_config(yaml_path: str | Path) -> DeepCTRConfig:
    """YAML 경로에서 `DeepCTRConfig`를 로드합니다."""
    return DeepCTRConfig.from_yaml(yaml_path)


def infer_deep_feature_columns(
    source: pd.DataFrame | pa.Schema | Sequence[str],
    cat_cols: Sequence[str] | None = None,
    num_cols: Sequence[str] | None = None,
    extra_cat_cols: Sequence[str] = (),
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> tuple[list[str], list[str]]:
    """DataFrame, PyArrow Schema 또는 컬럼 리스트에서 DeepCTR용 범주형/수치형 피처를 분리합니다."""
    reserved = set(contract.reserved_cols)
    explicit_cat_set = set(contract.explicit_cols) | set(extra_cat_cols)

    if isinstance(source, pd.DataFrame):
        all_cols = [c for c in source.columns if c not in reserved]
        is_non_numeric = {
            c: (
                isinstance(source[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(source[c].dtype)
                or pd.api.types.is_object_dtype(source[c].dtype)
                or pd.api.types.is_bool_dtype(source[c].dtype)
            )
            for c in all_cols
        }
    elif isinstance(source, pa.Schema):
        all_cols = [f.name for f in source if f.name not in reserved]
        is_non_numeric = {
            f.name: (
                pa.types.is_string(f.type)
                or pa.types.is_large_string(f.type)
                or pa.types.is_dictionary(f.type)
                or pa.types.is_boolean(f.type)
            )
            for f in source
            if f.name not in reserved
        }
    else:
        all_cols = [str(c) for c in source if str(c) not in reserved]
        is_non_numeric = {c: False for c in all_cols}

    all_set = set(all_cols)
    if cat_cols is not None:
        resolved_cat = [c for c in cat_cols if c in all_set and c not in reserved]
    else:
        resolved_cat = [
            c for c in all_cols if (c in explicit_cat_set) or is_non_numeric.get(c, False)
        ]

    cat_set = set(resolved_cat)
    if num_cols is not None:
        resolved_num = [c for c in num_cols if c in all_set and c not in reserved and c not in cat_set]
    else:
        resolved_num = [c for c in all_cols if c not in cat_set]

    return resolved_cat, resolved_num


def save_deep_predictions(
    oof_path: str | Path | None = None,
    row_ids: ArrayLike | None = None,
    fold_ids: ArrayLike | None = None,
    clicked: ArrayLike | None = None,
    oof_raw_logits: ArrayLike | None = None,
    oof_pred_probs: ArrayLike | None = None,
    test_path: str | Path | None = None,
    test_ids: ArrayLike | None = None,
    test_raw_logits: ArrayLike | None = None,
    test_pred_probs: ArrayLike | None = None,
    compression: str = "zstd",
) -> dict[str, Path]:
    """표준 규격 OOF(`row_id, fold_id, clicked, raw_logit, pred_prob`) 및 Test(`ID, raw_logit, pred_prob`) Parquet을 저장합니다."""
    saved: dict[str, Path] = {}

    if oof_path is not None:
        if any(v is None for v in (row_ids, fold_ids, clicked, oof_raw_logits)):
            raise ValueError("OOF 저장 시 row_ids, fold_ids, clicked, oof_raw_logits가 모두 필요합니다.")
        r_arr = np.asarray(row_ids, dtype=np.int64)
        f_arr = np.asarray(fold_ids, dtype=np.int8)
        y_arr = np.asarray(clicked, dtype=np.int8)
        l_arr = np.asarray(oof_raw_logits, dtype=np.float32)
        p_arr = (
            clip_probs(oof_pred_probs).astype(np.float32)
            if oof_pred_probs is not None
            else logits_to_probs(l_arr).astype(np.float32)
        )

        if not (len(r_arr) == len(f_arr) == len(y_arr) == len(l_arr) == len(p_arr)):
            raise ValueError("OOF 예측 배열들의 길이가 서로 일치하지 않습니다.")
        if not np.all(np.isfinite(l_arr)) or not np.all(np.isfinite(p_arr)):
            raise ValueError("OOF 예측값에 NaN 또는 Inf가 포함되어 있습니다.")

        # row_id 오름차순 정렬 보장
        if len(r_arr) > 1 and not np.all(r_arr[:-1] < r_arr[1:]):
            order = np.argsort(r_arr, kind="mergesort")
            r_arr = r_arr[order]
            f_arr = f_arr[order]
            y_arr = y_arr[order]
            l_arr = l_arr[order]
            p_arr = p_arr[order]

        oof_out = Path(oof_path)
        oof_out.parent.mkdir(parents=True, exist_ok=True)
        oof_table = pa.table(
            {
                "row_id": pa.array(r_arr, type=pa.int64()),
                "fold_id": pa.array(f_arr, type=pa.int8()),
                "clicked": pa.array(y_arr, type=pa.int8()),
                "raw_logit": pa.array(l_arr, type=pa.float32()),
                "pred_prob": pa.array(p_arr, type=pa.float32()),
            }
        )
        pq.write_table(oof_table, oof_out, compression=compression)
        saved["oof"] = oof_out

    if test_path is not None:
        if test_ids is None or test_raw_logits is None:
            raise ValueError("Test 저장 시 test_ids와 test_raw_logits가 모두 필요합니다.")
        id_arr = np.asarray(test_ids)
        tl_arr = np.asarray(test_raw_logits, dtype=np.float32)
        tp_arr = (
            clip_probs(test_pred_probs).astype(np.float32)
            if test_pred_probs is not None
            else logits_to_probs(tl_arr).astype(np.float32)
        )
        if not (len(id_arr) == len(tl_arr) == len(tp_arr)):
            raise ValueError("Test 예측 배열들의 길이가 서로 일치하지 않습니다.")
        if not np.all(np.isfinite(tl_arr)) or not np.all(np.isfinite(tp_arr)):
            raise ValueError("Test 예측값에 NaN 또는 Inf가 포함되어 있습니다.")

        test_out = Path(test_path)
        test_out.parent.mkdir(parents=True, exist_ok=True)
        test_table = pa.table(
            {
                "ID": pa.array(id_arr),
                "raw_logit": pa.array(tl_arr, type=pa.float32()),
                "pred_prob": pa.array(tp_arr, type=pa.float32()),
            }
        )
        pq.write_table(test_table, test_out, compression=compression)
        saved["test"] = test_out

    return saved


def validate_deep_prediction_artifacts(
    oof_path: str | Path | None = None,
    test_path: str | Path | None = None,
    expected_n_train: int | None = None,
    expected_n_test: int | None = None,
    expected_test_ids: ArrayLike | None = None,
) -> dict[str, Any]:
    """저장된 OOF/Test 예측 Parquet 아티팩트의 스키마, 정렬, 결측 및 확률 범위를 검증합니다."""
    report: dict[str, Any] = {"valid": True}

    if oof_path is not None:
        oof_df = pq.read_table(oof_path).to_pandas()
        missing = [c for c in OOF_PRED_COLUMNS if c not in oof_df.columns]
        if missing:
            raise ValueError(f"OOF 파일에 필수 컬럼이 누락되었습니다: {missing}")
        if expected_n_train is not None and len(oof_df) != expected_n_train:
            raise ValueError(f"OOF 행 수 불일치: expected={expected_n_train}, actual={len(oof_df)}")
        for col in OOF_PRED_COLUMNS:
            if oof_df[col].isna().any():
                raise ValueError(f"OOF 컬럼 '{col}'에 결측치가 존재합니다.")
        r_vals = oof_df["row_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(r_vals, np.arange(len(oof_df), dtype=np.int64)):
            raise ValueError("OOF row_id가 0..N-1 오름차순 연속이 아닙니다.")
        probs = oof_df["pred_prob"].to_numpy(dtype=np.float64)
        if np.any(probs <= 0.0) or np.any(probs >= 1.0) or not np.all(np.isfinite(probs)):
            raise ValueError("OOF pred_prob는 (0, 1) 유한 범위여야 합니다.")
        if not np.all(np.isfinite(oof_df["raw_logit"].to_numpy(dtype=np.float64))):
            raise ValueError("OOF raw_logit에 비유한 값이 있습니다.")
        report["oof_rows"] = int(len(oof_df))

    if test_path is not None:
        test_df = pq.read_table(test_path).to_pandas()
        missing = [c for c in TEST_PRED_COLUMNS if c not in test_df.columns]
        if missing:
            raise ValueError(f"Test 예측 파일에 필수 컬럼이 누락되었습니다: {missing}")
        if expected_n_test is not None and len(test_df) != expected_n_test:
            raise ValueError(
                f"Test 예측 행 수 불일치: expected={expected_n_test}, actual={len(test_df)}"
            )
        for col in TEST_PRED_COLUMNS:
            if test_df[col].isna().any():
                raise ValueError(f"Test 컬럼 '{col}'에 결측치가 존재합니다.")
        if expected_test_ids is not None:
            exp_ids = np.asarray(expected_test_ids)
            act_ids = test_df["ID"].to_numpy()
            if not np.array_equal(act_ids, exp_ids):
                raise ValueError("Test ID 순서가 원천/기준 ID와 일치하지 않습니다.")
        probs = test_df["pred_prob"].to_numpy(dtype=np.float64)
        if np.any(probs <= 0.0) or np.any(probs >= 1.0) or not np.all(np.isfinite(probs)):
            raise ValueError("Test pred_prob는 (0, 1) 유한 범위여야 합니다.")
        if not np.all(np.isfinite(test_df["raw_logit"].to_numpy(dtype=np.float64))):
            raise ValueError("Test raw_logit에 비유한 값이 있습니다.")
        report["test_rows"] = int(len(test_df))

    return report


def _get_torch_model_classes() -> dict[str, Any]:
    """PyTorch `nn.Module` 서브클래스들을 지연 임포트 후 한 번만 정의·캐싱합니다."""
    global _TORCH_CLASSES_CACHE
    if _TORCH_CLASSES_CACHE is not None:
        return _TORCH_CLASSES_CACHE

    torch = require_torch()
    import torch.nn as nn

    class TargetAttentionPooling(nn.Module):
        """DIN 스타일 Target Attention Pooling (PAD/빈 시퀀스/FP16 AMP 안전)."""

        def __init__(
            self,
            seq_embed_dim: int,
            query_dim: int,
            att_hidden_dim: int = 64,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.seq_embed_dim = int(seq_embed_dim)
            self.query_proj = nn.Sequential(
                nn.Linear(query_dim, seq_embed_dim),
                nn.LayerNorm(seq_embed_dim),
            )
            self.att_mlp = nn.Sequential(
                nn.Linear(4 * seq_embed_dim, att_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(att_hidden_dim, 1),
            )

        def forward(
            self,
            seq_emb: torch.Tensor,
            seq_mask: torch.Tensor,
            query_raw: torch.Tensor,
        ) -> torch.Tensor:
            """어텐션 가중합 시퀀스 표현 `(B, seq_embed_dim)`을 계산합니다."""
            _, seq_len, _ = seq_emb.shape
            query = self.query_proj(query_raw)  # (B, D)
            q_exp = query.unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, D)

            att_in = torch.cat([seq_emb, q_exp, seq_emb - q_exp, seq_emb * q_exp], dim=-1)
            att_logits = self.att_mlp(att_in).squeeze(-1)  # (B, L)

            has_seq = seq_mask.any(dim=1)  # (B,)
            safe_mask = seq_mask
            if not bool(torch.all(has_seq)):
                safe_mask = seq_mask.clone()
                safe_mask[~has_seq, 0] = True

            # FP16 표현 범위를 넘지 않는 -1e4 마스킹 및 float32 Softmax 후 dtype 정렬로 AMP bmm 오류 방지
            att_logits_f32 = att_logits.float().masked_fill(~safe_mask, -1.0e4)
            att_weights = torch.softmax(att_logits_f32, dim=1)
            att_weights = (att_weights * seq_mask.float()).to(dtype=seq_emb.dtype)

            pooled = torch.bmm(att_weights.unsqueeze(1), seq_emb).squeeze(1)
            return pooled

    class CrossNetworkV2(nn.Module):
        """DCN-v2 Residual Cross Network (Full-rank 또는 Low-rank 지원)."""

        def __init__(
            self,
            in_dim: int,
            num_layers: int = 3,
            low_rank_dim: int | None = None,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.in_dim = int(in_dim)
            self.num_layers = int(num_layers)
            self.low_rank_dim = int(low_rank_dim) if low_rank_dim is not None else None
            self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

            if self.low_rank_dim is not None:
                self.v_layers = nn.ModuleList(
                    [nn.Linear(in_dim, self.low_rank_dim, bias=False) for _ in range(num_layers)]
                )
                self.u_layers = nn.ModuleList(
                    [nn.Linear(self.low_rank_dim, in_dim, bias=True) for _ in range(num_layers)]
                )
                for u_layer in self.u_layers:
                    nn.init.normal_(u_layer.weight, mean=0.0, std=1e-3)
                    nn.init.zeros_(u_layer.bias)
            else:
                self.w_layers = nn.ModuleList(
                    [nn.Linear(in_dim, in_dim, bias=True) for _ in range(num_layers)]
                )
                for w_layer in self.w_layers:
                    nn.init.normal_(w_layer.weight, mean=0.0, std=1e-3)
                    nn.init.zeros_(w_layer.bias)

        def forward(self, x0: torch.Tensor) -> torch.Tensor:
            """잔차 크로스 상호작용 결과 `(B, in_dim)`을 반환합니다."""
            xl = x0
            for idx in range(self.num_layers):
                if self.low_rank_dim is not None:
                    cross_val = self.u_layers[idx](self.v_layers[idx](xl))
                else:
                    cross_val = self.w_layers[idx](xl)
                xl = x0 * self.drop(cross_val) + xl
            return xl

    class _SeqDCNv2CTRModule(nn.Module):
        """Target-Attention(시퀀스) + DCN-v2 + Deep MLP 하이브리드 CTR 모델."""

        def __init__(
            self,
            seq_vocab_size: int,
            cat_vocab_sizes: Sequence[int],
            num_numeric_features: int,
            query_cat_indices: Sequence[int] = (),
            seq_embed_dim: int = 32,
            cat_embed_dim: int = 16,
            att_hidden_dim: int = 64,
            num_proj_dim: int = 64,
            cross_layers: int = 3,
            cross_low_rank_dim: int | None = None,
            mlp_dims: Sequence[int] = (256, 128, 64),
            dropout: float = 0.15,
        ) -> None:
            super().__init__()
            self.seq_embed_dim = int(seq_embed_dim)
            self.cat_embed_dim = int(cat_embed_dim)
            self.num_numeric_features = int(num_numeric_features)
            self.query_cat_indices = tuple(int(i) for i in query_cat_indices)

            self.seq_embedding = nn.Embedding(
                int(seq_vocab_size), self.seq_embed_dim, padding_idx=0
            )
            self.seq_dropout = nn.Dropout(dropout)

            self.cat_embeddings = nn.ModuleList(
                [
                    nn.Embedding(int(vs), self.cat_embed_dim, padding_idx=0)
                    for vs in cat_vocab_sizes
                ]
            )
            self.num_cats = len(cat_vocab_sizes)

            if self.query_cat_indices:
                query_in_dim = len(self.query_cat_indices) * self.cat_embed_dim
                self.default_query = None
            elif self.num_cats > 0:
                query_in_dim = self.num_cats * self.cat_embed_dim
                self.default_query = None
            else:
                query_in_dim = self.seq_embed_dim
                self.default_query = nn.Parameter(torch.zeros(1, self.seq_embed_dim))

            self.target_attention = TargetAttentionPooling(
                seq_embed_dim=self.seq_embed_dim,
                query_dim=query_in_dim,
                att_hidden_dim=att_hidden_dim,
                dropout=dropout,
            )

            # Target Attention + Mean + Last Pooling 결합 + log1p(seq_len)
            seq_repr_dim = 3 * self.seq_embed_dim + 1
            cat_repr_dim = self.num_cats * self.cat_embed_dim

            if self.num_numeric_features > 0:
                self.num_proj = nn.Sequential(
                    nn.Linear(2 * self.num_numeric_features, num_proj_dim),
                    nn.LayerNorm(num_proj_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
                num_repr_dim = int(num_proj_dim)
            else:
                self.num_proj = None
                num_repr_dim = 0

            fusion_dim = seq_repr_dim + cat_repr_dim + num_repr_dim
            self.input_norm = nn.LayerNorm(fusion_dim)

            self.cross_net = CrossNetworkV2(
                in_dim=fusion_dim,
                num_layers=cross_layers,
                low_rank_dim=cross_low_rank_dim,
                dropout=dropout * 0.5,
            )

            mlp_layers: list[nn.Module] = []
            in_d = fusion_dim
            for hidden_d in mlp_dims:
                mlp_layers.extend(
                    [
                        nn.Linear(in_d, int(hidden_d)),
                        nn.LayerNorm(int(hidden_d)),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_d = int(hidden_d)
            self.deep_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
            deep_out_dim = in_d if mlp_layers else fusion_dim

            self.head = nn.Linear(fusion_dim + deep_out_dim, 1)

        def forward(
            self,
            seq_ids: torch.Tensor,
            seq_mask: torch.Tensor,
            seq_lengths: torch.Tensor,
            cat_ids: torch.Tensor | None = None,
            num_x: torch.Tensor | None = None,
            num_missing_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """단일 클릭 확률 로짓 `raw_logit` `(B,)`를 출력합니다."""
            batch_size, seq_len = seq_ids.shape
            mask_bool = seq_mask.bool()
            has_seq = mask_bool.any(dim=1)

            # 1) 범주형 임베딩
            cat_emb_list: list[torch.Tensor] = []
            if self.num_cats > 0:
                if cat_ids is None:
                    raise ValueError("cat_ids가 필요합니다.")
                for col_idx, emb_layer in enumerate(self.cat_embeddings):
                    cat_emb_list.append(emb_layer(cat_ids[:, col_idx]))
                cat_flat = torch.cat(cat_emb_list, dim=-1)
            else:
                cat_flat = seq_ids.new_zeros((batch_size, 0), dtype=torch.float32)

            # 2) Target Attention용 광고/지면 쿼리 구성
            if self.query_cat_indices:
                query_raw = torch.cat([cat_emb_list[i] for i in self.query_cat_indices], dim=-1)
            elif self.num_cats > 0:
                query_raw = cat_flat
            else:
                assert self.default_query is not None
                query_raw = self.default_query.expand(batch_size, -1)

            # 3) 시퀀스 임베딩 + PAD 완전 마스킹 (PAD 토큰 ID 변경 시에도 출력 불변 보장)
            seq_emb = self.seq_dropout(self.seq_embedding(seq_ids))
            mask_f = mask_bool.unsqueeze(-1).to(dtype=seq_emb.dtype)
            seq_emb = seq_emb * mask_f

            att_pool = self.target_attention(seq_emb, mask_bool, query_raw)

            denom = mask_bool.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=seq_emb.dtype)
            mean_pool = seq_emb.sum(dim=1) / denom

            # pre/post 패딩 모두 안전하게 마지막 유효 토큰 인덱스를 추출
            positions = torch.arange(seq_len, device=seq_ids.device).unsqueeze(0) * mask_bool.long()
            last_pos = positions.max(dim=1).values.view(batch_size, 1, 1).expand(
                -1, 1, self.seq_embed_dim
            )
            last_pool = seq_emb.gather(dim=1, index=last_pos).squeeze(1)
            last_pool = last_pool * has_seq.unsqueeze(-1).to(dtype=seq_emb.dtype)

            seq_len_feat = torch.log1p(seq_lengths.to(dtype=seq_emb.dtype).clamp(min=0.0)).unsqueeze(-1)
            seq_repr = torch.cat([att_pool, mean_pool, last_pool, seq_len_feat], dim=-1)

            # 4) 수치형 + 결측 마스크 투영
            parts: list[torch.Tensor] = [seq_repr]
            if self.num_cats > 0:
                parts.append(cat_flat.to(dtype=seq_repr.dtype))
            if self.num_proj is not None:
                if num_x is None or num_missing_mask is None:
                    raise ValueError("num_x와 num_missing_mask가 필요합니다.")
                num_in = torch.cat(
                    [num_x.to(dtype=seq_repr.dtype), num_missing_mask.to(dtype=seq_repr.dtype)],
                    dim=-1,
                )
                parts.append(self.num_proj(num_in))

            x0 = self.input_norm(torch.cat(parts, dim=-1))
            cross_out = self.cross_net(x0)
            deep_out = self.deep_mlp(x0)

            combined = torch.cat([cross_out, deep_out], dim=-1)
            return self.head(combined).squeeze(-1)

    _TORCH_CLASSES_CACHE = {
        "TargetAttentionPooling": TargetAttentionPooling,
        "CrossNetworkV2": CrossNetworkV2,
        "SeqDCNv2CTRModule": _SeqDCNv2CTRModule,
    }
    return _TORCH_CLASSES_CACHE


class SeqDCNv2CTRModel:
    """지연 임포트를 적용한 `SeqDCNv2CTRModel` 팩토리 래퍼 클래스.

    PyTorch 미설치 환경에서도 모듈 import가 성공하며, 인스턴스 생성 시점에만 `torch`를 로드합니다.
    """

    def __new__(
        cls,
        seq_vocab_size: int,
        cat_vocab_sizes: Sequence[int],
        num_numeric_features: int,
        query_cat_indices: Sequence[int] = (),
        seq_embed_dim: int = 32,
        cat_embed_dim: int = 16,
        att_hidden_dim: int = 64,
        num_proj_dim: int = 64,
        cross_layers: int = 3,
        cross_low_rank_dim: int | None = None,
        mlp_dims: Sequence[int] = (256, 128, 64),
        dropout: float = 0.15,
    ) -> Any:
        classes = _get_torch_model_classes()
        module_cls = classes["SeqDCNv2CTRModule"]
        return module_cls(
            seq_vocab_size=seq_vocab_size,
            cat_vocab_sizes=cat_vocab_sizes,
            num_numeric_features=num_numeric_features,
            query_cat_indices=query_cat_indices,
            seq_embed_dim=seq_embed_dim,
            cat_embed_dim=cat_embed_dim,
            att_hidden_dim=att_hidden_dim,
            num_proj_dim=num_proj_dim,
            cross_layers=cross_layers,
            cross_low_rank_dim=cross_low_rank_dim,
            mlp_dims=mlp_dims,
            dropout=dropout,
        )


def build_seq_dcn_model(
    seq_vocab_size: int,
    cat_vocab_sizes: Sequence[int],
    num_numeric_features: int,
    cat_cols: Sequence[str] = (),
    config: DeepCTRConfig | None = None,
    query_cat_indices: Sequence[int] | None = None,
) -> Any:
    """`DeepCTRConfig`와 어휘 크기를 바탕으로 Target-Attention + DCN-v2 모델을 생성합니다."""
    cfg = config or DeepCTRConfig()
    if query_cat_indices is None:
        cat_index_map = {name: idx for idx, name in enumerate(cat_cols)}
        resolved_query_indices = [cat_index_map[q] for q in cfg.query_cols if q in cat_index_map]
    else:
        resolved_query_indices = list(query_cat_indices)

    return SeqDCNv2CTRModel(
        seq_vocab_size=seq_vocab_size,
        cat_vocab_sizes=cat_vocab_sizes,
        num_numeric_features=num_numeric_features,
        query_cat_indices=resolved_query_indices,
        seq_embed_dim=cfg.seq_embed_dim,
        cat_embed_dim=cfg.cat_embed_dim,
        att_hidden_dim=cfg.att_hidden_dim,
        num_proj_dim=cfg.num_proj_dim,
        cross_layers=cfg.cross_layers,
        cross_low_rank_dim=cfg.cross_low_rank_dim,
        mlp_dims=cfg.mlp_dims,
        dropout=cfg.dropout,
    )


def _iter_block_shuffled_batches(
    n_rows: int,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
    block_size: int = 65_536,
) -> Iterator[np.ndarray]:
    """Memmap 순차 I/O 효율을 유지하면서 미니배치 단위 셔플을 수행하는 인덱스 제너레이터."""
    if not shuffle:
        for start in range(0, n_rows, batch_size):
            yield np.arange(start, min(start + batch_size, n_rows), dtype=np.int64)
        return

    block_starts = np.arange(0, n_rows, block_size, dtype=np.int64)
    rng.shuffle(block_starts)
    for b_start in block_starts:
        b_end = min(int(b_start) + block_size, n_rows)
        local_idx = np.arange(int(b_start), b_end, dtype=np.int64)
        rng.shuffle(local_idx)
        for pos in range(0, len(local_idx), batch_size):
            yield local_idx[pos : pos + batch_size]


def _predict_arrays_in_batches(
    model: Any,
    seq_ids: np.ndarray,
    seq_mask: np.ndarray,
    seq_lengths: np.ndarray,
    cat_ids: np.ndarray,
    num_x: np.ndarray,
    num_missing_mask: np.ndarray,
    batch_size: int,
    device: Any,
    use_amp: bool,
    amp_dtype: Any,
) -> np.ndarray:
    """인메모리 또는 Memmap 배열에서 미니배치 단위로 `raw_logit`을 추론합니다."""
    torch = require_torch()
    n_rows = len(seq_ids)
    preds = np.empty(n_rows, dtype=np.float32)
    amp_enabled = bool(use_amp and device.type == "cuda")

    model.eval()
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            b_seq_ids = torch.as_tensor(np.asarray(seq_ids[start:end]), dtype=torch.long, device=device)
            b_seq_mask = torch.as_tensor(np.asarray(seq_mask[start:end]), dtype=torch.bool, device=device)
            b_seq_len = torch.as_tensor(
                np.asarray(seq_lengths[start:end]), dtype=torch.long, device=device
            )
            b_cat = torch.as_tensor(np.asarray(cat_ids[start:end]), dtype=torch.long, device=device)
            b_num = torch.as_tensor(np.asarray(num_x[start:end]), dtype=torch.float32, device=device)
            b_num_mask = torch.as_tensor(
                np.asarray(num_missing_mask[start:end]), dtype=torch.float32, device=device
            )
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                logits = model(
                    seq_ids=b_seq_ids,
                    seq_mask=b_seq_mask,
                    seq_lengths=b_seq_len,
                    cat_ids=b_cat,
                    num_x=b_num,
                    num_missing_mask=b_num_mask,
                )
            preds[start:end] = logits.detach().float().cpu().numpy()
    return preds


def _fit_and_eval_single_fold(
    fold_id: int,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    y_val: np.ndarray,
    seq_vocab_size: int,
    cat_vocab_sizes: Sequence[int],
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    cfg: DeepCTRConfig,
    device: Any,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """단일 폴드의 PyTorch 모델 학습, AMP/Early Stopping, 최고 성능 OOF 예측을 수행합니다."""
    torch = require_torch()
    import torch.nn as nn

    # 전체 5-Fold 실행과 특정 폴드 단독 재실행 간 완전한 시드 재현성 보장
    set_all_seeds(cfg.seed + int(fold_id))

    amp_dtype = torch.bfloat16 if cfg.amp_dtype.lower() == "bfloat16" else torch.float16
    amp_enabled = bool(cfg.use_amp and device.type == "cuda")

    model = build_seq_dcn_model(
        seq_vocab_size=seq_vocab_size,
        cat_vocab_sizes=cat_vocab_sizes,
        num_numeric_features=len(num_cols),
        cat_cols=cat_cols,
        config=cfg,
    ).to(device)

    y_tr = np.asarray(train_data["y"], dtype=np.float32)
    if cfg.use_pos_weight_50_50:
        pos_weight_val = compute_pos_weight_50_50(y_tr)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
        )
    else:
        pos_weight_val = 1.0
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    n_train = len(y_tr)
    rng = np.random.default_rng(cfg.seed + fold_id * 1009)
    monitor_wll = cfg.monitor_metric == "wll"
    best_metric = float("inf") if monitor_wll else -float("inf")
    best_epoch = 0
    best_val_logits: np.ndarray | None = None
    best_state_dict: dict[str, Any] | None = None
    patience_left = cfg.early_stopping_patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for idx in _iter_block_shuffled_batches(n_train, cfg.batch_size, shuffle=True, rng=rng):
            b_seq_ids = torch.as_tensor(
                np.asarray(train_data["seq_ids"][idx]), dtype=torch.long, device=device
            )
            b_seq_mask = torch.as_tensor(
                np.asarray(train_data["seq_mask"][idx]), dtype=torch.bool, device=device
            )
            b_seq_len = torch.as_tensor(
                np.asarray(train_data["seq_lengths"][idx]), dtype=torch.long, device=device
            )
            b_cat = torch.as_tensor(
                np.asarray(train_data["cat_ids"][idx]), dtype=torch.long, device=device
            )
            b_num = torch.as_tensor(
                np.asarray(train_data["num_x"][idx]), dtype=torch.float32, device=device
            )
            b_num_mask = torch.as_tensor(
                np.asarray(train_data["num_missing_mask"][idx]), dtype=torch.float32, device=device
            )
            b_y = torch.as_tensor(np.asarray(y_tr[idx]), dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                logits = model(
                    seq_ids=b_seq_ids,
                    seq_mask=b_seq_mask,
                    seq_lengths=b_seq_len,
                    cat_ids=b_cat,
                    num_x=b_num,
                    num_missing_mask=b_num_mask,
                )
                loss = criterion(logits, b_y)

            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            b_cnt = int(len(idx))
            total_loss += float(loss.detach().item()) * b_cnt
            total_seen += b_cnt

        scheduler.step()
        train_loss = total_loss / max(total_seen, 1)

        val_logits = _predict_arrays_in_batches(
            model=model,
            seq_ids=val_data["seq_ids"],
            seq_mask=val_data["seq_mask"],
            seq_lengths=val_data["seq_lengths"],
            cat_ids=val_data["cat_ids"],
            num_x=val_data["num_x"],
            num_missing_mask=val_data["num_missing_mask"],
            batch_size=cfg.eval_batch_size,
            device=device,
            use_amp=cfg.use_amp,
            amp_dtype=amp_dtype,
        )
        val_probs = logits_to_probs(val_logits)
        eval_res = evaluate_predictions(y_val, val_probs)
        current_metric = eval_res.wll if monitor_wll else eval_res.score

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_ap": round(eval_res.ap, 6),
                "val_wll": round(eval_res.wll, 6),
                "val_score": round(eval_res.score, 6),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        improved = (
            current_metric < best_metric - 1e-6
            if monitor_wll
            else current_metric > best_metric + 1e-6
        )
        if improved or best_val_logits is None:
            best_metric = current_metric
            best_epoch = epoch
            best_val_logits = val_logits.copy()
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    assert best_val_logits is not None
    best_eval = evaluate_predictions(y_val, logits_to_probs(best_val_logits))
    fold_info = {
        "fold_id": int(fold_id),
        "n_train": int(n_train),
        "n_val": int(len(y_val)),
        "pos_weight": round(pos_weight_val, 4),
        "seq_vocab_size": int(seq_vocab_size),
        "best_epoch": int(best_epoch),
        "best_val_metrics": best_eval.to_dict(),
        "history": history,
    }
    return model, best_val_logits, fold_info


def train_deep_ctr_oof(
    train_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    config: DeepCTRConfig | None = None,
    cat_cols: Sequence[str] | None = None,
    num_cols: Sequence[str] | None = None,
    seq_col: str = "seq",
    oof_output_path: str | Path | None = None,
    test_output_path: str | Path | None = None,
    device_str: str | None = None,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """인메모리 DataFrame 기반 5-Fold 누수 방지 `SeqDCNv2CTRModel` 학습·OOF·Test 추론을 실행합니다."""
    torch = require_torch()
    cfg = config or DeepCTRConfig()
    set_all_seeds(cfg.seed)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    n_rows = len(train_df)
    if seq_col not in train_df.columns:
        raise ValueError(f"train_df에 '{seq_col}' 컬럼이 없습니다.")
    if contract.target_col not in train_df.columns:
        raise ValueError(f"train_df에 '{contract.target_col}' 컬럼이 없습니다.")

    unique_folds = sorted(int(f) for f in np.unique(folds_df[contract.fold_id_col].to_numpy()))
    validate_folds_contract(folds_df, n_rows=n_rows, n_splits=len(unique_folds), contract=contract)

    resolved_cat, resolved_num = infer_deep_feature_columns(
        train_df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        contract=contract,
    )

    fold_ids_all = folds_df[contract.fold_id_col].to_numpy(dtype=np.int8)
    row_ids_all = folds_df[contract.row_id_col].to_numpy(dtype=np.int64)
    y_all = train_df[contract.target_col].to_numpy(dtype=np.int8)

    oof_raw_logits = np.zeros(n_rows, dtype=np.float32)
    test_logits_sum = (
        np.zeros(len(test_df), dtype=np.float64) if test_df is not None else None
    )
    fold_reports: list[dict[str, Any]] = []
    amp_dtype = torch.bfloat16 if cfg.amp_dtype.lower() == "bfloat16" else torch.float16

    for f_id in unique_folds:
        val_mask = fold_ids_all == f_id
        train_mask = ~val_mask

        tr_df = train_df.iloc[train_mask]
        va_df = train_df.iloc[val_mask]

        # 오직 학습 폴드에서만 어휘집, 범주형 인덱서, 수치형 스케일러 fit (누수 0%)
        seq_vocab = FoldSeqVocabulary(
            min_freq=cfg.min_freq,
            max_vocab_size=cfg.max_vocab_size,
            delimiter=cfg.delimiter,
        ).fit(tr_df[seq_col].tolist())

        cat_indexer = FoldCategoricalIndexer(
            cat_cols=resolved_cat,
            min_freq=cfg.cat_min_freq,
            max_cardinality=cfg.cat_max_cardinality,
        ).fit(tr_df, cat_cols=resolved_cat)

        num_prep = FoldNumericPreprocessor(
            num_cols=resolved_num,
            apply_signed_log1p=cfg.apply_signed_log1p,
            clip_value=cfg.numeric_clip_value,
        ).fit(tr_df, num_cols=resolved_num)

        tr_seq_ids, tr_seq_mask, tr_seq_len = seq_vocab.encode_batch(
            tr_df[seq_col].tolist(),
            max_seq_len=cfg.max_seq_len,
            padding=cfg.padding,  # type: ignore[arg-type]
            truncating=cfg.truncating,  # type: ignore[arg-type]
        )
        va_seq_ids, va_seq_mask, va_seq_len = seq_vocab.encode_batch(
            va_df[seq_col].tolist(),
            max_seq_len=cfg.max_seq_len,
            padding=cfg.padding,  # type: ignore[arg-type]
            truncating=cfg.truncating,  # type: ignore[arg-type]
        )

        tr_num_x, tr_num_mask = num_prep.transform(tr_df)
        va_num_x, va_num_mask = num_prep.transform(va_df)

        train_data = {
            "seq_ids": tr_seq_ids,
            "seq_mask": tr_seq_mask,
            "seq_lengths": tr_seq_len,
            "cat_ids": cat_indexer.transform(tr_df),
            "num_x": tr_num_x,
            "num_missing_mask": tr_num_mask,
            "y": y_all[train_mask].astype(np.float32),
        }
        val_data = {
            "seq_ids": va_seq_ids,
            "seq_mask": va_seq_mask,
            "seq_lengths": va_seq_len,
            "cat_ids": cat_indexer.transform(va_df),
            "num_x": va_num_x,
            "num_missing_mask": va_num_mask,
        }
        cat_vocab_sizes = [cat_indexer.vocab_sizes[c] for c in resolved_cat]

        model, val_logits, fold_info = _fit_and_eval_single_fold(
            fold_id=f_id,
            train_data=train_data,
            val_data=val_data,
            y_val=y_all[val_mask],
            seq_vocab_size=seq_vocab.vocab_size,
            cat_vocab_sizes=cat_vocab_sizes,
            cat_cols=resolved_cat,
            num_cols=resolved_num,
            cfg=cfg,
            device=device,
        )
        oof_raw_logits[val_mask] = val_logits
        fold_reports.append(fold_info)

        if test_df is not None and test_logits_sum is not None:
            te_seq_ids, te_seq_mask, te_seq_len = seq_vocab.encode_batch(
                test_df[seq_col].tolist(),
                max_seq_len=cfg.max_seq_len,
                padding=cfg.padding,  # type: ignore[arg-type]
                truncating=cfg.truncating,  # type: ignore[arg-type]
            )
            te_num_x, te_num_mask = num_prep.transform(test_df)
            te_logits = _predict_arrays_in_batches(
                model=model,
                seq_ids=te_seq_ids,
                seq_mask=te_seq_mask,
                seq_lengths=te_seq_len,
                cat_ids=cat_indexer.transform(test_df),
                num_x=te_num_x,
                num_missing_mask=te_num_mask,
                batch_size=cfg.eval_batch_size,
                device=device,
                use_amp=cfg.use_amp,
                amp_dtype=amp_dtype,
            )
            test_logits_sum += te_logits.astype(np.float64)

    oof_pred_probs = logits_to_probs(oof_raw_logits)
    oof_eval = evaluate_predictions(y_all, oof_pred_probs)

    oof_df = pd.DataFrame(
        {
            "row_id": row_ids_all,
            "fold_id": fold_ids_all,
            "clicked": y_all,
            "raw_logit": oof_raw_logits,
            "pred_prob": oof_pred_probs.astype(np.float32),
        }
    )

    test_pred_df: pd.DataFrame | None = None
    test_raw_logits_avg: np.ndarray | None = None
    test_pred_probs_avg: np.ndarray | None = None
    if test_df is not None and test_logits_sum is not None:
        test_raw_logits_avg = (test_logits_sum / len(unique_folds)).astype(np.float32)
        test_pred_probs_avg = logits_to_probs(test_raw_logits_avg).astype(np.float32)
        test_pred_df = pd.DataFrame(
            {
                contract.id_col: test_df[contract.id_col].to_numpy(),
                "raw_logit": test_raw_logits_avg,
                "pred_prob": test_pred_probs_avg,
            }
        )

    save_deep_predictions(
        oof_path=oof_output_path,
        row_ids=row_ids_all,
        fold_ids=fold_ids_all,
        clicked=y_all,
        oof_raw_logits=oof_raw_logits,
        oof_pred_probs=oof_pred_probs,
        test_path=test_output_path,
        test_ids=test_df[contract.id_col].to_numpy() if test_df is not None else None,
        test_raw_logits=test_raw_logits_avg,
        test_pred_probs=test_pred_probs_avg,
    )

    oot_metrics: dict[str, float] | None = None
    if contract.oot_col in folds_df.columns:
        oot_mask = folds_df[contract.oot_col].to_numpy(dtype=bool)
        if np.any(oot_mask) and np.unique(y_all[oot_mask]).size == 2:
            oot_metrics = evaluate_predictions(y_all[oot_mask], oof_pred_probs[oot_mask]).to_dict()

    return {
        "oof_df": oof_df,
        "test_df": test_pred_df,
        "oof_metrics": oof_eval.to_dict(),
        "oot_metrics": oot_metrics,
        "fold_reports": fold_reports,
        "cat_cols": resolved_cat,
        "num_cols": resolved_num,
        "config": asdict(cfg),
    }


def _iter_aligned_parquet_batches(
    left_pf: pq.ParquetFile,
    right_pf: pq.ParquetFile,
    left_cols: Sequence[str],
    right_cols: Sequence[str],
    batch_size: int,
) -> Iterator[tuple[pa.Table, pa.Table]]:
    """두 ParquetFile의 Row-Group 크기가 달라도 정확히 동일한 행 수 단위로 정렬 배치를 생성합니다."""
    left_iter = left_pf.iter_batches(batch_size=batch_size, columns=list(left_cols))
    right_iter = right_pf.iter_batches(batch_size=batch_size, columns=list(right_cols))

    left_buf: pa.Table | None = None
    right_buf: pa.Table | None = None

    while True:
        if left_buf is None or left_buf.num_rows == 0:
            try:
                left_buf = pa.Table.from_batches([next(left_iter)])
            except StopIteration:
                left_buf = None
        if right_buf is None or right_buf.num_rows == 0:
            try:
                right_buf = pa.Table.from_batches([next(right_iter)])
            except StopIteration:
                right_buf = None

        if left_buf is None and right_buf is None:
            break
        if left_buf is None or right_buf is None:
            raise ValueError("두 Parquet 파일의 전체 행 수가 일치하지 않습니다.")

        take = min(left_buf.num_rows, right_buf.num_rows)
        yield left_buf.slice(0, take), right_buf.slice(0, take)
        left_buf = left_buf.slice(take) if take < left_buf.num_rows else None
        right_buf = right_buf.slice(take) if take < right_buf.num_rows else None


def _stream_build_fold_memmaps(
    train_tabular_path: Path,
    train_seq_path: Path,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    y_all: np.ndarray,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    seq_vocab: FoldSeqVocabulary,
    cat_indexer: FoldCategoricalIndexer,
    num_prep: FoldNumericPreprocessor,
    cfg: DeepCTRConfig,
    cache_dir: Path,
    seq_col: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Parquet을 배치 스트리밍 변환해 학습/검증 폴드 전용 Memmap 배열을 생성합니다."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_train = int(np.sum(train_mask))
    n_val = int(np.sum(val_mask))
    n_cats = len(cat_cols)
    n_nums = len(num_cols)
    max_len = cfg.max_seq_len

    def _safe_memmap(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
        if int(np.prod(shape)) == 0:
            return np.zeros(shape, dtype=dtype)
        return np.memmap(path, dtype=dtype, mode="w+", shape=shape)

    def _alloc_split(prefix: str, n_split: int) -> dict[str, np.ndarray]:
        return {
            "seq_ids": _safe_memmap(
                cache_dir / f"{prefix}_seq_ids.mmap",
                dtype=np.int32,
                shape=(n_split, max_len),
            ),
            "seq_mask": _safe_memmap(
                cache_dir / f"{prefix}_seq_mask.mmap",
                dtype=np.uint8,
                shape=(n_split, max_len),
            ),
            "seq_lengths": _safe_memmap(
                cache_dir / f"{prefix}_seq_len.mmap",
                dtype=np.int16,
                shape=(n_split,),
            ),
            "cat_ids": _safe_memmap(
                cache_dir / f"{prefix}_cat_ids.mmap",
                dtype=np.int32,
                shape=(n_split, n_cats),
            ),
            "num_x": _safe_memmap(
                cache_dir / f"{prefix}_num_x.mmap",
                dtype=np.float32,
                shape=(n_split, n_nums),
            ),
            "num_missing_mask": _safe_memmap(
                cache_dir / f"{prefix}_num_mask.mmap",
                dtype=np.uint8,
                shape=(n_split, n_nums),
            ),
            "y": _safe_memmap(
                cache_dir / f"{prefix}_y.mmap",
                dtype=np.float32,
                shape=(n_split,),
            ),
        }

    tr_mmap = _alloc_split("train", n_train)
    va_mmap = _alloc_split("val", n_val)

    tab_cols = list(cat_cols) + list(num_cols)
    tab_pf = pq.ParquetFile(train_tabular_path)
    seq_pf = pq.ParquetFile(train_seq_path)

    tr_pos = 0
    va_pos = 0
    global_offset = 0

    for tab_batch, seq_batch in _iter_aligned_parquet_batches(
        tab_pf, seq_pf, tab_cols, [seq_col], cfg.stream_batch_size
    ):
        n_b = tab_batch.num_rows
        if n_b == 0:
            continue
        b_tr_mask = train_mask[global_offset : global_offset + n_b]
        b_va_mask = val_mask[global_offset : global_offset + n_b]
        b_y = y_all[global_offset : global_offset + n_b]
        global_offset += n_b

        batch_df = tab_batch.to_pandas()
        batch_seqs = seq_batch.column(0).to_pylist()

        for split_mask, mmap_dict, is_train in (
            (b_tr_mask, tr_mmap, True),
            (b_va_mask, va_mmap, False),
        ):
            cnt = int(np.sum(split_mask))
            if cnt == 0:
                continue
            sub_df = batch_df.iloc[split_mask]
            sub_seqs = [batch_seqs[i] for i in np.flatnonzero(split_mask)]

            s_ids, s_mask, s_len = seq_vocab.encode_batch(
                sub_seqs,
                max_seq_len=max_len,
                padding=cfg.padding,  # type: ignore[arg-type]
                truncating=cfg.truncating,  # type: ignore[arg-type]
            )
            c_ids = cat_indexer.transform(sub_df)
            n_x, n_mask = num_prep.transform(sub_df)

            pos = tr_pos if is_train else va_pos
            end = pos + cnt

            mmap_dict["seq_ids"][pos:end] = s_ids
            mmap_dict["seq_mask"][pos:end] = s_mask.astype(np.uint8)
            mmap_dict["seq_lengths"][pos:end] = s_len.astype(np.int16)
            mmap_dict["cat_ids"][pos:end] = c_ids
            mmap_dict["num_x"][pos:end] = n_x
            mmap_dict["num_missing_mask"][pos:end] = n_mask.astype(np.uint8)
            mmap_dict["y"][pos:end] = b_y[split_mask].astype(np.float32)

            if is_train:
                tr_pos = end
            else:
                va_pos = end

    for m_dict in (tr_mmap, va_mmap):
        for mm in m_dict.values():
            if isinstance(mm, np.memmap):
                mm.flush()

    return tr_mmap, va_mmap


def _predict_test_parquet_streaming(
    model: Any,
    test_tabular_path: Path,
    test_seq_path: Path,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    seq_vocab: FoldSeqVocabulary,
    cat_indexer: FoldCategoricalIndexer,
    num_prep: FoldNumericPreprocessor,
    cfg: DeepCTRConfig,
    device: Any,
    seq_col: str,
    id_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """`test_tabular_diet.parquet`와 `test_seq.parquet`를 스트리밍으로 읽어 테스트 로짓을 예측합니다."""
    torch = require_torch()
    amp_dtype = torch.bfloat16 if cfg.amp_dtype.lower() == "bfloat16" else torch.float16

    tab_pf = pq.ParquetFile(test_tabular_path)
    seq_pf = pq.ParquetFile(test_seq_path)
    n_test = tab_pf.metadata.num_rows
    if seq_pf.metadata.num_rows != n_test:
        raise ValueError("test_tabular과 test_seq의 행 수가 일치하지 않습니다.")

    tab_cols = [id_col, *cat_cols, *num_cols]
    test_ids_chunks: list[np.ndarray] = []
    logits_out = np.empty(n_test, dtype=np.float32)
    offset = 0

    for tab_batch, seq_batch in _iter_aligned_parquet_batches(
        tab_pf, seq_pf, tab_cols, [seq_col], cfg.stream_batch_size
    ):
        n_b = tab_batch.num_rows
        if n_b == 0:
            continue
        batch_df = tab_batch.to_pandas()
        test_ids_chunks.append(batch_df[id_col].to_numpy())
        batch_seqs = seq_batch.column(0).to_pylist()

        s_ids, s_mask, s_len = seq_vocab.encode_batch(
            batch_seqs,
            max_seq_len=cfg.max_seq_len,
            padding=cfg.padding,  # type: ignore[arg-type]
            truncating=cfg.truncating,  # type: ignore[arg-type]
        )
        c_ids = cat_indexer.transform(batch_df)
        n_x, n_mask = num_prep.transform(batch_df)

        b_logits = _predict_arrays_in_batches(
            model=model,
            seq_ids=s_ids,
            seq_mask=s_mask,
            seq_lengths=s_len,
            cat_ids=c_ids,
            num_x=n_x,
            num_missing_mask=n_mask,
            batch_size=cfg.eval_batch_size,
            device=device,
            use_amp=cfg.use_amp,
            amp_dtype=amp_dtype,
        )
        logits_out[offset : offset + n_b] = b_logits
        offset += n_b

    test_ids = np.concatenate(test_ids_chunks, axis=0) if test_ids_chunks else np.array([])
    return test_ids, logits_out


def _validate_parquet_row_id_alignment(
    parquet_path: Path,
    expected_row_ids: np.ndarray,
    row_id_col: str,
    batch_size: int,
    dataset_label: str,
) -> None:
    """Parquet 파일의 `row_id` 컬럼만 경량 스트리밍으로 읽어 `0..N-1` 오름차순 일치를 검증합니다."""
    pf = pq.ParquetFile(parquet_path)
    if row_id_col not in pf.schema_arrow.names:
        raise ValueError(f"{dataset_label} 파일({parquet_path})에 '{row_id_col}' 컬럼이 없습니다.")

    n_expected = len(expected_row_ids)
    if pf.metadata.num_rows != n_expected:
        raise ValueError(
            f"{dataset_label} 행 수({pf.metadata.num_rows})가 folds({n_expected})와 일치하지 않습니다."
        )

    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[row_id_col]):
        col = batch.column(0)
        if col.null_count > 0:
            raise ValueError(f"{dataset_label} '{row_id_col}' 컬럼에 결측치가 포함되어 있습니다.")
        n_b = batch.num_rows
        actual_chunk = col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        expected_chunk = expected_row_ids[offset : offset + n_b]
        if not np.array_equal(actual_chunk, expected_chunk):
            raise ValueError(
                f"{dataset_label} '{row_id_col}' 순서가 folds.parquet의 0..N-1 오름차순과 일치하지 않습니다 "
                f"(offset={offset})."
            )
        offset += n_b


def train_deep_ctr_oof_from_parquet(
    train_tabular_path: str | Path,
    train_seq_path: str | Path,
    folds_path: str | Path,
    test_tabular_path: str | Path | None = None,
    test_seq_path: str | Path | None = None,
    config: DeepCTRConfig | None = None,
    cat_cols: Sequence[str] | None = None,
    num_cols: Sequence[str] | None = None,
    oof_output_path: str | Path | None = None,
    test_output_path: str | Path | None = None,
    memmap_cache_dir: str | Path = "/content/tmp_deepctr_cache",
    model_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    folds_to_run: Sequence[int] | None = None,
    device_str: str | None = None,
    cleanup_cache: bool = True,
    contract: DatasetContract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """1,070만 행 대용량 Parquet 파일을 스트리밍 + Memmap 기반으로 5-Fold AMP 학습·추론합니다."""
    torch = require_torch()
    cfg = config or DeepCTRConfig()
    set_all_seeds(cfg.seed)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_tab_p = Path(train_tabular_path)
    train_seq_p = Path(train_seq_path)
    folds_df = pq.read_table(folds_path).to_pandas()

    tab_pf = pq.ParquetFile(train_tab_p)
    seq_pf = pq.ParquetFile(train_seq_p)
    n_rows = tab_pf.metadata.num_rows
    if seq_pf.metadata.num_rows != n_rows:
        raise ValueError(
            f"train_tabular({n_rows})와 train_seq({seq_pf.metadata.num_rows})의 행 수가 다릅니다."
        )

    validate_folds_contract(folds_df, n_rows=n_rows, n_splits=cfg.n_splits, contract=contract)

    row_ids_all = folds_df[contract.row_id_col].to_numpy(dtype=np.int64)
    _validate_parquet_row_id_alignment(
        train_tab_p,
        expected_row_ids=row_ids_all,
        row_id_col=contract.row_id_col,
        batch_size=cfg.stream_batch_size,
        dataset_label="train_tabular",
    )
    _validate_parquet_row_id_alignment(
        train_seq_p,
        expected_row_ids=row_ids_all,
        row_id_col=contract.row_id_col,
        batch_size=cfg.stream_batch_size,
        dataset_label="train_seq",
    )

    resolved_cat, resolved_num = infer_deep_feature_columns(
        tab_pf.schema_arrow,
        cat_cols=cat_cols,
        num_cols=num_cols,
        contract=contract,
    )

    y_all = pq.read_table(train_tab_p, columns=[contract.target_col]).column(0).to_numpy().astype(np.int8)
    fold_ids_all = folds_df[contract.fold_id_col].to_numpy(dtype=np.int8)

    target_folds = (
        sorted(int(f) for f in folds_to_run)
        if folds_to_run is not None
        else list(range(cfg.n_splits))
    )

    oof_raw_logits = np.zeros(n_rows, dtype=np.float32)
    oof_filled_mask = np.zeros(n_rows, dtype=bool)
    test_logits_sum: np.ndarray | None = None
    test_ids_ref: np.ndarray | None = None
    fold_reports: list[dict[str, Any]] = []

    tab_cols_stream = [*resolved_cat, *resolved_num]
    cache_root = Path(memmap_cache_dir)

    for f_id in target_folds:
        val_mask = fold_ids_all == f_id
        train_mask = ~val_mask

        # 1) 학습 폴드 전용 Seq 어휘집 스트리밍 구축
        seq_vocab = FoldSeqVocabulary(
            min_freq=cfg.min_freq,
            max_vocab_size=cfg.max_vocab_size,
            delimiter=cfg.delimiter,
        ).fit_from_parquet_mask(
            train_seq_p,
            train_mask=train_mask,
            seq_col=contract.seq_col,
            batch_size=cfg.stream_batch_size,
        )

        # 2) 학습 폴드 전용 범주형/수치형 전처리기 스트리밍 통계 누적
        cat_indexer = FoldCategoricalIndexer(
            cat_cols=resolved_cat,
            min_freq=cfg.cat_min_freq,
            max_cardinality=cfg.cat_max_cardinality,
        )
        num_prep = FoldNumericPreprocessor(
            num_cols=resolved_num,
            apply_signed_log1p=cfg.apply_signed_log1p,
            clip_value=cfg.numeric_clip_value,
        )

        offset = 0
        for tab_batch in tab_pf.iter_batches(batch_size=cfg.stream_batch_size, columns=tab_cols_stream):
            n_b = tab_batch.num_rows
            b_tr_mask = train_mask[offset : offset + n_b]
            offset += n_b
            if not np.any(b_tr_mask):
                continue
            tr_chunk_df = tab_batch.to_pandas().iloc[b_tr_mask]
            cat_indexer.partial_fit(tr_chunk_df, cat_cols=resolved_cat)
            num_prep.partial_fit(tr_chunk_df, num_cols=resolved_num)

        cat_indexer.finalize_fit()
        num_prep.finalize_fit()

        # 3) Memmap 스트리밍 변환 및 폴드 학습
        fold_cache = cache_root / f"fold_{f_id}"
        try:
            tr_mmap, va_mmap = _stream_build_fold_memmaps(
                train_tabular_path=train_tab_p,
                train_seq_path=train_seq_p,
                train_mask=train_mask,
                val_mask=val_mask,
                y_all=y_all,
                cat_cols=resolved_cat,
                num_cols=resolved_num,
                seq_vocab=seq_vocab,
                cat_indexer=cat_indexer,
                num_prep=num_prep,
                cfg=cfg,
                cache_dir=fold_cache,
                seq_col=contract.seq_col,
            )

            cat_vocab_sizes = [cat_indexer.vocab_sizes[c] for c in resolved_cat]
            model, val_logits, fold_info = _fit_and_eval_single_fold(
                fold_id=f_id,
                train_data=tr_mmap,
                val_data=va_mmap,
                y_val=y_all[val_mask],
                seq_vocab_size=seq_vocab.vocab_size,
                cat_vocab_sizes=cat_vocab_sizes,
                cat_cols=resolved_cat,
                num_cols=resolved_num,
                cfg=cfg,
                device=device,
            )
            tr_mmap.clear()
            va_mmap.clear()
            oof_raw_logits[val_mask] = val_logits
            oof_filled_mask[val_mask] = True
            fold_reports.append(fold_info)

            if model_dir is not None:
                m_dir = Path(model_dir)
                m_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), m_dir / f"model_fold_{f_id}.pt")
                seq_vocab.save(m_dir / f"seq_vocab_fold_{f_id}.json")
                with (m_dir / f"cat_indexer_fold_{f_id}.json").open("w", encoding="utf-8") as fp:
                    json.dump(cat_indexer.to_dict(), fp, ensure_ascii=False)
                with (m_dir / f"num_prep_fold_{f_id}.json").open("w", encoding="utf-8") as fp:
                    json.dump(num_prep.to_dict(), fp, ensure_ascii=False)

            if test_tabular_path is not None and test_seq_path is not None:
                t_ids, t_logits = _predict_test_parquet_streaming(
                    model=model,
                    test_tabular_path=Path(test_tabular_path),
                    test_seq_path=Path(test_seq_path),
                    cat_cols=resolved_cat,
                    num_cols=resolved_num,
                    seq_vocab=seq_vocab,
                    cat_indexer=cat_indexer,
                    num_prep=num_prep,
                    cfg=cfg,
                    device=device,
                    seq_col=contract.seq_col,
                    id_col=contract.id_col,
                )
                if test_logits_sum is None:
                    test_logits_sum = t_logits.astype(np.float64)
                    test_ids_ref = t_ids
                else:
                    test_logits_sum += t_logits.astype(np.float64)
        finally:
            if cleanup_cache and fold_cache.exists():
                shutil.rmtree(fold_cache, ignore_errors=True)

    eval_mask = oof_filled_mask
    oof_probs = logits_to_probs(oof_raw_logits)
    oof_eval = evaluate_predictions(y_all[eval_mask], oof_probs[eval_mask])

    oot_metrics: dict[str, float] | None = None
    if contract.oot_col in folds_df.columns:
        oot_mask = folds_df[contract.oot_col].to_numpy(dtype=bool) & eval_mask
        if np.any(oot_mask) and np.unique(y_all[oot_mask]).size == 2:
            oot_metrics = evaluate_predictions(y_all[oot_mask], oof_probs[oot_mask]).to_dict()

    test_raw_avg: np.ndarray | None = None
    test_probs_avg: np.ndarray | None = None
    if test_logits_sum is not None and test_ids_ref is not None:
        test_raw_avg = (test_logits_sum / len(target_folds)).astype(np.float32)
        test_probs_avg = logits_to_probs(test_raw_avg).astype(np.float32)

    if np.all(oof_filled_mask):
        save_deep_predictions(
            oof_path=oof_output_path,
            row_ids=row_ids_all,
            fold_ids=fold_ids_all,
            clicked=y_all,
            oof_raw_logits=oof_raw_logits,
            oof_pred_probs=oof_probs,
            test_path=test_output_path,
            test_ids=test_ids_ref,
            test_raw_logits=test_raw_avg,
            test_pred_probs=test_probs_avg,
        )

    summary = {
        "evaluated_rows": int(np.sum(eval_mask)),
        "total_rows": int(n_rows),
        "all_folds_completed": bool(np.all(oof_filled_mask)),
        "oof_metrics": oof_eval.to_dict(),
        "oot_metrics": oot_metrics,
        "fold_reports": fold_reports,
        "cat_cols": resolved_cat,
        "num_cols": resolved_num,
        "config": asdict(cfg),
    }

    if report_path is not None:
        rep_p = Path(report_path)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        with rep_p.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)

    return summary
