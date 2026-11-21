"""행 단위 안전 정보영역(`feat_a~e`) 결측 패턴 및 과거 인기도(`history_a_*`) 통계 파생 모듈."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from toss_ctr.data.schema import DEFAULT_CONTRACT, DatasetContract, discover_feature_groups

DOMAIN_STAT_GROUPS: tuple[str, ...] = (
    "feat_a",
    "feat_b",
    "feat_c",
    "feat_d",
    "feat_e",
)
PATTERN_GROUPS: tuple[str, ...] = (*DOMAIN_STAT_GROUPS, "history_a")
_EPS: float = 1e-5


def _extract_numeric_cols(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    return [
        c
        for c in cols
        if c in df.columns
        and pd.api.types.is_numeric_dtype(df[c].dtype)
        and not pd.api.types.is_bool_dtype(df[c].dtype)
    ]


def _compute_null_counts(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    n = len(df)
    if not cols or n == 0:
        return np.zeros(n, dtype=np.int16)
    acc = np.zeros(n, dtype=np.int32)
    for col in cols:
        acc += df[col].isna().to_numpy(dtype=np.int32, copy=False)
    return acc.astype(np.int16)


def _compute_group_row_stats(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    chunk_size: int = 500_000,
) -> dict[str, np.ndarray]:
    """경고 없이 전체-NaN 행을 안전 처리하며 청크 단위로 행별 sum/mean/std/min/max를 계산합니다."""
    n = len(df)
    row_sum = np.full(n, np.nan, dtype=np.float32)
    row_mean = np.full(n, np.nan, dtype=np.float32)
    row_std = np.full(n, np.nan, dtype=np.float32)
    row_min = np.full(n, np.nan, dtype=np.float32)
    row_max = np.full(n, np.nan, dtype=np.float32)

    if not numeric_cols or n == 0:
        return {
            "sum": row_sum,
            "mean": row_mean,
            "std": row_std,
            "min": row_min,
            "max": row_max,
        }

    cols_list = list(numeric_cols)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_arr = df[cols_list].iloc[start:end].to_numpy(
            dtype=np.float64, na_value=np.nan, copy=False
        )
        valid_mask = np.isfinite(chunk_arr)
        valid_cnt = valid_mask.sum(axis=1)
        has_valid = valid_cnt > 0

        safe_vals = np.where(valid_mask, chunk_arr, 0.0)
        s_chunk = safe_vals.sum(axis=1)
        denom = np.maximum(valid_cnt, 1)
        m_chunk = s_chunk / denom

        diff = np.where(valid_mask, safe_vals - m_chunk[:, None], 0.0)
        var_chunk = (diff * diff).sum(axis=1) / denom
        std_chunk = np.sqrt(np.maximum(var_chunk, 0.0))

        pos_inf_safe = np.where(valid_mask, chunk_arr, np.inf)
        neg_inf_safe = np.where(valid_mask, chunk_arr, -np.inf)
        min_chunk = pos_inf_safe.min(axis=1)
        max_chunk = neg_inf_safe.max(axis=1)

        row_sum[start:end] = np.where(has_valid, s_chunk, np.nan).astype(np.float32)
        row_mean[start:end] = np.where(has_valid, m_chunk, np.nan).astype(np.float32)
        row_std[start:end] = np.where(has_valid, std_chunk, np.nan).astype(np.float32)
        row_min[start:end] = np.where(has_valid, min_chunk, np.nan).astype(np.float32)
        row_max[start:end] = np.where(has_valid, max_chunk, np.nan).astype(np.float32)

    return {
        "sum": row_sum,
        "mean": row_mean,
        "std": row_std,
        "min": row_min,
        "max": row_max,
    }


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = _EPS) -> np.ndarray:
    out = np.full(len(num), np.nan, dtype=np.float32)
    valid = np.isfinite(num) & np.isfinite(den)
    if np.any(valid):
        n_v = num[valid].astype(np.float64)
        d_v = np.abs(den[valid].astype(np.float64)) + eps
        out[valid] = (n_v / d_v).astype(np.float32)
    return out


def _safe_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.any(valid):
        out[valid] = (a[valid].astype(np.float64) - b[valid].astype(np.float64)).astype(np.float32)
    return out


def _suffix_sort_key(col_name: str, prefix: str) -> tuple[int, int | str]:
    suffix = col_name[len(prefix) :]
    if re.fullmatch(r"\d+", suffix):
        return (0, int(suffix))
    return (1, suffix)


def build_row_statistics(
    df: pd.DataFrame,
    contract: DatasetContract = DEFAULT_CONTRACT,
    ratio_domains: Sequence[str] = ("feat_a", "feat_b", "feat_e"),
    max_matched_pairs: int = 6,
    chunk_size: int = 500_000,
    keep_original: bool = False,
) -> pd.DataFrame:
    """타 행이나 미래 정보를 전혀 참조하지 않는 100% 행 단위 통계 피처를 생성합니다.

    포함 피처:
    1. 정보영역(`feat_a~e`)별 결측 개수(`null_cnt`), 결측 비율, 행 평균/표준편차
    2. 전체 결측 합계(`total_null_cnt`) 및 영역별 동시 결측 패턴 비트마스크(`domain_all_null_mask` 등)
    3. `history_a_*` 행 합계·평균·표준편차·최소·최대·범위
    4. 현재 정보영역(`feat_a` 등) 대비 과거 인기도(`history_a_*`) 비율 및 차이 파생
    """
    n = len(df)
    groups = discover_feature_groups(list(df.columns), contract=contract)

    all_feature_cols: list[str] = []
    for group_name, cols in groups.items():
        _ = group_name
        all_feature_cols.extend(cols)

    out_cols: dict[str, np.ndarray] = {}
    total_null_acc = np.zeros(n, dtype=np.int32)
    domain_all_null_mask = np.zeros(n, dtype=np.int16)
    domain_any_null_mask = np.zeros(n, dtype=np.int16)
    n_domains_all_null = np.zeros(n, dtype=np.int8)
    n_domains_any_null = np.zeros(n, dtype=np.int8)

    domain_means: dict[str, np.ndarray] = {}

    for bit_idx, g_name in enumerate(PATTERN_GROUPS):
        cols = groups.get(g_name, [])
        if not cols:
            continue

        null_cnt = _compute_null_counts(df, cols)
        total_null_acc += null_cnt.astype(np.int32)

        n_cols = len(cols)
        all_null_flag = (null_cnt == n_cols).astype(np.int8)
        any_null_flag = (null_cnt > 0).astype(np.int8)

        out_cols[f"{g_name}_null_cnt"] = null_cnt
        out_cols[f"{g_name}_null_ratio"] = (null_cnt.astype(np.float32) / float(n_cols)).astype(
            np.float32
        )
        out_cols[f"{g_name}_all_null"] = all_null_flag

        bit_val = np.int16(1 << bit_idx)
        domain_all_null_mask |= np.where(all_null_flag == 1, bit_val, np.int16(0))
        domain_any_null_mask |= np.where(any_null_flag == 1, bit_val, np.int16(0))
        n_domains_all_null += all_null_flag
        n_domains_any_null += any_null_flag

        num_cols = _extract_numeric_cols(df, cols)
        if num_cols:
            stats = _compute_group_row_stats(df, num_cols, chunk_size=chunk_size)
            out_cols[f"{g_name}_row_mean"] = stats["mean"]
            out_cols[f"{g_name}_row_std"] = stats["std"]
            domain_means[g_name] = stats["mean"]
            if g_name == "history_a":
                out_cols["history_a_row_sum"] = stats["sum"]
                out_cols["history_a_row_min"] = stats["min"]
                out_cols["history_a_row_max"] = stats["max"]
                out_cols["history_a_row_range"] = _safe_diff(stats["max"], stats["min"])

    # explicit, l_feat, other 결측 합산으로 전체 행 결측 총합 완성
    for extra_group in ("explicit", "l_feat", "other"):
        cols = groups.get(extra_group, [])
        if not cols:
            continue
        null_cnt = _compute_null_counts(df, cols)
        total_null_acc += null_cnt.astype(np.int32)
        if extra_group == "l_feat":
            out_cols["l_feat_null_cnt"] = null_cnt

    n_total_feats = max(len(all_feature_cols), 1)
    out_cols["total_null_cnt"] = total_null_acc.astype(np.int16)
    out_cols["total_null_ratio"] = (total_null_acc.astype(np.float32) / float(n_total_feats)).astype(
        np.float32
    )
    out_cols["domain_all_null_mask"] = domain_all_null_mask
    out_cols["domain_any_null_mask"] = domain_any_null_mask
    out_cols["n_domains_all_null"] = n_domains_all_null
    out_cols["n_domains_any_null"] = n_domains_any_null

    # history_a vs feat_* 영역 평균 차이 및 비율
    if "history_a" in domain_means:
        hist_mean = domain_means["history_a"]
        for dom in ratio_domains:
            if dom in domain_means:
                dom_mean = domain_means[dom]
                out_cols[f"history_a_minus_{dom}_mean"] = _safe_diff(hist_mean, dom_mean)
                out_cols[f"history_a_div_{dom}_mean"] = _safe_ratio(hist_mean, dom_mean)

        # 동일 접미사(예: history_a_1 vs feat_a_1) 1:1 비율 및 차이
        hist_num_cols = sorted(
            _extract_numeric_cols(df, groups.get("history_a", [])),
            key=lambda c: _suffix_sort_key(c, "history_a_"),
        )
        feat_a_num_set = set(_extract_numeric_cols(df, groups.get("feat_a", [])))
        matched_count = 0
        for h_col in hist_num_cols:
            if matched_count >= max_matched_pairs:
                break
            suffix = h_col[len("history_a_") :]
            f_col = f"feat_a_{suffix}"
            if f_col in feat_a_num_set:
                h_arr = df[h_col].to_numpy(dtype=np.float32, na_value=np.nan, copy=False)
                f_arr = df[f_col].to_numpy(dtype=np.float32, na_value=np.nan, copy=False)
                out_cols[f"{h_col}_minus_{f_col}"] = _safe_diff(h_arr, f_arr)
                out_cols[f"{h_col}_div_{f_col}"] = _safe_ratio(h_arr, f_arr)
                matched_count += 1

        if len(hist_num_cols) >= 2:
            h_first = df[hist_num_cols[0]].to_numpy(dtype=np.float32, na_value=np.nan, copy=False)
            h_last = df[hist_num_cols[-1]].to_numpy(dtype=np.float32, na_value=np.nan, copy=False)
            out_cols["history_a_first_last_diff"] = _safe_diff(h_first, h_last)
            out_cols["history_a_first_last_ratio"] = _safe_ratio(h_first, h_last)
            out_cols["history_a_max_vs_mean_ratio"] = _safe_ratio(
                out_cols["history_a_row_max"], hist_mean
            )

    out_df = pd.DataFrame(out_cols, index=df.index)
    if keep_original:
        return pd.concat([df.copy(), out_df], axis=1)
    return out_df
