"""유저 서버 로그 시퀀스(`seq`) 파싱, 경량 요약 피처 스트리밍 추출 및 폴드 전용 어휘집 모듈."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
PAD_ID: int = 0
UNK_ID: int = 1

DEFAULT_DELIMITER_CANDIDATES: tuple[str, ...] = (",", "|", ";", "\t")

SEQ_SUMMARY_FEATURE_COLS: tuple[str, ...] = (
    "seq_is_empty",
    "seq_char_len",
    "seq_token_len",
    "seq_unique_cnt",
    "seq_unique_ratio",
    "seq_last_1",
    "seq_last_2",
    "seq_last_3",
    "seq_first_1",
    "seq_top1_share",
    "seq_recent5_unique_ratio",
)


def _is_missing_scalar(value: Any) -> bool:
    """리스트/배열 예외 없이 스칼라 결측(`None`, `NaN`, `pd.NA`, `NaT`)을 판별합니다."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _coerce_seq_text(seq_value: Any) -> str | None:
    """입력 시퀀스 값을 공백 제거된 문자열로 정규화합니다. 결측이면 `None`을 반환합니다."""
    if isinstance(seq_value, pa.Scalar):
        seq_value = seq_value.as_py()
    if _is_missing_scalar(seq_value):
        return None
    if isinstance(seq_value, bytes):
        return seq_value.decode("utf-8", errors="ignore").strip()
    if isinstance(seq_value, str):
        return seq_value.strip()
    return str(seq_value).strip()


def detect_seq_delimiter(
    samples: str | Iterable[Any] | pd.Series | pa.Array | pa.ChunkedArray | None,
    candidates: Sequence[str] = DEFAULT_DELIMITER_CANDIDATES,
    max_samples: int = 1_000,
) -> str | None:
    """샘플 시퀀스 컬렉션에서 가장 일관된 구분자를 자동 감지합니다.

    후보 구분자(`,`, `|`, `;`, `\\t`)가 발견되면 최다 빈도 구분자를 반환하고,
    후보 구분자 없이 공백으로 구분된 다중 토큰이 존재하면 `" "`을 반환합니다.
    구분자가 전혀 감지되지 않거나 비어 있는 입력이면 `None`을 반환합니다.
    """
    if samples is None:
        return None

    if isinstance(samples, (str, bytes)):
        iterable: Iterable[Any] = [samples]
    elif isinstance(samples, (pa.Array, pa.ChunkedArray)):
        iterable = samples.to_pylist()
    elif isinstance(samples, pd.Series):
        iterable = samples.tolist()
    else:
        iterable = samples

    counts: dict[str, int] = {cand: 0 for cand in candidates}
    has_whitespace_split = False
    inspected = 0

    for item in iterable:
        if inspected >= max_samples:
            break
        if isinstance(item, (list, tuple, np.ndarray)):
            continue
        text = _coerce_seq_text(item)
        if not text:
            continue
        inspected += 1

        matched_cand = False
        for cand in candidates:
            cnt = text.count(cand)
            if cnt > 0:
                counts[cand] += cnt
                matched_cand = True
        if not matched_cand and len(text.split()) > 1:
            has_whitespace_split = True

    best_cand = max(candidates, key=lambda c: counts[c], default="")
    if best_cand and counts.get(best_cand, 0) > 0:
        return best_cand
    if has_whitespace_split:
        return " "
    return None


def parse_seq_string(
    seq_value: Any,
    delimiter: str | None = "auto",
) -> list[str]:
    """원시 `seq` 값을 안전하게 토큰 리스트로 파싱합니다.

    쉼표(`,`), 공백, 세미콜론(`;`), 파이프(`|`), 탭(`\\t`) 구분자를 지원하며,
    `None`, `NaN`, `pd.NA`, 빈 문자열(`""`) 입력은 빈 리스트 `[]`로 반환합니다.
    """
    if isinstance(seq_value, pa.Scalar):
        seq_value = seq_value.as_py()

    if isinstance(seq_value, (list, tuple, np.ndarray)):
        tokens: list[str] = []
        for elem in seq_value:
            text_elem = _coerce_seq_text(elem)
            if text_elem:
                tokens.append(text_elem)
        return tokens

    text = _coerce_seq_text(seq_value)
    if not text:
        return []

    resolved_delim = delimiter
    if resolved_delim is None or resolved_delim == "auto":
        best_cand = ""
        best_count = 0
        for cand in DEFAULT_DELIMITER_CANDIDATES:
            cnt = text.count(cand)
            if cnt > best_count:
                best_count = cnt
                best_cand = cand
        resolved_delim = best_cand if best_count > 0 else " "

    if resolved_delim == " ":
        return [tok for tok in text.split() if tok]

    return [tok.strip() for tok in text.split(resolved_delim) if tok.strip()]


def stable_token_to_int32(token: str) -> int:
    """문자열 토큰을 실행 환경과 무관한 결정론적 양의 `int32` ID(`[1, 2^31 - 1]`)로 변환합니다.

    `0`은 빈 시퀀스/패딩(`PAD = 0`) 전용으로 예약하므로 결과값은 항상 `1` 이상입니다.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    val = int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF
    return 1 if val == 0 else val


def _resolve_token_id_cached(token: str, cache: dict[str, int]) -> int:
    cached = cache.get(token)
    if cached is not None:
        return cached
    if len(cache) >= 500_000:
        cache.clear()
    val = stable_token_to_int32(token)
    cache[token] = val
    return val


def _extract_summary_arrays_from_values(
    seq_values: Sequence[Any],
    delimiter: str | None = "auto",
    empty_token_id: int = 0,
    token_cache: dict[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], Counter[int]]:
    """시퀀스 값 시퀀스로부터 경량 요약 피처 NumPy 배열과 길이 분포 카운터를 생성합니다."""
    n_rows = len(seq_values)
    resolved_delim = (
        detect_seq_delimiter(seq_values)
        if (delimiter is None or delimiter == "auto")
        else delimiter
    )
    cache = token_cache if token_cache is not None else {}

    seq_is_empty = np.ones(n_rows, dtype=np.int8)
    seq_char_len = np.zeros(n_rows, dtype=np.int32)
    seq_token_len = np.zeros(n_rows, dtype=np.int32)
    seq_unique_cnt = np.zeros(n_rows, dtype=np.int32)
    seq_unique_ratio = np.zeros(n_rows, dtype=np.float32)
    seq_last_1 = np.full(n_rows, empty_token_id, dtype=np.int32)
    seq_last_2 = np.full(n_rows, empty_token_id, dtype=np.int32)
    seq_last_3 = np.full(n_rows, empty_token_id, dtype=np.int32)
    seq_first_1 = np.full(n_rows, empty_token_id, dtype=np.int32)
    seq_top1_share = np.zeros(n_rows, dtype=np.float32)
    seq_recent5_unique_ratio = np.zeros(n_rows, dtype=np.float32)

    len_counter: Counter[int] = Counter()

    for idx, raw_val in enumerate(seq_values):
        if isinstance(raw_val, (list, tuple, np.ndarray)):
            tokens = parse_seq_string(raw_val, delimiter=resolved_delim)
            char_len = sum(len(t) for t in tokens)
        else:
            text = _coerce_seq_text(raw_val)
            if not text:
                len_counter[0] += 1
                continue
            char_len = len(text)
            if resolved_delim is None or resolved_delim == " ":
                tokens = text.split()
            else:
                tokens = [t.strip() for t in text.split(resolved_delim) if t.strip()]

        tok_len = len(tokens)
        seq_char_len[idx] = char_len
        len_counter[tok_len] += 1

        if tok_len == 0:
            continue

        seq_is_empty[idx] = 0
        seq_token_len[idx] = tok_len

        counts = Counter(tokens)
        uniq_cnt = len(counts)
        seq_unique_cnt[idx] = uniq_cnt
        seq_unique_ratio[idx] = np.float32(uniq_cnt / tok_len)

        top1_cnt = max(counts.values())
        seq_top1_share[idx] = np.float32(top1_cnt / tok_len)

        recent5 = tokens[-5:]
        seq_recent5_unique_ratio[idx] = np.float32(len(set(recent5)) / len(recent5))

        seq_first_1[idx] = _resolve_token_id_cached(tokens[0], cache)
        seq_last_1[idx] = _resolve_token_id_cached(tokens[-1], cache)
        if tok_len >= 2:
            seq_last_2[idx] = _resolve_token_id_cached(tokens[-2], cache)
        if tok_len >= 3:
            seq_last_3[idx] = _resolve_token_id_cached(tokens[-3], cache)

    arrays: dict[str, np.ndarray] = {
        "seq_is_empty": seq_is_empty,
        "seq_char_len": seq_char_len,
        "seq_token_len": seq_token_len,
        "seq_unique_cnt": seq_unique_cnt,
        "seq_unique_ratio": seq_unique_ratio,
        "seq_last_1": seq_last_1,
        "seq_last_2": seq_last_2,
        "seq_last_3": seq_last_3,
        "seq_first_1": seq_first_1,
        "seq_top1_share": seq_top1_share,
        "seq_recent5_unique_ratio": seq_recent5_unique_ratio,
    }
    return arrays, len_counter


def extract_seq_summary_features(
    df: pd.DataFrame,
    id_col: str | None = "row_id",
    seq_col: str = "seq",
    delimiter: str | None = "auto",
    empty_token_id: int = 0,
) -> pd.DataFrame:
    """DataFrame의 `seq` 컬럼에서 GBDT 조인용 경량 시퀀스 요약 피처 11종을 추출합니다.

    `id_col`이 존재하면 첫 번째 컬럼으로 원본 순서와 dtype을 그대로 보존합니다.
    """
    if seq_col not in df.columns:
        raise ValueError(f"DataFrame에 '{seq_col}' 컬럼이 존재하지 않습니다.")

    seq_values = df[seq_col].tolist()
    arrays, _ = _extract_summary_arrays_from_values(
        seq_values,
        delimiter=delimiter,
        empty_token_id=empty_token_id,
    )

    out_data: dict[str, Any] = {}
    if id_col is not None and id_col in df.columns:
        out_data[id_col] = df[id_col].to_numpy()
    for col in SEQ_SUMMARY_FEATURE_COLS:
        out_data[col] = arrays[col]

    return pd.DataFrame(out_data, index=df.index)


def _quantile_from_counter(counter: Counter[int], q: float) -> float:
    """빈도 카운터로부터 메모리 추가 사용 없이 분위수를 계산합니다."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    target_rank = q * (total - 1)
    low_idx = int(np.floor(target_rank))
    high_idx = int(np.ceil(target_rank))
    frac = target_rank - low_idx

    sorted_items = sorted(counter.items(), key=lambda kv: kv[0])
    cumulative = 0
    low_val = sorted_items[-1][0]
    high_val = sorted_items[-1][0]
    found_low = False

    for length_val, freq in sorted_items:
        cumulative += freq
        if not found_low and cumulative > low_idx:
            low_val = length_val
            found_low = True
        if cumulative > high_idx:
            high_val = length_val
            break

    return float(low_val + (high_val - low_val) * frac)


def extract_seq_summary_parquet(
    seq_parquet_path: str | Path,
    output_path: str | Path,
    id_col: str | None = None,
    seq_col: str = "seq",
    delimiter: str | None = "auto",
    batch_size: int = 131_072,
    compression: str = "zstd",
    compression_level: int | None = 3,
    empty_token_id: int = 0,
) -> dict[str, Any]:
    """대용량 `*_seq.parquet`을 청크 스트리밍으로 읽어 요약 피처 Parquet을 OOM 없이 저장합니다.

    `id_col=None`이면 스키마에서 `"row_id"` 또는 `"ID"`를 자동 탐색하여 순서를 100% 보존합니다.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size는 1 이상이어야 합니다: {batch_size}")

    src_path = Path(seq_parquet_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(src_path)
    schema_names = pf.schema_arrow.names
    if seq_col not in schema_names:
        raise ValueError(f"Parquet 파일({src_path})에 '{seq_col}' 컬럼이 없습니다.")

    resolved_id_col = id_col
    if resolved_id_col is None:
        if "row_id" in schema_names:
            resolved_id_col = "row_id"
        elif "ID" in schema_names:
            resolved_id_col = "ID"

    if resolved_id_col is not None and resolved_id_col not in schema_names:
        raise ValueError(f"Parquet 파일({src_path})에 식별자 '{resolved_id_col}' 컬럼이 없습니다.")

    read_cols = [resolved_id_col, seq_col] if resolved_id_col is not None else [seq_col]

    out_fields: list[pa.Field] = []
    if resolved_id_col is not None:
        src_id_field = pf.schema_arrow.field(resolved_id_col)
        out_fields.append(
            pa.field(resolved_id_col, src_id_field.type, nullable=src_id_field.nullable)
        )

    feature_arrow_types: dict[str, pa.DataType] = {
        "seq_is_empty": pa.int8(),
        "seq_char_len": pa.int32(),
        "seq_token_len": pa.int32(),
        "seq_unique_cnt": pa.int32(),
        "seq_unique_ratio": pa.float32(),
        "seq_last_1": pa.int32(),
        "seq_last_2": pa.int32(),
        "seq_last_3": pa.int32(),
        "seq_first_1": pa.int32(),
        "seq_top1_share": pa.float32(),
        "seq_recent5_unique_ratio": pa.float32(),
    }
    for col in SEQ_SUMMARY_FEATURE_COLS:
        out_fields.append(pa.field(col, feature_arrow_types[col], nullable=False))
    target_schema = pa.schema(out_fields)

    writer_kwargs: dict[str, Any] = {"compression": compression.lower()}
    if compression_level is not None and writer_kwargs["compression"] in {"zstd", "gzip", "brotli"}:
        writer_kwargs["compression_level"] = int(compression_level)

    total_rows = 0
    empty_rows = 0
    total_len_counter: Counter[int] = Counter()
    token_cache: dict[str, int] = {}
    locked_delimiter: str | None = None if delimiter == "auto" else delimiter

    with pq.ParquetWriter(out_path, target_schema, **writer_kwargs) as writer:
        for record_batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
            n_batch = record_batch.num_rows
            if n_batch == 0:
                continue

            seq_list = record_batch.column(seq_col).to_pylist()
            if locked_delimiter is None and delimiter == "auto":
                detected = detect_seq_delimiter(seq_list)
                if detected is not None:
                    locked_delimiter = detected

            batch_delim = locked_delimiter if locked_delimiter is not None else delimiter
            arrays, batch_counter = _extract_summary_arrays_from_values(
                seq_list,
                delimiter=batch_delim,
                empty_token_id=empty_token_id,
                token_cache=token_cache,
            )

            pa_cols: list[pa.Array | pa.ChunkedArray] = []
            if resolved_id_col is not None:
                pa_cols.append(record_batch.column(resolved_id_col))
            for col in SEQ_SUMMARY_FEATURE_COLS:
                pa_cols.append(pa.array(arrays[col], type=feature_arrow_types[col]))

            writer.write_table(pa.Table.from_arrays(pa_cols, schema=target_schema))

            total_rows += n_batch
            empty_rows += int(np.sum(arrays["seq_is_empty"] == 1))
            total_len_counter.update(batch_counter)

        if total_rows == 0:
            writer.write_table(target_schema.empty_table())

    total_tokens = sum(length * cnt for length, cnt in total_len_counter.items())
    max_len = max(total_len_counter.keys()) if total_len_counter else 0

    return {
        "source_path": str(src_path),
        "output_path": str(out_path),
        "id_col": resolved_id_col,
        "rows": int(total_rows),
        "empty_rows": int(empty_rows),
        "empty_ratio": round(empty_rows / total_rows, 6) if total_rows > 0 else 0.0,
        "mean_token_len": round(total_tokens / total_rows, 4) if total_rows > 0 else 0.0,
        "p50_token_len": _quantile_from_counter(total_len_counter, 0.50),
        "p90_token_len": _quantile_from_counter(total_len_counter, 0.90),
        "p95_token_len": _quantile_from_counter(total_len_counter, 0.95),
        "p99_token_len": _quantile_from_counter(total_len_counter, 0.99),
        "max_token_len": int(max_len),
        "detected_delimiter": locked_delimiter,
        "output_size_mb": round(out_path.stat().st_size / (1024 ** 2), 3),
    }


class FoldSeqVocabulary:
    """학습 폴드 데이터만으로 구축해 타겟/어휘 누수를 차단하는 시퀀스 어휘집.

    - `PAD = 0`: 패딩 및 빈 위치 전용 ID
    - `UNK = 1`: 학습 폴드에서 `min_freq` 미만이거나 검증/테스트 전용 신규 토큰 ID
    - Tail Truncation (`tokens[-max_seq_len:]`)을 기본 적용해 최근 행동 로그를 보존합니다.
    """

    PAD_ID: int = PAD_ID
    UNK_ID: int = UNK_ID

    def __init__(
        self,
        min_freq: int = 5,
        max_vocab_size: int = 100_000,
        delimiter: str | None = "auto",
    ) -> None:
        if min_freq < 1:
            raise ValueError(f"min_freq는 1 이상이어야 합니다: {min_freq}")
        if max_vocab_size < 2:
            raise ValueError(f"max_vocab_size는 PAD/UNK 포함 2 이상이어야 합니다: {max_vocab_size}")
        self.min_freq = int(min_freq)
        self.max_vocab_size = int(max_vocab_size)
        self.delimiter = delimiter

        self.token_to_id: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.id_to_token: list[str] = [PAD_TOKEN, UNK_TOKEN]
        self.resolved_delimiter_: str | None = None if delimiter == "auto" else delimiter
        self.is_fitted_: bool = False

    @property
    def vocab_size(self) -> int:
        """PAD(0) 및 UNK(1)을 포함한 전체 어휘 크기."""
        return len(self.id_to_token)

    def _reset_vocab(self) -> None:
        self.token_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self.id_to_token = [PAD_TOKEN, UNK_TOKEN]

    def _finalize_from_counter(self, counter: Counter[str]) -> FoldSeqVocabulary:
        self._reset_vocab()
        max_real_tokens = self.max_vocab_size - 2
        eligible = [
            (tok, freq)
            for tok, freq in counter.items()
            if freq >= self.min_freq and tok not in (PAD_TOKEN, UNK_TOKEN)
        ]
        # 빈도 내림차순 + 동일 빈도 시 토큰 사전순 정렬로 100% 결정론적 ID 할당
        eligible.sort(key=lambda item: (-item[1], item[0]))

        for tok, _ in eligible[:max_real_tokens]:
            idx = len(self.id_to_token)
            self.token_to_id[tok] = idx
            self.id_to_token.append(tok)

        self.is_fitted_ = True
        return self

    def fit(self, train_seqs: Iterable[Any]) -> FoldSeqVocabulary:
        """오직 학습 폴드 시퀀스만 순회해 빈도 기반 어휘집을 구축합니다."""
        seq_list = list(train_seqs)
        if self.delimiter == "auto":
            self.resolved_delimiter_ = detect_seq_delimiter(seq_list)
        else:
            self.resolved_delimiter_ = self.delimiter

        delim = self.resolved_delimiter_
        counter: Counter[str] = Counter()
        for seq_val in seq_list:
            tokens = parse_seq_string(seq_val, delimiter=delim)
            if tokens:
                counter.update(tokens)

        return self._finalize_from_counter(counter)

    def fit_from_parquet_mask(
        self,
        seq_parquet_path: str | Path,
        train_mask: np.ndarray,
        seq_col: str = "seq",
        batch_size: int = 262_144,
    ) -> FoldSeqVocabulary:
        """대용량 `train_seq.parquet`에서 `train_mask`가 `True`인 행만 스트리밍으로 읽어 어휘집을 구축합니다."""
        mask_arr = np.asarray(train_mask, dtype=bool)
        pf = pq.ParquetFile(seq_parquet_path)
        if pf.metadata.num_rows != len(mask_arr):
            raise ValueError(
                f"Parquet 행 수({pf.metadata.num_rows})와 train_mask 길이({len(mask_arr)})가 다릅니다."
            )

        self.resolved_delimiter_ = None if self.delimiter == "auto" else self.delimiter
        counter: Counter[str] = Counter()
        offset = 0
        seen_whitespace_split = False

        for record_batch in pf.iter_batches(batch_size=batch_size, columns=[seq_col]):
            n_batch = record_batch.num_rows
            batch_mask = mask_arr[offset : offset + n_batch]
            offset += n_batch
            if not np.any(batch_mask):
                continue

            seq_list = record_batch.column(seq_col).to_pylist()
            selected = [seq_list[i] for i in np.flatnonzero(batch_mask)]

            # 첫 배치에 단일 토큰/결측만 있는 경우 후속 배치에서도 구분자 자동 감지를 지속합니다.
            if self.resolved_delimiter_ is None and self.delimiter == "auto":
                detected = detect_seq_delimiter(selected)
                if detected in DEFAULT_DELIMITER_CANDIDATES:
                    self.resolved_delimiter_ = detected
                elif detected == " ":
                    seen_whitespace_split = True

            delim = self.resolved_delimiter_ if self.resolved_delimiter_ is not None else (
                " " if seen_whitespace_split else self.delimiter
            )
            for seq_val in selected:
                tokens = parse_seq_string(seq_val, delimiter=delim)
                if tokens:
                    counter.update(tokens)

        if self.resolved_delimiter_ is None and self.delimiter == "auto" and seen_whitespace_split:
            self.resolved_delimiter_ = " "

        return self._finalize_from_counter(counter)

    def encode_sequence(
        self,
        seq_value: Any,
        max_seq_len: int = 100,
        truncating: Literal["pre", "post"] = "pre",
    ) -> list[int]:
        """단일 시퀀스를 정수 토큰 ID 리스트로 변환합니다.

        기본값 `truncating="pre"`는 앞쪽 과거 토큰을 자르고 최근 로그(`tokens[-max_seq_len:]`)를 보존합니다.
        """
        if not self.is_fitted_:
            raise RuntimeError("FoldSeqVocabulary.fit()을 먼저 호출해야 합니다.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len은 1 이상이어야 합니다: {max_seq_len}")

        tokens = parse_seq_string(seq_value, delimiter=self.resolved_delimiter_)
        if len(tokens) > max_seq_len:
            tokens = tokens[-max_seq_len:] if truncating == "pre" else tokens[:max_seq_len]

        t2i = self.token_to_id
        return [t2i.get(tok, UNK_ID) for tok in tokens]

    def encode_batch(
        self,
        seqs: Sequence[Any] | Iterable[Any],
        max_seq_len: int = 100,
        padding: Literal["post", "pre"] = "post",
        truncating: Literal["pre", "post"] = "pre",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """시퀀스 배치를 고정 길이 `(padded_ids, seq_mask, seq_lengths)` NumPy 배열로 인코딩합니다.

        Args:
            seqs: 인코딩할 시퀀스 배치.
            max_seq_len: 최대 시퀀스 길이.
            padding: `"post"`(뒤쪽 패딩) 또는 `"pre"`(앞쪽 패딩).
            truncating: `"pre"`(최근 `max_seq_len`개 보존 Tail Truncation) 또는 `"post"`.

        Returns:
            `(padded_ids, seq_mask, seq_lengths)` 튜플:
            - `padded_ids`: `np.int32`, shape `(B, max_seq_len)`
            - `seq_mask`: `np.bool_`, shape `(B, max_seq_len)` (`True` = 유효 토큰, `False` = `PAD`)
            - `seq_lengths`: `np.int32`, shape `(B,)`
        """
        if not self.is_fitted_:
            raise RuntimeError("FoldSeqVocabulary.fit()을 먼저 호출해야 합니다.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len은 1 이상이어야 합니다: {max_seq_len}")
        if padding not in ("post", "pre"):
            raise ValueError(f"padding은 'post' 또는 'pre'여야 합니다: {padding!r}")
        if truncating not in ("pre", "post"):
            raise ValueError(f"truncating은 'pre' 또는 'post'여야 합니다: {truncating!r}")

        seq_items = list(seqs) if not isinstance(seqs, (list, tuple, np.ndarray, pd.Series)) else seqs
        n_batch = len(seq_items)

        padded_ids = np.zeros((n_batch, max_seq_len), dtype=np.int32)
        seq_mask = np.zeros((n_batch, max_seq_len), dtype=bool)
        seq_lengths = np.zeros(n_batch, dtype=np.int32)

        t2i = self.token_to_id
        delim = self.resolved_delimiter_

        for row_idx, seq_val in enumerate(seq_items):
            tokens = parse_seq_string(seq_val, delimiter=delim)
            if not tokens:
                continue
            if len(tokens) > max_seq_len:
                tokens = tokens[-max_seq_len:] if truncating == "pre" else tokens[:max_seq_len]

            length = len(tokens)
            seq_lengths[row_idx] = length
            ids = [t2i.get(tok, UNK_ID) for tok in tokens]
            if padding == "post":
                padded_ids[row_idx, :length] = ids
                seq_mask[row_idx, :length] = True
            else:
                padded_ids[row_idx, max_seq_len - length :] = ids
                seq_mask[row_idx, max_seq_len - length :] = True

        return padded_ids, seq_mask, seq_lengths

    def to_dict(self) -> dict[str, Any]:
        """어휘집 상태를 직렬화 가능한 딕셔너리로 반환합니다."""
        return {
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "delimiter": self.delimiter,
            "resolved_delimiter": self.resolved_delimiter_,
            "id_to_token": self.id_to_token,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FoldSeqVocabulary:
        """직렬화된 딕셔너리로부터 `FoldSeqVocabulary`를 복원합니다."""
        vocab = cls(
            min_freq=int(payload["min_freq"]),
            max_vocab_size=int(payload["max_vocab_size"]),
            delimiter=payload.get("delimiter", "auto"),
        )
        id_to_token = list(payload["id_to_token"])
        vocab.id_to_token = id_to_token
        vocab.token_to_id = {tok: idx for idx, tok in enumerate(id_to_token)}
        vocab.resolved_delimiter_ = payload.get("resolved_delimiter")
        vocab.is_fitted_ = True
        return vocab

    def save(self, path: str | Path) -> None:
        """어휘집을 JSON 파일로 저장합니다."""
        out_p = Path(path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> FoldSeqVocabulary:
        """JSON 파일에서 어휘집을 로드합니다."""
        with Path(path).open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return cls.from_dict(payload)
