"""CTR 예측용 피처 엔지니어링(교호작용·행 통계·타겟 인코딩·시퀀스 요약) 모듈."""

from toss_ctr.features.history_agg import (
    DOMAIN_STAT_GROUPS,
    PATTERN_GROUPS,
    build_row_statistics,
)
from toss_ctr.features.interactions import (
    CORE_INTERACTION_PAIRS,
    EXTENDED_INTERACTION_PAIRS,
    FoldConditionalStatsEncoder,
    FoldFrequencyEncoder,
    build_interaction_features,
    make_interaction_column_name,
)
from toss_ctr.features.seq_features import (
    PAD_ID,
    PAD_TOKEN,
    SEQ_SUMMARY_FEATURE_COLS,
    UNK_ID,
    UNK_TOKEN,
    FoldSeqVocabulary,
    detect_seq_delimiter,
    extract_seq_summary_features,
    extract_seq_summary_parquet,
    parse_seq_string,
    stable_token_to_int32,
)
from toss_ctr.features.tabular_pipeline import (
    DEFAULT_CONDITIONAL_SPECS,
    DEFAULT_FREQ_COLS,
    DEFAULT_TE_COLS,
    FoldFeatureBundle,
    TabularFeaturePipeline,
    join_seq_summary,
)
from toss_ctr.features.target_encoding import OutOfFoldTargetEncoder

__all__ = [
    "CORE_INTERACTION_PAIRS",
    "DEFAULT_CONDITIONAL_SPECS",
    "DEFAULT_FREQ_COLS",
    "DEFAULT_TE_COLS",
    "DOMAIN_STAT_GROUPS",
    "EXTENDED_INTERACTION_PAIRS",
    "FoldConditionalStatsEncoder",
    "FoldFeatureBundle",
    "FoldFrequencyEncoder",
    "FoldSeqVocabulary",
    "OutOfFoldTargetEncoder",
    "PAD_ID",
    "PAD_TOKEN",
    "PATTERN_GROUPS",
    "SEQ_SUMMARY_FEATURE_COLS",
    "TabularFeaturePipeline",
    "UNK_ID",
    "UNK_TOKEN",
    "build_interaction_features",
    "build_row_statistics",
    "detect_seq_delimiter",
    "extract_seq_summary_features",
    "extract_seq_summary_parquet",
    "join_seq_summary",
    "make_interaction_column_name",
    "parse_seq_string",
    "stable_token_to_int32",
]
