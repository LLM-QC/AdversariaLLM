"""
LM Utils - Modular language model utilities

This package provides utilities for language model operations including:
- Text generation (batched and ragged) with configurable filtering
- Token processing and conversation handling
- Sampling methods
- Filter system (JSON schema enforcement, repetition prevention, custom filters)
- Batch processing with automatic memory management
"""

# Core generation functions
from .generation import generate_ragged, generate_ragged_batched, get_losses_batched

# Batch processing utilities
from .batching import with_max_batchsize

# Tokenization and conversation handling
from .tokenization import (
    prepare_tokens,
    prepare_conversation,
    tokenize_chats,
    get_tokenized_no_attack,
    get_pre_post_suffix_tokens,
    generate_random_string,
    filter_suffix,
    TokenMergeError,
)

# Sampling utilities
from .sampling import top_p_filtering, top_k_filtering

# Text generation interface
from .text_generation import (
    generate_from_prompts,
    generate_with_conv,
    safe_generate_from_prompts,
    safe_generate_with_conv,
    generate_json,
    TextGenerator,
    LocalTextGenerator,
    APITextGenerator,
    GenerationResult,
    CommonGenerateArgs,
    APIRetryOverrides,
    LocalRetryOverrides,
    RetryOverrides
)

# Filter system
from .filters import (
    # Protocol and interface
    FilterProtocol,
    # Filter classes
    NullFilter,
    JSONFilter,
    RepetitionFilter,
    FilterPipeline,
    # Factory functions
    json_filter,
    repetition_filter,
    null_filter,
    # Registry and validation
    FILTER_REGISTRY,
    validate_json_strings,
    SchemaValidationError,
    forbid_extras,
)

# General utilities
from .utils import (
    get_disallowed_ids,
    get_stop_token_ids,
    get_flops,
    select_active_subset,
    update_masked_subset,
    build_single_turn_conversations,
)

__all__ = [
    # Generation
    "generate_ragged",
    "generate_ragged_batched",
    "get_losses_batched",
    # Batching
    "with_max_batchsize",
    # Tokenization
    "prepare_tokens",
    "prepare_conversation",
    "tokenize_chats",
    "get_tokenized_no_attack",
    "get_pre_post_suffix_tokens",
    "generate_random_string",
    "filter_suffix",
    "TokenMergeError",
    # Sampling
    "top_p_filtering",
    "top_k_filtering",
    # Text generation interface
    "generate_from_prompts",
    "generate_with_conv",
    "safe_generate_from_prompts",
    "safe_generate_with_conv",
    "generate_json",
    "TextGenerator",
    "LocalTextGenerator",
    "APITextGenerator",
    "GenerationResult",
    "CommonGenerateArgs",
    # Filter system
    "FilterProtocol",
    "NullFilter",
    "JSONFilter",
    "RepetitionFilter",
    "FilterPipeline",
    "json_filter",
    "repetition_filter",
    "null_filter",
    "FILTER_REGISTRY",
    "validate_json_strings",
    "SchemaValidationError",
    "forbid_extras",
    # Utils
    "get_disallowed_ids",
    "get_stop_token_ids",
    "get_flops",
    "select_active_subset",
    "update_masked_subset",
    "build_single_turn_conversations",
    # Retry
    "APIRetryOverrides",
    "LocalRetryOverrides",
    "RetryOverrides",
    "APIRetryPolicy",
    "LocalRetryPolicy",
    "RetryPolicy",
]
