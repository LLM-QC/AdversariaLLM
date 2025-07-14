"""
LM Utils - Modular language model utilities

This package provides utilities for language model operations including:
- Text generation (batched and ragged)
- Token processing and conversation handling
- Sampling methods
- JSON schema validation
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
    TokenMergeError
)

# Sampling utilities
from .sampling import top_p_filtering, top_k_filtering

# Text generation interface
from .text_generation import (
    TextGenerator,
    HFLocalTextGen,
    APITextGen,
    GenerationResult,
    CommonGenerateArgs
)

# JSON utilities
from .json_utils import (
    NullFilter,
    JSONFilter,
    validate_json_strings,
    SchemaValidationError,
    forbid_extras
)

# General utilities
from .utils import (
    get_disallowed_ids,
    get_stop_token_ids,
    get_flops
)

__all__ = [
    # Generation
    'generate_ragged',
    'generate_ragged_batched',
    'get_losses_batched',

    # Batching
    'with_max_batchsize',

    # Tokenization
    'prepare_tokens',
    'prepare_conversation',
    'tokenize_chats',
    'get_tokenized_no_attack',
    'get_pre_post_suffix_tokens',
    'generate_random_string',
    'filter_suffix',
    'TokenMergeError',

    # Sampling
    'top_p_filtering',
    'top_k_filtering',

    # Text generation interface
    'TextGenerator',
    'HFLocalTextGen',
    'APITextGen',
    'GenerationResult',
    'CommonGenerateArgs',

    # JSON utilities
    'NullFilter',
    'JSONFilter',
    'validate_json_strings',
    'SchemaValidationError',
    'forbid_extras',

    # Utils
    'get_disallowed_ids',
    'get_stop_token_ids',
    'get_flops',
]