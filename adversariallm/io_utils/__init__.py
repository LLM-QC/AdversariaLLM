"""
IO Utils - Modular I/O and data management utilities

This package provides utilities for:
- Model and tokenizer loading
- Database operations and MongoDB integration
- Attack result logging and management
- JSON encoding/decoding and file operations
- Data analysis and result collection
- Memory management
- Configuration management
"""

# Model loading utilities
from .model_loading import load_model_and_tokenizer, load_chat_template, num_model_params

# Database operations
from .database import (
    get_mongodb_connection,
    log_config_to_db,
    delete_orphaned_runs,
    check_match,
    get_filtered_and_grouped_paths
)

# Logging utilities
from .logging import log_attack, offload_tensors

# JSON utilities
from .json_utils import CompactJSONEncoder, cached_json_load

# Data analysis utilities
from .data_analysis import (
    collect_results,
    normalize_value_for_grouping,
    get_nested_value,
    load_embedding
)

# Memory management
from .memory import free_vram

# Configuration utilities
from .config import RunConfig, filter_config

__all__ = [
    # Model loading
    'load_model_and_tokenizer',
    'load_chat_template',
    'num_model_params',
    
    # Database
    'get_mongodb_connection',
    'log_config_to_db',
    'delete_orphaned_runs',
    'check_match',
    'get_filtered_and_grouped_paths',
    
    # Logging
    'log_attack',
    'offload_tensors',
    
    # JSON utilities
    'CompactJSONEncoder',
    'cached_json_load',
    
    # Data analysis
    'collect_results',
    'normalize_value_for_grouping',
    'get_nested_value',
    'load_embedding',
    
    # Memory management
    'free_vram',
    
    # Configuration
    'RunConfig',
    'filter_config',
]