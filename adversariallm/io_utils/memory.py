"""
Memory management utilities.

This module provides functions for managing GPU and system memory,
particularly for CUDA operations.
"""

import gc
import torch


def free_vram():
    for _ in range(2):
        gc.collect()
        torch.cuda.empty_cache()