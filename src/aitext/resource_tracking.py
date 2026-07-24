"""Measures wall-clock time, peak VRAM, and CPU RAM delta around a function call --
feeds the paper's Table 7 (inference time & peak VRAM per model/strategy)."""
from __future__ import annotations

import os
import time
from typing import Any, Callable

import psutil
import torch


def resource_wrapper(fn: Callable, *args, device: str = "cuda", **kwargs) -> tuple[Any, float, float, float]:
    """Runs `fn(*args, **kwargs)` and returns `(result, elapsed_seconds, peak_vram_gb,
    cpu_ram_delta_gb)`."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**3

    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start

    vram_peak = (
        torch.cuda.max_memory_allocated() / 1024**3
        if device == "cuda" and torch.cuda.is_available()
        else 0.0
    )
    mem_after = process.memory_info().rss / 1024**3

    return result, elapsed, vram_peak, mem_after - mem_before
