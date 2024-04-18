"""Module providing utility functions."""
from time import perf_counter

import torch

def timer(func):
    """Decorator to measure time taken by function."""
    def wrap_func(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        result['time'] = perf_counter() - start
        return result
    return wrap_func

def free_cuda_memory():
    """Utility functioon to free up cuda memory."""
    with torch.no_grad():
        torch.cuda.empty_cache()
