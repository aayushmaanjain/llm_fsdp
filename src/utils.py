"""Module providing utility functions."""
from time import perf_counter

def timer(func):
    """Decorator to measure time taken by function."""
    def wrap_func(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        result['time'] = perf_counter() - start
        return result
    return wrap_func
