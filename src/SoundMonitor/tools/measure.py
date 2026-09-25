import time

import functools


def time_it(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High-resolution clock
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds")
        return result

    return wrapper
