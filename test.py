import time
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds to execute.")
        return result
    return wrapper


import numpy as np

@timeit
def calculations(n):
    sum = 0
    for i in range(n+1):
        sum += i
    print(sum)



@timeit
def calculations_vectorized(n):
    sum = np.sum(np.arange(n+1))
    print(sum)

calculations(100000000)
calculations_vectorized(100000000)