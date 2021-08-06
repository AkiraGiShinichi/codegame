from functools import wraps
from time import time


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Start timer
        t_start = time()

        try:
            # 2. Execute function
            return func(*args, **kwargs)
        finally:
            # 3. Measure time
            t_total = time.time() - t_start
            print(f'Total execution time: {func.__name__} took {t_total}(s)')
    return wrapper


# @timing
def foo():
    """Just say foo!
    """
    time.sleep(10)
    print('foo!')


foo()
