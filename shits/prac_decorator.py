from functools import wraps
import time


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Start timer
        t_start = time.time()

        # 2. Execute function
        result = func(*args, **kwargs)

        # 3. Measure time
        t_total = time.time() - t_start
        print('{func.__name__} took {t_total}(s)')

        return result
    return wrapper


# @timer
def foo():
    """Just say foo!
    """
    time.sleep(10)
    print('foo!')


foo()
