
import cProfile, pstats, io
from functools import wraps
from collections import defaultdict
from mpi4py import MPI

_profiled_stats = defaultdict(list)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ENABLE_PROFILING = True

def profile_this(func):
    """
    Decorator to profile a function using cProfile.
    If profiling is disabled, the function runs normally without profiling.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not ENABLE_PROFILING:
            return func(*args, **kwargs)

        pr = cProfile.Profile()
        pr.enable()
        try:
            return func(*args, **kwargs)
        finally:
            pr.disable()
            _profiled_stats[func.__qualname__].append(pr)
    return wrapper

def print_profiled_summary(limit=10):
    """
    Print a summary of the profiled functions.
    This function aggregates profiling data across all ranks and prints it only from rank 0.
    """

    if rank != 0 or not ENABLE_PROFILING:
        return
    print("\n=== Profiling Summary ===")
    for func_name, profiles in _profiled_stats.items():
        combined = pstats.Stats()
        for pr in profiles:
            combined.add(pr)
        stream = io.StringIO()
        combined.strip_dirs().sort_stats('cumulative').print_stats(limit)
        combined.stream = stream
        combined.print_stats(limit)
        print(f"\n--- {func_name} ---")
        print(stream.getvalue())


if __name__ == "__main__":
    @profile_this
    def foo():
        for _ in range(10000):
            sum([i**2 for i in range(100)])

    @profile_this
    def bar():
        for _ in range(10000):
            sum([i for i in range(100)])

    foo()
    bar()
    foo()