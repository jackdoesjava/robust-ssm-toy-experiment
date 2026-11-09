"""Worker processes for the tuning and the Bayes filter.

The pool is module state so that the numerical code can use it without passing it around.
Without a pool everything runs in this process, with the same numbers.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))

_pool = None  # None: no worker processes
_workers = 1


@contextmanager
def worker_pool(n):
    """Run the enclosed code with n worker processes (n = 1: none)."""
    global _pool, _workers
    saved = _pool, _workers
    _workers = max(1, n)
    _pool = ProcessPoolExecutor(_workers) if _workers > 1 else None
    try:
        yield
    finally:
        if _pool is not None:
            _pool.shutdown()
        _pool, _workers = saved


def active():
    """Whether there are worker processes."""
    return _pool is not None


def n_workers():
    return _workers


def run_tasks(fn, tasks):
    """fn over tasks, in the worker processes when there are any; results in task order."""
    return list(_pool.map(fn, tasks)) if _pool is not None else [fn(task) for task in tasks]
