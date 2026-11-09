"""Worker processes give the same numbers as one process.

The two tests marked needs_workers use real worker processes and are skipped without
--workers 2 or more; the others run the same pool code with an in-process stand-in, so it is
checked with any --workers.
"""

import numpy as np
import pytest

from toy_experiment import parallel, tuning
from toy_experiment.bayes import bayes_filter, exact_filter
from toy_experiment.config import EPS, LAM
from toy_experiment.simulate import simulate_rw
from toy_experiment.tuning import fit_all

from .helpers import FakeScores, InProcessPool, fabricated_tables, use_pool


def same_scores(a, b):
    return set(a) == set(b) and all(np.array_equal(a[key], b[key]) for key in b)


def pool_test_data():
    """The data of the two worker-process tests, drawn from one generator."""
    rng = np.random.default_rng(16)
    bayes = [(Q, stay, simulate_rw(rng, 3 * parallel.n_workers() + 1, 120, Q, EPS, LAM, stay, 60))
             for Q, stay in ((0.1, None), (1.0, 0.8))]
    _, y_tr, _ = simulate_rw(rng, 3, 100, 0.1, EPS, LAM, None, None)
    _, y_va, _ = simulate_rw(rng, 3, 100, 0.1, EPS, LAM, None, None)
    return bayes, y_tr, y_va


@pytest.mark.needs_workers
def test_bayes_filter_split_across_worker_processes_equals_one_call():
    """Independent outliers at Q = 0.1; bursts at Q = 1; uneven chunks."""
    assert parallel.active()
    for Q, stay, (x, y, _) in pool_test_data()[0]:
        assert same_scores(bayes_filter(y, x, Q, EPS, LAM, stay), exact_filter(y, x, Q, EPS, LAM, stay))


@pytest.mark.needs_workers
def test_fit_all_in_worker_processes_equals_fit_all_in_one_process(monkeypatch):
    assert parallel.active()
    _, y_tr, y_va = pool_test_data()
    in_pool = fit_all(y_tr, y_va, 0.1)
    use_pool(monkeypatch, None, 1)
    assert in_pool == fit_all(y_tr, y_va, 0.1)


@pytest.mark.parametrize("n_workers, Q, stay, seed", [(2, 0.1, None, 61), (3, 0.1, 0.8, 62), (3, 1.0, None, 63),
                                                      (6, 0.1, 0.8, 64)])
def test_bayes_filter_through_a_stand_in_pool_equals_one_call(monkeypatch, n_workers, Q, stay, seed):
    """In 2, 3 or 6 uneven chunks, with bursts and with Q = 1."""
    x, y, _ = simulate_rw(np.random.default_rng(seed), 2 * n_workers + 1, 40, Q, EPS, LAM, stay, 20)
    pool = InProcessPool()
    use_pool(monkeypatch, pool, n_workers)
    split = bayes_filter(y, x, Q, EPS, LAM, stay)
    assert (pool.calls, pool.tasks) == (1, n_workers)  # the stand-in did the work, one chunk per worker
    assert same_scores(split, exact_filter(y, x, Q, EPS, LAM, stay))


def test_fit_all_through_the_pool_branch_of_run_tasks_equals_fit_all_without_a_pool(monkeypatch):
    S, Sp = fabricated_tables({(9, 1.5, 3.0): 0.5}, {(3, 2.0, 2.0): 0.45}, 7)
    fake = FakeScores(S, Sp)
    monkeypatch.setattr(tuning, "_scores_over_c", fake)  # where fit_all looks it up
    y = np.random.default_rng(5).normal(size=(2, 20))
    use_pool(monkeypatch, None, 1)
    alone = fit_all(y, y, 0.1)
    pool = InProcessPool()
    use_pool(monkeypatch, pool, 4)
    pooled = fit_all(y, y, 0.1)
    assert pool.calls == 1 and fake.calls == 2 * pool.tasks  # both fits read the fabricated tables
    assert alone == pooled
