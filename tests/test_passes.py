"""The repeated frozen-weight passes of experiment 2, and count_passes."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from toy_experiment.config import R2, SIGMA2, TOLS
from toy_experiment.filters import kalman_filter
from toy_experiment.passes import count_passes, frozen_passes
from toy_experiment.simulate import simulate_selective

from .helpers import scalar_filter


@pytest.fixture(scope="module")
def passes():
    """The first 61 passes on 3 sequences of 60 steps, with outliers at the start so the first
    weights matter, and the sequential robust and ordinary Kalman filters."""
    _, y, a, Qs = simulate_selective(np.random.default_rng(14), 3, 60, 0.9)
    y[:, :3] += 6.0
    return SimpleNamespace(seq=kalman_filter(y, R2, Qs, a, c=1.0, P0=SIGMA2),
                           plain=kalman_filter(y, R2, Qs, a, P0=SIGMA2),
                           runs=list(itertools.islice(frozen_passes(y, Qs, a, 1.0), 61)))


def test_pass_1_is_the_ordinary_kalman_filter(passes):
    assert np.abs(passes.runs[0]["m"] - passes.plain["m"]).max() == 0


def test_after_k_passes_the_first_k_minus_1_means_are_exact(passes):
    assert all((run["m"][:, :k - 1] == passes.seq["m"][:, :k - 1]).all()
               for k, run in enumerate(passes.runs, start=1))


def test_after_k_passes_the_kth_mean_is_not_yet_exact(passes):
    """Checked for k = 1, 2, 3."""
    assert all((passes.runs[k - 1]["m"][:, k - 1] != passes.seq["m"][:, k - 1]).any() for k in (1, 2, 3))


def test_T_plus_1_passes_give_the_sequential_robust_filter_exactly(passes):
    assert np.abs(passes.runs[60]["m"] - passes.seq["m"]).max() == 0


def first_pass_below(arr, tol):
    hit = arr < tol
    return np.where(hit.any(axis=0), hit.argmax(axis=0) + 1, -1)


@pytest.fixture(scope="module")
def counted():
    """count_passes, and the same iteration written again with the information-form filter."""
    _, y, a, Qs = simulate_selective(np.random.default_rng(107), 3, 40, 0.9)
    y[:, :3] += 6.0
    res = count_passes(y, Qs, a, 1.0, 60)
    ref_seq = scalar_filter(y, R2, Qs, a, c=1.0)  # its prior M0 = 0, P0 = 1 = SIGMA2, as in experiment 2
    sd = np.sqrt(ref_seq["P"])
    w, errs, moves, prev = np.ones_like(y), [], [], None
    for _ in range(60):
        run = scalar_filter(y, R2, Qs, a, w=w)
        errs.append((np.abs(run["m"] - ref_seq["m"]) / sd).max(axis=1))
        moves.append(np.full(len(y), np.inf) if prev is None else (np.abs(run["m"] - prev) / sd).max(axis=1))
        prev = run["m"]
        w = 1.0 / np.sqrt(1.0 + (y - run["m_pred"]) ** 2 / (run["P_pred"] + R2))
    return SimpleNamespace(y=y, a=a, Qs=Qs, res=res, ref_seq=ref_seq, errs=np.array(errs), moves=np.array(moves))


def test_count_passes_error_after_each_pass_equals_an_independent_iteration(counted):
    errs = counted.res["errs"]
    assert np.abs(errs - counted.errs[:len(errs)]).max() < 1e-9


def test_count_passes_counts_equal_the_independent_ones(counted):
    """Both the passes to each tolerance and the practical rule's passes."""
    for tol in TOLS:
        assert np.array_equal(counted.res["passes"][tol], first_pass_below(counted.errs, tol))
        assert np.array_equal(counted.res["practical"][tol], first_pass_below(counted.moves, tol))


def test_count_passes_contraction_factor(counted):
    """(error at the last pass / error at pass 2)^(1 / (K - 2))."""
    errs = counted.errs
    Kp = first_pass_below(errs, min(TOLS))
    fac = np.array([(errs[k - 1, i] / errs[1, i]) ** (1 / (k - 2)) if k > 2 and errs[1, i] > 0 else np.nan
                    for i, k in enumerate(Kp)])
    assert np.allclose(counted.res["factor"], fac, rtol=1e-6, atol=0, equal_nan=True)


def test_count_passes_memory_is_a_times_1_minus_K_of_the_sequential_filter(counted):
    """a_{t+1} (1 - K_t), t = 1, ..., T - 1."""
    ref = counted.ref_seq
    Kg = ref["w"] ** 2 * ref["P_pred"] / (ref["w"] ** 2 * ref["P_pred"] + R2)
    memory = counted.res["memory"]
    assert memory.shape == (3, 39)
    assert np.abs(memory - counted.a[:, 1:] * (1 - Kg[:, :-1])).max() < 1e-12


def test_count_passes_memory_is_the_effect_of_one_prediction_on_the_next(counted):
    """Restart the filter (weights frozen) from the independent filter's state, once as it is and
    once with the prediction for column t_n moved by h, and difference the next prediction."""
    y, a, Qs, ref = counted.y, counted.a, counted.Qs, counted.ref_seq
    t_n, h = 17, 1e-6

    def tail(m0v):
        return kalman_filter(y[:1, t_n:], R2, Qs[:1, t_n:], a[:1, t_n:], w_fixed=ref["w"][:1, t_n:], m0=m0v,
                             P0=ref["P"][0, t_n - 1])

    m_f = ref["m"][0, t_n - 1]
    fd = (tail(m_f + h / a[0, t_n])["m_pred"][0, 1] - tail(m_f)["m_pred"][0, 1]) / h
    assert abs(fd - counted.res["memory"][0, t_n]) < 1e-7
