"""The data generators."""

import numpy as np
import pytest

from toy_experiment.config import EPS, JUMP, LAM, R, R2, SIGMA2
from toy_experiment.simulate import outlier_mask, simulate_rw, simulate_selective


@pytest.fixture(scope="module")
def walk():
    """Random walks with 10% outliers, with a jump after step 500 and the same draws without it."""
    x, y, o = simulate_rw(np.random.default_rng(3), 400, 1000, 0.1, 0.1, 100.0, None, 500)
    x_flat, _, _ = simulate_rw(np.random.default_rng(3), 400, 1000, 0.1, 0.1, 100.0, None, None)
    return x, y, o, x_flat


def test_outlier_rate_is_10_percent(walk):
    _, _, o, _ = walk
    assert abs(o.mean() - 0.1) < 0.003


def test_clean_noise_variance_is_R(walk):
    x, y, o, _ = walk
    assert abs((y - x)[~o].var() / R - 1) < 0.01


def test_outlier_noise_variance_is_100_R(walk):
    x, y, o, _ = walk
    assert abs((y - x)[o].var() / (100 * R) - 1) < 0.03


def test_jump_is_5_from_step_501_on_and_absent_before(walk):
    x, _, _, x_flat = walk
    jump_err = max(np.abs((x - x_flat)[:, 500:] - JUMP).max(), np.abs((x - x_flat)[:, :500]).max())
    assert jump_err < 1e-12


def test_random_walk_step_variance_is_Q(walk):
    q_var = np.diff(walk[3], axis=1).var()
    assert abs(q_var / 0.1 - 1) < 0.01


@pytest.fixture(scope="module")
def bursts():
    return outlier_mask(np.random.default_rng(4), 400, 2000, 0.1, 0.8)


def test_bursty_outlier_rate_is_10_percent(bursts):
    assert abs(bursts.mean() - 0.1) < 0.005


def test_bursty_outliers_have_mean_run_length_5(bursts):
    edges = np.zeros((len(bursts), 1), bool)
    runs = np.diff(np.concatenate([edges, bursts, edges], axis=1).astype(int), axis=1)
    assert abs(bursts.sum() / (runs == 1).sum() - 5) < 0.2


def test_bursty_outlier_rate_is_10_percent_at_each_of_the_first_four_steps():
    r4 = outlier_mask(np.random.default_rng(5), 200_000, 4, 0.1, 0.8).mean(axis=0)
    assert np.abs(r4 - 0.1).max() < 0.004


@pytest.fixture(scope="module")
def decaying():
    return simulate_selective(np.random.default_rng(6), 50, 4000, 0.9)


def test_decaying_state_keeps_variance_sigma2(decaying):
    x = decaying[0]
    assert abs(x[:, 100:].var() / SIGMA2 - 1) < 0.1


def test_median_decay_is_a_med(decaying):
    assert abs(np.median(decaying[2]) - 0.9) < 0.02


def test_input_is_stationary_with_sd_075(decaying):
    u = np.log(np.log(decaying[2]) / np.log(0.9))
    assert abs(u.std() - 0.75) < 0.05


def test_decaying_state_observation_noise_variance(decaying):
    """(1 - eps) R2 + eps LAM R2."""
    x, y, _, _ = decaying
    want = (1 - EPS) * R2 + EPS * LAM * R2
    assert abs((y - x).var() / want - 1) < 0.1


def test_capped_decay_never_exceeds_the_cap():
    _, _, a, _ = simulate_selective(np.random.default_rng(7), 20, 2000, 0.3, cap=0.6)
    assert a.max() <= 0.6
