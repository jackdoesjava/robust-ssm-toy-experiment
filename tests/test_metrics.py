"""The Gaussian CRPS and the summary metrics, on cases with known answers."""

import numpy as np
import pytest

from toy_experiment.config import AFTER, JUMP, Z90
from toy_experiment.metrics import gaussian_crps, gaussian_scores, summarise


def test_closed_form_gaussian_crps_matches_monte_carlo():
    s, mv, xv = 1.5, 0.2, 0.9
    draws = np.random.default_rng(2).normal(mv, s, size=(2, 2_000_000))
    mc = np.abs(draws[0] - xv).mean() - 0.5 * np.abs(draws[0] - draws[1]).mean()
    assert abs(mc - gaussian_crps(xv, mv, s)) < 5e-3


@pytest.fixture(scope="module")
def calibrated():
    """The summary of N(0, 1) forecasts of N(0, 1) truths."""
    xs = np.random.default_rng(8).normal(size=(200, 1000))
    idx = np.random.default_rng(9).integers(0, 200, size=(200, 200))
    summ, _, _ = summarise(xs, gaussian_scores(xs, np.zeros_like(xs), np.ones_like(xs)), 500, idx)
    return summ


def test_calibrated_gaussian_has_90_percent_coverage(calibrated):
    assert abs(calibrated["cov90"][0] - 0.9) < 0.005


def test_calibrated_gaussian_has_50_percent_coverage(calibrated):
    assert abs(calibrated["cov50"][0] - 0.5) < 0.006


def test_calibrated_gaussian_rmse_is_the_sd(calibrated):
    assert abs(calibrated["rmse"][0] - 1) < 0.01


# four sequences of 300 steps with the jump after step 150, truth 0 and unit variance
N, T, JT = 4, 300, 150
ONES, ZEROS = np.ones((N, T)), np.zeros((N, T))


@pytest.fixture
def boot_idx():
    return np.random.default_rng(10).integers(0, N, size=(50, N))


def test_summarise_uses_only_the_steps_before_the_jump_for_accuracy_and_calibration(boot_idx):
    m = np.zeros((N, T))
    m[:, JT:] = -3.0
    summ, _, _ = summarise(ZEROS, gaussian_scores(ZEROS, m, ONES), JT, boot_idx)
    assert summ["rmse"][0] == 0
    assert abs(summ["crps"][0] - (np.sqrt(2) - 1) / np.sqrt(np.pi)) < 1e-12
    assert summ["cov90"][0] == 1
    assert abs(summ["width90"][0] - 2 * Z90) < 1e-12


def test_summarise_post_jump_rmse_covers_exactly_the_after_steps_from_the_jump_on(boot_idx):
    m = np.zeros((N, T))
    m[:, JT - 1], m[:, JT], m[:, JT + AFTER - 1], m[:, JT + AFTER] = 100.0, 7.0, 1.0, 100.0
    summ, _, _ = summarise(ZEROS, gaussian_scores(ZEROS, m, ONES), JT, boot_idx)
    assert abs(summ["rmse_after"][0] - np.sqrt(50 / AFTER)) < 1e-12


def test_summarise_delay_comes_from_the_mean_signed_error():
    """Offsets of +-2 cancel, and an error of 5 * 0.7^s first falls below 1 at s = 5."""
    m2 = np.zeros((2, T))
    lagged = JUMP * 0.7 ** np.arange(T - JT)
    m2[0, JT:], m2[1, JT:] = -(lagged + 2.0), -(lagged - 2.0)
    summ, _, _ = summarise(np.zeros((2, T)), gaussian_scores(np.zeros((2, T)), m2, np.ones((2, T))), JT,
                           np.array([[0, 1]]))
    assert summ["delay"][0] == 5


def test_summarise_interval_half_width_is_196_sd_over_sqrt_n(boot_idx):
    m = np.zeros((N, T))
    m[:, :JT] = np.arange(1, N + 1)[:, None]
    summ, _, _ = summarise(ZEROS, gaussian_scores(ZEROS, m, ONES), JT, boot_idx)
    want = 1.96 * np.std(np.arange(1, N + 1), ddof=1) / np.sqrt(N)
    assert abs(summ["rmse"][1] - want) < 1e-12
