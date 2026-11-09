"""The wiring of the experiments: their results rebuilt from the documented parts."""

import functools
from dataclasses import replace

import numpy as np
import pytest

from toy_experiment.bayes import exact_filter
from toy_experiment.config import BAYES, BOTH, C2, C_GRID, INDEP, N_BOOT, ONE_FC, ONE_ML, REGIMES, ROB, SCENARIOS
from toy_experiment.experiments import experiment2, run_scenario
from toy_experiment.filters import kalman_filter
from toy_experiment.metrics import gaussian_scores, summarise
from toy_experiment.passes import count_passes
from toy_experiment.simulate import simulate_rw, simulate_selective
from toy_experiment.tuning import filter_specs, fit_all

from .helpers import SMALL, SMALL_EXTRA, TINY_EXP2, capture_figures


def scenario_data(k, cfg):
    """Scenario k's documented data: training (seed 0) and validation (seed 1) sequences without
    a jump, test sequences (seed 2) with one, and the bootstrap resamples (seed 99)."""
    sc = SCENARIOS[k]
    _, y_tr, _ = simulate_rw(np.random.default_rng([2027, k, 0]), cfg.n_train, cfg.T, sc.Q, sc.train.eps,
                             sc.train.lam, sc.train.stay, None)
    _, y_va, _ = simulate_rw(np.random.default_rng([2027, k, 1]), cfg.n_val, cfg.T, sc.Q, sc.train.eps,
                             sc.train.lam, sc.train.stay, None)
    x_te, y_te, _ = simulate_rw(np.random.default_rng([2027, k, 2]), cfg.n_test, cfg.T, sc.Q, sc.test.eps,
                                sc.test.lam, sc.test.stay, cfg.jump_t)
    boot_idx = np.random.default_rng([2027, k, 99]).integers(0, cfg.n_test, size=(N_BOOT, cfg.n_test))
    return y_tr, y_va, x_te, y_te, boot_idx


def scored(spec, x, y, Q, jump_t, boot_idx):
    """summarise for the Gaussian filter that spec describes."""
    out = kalman_filter(y, spec.R, Q, c=spec.c, R_t=spec.R_t)
    return summarise(x, gaussian_scores(x, out["m"], out["P"]), jump_t, boot_idx)


@functools.cache
def scenario_and_rebuilt(k):
    y_tr, y_va, x_te, y_te, boot_idx = scenario_data(k, SMALL)
    return run_scenario(k, SMALL), fit_all(y_tr, y_va, SCENARIOS[k].Q), x_te, y_te, boot_idx


@pytest.mark.parametrize("k", [6, 7])
def test_run_scenario_is_fitted_on_the_jump_free_training_and_validation_data(k):
    res, fit, *_ = scenario_and_rebuilt(k)
    assert fit == res["fitted"]


@pytest.mark.parametrize("k", [6, 7])
def test_run_scenario_rows_equal_their_filters_rebuilt_from_the_documented_parts(k):
    res, fit, x_te, y_te, boot_idx = scenario_and_rebuilt(k)
    sc = SCENARIOS[k]
    bad = []
    for lab, spec in filter_specs(fit, y_te).items():
        summ, _, boots = scored(spec, x_te, y_te, sc.Q, SMALL.jump_t, boot_idx)
        if summ != res["metrics"][lab] or not np.array_equal(boots, res["boots"][lab]):
            bad.append(lab)
    summ, _, boots = summarise(x_te, exact_filter(y_te, x_te, sc.Q, sc.test.eps, sc.test.lam, sc.test.stay),
                               SMALL.jump_t, boot_idx)
    if summ != res["metrics"][BAYES] or not np.array_equal(boots, res["boots"][BAYES]):
        bad.append(BAYES)
    assert bad == []


def test_tradeoff_curve_has_a_point_for_every_c_with_the_R_tuned_for_that_c(small_main_run):
    res = small_main_run
    _, _, x_te, y_te, boot_idx = scenario_data(0, SMALL_EXTRA)
    assert [d["c"] for d in res["curve"]] == [float(c) for c in C_GRID]
    for d in res["curve"]:
        Rc = res["fitted"].R_by_c[d["c"]]
        out = kalman_filter(y_te, Rc, SCENARIOS[0].Q, c=d["c"])
        mt, _, _ = summarise(x_te, gaussian_scores(x_te, out["m"], out["P"]), SMALL_EXTRA.jump_t, boot_idx)
        assert d["R"] == Rc and all(d[key] == mt[key][0] for key in ("rmse", "crps", "cov90", "delay"))


def test_tradeoff_curve_inf_point_is_the_one_R_forecast_crps_row(small_main_run):
    res = small_main_run
    assert all(res["curve"][-1][key] == res["metrics"][ONE_FC][key][0] for key in ("rmse", "crps", "cov90", "delay"))


@functools.cache
def stability_refits():
    """The documented refits: fresh jump-free training (seed 10 + r) and validation (seed 20 + r) sequences."""
    sc = SCENARIOS[0]
    fits = []
    for r in range(SMALL_EXTRA.n_stability):
        _, y_tr2, _ = simulate_rw(np.random.default_rng([2027, 0, 10 + r]), SMALL_EXTRA.n_train, SMALL_EXTRA.T, sc.Q,
                                  sc.train.eps, sc.train.lam, sc.train.stay, None)
        _, y_va2, _ = simulate_rw(np.random.default_rng([2027, 0, 20 + r]), SMALL_EXTRA.n_val, SMALL_EXTRA.T, sc.Q,
                                  sc.train.eps, sc.train.lam, sc.train.stay, None)
        fits.append(fit_all(y_tr2, y_va2, sc.Q))
    return fits


def test_tuning_stability_rows_are_the_refits_rebuilt(small_main_run):
    res = small_main_run
    _, _, x_te, y_te, boot_idx = scenario_data(0, SMALL_EXTRA)
    for r, fit in enumerate(stability_refits()):
        assert fit == res["stability"][r]["fit"]
        for lab, spec in filter_specs(fit, y_te).items():
            if lab in (ONE_ML, INDEP, ROB, BOTH):
                mt, _, _ = scored(spec, x_te, y_te, SCENARIOS[0].Q, SMALL_EXTRA.jump_t, boot_idx[:1])
                assert res["stability"][r][lab] == dict(rmse=mt["rmse"][0], crps=mt["crps"][0], cov90=mt["cov90"][0],
                                                        delay=mt["delay"][0])


def test_tuning_stability_refits_differ_from_each_other_and_from_the_main_fit(small_main_run):
    fits = stability_refits()
    assert fits[0] != fits[1] and all(f != small_main_run["fitted"] for f in fits)


def test_exp2_memory_and_factor_are_the_medians_of_the_recomputed_counts(tiny_exp2):
    """Memory is the median of a_{t+1} (1 - K_t), as exp2_table.txt says, and Factor the median of
    the per-sequence factors over the sequences that have one (some have none here)."""
    T = TINY_EXP2.T2_mid
    nan_seen = 0
    for row, (label, a_med, cap, seed) in zip(tiny_exp2.rows, REGIMES):
        _, y, a, Qs = simulate_selective(np.random.default_rng([2027, seed, T]), TINY_EXP2.n2, T, a_med, cap)
        cp = count_passes(y, Qs, a, C2, min(TINY_EXP2.maxpass, T + 2))
        fac = cp["factor"]
        nan_seen += int(np.isnan(fac).any())
        want_f = float(np.nanmedian(fac)) if np.isfinite(fac).any() else None
        got = row[f"T={T} c={C2:g}"]
        assert abs(got["memory"] - float(np.median(cp["memory"]))) <= 1e-12
        assert (got["factor"] is None) == (want_f is None)
        assert want_f is None or abs(got["factor"] - want_f) <= 1e-12
    assert ("memory: median over steps and sequences of a_{t+1} (1 - K_t), the filter's memory per step"
            in tiny_exp2.text)
    assert nan_seen > 0


def test_exp2_practical_rule_lines_say_n_a_when_nothing_converges(tmp_path):
    cfg = replace(TINY_EXP2, T2_list=(30, 40), T2_mid=40, maxpass=1)
    with capture_figures():
        experiment2(cfg, tmp_path)
    lines = (tmp_path / "exp2_table.txt").read_text(encoding="utf-8").splitlines()
    n_seq = 2 * len(REGIMES) * cfg.n2
    for name in ("1%", "1e-8"):
        assert f"  tolerance {name} of sd: n/a ({n_seq} sequences not converged, left out)" in lines
