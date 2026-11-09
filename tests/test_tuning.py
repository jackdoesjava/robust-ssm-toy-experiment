"""The grid search of fit_all and the filters that filter_specs builds from its fits."""

import functools
from dataclasses import replace

import numpy as np
import pytest

from toy_experiment import tuning
from toy_experiment.config import (BAYES, BOTH, BOTH_PAST, C_GRID, C_PRIOR, FILTERS, INDEP, INDEP_FIX, INDEP_PAST,
                                   K_GRID, K_PRIOR, LAM, ONE_FC, ONE_ML, ONE_TRUE, R, R_GRID, ROB, ROB_FIX)
from toy_experiment.filters import kalman_filter
from toy_experiment.simulate import simulate_rw
from toy_experiment.tuning import FilterSpec, best_over, filter_specs, fit_all, grid_scores

from .helpers import (INF, FakeScores, d2_ref, expected_fit, fabricated_tables, fake_fit, k_none_iff_c_inf,
                      past_ref, use_pool)

INF_C = len(C_GRID) - 1  # index of c = inf

# Three data sets. On seed 104 the training and validation fits differ; on seed 231 the
# filters' scales and windows differ, so mixing them up is caught; seed 302 has bursts, so
# the noise from past readings is used. (With 3 short sequences several filters' best R land
# on the same grid value; test_filter_specs_gives_each_filter_its_own_parameters tells the R
# values apart with fabricated fits, and the fabricated tables below put the best entry on
# every edge of the search.)
SEEDS = {104: None, 231: None, 302: 0.8}  # seed: outlier persistence


def ic(c):
    return int(np.flatnonzero(C_GRID == c)[0])


def iR(Rv):
    return int(np.argmin(np.abs(R_GRID - Rv)))


def ik(k):  # the window does not matter when the noise c is inf; use the first row then
    return 0 if k is None else K_GRID.index(k)


@functools.cache
def brute_force(seed):
    """fit_all on short training and validation sequences, with its search done again by brute
    force: S[k, noise c, robust c, R] is the validation forecast CRPS, from one filter call per
    (k, R) with every (noise c, robust c) pair as its own rows (a noise c of inf leaves R_t = R);
    Sp the same with the noise from past readings; ll the training log-likelihood over R."""
    rng = np.random.default_rng(seed)
    _, y_tr, _ = simulate_rw(rng, 3, 150, 0.1, 0.1, 100.0, SEEDS[seed], None)
    _, y_va, _ = simulate_rw(rng, 3, 150, 0.1, 0.1, 100.0, SEEDS[seed], None)
    fit = fit_all(y_tr, y_va, 0.1)
    nv, nc = len(y_va), len(C_GRID)
    ll = np.array([kalman_filter(y_tr, Rv, 0.1, store=False, score=True)["loglik"].mean() for Rv in R_GRID])
    cin_rows, c_rows = np.repeat(C_GRID, nc * nv), np.tile(np.repeat(C_GRID, nv), nc)
    S, Sp = (np.empty((len(K_GRID), nc, nc, len(R_GRID))) for _ in range(2))
    for arr, dist in ((S, d2_ref), (Sp, past_ref)):
        for i, k in enumerate(K_GRID):
            extra = np.tile(dist(y_va, k), (nc * nc, 1)) / cin_rows[:, None] ** 2
            for j, Rv in enumerate(R_GRID):
                out = kalman_filter(np.tile(y_va, (nc * nc, 1)), Rv, 0.1, c=c_rows, R_t=Rv + extra, store=False,
                                    score=True)
                arr[i, :, :, j] = out["fc_crps"].reshape(nc, nc, nv).mean(axis=2)
    return fit, ll, S, Sp


seeds = pytest.mark.parametrize("seed", SEEDS)


@seeds
def test_one_R_ml_maximises_the_training_likelihood(seed):
    fit, ll, _, _ = brute_force(seed)
    assert ll[iR(fit.R_ml)] >= ll.max() - 1e-9 * abs(ll.max())


@seeds
def test_one_R_forecast_minimises_the_forecast_crps(seed):
    fit, _, S, _ = brute_force(seed)
    plain = S[0, INF_C, INF_C]
    assert plain[iR(fit.R_fc)] <= plain.min() + 1e-12


@seeds
def test_robust_R_and_c_minimise_the_forecast_crps(seed):
    fit, _, S, _ = brute_force(seed)
    rob = S[0, INF_C]  # (robust c, R) without input-dependent noise
    assert rob[ic(fit.c_rob), iR(fit.R_rob)] <= rob.min() + 1e-12


@seeds
def test_the_R_kept_for_every_c_is_the_best_R_for_that_c(seed):
    fit, _, S, _ = brute_force(seed)
    rob = S[0, INF_C]
    bad = sum(rob[i, iR(fit.R_by_c[float(c)])] > rob[i].min() + 1e-12 for i, c in enumerate(C_GRID))
    assert bad == 0


@seeds
def test_robust_filter_with_fixed_c_gets_the_best_R_for_that_c(seed):
    fit, _, S, _ = brute_force(seed)
    rob = S[0, INF_C]
    assert fit.R_prior == fit.R_by_c[C_PRIOR]
    assert rob[ic(C_PRIOR), iR(fit.R_prior)] <= rob[ic(C_PRIOR)].min() + 1e-12


@seeds
def test_input_dependent_R_c_k_minimise_the_forecast_crps(seed):
    fit, _, S, _ = brute_force(seed)
    ind = S[:, :, INF_C]  # (k, noise c, R) without the robust weight
    assert ind[ik(fit.k_in), ic(fit.c_in), iR(fit.R_in)] <= ind.min() + 1e-12


@seeds
def test_input_dependent_filter_with_c_and_k_fixed_gets_its_best_R(seed):
    fit, _, S, _ = brute_force(seed)
    fix = S[K_GRID.index(K_PRIOR), ic(C_PRIOR), INF_C]
    assert fix[iR(fit.R_in_prior)] <= fix.min() + 1e-12


@seeds
def test_robust_plus_input_dependent_minimises_the_forecast_crps_over_the_whole_grid(seed):
    fit, _, S, _ = brute_force(seed)
    assert S[ik(fit.k_both), ic(fit.c_in_both), ic(fit.c_both), iR(fit.R_both)] <= S.min() + 1e-12


@seeds
def test_past_only_input_dependent_R_c_k_minimise_the_forecast_crps(seed):
    fit, _, _, Sp = brute_force(seed)
    ind_p = Sp[:, :, INF_C]
    assert ind_p[ik(fit.k_past), ic(fit.c_past), iR(fit.R_past)] <= ind_p.min() + 1e-12


@seeds
def test_past_only_robust_plus_input_dependent_minimises_the_forecast_crps_over_the_whole_grid(seed):
    fit, _, _, Sp = brute_force(seed)
    v = Sp[ik(fit.k_both_past), ic(fit.c_in_both_past), ic(fit.c_both_past), iR(fit.R_both_past)]
    assert v <= Sp.min() + 1e-12


@seeds
def test_k_is_none_exactly_when_the_noise_c_is_inf(seed):
    assert k_none_iff_c_inf(brute_force(seed)[0])


def test_the_data_of_seed_231_tell_the_fits_apart():
    """So the tests above are not vacuous."""
    fit = brute_force(231)[0]
    assert fit.k_in not in (None, K_PRIOR) and fit.c_in not in (C_PRIOR, np.inf)
    assert fit.R_in != fit.R_in_prior and fit.R_by_c[C_PRIOR] != fit.R_by_c[2.25]
    assert (fit.k_both, fit.c_in_both) != (fit.k_in, fit.c_in)
    assert np.isfinite(fit.c_both) and np.isfinite(fit.c_in_both) and fit.c_both != fit.c_rob


def test_with_bursts_the_past_only_noise_is_used_alone_and_with_the_weight():
    """On seed 302, with its own fit, so its tests above are not vacuous."""
    fit = brute_force(302)[0]
    assert np.isfinite(fit.c_past) and np.isfinite(fit.c_in_both_past) and np.isfinite(fit.c_both_past)
    assert (fit.k_past, fit.c_past) != (fit.k_in, fit.c_in)
    assert (fit.k_both_past, fit.c_in_both_past) != (fit.k_past, fit.c_past)
    assert (fit.k_both_past, fit.c_in_both_past, fit.c_both_past) != (fit.k_both, fit.c_in_both, fit.c_both)


def test_maximum_likelihood_on_clean_data_learns_R_within_one_grid_step():
    _, y_clean, _ = simulate_rw(np.random.default_rng(12), 20, 500, 0.1, 0.0, LAM, None, None)
    R_ml = R_GRID[np.argmax(grid_scores(y_clean, 0.1, "loglik"))]
    assert abs(np.log10(R_ml / R)) <= 0.05 + 1e-12


# filter_specs
def close(a, b):
    return a is not None and np.allclose(a, b, rtol=0, atol=1e-12)


def test_filter_specs_gives_each_filter_its_own_parameters():
    """Each noise from its own window k, the past-only ones from y_{t-1}."""
    fake = fake_fit()
    assert len({fake.k_in, fake.k_both, fake.k_past, fake.k_both_past, K_PRIOR}) == 5  # all windows differ
    yv = np.random.default_rng(13).normal(size=(2, 30))
    sp = filter_specs(fake, yv)
    assert set(sp) == set(FILTERS) - {BAYES}
    assert sp[ONE_TRUE] == FilterSpec(R=R) and sp[ONE_ML] == FilterSpec(R=1.1) and sp[ONE_FC] == FilterSpec(R=2.2)
    assert sp[ROB] == FilterSpec(R=3.3, c=4.4) and sp[ROB_FIX] == FilterSpec(R=5.5, c=C_PRIOR)
    assert sp[INDEP].R == 6.6 and close(sp[INDEP].R_t, 6.6 + d2_ref(yv, 3) / 7.7 ** 2)
    assert sp[INDEP_FIX].R == 8.8 and close(sp[INDEP_FIX].R_t, 8.8 + d2_ref(yv, K_PRIOR) / C_PRIOR ** 2)
    assert sp[BOTH].R == 9.9 and sp[BOTH].c == 1.25 and close(sp[BOTH].R_t, 9.9 + d2_ref(yv, 9) / 0.75 ** 2)
    assert sp[INDEP_PAST].R == 7.1 and sp[INDEP_PAST].c is None
    assert close(sp[INDEP_PAST].R_t, 7.1 + past_ref(yv, 7) / 1.9 ** 2)
    assert sp[BOTH_PAST].R == 8.1 and sp[BOTH_PAST].c == 2.9
    assert close(sp[BOTH_PAST].R_t, 8.1 + past_ref(yv, 4) / 1.1 ** 2)


def test_filter_specs_noise_c_inf_gives_R_t_equal_to_R_exactly():
    yv = np.random.default_rng(13).normal(size=(2, 30))
    sp = filter_specs(replace(fake_fit(), k_in=None, c_in=np.inf, k_past=None, c_past=np.inf), yv)
    assert np.array_equal(sp[INDEP].R_t, np.full(yv.shape, 6.6))
    assert np.array_equal(sp[INDEP_PAST].R_t, np.full(yv.shape, 7.1))


# fit_all on fabricated tables, built so that the best entry is "noise off, robust on",
# "noise on, robust off", "both on" or "everything off", for both kinds of noise:
# {name: (special levels in S, special levels in Sp, seed)}
VARIANTS = {
    "noise off, robust on": ({(None, INF, 2.25): 0.5, (None, INF, INF): 0.8, (K_PRIOR, C_PRIOR, INF): 0.9}, {}, 100),
    "noise on, robust off": ({(3, 4.0, INF): 0.5, (None, INF, 1.5): 0.7}, {(5, 2.5, INF): 0.45}, 101),
    "both on": ({(9, 1.5, 3.0): 0.5, (5, 2.5, INF): 0.7, (None, INF, 2.75): 0.75},
                {(9, 5.0, 2.5): 0.4, (3, 1.75, INF): 0.6}, 102),
    "everything off": ({(None, INF, INF): 0.5}, {}, 103),
}


def fit_on_fabricated(name, monkeypatch):
    """fit_all with every validation score read from a fabricated table, and the brute-force choices."""
    S, Sp = fabricated_tables(*VARIANTS[name])
    fake = FakeScores(S, Sp)
    monkeypatch.setattr(tuning, "_scores_over_c", fake)  # where fit_all looks it up
    use_pool(monkeypatch, None, 1)  # the stand-in cannot be sent to worker processes
    y = np.random.default_rng(5).normal(size=(2, 20))
    fit = fit_all(y, y, 0.1)
    assert fake.calls == 1 + 2 * len(K_GRID) * (len(C_GRID) - 1)  # every task read the fabricated tables
    return fit, expected_fit(S, Sp)


@pytest.mark.parametrize("name", VARIANTS)
def test_fit_all_on_fabricated_tables_makes_every_choice_as_brute_force(name, monkeypatch):
    fit, want = fit_on_fabricated(name, monkeypatch)
    assert [key for key in want if getattr(fit, key) != want[key]] == []


@pytest.mark.parametrize("name", VARIANTS)
def test_fit_all_on_fabricated_tables_k_is_none_exactly_when_the_noise_c_is_inf(name, monkeypatch):
    fit, _ = fit_on_fabricated(name, monkeypatch)
    assert k_none_iff_c_inf(fit)


def test_the_fabricated_tables_put_the_best_entry_on_each_edge_of_the_search():
    e = [expected_fit(*fabricated_tables(*VARIANTS[name])) for name in VARIANTS]
    assert e[0]["c_in_both"] == INF and np.isfinite(e[0]["c_both"]) and e[0]["c_in_both_past"] == INF
    assert e[1]["c_both"] == INF and np.isfinite(e[1]["c_in_both"])
    assert e[1]["c_both_past"] == INF and np.isfinite(e[1]["c_in_both_past"])
    assert np.isfinite(e[2]["c_in_both"]) and np.isfinite(e[2]["c_both"])
    assert np.isfinite(e[2]["c_in_both_past"]) and np.isfinite(e[2]["c_both_past"])
    assert e[3]["c_in"] == INF and e[3]["k_in"] is None and e[3]["c_both"] == INF
    assert e[3]["c_past"] == INF and e[3]["c_both_past"] == INF


def test_best_over_breaks_ties_by_the_first_parameter_and_the_first_R():
    s_tie = np.array([3.0, 1.0, 2.0, 1.0])  # the best R ties within a row too: argmin takes the first
    assert best_over({"a": s_tie + 1.0, "b": s_tie, "c": s_tie.copy()}) == (1.0, float(R_GRID[1]), "b")
    assert best_over({"c": s_tie.copy(), "b": s_tie}) == (1.0, float(R_GRID[1]), "c")
