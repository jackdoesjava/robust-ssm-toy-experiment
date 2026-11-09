"""The grid Bayes filter."""

import numpy as np
import pytest
from scipy.special import ndtr

from toy_experiment.bayes import exact_filter
from toy_experiment.config import EPS, LAM, R
from toy_experiment.filters import kalman_filter
from toy_experiment.metrics import gaussian_scores
from toy_experiment.simulate import simulate_rw

SCORES = ("mean", "pit", "crps", "width90")
TOLS = dict(mean=1e-9, pit=2e-7, crps=2e-7, width90=2e-7)


def absmean(mu, var):
    """E|Z| for Z ~ N(mu, var)."""
    sd = np.sqrt(var)
    return sd * np.sqrt(2 / np.pi) * np.exp(-0.5 * mu * mu / var) + mu * (2 * ndtr(mu / sd) - 1)


def mixture_scores(wts, ms, Ps, xv):
    """Mean, PIT, CRPS and 90% width of a Gaussian mixture, scored at xv."""
    sd = np.sqrt(Ps)

    def cdf(q):
        return (wts * ndtr((q - ms) / sd)).sum()

    crps = ((wts * absmean(xv - ms, Ps)).sum()
            - 0.5 * (np.outer(wts, wts) * absmean(ms[:, None] - ms[None, :], Ps[:, None] + Ps[None, :])).sum())
    qs = []
    for level in (0.05, 0.95):
        lo, hi = ms.min() - 12 * sd.max(), ms.max() + 12 * sd.max()
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cdf(mid) < level else (lo, mid)
        qs.append(0.5 * (lo + hi))
    return dict(mean=(wts * ms).sum(), pit=cdf(xv), crps=crps, width90=qs[1] - qs[0])


@pytest.mark.parametrize("kind, eps, stay", [("independent outliers", 0.3, None), ("bursty outliers", 0.3, 0.8),
                                             ("a real jump of 5 at step 6", 0.1, None)])
def test_grid_filter_equals_exact_enumeration_of_outlier_patterns(shared, kind, eps, stay):
    """With every outlier pattern enumerated, the posterior is a Gaussian mixture whose mean,
    PIT, CRPS and 90% width are computed exactly."""
    Tn, n, Q, lam = 10, 3, 0.1, 100.0
    if kind.startswith("a real jump"):
        x, y, _ = simulate_rw(np.random.default_rng(31), n, Tn, Q, eps, lam, stay, 5)
    else:
        x, y = shared.patterns[kind]
    pats = ((np.arange(2 ** Tn)[:, None] >> np.arange(Tn)) & 1).astype(bool)
    p_stay, p_start = (eps, eps) if stay is None else (stay, eps * (1 - stay) / (1 - eps))
    lp = np.where(pats[:, 0], np.log(eps), np.log1p(-eps))
    p_now = np.where(pats[:, :-1], p_stay, p_start)
    lp = lp + np.where(pats[:, 1:], np.log(p_now), np.log1p(-p_now)).sum(axis=1)
    ex = exact_filter(y, x, Q, eps, lam, stay)
    worst = dict.fromkeys(SCORES, 0.0)
    for i in range(n):
        Rts = np.where(pats, lam * R, R)
        out = kalman_filter(np.tile(y[i], (len(pats), 1)), R, Q, R_t=Rts)
        S = out["P_pred"] + Rts
        ll = -0.5 * (np.log(2 * np.pi * S) + (y[i] - out["m_pred"]) ** 2 / S)
        logw = lp[:, None] + np.cumsum(ll, axis=1)
        wts = np.exp(logw - logw.max(axis=0))
        wts /= wts.sum(axis=0)
        for t in range(Tn):
            # patterns that agree up to step t share their filter output; pattern j < 2^(t+1)
            # stands for all of them, with their weights added up
            comp = np.arange(2 ** (t + 1))
            W = np.bincount(np.arange(2 ** Tn) % 2 ** (t + 1), weights=wts[:, t], minlength=len(comp))
            mix = mixture_scores(W, out["m"][comp, t], out["P"][comp, t], x[i, t])
            for key in SCORES:
                worst[key] = max(worst[key], abs(mix[key] - ex[key][i, t]))
    assert {key: worst[key] for key in SCORES if not worst[key] < TOLS[key]} == {}


@pytest.mark.parametrize("Q", [0.1, 1.0])
def test_grid_filter_equals_the_kalman_filter_on_clean_data_with_a_jump(shared, Q):
    x, y = shared.clean[Q]
    kf = kalman_filter(y, R, Q)
    ex = exact_filter(y, x, Q, 0.0, LAM)
    gs = gaussian_scores(x, kf["m"], kf["P"])
    worst = {key: np.abs(ex[key] - gs[key]).max() for key in SCORES}
    assert {key: d for key, d in worst.items() if not d < TOLS[key]} == {}


@pytest.mark.parametrize("seed, Q, stay", [(151, 0.1, None), (152, 0.1, 0.8), (153, 1.0, None)],
                         ids=["Q=0.1", "Q=0.1, bursts", "Q=1"])
def test_widening_the_grid_to_40_changes_no_score(seed, Q, stay):
    """The default grid is wide enough, with outliers."""
    x, y, _ = simulate_rw(np.random.default_rng(seed), 20, 300, Q, EPS, LAM, stay, 150)
    narrow, wide = exact_filter(y, x, Q, EPS, LAM, stay), exact_filter(y, x, Q, EPS, LAM, stay, half_width=40.0)
    d = max(np.abs(narrow[key] - wide[key]).max() for key in narrow)
    assert d < 1e-9


def test_grid_filter_renormalises_over_many_steps():
    """Clean readings swinging between +6 and -6, where the Kalman filter is still exact but the
    unnormalised posterior mass shrinks by about e^-15 per step, so a filter that did not
    renormalise what it carries would underflow within 50 steps."""
    yb = np.tile(np.where(np.arange(100) % 2 == 0, 6.0, -6.0), (2, 1))
    yb[1] += 0.5
    xb = np.zeros_like(yb)
    worst = 0.0
    for Q in (0.1, 1.0):
        kf = kalman_filter(yb, R, Q)
        exb = exact_filter(yb, xb, Q, 0.0, LAM)
        gs = gaussian_scores(xb, kf["m"], kf["P"])
        worst = max(worst, max(np.abs(exb[key] - gs[key]).max() if np.isfinite(exb[key]).all() else np.inf
                               for key in SCORES))
    assert worst < 1e-6
