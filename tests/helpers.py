"""Reference implementations written independently of the package, stand-ins, and fabricated results."""

import contextlib
from dataclasses import replace
from types import SimpleNamespace

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest
from scipy.special import ndtr

from toy_experiment import parallel
from toy_experiment.config import (BOTH, C_GRID, C_PRIOR, FILTERS, INDEP, JUMP, K_GRID, K_PRIOR, M0, ONE_ML, P0,
                                   QUICK, R_GRID, ROB)
from toy_experiment.figures import plt
from toy_experiment.tuning import FittedParams

INF = np.inf

# Small configurations for the tests
SMALL = replace(QUICK, T=80, jump_t=40, n_train=4, n_val=4, n_test=6, n_stability=0)
SMALL_EXTRA = replace(SMALL, n_stability=2)
FABRICATED_CFG = replace(QUICK, T=300, jump_t=150, n_train=4, n_val=4, n_test=4, n_stability=2)
TINY_EXP2 = replace(QUICK, T2_list=(40, 60), T2_mid=60, n2=3, maxpass=10)  # some 1e-8 counts not reached


# Reference implementations
def crps_ref(xv, mv, sv):
    z = (xv - mv) / sv
    return sv * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))


def d2_ref(yv, k):
    """local_distance2 as a direct loop, the prior mean standing in for missing readings."""
    out = np.zeros_like(yv)
    for i in range(yv.shape[0]):
        for t in range(yv.shape[1]):
            window = [M0] * max(0, k - t) + list(yv[i, max(0, t - k):t])
            out[i, t] = (yv[i, t] - np.median(window)) ** 2
    return out


def past_ref(yv, k):
    """past_distance2 as a direct loop: y_{t-1} against the median of the k readings before it; 0 at t = 1."""
    out = np.zeros_like(yv)
    for i in range(yv.shape[0]):
        for t in range(1, yv.shape[1]):
            s = t - 1
            window = [M0] * max(0, k - s) + list(yv[i, max(0, s - k):s])
            out[i, t] = (yv[i, s] - np.median(window)) ** 2
    return out


def scalar_filter(yv, Rv, Qs, a, c=None, R_t=None, w=None):
    """The (robust) Kalman filter in information form."""
    n, T = yv.shape
    m, P = np.full(n, M0), np.full(n, P0)
    res = {k: np.empty((n, T)) for k in ("m_pred", "P_pred", "m", "P", "w")}
    for t in range(T):
        mp, Pp = a[:, t] * m, a[:, t] ** 2 * P + Qs[:, t]
        Ru = Rv if R_t is None else R_t[:, t]
        if w is not None:
            w2 = w[:, t] ** 2
        elif c is None:
            w2 = np.ones(n)
        else:
            w2 = 1.0 / (1.0 + (yv[:, t] - mp) ** 2 / (c ** 2 * (Pp + Ru)))
        P = 1.0 / (1.0 / Pp + w2 / Ru)
        m = P * (mp / Pp + w2 * yv[:, t] / Ru)
        for key, v in zip(res, (mp, Pp, m, P, np.sqrt(w2))):
            res[key][:, t] = v
    return res


def k_none_iff_c_inf(fit):
    """Whether each input-dependent window is None exactly when its noise c is inf."""
    return all((getattr(fit, k) is None) == (not np.isfinite(getattr(fit, c))) for k, c in
               (("k_in", "c_in"), ("k_both", "c_in_both"), ("k_past", "c_past"), ("k_both_past", "c_in_both_past")))


# Stand-ins
class InProcessPool:
    """Stands in for the worker processes: map runs the tasks here, in order, and counts them."""

    def __init__(self):
        self.calls = self.tasks = 0

    def map(self, fn, tasks):
        tasks = list(tasks)
        self.calls += 1
        self.tasks += len(tasks)
        return [fn(task) for task in tasks]


def use_pool(monkeypatch, pool, n):
    """Run the code under test with this pool (None: none) and n workers."""
    monkeypatch.setattr(parallel, "_pool", pool)
    monkeypatch.setattr(parallel, "_workers", n)


TABLE_KEYS = ([(None, INF, float(c)) for c in C_GRID]
              + [(k, float(ci), float(c)) for k in K_GRID for ci in C_GRID if np.isfinite(ci) for c in C_GRID])


def fabricated_tables(special, special_past, seed):
    """Validation tables {(k, noise c, robust c): scores over R_GRID} for the noise from y_t
    (S) and from past readings (Sp), sharing the rows without noise. Every entry has its own
    level, distinct within each table, and its own best R; `special` entries get the given
    lower levels."""
    rng = np.random.default_rng(seed)
    iRs = np.arange(len(R_GRID))
    level = dict(zip(TABLE_KEYS, 1.0 + 1e-3 * rng.permutation(len(TABLE_KEYS))))
    level_p = dict(zip(TABLE_KEYS, 1.0 + 1e-3 * (rng.permutation(len(TABLE_KEYS)) + 0.5)))
    level.update(special)
    level_p.update(special_past)
    best = dict(zip(TABLE_KEYS, rng.integers(10, len(R_GRID) - 10, size=len(TABLE_KEYS))))
    S = {key: level[key] + 1e-4 * ((iRs - best[key]) / 10.0) ** 2 for key in TABLE_KEYS}
    Sp = {key: S[key] if key[0] is None else level_p[key] + 1e-4 * ((iRs - best[key]) / 10.0) ** 2
          for key in TABLE_KEYS}
    return S, Sp


class FakeScores:
    """Stands in for tuning._scores_over_c, reading fabricated tables, and counts its tasks."""

    def __init__(self, S, Sp):
        self.S, self.Sp, self.calls = S, Sp, 0

    def __call__(self, args):
        self.calls += 1
        _, _, k, c_in, past = args
        if k is None or not np.isfinite(c_in):  # noise off: R_t = R, whatever the window
            return [self.S[(None, INF, float(c))] for c in C_GRID]
        return [(self.Sp if past else self.S)[(k, c_in, float(c))] for c in C_GRID]


def expected_fit(S, Sp):
    """fit_all's documented choices on fabricated tables, by brute force (the tables have no ties).
    Everything but R_ml, which comes from the training data."""
    def argbest(sub):
        key = min(sub, key=lambda kk: sub[kk].min())
        return key, float(R_GRID[int(np.argmin(sub[key]))])
    e = {"R_fc": float(R_GRID[np.argmin(S[(None, INF, INF)])])}
    off = {key: s for key, s in S.items() if key[0] is None}
    e["R_by_c"] = {key[2]: float(R_GRID[np.argmin(s)]) for key, s in off.items()}
    (_, _, e["c_rob"]), e["R_rob"] = argbest(off)
    e["R_prior"] = e["R_by_c"][C_PRIOR]
    (e["k_in"], e["c_in"], _), e["R_in"] = argbest({key: s for key, s in S.items() if key[2] == INF})
    e["R_in_prior"] = float(R_GRID[np.argmin(S[(K_PRIOR, C_PRIOR, INF)])])
    (e["k_both"], e["c_in_both"], e["c_both"]), e["R_both"] = argbest(S)
    (e["k_past"], e["c_past"], _), e["R_past"] = argbest({key: s for key, s in Sp.items() if key[2] == INF})
    (e["k_both_past"], e["c_in_both_past"], e["c_both_past"]), e["R_both_past"] = argbest(Sp)
    return e


@contextlib.contextmanager
def capture_figures():
    """Records the calls the figure code makes to a few matplotlib methods, and keeps the
    figures it closes open until the end, so their artists can be checked."""
    cap = SimpleNamespace(calls=[], figs=[])
    close = plt.close
    try:
        with pytest.MonkeyPatch.context() as mp:
            targets = [(matplotlib.axes.Axes, name) for name in ("scatter", "fill_between", "legend", "plot")]
            for cls, name in targets + [(matplotlib.figure.Figure, "legend")]:
                def wrapper(obj, *a, _orig=getattr(cls, name), _name=f"{cls.__name__}.{name}", **kw):
                    out = _orig(obj, *a, **kw)
                    cap.calls.append((_name, obj, a, kw, out))
                    return out
                mp.setattr(cls, name, wrapper)
            mp.setattr(plt, "close", lambda fig=None: cap.figs.append(fig))
            yield cap
    finally:
        for f in cap.figs:
            close(f)


# Fabricated results
def fake_fit():
    """Fitted parameters in which every value is distinct, so each says which filter it belongs to."""
    return FittedParams(R_ml=1.1, R_fc=2.2, R_by_c={}, R_rob=3.3, c_rob=4.4, R_prior=5.5, R_in=6.6, k_in=3, c_in=7.7,
                        R_in_prior=8.8, R_both=9.9, k_both=9, c_in_both=0.75, c_both=1.25,
                        R_past=7.1, k_past=7, c_past=1.9, R_both_past=8.1, k_both_past=4, c_in_both_past=1.1,
                        c_both_past=2.9)


def _per_filter(res, half):
    """Per-sequence values, metrics and bootstrap delays in which every value names its filter and metric."""
    spread = np.arange(4.0)
    for i, lab in enumerate(FILTERS):
        ps = dict(rmse=0.1 * (i + 1) + 0.0013 * (i + 1) * spread,
                  crps=0.5 + 0.017 * (i + 1) + 0.0021 * (i + 1) * spread,
                  rmse_after=1.0 + 0.071 * (i + 1) + 0.0037 * (i + 1) * spread,
                  cov90=0.3031 + i / 1000 + 0 * spread, cov50=0.6061 + i / 1000 + 0 * spread,
                  width90=4.041 + i / 100 + 0 * spread)
        res["per_seq"][lab] = ps
        res["metrics"][lab] = {key: (float(v.mean()), half(i)) for key, v in ps.items()}
        res["boots"][lab] = 10 + i + (np.arange(200) % (i + 2))
    return res


def simple_result():
    """A fabricated run_scenario result with no intervals on the metrics."""
    res = _per_filter(dict(name="test", fitted=fake_fit(), metrics={}, per_seq={}, boots={}), lambda i: 0.0)
    for i, lab in enumerate(FILTERS):
        res["metrics"][lab]["delay"] = (float(20 + 3 * i), 0.0, 0.0)
    return res


def full_result():
    """A fabricated run_scenario(..., extra=True) result for FABRICATED_CFG, in which every number
    says which filter, metric, c or refit it belongs to, with readings beyond each edge of figure 1(a)."""
    T, jt = FABRICATED_CFG.T, FABRICATED_CFG.jump_t
    rng = np.random.default_rng(21)
    x = np.cumsum(rng.normal(0, 0.3, T)) + np.where(np.arange(T) >= jt, JUMP, 0.0)
    outlier = rng.random(T) < 0.1
    y = x + np.where(outlier, 10.0, 1.0) * rng.normal(size=T)
    y[100], y[120], outlier[100], outlier[120] = x[100] + 60.0, x[120] - 60.0, True, True  # beyond the axis
    res = _per_filter(dict(name="fabricated", metrics={}, per_seq={}, boots={}), lambda i: 0.001 * (i + 1))
    for i, lab in enumerate(FILTERS):
        res["metrics"][lab]["delay"] = (float(20 + 3 * i), float(18 + 3 * i), float(23 + 3 * i))
    res["fitted"] = FittedParams(R_ml=1.1, R_fc=2.2, R_by_c={float(c): 0.5 + 0.1 * j for j, c in enumerate(C_GRID)},
                                 R_rob=3.3, c_rob=2.25, R_prior=5.5, R_in=6.6, k_in=None, c_in=INF, R_in_prior=8.8,
                                 R_both=9.9, k_both=9, c_in_both=0.75, c_both=1.25, R_past=7.7, k_past=None,
                                 c_past=INF, R_both_past=4.4, k_both_past=3, c_in_both_past=1.1, c_both_past=2.9)
    res["curve"] = [dict(c=float(c), R=0.5 + 0.1 * j, rmse=0.9 - 0.01 * j, crps=0.5 - 0.004 * j, cov90=0.8 + 0.005 * j,
                         delay=float(40 - 2 * j)) for j, c in enumerate(C_GRID)]
    res["stability"] = []
    for r in range(2):
        f2 = replace(res["fitted"], R_ml=11.0 + r, R_rob=1.31 + r, c_rob=[2.5, INF][r], R_in=1.41 + r,
                     c_in=[3.0, INF][r], k_in=[9, None][r], R_both=1.51 + r, c_both=[3.5, 1.5][r],
                     c_in_both=[6.0, INF][r], k_both=[3, None][r])
        row = dict(fit=f2)
        for j, lab in enumerate((ONE_ML, INDEP, ROB, BOTH)):
            row[lab] = dict(rmse=0.611 + 0.01 * j + 0.1 * r, crps=0.333 + 0.01 * j + 0.1 * r, cov90=0.9,
                            delay=float(7 + j + 10 * r))
        res["stability"].append(row)
    res["example"] = dict(x=x, y=y, outlier=outlier,
                          fits={lab: (x + 0.05 * (i + 1) * np.sin(np.arange(T) / (5.0 + i)), 0.2 + 0.01 * i + 0.0 * x)
                                for i, lab in enumerate(FILTERS[:-1])})
    return res
