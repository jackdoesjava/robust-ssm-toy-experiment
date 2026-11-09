"""Tuning of the filters' free parameters by grid search, from observations only."""

from dataclasses import dataclass

import numpy as np

from .config import (BOTH, BOTH_PAST, C_GRID, C_PRIOR, INDEP, INDEP_FIX, INDEP_PAST, K_GRID, K_PRIOR,
                     ONE_FC, ONE_ML, ONE_TRUE, R, R_GRID, ROB, ROB_FIX)
from .filters import kalman_filter, local_distance2, past_distance2
from .output import log
from .parallel import run_tasks


@dataclass(frozen=True)
class FittedParams:
    """The tuned parameters of experiment 1. A noise c of inf switches the input-dependent
    noise off, and its window k is then None."""
    R_ml: float             # one R, by maximum likelihood
    R_fc: float             # one R, by forecast CRPS
    R_by_c: dict            # robust filter: the best R for every c in C_GRID
    R_rob: float            # robust filter, tuned
    c_rob: float
    R_prior: float          # robust filter with c = C_PRIOR
    R_in: float             # input-dependent noise from y_t, tuned
    k_in: int | None
    c_in: float
    R_in_prior: float       # the same with c = C_PRIOR and k = K_PRIOR
    R_both: float           # robust filter on input-dependent noise from y_t
    k_both: int | None
    c_in_both: float        # the noise's c
    c_both: float           # the robust c
    R_past: float           # input-dependent noise from past readings only
    k_past: int | None
    c_past: float
    R_both_past: float      # robust filter on input-dependent noise from past readings
    k_both_past: int | None
    c_in_both_past: float
    c_both_past: float


@dataclass(frozen=True)
class FilterSpec:
    """A Gaussian filter of experiment 1: base R, robust scale c (None: no weight) and
    input-dependent noise R_t (None: R at every step)."""
    R: float
    c: float | None = None
    R_t: np.ndarray | None = None


def grid_scores(y, Q, key, c=None, d2=None, c_in=None):
    """Run the filter for every R in R_GRID at once (one row per R and sequence) and return
    the forecast score `key` averaged over sequences, one value per R. c is the robust scale
    (None = no robust weight); d2 (squared local distances) and c_in turn on the
    input-dependent noise R_t = R + d2 / c_in^2."""
    nR, n = len(R_GRID), len(y)
    Rs = np.repeat(R_GRID, n)
    R_t = None if d2 is None else Rs[:, None] + np.tile(d2, (nR, 1)) / c_in ** 2
    out = kalman_filter(np.tile(y, (nR, 1)), Rs, Q, c=c, R_t=R_t, store=False, score=key)
    return out[key].reshape(nR, n).mean(axis=1)


def best_over(scores_by_param):
    """(best score, best R, parameter) over a dict {parameter: scores over R_GRID}. Ties go to
    the first parameter in the dict, so the order of the table decides between equal fits."""
    best = (np.inf, None, None)
    for param, s in scores_by_param.items():
        j = int(np.argmin(s))
        if s[j] < best[0]:
            best = (float(s[j]), float(R_GRID[j]), param)
    return best


def _scores_over_c(args):
    """One task of fit_all: the validation forecast CRPS over R_GRID for every robust c in
    C_GRID, with input-dependent noise of window k and scale c_in (k = None: none), from y_t
    itself (past = False) or from past readings only (past = True)."""
    y, Q, k, c_in, past = args
    d2 = None if k is None else (past_distance2 if past else local_distance2)(y, k)
    return [grid_scores(y, Q, "fc_crps", c=c, d2=d2, c_in=c_in) for c in C_GRID]


def fit_all(y_tr, y_va, Q):
    """Free parameters of every filter, from observations only.

    The one-R Kalman filter's R maximises the likelihood on the training sequences;
    everything else minimises the CRPS of the one-step forecast of y_t on the validation
    sequences, over R_GRID, C_GRID and K_GRID. One table of scores, indexed by (k, noise c,
    robust c), serves every filter: each filter's search is the slice of it where its
    unused parts are switched off (c = inf), and the combined filters search the whole
    table, so their search includes both filters alone. A noise or robust scale of inf
    switches that part off; when the noise scale is inf the window k is not used and is
    stored as None.
    """
    finite = [float(c) for c in C_GRID if np.isfinite(c)]
    tasks = [(y_va, Q, None, None, False)] + [(y_va, Q, k, c_in, past) for past in (False, True)
                                              for k in K_GRID for c_in in finite]
    # table[(k, noise c, robust c)] = validation forecast CRPS for every R in R_GRID, with the
    # noise from y_t itself; table_past the same with the noise from past readings only. The
    # rows without noise (k = None, noise c = inf) are the same in both, and come first in each.
    table, table_past = {}, {}
    for (_, _, k, c_noise, past), scores in zip(tasks, run_tasks(_scores_over_c, tasks)):
        for c, s in zip(C_GRID, scores):
            key = (k, np.inf if c_noise is None else c_noise, float(c))
            if k is None:
                table[key] = table_past[key] = s
            else:
                (table_past if past else table)[key] = s
    R_ml = float(R_GRID[np.argmax(grid_scores(y_tr, Q, "loglik"))])
    R_fc = float(R_GRID[np.argmin(table[(None, np.inf, np.inf)])])
    # robust filter (no input-dependent noise): the best R for every c, then the best c
    rob = {c: s for (k, c_in, c), s in table.items() if k is None}
    R_by_c = {c: float(R_GRID[np.argmin(s)]) for c, s in rob.items()}
    _, R_rob, c_rob = best_over(rob)
    # input-dependent noise (no robust weight): tuned (R, k, c), and with c and k fixed in advance
    ind = {(k, c_in): s for (k, c_in, c), s in table.items() if c == np.inf}
    _, R_in, (k_in, c_in) = best_over(ind)
    R_in_prior = float(R_GRID[np.argmin(table[(K_PRIOR, C_PRIOR, np.inf)])])
    # both together: (R, k, noise c, robust c) over the whole table, which holds both filters above
    _, R_both, (k_both, c_in_both, c_both) = best_over(table)
    # the same two with the noise from past readings only
    ind_past = {(k, c_in): s for (k, c_in, c), s in table_past.items() if c == np.inf}
    _, R_past, (k_past, c_past) = best_over(ind_past)
    _, R_both_past, (k_both_past, c_in_both_past, c_both_past) = best_over(table_past)
    fit = FittedParams(R_ml=R_ml, R_fc=R_fc, R_by_c=R_by_c, R_rob=R_rob, c_rob=c_rob, R_prior=R_by_c[C_PRIOR],
                       R_in=R_in, k_in=k_in, c_in=c_in, R_in_prior=R_in_prior,
                       R_both=R_both, k_both=k_both, c_in_both=c_in_both, c_both=c_both,
                       R_past=R_past, k_past=k_past, c_past=c_past,
                       R_both_past=R_both_past, k_both_past=k_both_past, c_in_both_past=c_in_both_past,
                       c_both_past=c_both_past)
    for name in ("R_ml", "R_fc", "R_rob", "R_prior", "R_in", "R_in_prior", "R_both", "R_past", "R_both_past"):
        value = getattr(fit, name)
        if value in (R_GRID[0], R_GRID[-1]):
            log(f"  warning: {name} = {value:.3g} is at the edge of the search grid")
    return fit


def filter_specs(fit, y):
    """The Gaussian filters of experiment 1 with their fitted parameters, for data y."""
    def noise(R_, k, c, past=False):  # R_t = R + d_t^2 / c^2 (d_{t-1} if past); just R when c = inf
        if not np.isfinite(c):
            return np.full(y.shape, R_)
        return R_ + (past_distance2 if past else local_distance2)(y, k) / c ** 2

    return {
        ONE_TRUE: FilterSpec(R=R),
        ONE_ML: FilterSpec(R=fit.R_ml),
        ONE_FC: FilterSpec(R=fit.R_fc),
        INDEP: FilterSpec(R=fit.R_in, R_t=noise(fit.R_in, fit.k_in, fit.c_in)),
        INDEP_FIX: FilterSpec(R=fit.R_in_prior, R_t=noise(fit.R_in_prior, K_PRIOR, C_PRIOR)),
        INDEP_PAST: FilterSpec(R=fit.R_past, R_t=noise(fit.R_past, fit.k_past, fit.c_past, past=True)),
        ROB: FilterSpec(R=fit.R_rob, c=fit.c_rob),
        ROB_FIX: FilterSpec(R=fit.R_prior, c=C_PRIOR),
        BOTH: FilterSpec(R=fit.R_both, c=fit.c_both, R_t=noise(fit.R_both, fit.k_both, fit.c_in_both)),
        BOTH_PAST: FilterSpec(R=fit.R_both_past, c=fit.c_both_past,
                              R_t=noise(fit.R_both_past, fit.k_both_past, fit.c_in_both_past, past=True)),
    }


def run_filter(spec, y, Q):
    """Run the filter that spec describes on y."""
    return kalman_filter(y, spec.R, Q, c=spec.c, R_t=spec.R_t)
