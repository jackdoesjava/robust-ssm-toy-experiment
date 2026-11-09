"""The Kalman filter, its robust version, and the distances behind the input-dependent noise."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import M0, P0
from .metrics import gaussian_crps


def local_distance2(y, k):
    """Squared distance from each y_t to the median of the previous k readings. Before k
    readings exist, the prior mean M0 stands in for the missing ones, so the first reading
    is compared with the prior mean rather than accepted unchecked."""
    n, T = y.shape
    padded = np.concatenate([np.full((n, k), M0), y], axis=1)
    windows = sliding_window_view(padded, k, axis=1)[:, :T, :]  # window t = y[t-k : t]
    d = y - np.median(windows, axis=2)
    return d * d


def past_distance2(y, k):
    """local_distance2 one step later: at step t, the squared distance of the previous reading
    y_{t-1} from the median of the k readings before it, and 0 at the first step. It uses
    only y_1, ..., y_{t-1}, so noise built from it is set before y_t is seen."""
    d2 = local_distance2(y, k)
    return np.concatenate([np.zeros((len(y), 1)), d2[:, :-1]], axis=1)


def imq_weight(e, scale, c):
    """Inverse multiquadric weight w = (1 + e^2 / (c^2 * scale))^(-1/2), eq. (3)."""
    return 1.0 / np.sqrt(1.0 + e * e / (c * c * scale))


def _at(v, t):
    """Step t of a parameter given per row (or as a number) or per step."""
    return v[:, t] if v.ndim == 2 else v


def kalman_filter(y, R, Q, a=1.0, c=None, w_fixed=None, R_t=None, m0=M0, P0=P0,
                  store=True, score=False):
    """(Robust) Kalman filter for x_t = a_t x_{t-1} + q_t, y_t = x_t + r_t.

    y        (N, T) array, one sequence per row.
    R, Q, a  a number, an (N,) array (one value per row) or an (N, T) array (one per step).
    c        None gives the ordinary Kalman filter. A number (or an (N,) array) gives the
             robust filter of eqs. (3)-(4) with that scale; c = inf gives the ordinary
             filter again. The weight is measured against P_{t|t-1} + R_t.
    w_fixed  (N, T) weights to use as given, for the repeated passes of experiment 2.
    R_t      (N, T) noise to use instead of R in the update and the weight (the
             input-dependent noise). R is then only the base level used in the forecast.
    store    False skips storing the per-step outputs, to save memory when tuning.
    score    True also returns, per row, the mean CRPS ("fc_crps") and the summed log
             density ("loglik") of the one-step forecast N(m_{t|t-1}, P_{t|t-1} + R) of y_t.
             It uses the base R for every filter alike, because an R_t that depends on y_t
             would not give a density for y_t. score = "fc_crps" or "loglik" returns only
             that one.
    """
    N, T = y.shape
    R, Q, a = (np.asarray(v, dtype=float) for v in (R, Q, a))
    want_crps, want_ll = score is True or score == "fc_crps", score is True or score == "loglik"

    m = np.full(N, m0, dtype=float)
    P = np.full(N, P0, dtype=float)
    fc_crps, loglik = np.zeros(N), np.zeros(N)
    out = {k: np.empty((N, T)) for k in ("m_pred", "P_pred", "m", "P", "w")} if store else {}
    for t in range(T):
        a_t, Q_t, R_base = _at(a, t), _at(Q, t), _at(R, t)
        R_use = R_base if R_t is None else R_t[:, t]
        m_pred = a_t * m
        P_pred = a_t * a_t * P + Q_t
        e = y[:, t] - m_pred
        if want_crps or want_ll:
            S_fc = P_pred + R_base  # forecast variance of y_t
            if want_crps:
                fc_crps += gaussian_crps(y[:, t], m_pred, np.sqrt(S_fc))
            if want_ll:
                loglik += -0.5 * (np.log(2 * np.pi * S_fc) + e * e / S_fc)
        if w_fixed is not None:
            w = w_fixed[:, t]
        elif c is None:
            w = np.ones(N)
        else:
            w = imq_weight(e, P_pred + R_use, c)
        # the Kalman update with R replaced by R / w^2, eq. (4)
        S = P_pred + R_use / (w * w)
        K = P_pred / S
        m = m_pred + K * e
        P = P_pred - K * S * K
        if store:
            out["m_pred"][:, t], out["P_pred"][:, t] = m_pred, P_pred
            out["m"][:, t], out["P"][:, t], out["w"][:, t] = m, P, w
    if want_crps:
        out["fc_crps"] = fc_crps / T
    if want_ll:
        out["loglik"] = loglik
    return out
