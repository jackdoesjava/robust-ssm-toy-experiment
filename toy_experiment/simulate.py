"""Data generators for the two experiments."""

import numpy as np

from .config import EPS, JUMP, LAM, M0, P0, R, R2, SIGMA2


def outlier_mask(rng, n, T, eps, stay):
    """Which readings are outliers. stay = None: each is an outlier with probability eps,
    independently. Otherwise a Markov chain with P(outlier | outlier before) = stay and the
    same long-run rate eps, so outliers come in runs of mean length 1 / (1 - stay)."""
    if stay is None:
        return rng.random((n, T)) < eps
    start = eps * (1 - stay) / (1 - eps)
    u = rng.random((n, T))
    o = np.empty((n, T), dtype=bool)
    o[:, 0] = u[:, 0] < eps
    for t in range(1, T):
        o[:, t] = u[:, t] < np.where(o[:, t - 1], stay, start)
    return o


def simulate_rw(rng, n, T, Q, eps, lam, stay, jump_t):
    """n random-walk sequences of length T with outliers and, if jump_t is set, one jump.

    Returns the true states x, the observations y and a mask marking the outliers.
    Column t holds step t + 1; the initial state x_0 ~ N(M0, P0) is not stored.
    """
    x0 = rng.normal(M0, np.sqrt(P0), size=n)
    x = x0[:, None] + np.cumsum(rng.normal(0.0, np.sqrt(Q), size=(n, T)), axis=1)
    if jump_t is not None:
        x[:, jump_t:] += JUMP
    outlier = outlier_mask(rng, n, T, eps, stay)
    r = rng.normal(size=(n, T)) * np.sqrt(np.where(outlier, lam * R, R))
    return x, x + r, outlier


def simulate_selective(rng, n, T, a_med, cap=None, eps=EPS, lam=LAM, phi=0.98, s_u=0.75):
    """Sequences with input-dependent decay a_t = exp(-Delta_t), as in eq. (1).

    The input u_t is a slowly varying AR(1) signal and Delta_t = -log(a_med) * exp(u_t),
    so a_t moves around a_med (and is capped at `cap` if given). Q_t = (1 - a_t^2) SIGMA2
    keeps the state's variance at SIGMA2. a_med = None gives a random walk (a_t = 1,
    Q_t = 0.1). Returns x, y and the per-step a_t and Q_t, which the filter is given.
    """
    if a_med is None:
        a = np.ones((n, T))
        Qs = np.full((n, T), 0.1)
    else:
        u = np.empty((n, T))
        u[:, 0] = rng.normal(0.0, s_u, size=n)
        z = rng.normal(size=(n, T))
        for t in range(1, T):
            u[:, t] = phi * u[:, t - 1] + np.sqrt(1 - phi ** 2) * s_u * z[:, t]
        a = np.exp(np.log(a_med) * np.exp(u))
        if cap is not None:
            a = np.minimum(a, cap)
        Qs = (1 - a ** 2) * SIGMA2
    x = np.empty((n, T))
    prev = rng.normal(0.0, np.sqrt(SIGMA2), size=n)
    noise = rng.normal(size=(n, T))
    for t in range(T):
        prev = a[:, t] * prev + np.sqrt(Qs[:, t]) * noise[:, t]
        x[:, t] = prev
    outlier = rng.random((n, T)) < eps
    y = x + rng.normal(size=(n, T)) * np.sqrt(np.where(outlier, lam * R2, R2))
    return x, y, a, Qs
