"""The Bayes filter for the random walk with outliers, computed on a moving grid."""

import numpy as np
from scipy.ndimage import convolve1d

from . import parallel
from .config import GRID_DX, M0, P0, R

_u, _w = np.polynomial.legendre.leggauss(4)
GL_U, GL_W = (_u + 1) / 2, _w / 2  # 4-point Gauss-Legendre rule on [0, 1], exact up to degree 7


def normal_logpdf(x, mean, var):
    return -0.5 * (x - mean) ** 2 / var - 0.5 * np.log(2 * np.pi * var)


def exact_filter(y, x, Q, eps, lam, stay=None, half_width=None, dx=GRID_DX):
    """Bayes filter for the random walk with outliers, computed on a fine grid.

    It tracks two densities, p(x_t, reading t is clean | y_1:t) and p(x_t, reading t is an
    outlier | y_1:t). Prediction convolves both with the N(0, Q) density (directly, with the
    kernel cut at 9 sd, so the tiny tail values that matter after the jump stay accurate;
    the cut is accurate for clean readings up to about 8 predictive sd from the prediction)
    and applies the outlier chain's transition; the update multiplies by N(y_t; x_t, R) or
    N(y_t; x_t, lam R). With independent outliers this is Theorem 4.1 of Sarkka (2013),
    eqs. (4.11)-(4.12), done numerically. The mean is the grid sum. The CDF at the grid
    points is the cumulative trapezoid sum with the Euler-Maclaurin end correction, which
    is accurate to O(dx^4); between grid points it is the cubic that matches those values
    and has the density as its slope. The PIT, the 90% interval and the CRPS are computed
    exactly for that CDF (Gauss-Legendre quadrature, exact for these polynomials, and
    bisection to round-off for the quantiles). The grid covers +-half_width around the
    current mean (15 + 10 sqrt(Q) unless given) and moves with it. x (the true states) is
    only used to score each step. Returns per-step scores like gaussian_scores().
    """
    N, T = y.shape
    if stay is None:
        p_start = p_stay = eps
    else:
        p_stay, p_start = stay, eps * (1 - stay) / (1 - eps)
    if half_width is None:
        half_width = 15.0 + 10.0 * np.sqrt(Q)
    G = int(round(half_width / dx))
    n_pts = 2 * G + 1
    offs = np.arange(-G, G + 1) * dx
    centre = np.full(N, float(M0))
    base = np.exp(normal_logpdf(offs, 0.0, P0))
    base /= (base.sum() - 0.5 * (base[0] + base[-1])) * dx
    # joint density of x and the previous reading's type, which starts at the long-run rate
    g = np.stack([np.tile((1 - eps) * base, (N, 1)), np.tile(eps * base, (N, 1))])
    k = int(np.ceil(9 * np.sqrt(Q) / dx))
    kern = np.exp(normal_logpdf(np.arange(-k, k + 1) * dx, 0.0, Q))
    kern /= kern.sum()
    sc = {key: np.empty((N, T)) for key in ("mean", "pit", "crps", "width90")}

    for t in range(T):
        gc, go = _predict(g, kern, p_start, p_stay)
        grid = centre[:, None] + offs[None, :]
        gc, go = _update(gc, go, grid, y[:, t:t + 1], eps, lam)
        p = gc + go
        F = _grid_cdf(p, dx)
        Z = np.maximum(F[:, -1:], 1e-300)
        gc, go, p, F = gc / Z, go / Z, p / Z, F / Z
        sc["mean"][:, t] = (grid * p).sum(axis=1) * dx
        u = (x[:, t] - grid[:, 0]) / dx  # position of the true state, in cells
        j = np.clip(np.floor(u).astype(int), 0, n_pts - 2)
        th = np.clip(u - j, 0.0, 1.0)
        co = _cell_cubic(F, p, j, dx)
        sc["pit"][:, t] = _cubic_at(co, th)
        sc["crps"][:, t] = _crps(F, p, co, u, j, th, dx)
        sc["width90"][:, t] = _width90(F, p, offs, dx)
        gc, go, centre = _recentre(gc, go, centre, sc["mean"][:, t], dx)
        g = np.stack([gc, go])
    return sc


def _predict(g, kern, p_start, p_stay):
    """Convolve both densities with the process-noise density, then apply the outlier chain."""
    N = g.shape[1]
    h = np.clip(convolve1d(g.reshape(2 * N, -1), kern, axis=1, mode="constant", cval=0.0),
                0.0, None).reshape(2, N, -1)
    return (1 - p_start) * h[0] + (1 - p_stay) * h[1], p_start * h[0] + p_stay * h[1]


def _update(gc, go, grid, yt, eps, lam):
    """Multiply by the likelihood of y_t under each reading type, scaled so the largest is 1."""
    llc = normal_logpdf(yt, grid, R)
    llo = normal_logpdf(yt, grid, lam * R)
    top = llc.max(axis=1, keepdims=True)
    if eps > 0:
        top = np.maximum(top, llo.max(axis=1, keepdims=True))
    gc = gc * np.exp(llc - top)
    go = go * np.exp(llo - top) if eps > 0 else np.zeros_like(gc)
    return gc, go


def _grid_cdf(p, dx):
    """CDF of the density p at the grid points: cumulative trapezoid sums minus
    (dx^2 / 12) (p'_i - p'_0), with p' by central differences and the density zero beyond
    the grid (the Euler-Maclaurin correction, which makes the CDF accurate to O(dx^4))."""
    dp = np.empty_like(p)
    dp[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dx)
    dp[:, 0], dp[:, -1] = p[:, 1] / (2 * dx), -p[:, -2] / (2 * dx)
    return dx * (np.cumsum(p, axis=1) - 0.5 * (p + p[:, :1])) - dx * dx / 12 * (dp - dp[:, :1])


def _cell_cubic(F, p, j, dx):
    """The CDF in cell j as a cubic in the fraction th of the cell crossed: values F_j and
    F_j+1 at the ends, slopes p_j and p_j+1. Returns its four power-basis coefficients."""
    rows = np.arange(len(F))
    F0 = F[rows, j]
    m0, m1 = p[rows, j] * dx, p[rows, j + 1] * dx
    D = F[rows, j + 1] - F0
    return F0, m0, 3 * D - 2 * m0 - m1, m0 + m1 - 2 * D


def _cubic_at(co, th):
    c0, c1, c2, c3 = co
    return c0 + th * (c1 + th * (c2 + th * c3))


def _crps(F, p, co, u, j, th, dx):
    """CRPS of the grid CDF at the truth, which lies a fraction th into cell j (u cells from
    the grid's first point; co is the CDF's cubic in that cell).

    CRPS = integral of F^2 left of the truth + (1 - F)^2 right of it. Both integrands are
    degree-6 polynomials in each cell, so the 4-point rule is exact; the truth's cell is
    split at the truth.
    """
    N, n_pts = F.shape
    rows = np.arange(N)
    F0, m0, m1 = F[:, :-1], p[:, :-1] * dx, p[:, 1:] * dx
    D = F[:, 1:] - F0
    c2, c3 = 3 * D - 2 * m0 - m1, m0 + m1 - 2 * D
    A, B = np.zeros_like(F0), np.zeros_like(F0)
    for uq, wq in zip(GL_U, GL_W):
        Fq = F0 + uq * (m0 + uq * (c2 + uq * c3))
        A += wq * Fq * Fq
        B += wq * (1 - Fq) * (1 - Fq)
    cA = np.concatenate([np.zeros((N, 1)), np.cumsum(A * dx, axis=1)], axis=1)
    cB = np.concatenate([np.cumsum(B[:, ::-1] * dx, axis=1)[:, ::-1], np.zeros((N, 1))], axis=1)
    left = right = 0.0
    for uq, wq in zip(GL_U, GL_W):
        left = left + wq * _cubic_at(co, th * uq) ** 2
        right = right + wq * (1 - _cubic_at(co, th + (1 - th) * uq)) ** 2
    left = cA[rows, j] + dx * th * left
    right = cB[rows, j + 1] + dx * (1 - th) * right
    beyond = dx * (np.maximum(-u, 0.0) + np.maximum(u - (n_pts - 1), 0.0))  # truth outside the grid
    return left + right + beyond


def _width90(F, p, offs, dx):
    """Width of the 90% interval: the 5% and 95% quantiles, by bisection on the cubic in the
    cell where the CDF crosses the level."""
    N, n_pts = F.shape
    q = []
    for level in (0.05, 0.95):
        jq = np.clip(np.argmax(F > level, axis=1) - 1, 0, n_pts - 2)
        cq = _cell_cubic(F, p, jq, dx)
        lo, hi = np.zeros(N), np.ones(N)
        for _ in range(52):
            mid = 0.5 * (lo + hi)
            below = _cubic_at(cq, mid) < level
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        q.append(offs[jq] + 0.5 * (lo + hi) * dx)
    return q[1] - q[0]


def _recentre(gc, go, centre, mean, dx):
    """Move the grid by whole cells so it stays centred on the estimate."""
    n_pts = gc.shape[1]
    shift = np.rint((mean - centre) / dx).astype(int)
    idx = np.arange(n_pts)[None, :] + shift[:, None]
    inside = (idx >= 0) & (idx < n_pts)
    safe = np.clip(idx, 0, n_pts - 1)
    gc = np.where(inside, np.take_along_axis(gc, safe, axis=1), 0.0)
    go = np.where(inside, np.take_along_axis(go, safe, axis=1), 0.0)
    return gc, go, centre + shift * dx


def _exact_task(args):
    return exact_filter(*args)


def bayes_filter(y, x, Q, eps, lam, stay=None):
    """exact_filter, with the sequences split across the worker processes when there are any.
    Each sequence is filtered on its own, so the result is the same as one call."""
    n = parallel.n_workers()
    if not parallel.active() or len(y) < 2 * n:
        return exact_filter(y, x, Q, eps, lam, stay)
    chunks = np.array_split(np.arange(len(y)), n)
    parts = parallel.run_tasks(_exact_task, [(y[i], x[i], Q, eps, lam, stay) for i in chunks])
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
