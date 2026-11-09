"""The repeated frozen-weight iteration of Section 5.2, and the passes it needs (experiment 2)."""

import numpy as np

from .config import R2, SIGMA2, TOLS
from .filters import imq_weight, kalman_filter


def frozen_passes(y, Qs, a, c):
    """The repeated frozen-weight iteration. Yields the filter output of pass 1 (every
    weight 1), pass 2, and so on, without end.

    Each pass is a Kalman filter with the weights frozen, which is what one parallel scan
    computes; the next pass's weights are recomputed from this pass's predictions. Since a
    weight only depends on the steps before it, pass k gets the first k - 1 means exactly.
    """
    w = np.ones_like(y)
    while True:
        run = kalman_filter(y, R2, Qs, a, w_fixed=w, P0=SIGMA2)
        yield run
        w = imq_weight(y - run["m_pred"], run["P_pred"] + R2, c)


def _first_pass_below(arr, tol):
    """Per sequence, the first pass (counting from 1) at which arr is below tol; -1 if none."""
    hit = arr < tol
    return np.where(hit.any(axis=0), hit.argmax(axis=0) + 1, -1)


def count_passes(y, Qs, a, c, maxpass):
    """Passes of the repeated iteration against the sequential robust filter.

    For each tolerance in TOLS: the passes each sequence needs before every filtered mean
    is within TOL posterior sd of the sequential filter, and the passes a practical stopping
    rule needs (stop once no mean moved by more than TOL sd since the previous pass); -1 if
    not reached within maxpass. Also the error after each pass; the contraction factor, the
    geometric-mean shrink of the error per pass from pass 2 to the first pass within the
    tightest tolerance; and the filter's memory a_{t+1} (1 - K_t) along the sequential run
    (the scalar A_{t+1}(I - K_t H): how much of the prediction for step t is still in the
    prediction for step t + 1), for t = 1, ..., T - 1.
    """
    seq = kalman_filter(y, R2, Qs, a, c=c, P0=SIGMA2)
    sd = np.sqrt(seq["P"])
    errs, moves, prev = [], [], None
    for k, run in enumerate(frozen_passes(y, Qs, a, c), start=1):
        errs.append((np.abs(run["m"] - seq["m"]) / sd).max(axis=1))
        moves.append(np.full(len(y), np.inf) if prev is None else (np.abs(run["m"] - prev) / sd).max(axis=1))
        prev = run["m"]
        if ((errs[-1] < min(TOLS)) & (moves[-1] < min(TOLS))).all() or k >= maxpass:
            break
    errs, moves = np.array(errs), np.array(moves)  # (passes, sequences)

    passes = {tol: _first_pass_below(errs, tol) for tol in TOLS}
    practical = {tol: _first_pass_below(moves, tol) for tol in TOLS}
    factor = np.full(len(y), np.nan)
    for i, Ki in enumerate(passes[min(TOLS)]):
        if Ki > 2 and errs[1, i] > 0:
            factor[i] = (errs[Ki - 1, i] / errs[1, i]) ** (1 / (Ki - 2))
    K = seq["P_pred"] / (seq["P_pred"] + R2 / seq["w"] ** 2)
    return dict(passes=passes, practical=practical, errs=errs, factor=factor, memory=a[:, 1:] * (1 - K[:, :-1]))
