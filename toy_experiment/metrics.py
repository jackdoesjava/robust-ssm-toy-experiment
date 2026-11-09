"""Scores of Gaussian forecasts and the summary metrics of experiment 1."""

import numpy as np
from scipy.special import ndtr  # standard normal CDF

from .config import AFTER, NEAR, Z90


def gaussian_crps(x, m, s):
    """CRPS of N(m, s^2) at the true value x, lower is better. This is the closed form of
    Gneiting and Raftery (2007, Section 4.2) in its negative orientation (their CRPS*)."""
    z = (x - m) / s
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return s * (z * (2 * ndtr(z) - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def gaussian_scores(x, m, P):
    """Per-step mean, PIT, CRPS and 90% width of the posteriors N(m, P), scored at x."""
    s = np.sqrt(P)
    return dict(mean=m, pit=ndtr((x - m) / s), crps=gaussian_crps(x, m, s), width90=2 * Z90 * s)


def first_below(bias):
    """Index of the first step with bias below NEAR, or len(bias) if there is none."""
    below = bias < NEAR
    return int(np.argmax(below)) if below.any() else len(bias)


def summarise(x, sc, jump_t, boot_idx):
    """Metrics for one filter on the test sequences.

    Accuracy and calibration use the steps before the jump: per-sequence values, then their
    mean over sequences and the half-width of a 95% interval for that mean. The 90%
    interval covers the truth when its PIT value F(x_t) is between 0.05 and 0.95. The
    response to the jump is the delay: steps after the jump until the error averaged across
    sequences, x_t - m_t, first falls below NEAR (80% of the jump recovered; 0 = already at
    the first step after it), with a 95% interval from resampling sequences, plus the RMSE
    over the AFTER steps from the jump on. Returns the summary, the per-sequence values and
    the bootstrap delays.
    """
    pre = slice(0, jump_t)
    err = sc["mean"] - x
    pit = sc["pit"][:, pre]
    per_seq = dict(
        rmse=np.sqrt((err[:, pre] ** 2).mean(axis=1)),
        crps=sc["crps"][:, pre].mean(axis=1),
        cov90=((pit >= 0.05) & (pit <= 0.95)).mean(axis=1),
        cov50=((pit >= 0.25) & (pit <= 0.75)).mean(axis=1),
        width90=sc["width90"][:, pre].mean(axis=1),
        rmse_after=np.sqrt((err[:, jump_t:jump_t + AFTER] ** 2).mean(axis=1)),
    )
    n = len(x)
    summ = {key: (float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(n))) for key, v in per_seq.items()}
    lag = -err[:, jump_t:]
    boots = np.array([first_below(lag[i].mean(axis=0)) for i in boot_idx])
    summ["delay"] = (float(first_below(lag.mean(axis=0))),
                     float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return summ, per_seq, boots
