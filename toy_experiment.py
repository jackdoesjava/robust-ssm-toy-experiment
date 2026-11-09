"""
Toy experiments for the Warwick MSc by Research proposal (Section 4).

How to run, from this folder (Python 3 with numpy, scipy and matplotlib):

    python toy_experiment.py               full run, about 80 minutes with 6 workers, into results/
    python toy_experiment.py --quick       small run (a few minutes), into results_quick/
    python toy_experiment.py --only exp2   one part (checks, exp1, exp2 or sweep), into results_partial/
    --workers N                            processes for the tuning and the Bayes filter (default up
                                           to 6; 1 runs everything in one process, with the same
                                           numbers, but takes several hours)

Everything is seeded, so a rerun gives the same numbers. Each part writes its files as
soon as it finishes. The log is run_log.txt, and DONE.txt appears when a run that did
every part has finished.

Experiment 1 (RQ1 in miniature)
    The Gaussian random walk of Sarkka (2013), Example 4.1, here with Q = 0.1:
        x_t = x_{t-1} + q_t,   q_t ~ N(0, Q),   Q = 0.1
        y_t = x_t + r_t
    Most r_t ~ N(0, R) with R = 1, but 10% of the readings are outliers, r_t ~ N(0, 100 R).
    In the test sequences the true state is 5 higher from step 501 on. The filters (all
    know Q and the prior on x_0):
      * Kalman filter with R = 1, the clean noise level, treating every reading as clean.
      * Kalman filter with one R for every step: learned by maximum likelihood, or chosen
        by the forecast criterion below.
      * Kalman filter with input-dependent noise R_t = R + d_t^2 / c^2, where d_t is the
        distance from y_t to the median of the previous k readings (the prior mean stands
        in for readings before the first). It stands in for a network that sets the noise
        from the inputs, y_t included. Tuned (R, c, k with k in {3, 5, 9}), and with
        c = 2, k = 5 fixed.
      * The same with noise from past readings only, R_t = R + d_{t-1}^2 / c^2 (R_1 = R),
        which uses y_1, ..., y_{t-1}: a network that must set the noise before it sees
        y_t, as a proper forecast density of y_t requires. Tuned (R, c, k).
      * Robust filter: the Kalman update with R replaced by R / w_t^2, using the weight of
        proposal eq. (3). Tuned (R, c), and with c = 2 fixed.
      * Robust filter on top of the input-dependent noise, a stand-in for the model RQ1 asks
        about: R_t as above, and the weight of eq. (3) measured against P_{t|t-1} + R_t.
        R, the robust c and the noise's c and k are tuned together, so the search includes
        the input-dependent filter (robust c = inf) and the robust filter (noise c = inf).
        Once with each of the two kinds of input-dependent noise.
      * The Bayes filter for the outlier model, computed on a fine grid. It knows how the
        outliers are generated but, like every other filter, not about the jump, so it has
        the lowest expected error before the jump and is only a reference after it.
    Every free parameter is fitted on training and validation sequences that have no jump,
    using observations only: the one-R Kalman filter's R maximises the likelihood on the
    training sequences, and everything else minimises the CRPS of the one-step forecast
    N(m_{t|t-1}, P_{t|t-1} + R) of y_t on the validation sequences. The forecast uses the
    base R for every filter, the past-only ones included, so that all are scored alike: with
    an R_t that depends on y_t, N(y_t; m, P + R_t) is not a density in y_t (its integral
    over y_t is infinite), so it cannot score a forecast.
    Accuracy and calibration are measured before the jump, and the response to the jump
    separately.

Experiment 2 (RQ2 in miniature)
    The same kind of outliers, with input-dependent decay as in proposal eq. (1):
        x_t = a_t x_{t-1} + q_t,   a_t = exp(-Delta_t),   q_t ~ N(0, (1 - a_t^2) SIGMA2)
    Delta_t follows a slowly varying input. It never depends on the weights, as in the
    proposal, where the network computes it from the observations. The robust filter is
    computed the way Section 5.2 proposes: start with every weight at 1, run a Kalman
    filter with the weights frozen, recompute every weight from that run's predictions,
    and repeat. One frozen-weight run is what one parallel scan computes (Sarkka and
    Garcia-Fernandez, 2021). Here it is run step by step, which gives the same numbers up
    to round-off.
    The experiment counts the passes needed to match the one-step-at-a-time robust filter.

Sweep
    Experiment 1 under eight other settings: clean data, rarer, more frequent, milder and
    larger outliers, outliers that only appear at test time, outliers in bursts, and a
    faster-moving state with the main setting's outliers.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402
import scipy  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy.ndimage import convolve1d  # noqa: E402
from scipy.special import ndtr  # noqa: E402  (standard normal CDF)

HERE = Path(__file__).resolve().parent
_LOG = None   # log file of the current run
_POOL = None  # worker processes of the current run (None: everything runs in this process)
_WORKERS = 1
WORKERS_DEFAULT = max(1, min(6, (os.cpu_count() or 2) - 2))

# ---------------------------------------------------------------- settings
R = 1.0            # clean observation noise variance (experiment 1)
EPS = 0.1          # long-run fraction of readings that are outliers
LAM = 100.0        # outliers have variance LAM * R, i.e. standard deviation 10
JUMP = 5.0         # size of the real change in the state
M0, P0 = 0.0, 1.0  # prior on the initial state
NEAR = 1.0         # the jump counts as followed once the average error is below NEAR (80%)
AFTER = 50         # steps after the jump used for the post-jump RMSE
C_PRIOR = 2.0      # scale fixed in advance: a 1-sd residual keeps w = 0.89
K_PRIOR = 5        # window fixed in advance for the input-dependent noise
N_BOOT = 2000      # bootstrap resamples of the test sequences, for the jump delay
GRID_DX = 0.025    # grid spacing of the Bayes filter

R_GRID = np.logspace(-2, 3, 101)  # candidate R values from 0.01 to 1000, 20 per decade
C_GRID = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 5, 6, 8, 10, 15,
                   np.inf])     # inf switches the down-weighting off
K_GRID = (3, 5, 9)              # window lengths tried for the input-dependent noise

# Experiment 1 and the sweep: (name, Q, outliers when fitting, outliers at test time).
# Outliers are (long-run fraction, variance multiplier, persistence). Persistence None means
# independent outliers; a number p means P(outlier | outlier at the previous step) = p.
SCENARIOS = [
    ("main: 10% outliers, sd 10", 0.1, (0.1, 100.0, None), (0.1, 100.0, None)),
    ("clean data", 0.1, (0.0, 100.0, None), (0.0, 100.0, None)),
    ("rare outliers: 5%, sd 10", 0.1, (0.05, 100.0, None), (0.05, 100.0, None)),
    ("frequent outliers: 20%, sd 10", 0.1, (0.2, 100.0, None), (0.2, 100.0, None)),
    ("frequent mild outliers: 20%, sd 3", 0.1, (0.2, 9.0, None), (0.2, 9.0, None)),
    ("huge outliers: 10%, sd 32", 0.1, (0.1, 1000.0, None), (0.1, 1000.0, None)),
    ("outliers at test time only: 10%, sd 10", 0.1, (0.0, 100.0, None), (0.1, 100.0, None)),
    ("outlier bursts: 10%, sd 10, mean run 5", 0.1, (0.1, 100.0, 0.8), (0.1, 100.0, 0.8)),
    ("fast-moving state: Q = 1, 10% outliers, sd 10", 1.0, (0.1, 100.0, None), (0.1, 100.0, None)),
]
SHORT_TEX = [  # row labels for the sweep tables, in SCENARIOS order
    r"10\% outliers, sd 10 (main)", r"none (clean data)", r"5\%, sd 10", r"20\%, sd 10", r"20\%, sd 3",
    r"10\%, sd 32", r"10\%, sd 10, test time only", r"10\%, sd 10, in bursts", r"10\%, sd 10, fast state",
]

# Experiment 2
SIGMA2 = 1.0         # stationary variance of the decaying state
R2 = 0.25            # clean observation noise variance
C2 = 2.0             # robust scale
TOLS = (1e-2, 1e-8)  # passes are counted until every mean is within TOL posterior sd
REGIMES = [          # (label, median a_t, cap on a_t, seed id); None = random walk, a_t = 1
    ("a_t <= 0.6", 0.3, 0.6, 105),
    ("a_t ~ 0.5", 0.5, None, 100),
    ("a_t ~ 0.9", 0.9, None, 101),
    ("a_t ~ 0.99", 0.99, None, 102),
    ("a_t ~ 0.999", 0.999, None, 103),
    ("random walk", None, None, 104),
]

FULL = dict(T=1000, jump_t=500, n_train=100, n_val=100, n_test=200, n_stability=5,
            T2_list=[100, 1000, 10000], T2_mid=1000, n2=10, maxpass=400,
            scenarios=list(range(len(SCENARIOS))))
QUICK = dict(T=300, jump_t=150, n_train=10, n_val=10, n_test=20, n_stability=2,
             T2_list=[100, 300], T2_mid=300, n2=3, maxpass=100,
             scenarios=[0, 1, 6, 7, 8])

ONE_TRUE = "Kalman, R = 1"
ONE_ML, ONE_FC = "Kalman, one R (max likelihood)", "Kalman, one R (forecast CRPS)"
INDEP, INDEP_FIX = "Kalman, input-dependent R_t", f"Kalman, input-dependent R_t, c = {C_PRIOR:g} fixed"
INDEP_PAST = "Kalman, input-dependent R_t (past only)"
ROB, ROB_FIX, BOTH = "Robust, tuned", f"Robust, c = {C_PRIOR:g} fixed", "Robust + input-dependent R_t"
BOTH_PAST = "Robust + input-dependent R_t (past only)"
BAYES = "Bayes filter (reference)"
FILTERS = [ONE_TRUE, ONE_ML, ONE_FC, INDEP, INDEP_FIX, INDEP_PAST, ROB, ROB_FIX, BOTH, BOTH_PAST, BAYES]
PAIRS = [(ROB, ONE_ML), (ROB, ONE_FC), (ROB, INDEP), (BOTH, INDEP), (BOTH_PAST, INDEP_PAST), (ROB_FIX, INDEP_FIX),
         (ROB, BAYES)]

Z90 = 1.6448536269514722  # 95% point of N(0, 1)

# Chart style: categorical slots 1-3, an ordinal blue ramp, neutral inks, TrueType fonts
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "font.family": "sans-serif",
         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:  # a console that cannot show a character (the log file is UTF-8)
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    if _LOG is not None:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def clean_json(o):
    """Make numbers JSON-safe: numpy types to Python, inf to the string "inf"."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if np.isnan(v) else (v if np.isfinite(v) else ("inf" if v > 0 else "-inf"))
    return o


def write_json(path, obj):
    path.write_text(json.dumps(clean_json(obj), indent=1), encoding="utf-8")


# ---------------------------------------------------------------- data
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
    """Sequences with input-dependent decay a_t = exp(-Delta_t), as in proposal eq. (1).

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


# ---------------------------------------------------------------- filters
def imq_weight(e, scale, c):
    """Inverse multiquadric weight w = (1 + e^2 / (c^2 * scale))^(-1/2), proposal eq. (3)."""
    return 1.0 / np.sqrt(1.0 + e * e / (c * c * scale))


def kalman_filter(y, R, Q, a=1.0, c=None, w_fixed=None, R_t=None, m0=M0, P0=P0,
                  store=True, score=False):
    """(Robust) Kalman filter for x_t = a_t x_{t-1} + q_t, y_t = x_t + r_t.

    y        (N, T) array, one sequence per row.
    R, Q, a  a number, an (N,) array (one value per row) or an (N, T) array (one per step).
    c        None gives the ordinary Kalman filter. A number (or an (N,) array) gives the
             robust filter of proposal eqs. (3)-(4) with that scale; c = inf gives the
             ordinary filter again. The weight is measured against P_{t|t-1} + R_t.
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

    def at(v, t):
        return v[:, t] if v.ndim == 2 else v

    m = np.full(N, m0, dtype=float)
    P = np.full(N, P0, dtype=float)
    fc_crps, loglik = np.zeros(N), np.zeros(N)
    out = {k: np.empty((N, T)) for k in ("m_pred", "P_pred", "m", "P", "w")} if store else {}
    for t in range(T):
        a_t, Q_t, R_base = at(a, t), at(Q, t), at(R, t)
        R_use = R_base if R_t is None else R_t[:, t]
        # Prediction step
        m_pred = a_t * m
        P_pred = a_t * a_t * P + Q_t
        e = y[:, t] - m_pred  # residual y_t - yhat_t
        if want_crps or want_ll:
            S_fc = P_pred + R_base  # forecast variance of y_t
            if want_crps:
                fc_crps += gaussian_crps(y[:, t], m_pred, np.sqrt(S_fc))
            if want_ll:
                loglik += -0.5 * (np.log(2 * np.pi * S_fc) + e * e / S_fc)
        # Weight
        if w_fixed is not None:
            w = w_fixed[:, t]
        elif c is None:
            w = np.ones(N)
        else:
            w = imq_weight(e, P_pred + R_use, c)
        # Update step: the Kalman update with R replaced by R / w^2 (proposal eq. 4)
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
    rows, cols = np.arange(N), np.arange(n_pts)
    sc = {key: np.empty((N, T)) for key in ("mean", "pit", "crps", "width90")}
    gl_x, gl_w = np.polynomial.legendre.leggauss(4)
    gl_u, gl_w = (gl_x + 1) / 2, gl_w / 2  # 4-point rule on [0, 1], exact up to degree 7

    def cubic(F, p, j):
        """The CDF in cell j as a cubic in the fraction th of the cell crossed: values F_j and
        F_j+1 at the ends, slopes p_j and p_j+1. Returns its four power-basis coefficients."""
        F0 = F[rows, j]
        m0, m1 = p[rows, j] * dx, p[rows, j + 1] * dx
        D = F[rows, j + 1] - F0
        return F0, m0, 3 * D - 2 * m0 - m1, m0 + m1 - 2 * D

    def at(co, th):
        c0, c1, c2, c3 = co
        return c0 + th * (c1 + th * (c2 + th * c3))

    for t in range(T):
        # Prediction: convolve with the process-noise density, then the outlier chain
        h = np.clip(convolve1d(g.reshape(2 * N, -1), kern, axis=1, mode="constant", cval=0.0),
                    0.0, None).reshape(2, N, -1)
        gc = (1 - p_start) * h[0] + (1 - p_stay) * h[1]
        go = p_start * h[0] + p_stay * h[1]
        # Update: multiply by the likelihood of y_t under each reading type, normalise
        grid = centre[:, None] + offs[None, :]
        yt = y[:, t:t + 1]
        llc = normal_logpdf(yt, grid, R)
        llo = normal_logpdf(yt, grid, lam * R)
        top = llc.max(axis=1, keepdims=True)
        if eps > 0:
            top = np.maximum(top, llo.max(axis=1, keepdims=True))
        gc = gc * np.exp(llc - top)
        go = go * np.exp(llo - top) if eps > 0 else np.zeros_like(gc)
        p = gc + go
        # CDF at the grid points: cumulative trapezoid sums minus (dx^2 / 12) (p'_i - p'_0), with
        # p' by central differences and the density zero beyond the grid
        dp = np.empty_like(p)
        dp[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dx)
        dp[:, 0], dp[:, -1] = p[:, 1] / (2 * dx), -p[:, -2] / (2 * dx)
        F = dx * (np.cumsum(p, axis=1) - 0.5 * (p + p[:, :1])) - dx * dx / 12 * (dp - dp[:, :1])
        Z = np.maximum(F[:, -1:], 1e-300)
        gc, go, p, F = gc / Z, go / Z, p / Z, F / Z
        # Summaries and scores
        sc["mean"][:, t] = (grid * p).sum(axis=1) * dx
        u = (x[:, t] - grid[:, 0]) / dx  # position of the true state, in cells
        j = np.clip(np.floor(u).astype(int), 0, n_pts - 2)
        th = np.clip(u - j, 0.0, 1.0)
        co = cubic(F, p, j)
        sc["pit"][:, t] = at(co, th)
        # CRPS = integral of F^2 left of the truth + (1 - F)^2 right of it. Both integrands are
        # degree-6 polynomials in each cell, so the 4-point rule is exact; the truth's cell is
        # split at the truth.
        F0, m0, m1 = F[:, :-1], p[:, :-1] * dx, p[:, 1:] * dx
        D = F[:, 1:] - F0
        c2, c3 = 3 * D - 2 * m0 - m1, m0 + m1 - 2 * D
        A, B = np.zeros_like(F0), np.zeros_like(F0)
        for uq, wq in zip(gl_u, gl_w):
            Fq = F0 + uq * (m0 + uq * (c2 + uq * c3))
            A += wq * Fq * Fq
            B += wq * (1 - Fq) * (1 - Fq)
        cA = np.concatenate([np.zeros((N, 1)), np.cumsum(A * dx, axis=1)], axis=1)
        cB = np.concatenate([np.cumsum(B[:, ::-1] * dx, axis=1)[:, ::-1], np.zeros((N, 1))], axis=1)
        left = right = 0.0
        for uq, wq in zip(gl_u, gl_w):
            left = left + wq * at(co, th * uq) ** 2
            right = right + wq * (1 - at(co, th + (1 - th) * uq)) ** 2
        left = cA[rows, j] + dx * th * left
        right = cB[rows, j + 1] + dx * (1 - th) * right
        beyond = dx * (np.maximum(-u, 0.0) + np.maximum(u - (n_pts - 1), 0.0))  # truth outside the grid
        sc["crps"][:, t] = left + right + beyond
        # 90% interval: the 5% and 95% quantiles, by bisection on the cubic in the cell where
        # the CDF crosses the level
        q = []
        for level in (0.05, 0.95):
            jq = np.clip(np.argmax(F > level, axis=1) - 1, 0, n_pts - 2)
            cq = cubic(F, p, jq)
            lo, hi = np.zeros(N), np.ones(N)
            for _ in range(52):
                mid = 0.5 * (lo + hi)
                below = at(cq, mid) < level
                lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
            q.append(offs[jq] + 0.5 * (lo + hi) * dx)
        sc["width90"][:, t] = q[1] - q[0]
        # Move the grid so it stays centred on the estimate
        shift = np.rint((sc["mean"][:, t] - centre) / dx).astype(int)
        idx = cols[None, :] + shift[:, None]
        inside = (idx >= 0) & (idx < n_pts)
        safe = np.clip(idx, 0, n_pts - 1)
        gc = np.where(inside, np.take_along_axis(gc, safe, axis=1), 0.0)
        go = np.where(inside, np.take_along_axis(go, safe, axis=1), 0.0)
        g = np.stack([gc, go])
        centre = centre + shift * dx
    return sc


def _exact_task(args):
    return exact_filter(*args)


def bayes_filter(y, x, Q, eps, lam, stay=None):
    """exact_filter, with the sequences split across the worker processes when there are any.
    Each sequence is filtered on its own, so the result is the same as one call."""
    if _POOL is None or len(y) < 2 * _WORKERS:
        return exact_filter(y, x, Q, eps, lam, stay)
    chunks = np.array_split(np.arange(len(y)), _WORKERS)
    parts = list(_POOL.map(_exact_task, [(y[i], x[i], Q, eps, lam, stay) for i in chunks]))
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


# ---------------------------------------------------------------- scores
def gaussian_crps(x, m, s):
    """CRPS of N(m, s^2) at the true value x, lower is better. This is the closed form of
    Gneiting and Raftery (2007, Section 4.2) in its negative orientation (their CRPS*)."""
    z = (x - m) / s
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return s * (z * (2 * ndtr(z) - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def gaussian_scores(x, m, P):
    s = np.sqrt(P)
    return dict(mean=m, pit=ndtr((x - m) / s), crps=gaussian_crps(x, m, s), width90=2 * Z90 * s)


def first_below(bias):
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


# ---------------------------------------------------------------- fitting
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
    the first parameter in the dict."""
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


def run_tasks(fn, tasks):
    """fn over tasks, in the worker processes when there are any; results in task order."""
    return list(_POOL.map(fn, tasks)) if _POOL is not None else [fn(task) for task in tasks]


def fit_all(y_tr, y_va, Q):
    """Free parameters of every filter, from observations only. The one-R Kalman filter's R
    maximises the likelihood on the training sequences; everything else minimises the CRPS
    of the one-step forecast of y_t on the validation sequences, over R_GRID, C_GRID and
    K_GRID. A noise or robust scale of inf switches that part off; when the noise scale is
    inf the window k is not used and is stored as None."""
    finite = [float(c) for c in C_GRID if np.isfinite(c)]
    tasks = [(y_va, Q, None, None, False)] + [(y_va, Q, k, c_in, past) for past in (False, True)
                                              for k in K_GRID for c_in in finite]
    # table[(k, noise c, robust c)] = validation forecast CRPS for every R in R_GRID, with the
    # noise from y_t itself; table_past the same with the noise from past readings only. The
    # rows without noise (k = None, noise c = inf) are the same in both, and come first in each.
    table, table_past = {}, {}
    for (_, _, k, c_in, past), scores in zip(tasks, run_tasks(_scores_over_c, tasks)):
        for c, s in zip(C_GRID, scores):
            key = (k, np.inf if c_in is None else c_in, float(c))
            if k is None:
                table[key] = table_past[key] = s
            else:
                (table_past if past else table)[key] = s
    fit = {"R_ml": float(R_GRID[np.argmax(grid_scores(y_tr, Q, "loglik"))]),
           "R_fc": float(R_GRID[np.argmin(table[(None, np.inf, np.inf)])])}
    # robust filter (no input-dependent noise): the best R for every c, then the best c
    rob = {c: s for (k, c_in, c), s in table.items() if k is None}
    fit["R_by_c"] = {c: float(R_GRID[np.argmin(s)]) for c, s in rob.items()}
    _, fit["R_rob"], fit["c_rob"] = best_over(rob)
    fit["R_prior"] = fit["R_by_c"][C_PRIOR]
    # input-dependent noise (no robust weight): tuned (R, k, c), and with c and k fixed in advance
    ind = {(k, c_in): s for (k, c_in, c), s in table.items() if c == np.inf}
    _, fit["R_in"], (fit["k_in"], fit["c_in"]) = best_over(ind)
    fit["R_in_prior"] = float(R_GRID[np.argmin(table[(K_PRIOR, C_PRIOR, np.inf)])])
    # both together: (R, k, noise c, robust c) tuned jointly; the table holds both filters above
    _, fit["R_both"], (fit["k_both"], fit["c_in_both"], fit["c_both"]) = best_over(table)
    # the same two with the noise from past readings only
    ind_past = {(k, c_in): s for (k, c_in, c), s in table_past.items() if c == np.inf}
    _, fit["R_past"], (fit["k_past"], fit["c_past"]) = best_over(ind_past)
    _, fit["R_both_past"], (fit["k_both_past"], fit["c_in_both_past"], fit["c_both_past"]) = best_over(table_past)
    for name in ("R_ml", "R_fc", "R_rob", "R_prior", "R_in", "R_in_prior", "R_both", "R_past", "R_both_past"):
        if fit[name] in (R_GRID[0], R_GRID[-1]):
            log(f"  warning: {name} = {fit[name]:.3g} is at the edge of the search grid")
    return fit


def filter_specs(fit, y):
    """The Gaussian filters of experiment 1 with their fitted parameters, for data y."""
    def noise(R_, k, c, past=False):  # R_t = R + d_t^2 / c^2 (d_{t-1} if past); just R when c = inf
        if not np.isfinite(c):
            return np.full(y.shape, R_)
        return R_ + (past_distance2 if past else local_distance2)(y, k) / c ** 2

    return {
        ONE_TRUE: dict(R=R),
        ONE_ML: dict(R=fit["R_ml"]),
        ONE_FC: dict(R=fit["R_fc"]),
        INDEP: dict(R=fit["R_in"], R_t=noise(fit["R_in"], fit["k_in"], fit["c_in"])),
        INDEP_FIX: dict(R=fit["R_in_prior"], R_t=noise(fit["R_in_prior"], K_PRIOR, C_PRIOR)),
        INDEP_PAST: dict(R=fit["R_past"], R_t=noise(fit["R_past"], fit["k_past"], fit["c_past"], past=True)),
        ROB: dict(R=fit["R_rob"], c=fit["c_rob"]),
        ROB_FIX: dict(R=fit["R_prior"], c=C_PRIOR),
        BOTH: dict(R=fit["R_both"], c=fit["c_both"], R_t=noise(fit["R_both"], fit["k_both"], fit["c_in_both"])),
        BOTH_PAST: dict(R=fit["R_both_past"], c=fit["c_both_past"],
                        R_t=noise(fit["R_both_past"], fit["k_both_past"], fit["c_in_both_past"], past=True)),
    }


# ---------------------------------------------------------------- self-checks
def self_checks(out_dir):
    """Checks of the code against independent calculations or hand-made cases with known
    answers. Each writes PASS or FAIL; any FAIL stops the run before results are written.
    The two checks that compare worker processes with one process are written as SKIP when
    there are no worker processes (--workers 1)."""
    global _POOL, _WORKERS  # checks 18-20 run the pool code, with real or stand-in workers
    lines, ok, skipped = [], True, []

    def check(name, passed, value):
        nonlocal ok
        ok &= bool(passed)
        try:
            vs = f"{float(value):.3g}"
        except (TypeError, ValueError):
            vs = str(value)
        lines.append(f"{'PASS' if passed else 'FAIL'}  {name}  ({vs})")
        log(lines[-1])

    def skip(name, reason):
        lines.append(f"SKIP  {name}  ({reason})")
        skipped.append(name)
        log(lines[-1])

    @contextlib.contextmanager
    def quiet():  # no log lines or printing from the code under test
        global _LOG
        saved = _LOG
        _LOG = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            _LOG = saved

    class InProcessPool:  # stands in for the worker processes: map runs the tasks here, in order
        def map(self, fn, tasks):
            return [fn(task) for task in list(tasks)]

    @contextlib.contextmanager
    def pool_as(p, n):  # run the code under test with this pool (None: no pool) and n workers
        global _POOL, _WORKERS
        saved = _POOL, _WORKERS
        _POOL, _WORKERS = p, n
        try:
            yield
        finally:
            _POOL, _WORKERS = saved

    @contextlib.contextmanager
    def replaced(name, fn):  # a module-level function replaced for the duration
        g = globals()
        saved = g[name]
        g[name] = fn
        try:
            yield
        finally:
            g[name] = saved

    def crps_ref(xv, mv, sv):  # written out again, not the module's function
        z = (xv - mv) / sv
        return sv * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))

    def d2_ref(yv, k):  # direct loop, prior mean standing in for missing readings
        out = np.zeros_like(yv)
        for i in range(yv.shape[0]):
            for t in range(yv.shape[1]):
                window = [M0] * max(0, k - t) + list(yv[i, max(0, t - k):t])
                out[i, t] = (yv[i, t] - np.median(window)) ** 2
        return out

    def past_ref(yv, k):  # direct loop: y_{t-1} against the median of the k readings before it; 0 at t = 1
        out = np.zeros_like(yv)
        for i in range(yv.shape[0]):
            for t in range(1, yv.shape[1]):
                s = t - 1
                window = [M0] * max(0, k - s) + list(yv[i, max(0, s - k):s])
                out[i, t] = (yv[i, s] - np.median(window)) ** 2
        return out

    def scalar_filter(yv, Rv, Qs, a, c=None, R_t=None, w=None):  # information form, written again
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

    rng = np.random.default_rng(1)

    # 1. The weight itself
    w = imq_weight(np.array(1.0), np.array(1.0), 2.0)
    check("IMQ weight at a residual of one predictive sd with c = 2 is 1/sqrt(1.25)",
          abs(w - 1 / np.sqrt(1.25)) < 1e-15, abs(w - 1 / np.sqrt(1.25)))

    # 2. Kalman filter with time-varying a_t, Q_t against exact Gaussian conditioning; the
    #    forecast CRPS and log-likelihood against the exact one-step predictive of y_t
    Tn, n, Rv = 25, 3, 0.7
    a = rng.uniform(0.3, 1.0, size=(n, Tn))
    Qs = rng.uniform(0.05, 0.5, size=(n, Tn))
    xs = np.empty((n, Tn))
    prev = rng.normal(M0, np.sqrt(P0), size=n)
    for t in range(Tn):
        prev = a[:, t] * prev + np.sqrt(Qs[:, t]) * rng.normal(size=n)
        xs[:, t] = prev
    ys = xs + np.sqrt(Rv) * rng.normal(size=(n, Tn))
    ys[:, [4, 11]] += 8.0
    Rt = Rv + rng.uniform(0.0, 5.0, size=(n, Tn))
    for label, R_t in (("one R", None), ("R_t in the update", Rt)):
        kf = kalman_filter(ys, Rv, Qs, a, R_t=R_t, score=True)
        dm = dP = dc = dl = 0.0
        for i in range(n):
            A = np.zeros((Tn, Tn + 1))  # x = A [x_0, q_1, ..., q_T]
            for t in range(Tn):
                A[t, 0] = np.prod(a[i, :t + 1])
                for jj in range(1, t + 2):
                    A[t, jj] = np.prod(a[i, jj:t + 1])
            Cx = A @ np.diag(np.concatenate([[P0], Qs[i]])) @ A.T
            mx = A[:, 0] * M0
            Cy = Cx + np.diag(np.full(Tn, Rv) if R_t is None else R_t[i])
            crps = ll = 0.0
            for t in range(Tn):
                cxy = Cx[t, :t + 1]
                sol = np.linalg.solve(Cy[:t + 1, :t + 1], ys[i, :t + 1] - mx[:t + 1])
                dm = max(dm, abs(mx[t] + cxy @ sol - kf["m"][i, t]))
                dP = max(dP, abs(Cx[t, t] - cxy @ np.linalg.solve(Cy[:t + 1, :t + 1], cxy) - kf["P"][i, t]))
                if t == 0:
                    mu, Pp = mx[0], Cx[0, 0]
                else:
                    cp = Cx[t, :t]
                    mu = mx[t] + cp @ np.linalg.solve(Cy[:t, :t], ys[i, :t] - mx[:t])
                    Pp = Cx[t, t] - cp @ np.linalg.solve(Cy[:t, :t], cp)
                crps += crps_ref(ys[i, t], mu, np.sqrt(Pp + Rv)) / Tn
                ll += normal_logpdf(ys[i, t], mu, Pp + Rv)
            dc = max(dc, abs(crps - kf["fc_crps"][i]))
            dl = max(dl, abs(ll - kf["loglik"][i]))
        check(f"Kalman means equal exact Gaussian conditioning, time-varying a_t, Q_t ({label})", dm < 1e-10, dm)
        check(f"Kalman variances equal exact Gaussian conditioning ({label})", dP < 1e-10, dP)
        check(f"forecast CRPS equals the exact one-step predictive with base R ({label})", dc < 1e-10, dc)
        check(f"forecast log-likelihood equals the exact one-step predictive ({label})", dl < 1e-8, dl)

    # 3. Robust filter against an independently written information-form update, with one
    #    R and with input-dependent R_t (the weight measured against P_pred + R_t)
    for label, R_t in (("one R", None), ("R_t", Rt)):
        rob = kalman_filter(ys, Rv, Qs, a, c=1.5, R_t=R_t)
        ref = scalar_filter(ys, Rv, Qs, a, c=1.5, R_t=R_t)
        d = max(np.abs(rob[k] - ref[k]).max() for k in ("m", "P", "w"))
        check(f"robust filter equals an information-form robust update ({label})", d < 1e-12, d)

    # 4. Limits of the robust filter, and the steady state of the Kalman filter
    x, y, _ = simulate_rw(rng, 5, 300, 0.1, EPS, LAM, None, None)
    kf = kalman_filter(y, R, 0.1)
    d = np.abs(kalman_filter(y, R, 0.1, c=np.inf)["m"] - kf["m"]).max()
    check("robust filter with c = inf is the Kalman filter", d == 0, d)
    d = np.abs(kalman_filter(y, R, 0.1, c=1e9)["m"] - kf["m"]).max()
    check("robust filter with c = 1e9 is within 1e-9 of the Kalman filter", d < 1e-9, d)
    d = abs(kf["P_pred"][0, -1] - (0.1 + np.sqrt(0.01 + 0.4 * R)) / 2)
    check("Kalman predictive variance reaches the steady-state Riccati value", d < 1e-10, d)

    # 5. Grid Bayes filter against exact enumeration of every outlier pattern: the posterior is
    #    then a Gaussian mixture, whose mean, PIT, CRPS and 90% width are computed exactly
    def absmean(mu, var):  # E|Z| for Z ~ N(mu, var)
        sd = np.sqrt(var)
        return sd * np.sqrt(2 / np.pi) * np.exp(-0.5 * mu * mu / var) + mu * (2 * ndtr(mu / sd) - 1)

    def mixture_scores(wts, ms, Ps, xv):
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

    Tn, n, Qv, lam_c = 10, 3, 0.1, 100.0
    pats = ((np.arange(2 ** Tn)[:, None] >> np.arange(Tn)) & 1).astype(bool)
    tols5 = dict(mean=1e-9, pit=2e-7, crps=2e-7, width90=2e-7)
    for kind, eps_c, stay in (("independent outliers", 0.3, None), ("bursty outliers", 0.3, 0.8),
                              ("a real jump of 5 at step 6", 0.1, None)):
        if kind.startswith("a real jump"):
            x, y, _ = simulate_rw(np.random.default_rng(31), n, Tn, Qv, eps_c, lam_c, stay, 5)
        else:
            x, y, _ = simulate_rw(rng, n, Tn, Qv, eps_c, lam_c, stay, None)
            y[:, 4] += 15.0
        p_stay, p_start = (eps_c, eps_c) if stay is None else (stay, eps_c * (1 - stay) / (1 - eps_c))
        lp = np.where(pats[:, 0], np.log(eps_c), np.log1p(-eps_c))
        p_now = np.where(pats[:, :-1], p_stay, p_start)
        lp = lp + np.where(pats[:, 1:], np.log(p_now), np.log1p(-p_now)).sum(axis=1)
        ex = exact_filter(y, x, Qv, eps_c, lam_c, stay)
        worst = dict.fromkeys(tols5, 0.0)
        for i in range(n):
            Rts = np.where(pats, lam_c * R, R)
            out = kalman_filter(np.tile(y[i], (len(pats), 1)), R, Qv, R_t=Rts)
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
                for key in tols5:
                    worst[key] = max(worst[key], abs(mix[key] - ex[key][i, t]))
        for key, tol in tols5.items():
            check(f"grid Bayes filter equals exact enumeration of outlier patterns ({kind}): {key}",
                  worst[key] < tol, worst[key])

    # 6. Grid Bayes filter equals the Kalman filter on clean data, including after a jump,
    #    and at both process-noise levels used
    for Qv in (0.1, 1.0):
        x, y, _ = simulate_rw(rng, 5, 300, Qv, 0.0, LAM, None, 150)
        kf = kalman_filter(y, R, Qv)
        ex = exact_filter(y, x, Qv, 0.0, LAM)
        gs = gaussian_scores(x, kf["m"], kf["P"])
        for key, tol in (("mean", 1e-9), ("pit", 2e-7), ("crps", 2e-7), ("width90", 2e-7)):
            d = np.abs(ex[key] - gs[key]).max()
            check(f"grid Bayes filter equals the Kalman filter on clean data with a jump, Q = {Qv:g}: {key}",
                  d < tol, d)
    # ... and the grid is wide enough: widening it to +-40 changes no score, with outliers
    for seed, Qv, stay in ((151, 0.1, None), (152, 0.1, 0.8), (153, 1.0, None)):
        x, y, _ = simulate_rw(np.random.default_rng(seed), 20, 300, Qv, EPS, LAM, stay, 150)
        narrow, wide = exact_filter(y, x, Qv, EPS, LAM, stay), exact_filter(y, x, Qv, EPS, LAM, stay, half_width=40.0)
        d = max(np.abs(narrow[key] - wide[key]).max() for key in narrow)
        kind = "independent outliers" if stay is None else "bursts"
        check(f"grid Bayes filter: widening the grid to +-40 changes no score by 1e-9 (Q = {Qv:g}, {kind})",
              d < 1e-9, d)

    # 7. Closed-form Gaussian CRPS against Monte Carlo
    s, mv, xv = 1.5, 0.2, 0.9
    draws = np.random.default_rng(2).normal(mv, s, size=(2, 2_000_000))
    mc = np.abs(draws[0] - xv).mean() - 0.5 * np.abs(draws[0] - draws[1]).mean()
    d = abs(mc - gaussian_crps(xv, mv, s))
    check("closed-form Gaussian CRPS matches Monte Carlo", d < 5e-3, d)

    # 8. Simulators
    x, y, o = simulate_rw(np.random.default_rng(3), 400, 1000, 0.1, 0.1, 100.0, None, 500)
    x2, _, _ = simulate_rw(np.random.default_rng(3), 400, 1000, 0.1, 0.1, 100.0, None, None)
    check("outlier rate is 10%", abs(o.mean() - 0.1) < 0.003, o.mean())
    check("clean noise variance is R", abs((y - x)[~o].var() / R - 1) < 0.01, (y - x)[~o].var())
    check("outlier noise variance is 100 R", abs((y - x)[o].var() / (100 * R) - 1) < 0.03, (y - x)[o].var())
    jump_err = max(np.abs((x - x2)[:, 500:] - JUMP).max(), np.abs((x - x2)[:, :500]).max())
    check("the jump is +5 from column 500 (step 501) on and absent before", jump_err < 1e-12, jump_err)
    q_var = np.diff(x2, axis=1).var()
    check("random-walk step variance is Q", abs(q_var / 0.1 - 1) < 0.01, q_var)
    o = outlier_mask(np.random.default_rng(4), 400, 2000, 0.1, 0.8)
    runs = np.diff(np.concatenate([np.zeros((400, 1), bool), o, np.zeros((400, 1), bool)], axis=1).astype(int), axis=1)
    check("bursty outliers: rate is 10%", abs(o.mean() - 0.1) < 0.005, o.mean())
    check("bursty outliers: mean run length is 5", abs(o.sum() / (runs == 1).sum() - 5) < 0.2, o.sum() / (runs == 1).sum())
    r4 = outlier_mask(np.random.default_rng(5), 200_000, 4, 0.1, 0.8).mean(axis=0)
    check("bursty outliers: rate is 10% at each of the first four steps", np.abs(r4 - 0.1).max() < 0.004,
          np.abs(r4 - 0.1).max())
    x, y, a, Qs = simulate_selective(np.random.default_rng(6), 50, 4000, 0.9)
    check("decaying state keeps variance SIGMA2", abs(x[:, 100:].var() / SIGMA2 - 1) < 0.1, x[:, 100:].var())
    check("median decay is a_med", abs(np.median(a) - 0.9) < 0.02, np.median(a))
    u = np.log(np.log(a) / np.log(0.9))
    check("the input u_t is stationary with sd 0.75", abs(u.std() - 0.75) < 0.05, u.std())
    want = (1 - EPS) * R2 + EPS * LAM * R2
    check("decaying-state observation noise variance is (1 - eps) R2 + eps LAM R2",
          abs((y - x).var() / want - 1) < 0.1, (y - x).var())
    _, _, a, _ = simulate_selective(np.random.default_rng(7), 20, 2000, 0.3, cap=0.6)
    check("capped decay never exceeds the cap", a.max() <= 0.6, a.max())

    # 9. Metrics, on hand-made cases with known answers
    xs = np.random.default_rng(8).normal(size=(200, 1000))
    idx = np.random.default_rng(9).integers(0, 200, size=(200, 200))
    summ, _, _ = summarise(xs, gaussian_scores(xs, np.zeros_like(xs), np.ones_like(xs)), 500, idx)
    check("calibrated Gaussian: 90% coverage", abs(summ["cov90"][0] - 0.9) < 0.005, summ["cov90"][0])
    check("calibrated Gaussian: 50% coverage", abs(summ["cov50"][0] - 0.5) < 0.006, summ["cov50"][0])
    check("calibrated Gaussian: RMSE is the sd", abs(summ["rmse"][0] - 1) < 0.01, summ["rmse"][0])
    nn, TT, jt = 4, 300, 150
    idx4 = np.random.default_rng(10).integers(0, nn, size=(50, nn))
    ones, xz = np.ones((nn, TT)), np.zeros((nn, TT))
    m = np.zeros((nn, TT))
    m[:, jt:] = -3.0
    summ, _, _ = summarise(xz, gaussian_scores(xz, m, ones), jt, idx4)
    good = (summ["rmse"][0] == 0 and abs(summ["crps"][0] - (np.sqrt(2) - 1) / np.sqrt(np.pi)) < 1e-12
            and summ["cov90"][0] == 1 and abs(summ["width90"][0] - 2 * Z90) < 1e-12)
    check("summarise: accuracy and calibration use only the steps before the jump", good, summ["rmse"][0])
    m = np.zeros((nn, TT))
    m[:, jt - 1], m[:, jt], m[:, jt + AFTER - 1], m[:, jt + AFTER] = 100.0, 7.0, 1.0, 100.0
    summ, _, _ = summarise(xz, gaussian_scores(xz, m, ones), jt, idx4)
    check("summarise: post-jump RMSE covers exactly the AFTER steps from the jump on",
          abs(summ["rmse_after"][0] - np.sqrt(50 / AFTER)) < 1e-12, summ["rmse_after"][0])
    m2 = np.zeros((2, TT))
    lagged = JUMP * 0.7 ** np.arange(TT - jt)
    m2[0, jt:], m2[1, jt:] = -(lagged + 2.0), -(lagged - 2.0)
    summ, _, _ = summarise(np.zeros((2, TT)), gaussian_scores(np.zeros((2, TT)), m2, np.ones((2, TT))), jt,
                           np.array([[0, 1]]))
    check("summarise: delay from the mean signed error (+-2 offsets cancel; 5 * 0.7^s gives 5)",
          summ["delay"][0] == 5, summ["delay"][0])
    m = np.zeros((nn, TT))
    m[:, :jt] = np.arange(1, nn + 1)[:, None]
    summ, _, _ = summarise(xz, gaussian_scores(xz, m, ones), jt, idx4)
    want = 1.96 * np.std(np.arange(1, nn + 1), ddof=1) / np.sqrt(nn)
    check("summarise: interval half-width is 1.96 sd / sqrt(n)", abs(summ["rmse"][1] - want) < 1e-12, summ["rmse"][1])

    # 10. Input-dependent noise: local distances against a direct loop
    yv = np.random.default_rng(11).normal(size=(3, 40)) * 3.0
    d = max(np.abs(local_distance2(yv, k) - d2_ref(yv, k)).max() for k in K_GRID)
    check("local_distance2 equals a direct loop (prior mean before the first readings)", d < 1e-12, d)
    d = max(np.abs(past_distance2(yv, k) - past_ref(yv, k)).max() for k in K_GRID)
    check("past_distance2 equals a direct loop on y_{t-1} (0 at the first step), so it never uses y_t", d < 1e-12, d)
    d = 0.0
    for t in (0, 5, 17, 39):  # change y_t by 100: nothing past_distance2 gives at steps up to t may move
        y_changed = yv.copy()
        y_changed[:, t] += 100.0
        d = max(d, max(np.abs(past_distance2(y_changed, k)[:, :t + 1] - past_distance2(yv, k)[:, :t + 1]).max()
                       for k in K_GRID))
    check("past_distance2 up to step t does not depend on y_t (y_t changed by 100, four steps t)", d == 0, d)

    # 11. Tuning: brute force over every grid at once, written independently of fit_all, on
    #     three data sets. On seed 104 the training and validation fits differ; on seed 231 the
    #     filters' scales and windows differ, so mixing them up is caught; seed 302 has bursts,
    #     so the noise from past readings is used. (With 3 short sequences several filters'
    #     best R land on the same grid value; check 12 tells the R values apart with fabricated
    #     fits, and check 19 puts the best entry on every edge of the search.)
    nc, inf_c = len(C_GRID), len(C_GRID) - 1

    def ic(c):
        return int(np.flatnonzero(C_GRID == c)[0])

    def iR(Rv):
        return int(np.argmin(np.abs(R_GRID - Rv)))

    def ik(k):  # the window does not matter when the noise c is inf; use the first row then
        return 0 if k is None else K_GRID.index(k)

    for seed, stay in ((104, None), (231, None), (302, 0.8)):
        rng_t = np.random.default_rng(seed)
        _, y_tr, _ = simulate_rw(rng_t, 3, 150, 0.1, 0.1, 100.0, stay, None)
        _, y_va, _ = simulate_rw(rng_t, 3, 150, 0.1, 0.1, 100.0, stay, None)
        fit = fit_all(y_tr, y_va, 0.1)
        nv = len(y_va)
        ll = np.array([kalman_filter(y_tr, Rv, 0.1, store=False, score=True)["loglik"].mean() for Rv in R_GRID])
        # S[k, noise c, robust c, R]: validation forecast CRPS; one filter call per (k, R) with every
        # (noise c, robust c) pair as its own rows; a noise c of inf leaves R_t = R. Sp: the same with
        # the noise from past readings.
        cin_rows, c_rows = np.repeat(C_GRID, nc * nv), np.tile(np.repeat(C_GRID, nv), nc)
        S, Sp = (np.empty((len(K_GRID), nc, nc, len(R_GRID))) for _ in range(2))
        for arr, dist in ((S, d2_ref), (Sp, past_ref)):
            for i, k in enumerate(K_GRID):
                extra = np.tile(dist(y_va, k), (nc * nc, 1)) / cin_rows[:, None] ** 2
                for j, Rv in enumerate(R_GRID):
                    out = kalman_filter(np.tile(y_va, (nc * nc, 1)), Rv, 0.1, c=c_rows, R_t=Rv + extra, store=False,
                                        score=True)
                    arr[i, :, :, j] = out["fc_crps"].reshape(nc, nc, nv).mean(axis=2)
        tag = f"(seed {seed})"
        check(f"fit_all: the one-R maximum-likelihood R maximises the training log-likelihood {tag}",
              ll[iR(fit["R_ml"])] >= ll.max() - 1e-9 * abs(ll.max()), fit["R_ml"])
        plain = S[0, inf_c, inf_c]
        check(f"fit_all: the one-R forecast-CRPS R minimises the forecast CRPS {tag}",
              plain[iR(fit["R_fc"])] <= plain.min() + 1e-12, fit["R_fc"])
        rob = S[0, inf_c]  # (robust c, R) without input-dependent noise
        check(f"fit_all: the robust (R, c) minimise the forecast CRPS {tag}",
              rob[ic(fit["c_rob"]), iR(fit["R_rob"])] <= rob.min() + 1e-12, fit["c_rob"])
        bad = sum(rob[i, iR(fit["R_by_c"][float(c)])] > rob[i].min() + 1e-12 for i, c in enumerate(C_GRID))
        check(f"fit_all: the R kept for every c is the best R for that c {tag}", bad == 0, bad)
        check(f"fit_all: the c = {C_PRIOR:g} robust filter gets the best R for that c {tag}",
              fit["R_prior"] == fit["R_by_c"][C_PRIOR]
              and rob[ic(C_PRIOR), iR(fit["R_prior"])] <= rob[ic(C_PRIOR)].min() + 1e-12, fit["R_prior"])
        ind = S[:, :, inf_c]  # (k, noise c, R) without the robust weight
        v = ind[ik(fit["k_in"]), ic(fit["c_in"]), iR(fit["R_in"])]
        check(f"fit_all: the input-dependent (R, c, k) minimise the forecast CRPS {tag}", v <= ind.min() + 1e-12,
              v - ind.min())
        fix = ind[K_GRID.index(K_PRIOR), ic(C_PRIOR)]
        check(f"fit_all: the input-dependent filter with c, k fixed gets its best R {tag}",
              fix[iR(fit["R_in_prior"])] <= fix.min() + 1e-12, fit["R_in_prior"])
        v = S[ik(fit["k_both"]), ic(fit["c_in_both"]), ic(fit["c_both"]), iR(fit["R_both"])]
        check(f"fit_all: the robust + input-dependent (R, k, noise c, robust c) minimise the forecast CRPS over the "
              f"whole grid {tag}", v <= S.min() + 1e-12, v - S.min())
        ind_p = Sp[:, :, inf_c]  # (k, noise c, R): noise from past readings, no robust weight
        v = ind_p[ik(fit["k_past"]), ic(fit["c_past"]), iR(fit["R_past"])]
        check(f"fit_all: the past-only input-dependent (R, c, k) minimise the forecast CRPS {tag}",
              v <= ind_p.min() + 1e-12, v - ind_p.min())
        v = Sp[ik(fit["k_both_past"]), ic(fit["c_in_both_past"]), ic(fit["c_both_past"]), iR(fit["R_both_past"])]
        check(f"fit_all: the past-only robust + input-dependent (R, k, noise c, robust c) minimise the forecast CRPS "
              f"over the whole grid {tag}", v <= Sp.min() + 1e-12, v - Sp.min())
        good = all((fit[k] is None) == (not np.isfinite(fit[c])) for k, c in
                   (("k_in", "c_in"), ("k_both", "c_in_both"), ("k_past", "c_past"), ("k_both_past", "c_in_both_past")))
        check(f"fit_all: k is stored as None exactly when the noise c is inf {tag}", good, float(good))
        if seed == 231:
            apart = (fit["k_in"] not in (None, K_PRIOR) and fit["c_in"] not in (C_PRIOR, np.inf)
                     and fit["R_in"] != fit["R_in_prior"] and fit["R_by_c"][C_PRIOR] != fit["R_by_c"][2.25]
                     and (fit["k_both"], fit["c_in_both"]) != (fit["k_in"], fit["c_in"])
                     and np.isfinite(fit["c_both"]) and np.isfinite(fit["c_in_both"]) and fit["c_both"] != fit["c_rob"])
            check(f"the data of seed 231 tell the fits apart, so the checks above are not vacuous {tag}", apart,
                  float(apart))
        if seed == 302:  # the past-only noise is used here, alone and with the weight, so its checks are not vacuous
            apart = (np.isfinite(fit["c_past"]) and np.isfinite(fit["c_in_both_past"])
                     and np.isfinite(fit["c_both_past"]) and (fit["k_past"], fit["c_past"]) != (fit["k_in"], fit["c_in"])
                     and (fit["k_both_past"], fit["c_in_both_past"]) != (fit["k_past"], fit["c_past"])
                     and (fit["k_both_past"], fit["c_in_both_past"], fit["c_both_past"])
                     != (fit["k_both"], fit["c_in_both"], fit["c_both"]))
            check(f"with bursts the past-only noise is used alone and with the weight, with its own fit {tag}", apart,
                  float(apart))
    _, y_clean, _ = simulate_rw(np.random.default_rng(12), 20, 500, 0.1, 0.0, LAM, None, None)
    R_ml = R_GRID[np.argmax(grid_scores(y_clean, 0.1, "loglik"))]
    check("maximum likelihood on clean data learns R = 1 (within one grid step)",
          abs(np.log10(R_ml / R)) <= 0.05 + 1e-12, R_ml)

    # 12. Wiring: every filter gets its own fitted parameters; the paired lines and the .tex
    #     tables show the right numbers in the right places (fabricated results in which every
    #     value says which filter and which metric it belongs to)
    fake = dict(R_ml=1.1, R_fc=2.2, R_rob=3.3, c_rob=4.4, R_prior=5.5, R_in=6.6, c_in=7.7, k_in=3,
                R_in_prior=8.8, R_both=9.9, c_both=1.25, k_both=9, c_in_both=0.75,
                R_past=7.1, c_past=1.9, k_past=7, R_both_past=8.1, c_both_past=2.9, c_in_both_past=1.1, k_both_past=4)
    assert len({fake["k_in"], fake["k_both"], fake["k_past"], fake["k_both_past"], K_PRIOR}) == 5  # all windows differ
    yv = np.random.default_rng(13).normal(size=(2, 30))
    sp = filter_specs(fake, yv)

    def close(a, b):
        return a is not None and np.allclose(a, b, rtol=0, atol=1e-12)

    good = (sp[ONE_TRUE] == dict(R=R) and sp[ONE_ML] == dict(R=1.1) and sp[ONE_FC] == dict(R=2.2)
            and sp[ROB] == dict(R=3.3, c=4.4) and sp[ROB_FIX] == dict(R=5.5, c=C_PRIOR)
            and sp[INDEP]["R"] == 6.6 and close(sp[INDEP].get("R_t"), 6.6 + d2_ref(yv, 3) / 7.7 ** 2)
            and sp[INDEP_FIX]["R"] == 8.8 and close(sp[INDEP_FIX].get("R_t"), 8.8 + d2_ref(yv, K_PRIOR) / C_PRIOR ** 2)
            and sp[BOTH]["R"] == 9.9 and sp[BOTH].get("c") == 1.25
            and close(sp[BOTH].get("R_t"), 9.9 + d2_ref(yv, 9) / 0.75 ** 2)
            and sp[INDEP_PAST]["R"] == 7.1 and "c" not in sp[INDEP_PAST]
            and close(sp[INDEP_PAST].get("R_t"), 7.1 + past_ref(yv, 7) / 1.9 ** 2)
            and sp[BOTH_PAST]["R"] == 8.1 and sp[BOTH_PAST].get("c") == 2.9
            and close(sp[BOTH_PAST].get("R_t"), 8.1 + past_ref(yv, 4) / 1.1 ** 2)
            and set(sp) == set(FILTERS) - {BAYES})
    check("filter_specs gives each filter its own fitted parameters, each noise from its own window k (the past-only "
          "ones from y_{t-1})", good, float(good))
    sp = filter_specs(dict(fake, k_in=None, c_in=np.inf, k_past=None, c_past=np.inf), yv)
    good = (np.array_equal(sp[INDEP]["R_t"], np.full(yv.shape, 6.6))
            and np.array_equal(sp[INDEP_PAST]["R_t"], np.full(yv.shape, 7.1)))
    check("filter_specs: a noise c of inf gives R_t = R exactly", good, float(good))
    n4, spread = 4, np.arange(4.0)
    res = dict(name="test", fitted=dict(fake, R_by_c={}), metrics={}, per_seq={}, boots={})
    for i, lab in enumerate(FILTERS):
        ps = dict(rmse=0.1 * (i + 1) + 0.0013 * (i + 1) * spread, crps=0.5 + 0.017 * (i + 1) + 0.0021 * (i + 1) * spread,
                  rmse_after=1.0 + 0.071 * (i + 1) + 0.0037 * (i + 1) * spread,
                  cov90=0.3031 + i / 1000 + 0 * spread, cov50=0.6061 + i / 1000 + 0 * spread,
                  width90=4.041 + i / 100 + 0 * spread)
        res["per_seq"][lab] = ps
        res["metrics"][lab] = {key: (float(v.mean()), 0.0) for key, v in ps.items()}
        res["metrics"][lab]["delay"] = (float(20 + 3 * i), 0.0, 0.0)
        res["boots"][lab] = 10 + i + (np.arange(200) % (i + 2))
    text = scenario_text(res)
    want = [(ROB, ONE_ML), (ROB, ONE_FC), (ROB, INDEP), (BOTH, INDEP), (BOTH_PAST, INDEP_PAST), (ROB_FIX, INDEP_FIX),
            (ROB, BAYES)]
    got = re.findall(r"^  (.+?) vs (.+?): ", text, flags=re.M)
    check("scenario_text: the paired lines are the seven documented comparisons, in order", got == want, len(got))

    def sign4(v):  # +/- and 4 decimals, with no "-0.0000"
        return "+0.0000" if abs(v) < 5e-5 else f"{v:+.4f}"

    bad = 0
    for f1, f2 in want:
        parts = []
        for key, name in (("rmse", "RMSE"), ("crps", "CRPS"), ("rmse_after", "RMSE after the jump")):
            dv = res["per_seq"][f1][key] - res["per_seq"][f2][key]
            parts.append(f"{name} {sign4(dv.mean())} ±{1.96 * dv.std(ddof=1) / np.sqrt(n4):.4f}")
        db = res["boots"][f1] - res["boots"][f2]
        parts.append(f"delay {res['metrics'][f1]['delay'][0] - res['metrics'][f2]['delay'][0]:+.0f} "
                     f"[{np.percentile(db, 2.5):+.0f}, {np.percentile(db, 97.5):+.0f}]")
        bad += f"  {f1} vs {f2}: " + ", ".join(parts) not in text.splitlines()
    check("scenario_text: each paired line is first minus second (mean ± 1.96 sd / sqrt(n)) for RMSE, CRPS and "
          "RMSE after the jump, and the delay with its paired-bootstrap interval", bad == 0, bad)
    check(f"scenario_text: the post-jump column names its window of {AFTER} steps", f"RMSE ({AFTER} steps)" in text,
          AFTER)
    good = (signed(-1e-9) == "+0.0000" and signed(-0.00006) == "-0.0001" and signed(0.0) == "+0.0000"
            and signed(-0.04, 1) == "+0.0" and signed(-0.06, 1) == "-0.1")
    check("signed: a value that rounds to zero prints as +0.0000, any other keeps its sign", good, float(good))
    # exp1_table.tex: every filter in the documented order, each row with its own label and metrics
    order = [ONE_TRUE, ONE_ML, ONE_FC, INDEP, INDEP_FIX, INDEP_PAST, ROB, ROB_FIX, BOTH, BOTH_PAST, BAYES]
    words = {ONE_TRUE: "$R = 1$", ONE_ML: "likelihood", ONE_FC: "forecast", INDEP: "Kalman, input-dep.\\ $R_t$, tuned",
             INDEP_FIX: "Kalman, input-dep.\\ $R_t$, $c", INDEP_PAST: "Kalman, input-dep.\\ $R_t$ (past), tuned",
             ROB: "Robust, tuned", ROB_FIX: "Robust, $c", BOTH: "Robust + input-dep.\\ $R_t$, tuned",
             BOTH_PAST: "Robust + input-dep.\\ $R_t$ (past), tuned", BAYES: "Bayes"}
    body = [ln for ln in table_exp1_tex(res).split(r"\midrule", 1)[1].splitlines() if ln.endswith(r"\\")]
    bad = len(body) != len(order)
    for ln, lab in zip(body, order):
        cells = [c.strip() for c in ln[:-2].split("&")]
        mt = res["metrics"][lab]
        bad += words[lab] not in cells[0] or sum(words[other] in cells[0] for other in order) != 1
        bad += cells[1:] != [f"{mt['rmse'][0]:.3f}", f"{mt['crps'][0]:.3f}", f"{100 * mt['cov90'][0]:.1f}",
                             f"{mt['width90'][0]:.2f}", f"{mt['delay'][0]:.0f}", f"{mt['rmse_after'][0]:.2f}"]
    check("exp1_table.tex: every filter in order, with its label, RMSE, CRPS, 90% cover, 90% width, delay and "
          "post-jump RMSE", bad == 0, bad)
    # sweep tables: the documented columns, and row labels that describe their scenarios
    cols = [ONE_ML, INDEP, ROB, BOTH, INDEP_FIX, ROB_FIX, BAYES]
    bad = 0
    for key, fmt in (("rmse", lambda v: f"{v:.3f}"), ("delay", lambda v: f"{v:.0f}")):
        st = sweep_tex([(k, res) for k in range(len(SCENARIOS))], key, fmt, "TITLE")
        bad += r"Setting & one $R$ & input-dep. & robust & both & input-dep. & robust & Bayes \\" not in st
        bad += not any(ln.startswith("TITLE & ") for ln in st.splitlines())
        rows_ = [ln for ln in st.split(r"\midrule", 1)[1].splitlines() if ln.endswith(r"\\")]
        bad += len(rows_) != len(SCENARIOS)
        for ln in rows_:
            bad += [c.strip() for c in ln[:-2].split("&")][1:] != [fmt(res["metrics"][f][key][0]) for f in cols]
    for label, (name, Qk, (ef, lf, sf), (et, lt, st)) in zip(SHORT_TEX, SCENARIOS):
        if et == 0:
            bad += not label.startswith("none")
            continue
        pct, sd = re.match(r"(\d+)\\%", label), re.search(r"sd (\d+)", label)
        bad += pct is None or int(pct.group(1)) != round(100 * et) or sd is None or int(sd.group(1)) != round(np.sqrt(lt))
        bad += ("test time only" in label) != (ef == 0 < et)
        bad += ("bursts" in label) != (st is not None)
        bad += ("fast state" in label) != (Qk != 0.1)
    check("sweep tables: the documented columns, and every row label describes its scenario", bad == 0, bad)
    # sweep_weight_table.tex: per setting, the paired RMSE change from adding the robust weight to each
    # kind of input-dependent noise, with its interval and as a share of the noise-alone RMSE
    wt = weight_tex([(k, res) for k in range(len(SCENARIOS))])
    rows_ = [ln for ln in wt.split(r"\midrule", 1)[1].splitlines() if ln.endswith(r"\\")]
    bad = (r"\begin{tabular}{@{}lrr@{}}" not in wt
           or r"Change in RMSE & \multicolumn{2}{c}{Input-dependent $R_t$ computed from} \\" not in wt.splitlines()
           or r"Setting & $y_1, \dots, y_t$ & $y_1, \dots, y_{t-1}$ \\" not in wt.splitlines()
           or len(rows_) != len(SCENARIOS))
    for ln, label in zip(rows_, SHORT_TEX):
        cells = [c.strip() for c in ln[:-2].split("&")]
        expect = [label]
        for f1, f2 in ((BOTH, INDEP), (BOTH_PAST, INDEP_PAST)):
            dv = res["per_seq"][f1]["rmse"] - res["per_seq"][f2]["rmse"]
            pct = 100 * dv.mean() / res["per_seq"][f2]["rmse"].mean()  # +125.0 and +66.7 here: no padding
            expect.append(f"${sign4(dv.mean())} \\pm {1.96 * dv.std(ddof=1) / np.sqrt(n4):.4f}$ (${pct:+.1f}\\%$)")
        bad += cells != expect
    # a tiny negative change keeps its sign in the share, and a one-digit share is padded by a digit
    res2 = dict(metrics={}, per_seq={})
    for lab, v in ((INDEP, 1.0), (BOTH, 0.9998), (INDEP_PAST, 1.0), (BOTH_PAST, 0.912)):
        res2["per_seq"][lab] = dict(rmse=np.full(4, v))
        res2["metrics"][lab] = dict(rmse=(v, 0.0))
    row2 = weight_tex([(0, res2)]).split(r"\midrule", 1)[1].splitlines()[1]
    bad += row2 != (SHORT_TEX[0] + r" & $-0.0002 \pm 0.0000$ ($-\phantom{0}0.0\%$) & "
                    r"$-0.0880 \pm 0.0000$ ($-\phantom{0}8.8\%$) \\")
    check("sweep_weight_table.tex: both header rows, and each setting's paired RMSE change from adding the robust "
          "weight, to noise from y_1..y_t and to noise from y_1..y_{t-1}, with its interval and share (the share "
          "signed as the change, one-digit shares padded)", bad == 0, bad)

    # 13. Repeated passes (Section 5.2), using the same code as experiment 2
    x, y, a, Qs = simulate_selective(np.random.default_rng(14), 3, 60, 0.9)
    y[:, :3] += 6.0  # outliers at the start, so the first weights matter
    seq = kalman_filter(y, R2, Qs, a, c=1.0, P0=SIGMA2)
    plain = kalman_filter(y, R2, Qs, a, P0=SIGMA2)
    prefix_exact, tight = True, True
    for k, run in enumerate(frozen_passes(y, Qs, a, 1.0), start=1):
        if k == 1:
            d = np.abs(run["m"] - plain["m"]).max()
            check("pass 1 (all weights 1) is the ordinary Kalman filter", d == 0, d)
        prefix_exact &= bool((run["m"][:, :k - 1] == seq["m"][:, :k - 1]).all())
        if k <= 3:
            tight &= bool((run["m"][:, k - 1] != seq["m"][:, k - 1]).any())
        if k == 61:
            d = np.abs(run["m"] - seq["m"]).max()
            break
    check("after k passes the first k - 1 filtered means are exact", prefix_exact, float(prefix_exact))
    check("and the k-th is not yet exact (checked for k = 1, 2, 3)", tight, float(tight))
    check("T + 1 passes give the sequential robust filter exactly", d == 0, d)
    # count_passes against an independent iteration written with the information form
    x, y, a, Qs = simulate_selective(np.random.default_rng(107), 3, 40, 0.9)
    y[:, :3] += 6.0
    res = count_passes(y, Qs, a, 1.0, 60)
    ref_seq = scalar_filter(y, R2, Qs, a, c=1.0)  # its prior M0 = 0, P0 = 1 = SIGMA2, as in experiment 2
    sd = np.sqrt(ref_seq["P"])
    w, errs, moves, prev = np.ones_like(y), [], [], None
    for _ in range(60):
        run = scalar_filter(y, R2, Qs, a, w=w)
        errs.append((np.abs(run["m"] - ref_seq["m"]) / sd).max(axis=1))
        moves.append(np.full(len(y), np.inf) if prev is None else (np.abs(run["m"] - prev) / sd).max(axis=1))
        prev = run["m"]
        w = 1.0 / np.sqrt(1.0 + (y - run["m_pred"]) ** 2 / (run["P_pred"] + R2))
    errs, moves = np.array(errs), np.array(moves)

    def first(arr, tol):
        hit = arr < tol
        return np.where(hit.any(axis=0), hit.argmax(axis=0) + 1, -1)

    d = np.abs(res["errs"] - errs[:len(res["errs"])]).max()
    check("count_passes: error after each pass equals an independent iteration", d < 1e-9, d)
    good = all(np.array_equal(res["passes"][tol], first(errs, tol)) and np.array_equal(res["practical"][tol], first(moves, tol))
               for tol in TOLS)
    check("count_passes: pass counts and practical-rule counts equal the independent ones", good, float(good))
    Kp = first(errs, min(TOLS))
    fac = np.array([(errs[k - 1, i] / errs[1, i]) ** (1 / (k - 2)) if k > 2 and errs[1, i] > 0 else np.nan
                    for i, k in enumerate(Kp)])
    good = np.allclose(res["factor"], fac, rtol=1e-6, atol=0, equal_nan=True)
    check("count_passes: contraction factor is (error at the last pass / error at pass 2)^(1/(K - 2))", good, float(good))
    Kg = ref_seq["w"] ** 2 * ref_seq["P_pred"] / (ref_seq["w"] ** 2 * ref_seq["P_pred"] + R2)
    d = np.abs(res["memory"] - a[:, 1:] * (1 - Kg[:, :-1])).max() if res["memory"].shape == (3, 39) else np.inf
    check("count_passes: memory equals a_{t+1} (1 - K_t) of the sequential robust filter, t = 1..T-1", d < 1e-12, d)
    # ... and it is how much of the prediction for one step is still in the prediction for the next:
    # restart the filter (weights frozen, as in l.130) from the independent filter's state, once as it
    # is and once with the prediction for column t_n moved by h, and difference the next prediction
    t_n, h = 17, 1e-6

    def tail(m0v):
        return kalman_filter(y[:1, t_n:], R2, Qs[:1, t_n:], a[:1, t_n:], w_fixed=ref_seq["w"][:1, t_n:], m0=m0v,
                             P0=ref_seq["P"][0, t_n - 1])
    m_f = ref_seq["m"][0, t_n - 1]
    fd = (tail(m_f + h / a[0, t_n])["m_pred"][0, 1] - tail(m_f)["m_pred"][0, 1]) / h
    d = abs(fd - res["memory"][0, t_n])
    check("count_passes: memory equals the finite-difference effect of one step's prediction on the next one's", d < 1e-7,
          d)

    # 14. Wiring of experiment 1: run_scenario rebuilt from its documented parts
    cfg = dict(T=80, jump_t=40, n_train=4, n_val=4, n_test=6, n_stability=0)
    for k in (6, 7):
        res = run_scenario(k, cfg)
        name, Qk, (ef, lf, sf), (et, lt, st) = SCENARIOS[k]
        _, y_tr, _ = simulate_rw(np.random.default_rng([2027, k, 0]), 4, 80, Qk, ef, lf, sf, None)
        _, y_va, _ = simulate_rw(np.random.default_rng([2027, k, 1]), 4, 80, Qk, ef, lf, sf, None)
        x_te, y_te, _ = simulate_rw(np.random.default_rng([2027, k, 2]), 6, 80, Qk, et, lt, st, 40)
        boot_idx = np.random.default_rng([2027, k, 99]).integers(0, 6, size=(N_BOOT, 6))
        fit = fit_all(y_tr, y_va, Qk)
        check(f"run_scenario {k}: fitted on jump-free training (seed 0) and validation (seed 1) data",
              fit == res["fitted"], float(fit == res["fitted"]))
        bad = 0
        for lab, kw in filter_specs(fit, y_te).items():
            out = kalman_filter(y_te, kw["R"], Qk, c=kw.get("c"), R_t=kw.get("R_t"))
            summ, _, boots = summarise(x_te, gaussian_scores(x_te, out["m"], out["P"]), 40, boot_idx)
            bad += summ != res["metrics"][lab] or not np.array_equal(boots, res["boots"][lab])
        summ, _, boots = summarise(x_te, exact_filter(y_te, x_te, Qk, et, lt, st), 40, boot_idx)
        bad += summ != res["metrics"][BAYES] or not np.array_equal(boots, res["boots"][BAYES])
        check(f"run_scenario {k}: every row equals its filter rebuilt from the documented parts", bad == 0, bad)

    # 15. The trade-off curve and the tuning-stability refits of experiment 1, rebuilt
    k = 0
    cfg = dict(T=80, jump_t=40, n_train=4, n_val=4, n_test=6, n_stability=2)
    with quiet():
        res = run_scenario(k, cfg, extra=True)
    _, Qk, (ef, lf, sf), (et, lt, st) = SCENARIOS[k]
    x_te, y_te, _ = simulate_rw(np.random.default_rng([2027, k, 2]), 6, 80, Qk, et, lt, st, 40)
    boot_idx = np.random.default_rng([2027, k, 99]).integers(0, 6, size=(N_BOOT, 6))
    cs = [d["c"] for d in res["curve"]]
    bad = cs != [float(c) for c in C_GRID]
    for d in res["curve"]:
        Rc = res["fitted"]["R_by_c"][d["c"]]
        out = kalman_filter(y_te, Rc, Qk, c=d["c"])
        mt, _, _ = summarise(x_te, gaussian_scores(x_te, out["m"], out["P"]), 40, boot_idx)
        bad += d["R"] != Rc or any(d[key] != mt[key][0] for key in ("rmse", "crps", "cov90", "delay"))
    check("trade-off curve: one point for every c, each the robust filter with the R tuned for that c", bad == 0, bad)
    same = all(res["curve"][-1][key] == res["metrics"][ONE_FC][key][0] for key in ("rmse", "crps", "cov90", "delay"))
    check("trade-off curve: its c = inf point is the 'Kalman, one R (forecast CRPS)' row", same, float(same))
    bad, fits = 0, []
    for r in range(cfg["n_stability"]):
        _, y_tr2, _ = simulate_rw(np.random.default_rng([2027, k, 10 + r]), 4, 80, Qk, ef, lf, sf, None)
        _, y_va2, _ = simulate_rw(np.random.default_rng([2027, k, 20 + r]), 4, 80, Qk, ef, lf, sf, None)
        fits.append(fit_all(y_tr2, y_va2, Qk))
        bad += fits[-1] != res["stability"][r]["fit"]
        for lab, kw in filter_specs(fits[-1], y_te).items():
            if lab in (ONE_ML, INDEP, ROB, BOTH):
                out = kalman_filter(y_te, kw["R"], Qk, c=kw.get("c"), R_t=kw.get("R_t"))
                mt, _, _ = summarise(x_te, gaussian_scores(x_te, out["m"], out["P"]), 40, boot_idx[:1])
                bad += res["stability"][r][lab] != dict(rmse=mt["rmse"][0], crps=mt["crps"][0], cov90=mt["cov90"][0],
                                                        delay=mt["delay"][0])
    check("tuning stability: refits on fresh jump-free training (seed 10 + r) and validation (seed 20 + r) "
          "sequences, rows rebuilt", bad == 0, bad)
    apart = fits[0] != fits[1] and all(f != res["fitted"] for f in fits)
    check("tuning stability check: the refits differ from each other and from the main fit", apart, float(apart))

    # 16. Figure 1: every line and mark shows the values it stands for
    res_f = dict(res, fitted=dict(res["fitted"], c_rob=2.25))  # a tuned c distinct from C_PRIOR
    captured, close_fig = [], plt.close
    plt.close = lambda fig=None: captured.append(fig)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            figure_exp1(res_f, cfg, Path(tmp) / "fig")
    finally:
        plt.close = close_fig
    fig = captured[0]
    t0, t1 = max(cfg["jump_t"] - 70, 0), min(cfg["jump_t"] + 100, cfg["T"])
    lines_a = {ln.get_label(): ln for ln in fig.axes[0].get_lines()}
    bad = sum(lab not in lines_a or not np.array_equal(lines_a[lab].get_ydata(), res_f["example"]["fits"][lab][0][t0:t1])
              for lab in (ONE_TRUE, ONE_ML, INDEP, ROB))
    curve = {d["c"]: d for d in res_f["curve"]}
    finite = [d for d in res_f["curve"] if np.isfinite(d["c"])]
    for axb, key in zip(fig.axes[1:3], ("rmse", "delay")):
        ln = {line.get_label(): line for line in axb.get_lines()}
        if not all(name in ln for name in (CURVE_NAME, INF_NAME, PRIOR_NAME, TUNED_NAME)):
            bad += 1
            continue
        bad += ln[INF_NAME].get_ydata()[0] != curve[np.inf][key]
        bad += ln[PRIOR_NAME].get_xdata()[0] != C_PRIOR or ln[PRIOR_NAME].get_ydata()[0] != curve[C_PRIOR][key]
        bad += ln[TUNED_NAME].get_xdata()[0] != 2.25 or ln[TUNED_NAME].get_ydata()[0] != curve[2.25][key]
        bad += list(ln[CURVE_NAME].get_xdata()) != [d["c"] for d in finite]
        bad += list(ln[CURVE_NAME].get_ydata()) != [d[key] for d in finite]
        for lab, name in ((ONE_ML, ML_NAME), (INDEP, INDEP_NAME), (BAYES, "Bayes filter (reference)")):
            bad += name not in ln or ln[name].get_ydata()[0] != res_f["metrics"][lab][key][0]
    plt.close(fig)
    check("figure 1: the filters in (a); the curve, the c = inf point, the fixed and tuned c marks and the reference "
          "lines in (b) and (c)", bad == 0, bad)

    # 17. Experiment 2's tables, on a tiny configuration where some 1e-8 counts are not reached
    cfg2 = dict(T2_list=[40, 60], T2_mid=60, n2=3, maxpass=10)
    with tempfile.TemporaryDirectory() as tmp, quiet():
        experiment2(cfg2, Path(tmp))
        rows = json.loads((Path(tmp) / "exp2_results.json").read_text(encoding="utf-8"))
        txt = (Path(tmp) / "exp2_table.txt").read_text(encoding="utf-8").splitlines()
        tex = (Path(tmp) / "exp2_table.tex").read_text(encoding="utf-8")
    c2, bad, unreached = f"c={C2:g}", 0, 0
    for tol, name in zip(TOLS, ("1%", "1e-8")):
        extra, left = [], 0
        for row in rows:
            for T in cfg2["T2_list"]:
                pr, pa = (np.array(row[f"T={T} {c2}"][part][str(tol)]) for part in ("practical", "passes"))
                fine = (pr > 0) & (pa > 0)
                extra += list(pr[fine] - pa[fine])
                left += int((~fine).sum())
        unreached += left
        want = f"  tolerance {name} of sd: " + (f"{np.median(extra):g} ({max(extra)})" if extra else "n/a") + \
               (f" ({left} sequences not converged, left out)" if left else "")
        bad += want not in txt
    check("experiment 2: the practical-rule line is the median (max) of the extra passes over sequences where both "
          "counts were reached", bad == 0 and unreached > 0, bad)

    def cell(p, maxpass):  # median (max), counting a sequence that was not reached as maxpass
        p = np.array(p)
        pc = np.where(p < 0, maxpass, p)
        lead = ">=" if 2 * (p < 0).sum() >= len(p) else ""
        return f"{lead}{np.median(pc):g} ({'>=' if (p < 0).any() else ''}{pc.max()})".replace(">=", r"$\ge$")

    trs = [ln for ln in tex.split(r"\midrule", 1)[1].splitlines() if ln.endswith(r"\\")]
    bad = len(trs) != len(rows)
    for ln, row in zip(trs, rows):
        m = row[f"T={cfg2['T2_mid']} {c2}"]
        factor = np.nan if m["factor"] is None else m["factor"]
        bad += [c.strip() for c in ln[:-2].split("&")][1:] != (
            [f"{m['memory']:.3f}", f"{factor:.3f}"]
            + [cell(row[f"T={T} {c2}"]["passes"][str(tol)], row[f"T={T} {c2}"]["maxpass"])
               for tol in TOLS for T in cfg2["T2_list"]])
    check("experiment 2: every cell of exp2_table.tex equals the saved results", bad == 0, bad)

    # 18. Worker processes give the same numbers as one process (check 20 runs the same code with an
    #     in-process stand-in pool, so it is also checked with --workers 1)
    names18 = ("Bayes filter split across worker processes equals one call (independent outliers at Q = 0.1; "
               "bursts at Q = 1; uneven chunks)", "fit_all in worker processes equals fit_all in one process")
    if _POOL is None:
        for name in names18:
            skip(name, "needs --workers 2 or more")
    else:
        rng_p = np.random.default_rng(16)
        same = True
        for Qv, stay in ((0.1, None), (1.0, 0.8)):
            x, y, _ = simulate_rw(rng_p, 3 * _WORKERS + 1, 120, Qv, EPS, LAM, stay, 60)
            split, whole = bayes_filter(y, x, Qv, EPS, LAM, stay), exact_filter(y, x, Qv, EPS, LAM, stay)
            same &= set(split) == set(whole) and all(np.array_equal(split[key], whole[key]) for key in whole)
        check(names18[0], same, float(same))
        _, y_tr, _ = simulate_rw(rng_p, 3, 100, 0.1, EPS, LAM, None, None)
        _, y_va, _ = simulate_rw(rng_p, 3, 100, 0.1, EPS, LAM, None, None)
        in_pool = fit_all(y_tr, y_va, 0.1)
        with pool_as(None, 1):
            alone = fit_all(y_tr, y_va, 0.1)
        check(names18[1], in_pool == alone, float(in_pool == alone))

    more_checks(check, quiet, InProcessPool, pool_as, replaced)

    # the record of the run: run_info.json (main() writes it before the checks) holds this file's SHA-256;
    # and the two worker-process checks are SKIP exactly when there are no worker processes
    info_path = out_dir / "run_info.json"
    recorded = json.loads(info_path.read_text(encoding="utf-8")).get("script_sha256") if info_path.exists() else None
    check("run_info.json records the SHA-256 of this script", recorded == hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest(), "missing" if recorded is None else recorded[:12])
    good = skipped == (list(names18) if _POOL is None else []) and not any(
        ln.startswith("PASS") and "needs --workers" in ln for ln in lines)
    check("the two worker-process checks are SKIP exactly when there are no worker processes", good, float(good))

    (out_dir / "checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit("A self-check failed; see checks.txt. Results were not produced.")
    n_pass = len(lines) - len(skipped)
    log(f"all {n_pass} self-checks passed"
        + (f"; {len(skipped)} skipped (they compare worker processes with one process; run with --workers 2 or "
           "more to include them)" if skipped else ""))


def more_checks(check, quiet, InProcessPool, pool_as, replaced):
    """Checks 19-25 of self_checks: the choices of fit_all on the edges of its search, the pool
    code without worker processes, the .tex layouts, the figures' artists, experiment 2's
    summaries, the printed text, and the grid filter over many steps."""
    import matplotlib.axes
    import matplotlib.figure

    inf = np.inf
    C_list = [float(c) for c in C_GRID]
    keys = [(None, inf, c) for c in C_list] + [(k, ci, c) for k in K_GRID for ci in C_list if np.isfinite(ci)
                                                for c in C_list]

    # 19. fit_all's choices on fabricated tables, built so that the best entry is "noise off,
    #     robust on", "noise on, robust off", "both on" or "everything off", for both kinds of
    #     noise; the expected fits are found by brute force over the fabricated tables
    def fabricated(special, special_past, seed):
        """Validation tables {(k, noise c, robust c): scores over R_GRID} for the noise from y_t
        (S) and from past readings (Sp), sharing the rows without noise. Every entry has its own
        level, distinct within each table, and its own best R; `special` entries get the given
        lower levels."""
        rng_f = np.random.default_rng(seed)
        iRs = np.arange(len(R_GRID))
        level = dict(zip(keys, 1.0 + 1e-3 * rng_f.permutation(len(keys))))
        level_p = dict(zip(keys, 1.0 + 1e-3 * (rng_f.permutation(len(keys)) + 0.5)))
        level.update(special)
        level_p.update(special_past)
        best = dict(zip(keys, rng_f.integers(10, len(R_GRID) - 10, size=len(keys))))
        S = {key: level[key] + 1e-4 * ((iRs - best[key]) / 10.0) ** 2 for key in keys}
        Sp = {key: S[key] if key[0] is None else level_p[key] + 1e-4 * ((iRs - best[key]) / 10.0) ** 2 for key in keys}
        return S, Sp

    def fake_scores(S, Sp):  # stands in for _scores_over_c, reading the fabricated tables
        def fn(args):
            _, _, k, c_in, past = args
            if k is None or not np.isfinite(c_in):  # noise off: R_t = R, whatever the window
                return [S[(None, inf, c)] for c in C_list]
            return [(Sp if past else S)[(k, c_in, c)] for c in C_list]
        return fn

    def expected_fit(S, Sp):  # the documented choices, by brute force; the tables have no ties
        def argbest(sub):
            key = min(sub, key=lambda kk: sub[kk].min())
            return key, float(R_GRID[int(np.argmin(sub[key]))])
        e = {"R_fc": float(R_GRID[np.argmin(S[(None, inf, inf)])])}
        off = {key: s for key, s in S.items() if key[0] is None}
        e["R_by_c"] = {key[2]: float(R_GRID[np.argmin(s)]) for key, s in off.items()}
        (_, _, e["c_rob"]), e["R_rob"] = argbest(off)
        e["R_prior"] = e["R_by_c"][C_PRIOR]
        (e["k_in"], e["c_in"], _), e["R_in"] = argbest({key: s for key, s in S.items() if key[2] == inf})
        e["R_in_prior"] = float(R_GRID[np.argmin(S[(K_PRIOR, C_PRIOR, inf)])])
        (e["k_both"], e["c_in_both"], e["c_both"]), e["R_both"] = argbest(S)
        (e["k_past"], e["c_past"], _), e["R_past"] = argbest({key: s for key, s in Sp.items() if key[2] == inf})
        (e["k_both_past"], e["c_in_both_past"], e["c_both_past"]), e["R_both_past"] = argbest(Sp)
        return e

    variants = [
        ("noise off, robust on", {(None, inf, 2.25): 0.5, (None, inf, inf): 0.8, (K_PRIOR, C_PRIOR, inf): 0.9}, {}),
        ("noise on, robust off", {(3, 4.0, inf): 0.5, (None, inf, 1.5): 0.7}, {(5, 2.5, inf): 0.45}),
        ("both on", {(9, 1.5, 3.0): 0.5, (5, 2.5, inf): 0.7, (None, inf, 2.75): 0.75},
         {(9, 5.0, 2.5): 0.4, (3, 1.75, inf): 0.6}),
        ("everything off", {(None, inf, inf): 0.5}, {}),
    ]
    y = np.random.default_rng(5).normal(size=(2, 20))
    exps = []
    for i, (name, special, special_past) in enumerate(variants):
        S, Sp = fabricated(special, special_past, 100 + i)
        with replaced("_scores_over_c", fake_scores(S, Sp)), pool_as(None, 1), quiet():
            fit = fit_all(y, y, 0.1)
        exps.append(expected_fit(S, Sp))
        bad = [key for key in exps[-1] if fit[key] != exps[-1][key]]
        check(f"fit_all on fabricated tables whose best entry is '{name}': every choice is the brute-force one "
              f"({', '.join(bad) or 'none differ'})", not bad, len(bad))
        good = all((fit[k] is None) == (not np.isfinite(fit[c])) for k, c in
                   (("k_in", "c_in"), ("k_both", "c_in_both"), ("k_past", "c_past"), ("k_both_past", "c_in_both_past")))
        check(f"fit_all on fabricated tables ('{name}'): k is None exactly when the noise c is inf", good, float(good))
    e = exps
    apart = (e[0]["c_in_both"] == inf and np.isfinite(e[0]["c_both"]) and e[0]["c_in_both_past"] == inf
             and e[1]["c_both"] == inf and np.isfinite(e[1]["c_in_both"])
             and e[1]["c_both_past"] == inf and np.isfinite(e[1]["c_in_both_past"])
             and np.isfinite(e[2]["c_in_both"]) and np.isfinite(e[2]["c_both"])
             and np.isfinite(e[2]["c_in_both_past"]) and np.isfinite(e[2]["c_both_past"])
             and e[3]["c_in"] == inf and e[3]["k_in"] is None and e[3]["c_both"] == inf
             and e[3]["c_past"] == inf and e[3]["c_both_past"] == inf)
    check("the fabricated tables put the best entry on each edge of the search, for both kinds of noise", apart,
          float(apart))
    s_tie = np.array([3.0, 1.0, 2.0, 1.0])  # the best R ties within a row too: argmin takes the first
    good = (best_over({"a": s_tie + 1.0, "b": s_tie, "c": s_tie.copy()}) == (1.0, float(R_GRID[1]), "b")
            and best_over({"c": s_tie.copy(), "b": s_tie}) == (1.0, float(R_GRID[1]), "c"))
    check("best_over: an exact tie goes to the first parameter in the dict, and to the first R in a row", good,
          float(good))

    # 20. The pool code with an in-process stand-in pool, so it is checked with any --workers
    bad = 0
    for n_w, Qv, stay, seed in ((2, 0.1, None, 61), (3, 0.1, 0.8, 62), (3, 1.0, None, 63), (6, 0.1, 0.8, 64)):
        x, y, _ = simulate_rw(np.random.default_rng(seed), 2 * n_w + 1, 40, Qv, EPS, LAM, stay, 20)
        with pool_as(InProcessPool(), n_w):
            split = bayes_filter(y, x, Qv, EPS, LAM, stay)
        whole = exact_filter(y, x, Qv, EPS, LAM, stay)
        bad += set(split) != set(whole) or not all(np.array_equal(split[key], whole[key]) for key in whole)
    check("Bayes filter through a stand-in pool, in 2, 3 or 6 uneven chunks (bursts; Q = 1), equals one call",
          bad == 0, bad)
    S, Sp = fabricated({(9, 1.5, 3.0): 0.5}, {(3, 2.0, 2.0): 0.45}, 7)
    y = np.random.default_rng(5).normal(size=(2, 20))
    with replaced("_scores_over_c", fake_scores(S, Sp)), quiet():
        with pool_as(None, 1):
            alone = fit_all(y, y, 0.1)
        with pool_as(InProcessPool(), 4):
            pooled = fit_all(y, y, 0.1)
    check("fit_all through the pool branch of run_tasks equals fit_all without a pool", alone == pooled,
          float(alone == pooled))

    # A fabricated run_scenario(..., extra=True) result in which every number says which filter,
    # metric, c or refit it belongs to, and a tiny experiment 2 with its figure recorded
    cfg = dict(T=300, jump_t=150, n_train=4, n_val=4, n_test=4, n_stability=2)
    T, jt = cfg["T"], cfg["jump_t"]
    rng_s = np.random.default_rng(21)
    x = np.cumsum(rng_s.normal(0, 0.3, T)) + np.where(np.arange(T) >= jt, JUMP, 0.0)
    outlier = rng_s.random(T) < 0.1
    y = x + np.where(outlier, 10.0, 1.0) * rng_s.normal(size=T)
    y[100], y[120], outlier[100], outlier[120] = x[100] + 60.0, x[120] - 60.0, True, True  # beyond the axis
    res = dict(name="fabricated", metrics={}, per_seq={}, boots={})
    spread = np.arange(4.0)
    for i, lab in enumerate(FILTERS):
        ps = dict(rmse=0.1 * (i + 1) + 0.0013 * (i + 1) * spread, crps=0.5 + 0.017 * (i + 1) + 0.0021 * (i + 1) * spread,
                  rmse_after=1.0 + 0.071 * (i + 1) + 0.0037 * (i + 1) * spread, cov90=0.3031 + i / 1000 + 0 * spread,
                  cov50=0.6061 + i / 1000 + 0 * spread, width90=4.041 + i / 100 + 0 * spread)
        res["per_seq"][lab] = ps
        res["metrics"][lab] = {key: (float(v.mean()), 0.001 * (i + 1)) for key, v in ps.items()}
        res["metrics"][lab]["delay"] = (float(20 + 3 * i), float(18 + 3 * i), float(23 + 3 * i))
        res["boots"][lab] = 10 + i + (np.arange(200) % (i + 2))
    res["fitted"] = dict(R_ml=1.1, R_fc=2.2, R_rob=3.3, c_rob=2.25, R_prior=5.5, R_in=6.6, c_in=inf, k_in=None,
                         R_in_prior=8.8, R_both=9.9, c_both=1.25, k_both=9, c_in_both=0.75, R_past=7.7, c_past=inf,
                         k_past=None, R_both_past=4.4, c_both_past=2.9, c_in_both_past=1.1, k_both_past=3,
                         R_by_c={float(c): 0.5 + 0.1 * j for j, c in enumerate(C_GRID)})
    res["curve"] = [dict(c=float(c), R=0.5 + 0.1 * j, rmse=0.9 - 0.01 * j, crps=0.5 - 0.004 * j, cov90=0.8 + 0.005 * j,
                         delay=float(40 - 2 * j)) for j, c in enumerate(C_GRID)]
    res["stability"] = []
    for r in range(2):
        f2 = dict(res["fitted"], R_ml=11.0 + r, R_rob=1.31 + r, c_rob=[2.5, inf][r], R_in=1.41 + r, c_in=[3.0, inf][r],
                  k_in=[9, None][r], R_both=1.51 + r, c_both=[3.5, 1.5][r], c_in_both=[6.0, inf][r], k_both=[3, None][r])
        row = dict(fit=f2)
        for j, lab in enumerate((ONE_ML, INDEP, ROB, BOTH)):
            row[lab] = dict(rmse=0.611 + 0.01 * j + 0.1 * r, crps=0.333 + 0.01 * j + 0.1 * r, cov90=0.9,
                            delay=float(7 + j + 10 * r))
        res["stability"].append(row)
    res["example"] = dict(x=x, y=y, outlier=outlier,
                          fits={lab: (x + 0.05 * (i + 1) * np.sin(np.arange(T) / (5.0 + i)), 0.2 + 0.01 * i + 0.0 * x)
                                for i, lab in enumerate(FILTERS[:-1])})

    class Capture:
        """Records the calls the figure code makes to a few matplotlib methods, and the figures."""

        def __init__(self):
            self.calls, self.figs = [], []
            self.targets = [(matplotlib.axes.Axes, name) for name in ("scatter", "fill_between", "legend", "plot")]
            self.targets.append((matplotlib.figure.Figure, "legend"))

        def __enter__(self):
            self.saved = [(cls, name, getattr(cls, name)) for cls, name in self.targets]
            for cls, name, orig in self.saved:
                def wrapper(obj, *a, _orig=orig, _name=f"{cls.__name__}.{name}", **kw):
                    out = _orig(obj, *a, **kw)
                    self.calls.append((_name, obj, a, kw, out))
                    return out
                setattr(cls, name, wrapper)
            self.close = plt.close
            plt.close = lambda fig=None: self.figs.append(fig)
            return self

        def __exit__(self, *exc):
            for cls, name, orig in self.saved:
                setattr(cls, name, orig)
            plt.close = self.close
            for f in self.figs:
                self.close(f)

    cfg2 = dict(T2_list=[40, 60], T2_mid=60, n2=3, maxpass=10)
    with tempfile.TemporaryDirectory() as tmp, Capture() as cap2, quiet():
        experiment2(cfg2, Path(tmp))
        rows2 = json.loads((Path(tmp) / "exp2_results.json").read_text(encoding="utf-8"))
        tex2 = (Path(tmp) / "exp2_table.tex").read_text(encoding="utf-8")
        txt2 = (Path(tmp) / "exp2_table.txt").read_text(encoding="utf-8")

    def tex_rows(tex):
        return [ln for ln in tex.splitlines() if ln.endswith(r"\\")]

    def ncols(tex):
        m = re.search(r"\\begin\{tabular\}\{@\{\}(l*r*)@\{\}\}", tex)
        return len(m.group(1)) if m else -1

    # 21. The .tex layouts: column specs, spans and rules over the right cells, headers in cell order
    tex = table_exp1_tex(res)
    lines_t, head = tex.splitlines(), tex_rows(tex)
    bad = ncols(tex) != 7 or any(ln.count("&") != 6 for ln in head[1:])
    bad += head[0].strip() != r"& \multicolumn{4}{c}{Before the jump} & \multicolumn{2}{c}{After the jump} \\"
    bad += r"\cmidrule(lr){2-5} \cmidrule(l){6-7}" not in lines_t
    bad += head[1] != r"Filter & RMSE & CRPS & 90\% cover & 90\% width & Delay & RMSE \\"
    check("exp1_table.tex: 7 columns in the spec and every row; spans 4 + 2 over (RMSE, CRPS, cover, width) and "
          "(delay, RMSE); the header names the cells in their order", bad == 0, bad)
    want = [r"\midrule"]
    for label, _ in TEX_ROWS:
        if label in (INDEP, ROB, BAYES):
            want.append(r"\midrule")
        want.append(label)
    got = []
    for ln in lines_t[lines_t.index(head[1]) + 1:]:
        if ln == r"\midrule":
            got.append(ln)
        elif ln.endswith(r"\\"):
            got.append(next((lab for lab, name in TEX_ROWS if ln.startswith(name + " & ")), "?"))
    check("exp1_table.tex: a rule above the input-dependent, robust and Bayes groups and nowhere else", got == want,
          sum(a != b for a, b in zip(got, want)) + abs(len(got) - len(want)))
    bad = 0
    for key, fmt in (("rmse", lambda v: f"{v:.3f}"), ("delay", lambda v: f"{v:.0f}")):
        st = sweep_tex([(k, res) for k in range(len(SCENARIOS))], key, fmt, "TITLE")
        rows_ = tex_rows(st)
        bad += ncols(st) != 1 + len(SWEEP_COLS) or any(ln.count("&") != len(SWEEP_COLS) for ln in rows_[1:])
        bad += not rows_[0].startswith(r"TITLE & \multicolumn{4}{c}{Tuned} & \multicolumn{2}{c}{$c = "
                                       f"{C_PRIOR:g}" r"$ fixed} & \\")
        bad += r"\cmidrule(lr){2-5} \cmidrule(lr){6-7}" not in st.splitlines()
        bad += [lab for lab, _ in SWEEP_COLS[:4]] != [ONE_ML, INDEP, ROB, BOTH]
        bad += [lab for lab, _ in SWEEP_COLS[4:6]] != [INDEP_FIX, ROB_FIX]
    wt = weight_tex([(k, res) for k in range(len(SCENARIOS))])
    wrows = tex_rows(wt)
    bad += ncols(wt) != 3 or any(ln.count("&") != 2 for ln in wrows[1:])
    bad += wrows[0].count("&") != 1 or r"\multicolumn{2}{c}{" not in wrows[0]
    bad += r"\cmidrule(l){2-3}" not in wt.splitlines() or WEIGHT_PAIRS != [(BOTH, INDEP), (BOTH_PAST, INDEP_PAST)]
    check("sweep tables: 8 columns, 'Tuned' over the four tuned filters and 'c = 2 fixed' over the two fixed ones; "
          "the weight table: 3 columns, its span over both", bad == 0, bad)
    Ts = cfg2["T2_list"]
    th = [f"{T_:,}".replace(",", "{,}") for T_ in Ts]
    cells = [f"$T = {th[0]}$"] + [f"${t_}$" for t_ in th[1:]]
    r2 = tex_rows(tex2)
    bad = r2[1] != "Dynamics & Memory & Factor & " + " & ".join(cells) + " & " + " & ".join(cells) + r" \\"
    spans = [int(v) for v in re.findall(r"\\multicolumn\{(\d+)\}", r2[0])]
    titles = re.findall(r"Passes to (1\\% of sd|\$10\^\{-8\}\$ sd)", r2[0])
    bad += titles != [r"1\% of sd", r"$10^{-8}$ sd"] or spans != [len(Ts), len(Ts)]
    bad += ncols(tex2) != 3 + 2 * len(Ts) or any(ln.count("&") != 2 + 2 * len(Ts) for ln in r2[1:])
    bad += list(TOLS) != [1e-2, 1e-8]
    names = {"a_t <= 0.6": r"$a_t \le 0.6$", "a_t ~ 0.5": r"$a_t \approx 0.5$", "a_t ~ 0.9": r"$a_t \approx 0.9$",
             "a_t ~ 0.99": r"$a_t \approx 0.99$", "a_t ~ 0.999": r"$a_t \approx 0.999$", "random walk": "Random walk"}
    bad += [ln.split(" & ")[0] for ln in r2[2:]] != [names[row["regime"]] for row in rows2]
    check("exp2_table.tex: 'T = ' header cells in T order, tolerance titles in TOLS order, the regimes' row names",
          bad == 0, bad)

    # 22. The figures: each legend entry against the artist it names; the band; the off-axis markers;
    #     the c = inf connector; figure 2's points and curves against recomputed medians
    with tempfile.TemporaryDirectory() as tmp, Capture() as cap:
        figure_exp1(res, cfg, Path(tmp) / "fig")
    t0, t1 = max(jt - 70, 0), min(jt + 100, T)
    ex = res["example"]
    fig = cap.figs[0]
    ax = fig.axes[0]
    lines_a = {ln.get_label(): ln for ln in ax.get_lines()}
    leg = [c for c in cap.calls if c[0] == "Axes.legend" and c[1] is ax][-1]
    handles, labels = leg[2][0], leg[2][1]
    named = {INDEP_NAME: INDEP, "Kalman, $R = 1$": ONE_TRUE, ML_NAME: ONE_ML}
    bad = 0
    for h, lab in zip(handles, labels):
        if lab in named:
            bad += h is not lines_a.get(named[lab])
        elif lab.startswith("robust, tuned"):
            bad += not (isinstance(h, tuple) and h[-1] is lines_a.get(ROB))
        elif lab == "true state $x_t$":
            bad += not np.array_equal(h.get_ydata(), ex["x"][t0:t1])
        elif lab == "outlier beyond the axis":
            bad += not (isinstance(h, tuple) and [m.get_marker() for m in h] == ["^", "v"])
    bad += sorted(named) != sorted(lab for lab in labels if lab in named)
    check("figure 1(a): each legend entry is paired with the line it names", bad == 0, bad)
    m_r, P_r = ex["fits"][ROB]
    fb = [c for c in cap.calls if c[0] == "Axes.fill_between" and c[1] is ax][0]
    bad = not (np.allclose(fb[2][1], (m_r - Z90 * np.sqrt(P_r))[t0:t1], rtol=0, atol=1e-12)
               and np.allclose(fb[2][2], (m_r + Z90 * np.sqrt(P_r))[t0:t1], rtol=0, atol=1e-12))
    lo, hi = ax.get_ylim()
    pad = 0.05 / 1.1 * (hi - lo)
    yy, oo, tt = ex["y"][t0:t1], ex["outlier"][t0:t1], np.arange(t0, t1) + 1
    n_edge = 0
    for c in cap.calls:
        if c[0] == "Axes.scatter" and c[1] is ax and c[3].get("marker") in ("^", "v"):
            n_edge += 1
            mk, ys, xs = c[3]["marker"], np.asarray(c[2][1]), np.asarray(c[2][0])
            beyond = oo & ((yy > hi - pad) if mk == "^" else (yy < lo + pad))
            bad += not (np.array_equal(xs, tt[beyond]) and (np.all(ys > hi - pad) if mk == "^" else np.all(ys < lo + pad)))
    bad += n_edge != 2  # the fabricated example has a reading far beyond each edge
    ticks = ax.get_yticks()
    bad += len(ticks) < 3 or not np.all((ticks >= lo + pad) & (ticks <= hi - pad))
    check("figure 1(a): the band is m +- Z90 sqrt(P) of the robust filter; up markers at the top for readings above "
          "the axis, down markers at the bottom; no tick in the edge strips", bad == 0, bad)
    fl = [c for c in cap.calls if c[0] == "Figure.legend"][-1]
    hs, ls = fl[2][0], fl[2][1]
    bad = sum(lab != "" and h.get_label() != lab for h, lab in zip(hs, ls))
    bad += sorted(lab for lab in ls if lab) != sorted([CURVE_NAME, PRIOR_NAME, TUNED_NAME, INF_NAME, ML_NAME,
                                                       INDEP_NAME, "Bayes filter (reference)"])
    curve = [d for d in res["curve"] if np.isfinite(d["c"])]
    inf_pt = res["curve"][-1]
    for axb, key in zip(fig.axes[1:3], ("rmse", "delay")):
        dotted = [ln for ln in axb.get_lines() if ln.get_linestyle() == ":"]
        bad += len(dotted) != 1 or list(dotted[0].get_ydata()) != [curve[-1][key], inf_pt[key]]
    check("figure 1(b)/(c): each legend entry is the line with that label, all seven present; the dotted connector "
          "runs from c = 15 to the c = inf point", bad == 0, bad)
    fig2 = cap2.figs[0]
    bad = 0
    for axb, tol in zip(fig2.axes[1:3], TOLS):
        for row, (label, *_) in zip(rows2, REGIMES):
            name = ("random walk ($a_t = 1$)" if label == "random walk"
                    else "$" + label.replace("<=", r"\leq").replace("~", r"\approx") + "$")
            ln = [line for line in axb.get_lines() if line.get_label() == name]
            rr = [row[f"T={T_} c={C2:g}"] for T_ in Ts]
            med = [np.median(np.where(np.array(q["passes"][str(tol)]) < 0, q["maxpass"], q["passes"][str(tol)])) for q in rr]
            bad += not (len(ln) == 1 and list(ln[0].get_ydata()) == med
                        and np.all(np.abs(np.log10(np.asarray(ln[0].get_xdata()) / Ts)) <= 0.16))
    for label, a_med, cap_a, seed in REGIMES:
        T_ = cfg2["T2_mid"]
        _, yy2, aa, QQ = simulate_selective(np.random.default_rng([2027, seed, T_]), cfg2["n2"], T_, a_med, cap_a)
        cp = count_passes(yy2, QQ, aa, C2, min(cfg2["maxpass"], T_ + 2))
        e2 = np.maximum(np.median(cp["errs"], axis=1), 1e-13)
        bad += not any(np.array_equal(line.get_ydata(), e2) for line in fig2.axes[0].get_lines())
    check("figure 2: (b) and (c) show each regime's median passes at their own tolerance, at T up to the dodge; "
          "(a) shows the c = C2 runs' median errors", bad == 0, bad)
    # the details that keep the figures legible: 1(a)'s legend wholly below the x-axis label, and its
    # input-dependent entry marked "tuned"; in 1(b)/(c) the tuned-c diamond as large as the c = 2 square;
    # in figure 2 every regime at its own, dodged x in (b) and (c), each regime's markers drawn again
    # above every line at the same points, and diamonds shrunk by sqrt(2) to the other markers' size
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # the capture closed the figure, which drops its canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rend = canvas.get_renderer()
    leg_a = ax.get_legend()
    bad = int(leg_a is None or leg_a.get_window_extent(rend).y1 > ax.xaxis.label.get_window_extent(rend).y0 + 0.5)
    bad += leg_a is None or not any("input-dependent" in t.get_text() and "tuned" in t.get_text()
                                    for t in leg_a.get_texts())
    for axb in fig.axes[1:3]:
        marks = {ln.get_label(): ln for ln in axb.get_lines()}
        bad += abs(marks[TUNED_NAME].get_markersize() * np.sqrt(2) - marks[PRIOR_NAME].get_markersize()) > 1e-9
    names2 = [("random walk ($a_t = 1$)" if lab == "random walk"
               else "$" + lab.replace("<=", r"\leq").replace("~", r"\approx") + "$") for lab, *_ in REGIMES]
    for axb in fig2.axes[1:3]:
        by_name = {ln.get_label(): ln for ln in axb.get_lines()}
        xs = np.sort(np.array([np.asarray(by_name[n].get_xdata(), float) for n in names2]), axis=0)
        bad += not np.all(xs[1:] / xs[:-1] > 10 ** 0.03)
    for j, axb in enumerate(fig2.axes[:3]):
        lns = axb.get_lines()
        regime = [ln for ln in lns if ln.get_linestyle() != "None" and ln.get_marker() not in ("None", "", None)]
        top = max(ln.get_zorder() for ln in lns if ln.get_linestyle() != "None")
        bad += len(regime) != len(REGIMES)
        for ln in regime:
            twins = [o for o in lns if o.get_linestyle() == "None" and o.get_marker() == ln.get_marker()
                     and np.array_equal(o.get_xdata(), ln.get_xdata()) and np.array_equal(o.get_ydata(), ln.get_ydata())
                     and o.get_markevery() == ln.get_markevery() and o.get_zorder() > top]
            size = 2.6 if j == 0 else 3.6
            bad += len(twins) != 1 or abs(ln.get_markersize() - (size / np.sqrt(2) if ln.get_marker() == "D" else size)) > 1e-9
    check("figures: 1(a)'s legend below the x-axis label, its input-dependent entry 'tuned'; the tuned-c diamond as "
          "large as the square; figure 2's regimes dodged apart, their markers drawn again on top, diamonds shrunk",
          bad == 0, bad)

    # 23. Experiment 2's Memory and Factor recomputed; and the practical-rule lines when nothing converges
    bad, nan_seen = 0, 0
    T_ = cfg2["T2_mid"]
    for row, (label, a_med, cap_a, seed) in zip(rows2, REGIMES):
        _, yy2, aa, QQ = simulate_selective(np.random.default_rng([2027, seed, T_]), cfg2["n2"], T_, a_med, cap_a)
        cp = count_passes(yy2, QQ, aa, C2, min(cfg2["maxpass"], T_ + 2))
        fac = cp["factor"]
        nan_seen += int(np.isnan(fac).any())
        want_f = float(np.nanmedian(fac)) if np.isfinite(fac).any() else None
        got = row[f"T={T_} c={C2:g}"]
        bad += abs(got["memory"] - float(np.median(cp["memory"]))) > 1e-12
        bad += (got["factor"] is None) != (want_f is None) or (want_f is not None and abs(got["factor"] - want_f) > 1e-12)
    bad += "memory: median over steps and sequences of a_{t+1} (1 - K_t), the filter's memory per step" not in txt2
    check("experiment 2: Memory is the median of a_{t+1} (1 - K_t), as exp2_table.txt says, and Factor the median of "
          "the per-sequence factors over the sequences that have one", bad == 0 and nan_seen > 0, bad)
    cfg_na = dict(T2_list=[30, 40], T2_mid=40, n2=3, maxpass=1)
    with tempfile.TemporaryDirectory() as tmp, Capture(), quiet():
        experiment2(cfg_na, Path(tmp))
        txt = (Path(tmp) / "exp2_table.txt").read_text(encoding="utf-8")
    n_seq = 2 * len(REGIMES) * cfg_na["n2"]
    good = all(f"  tolerance {name} of sd: n/a ({n_seq} sequences not converged, left out)" in txt.splitlines()
               for name in ("1%", "1e-8"))
    check("experiment 2: when no sequence converges, the practical-rule lines say n/a and count them", good, float(good))

    # 24. Printed text: the fit lines, the stability and trade-off lines, the JSON, the notes' windows,
    #     and the labels and docstring against C_PRIOR and K_PRIOR
    f = res["fitted"]
    flat = " ".join(fitted_text(f).split())
    want = [f"input-dependent R = {f['R_in']:.3g}, c = inf (off), k = - (not used)",
            f"robust + input-dependent R = {f['R_both']:.3g}, robust c = {f['c_both']:g}, noise c = {f['c_in_both']:g}, "
            f"k = {f['k_both']}", f"robust R = {f['R_rob']:.3g}, c = {f['c_rob']:g}",
            f"past only: input-dependent R = {f['R_past']:.3g}, c = inf (off), k = - (not used); robust + input-dependent "
            f"R = {f['R_both_past']:.3g}, robust c = {f['c_both_past']:g}, noise c = {f['c_in_both_past']:g}, "
            f"k = {f['k_both_past']}"]
    bad = sum(w not in flat for w in want)

    def cv(v):
        return "inf (off)" if not np.isfinite(v) else f"{v:g}"

    def kv(k_):
        return "- (not used)" if k_ is None else f"{k_}"
    for over in (dict(c_past=1.9, k_past=7), dict(c_in=2.5, k_in=5)):  # one kind of noise on, the other off
        g = dict(f, **over)
        flat_g = " ".join(fitted_text(g).split())
        bad += sum(w not in flat_g for w in (
            f"input-dependent R = {g['R_in']:.3g}, c = {cv(g['c_in'])}, k = {kv(g['k_in'])};",
            f"past only: input-dependent R = {g['R_past']:.3g}, c = {cv(g['c_past'])}, k = {kv(g['k_past'])};"))
    check("fitted_text: each filter's R, c and k in its own place, also with one kind of noise on and the other off; "
          "'inf (off)' and '- (not used)' when off", bad == 0, bad)
    with replaced("run_scenario", lambda k, cfg_, extra=False: res), tempfile.TemporaryDirectory() as tmp, quiet(), \
            Capture():
        experiment1(cfg, Path(tmp), {})
        text = (Path(tmp) / "exp1_table.txt").read_text(encoding="utf-8")
        js = json.loads((Path(tmp) / "exp1_results.json").read_text(encoding="utf-8"))
    bad = 0
    sets = [ln for ln in text.splitlines() if ln.startswith("set ")]
    bad += len(sets) != len(res["stability"])
    for r, (ln, row) in enumerate(zip(sets, res["stability"]), start=1):
        g = row["fit"]
        parts = [f"set {r}: robust R = {g['R_rob']:.3g}, c = {cv(g['c_rob'])};",
                 f"input-dependent R = {g['R_in']:.3g}, c = {cv(g['c_in'])}, "
                 f"k = {'- (not used)' if g['k_in'] is None else g['k_in']};",
                 f"robust + input-dependent R = {g['R_both']:.3g}, robust c = {cv(g['c_both'])}, noise c = "
                 f"{cv(g['c_in_both'])}, k = {'- (not used)' if g['k_both'] is None else g['k_both']};",
                 f"one R = {g['R_ml']:.3g}"]
        bad += sum(p not in ln for p in parts)
        block = text.split(ln, 1)[1].splitlines()[1:5]
        for lab, bl in zip((ONE_ML, INDEP, ROB, BOTH), block):
            bad += bl.split()[-3:] != [f"{row[lab]['rmse']:.3f}", "/", f"{row[lab]['delay']:.0f}"] or lab not in bl
    check("exp1_table.txt: each stability line shows its refit's R, c and k, and each filter's RMSE / delay", bad == 0,
          bad)
    curve_lines = [ln for ln in text.splitlines() if re.match(r"^\s+(\d|inf)", ln) and ln.rstrip()[-1:].isdigit()]
    bad = len(curve_lines) != len(res["curve"])
    for ln, d in zip(curve_lines, res["curve"]):
        bad += ln.split() != [f"{d['c']:g}", f"{d['R']:.3g}", f"{d['rmse']:.3f}", f"{d['crps']:.3f}",
                              f"{100 * d['cov90']:.1f}%", f"{d['delay']:.0f}"]
    check("exp1_table.txt: the trade-off table has c, R, RMSE, CRPS, 90% cover and delay of every point", bad == 0, bad)
    bad = sum(js["bootstrap_delays"][lab] != [int(v) for v in res["boots"][lab]] for lab in FILTERS)
    bad += sum(js["per_sequence"][lab]["rmse_after"] != list(res["per_seq"][lab]["rmse_after"]) for lab in FILTERS)
    check("exp1_results.json: every filter's own bootstrap delays and per-sequence values", bad == 0, bad)
    bad = 0
    for cfg_n in (dict(cfg, jump_t=150), dict(n_test=7, n_train=5, n_val=6, T=90, jump_t=45)):
        for sw in (False, True):
            notes_l = table_notes(cfg_n, sweep=sw)
            text_n = " ".join(notes_l)
            frags = [f"means over {cfg_n['n_test']} test sequences of {cfg_n['T']} steps",
                     f"All over steps 1-{cfg_n['jump_t']} (before the jump)",
                     f"RMSE after the jump: steps {cfg_n['jump_t'] + 1}-{cfg_n['jump_t'] + AFTER}.",
                     f"fitted on {cfg_n['n_train']} training and {cfg_n['n_val']} validation sequences",
                     "the one-R maximum-likelihood filter by the likelihood of the training sequences",
                     "about ±0.001 in RMSE before the jump", "it can be as wide as the narrower row's ±, or wider",
                     "R_t = R + d_t^2 / c^2 equals R / w_t^2 for an IMQ weight on d_t",
                     "(past only): R_t = R + d_{t-1}^2 / c^2, which uses y_1, ..., y_{t-1} only",
                     "is a proper density for y_t, which it would not be with d_t in place of d_{t-1}",
                     "like every filter it was tuned on the forecast with the base R",
                     "in those units d_t is the larger, so the input-dependent weight down-weights more",
                     "the weight stays above about (1 + noise c^2 / robust c^2)^(-1/2)"]
            bad += sum(fr not in text_n for fr in frags) + (len(notes_l) != 9)
            bad += ("which it expects to come in runs" in text_n) != sw  # the Bayes note on bursts: sweep only
    d_ = np.logspace(-1, 4, 200)  # the last note's bound: with e_t = d_t the weight stays above it, and tends to it
    for cr, cn, P_, R_ in ((2.0, 4.0, 0.3, 1.0), (3.0, 1.5, 0.1, 1.4)):
        w_ = imq_weight(d_, P_ + R_ + d_ * d_ / cn ** 2, cr)
        bnd = (1 + cn ** 2 / cr ** 2) ** -0.5
        bad += not (np.all(w_ >= bnd) and abs(w_[-1] - bnd) < 1e-6)
    check("notes: all nine, each with its claim; the counts and step windows of the configuration; the Bayes note on "
          "bursts only under the sweep; the combined filter's weight bound holds", bad == 0, bad)
    cfg_s = dict(cfg, scenarios=list(range(len(SCENARIOS))))
    with replaced("run_scenario", lambda k, cfg_, extra=False: res), tempfile.TemporaryDirectory() as tmp, quiet():
        sweep(cfg_s, Path(tmp), {})
        got_s = {name: (Path(tmp) / name).read_text(encoding="utf-8")
                 for name in ("sweep.txt", "sweep_rmse_table.tex", "sweep_delay_table.tex", "sweep_weight_table.tex")}
    done_s = [(k, res) for k in cfg_s["scenarios"]]
    good = (got_s["sweep_weight_table.tex"] == weight_tex(done_s)
            and got_s["sweep_rmse_table.tex"] == sweep_tex(done_s, "rmse", lambda v: f"{v:.3f}", "RMSE before the jump")
            and got_s["sweep_delay_table.tex"] == sweep_tex(done_s, "delay", lambda v: f"{v:.0f}", "Delay after the jump")
            and got_s["sweep.txt"].count(scenario_text(res)) == len(SCENARIOS)
            and got_s["sweep.txt"].endswith("Notes\n" + "\n".join(f"- {n}" for n in table_notes(cfg_s, sweep=True)) + "\n"))
    check("sweep: every setting's block, the sweep notes (with the Bayes note on bursts), and the RMSE, delay and "
          "weight tables of every setting", good, float(good))
    doc = " ".join(__doc__.split())
    labels = [INDEP_FIX, ROB_FIX] + [name for lab, name in TEX_ROWS if lab in (INDEP_FIX, ROB_FIX)]
    good = (all(f"c = {C_PRIOR:g}" in s for s in labels) and f"c = {C_PRIOR:g}, k = {K_PRIOR} fixed" in doc
            and f"with c = {C_PRIOR:g} fixed" in doc)
    check("labels and the docstring name the fixed c and k that the code uses (C_PRIOR, K_PRIOR)", good, float(good))

    # 25. The grid Bayes filter over many steps: clean readings swinging between +6 and -6, where the
    #     Kalman filter is still exact but the unnormalised posterior mass shrinks by about e^-15 per
    #     step, so a filter that did not renormalise what it carries would underflow within 50 steps
    yb = np.tile(np.where(np.arange(100) % 2 == 0, 6.0, -6.0), (2, 1))
    yb[1] += 0.5
    xb = np.zeros_like(yb)
    worst = 0.0
    for Qv in (0.1, 1.0):
        kf = kalman_filter(yb, R, Qv)
        exb = exact_filter(yb, xb, Qv, 0.0, LAM)
        gs = gaussian_scores(xb, kf["m"], kf["P"])
        worst = max(worst, max(np.abs(exb[key] - gs[key]).max() if np.isfinite(exb[key]).all() else np.inf
                               for key in ("mean", "pit", "crps", "width90")))
    check("grid Bayes filter equals the Kalman filter on clean readings swinging +-6 for 100 steps (Q = 0.1, 1)",
          worst < 1e-6, worst)


# ---------------------------------------------------------------- experiment 1 and sweep
def run_scenario(k, cfg, extra=False):
    name, Q, (eps_f, lam_f, stay_f), (eps_t, lam_t, stay_t) = SCENARIOS[k]
    T, jt = cfg["T"], cfg["jump_t"]

    def sim(j, n, eps, lam, stay, jump):
        return simulate_rw(np.random.default_rng([2027, k, j]), n, T, Q, eps, lam, stay, jump)

    # training and validation data have no jump, so no filter is tuned on one
    _, y_tr, _ = sim(0, cfg["n_train"], eps_f, lam_f, stay_f, None)
    _, y_va, _ = sim(1, cfg["n_val"], eps_f, lam_f, stay_f, None)
    x_te, y_te, o_te = sim(2, cfg["n_test"], eps_t, lam_t, stay_t, jt)
    boot_idx = np.random.default_rng([2027, k, 99]).integers(0, len(x_te), size=(N_BOOT, len(x_te)))

    fit = fit_all(y_tr, y_va, Q)
    res = dict(name=name, fitted=fit, metrics={}, per_seq={}, boots={})
    outs = {}
    for label, kw in filter_specs(fit, y_te).items():
        outs[label] = kalman_filter(y_te, kw["R"], Q, c=kw.get("c"), R_t=kw.get("R_t"))
        sc = gaussian_scores(x_te, outs[label]["m"], outs[label]["P"])
        res["metrics"][label], res["per_seq"][label], res["boots"][label] = summarise(x_te, sc, jt, boot_idx)
    sc = bayes_filter(y_te, x_te, Q, eps_t, lam_t, stay_t)
    res["metrics"][BAYES], res["per_seq"][BAYES], res["boots"][BAYES] = summarise(x_te, sc, jt, boot_idx)

    if extra:
        # trade-off over c: the robust filter with R tuned for each c (c = inf is the one-R
        # Kalman filter tuned by forecast CRPS), on the test sequences
        res["curve"] = []
        for c in C_GRID:
            out = kalman_filter(y_te, fit["R_by_c"][float(c)], Q, c=c)
            mt, _, _ = summarise(x_te, gaussian_scores(x_te, out["m"], out["P"]), jt, boot_idx)
            res["curve"].append(dict(c=float(c), R=fit["R_by_c"][float(c)], rmse=mt["rmse"][0],
                                     crps=mt["crps"][0], cov90=mt["cov90"][0], delay=mt["delay"][0]))
        # tuning stability: refit on fresh training and validation sequences
        res["stability"] = []
        for r in range(cfg["n_stability"]):
            _, y_tr2, _ = sim(10 + r, cfg["n_train"], eps_f, lam_f, stay_f, None)
            _, y_va2, _ = sim(20 + r, cfg["n_val"], eps_f, lam_f, stay_f, None)
            fit2 = fit_all(y_tr2, y_va2, Q)
            row = dict(fit=fit2)
            for label, kw in filter_specs(fit2, y_te).items():
                if label in (ONE_ML, INDEP, ROB, BOTH):
                    out = kalman_filter(y_te, kw["R"], Q, c=kw.get("c"), R_t=kw.get("R_t"))
                    mt, _, _ = summarise(x_te, gaussian_scores(x_te, out["m"], out["P"]), jt, boot_idx[:1])
                    row[label] = dict(rmse=mt["rmse"][0], crps=mt["crps"][0], cov90=mt["cov90"][0],
                                      delay=mt["delay"][0])
            res["stability"].append(row)
            log(f"  tuning stability {r + 1}/{cfg['n_stability']} done")
        res["example"] = dict(x=x_te[0], y=y_te[0], outlier=o_te[0],
                              fits={lab: (outs[lab]["m"][0], outs[lab]["P"][0]) for lab in outs})
    return res


def fmt_c(c, tex=False):
    if not np.isfinite(c):
        return r"\infty" if tex else "inf (off)"
    return f"{c:g}"


def fmt_k(k):
    return "- (not used)" if k is None else f"{k}"


def fitted_text(fit):
    return (f"fitted: one R = {fit['R_ml']:.3g} (max likelihood) or {fit['R_fc']:.3g} (forecast CRPS); "
            f"input-dependent R = {fit['R_in']:.3g}, c = {fmt_c(fit['c_in'])}, k = {fmt_k(fit['k_in'])}; "
            f"input-dependent with c = {C_PRIOR:g}, k = {K_PRIOR} fixed: R = {fit['R_in_prior']:.3g};\n"
            f"        robust R = {fit['R_rob']:.3g}, c = {fmt_c(fit['c_rob'])}; robust with c = {C_PRIOR:g} fixed: "
            f"R = {fit['R_prior']:.3g}; robust + input-dependent R = {fit['R_both']:.3g}, robust c = "
            f"{fmt_c(fit['c_both'])}, noise c = {fmt_c(fit['c_in_both'])}, k = {fmt_k(fit['k_both'])};\n"
            f"        past only: input-dependent R = {fit['R_past']:.3g}, c = {fmt_c(fit['c_past'])}, "
            f"k = {fmt_k(fit['k_past'])}; robust + input-dependent R = {fit['R_both_past']:.3g}, robust c = "
            f"{fmt_c(fit['c_both_past'])}, noise c = {fmt_c(fit['c_in_both_past'])}, k = {fmt_k(fit['k_both_past'])}")


def signed(v, digits=4):
    """v with its sign, at the given decimals; a value that rounds to zero prints as +0.000..."""
    s = f"{v:+.{digits}f}"
    return "+" + s[1:] if float(s) == 0 else s


def paired(res, f1, f2, key):
    """Mean and 95% half-width of the per-sequence differences f1 - f2 of a metric."""
    dv = res["per_seq"][f1][key] - res["per_seq"][f2][key]
    return float(dv.mean()), float(1.96 * dv.std(ddof=1) / np.sqrt(len(dv)))


def scenario_text(res):
    lines = [f"== {res['name']} ==", fitted_text(res["fitted"]), "",
             f"{'':<42} {' before the jump ':-^60} {' after the jump ':-^33}",
             f"{'filter':<42}{'RMSE':>14}{'CRPS':>14}{'90% cover':>11}{'50% cover':>11}{'90% width':>11}"
             f"{'delay [95% CI]':>16}{f'RMSE ({AFTER} steps)':>18}"]
    for label in FILTERS:
        mt = res["metrics"][label]
        d = mt["delay"]
        delay = f"{d[0]:.0f} [{d[1]:.0f}, {d[2]:.0f}]"
        lines.append(
            f"{label:<42}{mt['rmse'][0]:>7.3f} ±{mt['rmse'][1]:.3f}{mt['crps'][0]:>7.3f} ±{mt['crps'][1]:.3f}"
            f"{100 * mt['cov90'][0]:>10.1f}%{100 * mt['cov50'][0]:>10.1f}%{mt['width90'][0]:>11.2f}"
            f"{delay:>16}{mt['rmse_after'][0]:>12.2f} ±{mt['rmse_after'][1]:.2f}")
    lines += ["", "Paired differences on the same test sequences (first minus second; mean and 95% interval):"]
    for f1, f2 in PAIRS:
        parts = []
        for key, name in (("rmse", "RMSE"), ("crps", "CRPS"), ("rmse_after", "RMSE after the jump")):
            mean, half = paired(res, f1, f2, key)
            parts.append(f"{name} {signed(mean)} ±{half:.4f}")
        db = res["boots"][f1] - res["boots"][f2]
        dd = res["metrics"][f1]["delay"][0] - res["metrics"][f2]["delay"][0]
        parts.append(f"delay {dd:+.0f} [{np.percentile(db, 2.5):+.0f}, {np.percentile(db, 97.5):+.0f}]")
        lines.append(f"  {f1} vs {f2}: " + ", ".join(parts))
    return "\n".join(lines)


TEX_ROWS = [  # (filter, row label) for exp1_table.tex: every filter, in the order of FILTERS
    (ONE_TRUE, r"Kalman, $R = 1$"),
    (ONE_ML, r"Kalman, one $R$ (max.\ likelihood)"),
    (ONE_FC, r"Kalman, one $R$ (forecast CRPS)"),
    (INDEP, r"Kalman, input-dep.\ $R_t$, tuned"),
    (INDEP_FIX, rf"Kalman, input-dep.\ $R_t$, $c = {C_PRIOR:g}$ fixed"),
    (INDEP_PAST, r"Kalman, input-dep.\ $R_t$ (past), tuned"),
    (ROB, r"Robust, tuned"),
    (ROB_FIX, rf"Robust, $c = {C_PRIOR:g}$ fixed"),
    (BOTH, r"Robust + input-dep.\ $R_t$, tuned"),
    (BOTH_PAST, r"Robust + input-dep.\ $R_t$ (past), tuned"),
    (BAYES, r"Bayes filter (reference)"),
]
TEX_GROUP_STARTS = (INDEP, ROB, BAYES)  # a rule above each of these rows


def table_exp1_tex(res):
    rows = [r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r" & \multicolumn{4}{c}{Before the jump} & \multicolumn{2}{c}{After the jump} \\",
            r"\cmidrule(lr){2-5} \cmidrule(l){6-7}",
            r"Filter & RMSE & CRPS & 90\% cover & 90\% width & Delay & RMSE \\", r"\midrule"]
    for label, name in TEX_ROWS:
        if label in TEX_GROUP_STARTS:
            rows.append(r"\midrule")
        mt = res["metrics"][label]
        rows.append(f"{name} & {mt['rmse'][0]:.3f} & {mt['crps'][0]:.3f} & {100 * mt['cov90'][0]:.1f} & "
                    rf"{mt['width90'][0]:.2f} & {mt['delay'][0]:.0f} & {mt['rmse_after'][0]:.2f} \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK2, length=2.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


CURVE_NAME = "robust, $R$ tuned for each $c$"  # legend labels of figure 1 (b) and (c)
INF_NAME = r"$c = \infty$: Kalman, one $R$ (forecast CRPS)"
PRIOR_NAME = f"$c = {C_PRIOR:g}$ fixed"
TUNED_NAME = "tuned $c$"
INDEP_NAME = "Kalman, input-dependent $R_t$, tuned"  # in all three panels
ML_NAME = "Kalman, one $R$ (max. likelihood)"


def figure_exp1(res, cfg, stem):
    ex = res["example"]
    jt = cfg["jump_t"]
    t0, t1 = max(jt - 70, 0), min(jt + 100, cfg["T"])
    t = np.arange(t0, t1) + 1
    x, y, o = ex["x"][t0:t1], ex["y"][t0:t1], ex["outlier"][t0:t1]
    shown = [ex["fits"][lab][0][t0:t1] for lab in (ONE_TRUE, ONE_ML, INDEP, ROB)]
    lo = min(x.min() - 3.5, min(s.min() for s in shown) - 0.5)
    hi = max(x.max() + 3.5, max(s.max() for s in shown) + 0.5)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad  # a strip at each edge for the markers of readings beyond the axis

    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(6.3, 4.75), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, :])
    style_axes(ax)
    ax.axvline(jt + 0.5, color=AXIS, linewidth=0.8)
    ax.text(jt + 1.5, hi - 0.4, "real jump (+5)", color=INK2, va="top", fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    clean = ~o
    h_obs = ax.scatter(t[clean], y[clean], s=6, color=MUTED, linewidths=0, zorder=2)
    inside = o & (y >= lo + pad) & (y <= hi - pad)
    h_out = ax.scatter(t[inside], y[inside], s=16, marker="x", color=INK2, linewidths=0.9, zorder=6)
    for mask, edge, marker in ((o & (y > hi - pad), hi - pad / 2, "^"), (o & (y < lo + pad), lo + pad / 2, "v")):
        if mask.any():
            ax.scatter(t[mask], np.full(mask.sum(), edge), s=16, marker=marker, color=INK2, linewidths=0, zorder=6)
    m_r, P_r = ex["fits"][ROB]
    ax.fill_between(t, (m_r - Z90 * np.sqrt(P_r))[t0:t1], (m_r + Z90 * np.sqrt(P_r))[t0:t1],
                    color=AQUA, alpha=0.18, linewidth=0, zorder=1)
    h_true, = ax.plot(t, x, color=INK, linewidth=1.3, zorder=4)
    h_k1, = ax.plot(t, ex["fits"][ONE_TRUE][0][t0:t1], color=ORANGE, linewidth=1.2, linestyle=(0, (4, 1.5)), zorder=5,
                    label=ONE_TRUE)
    h_ml, = ax.plot(t, ex["fits"][ONE_ML][0][t0:t1], color=BLUE, linewidth=1.3, zorder=5, label=ONE_ML)
    h_in, = ax.plot(t, ex["fits"][INDEP][0][t0:t1], color=INK2, linewidth=1.0, linestyle=(0, (4, 2)), zorder=5,
                    label=INDEP)
    h_rob, = ax.plot(t, m_r[t0:t1], color=AQUA, linewidth=1.3, zorder=5, label=ROB)
    h_edge = (Line2D([], [], marker="^", linestyle="none", color=INK2, markersize=4),
              Line2D([], [], marker="v", linestyle="none", color=INK2, markersize=4))
    h_band = (Patch(facecolor=AQUA, alpha=0.18, edgecolor="none"), h_rob)
    # three rows (data; truth and the one-R Kalman filters; input-dependent and robust),
    # listed column by column because that is how matplotlib fills a legend
    handles = [h_obs, h_true, h_in, h_out, h_k1, h_band, h_edge, h_ml, Line2D([], [], linestyle="none")]
    labels = ["reading $y_t$", "true state $x_t$", INDEP_NAME, "outlier", "Kalman, $R = 1$",
              "robust, tuned (90% band)", "outlier beyond the axis", ML_NAME, ""]
    # anchored a fixed 22 pt below the axes, so the legend clears the x-axis label at any panel height
    ax.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0),
              bbox_transform=ax.transAxes + ScaledTranslation(0, -22 / 72, fig.dpi_scale_trans),
              columnspacing=0.8, handlelength=1.6, handletextpad=0.5,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    ax.set_ylim(lo, hi)
    # ticks (and their gridlines) well clear of the edge strips, so an off-axis marker never sits on a value
    ax.set_yticks([v for v in MaxNLocator(nbins=5).tick_values(lo + pad, hi - pad)
                   if lo + 1.5 * pad <= v <= hi - 1.5 * pad])
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("step $t$")
    ax.set_ylabel("state and readings")
    ax.set_title("(a) the first test sequence around the jump", loc="left")

    curve = [d for d in res["curve"] if np.isfinite(d["c"])]
    inf_pt = next(d for d in res["curve"] if not np.isfinite(d["c"]))
    cs = np.array([d["c"] for d in curve])
    x_inf = 26.0  # where c = inf is drawn, after a gap
    panels = (("rmse", "(b) RMSE before the jump"), ("delay", "(c) delay after the jump (steps)"))
    for j, (key, title) in enumerate(panels):
        axb = fig.add_subplot(gs[1, j])
        style_axes(axb)
        refs = ((ONE_ML, BLUE, "-", ML_NAME),
                (INDEP, INK2, (0, (4, 2)), INDEP_NAME),
                (BAYES, INK, "-", "Bayes filter (reference)"))
        for label, col, ls, name in refs:
            axb.axhline(res["metrics"][label][key][0], color=col, linewidth=1.0, linestyle=ls, label=name, zorder=2)
        vals = [d[key] for d in curve]
        axb.plot(cs, vals, color=AQUA, linewidth=1.5, marker="o", markersize=3.2, label=CURVE_NAME, zorder=3)
        axb.plot([cs[-1], x_inf], [vals[-1], inf_pt[key]], color=AQUA, linewidth=1.0, linestyle=":", zorder=3)
        axb.plot([x_inf], [inf_pt[key]], marker="o", markersize=3.6, markerfacecolor="white", color=AQUA,
                 linestyle="none", label=INF_NAME, zorder=4)
        for c_mark, marker, name in ((C_PRIOR, "s", PRIOR_NAME), (res["fitted"]["c_rob"], "D", TUNED_NAME)):
            x_mark = c_mark if np.isfinite(c_mark) else x_inf
            v = next(d[key] for d in res["curve"] if d["c"] == c_mark)
            axb.plot([x_mark], [v], marker=marker, markersize=6 / np.sqrt(2) if marker == "D" else 6,
                     markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.9, linestyle="none", label=name,
                     zorder=5)  # matplotlib's "D" is sqrt(2) wider than "s" at the same size
        axb.set_xscale("log")
        ticks = [0.5, 1, 2, 4, 8, 15, x_inf]
        axb.set_xticks(ticks)
        axb.set_xticklabels(["0.5", "1", "2", "4", "8", "15", r"$\infty$"])
        axb.minorticks_off()
        axb.set_xlabel("robust scale $c$ (smaller = more robust)")
        axb.set_title(title, loc="left")
        top = max(max(vals), inf_pt[key], *(res["metrics"][lab][key][0] for lab, *_ in refs))
        if key == "rmse":
            bottom = min(min(vals), res["metrics"][BAYES][key][0])
            axb.set_ylim(max(0.0, bottom - 0.15 * (top - bottom)), top + 0.08 * (top - bottom))
        else:
            axb.set_ylim(0, 1.08 * top)
            axb.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        if j == 0:
            handles, labels = axb.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
    # one legend for (b) and (c), below them, in three rows: the robust curve and its marks;
    # the reference lines; the c = inf point. Listed column by column, as matplotlib fills it.
    layout = [[CURVE_NAME, PRIOR_NAME, TUNED_NAME], [name for *_, name in refs], [INF_NAME, None, None]]
    blank = Line2D([], [], linestyle="none")
    cells = [layout[r][col] for col in range(3) for r in range(len(layout))]
    fig.legend([by_label[name] if name else blank for name in cells], [name or "" for name in cells],
               loc="outside lower center", ncol=3, frameon=False, columnspacing=1.4, handlelength=1.8)
    fig.draw_without_rendering()  # let the constrained layout settle, so the PDF has the same layout as the PNG
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def experiment1(cfg, out_dir, cache):
    log("experiment 1: main scenario")
    res = run_scenario(0, cfg, extra=True)
    cache[0] = res
    text = [scenario_text(res), "",
            "Trade-off over c (robust filter with R tuned for each c; c = inf is the one-R Kalman filter "
            "tuned by forecast CRPS; test sequences):",
            f"{'c':>6}{'R':>8}{'RMSE':>8}{'CRPS':>8}{'90% cover':>11}{'delay':>8}"]
    for d in res["curve"]:
        text.append(f"{d['c']:>6g}{d['R']:>8.3g}{d['rmse']:>8.3f}{d['crps']:>8.3f}{100 * d['cov90']:>10.1f}%"
                    f"{d['delay']:>8.0f}")
    text += ["", f"Tuning stability: everything refitted on {len(res['stability'])} fresh sets of training and "
             "validation sequences, scored on the same test sequences (RMSE before the jump / delay):"]
    for r, row in enumerate(res["stability"], start=1):
        f = row["fit"]
        text.append(f"set {r}: robust R = {f['R_rob']:.3g}, c = {fmt_c(f['c_rob'])}; input-dependent R = "
                    f"{f['R_in']:.3g}, c = {fmt_c(f['c_in'])}, k = {fmt_k(f['k_in'])}; robust + input-dependent R = "
                    f"{f['R_both']:.3g}, robust c = {fmt_c(f['c_both'])}, noise c = {fmt_c(f['c_in_both'])}, "
                    f"k = {fmt_k(f['k_both'])}; one R = {f['R_ml']:.3g}")
        for label in (ONE_ML, INDEP, ROB, BOTH):
            text.append(f"    {label:<34} {row[label]['rmse']:.3f} / {row[label]['delay']:.0f}")
    text += ["", "Notes"] + [f"- {note}" for note in table_notes(cfg)]
    (out_dir / "exp1_table.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    (out_dir / "exp1_table.tex").write_text(table_exp1_tex(res), encoding="utf-8")
    figure_exp1(res, cfg, out_dir / "exp1_figure")
    write_json(out_dir / "exp1_results.json", results_json(res, ("name", "fitted", "metrics", "curve", "stability")))
    log("experiment 1 written: exp1_table.txt/.tex, exp1_figure.pdf/.png")


def table_notes(cfg, sweep=False):
    """The notes printed under the experiment 1 table, and with sweep = True under the sweep."""
    return [
        f"RMSE and CRPS: means over {cfg['n_test']} test sequences of {cfg['T']} steps, ± the half-width of a 95% "
        f"interval for the mean. Coverage and width: means. All over steps 1-{cfg['jump_t']} (before the jump).",
        f"Delay: steps after the jump until the error averaged over sequences is below {NEAR:g} (80% of the jump "
        "recovered); 0 means already at the first step after it. Its interval resamples sequences. "
        f"RMSE after the jump: steps {cfg['jump_t'] + 1}-{cfg['jump_t'] + AFTER}.",
        "The ± in a row is for that row alone. Every row is scored on the same test sequences, and each paired line "
        "gives a difference its own interval. Between filters that behave alike, such as the robust filter and "
        "input-dependent noise from y_t, that interval is often much narrower than either row's (about ±0.001 in "
        "RMSE before the jump; wider with bursts or a fast state). Between filters that behave differently, such as "
        "the robust filter and one R, it can be as wide as the narrower row's ±, or wider; so can the past-only "
        "pair, which is the robust filter against one R wherever tuning switched its noise off. Judge differences "
        "by the paired lines, not by the printed digits.",
        f"Parameters were fitted on {cfg['n_train']} training and {cfg['n_val']} validation sequences without a jump, "
        "using observations only (see the docstring): the one-R maximum-likelihood filter by the likelihood of the "
        "training sequences, every other tuned filter by the CRPS of the one-step forecast "
        "N(m_{t|t-1}, P_{t|t-1} + R), with the base R, on the validation sequences. Every filter knows Q and the "
        "prior on x_0. Kalman, R = 1 is given the true clean noise level.",
        "The Bayes filter knows the true outlier model, including the test-time outlier rate when it differs from "
        "the training data, but not the jump, so it has the lowest expected error before the jump only."
        + (" Where its model has outliers, it first takes the jump for outliers, like the robust and input-dependent "
           "filters; with bursts, which it expects to come in runs, it does so for longer." if sweep else ""),
        "The input-dependent noise R_t = R + d_t^2 / c^2 equals R / w_t^2 for an IMQ weight on d_t, the distance from "
        "y_t to the median of the previous k readings; the robust weight uses e_t, the distance from the filter's "
        "prediction.",
        "(past only): R_t = R + d_{t-1}^2 / c^2, which uses y_1, ..., y_{t-1} only. It cannot tell whether y_t itself "
        "is an outlier, only expect one after an outlier, which can help when outliers come in bursts. Its forecast "
        "N(m_{t|t-1}, P_{t|t-1} + R_t) is a proper density for y_t, which it would not be with d_t in place of "
        "d_{t-1}; but like every filter it was tuned on the forecast with the base R.",
        f"With c = {C_PRIOR:g} fixed the two weights are not equally strong: d_t is measured in units of sqrt(R) and "
        "e_t in units of sqrt(P_{t|t-1} + R), and in those units d_t is the larger, so the input-dependent weight "
        "down-weights more. Their paired line mixes the mechanism with the strength.",
        "Robust + input-dependent R_t, both versions: R, the robust c and the noise's c and k were tuned together, "
        "over a grid that includes both filters alone. The weight is measured against P_{t|t-1} + R_t. Noise from "
        "d_t already grows with the reading, so where e_t is close to d_t the weight stays above about "
        "(1 + noise c^2 / robust c^2)^(-1/2); noise from past readings leaves the weight free.",
    ]


def results_json(res, keys):
    """What is saved of one scenario: the listed parts, plus per-sequence values and the
    bootstrap delays, so every interval can be recomputed."""
    keep = {key: res[key] for key in keys}
    keep["per_sequence"] = {lab: {key: res["per_seq"][lab][key] for key in ("rmse", "crps", "rmse_after")}
                            for lab in FILTERS}
    keep["bootstrap_delays"] = {lab: res["boots"][lab] for lab in FILTERS}
    return keep


SWEEP_COLS = [(ONE_ML, r"one $R$"), (INDEP, r"input-dep."), (ROB, r"robust"), (BOTH, r"both"),
              (INDEP_FIX, r"input-dep."), (ROB_FIX, r"robust"), (BAYES, r"Bayes")]


def sweep_tex(results, key, fmt, title):
    head = [r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
            title + r" & \multicolumn{4}{c}{Tuned} & \multicolumn{2}{c}{" + f"$c = {C_PRIOR:g}$ fixed" + r"} & \\",
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-7}",
            "Setting & " + " & ".join(name for _, name in SWEEP_COLS) + r" \\", r"\midrule"]
    body = [SHORT_TEX[k] + " & " + " & ".join(fmt(res["metrics"][lab][key][0]) for lab, _ in SWEEP_COLS) + r" \\"
            for k, res in results]
    return "\n".join(head + body + [r"\bottomrule", r"\end{tabular}"]) + "\n"


WEIGHT_PAIRS = [(BOTH, INDEP), (BOTH_PAST, INDEP_PAST)]  # the columns of sweep_weight_table.tex


def weight_tex(results):
    """What adding the robust weight to input-dependent noise did to the RMSE before the jump,
    for noise from y_1..y_t and from y_1..y_{t-1}: the paired difference (robust + noise minus
    the noise alone, each tuned) with its 95% interval, and as a share of the noise-alone RMSE.
    The share takes the sign of the printed difference, and a one-digit share is padded by a
    digit's width so that every row lines up."""
    def cell(res, f1, f2):
        mean, half = paired(res, f1, f2, "rmse")
        share = 100 * mean / res["metrics"][f2]["rmse"][0]
        m = signed(mean)
        pad = r"\phantom{0}" if abs(share) < 9.95 else ""
        return rf"${m} \pm {half:.4f}$ (${m[0]}{pad}{abs(share):.1f}\%$)"

    head = [r"\small", r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
            r"Change in RMSE & \multicolumn{2}{c}{Input-dependent $R_t$ computed from} \\", r"\cmidrule(l){2-3}",
            r"Setting & $y_1, \dots, y_t$ & $y_1, \dots, y_{t-1}$ \\", r"\midrule"]
    body = [SHORT_TEX[k] + " & " + " & ".join(cell(res, f1, f2) for f1, f2 in WEIGHT_PAIRS) + r" \\"
            for k, res in results]
    return "\n".join(head + body + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def sweep(cfg, out_dir, cache):
    blocks, keep, done = [], [], []
    for k in cfg["scenarios"]:
        log(f"sweep: {SCENARIOS[k][0]}")
        res = cache.get(k) or run_scenario(k, cfg)
        blocks.append(scenario_text(res))
        keep.append(results_json(res, ("name", "fitted", "metrics")))
        done.append((k, res))
        # write after every scenario, so a stopped run still leaves the finished ones
        notes = "Notes\n" + "\n".join(f"- {note}" for note in table_notes(cfg, sweep=True))
        (out_dir / "sweep.txt").write_text("\n\n".join(blocks + [notes]) + "\n", encoding="utf-8")
        write_json(out_dir / "sweep_results.json", keep)
        (out_dir / "sweep_rmse_table.tex").write_text(
            sweep_tex(done, "rmse", lambda v: f"{v:.3f}", "RMSE before the jump"), encoding="utf-8")
        (out_dir / "sweep_delay_table.tex").write_text(
            sweep_tex(done, "delay", lambda v: f"{v:.0f}", "Delay after the jump"), encoding="utf-8")
        (out_dir / "sweep_weight_table.tex").write_text(weight_tex(done), encoding="utf-8")
    log("sweep written: sweep.txt, sweep_rmse_table.tex (RMSE before the jump), sweep_delay_table.tex, "
        "sweep_weight_table.tex (the robust weight added to each kind of input-dependent noise)")


# ---------------------------------------------------------------- experiment 2
def frozen_passes(y, Qs, a, c):
    """The repeated frozen-weight iteration of proposal Section 5.2. Yields the filter
    output of pass 1 (all weights 1), pass 2, and so on, without end."""
    w = np.ones_like(y)
    while True:
        run = kalman_filter(y, R2, Qs, a, w_fixed=w, P0=SIGMA2)
        yield run
        w = imq_weight(y - run["m_pred"], run["P_pred"] + R2, c)


def count_passes(y, Qs, a, c, maxpass):
    """Passes of the repeated iteration against the sequential robust filter.

    For each tolerance in TOLS: the passes each sequence needs before every filtered mean
    is within TOL posterior sd of the sequential filter, and the passes a practical stopping
    rule needs (stop once no mean moved by more than TOL sd since the previous pass); -1 if
    not reached within maxpass. Also the error after each pass; the contraction factor, the
    geometric-mean shrink of the error per pass from pass 2 to the first pass within the
    tightest tolerance; and the filter's memory a_{t+1} (1 - K_t) along the sequential run
    (proposal l.130's A_{t+1}(I - K_t H): how much of the prediction for step t is still in
    the prediction for step t + 1), for t = 1, ..., T - 1.
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

    def first(arr, tol):
        hit = arr < tol
        return np.where(hit.any(axis=0), hit.argmax(axis=0) + 1, -1)

    passes = {tol: first(errs, tol) for tol in TOLS}
    practical = {tol: first(moves, tol) for tol in TOLS}
    factor = np.full(len(y), np.nan)
    for i, Ki in enumerate(passes[min(TOLS)]):
        if Ki > 2 and errs[1, i] > 0:
            factor[i] = (errs[Ki - 1, i] / errs[1, i]) ** (1 / (Ki - 2))
    K = seq["P_pred"] / (seq["P_pred"] + R2 / seq["w"] ** 2)
    return dict(passes=passes, practical=practical, errs=errs, factor=factor, memory=a[:, 1:] * (1 - K[:, :-1]))


def fmt_passes(p, maxpass):
    pc = np.where(p < 0, maxpass, p)
    med = np.median(pc)
    lead = ">=" if (p < 0).sum() * 2 >= len(p) else ""
    return f"{lead}{med:g} ({'>=' if (p < 0).any() else ''}{pc.max()})"


def experiment2(cfg, out_dir):
    mid = cfg["T2_mid"]
    rows, curves = [], {}
    for label, a_med, cap, seed in REGIMES:
        row = dict(regime=label)
        for T in cfg["T2_list"]:
            maxpass = min(cfg["maxpass"], T + 2)
            x, y, a, Qs = simulate_selective(np.random.default_rng([2027, seed, T]), cfg["n2"], T, a_med, cap)
            assert cap is None or a.max() <= cap
            for c in ((C2, 1.0) if T == mid else (C2,)):
                t_start = time.time()
                res = count_passes(y, Qs, a, c, maxpass)
                key = f"T={T} c={c:g}"
                row[key] = dict(maxpass=maxpass,
                                passes={str(tol): res["passes"][tol] for tol in TOLS},
                                practical={str(tol): res["practical"][tol] for tol in TOLS},
                                factor=float(np.nanmedian(res["factor"])) if np.isfinite(res["factor"]).any()
                                else float("nan"),
                                memory=float(np.median(res["memory"])),
                                a=float(np.median(a)), a_max=float(a.max()))
                if T == mid and c == C2:
                    curves[label] = np.median(res["errs"], axis=1)
                log(f"experiment 2: {label}, T = {T}, c = {c:g}: passes to 1% sd "
                    f"{fmt_passes(res['passes'][TOLS[0]], maxpass)}, to 1e-8 sd "
                    f"{fmt_passes(res['passes'][TOLS[1]], maxpass)} [{time.time() - t_start:.0f}s]")
        rows.append(row)
        write_json(out_dir / "exp2_results.json", rows)

    Ts = cfg["T2_list"]
    lines = ["Passes of the repeated frozen-weight iteration needed to match the sequential robust filter:",
             f"median (max) over {cfg['n2']} sequences, weight of proposal eq. (3) with c = {C2:g}.",
             "A pass is one frozen-weight Kalman filter, i.e. one parallel scan; pass 1 has every weight 1.",
             f"memory: median over steps and sequences of a_{{t+1}} (1 - K_t), the filter's memory per step, at T = {mid}.",
             "factor: median over sequences of the geometric-mean shrink of the largest error per pass, from pass 2 "
             f"to the first pass within 1e-8 sd, at T = {mid}. It rises with T.",
             "", f"{'regime':<13}{'a_t':>7}{'max a_t':>9}{'memory':>8}{'factor':>8}"
             + "".join(f"{'1% T=' + str(T):>14}" for T in Ts) + "".join(f"{'1e-8 T=' + str(T):>15}" for T in Ts)]
    for row in rows:
        m = row[f"T={mid} c={C2:g}"]
        cells = [f"{row['regime']:<13}{m['a']:>7.3f}{m['a_max']:>9.3f}{m['memory']:>8.3f}{m['factor']:>8.3f}"]
        for tol, width in ((TOLS[0], 14), (TOLS[1], 15)):
            for T in Ts:
                r = row[f"T={T} c={C2:g}"]
                cells.append(f"{fmt_passes(np.array(r['passes'][str(tol)]), r['maxpass']):>{width}}")
        lines.append("".join(cells))
    lines += ["", f"Practical stopping rule (stop once no mean moves by more than the tolerance), c = {C2:g}: "
              "extra passes beyond the counts above, median (max) over all regimes, lengths and sequences:"]
    for tol, name in zip(TOLS, ("1%", "1e-8")):
        extra = np.concatenate([np.array(row[f"T={T} c={C2:g}"]["practical"][str(tol)])
                                - np.array(row[f"T={T} c={C2:g}"]["passes"][str(tol)]) for row in rows for T in Ts])
        valid = np.concatenate([(np.array(row[f"T={T} c={C2:g}"]["practical"][str(tol)]) > 0)
                                & (np.array(row[f"T={T} c={C2:g}"]["passes"][str(tol)]) > 0) for row in rows for T in Ts])
        note = "" if valid.all() else f" ({(~valid).sum()} sequences not converged, left out)"
        value = f"{np.median(extra[valid]):g} ({extra[valid].max()})" if valid.any() else "n/a"
        lines.append(f"  tolerance {name} of sd: {value}{note}")
    lines += ["", f"Stronger down-weighting, c = 1, at T = {mid}: passes to 1% sd / to 1e-8 sd"]
    for row in rows:
        r = row[f"T={mid} c=1"]
        lines.append(f"  {row['regime']:<13}{fmt_passes(np.array(r['passes'][str(TOLS[0])]), r['maxpass']):>10}"
                     f"{fmt_passes(np.array(r['passes'][str(TOLS[1])]), r['maxpass']):>12}")
    lines += ["", "Tolerances are relative to the posterior sd. In float64 the iteration keeps converging until it "
              "matches the sequential filter exactly; 1e-8 is a strict target, about the square root of float64 "
              "precision. In float32, usual on a GPU, the error stalls at about 1e-6 sd, so only the 1% counts "
              "carry over to a GPU.",
              "Delta_t follows an exogenous input here; in the proposal it is computed from the observations. "
              "Neither depends on the weights, so the pass structure is the same."]
    (out_dir / "exp2_table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def tex_name(label):
        if label == "random walk":
            return "Random walk"
        return "$" + label.replace("<=", r"\le").replace("~", r"\approx") + "$"

    t_head = [f"{T:,}".replace(",", "{,}") for T in Ts]
    t_cells = [f"$T = {t_head[0]}$"] + [f"${T}$" for T in t_head[1:]]
    tex = [r"\small", r"\begin{tabular}{@{}lrr" + "r" * (2 * len(Ts)) + r"@{}}", r"\toprule",
           r" & & & \multicolumn{" + str(len(Ts)) + r"}{c}{Passes to 1\% of sd} & \multicolumn{"
           + str(len(Ts)) + r"}{c}{Passes to $10^{-8}$ sd} \\",
           rf"\cmidrule(lr){{4-{3 + len(Ts)}}} \cmidrule(l){{{4 + len(Ts)}-{3 + 2 * len(Ts)}}}",
           r"Dynamics & Memory & Factor & " + " & ".join(t_cells) + " & " + " & ".join(t_cells) + r" \\", r"\midrule"]
    for row in rows:
        m = row[f"T={mid} c={C2:g}"]
        cells = [fmt_passes(np.array(row[f"T={T} c={C2:g}"]["passes"][str(tol)]), row[f"T={T} c={C2:g}"]["maxpass"])
                 .replace(">=", r"$\ge$") for tol in TOLS for T in Ts]
        tex.append(f"{tex_name(row['regime'])} & {m['memory']:.3f} & {m['factor']:.3f} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "exp2_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.95), layout="constrained")
    colours = RAMP + [MUTED]
    markers = ["o", "s", "^", "D", "v", "P"]
    names = {r[0]: ("random walk ($a_t = 1$)" if r[0] == "random walk" else
                    "$" + r[0].replace("<=", r"\leq").replace("~", r"\approx") + "$") for r in REGIMES}
    ax = axes[0]

    def msize(mk, size):  # matplotlib's "D" is sqrt(2) wider than the other markers at the same size
        return size / np.sqrt(2) if mk == "D" else size

    for i, ((label, *_), col, mk) in enumerate(zip(REGIMES, colours, markers)):
        e = np.maximum(curves[label], 1e-13)
        # markers on every third pass, starting one pass later for each regime, and drawn again above
        # every line, so curves that coincide still show each regime's markers
        ax.plot(np.arange(1, len(e) + 1), e, color=col, linewidth=1.3, marker=mk, markersize=msize(mk, 2.6),
                markevery=(i % 3, 3))
        ax.plot(np.arange(1, len(e) + 1), e, color=col, linestyle="none", marker=mk, markersize=msize(mk, 2.6),
                markevery=(i % 3, 3), zorder=3)
    style_axes(ax)
    kmax = len(max(curves.values(), key=len))
    for tol, x_text, ha, text in ((TOLS[0], kmax, "right", "1% of sd"), (TOLS[1], 1, "left", "$10^{-8}$ sd")):
        ax.axhline(tol, color=AXIS, linewidth=0.8)
        ax.text(x_text, tol * 2.5, text, color=INK2, fontsize=7.5, ha=ha)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, None)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("pass $k$")
    ax.set_ylabel("largest error, in posterior sd\n(median over sequences)")
    ax.set_title("(a) error after each pass", loc="left")
    ax.text(0.97, 0.97, f"$T = {mid:,}$".replace(",", "{,}"), transform=ax.transAxes, ha="right", va="top",
            color=INK2, fontsize=7.5)
    for axb, tol, title in ((axes[1], TOLS[0], "(b) passes to 1% of sd"), (axes[2], TOLS[1], "(c) passes to $10^{-8}$ sd")):
        for i, ((label, *_), col, mk) in enumerate(zip(REGIMES, colours, markers)):
            r = next(rw for rw in rows if rw["regime"] == label)
            med = [np.median(np.where(np.array(r[f"T={T} c={C2:g}"]["passes"][str(tol)]) < 0,
                                      r[f"T={T} c={C2:g}"]["maxpass"],
                                      r[f"T={T} c={C2:g}"]["passes"][str(tol)])) for T in Ts]
            dodge = 10 ** (0.06 * (i - 2.5))  # sideways offset of about 2.6 pt per regime, so equal medians stay visible
            axb.plot(np.array(Ts) * dodge, med, color=col, linewidth=1.3, marker=mk, markersize=msize(mk, 3.6),
                     label=names[label])
            # the markers again, above every regime's line, so no other line runs through them
            axb.plot(np.array(Ts) * dodge, med, color=col, linestyle="none", marker=mk, markersize=msize(mk, 3.6),
                     zorder=3)
        style_axes(axb)
        axb.set_xscale("log")
        axb.set_xticks(Ts)
        axb.set_xticklabels([f"{T:,}" for T in Ts])
        axb.minorticks_off()
        axb.yaxis.set_major_locator(MaxNLocator(integer=True))
        axb.set_xlabel("sequence length $T$")
        axb.set_ylabel("passes (median over sequences)")
        axb.set_title(title, loc="left")
        axb.set_ylim(bottom=0)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, columnspacing=1.0,
               handlelength=1.6, handletextpad=0.4)
    fig.savefig((out_dir / "exp2_figure").with_suffix(".pdf"))
    fig.savefig((out_dir / "exp2_figure").with_suffix(".png"), dpi=200)
    plt.close(fig)
    log("experiment 2 written: exp2_table.txt/.tex, exp2_figure.pdf/.png")


# ---------------------------------------------------------------- main
def main():
    global _LOG, _POOL, _WORKERS
    ap = argparse.ArgumentParser(description="Toy experiments for the proposal (Section 4).")
    ap.add_argument("--quick", action="store_true", help="small, fast version to check the code runs")
    ap.add_argument("--only", choices=["checks", "exp1", "exp2", "sweep"], help="run one part only")
    ap.add_argument("--workers", type=int, default=WORKERS_DEFAULT,
                    help=f"processes for the tuning and the Bayes filter (default {WORKERS_DEFAULT}; same numbers)")
    args = ap.parse_args()
    cfg = QUICK if args.quick else FULL
    out_dir = HERE / ("results_quick" if args.quick else "results_partial" if args.only else "results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "DONE.txt").unlink(missing_ok=True)
    _LOG = out_dir / "run_log.txt"
    _LOG.write_text("", encoding="utf-8")
    info = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                quick=args.quick, only=args.only, workers=args.workers,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())  # the file that made these results
    write_json(out_dir / "run_info.json", info)

    t_start = time.time()
    cache = {}
    parts = [args.only] if args.only else ["checks", "exp1", "exp2", "sweep"]
    _WORKERS = max(1, args.workers)
    _POOL = ProcessPoolExecutor(_WORKERS) if _WORKERS > 1 else None
    try:
        for part in parts:
            if part == "checks":
                self_checks(out_dir)
            elif part == "exp1":
                experiment1(cfg, out_dir, cache)
            elif part == "exp2":
                experiment2(cfg, out_dir)
            elif part == "sweep":
                sweep(cfg, out_dir, cache)
    finally:
        if _POOL is not None:
            _POOL.shutdown()
    minutes = (time.time() - t_start) / 60
    log(f"finished in {minutes:.1f} minutes")
    if args.only is None:
        (out_dir / "DONE.txt").write_text(f"Run of every part finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
                                          f"in {minutes:.1f} minutes ({'quick' if args.quick else 'full'}).\n",
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
