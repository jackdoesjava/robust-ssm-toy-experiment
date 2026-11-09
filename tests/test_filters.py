"""The Kalman and robust filters, the IMQ weight, and the local distances."""

import numpy as np
import pytest

from toy_experiment.bayes import normal_logpdf
from toy_experiment.config import K_GRID, M0, P0, R
from toy_experiment.filters import imq_weight, kalman_filter, local_distance2, past_distance2

from .helpers import crps_ref, d2_ref, past_ref, scalar_filter


def test_imq_weight_at_one_predictive_sd():
    """A residual of one predictive sd with c = 2 gets the weight 1/sqrt(1.25)."""
    w = imq_weight(np.array(1.0), np.array(1.0), 2.0)
    assert abs(w - 1 / np.sqrt(1.25)) < 1e-15


def exact_gaussian(y, a, Qs, Rv, R_t, kf):
    """Largest differences between the filter output kf and exact Gaussian conditioning on
    the stacked states and readings: filtered means, variances, and the mean CRPS and summed
    log density of the one-step forecast of y_t with the base R."""
    n, Tn = y.shape
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
            sol = np.linalg.solve(Cy[:t + 1, :t + 1], y[i, :t + 1] - mx[:t + 1])
            dm = max(dm, abs(mx[t] + cxy @ sol - kf["m"][i, t]))
            dP = max(dP, abs(Cx[t, t] - cxy @ np.linalg.solve(Cy[:t + 1, :t + 1], cxy) - kf["P"][i, t]))
            if t == 0:
                mu, Pp = mx[0], Cx[0, 0]
            else:
                cp = Cx[t, :t]
                mu = mx[t] + cp @ np.linalg.solve(Cy[:t, :t], y[i, :t] - mx[:t])
                Pp = Cx[t, t] - cp @ np.linalg.solve(Cy[:t, :t], cp)
            crps += crps_ref(y[i, t], mu, np.sqrt(Pp + Rv)) / Tn
            ll += normal_logpdf(y[i, t], mu, Pp + Rv)
        dc = max(dc, abs(crps - kf["fc_crps"][i]))
        dl = max(dl, abs(ll - kf["loglik"][i]))
    return dict(mean=dm, var=dP, crps=dc, loglik=dl)


@pytest.mark.parametrize("noise", ["one R", "R_t in the update"])
def test_kalman_filter_equals_exact_gaussian_conditioning(shared, noise):
    """Means and variances with time-varying a_t, Q_t; the forecast CRPS and log-likelihood
    with the base R."""
    v = shared.varying
    R_t = None if noise == "one R" else v["R_t"]
    kf = kalman_filter(v["y"], v["R"], v["Q"], v["a"], R_t=R_t, score=True)
    d = exact_gaussian(v["y"], v["a"], v["Q"], v["R"], R_t, kf)
    assert d["mean"] < 1e-10
    assert d["var"] < 1e-10
    assert d["crps"] < 1e-10
    assert d["loglik"] < 1e-8


@pytest.mark.parametrize("noise", ["one R", "R_t"])
def test_robust_filter_equals_an_information_form_update(shared, noise):
    """With input-dependent R_t the weight is measured against P_pred + R_t."""
    v = shared.varying
    R_t = None if noise == "one R" else v["R_t"]
    rob = kalman_filter(v["y"], v["R"], v["Q"], v["a"], c=1.5, R_t=R_t)
    ref = scalar_filter(v["y"], v["R"], v["Q"], v["a"], c=1.5, R_t=R_t)
    d = max(np.abs(rob[k] - ref[k]).max() for k in ("m", "P", "w"))
    assert d < 1e-12


def test_robust_filter_with_c_inf_is_the_kalman_filter(shared):
    d = np.abs(kalman_filter(shared.y_rw, R, 0.1, c=np.inf)["m"] - kalman_filter(shared.y_rw, R, 0.1)["m"]).max()
    assert d == 0


def test_robust_filter_with_c_1e9_is_within_1e_9_of_the_kalman_filter(shared):
    d = np.abs(kalman_filter(shared.y_rw, R, 0.1, c=1e9)["m"] - kalman_filter(shared.y_rw, R, 0.1)["m"]).max()
    assert d < 1e-9


def test_kalman_predictive_variance_reaches_the_steady_state(shared):
    """The Riccati fixed point for a random walk: P = (Q + sqrt(Q^2 + 4 Q R)) / 2."""
    d = abs(kalman_filter(shared.y_rw, R, 0.1)["P_pred"][0, -1] - (0.1 + np.sqrt(0.01 + 0.4 * R)) / 2)
    assert d < 1e-10


@pytest.fixture(scope="module")
def readings():
    return np.random.default_rng(11).normal(size=(3, 40)) * 3.0


def test_local_distance2_equals_a_direct_loop(readings):
    """The prior mean stands in before the first readings."""
    d = max(np.abs(local_distance2(readings, k) - d2_ref(readings, k)).max() for k in K_GRID)
    assert d < 1e-12


def test_past_distance2_equals_a_direct_loop(readings):
    """y_{t-1} against the median of the k readings before it, and 0 at the first step."""
    d = max(np.abs(past_distance2(readings, k) - past_ref(readings, k)).max() for k in K_GRID)
    assert d < 1e-12


def test_past_distance2_never_uses_the_current_reading(readings):
    """Changing y_t by 100 moves nothing past_distance2 gives at steps up to t."""
    d = 0.0
    for t in (0, 5, 17, 39):
        y_changed = readings.copy()
        y_changed[:, t] += 100.0
        d = max(d, max(np.abs(past_distance2(y_changed, k)[:, :t + 1] - past_distance2(readings, k)[:, :t + 1]).max()
                       for k in K_GRID))
    assert d == 0
