"""Options, the session's worker processes, and data and runs that several test files share."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from toy_experiment import output, parallel
from toy_experiment.config import EPS, LAM, M0, P0
from toy_experiment.experiments import experiment2, run_scenario
from toy_experiment.simulate import simulate_rw

from .helpers import SMALL_EXTRA, TINY_EXP2, capture_figures


def pytest_addoption(parser):
    parser.addoption("--workers", type=int, default=1,
                     help="worker processes for the session; 2 or more also runs the two pool tests")
    parser.addoption("--run-info", default=None, help="run_info.json of the current run, whose hashes are checked")


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_workers: compares worker processes with one process")


def pytest_collection_modifyitems(config, items):
    # decided at collection, so test_record can check which tests are skipped
    if config.getoption("workers") < 2:
        skip = pytest.mark.skip(reason="needs --workers 2 or more")
        for item in items:
            if item.get_closest_marker("needs_workers"):
                item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def worker_processes(request):
    """As in a run, the code under test uses the worker processes when there are any."""
    with parallel.worker_pool(request.config.getoption("workers")):
        yield


@pytest.fixture(scope="session", autouse=True)
def no_log_file():
    """The code under test may log, but not into the log file of a run."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(output, "_log_file", None)
        yield


@pytest.fixture(scope="session")
def shared():
    """Data for the filter and Bayes-filter tests, drawn from one generator in a fixed order,
    so every test sees the same data whichever tests are run."""
    rng = np.random.default_rng(1)
    # a short model with time-varying a_t, Q_t and two outlying readings
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
    varying = dict(y=ys, a=a, Q=Qs, R=Rv, R_t=Rt)
    _, y_rw, _ = simulate_rw(rng, 5, 300, 0.1, EPS, LAM, None, None)
    patterns = {}  # short sequences with a large outlier at step 5, for the enumeration of outlier patterns
    for kind, stay in (("independent outliers", None), ("bursty outliers", 0.8)):
        x, y, _ = simulate_rw(rng, 3, 10, 0.1, 0.3, 100.0, stay, None)
        y[:, 4] += 15.0
        patterns[kind] = (x, y)
    clean = {Q: simulate_rw(rng, 5, 300, Q, 0.0, LAM, None, 150)[:2] for Q in (0.1, 1.0)}
    return SimpleNamespace(varying=varying, y_rw=y_rw, patterns=patterns, clean=clean)


@pytest.fixture(scope="session")
def small_main_run():
    """Experiment 1's main scenario with the trade-off curve and two refits, on short sequences."""
    return run_scenario(0, SMALL_EXTRA, extra=True)


@pytest.fixture(scope="session")
def tiny_exp2(tmp_path_factory):
    """Experiment 2 on a tiny configuration, with what it wrote and its figure."""
    out = tmp_path_factory.mktemp("exp2")
    with capture_figures() as cap:
        experiment2(TINY_EXP2, out)
    return SimpleNamespace(rows=json.loads((out / "exp2_results.json").read_text(encoding="utf-8")),
                           text=(out / "exp2_table.txt").read_text(encoding="utf-8"),
                           tex=(out / "exp2_table.tex").read_text(encoding="utf-8"),
                           fig=cap.figs[0])
