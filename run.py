"""Run the toy experiments: the tests, then experiment 1, experiment 2 and the sweep.

    python run.py               full run, about 80 minutes with 6 workers, into results/
    python run.py --quick       small run (a few minutes), into results_quick/
    python run.py --only exp2   one part (checks, exp1, exp2 or sweep), into results_partial/
    --workers N                 processes for the tuning and the Bayes filter (default up to 6;
                                1 runs everything in one process, with the same numbers, but
                                takes several hours)

Everything is seeded, so a rerun gives the same numbers. The tests in tests/ run first, and
nothing else is written if one fails. Each part writes its files as soon as it finishes.
The log is run_log.txt, and DONE.txt appears when a run that did every part has finished.
"""

import argparse
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import scipy

from toy_experiment import parallel
from toy_experiment.config import FULL, QUICK
from toy_experiment.experiments import experiment1, experiment2, sweep
from toy_experiment.output import log, set_log_file, source_hashes, write_json

HERE = Path(__file__).resolve().parent


class Recorder:
    """pytest plugin that keeps one outcome per test, in the order the tests ran."""

    def __init__(self):
        self.outcomes = {}  # node id -> (PASS, FAIL or SKIP, detail)
        self.broken = []    # files that could not be collected

    def pytest_runtest_logreport(self, report):
        if report.failed:  # in setup, the test itself or teardown
            crash = getattr(report.longrepr, "reprcrash", None)
            self.outcomes[report.nodeid] = ("FAIL", crash.message.splitlines()[0] if crash else report.when)
        elif report.skipped and report.nodeid not in self.outcomes:
            reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else ""
            self.outcomes[report.nodeid] = ("SKIP", reason.removeprefix("Skipped: "))
        elif report.when == "call" and report.nodeid not in self.outcomes:
            self.outcomes[report.nodeid] = ("PASS", "")

    def pytest_collectreport(self, report):
        if report.failed:
            self.broken.append(report.nodeid)


def run_tests(out_dir, workers):
    """Run the tests, write checks.txt, and stop the run unless every test passed or was skipped."""
    rec = Recorder()
    code = pytest.main([str(HERE / "tests"), "-q", "-p", "no:cacheprovider", "--workers", str(workers),
                        "--run-info", str(out_dir / "run_info.json")], plugins=[rec])
    lines = [f"{status}  {nodeid}" + (f"  ({detail})" if detail else "")
             for nodeid, (status, detail) in rec.outcomes.items()]
    lines += [f"FAIL  {nodeid}  (could not be collected)" for nodeid in rec.broken]
    (out_dir / "checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        log(line)
    if code != 0 or any(line.startswith("FAIL") for line in lines):
        raise SystemExit("A test failed; see checks.txt. Results were not produced.")
    n_skip = sum(line.startswith("SKIP") for line in lines)
    log(f"all {len(lines) - n_skip} tests passed"
        + (f"; {n_skip} skipped (they compare worker processes with one process; run with --workers 2 or "
           "more to include them)" if n_skip else ""))


def main():
    ap = argparse.ArgumentParser(description="Toy experiments for the proposal (Section 4).")
    ap.add_argument("--quick", action="store_true", help="small, fast version to check the code runs")
    ap.add_argument("--only", choices=["checks", "exp1", "exp2", "sweep"], help="run one part only")
    ap.add_argument("--workers", type=int, default=parallel.DEFAULT_WORKERS,
                    help=f"processes for the tuning and the Bayes filter (default {parallel.DEFAULT_WORKERS}; "
                         "same numbers)")
    args = ap.parse_args()
    cfg = QUICK if args.quick else FULL
    out_dir = HERE / ("results_quick" if args.quick else "results_partial" if args.only else "results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "DONE.txt").unlink(missing_ok=True)
    log_file = out_dir / "run_log.txt"
    log_file.write_text("", encoding="utf-8")
    set_log_file(log_file)
    # written before the tests, which check that it records the code that made these results
    write_json(out_dir / "run_info.json",
               dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                    pytest=pytest.__version__, quick=args.quick, only=args.only, workers=args.workers,
                    sha256=source_hashes()))

    t_start = time.time()
    parts = [args.only] if args.only else ["checks", "exp1", "exp2", "sweep"]
    if "checks" in parts:
        run_tests(out_dir, args.workers)  # the tests start their own worker processes
    cache = {}
    with parallel.worker_pool(args.workers):
        if "exp1" in parts:
            experiment1(cfg, out_dir, cache)
        if "exp2" in parts:
            experiment2(cfg, out_dir)
        if "sweep" in parts:
            sweep(cfg, out_dir, cache)
    minutes = (time.time() - t_start) / 60
    log(f"finished in {minutes:.1f} minutes")
    if args.only is None:
        (out_dir / "DONE.txt").write_text(f"Run of every part finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
                                          f"in {minutes:.1f} minutes ({'quick' if args.quick else 'full'}).\n",
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
