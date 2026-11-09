"""The record of a run: run_info.json and which tests are skipped."""

import hashlib
import json
import pkgutil
from pathlib import Path

import toy_experiment
from toy_experiment import parallel
from toy_experiment.output import source_hashes

POOL_TESTS = {"test_bayes_filter_split_across_worker_processes_equals_one_call",
              "test_fit_all_in_worker_processes_equals_fit_all_in_one_process"}


def test_run_info_records_the_sha256_of_every_source_file(request):
    """The run_info.json that run.py wrote before the tests, or without run.py what it would write."""
    path = request.config.getoption("run_info")
    recorded = json.loads(Path(path).read_text(encoding="utf-8"))["sha256"] if path else source_hashes()
    package = Path(toy_experiment.__file__).resolve().parent
    root = package.parent
    files = ([root / "run.py", package / "__init__.py"]
             + [package / f"{m.name}.py" for m in pkgutil.iter_modules([str(package)])]
             + list((root / "tests").glob("*.py")))
    assert recorded == {f.relative_to(root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def test_the_pool_tests_are_skipped_exactly_when_there_are_no_worker_processes(request):
    """And no other test is skipped."""
    items = request.session.items
    no_workers = request.config.getoption("workers") < 2
    marked = {item.name for item in items if item.get_closest_marker("needs_workers")}
    skipped = {item.name for item in items if item.get_closest_marker("skip")}
    assert marked == POOL_TESTS & {item.name for item in items}
    assert skipped == (marked if no_workers else set())
    assert parallel.active() != no_workers
