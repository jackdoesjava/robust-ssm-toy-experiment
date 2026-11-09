"""The run log, JSON output, and the hashes that tie results to the code that made them."""

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # the folder that holds run.py

_log_file = None  # run_log.txt of the current run; None: print only


def set_log_file(path):
    global _log_file
    _log_file = path


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:  # a console that cannot show a character (the log file is UTF-8)
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    if _log_file is not None:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def clean_json(o):
    """Make numbers JSON-safe: numpy types to Python, inf to the string "inf"."""
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return clean_json(dataclasses.asdict(o))
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


def source_files():
    """run.py, the package and the tests."""
    return [ROOT / "run.py", *sorted((ROOT / "toy_experiment").glob("*.py")),
            *sorted((ROOT / "tests").glob("*.py"))]


def source_hashes():
    """SHA-256 of every source file, keyed by its path relative to the folder of run.py."""
    return {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files()}
