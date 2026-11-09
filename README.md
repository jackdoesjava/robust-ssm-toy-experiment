# Robust filtering toy experiment

A small experiment for my MSc by Research proposal on outlier-robust filtering in selective state space models.

It asks two things on a scalar random walk with 10% outliers:

1. Does a robust (generalised-Bayes) Kalman update beat learned noise, and does it matter whether the noise sees the current reading?
2. How many repeated parallel scans does the robust filter need to match the sequential one?

## Run

```
pip install -r requirements.txt
python run.py --quick   # a few minutes, into results_quick/
python run.py           # full run, about 80 minutes with 6 workers, into results/
```

`--only checks|exp1|exp2|sweep` runs one part, into `results_partial/`. `--workers N` sets the number of processes; the numbers are the same for any N.

Everything is seeded. The tests run before any results are written, and nothing is written if one fails. To run them on their own:

```
python -m pytest -q tests              # skips the two tests that need worker processes
python -m pytest -q tests --workers 2
```

## Layout

- `run.py` runs the tests, then the experiments.
- `toy_experiment/` holds the code: settings and scenarios (`config.py`), data (`simulate.py`), the Kalman, robust and grid Bayes filters (`filters.py`, `bayes.py`), tuning (`tuning.py`), scores (`metrics.py`), the repeated scans (`passes.py`), the experiments (`experiments.py`), tables and figures (`tables.py`, `figures.py`) and the worker pool (`parallel.py`).
- `tests/` holds the tests, one file per part of the code.
- `results/` holds the outputs of a full run.
