# Robust filtering toy experiment

A small experiment for my MSc by Research proposal on outlier-robust filtering in selective state space models.

It asks two things on a scalar random walk with 10% outliers:

1. Does a robust (generalised-Bayes) Kalman update beat learned noise, and does it matter whether the noise sees the current reading?
2. How many repeated parallel scans does the robust filter need to match the sequential one?

## Run

```
pip install numpy scipy matplotlib
python toy_experiment.py --quick   # a few minutes
python toy_experiment.py           # full run, about 80 minutes with 6 workers
```

Results go to `results/`. Everything is seeded, and the script runs its own self-checks before any results are written.
