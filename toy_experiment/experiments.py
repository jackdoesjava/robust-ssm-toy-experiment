"""Experiment 1, the sweep over settings, and experiment 2."""

import time

import numpy as np

from .bayes import bayes_filter
from .config import BAYES, BOTH, C2, C_GRID, INDEP, N_BOOT, ONE_ML, REGIMES, ROB, SCENARIOS, TOLS
from .figures import figure_exp1, figure_exp2
from .filters import kalman_filter
from .metrics import gaussian_scores, summarise
from .output import log, write_json
from .passes import count_passes
from .simulate import simulate_rw, simulate_selective
from .tables import (exp1_text, exp2_tex, exp2_text, fmt_passes, results_json, scenario_text, sweep_tex,
                     sweep_text, table_exp1_tex, weight_tex)
from .tuning import filter_specs, fit_all, run_filter


# Experiment 1 and the sweep
def _simulate(k, stream, n, cfg, outliers, jump_t):
    """Sequences of scenario k. Every set has its own seeded stream: 0 training, 1 validation,
    2 test, 10 + r and 20 + r the training and validation sets of refit r."""
    rng = np.random.default_rng([2027, k, stream])
    return simulate_rw(rng, n, cfg.T, SCENARIOS[k].Q, outliers.eps, outliers.lam, outliers.stay, jump_t)


def _test_metrics(spec, x, y, Q, jump_t, boot_idx):
    """The filter's output on the test sequences, and its summary."""
    out = run_filter(spec, y, Q)
    return out, summarise(x, gaussian_scores(x, out["m"], out["P"]), jump_t, boot_idx)


def run_scenario(k, cfg, extra=False):
    """Experiment 1 under scenario k: every filter fitted, then scored on the test sequences.

    With extra = True also the trade-off over c, the tuning-stability refits, and the first
    test sequence with each filter's output, for the figure.
    """
    sc = SCENARIOS[k]
    # training and validation data have no jump, so no filter is tuned on one
    _, y_tr, _ = _simulate(k, 0, cfg.n_train, cfg, sc.train, None)
    _, y_va, _ = _simulate(k, 1, cfg.n_val, cfg, sc.train, None)
    x_te, y_te, o_te = _simulate(k, 2, cfg.n_test, cfg, sc.test, cfg.jump_t)
    boot_idx = np.random.default_rng([2027, k, 99]).integers(0, len(x_te), size=(N_BOOT, len(x_te)))

    fit = fit_all(y_tr, y_va, sc.Q)
    res = dict(name=sc.name, fitted=fit, metrics={}, per_seq={}, boots={})
    outs = {}
    for label, spec in filter_specs(fit, y_te).items():
        outs[label], summary = _test_metrics(spec, x_te, y_te, sc.Q, cfg.jump_t, boot_idx)
        res["metrics"][label], res["per_seq"][label], res["boots"][label] = summary
    scores = bayes_filter(y_te, x_te, sc.Q, sc.test.eps, sc.test.lam, sc.test.stay)
    res["metrics"][BAYES], res["per_seq"][BAYES], res["boots"][BAYES] = summarise(x_te, scores, cfg.jump_t, boot_idx)

    if extra:
        res["curve"] = _tradeoff_curve(fit, x_te, y_te, sc.Q, cfg.jump_t, boot_idx)
        res["stability"] = _stability(k, cfg, x_te, y_te, boot_idx)
        res["example"] = dict(x=x_te[0], y=y_te[0], outlier=o_te[0],
                              fits={lab: (outs[lab]["m"][0], outs[lab]["P"][0]) for lab in outs})
    return res


def _tradeoff_curve(fit, x_te, y_te, Q, jump_t, boot_idx):
    """The robust filter with R tuned for each c, on the test sequences. c = inf is the one-R
    Kalman filter tuned by forecast CRPS."""
    curve = []
    for c in C_GRID:
        out = kalman_filter(y_te, fit.R_by_c[float(c)], Q, c=c)
        mt, _, _ = summarise(x_te, gaussian_scores(x_te, out["m"], out["P"]), jump_t, boot_idx)
        curve.append(dict(c=float(c), R=fit.R_by_c[float(c)], rmse=mt["rmse"][0],
                          crps=mt["crps"][0], cov90=mt["cov90"][0], delay=mt["delay"][0]))
    return curve


def _stability(k, cfg, x_te, y_te, boot_idx):
    """Everything refitted on fresh training and validation sequences, scored on the same test
    sequences; the delay without its bootstrap interval."""
    sc = SCENARIOS[k]
    rows = []
    for r in range(cfg.n_stability):
        _, y_tr2, _ = _simulate(k, 10 + r, cfg.n_train, cfg, sc.train, None)
        _, y_va2, _ = _simulate(k, 20 + r, cfg.n_val, cfg, sc.train, None)
        fit2 = fit_all(y_tr2, y_va2, sc.Q)
        row = dict(fit=fit2)
        for label, spec in filter_specs(fit2, y_te).items():
            if label in (ONE_ML, INDEP, ROB, BOTH):
                _, (mt, _, _) = _test_metrics(spec, x_te, y_te, sc.Q, cfg.jump_t, boot_idx[:1])
                row[label] = dict(rmse=mt["rmse"][0], crps=mt["crps"][0], cov90=mt["cov90"][0], delay=mt["delay"][0])
        rows.append(row)
        log(f"  tuning stability {r + 1}/{cfg.n_stability} done")
    return rows


def experiment1(cfg, out_dir, cache):
    """The main scenario with the trade-off curve and the stability refits; cache[0] keeps it for the sweep."""
    log("experiment 1: main scenario")
    res = run_scenario(0, cfg, extra=True)
    cache[0] = res
    (out_dir / "exp1_table.txt").write_text(exp1_text(res, cfg), encoding="utf-8")
    (out_dir / "exp1_table.tex").write_text(table_exp1_tex(res), encoding="utf-8")
    figure_exp1(res, cfg, out_dir / "exp1_figure")
    write_json(out_dir / "exp1_results.json", results_json(res, ("name", "fitted", "metrics", "curve", "stability")))
    log("experiment 1 written: exp1_table.txt/.tex, exp1_figure.pdf/.png")


def sweep(cfg, out_dir, cache):
    """Experiment 1 under every setting in cfg.scenarios, written after each one so a stopped
    run still leaves the finished ones."""
    blocks, keep, done = [], [], []
    for k in cfg.scenarios:
        log(f"sweep: {SCENARIOS[k].name}")
        res = cache.get(k) or run_scenario(k, cfg)
        blocks.append(scenario_text(res))
        keep.append(results_json(res, ("name", "fitted", "metrics")))
        done.append((k, res))
        (out_dir / "sweep.txt").write_text(sweep_text(blocks, cfg), encoding="utf-8")
        write_json(out_dir / "sweep_results.json", keep)
        (out_dir / "sweep_rmse_table.tex").write_text(
            sweep_tex(done, "rmse", lambda v: f"{v:.3f}", "RMSE before the jump"), encoding="utf-8")
        (out_dir / "sweep_delay_table.tex").write_text(
            sweep_tex(done, "delay", lambda v: f"{v:.0f}", "Delay after the jump"), encoding="utf-8")
        (out_dir / "sweep_weight_table.tex").write_text(weight_tex(done), encoding="utf-8")
    log("sweep written: sweep.txt, sweep_rmse_table.tex (RMSE before the jump), sweep_delay_table.tex, "
        "sweep_weight_table.tex (the robust weight added to each kind of input-dependent noise)")


# Experiment 2
def _regime_row(label, a_med, cap, seed, cfg):
    """One regime's pass counts at every length (c = C2, and also c = 1 at T2_mid), and its
    median error curve at T2_mid."""
    row, curve = dict(regime=label), None
    for T in cfg.T2_list:
        maxpass = min(cfg.maxpass, T + 2)
        x, y, a, Qs = simulate_selective(np.random.default_rng([2027, seed, T]), cfg.n2, T, a_med, cap)
        assert cap is None or a.max() <= cap
        for c in ((C2, 1.0) if T == cfg.T2_mid else (C2,)):
            t_start = time.time()
            res = count_passes(y, Qs, a, c, maxpass)
            row[f"T={T} c={c:g}"] = dict(
                maxpass=maxpass,
                passes={str(tol): res["passes"][tol] for tol in TOLS},
                practical={str(tol): res["practical"][tol] for tol in TOLS},
                factor=float(np.nanmedian(res["factor"])) if np.isfinite(res["factor"]).any() else float("nan"),
                memory=float(np.median(res["memory"])),
                a=float(np.median(a)), a_max=float(a.max()))
            if T == cfg.T2_mid and c == C2:
                curve = np.median(res["errs"], axis=1)
            log(f"experiment 2: {label}, T = {T}, c = {c:g}: passes to 1% sd "
                f"{fmt_passes(res['passes'][TOLS[0]], maxpass)}, to 1e-8 sd "
                f"{fmt_passes(res['passes'][TOLS[1]], maxpass)} [{time.time() - t_start:.0f}s]")
    return row, curve


def experiment2(cfg, out_dir):
    """Passes of the frozen-weight iteration in every regime; the results are written after each regime."""
    rows, curves = [], {}
    for label, a_med, cap, seed in REGIMES:
        row, curves[label] = _regime_row(label, a_med, cap, seed, cfg)
        rows.append(row)
        write_json(out_dir / "exp2_results.json", rows)
    (out_dir / "exp2_table.txt").write_text(exp2_text(rows, cfg), encoding="utf-8")
    (out_dir / "exp2_table.tex").write_text(exp2_tex(rows, cfg), encoding="utf-8")
    figure_exp2(rows, curves, cfg, out_dir / "exp2_figure")
    log("experiment 2 written: exp2_table.txt/.tex, exp2_figure.pdf/.png")
