"""Text and LaTeX tables, the notes under them, and what is saved of each scenario."""

import numpy as np

from .config import (AFTER, BAYES, BOTH, BOTH_PAST, C2, C_PRIOR, FILTERS, INDEP, INDEP_FIX, INDEP_PAST,
                     K_PRIOR, NEAR, ONE_FC, ONE_ML, ONE_TRUE, PAIRS, ROB, ROB_FIX, SCENARIOS, TOLS)


def fmt_c(c, tex=False):
    if not np.isfinite(c):
        return r"\infty" if tex else "inf (off)"
    return f"{c:g}"


def fmt_k(k):
    return "- (not used)" if k is None else f"{k}"


def signed(v, digits=4):
    """v with its sign, at the given decimals; a value that rounds to zero prints as +0.000..."""
    s = f"{v:+.{digits}f}"
    return "+" + s[1:] if float(s) == 0 else s


def paired(res, f1, f2, key):
    """Mean and 95% half-width of the per-sequence differences f1 - f2 of a metric."""
    dv = res["per_seq"][f1][key] - res["per_seq"][f2][key]
    return float(dv.mean()), float(1.96 * dv.std(ddof=1) / np.sqrt(len(dv)))


# Experiment 1 and the sweep
def fitted_text(fit):
    """The fitted parameters, as printed at the top of each scenario's block."""
    return (f"fitted: one R = {fit.R_ml:.3g} (max likelihood) or {fit.R_fc:.3g} (forecast CRPS); "
            f"input-dependent R = {fit.R_in:.3g}, c = {fmt_c(fit.c_in)}, k = {fmt_k(fit.k_in)}; "
            f"input-dependent with c = {C_PRIOR:g}, k = {K_PRIOR} fixed: R = {fit.R_in_prior:.3g};\n"
            f"        robust R = {fit.R_rob:.3g}, c = {fmt_c(fit.c_rob)}; robust with c = {C_PRIOR:g} fixed: "
            f"R = {fit.R_prior:.3g}; robust + input-dependent R = {fit.R_both:.3g}, robust c = "
            f"{fmt_c(fit.c_both)}, noise c = {fmt_c(fit.c_in_both)}, k = {fmt_k(fit.k_both)};\n"
            f"        past only: input-dependent R = {fit.R_past:.3g}, c = {fmt_c(fit.c_past)}, "
            f"k = {fmt_k(fit.k_past)}; robust + input-dependent R = {fit.R_both_past:.3g}, robust c = "
            f"{fmt_c(fit.c_both_past)}, noise c = {fmt_c(fit.c_in_both_past)}, k = {fmt_k(fit.k_both_past)}")


def scenario_text(res):
    """One scenario's block of the text output: the fits, every filter's metrics and the paired lines."""
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


def exp1_text(res, cfg):
    """exp1_table.txt: the main scenario's block, the trade-off over c, the tuning stability and the notes."""
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
        text.append(f"set {r}: robust R = {f.R_rob:.3g}, c = {fmt_c(f.c_rob)}; input-dependent R = "
                    f"{f.R_in:.3g}, c = {fmt_c(f.c_in)}, k = {fmt_k(f.k_in)}; robust + input-dependent R = "
                    f"{f.R_both:.3g}, robust c = {fmt_c(f.c_both)}, noise c = {fmt_c(f.c_in_both)}, "
                    f"k = {fmt_k(f.k_both)}; one R = {f.R_ml:.3g}")
        for label in (ONE_ML, INDEP, ROB, BOTH):
            text.append(f"    {label:<34} {row[label]['rmse']:.3f} / {row[label]['delay']:.0f}")
    text += ["", "Notes"] + [f"- {note}" for note in table_notes(cfg)]
    return "\n".join(text) + "\n"


def sweep_text(blocks, cfg):
    """sweep.txt: the finished scenarios' blocks and the notes."""
    notes = "Notes\n" + "\n".join(f"- {note}" for note in table_notes(cfg, sweep=True))
    return "\n\n".join(blocks + [notes]) + "\n"


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
    """exp1_table.tex: every filter's metrics in the main scenario, in groups."""
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


SWEEP_COLS = [(ONE_ML, r"one $R$"), (INDEP, r"input-dep."), (ROB, r"robust"), (BOTH, r"both"),
              (INDEP_FIX, r"input-dep."), (ROB_FIX, r"robust"), (BAYES, r"Bayes")]


def sweep_tex(results, key, fmt, title):
    """A sweep table of one metric: a row per finished scenario (k, res), a column per SWEEP_COLS filter."""
    head = [r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
            title + r" & \multicolumn{4}{c}{Tuned} & \multicolumn{2}{c}{" + f"$c = {C_PRIOR:g}$ fixed" + r"} & \\",
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-7}",
            "Setting & " + " & ".join(name for _, name in SWEEP_COLS) + r" \\", r"\midrule"]
    body = [SCENARIOS[k].tex + " & " + " & ".join(fmt(res["metrics"][lab][key][0]) for lab, _ in SWEEP_COLS)
            + r" \\" for k, res in results]
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
    body = [SCENARIOS[k].tex + " & " + " & ".join(cell(res, f1, f2) for f1, f2 in WEIGHT_PAIRS) + r" \\"
            for k, res in results]
    return "\n".join(head + body + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_notes(cfg, sweep=False):
    """The notes printed under the experiment 1 table, and with sweep = True under the sweep."""
    return [
        f"RMSE and CRPS: means over {cfg.n_test} test sequences of {cfg.T} steps, ± the half-width of a 95% "
        f"interval for the mean. Coverage and width: means. All over steps 1-{cfg.jump_t} (before the jump).",
        f"Delay: steps after the jump until the error averaged over sequences is below {NEAR:g} (80% of the jump "
        "recovered); 0 means already at the first step after it. Its interval resamples sequences. "
        f"RMSE after the jump: steps {cfg.jump_t + 1}-{cfg.jump_t + AFTER}.",
        "The ± in a row is for that row alone. Every row is scored on the same test sequences, and each paired line "
        "gives a difference its own interval. Between filters that behave alike, such as the robust filter and "
        "input-dependent noise from y_t, that interval is often much narrower than either row's (about ±0.001 in "
        "RMSE before the jump; wider with bursts or a fast state). Between filters that behave differently, such as "
        "the robust filter and one R, it can be as wide as the narrower row's ±, or wider; so can the past-only "
        "pair, which is the robust filter against one R wherever tuning switched its noise off. Judge differences "
        "by the paired lines, not by the printed digits.",
        f"Parameters were fitted on {cfg.n_train} training and {cfg.n_val} validation sequences without a jump, "
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


# Experiment 2
def fmt_passes(p, maxpass):
    """Median (max) of pass counts, with a sequence that was not reached (-1) counted as maxpass
    and marked >=; the median too when half or more were not reached."""
    pc = np.where(p < 0, maxpass, p)
    med = np.median(pc)
    lead = ">=" if (p < 0).sum() * 2 >= len(p) else ""
    return f"{lead}{med:g} ({'>=' if (p < 0).any() else ''}{pc.max()})"


def _passes(row, T, tol):
    """The pass counts of one regime at length T and tolerance tol, with c = C2."""
    r = row[f"T={T} c={C2:g}"]
    return fmt_passes(np.array(r["passes"][str(tol)]), r["maxpass"])


def exp2_text(rows, cfg):
    """exp2_table.txt: the pass counts of every regime, the practical rule, and c = 1."""
    mid, Ts = cfg.T2_mid, cfg.T2_list
    lines = ["Passes of the repeated frozen-weight iteration needed to match the sequential robust filter:",
             f"median (max) over {cfg.n2} sequences, weight of proposal eq. (3) with c = {C2:g}.",
             "A pass is one frozen-weight Kalman filter, i.e. one parallel scan; pass 1 has every weight 1.",
             f"memory: median over steps and sequences of a_{{t+1}} (1 - K_t), the filter's memory per step, "
             f"at T = {mid}.",
             "factor: median over sequences of the geometric-mean shrink of the largest error per pass, from pass 2 "
             f"to the first pass within 1e-8 sd, at T = {mid}. It rises with T.",
             "", f"{'regime':<13}{'a_t':>7}{'max a_t':>9}{'memory':>8}{'factor':>8}"
             + "".join(f"{'1% T=' + str(T):>14}" for T in Ts) + "".join(f"{'1e-8 T=' + str(T):>15}" for T in Ts)]
    for row in rows:
        m = row[f"T={mid} c={C2:g}"]
        cells = [f"{row['regime']:<13}{m['a']:>7.3f}{m['a_max']:>9.3f}{m['memory']:>8.3f}{m['factor']:>8.3f}"]
        for tol, width in ((TOLS[0], 14), (TOLS[1], 15)):
            for T in Ts:
                cells.append(f"{_passes(row, T, tol):>{width}}")
        lines.append("".join(cells))
    lines += ["", f"Practical stopping rule (stop once no mean moves by more than the tolerance), c = {C2:g}: "
              "extra passes beyond the counts above, median (max) over all regimes, lengths and sequences:"]
    runs = [row[f"T={T} c={C2:g}"] for row in rows for T in Ts]
    for tol, name in zip(TOLS, ("1%", "1e-8")):
        practical = np.concatenate([np.array(r["practical"][str(tol)]) for r in runs])
        passes = np.concatenate([np.array(r["passes"][str(tol)]) for r in runs])
        extra = practical - passes
        valid = (practical > 0) & (passes > 0)  # both counts reached
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
    return "\n".join(lines) + "\n"


def _tex_regime(label):
    if label == "random walk":
        return "Random walk"
    return "$" + label.replace("<=", r"\le").replace("~", r"\approx") + "$"


def exp2_tex(rows, cfg):
    """exp2_table.tex: memory, factor and the pass counts of every regime."""
    mid, Ts = cfg.T2_mid, cfg.T2_list
    t_head = [f"{T:,}".replace(",", "{,}") for T in Ts]
    t_cells = [f"$T = {t_head[0]}$"] + [f"${T}$" for T in t_head[1:]]
    tex = [r"\small", r"\begin{tabular}{@{}lrr" + "r" * (2 * len(Ts)) + r"@{}}", r"\toprule",
           r" & & & \multicolumn{" + str(len(Ts)) + r"}{c}{Passes to 1\% of sd} & \multicolumn{"
           + str(len(Ts)) + r"}{c}{Passes to $10^{-8}$ sd} \\",
           rf"\cmidrule(lr){{4-{3 + len(Ts)}}} \cmidrule(l){{{4 + len(Ts)}-{3 + 2 * len(Ts)}}}",
           r"Dynamics & Memory & Factor & " + " & ".join(t_cells) + " & " + " & ".join(t_cells) + r" \\", r"\midrule"]
    for row in rows:
        m = row[f"T={mid} c={C2:g}"]
        cells = [_passes(row, T, tol).replace(">=", r"$\ge$") for tol in TOLS for T in Ts]
        tex.append(f"{_tex_regime(row['regime'])} & {m['memory']:.3f} & {m['factor']:.3f} & " + " & ".join(cells)
                   + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex) + "\n"
