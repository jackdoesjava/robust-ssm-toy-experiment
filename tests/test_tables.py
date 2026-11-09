"""The text and LaTeX tables, the notes, and the saved results, on fabricated results in which
every value says which filter and metric it belongs to."""

import json
import re
from dataclasses import replace

import numpy as np
import pytest

import toy_experiment
from toy_experiment import experiments
from toy_experiment.config import (AFTER, BAYES, BOTH, BOTH_PAST, C2, C_PRIOR, FILTERS, INDEP, INDEP_FIX, INDEP_PAST,
                                   K_PRIOR, ONE_FC, ONE_ML, ONE_TRUE, ROB, ROB_FIX, SCENARIOS, TOLS)
from toy_experiment.experiments import experiment1, sweep
from toy_experiment.filters import imq_weight
from toy_experiment.tables import (SWEEP_COLS, TEX_ROWS, WEIGHT_PAIRS, fitted_text, scenario_text, signed, sweep_tex,
                                   table_exp1_tex, table_notes, weight_tex)

from .helpers import FABRICATED_CFG, TINY_EXP2, capture_figures, full_result, simple_result

PAIRS = [(ROB, ONE_ML), (ROB, ONE_FC), (ROB, INDEP), (BOTH, INDEP), (BOTH_PAST, INDEP_PAST), (ROB_FIX, INDEP_FIX),
         (ROB, BAYES)]  # the documented comparisons, written out again
EXP1_ORDER = [ONE_TRUE, ONE_ML, ONE_FC, INDEP, INDEP_FIX, INDEP_PAST, ROB, ROB_FIX, BOTH, BOTH_PAST, BAYES]


def rmse_fmt(v):
    return f"{v:.3f}"


def delay_fmt(v):
    return f"{v:.0f}"


def sign4(v):  # +/- and 4 decimals, with no "-0.0000"
    return "+0.0000" if abs(v) < 5e-5 else f"{v:+.4f}"


def tex_rows(tex):
    return [ln for ln in tex.splitlines() if ln.endswith(r"\\")]


def body_rows(tex):
    """The rows below the header."""
    return [ln for ln in tex.split(r"\midrule", 1)[1].splitlines() if ln.endswith(r"\\")]


def cells(row):
    return [c.strip() for c in row[:-2].split("&")]


def ncols(tex):
    m = re.search(r"\\begin\{tabular\}\{@\{\}(l*r*)@\{\}\}", tex)
    return len(m.group(1)) if m else -1


# Experiment 1 and the sweep
def test_scenario_text_paired_lines_are_the_seven_documented_comparisons_in_order():
    got = re.findall(r"^  (.+?) vs (.+?): ", scenario_text(simple_result()), flags=re.M)
    assert got == PAIRS


def test_scenario_text_paired_lines_are_first_minus_second():
    """Mean ± 1.96 sd / sqrt(n) for RMSE, CRPS and RMSE after the jump, and the delay with its
    paired-bootstrap interval."""
    res = simple_result()
    lines = scenario_text(res).splitlines()
    for f1, f2 in PAIRS:
        parts = []
        for key, name in (("rmse", "RMSE"), ("crps", "CRPS"), ("rmse_after", "RMSE after the jump")):
            dv = res["per_seq"][f1][key] - res["per_seq"][f2][key]
            parts.append(f"{name} {sign4(dv.mean())} ±{1.96 * dv.std(ddof=1) / np.sqrt(len(dv)):.4f}")
        db = res["boots"][f1] - res["boots"][f2]
        parts.append(f"delay {res['metrics'][f1]['delay'][0] - res['metrics'][f2]['delay'][0]:+.0f} "
                     f"[{np.percentile(db, 2.5):+.0f}, {np.percentile(db, 97.5):+.0f}]")
        assert f"  {f1} vs {f2}: " + ", ".join(parts) in lines


def test_scenario_text_post_jump_column_names_its_window():
    assert f"RMSE ({AFTER} steps)" in scenario_text(simple_result())


def test_signed_prints_a_value_that_rounds_to_zero_as_plus_zero():
    assert signed(-1e-9) == "+0.0000" and signed(-0.00006) == "-0.0001" and signed(0.0) == "+0.0000"
    assert signed(-0.04, 1) == "+0.0" and signed(-0.06, 1) == "-0.1"


def test_exp1_tex_has_every_filter_in_order_with_its_label_and_metrics():
    """RMSE, CRPS, 90% cover, 90% width, delay and post-jump RMSE."""
    words = {ONE_TRUE: "$R = 1$", ONE_ML: "likelihood", ONE_FC: "forecast", INDEP: "Kalman, input-dep.\\ $R_t$, tuned",
             INDEP_FIX: "Kalman, input-dep.\\ $R_t$, $c", INDEP_PAST: "Kalman, input-dep.\\ $R_t$ (past), tuned",
             ROB: "Robust, tuned", ROB_FIX: "Robust, $c", BOTH: "Robust + input-dep.\\ $R_t$, tuned",
             BOTH_PAST: "Robust + input-dep.\\ $R_t$ (past), tuned", BAYES: "Bayes"}
    res = simple_result()
    body = body_rows(table_exp1_tex(res))
    assert len(body) == len(EXP1_ORDER)
    for ln, lab in zip(body, EXP1_ORDER):
        row, mt = cells(ln), res["metrics"][lab]
        assert words[lab] in row[0] and sum(words[other] in row[0] for other in EXP1_ORDER) == 1
        assert row[1:] == [f"{mt['rmse'][0]:.3f}", f"{mt['crps'][0]:.3f}", f"{100 * mt['cov90'][0]:.1f}",
                           f"{mt['width90'][0]:.2f}", f"{mt['delay'][0]:.0f}", f"{mt['rmse_after'][0]:.2f}"]


def test_sweep_tables_have_the_documented_columns_and_row_labels_that_describe_their_settings():
    res = simple_result()
    cols = [ONE_ML, INDEP, ROB, BOTH, INDEP_FIX, ROB_FIX, BAYES]
    for key, fmt in (("rmse", rmse_fmt), ("delay", delay_fmt)):
        st = sweep_tex([(k, res) for k in range(len(SCENARIOS))], key, fmt, "TITLE")
        assert r"Setting & one $R$ & input-dep. & robust & both & input-dep. & robust & Bayes \\" in st
        assert any(ln.startswith("TITLE & ") for ln in st.splitlines())
        rows = body_rows(st)
        assert len(rows) == len(SCENARIOS)
        for ln in rows:
            assert cells(ln)[1:] == [fmt(res["metrics"][f][key][0]) for f in cols]
    for sc in SCENARIOS:
        label, test = sc.tex, sc.test
        if test.eps == 0:
            assert label.startswith("none")
            continue
        pct, sd = re.match(r"(\d+)\\%", label), re.search(r"sd (\d+)", label)
        assert pct is not None and int(pct.group(1)) == round(100 * test.eps)
        assert sd is not None and int(sd.group(1)) == round(np.sqrt(test.lam))
        assert ("test time only" in label) == (sc.train.eps == 0 < test.eps)
        assert ("bursts" in label) == (test.stay is not None)
        assert ("fast state" in label) == (sc.Q != 0.1)


def test_sweep_weight_table_shows_each_settings_rmse_change_from_adding_the_weight():
    """Both header rows; for noise from y_1..y_t and from y_1..y_{t-1}, the paired change with
    its interval and share, the share signed as the change and one-digit shares padded."""
    res = simple_result()
    wt = weight_tex([(k, res) for k in range(len(SCENARIOS))])
    assert r"\begin{tabular}{@{}lrr@{}}" in wt
    assert r"Change in RMSE & \multicolumn{2}{c}{Input-dependent $R_t$ computed from} \\" in wt.splitlines()
    assert r"Setting & $y_1, \dots, y_t$ & $y_1, \dots, y_{t-1}$ \\" in wt.splitlines()
    rows = body_rows(wt)
    assert len(rows) == len(SCENARIOS)
    for ln, sc in zip(rows, SCENARIOS):
        expect = [sc.tex]
        for f1, f2 in ((BOTH, INDEP), (BOTH_PAST, INDEP_PAST)):
            dv = res["per_seq"][f1]["rmse"] - res["per_seq"][f2]["rmse"]
            pct = 100 * dv.mean() / res["per_seq"][f2]["rmse"].mean()  # +125.0 and +66.7 here: no padding
            expect.append(f"${sign4(dv.mean())} \\pm {1.96 * dv.std(ddof=1) / np.sqrt(len(dv)):.4f}$ (${pct:+.1f}\\%$)")
        assert cells(ln) == expect
    # a tiny negative change keeps its sign in the share, and a one-digit share is padded by a digit
    res2 = dict(metrics={}, per_seq={})
    for lab, v in ((INDEP, 1.0), (BOTH, 0.9998), (INDEP_PAST, 1.0), (BOTH_PAST, 0.912)):
        res2["per_seq"][lab] = dict(rmse=np.full(4, v))
        res2["metrics"][lab] = dict(rmse=(v, 0.0))
    row2 = weight_tex([(0, res2)]).split(r"\midrule", 1)[1].splitlines()[1]
    assert row2 == (SCENARIOS[0].tex + r" & $-0.0002 \pm 0.0000$ ($-\phantom{0}0.0\%$) & "
                    r"$-0.0880 \pm 0.0000$ ($-\phantom{0}8.8\%$) \\")


def test_exp1_tex_layout():
    """7 columns in the spec and every row; spans 4 + 2 over (RMSE, CRPS, cover, width) and
    (delay, RMSE); the header names the cells in their order."""
    tex = table_exp1_tex(full_result())
    head = tex_rows(tex)
    assert ncols(tex) == 7 and all(ln.count("&") == 6 for ln in head[1:])
    assert head[0].strip() == r"& \multicolumn{4}{c}{Before the jump} & \multicolumn{2}{c}{After the jump} \\"
    assert r"\cmidrule(lr){2-5} \cmidrule(l){6-7}" in tex.splitlines()
    assert head[1] == r"Filter & RMSE & CRPS & 90\% cover & 90\% width & Delay & RMSE \\"


def test_exp1_tex_has_a_rule_above_the_input_dependent_robust_and_bayes_groups_and_nowhere_else():
    tex = table_exp1_tex(full_result())
    lines, head = tex.splitlines(), tex_rows(tex)
    want = [r"\midrule"]
    for label, _ in TEX_ROWS:
        if label in (INDEP, ROB, BAYES):
            want.append(r"\midrule")
        want.append(label)
    got = []
    for ln in lines[lines.index(head[1]) + 1:]:
        if ln == r"\midrule":
            got.append(ln)
        elif ln.endswith(r"\\"):
            got.append(next((lab for lab, name in TEX_ROWS if ln.startswith(name + " & ")), "?"))
    assert got == want


def test_sweep_tex_layout():
    """8 columns, 'Tuned' over the four tuned filters and 'c = 2 fixed' over the two fixed ones;
    the weight table: 3 columns, its span over both."""
    res = full_result()
    for key, fmt in (("rmse", rmse_fmt), ("delay", delay_fmt)):
        st = sweep_tex([(k, res) for k in range(len(SCENARIOS))], key, fmt, "TITLE")
        rows = tex_rows(st)
        assert ncols(st) == 1 + len(SWEEP_COLS) and all(ln.count("&") == len(SWEEP_COLS) for ln in rows[1:])
        assert rows[0].startswith(r"TITLE & \multicolumn{4}{c}{Tuned} & \multicolumn{2}{c}{$c = "
                                  f"{C_PRIOR:g}" r"$ fixed} & \\")
        assert r"\cmidrule(lr){2-5} \cmidrule(lr){6-7}" in st.splitlines()
    assert [lab for lab, _ in SWEEP_COLS[:4]] == [ONE_ML, INDEP, ROB, BOTH]
    assert [lab for lab, _ in SWEEP_COLS[4:6]] == [INDEP_FIX, ROB_FIX]
    wt = weight_tex([(k, res) for k in range(len(SCENARIOS))])
    wrows = tex_rows(wt)
    assert ncols(wt) == 3 and all(ln.count("&") == 2 for ln in wrows[1:])
    assert wrows[0].count("&") == 1 and r"\multicolumn{2}{c}{" in wrows[0]
    assert r"\cmidrule(l){2-3}" in wt.splitlines() and WEIGHT_PAIRS == [(BOTH, INDEP), (BOTH_PAST, INDEP_PAST)]


def cv(v):
    return "inf (off)" if not np.isfinite(v) else f"{v:g}"


def kv(k):
    return "- (not used)" if k is None else f"{k}"


def test_fitted_text_puts_each_filters_parameters_in_its_own_place():
    """Also with one kind of noise on and the other off; 'inf (off)' and '- (not used)' when off."""
    f = full_result()["fitted"]
    flat = " ".join(fitted_text(f).split())
    for w in (f"input-dependent R = {f.R_in:.3g}, c = inf (off), k = - (not used)",
              f"robust + input-dependent R = {f.R_both:.3g}, robust c = {f.c_both:g}, noise c = {f.c_in_both:g}, "
              f"k = {f.k_both}", f"robust R = {f.R_rob:.3g}, c = {f.c_rob:g}",
              f"past only: input-dependent R = {f.R_past:.3g}, c = inf (off), k = - (not used); robust + "
              f"input-dependent R = {f.R_both_past:.3g}, robust c = {f.c_both_past:g}, noise c = "
              f"{f.c_in_both_past:g}, k = {f.k_both_past}"):
        assert w in flat
    for g in (replace(f, c_past=1.9, k_past=7), replace(f, c_in=2.5, k_in=5)):  # one kind of noise on, the other off
        flat_g = " ".join(fitted_text(g).split())
        assert f"input-dependent R = {g.R_in:.3g}, c = {cv(g.c_in)}, k = {kv(g.k_in)};" in flat_g
        assert f"past only: input-dependent R = {g.R_past:.3g}, c = {cv(g.c_past)}, k = {kv(g.k_past)};" in flat_g


def fake_run_scenario(res, calls):
    """Stands in for run_scenario, returning res and recording each call."""
    def run(k, cfg, extra=False):
        calls.append((k, extra))
        return res
    return run


@pytest.fixture(scope="module")
def exp1_written(tmp_path_factory):
    """What experiment1 writes for the fabricated result."""
    res, calls, out = full_result(), [], tmp_path_factory.mktemp("exp1")
    with pytest.MonkeyPatch.context() as mp, capture_figures():
        mp.setattr(experiments, "run_scenario", fake_run_scenario(res, calls))  # where experiment1 looks it up
        experiment1(FABRICATED_CFG, out, {})
    assert calls == [(0, True)]
    return res, (out / "exp1_table.txt").read_text(encoding="utf-8"), \
        json.loads((out / "exp1_results.json").read_text(encoding="utf-8"))


def test_exp1_text_stability_lines_show_each_refit_and_each_filters_rmse_and_delay(exp1_written):
    res, text, _ = exp1_written
    sets = [ln for ln in text.splitlines() if ln.startswith("set ")]
    assert len(sets) == len(res["stability"])
    for r, (ln, row) in enumerate(zip(sets, res["stability"]), start=1):
        g = row["fit"]
        for part in (f"set {r}: robust R = {g.R_rob:.3g}, c = {cv(g.c_rob)};",
                     f"input-dependent R = {g.R_in:.3g}, c = {cv(g.c_in)}, k = {kv(g.k_in)};",
                     f"robust + input-dependent R = {g.R_both:.3g}, robust c = {cv(g.c_both)}, noise c = "
                     f"{cv(g.c_in_both)}, k = {kv(g.k_both)};", f"one R = {g.R_ml:.3g}"):
            assert part in ln
        block = text.split(ln, 1)[1].splitlines()[1:5]
        for lab, bl in zip((ONE_ML, INDEP, ROB, BOTH), block):
            assert bl.split()[-3:] == [f"{row[lab]['rmse']:.3f}", "/", f"{row[lab]['delay']:.0f}"] and lab in bl


def test_exp1_text_tradeoff_table_has_every_point(exp1_written):
    """c, R, RMSE, CRPS, 90% cover and delay."""
    res, text, _ = exp1_written
    curve_lines = [ln for ln in text.splitlines() if re.match(r"^\s+(\d|inf)", ln) and ln.rstrip()[-1:].isdigit()]
    assert len(curve_lines) == len(res["curve"])
    for ln, d in zip(curve_lines, res["curve"]):
        assert ln.split() == [f"{d['c']:g}", f"{d['R']:.3g}", f"{d['rmse']:.3f}", f"{d['crps']:.3f}",
                              f"{100 * d['cov90']:.1f}%", f"{d['delay']:.0f}"]


def test_exp1_results_json_has_every_filters_own_bootstrap_delays_and_per_sequence_values(exp1_written):
    res, _, js = exp1_written
    for lab in FILTERS:
        assert js["bootstrap_delays"][lab] == [int(v) for v in res["boots"][lab]]
        assert js["per_sequence"][lab]["rmse_after"] == list(res["per_seq"][lab]["rmse_after"])


def test_notes_state_their_claims_with_the_counts_and_windows_of_the_configuration():
    """All nine notes; the Bayes note on bursts only under the sweep; and the combined filter's
    weight bound holds: with e_t = d_t the weight stays above it, and tends to it."""
    for cfg in (FABRICATED_CFG, replace(FABRICATED_CFG, n_test=7, n_train=5, n_val=6, T=90, jump_t=45)):
        for sw in (False, True):
            notes = table_notes(cfg, sweep=sw)
            text = " ".join(notes)
            frags = [f"means over {cfg.n_test} test sequences of {cfg.T} steps",
                     f"All over steps 1-{cfg.jump_t} (before the jump)",
                     f"RMSE after the jump: steps {cfg.jump_t + 1}-{cfg.jump_t + AFTER}.",
                     f"fitted on {cfg.n_train} training and {cfg.n_val} validation sequences",
                     "the one-R maximum-likelihood filter by the likelihood of the training sequences",
                     "about ±0.001 in RMSE before the jump", "it can be as wide as the narrower row's ±, or wider",
                     "R_t = R + d_t^2 / c^2 equals R / w_t^2 for an IMQ weight on d_t",
                     "(past only): R_t = R + d_{t-1}^2 / c^2, which uses y_1, ..., y_{t-1} only",
                     "is a proper density for y_t, which it would not be with d_t in place of d_{t-1}",
                     "like every filter it was tuned on the forecast with the base R",
                     "in those units d_t is the larger, so the input-dependent weight down-weights more",
                     "the weight stays above about (1 + noise c^2 / robust c^2)^(-1/2)"]
            assert [fr for fr in frags if fr not in text] == [] and len(notes) == 9
            assert ("which it expects to come in runs" in text) == sw
    d = np.logspace(-1, 4, 200)
    for cr, cn, P, R in ((2.0, 4.0, 0.3, 1.0), (3.0, 1.5, 0.1, 1.4)):
        w = imq_weight(d, P + R + d * d / cn ** 2, cr)
        bound = (1 + cn ** 2 / cr ** 2) ** -0.5
        assert np.all(w >= bound) and abs(w[-1] - bound) < 1e-6


def test_sweep_writes_every_settings_block_the_notes_and_the_three_tables(tmp_path, monkeypatch):
    """The sweep notes include the Bayes note on bursts."""
    res, calls = full_result(), []
    cfg = replace(FABRICATED_CFG, scenarios=tuple(range(len(SCENARIOS))))
    monkeypatch.setattr(experiments, "run_scenario", fake_run_scenario(res, calls))  # where sweep looks it up
    sweep(cfg, tmp_path, {})
    assert calls == [(k, False) for k in cfg.scenarios]
    got = {name: (tmp_path / name).read_text(encoding="utf-8")
           for name in ("sweep.txt", "sweep_rmse_table.tex", "sweep_delay_table.tex", "sweep_weight_table.tex")}
    done = [(k, res) for k in cfg.scenarios]
    assert got["sweep_weight_table.tex"] == weight_tex(done)
    assert got["sweep_rmse_table.tex"] == sweep_tex(done, "rmse", rmse_fmt, "RMSE before the jump")
    assert got["sweep_delay_table.tex"] == sweep_tex(done, "delay", delay_fmt, "Delay after the jump")
    assert got["sweep.txt"].count(scenario_text(res)) == len(SCENARIOS)
    assert got["sweep.txt"].endswith("Notes\n" + "\n".join(f"- {n}" for n in table_notes(cfg, sweep=True)) + "\n")


def test_labels_and_the_docstring_name_the_fixed_c_and_k_that_the_code_uses():
    doc = " ".join(toy_experiment.__doc__.split())
    labels = [INDEP_FIX, ROB_FIX] + [name for lab, name in TEX_ROWS if lab in (INDEP_FIX, ROB_FIX)]
    assert all(f"c = {C_PRIOR:g}" in s for s in labels)
    assert f"c = {C_PRIOR:g}, k = {K_PRIOR} fixed" in doc and f"with c = {C_PRIOR:g} fixed" in doc


# Experiment 2
def test_exp2_practical_rule_line_is_the_median_and_max_of_the_extra_passes(tiny_exp2):
    """Over the sequences where both counts were reached; some are not reached here."""
    lines, c2, unreached = tiny_exp2.text.splitlines(), f"c={C2:g}", 0
    for tol, name in zip(TOLS, ("1%", "1e-8")):
        extra, left = [], 0
        for row in tiny_exp2.rows:
            for T in TINY_EXP2.T2_list:
                pr, pa = (np.array(row[f"T={T} {c2}"][part][str(tol)]) for part in ("practical", "passes"))
                fine = (pr > 0) & (pa > 0)
                extra += list(pr[fine] - pa[fine])
                left += int((~fine).sum())
        unreached += left
        want = f"  tolerance {name} of sd: " + (f"{np.median(extra):g} ({max(extra)})" if extra else "n/a") + \
               (f" ({left} sequences not converged, left out)" if left else "")
        assert want in lines
    assert unreached > 0


def test_exp2_tex_cells_equal_the_saved_results(tiny_exp2):
    def cell(p, maxpass):  # median (max), counting a sequence that was not reached as maxpass
        p = np.array(p)
        pc = np.where(p < 0, maxpass, p)
        lead = ">=" if 2 * (p < 0).sum() >= len(p) else ""
        return f"{lead}{np.median(pc):g} ({'>=' if (p < 0).any() else ''}{pc.max()})".replace(">=", r"$\ge$")

    c2, rows = f"c={C2:g}", tiny_exp2.rows
    trs = body_rows(tiny_exp2.tex)
    assert len(trs) == len(rows)
    for ln, row in zip(trs, rows):
        m = row[f"T={TINY_EXP2.T2_mid} {c2}"]
        factor = np.nan if m["factor"] is None else m["factor"]
        assert cells(ln)[1:] == ([f"{m['memory']:.3f}", f"{factor:.3f}"]
                                 + [cell(row[f"T={T} {c2}"]["passes"][str(tol)], row[f"T={T} {c2}"]["maxpass"])
                                    for tol in TOLS for T in TINY_EXP2.T2_list])


def test_exp2_tex_layout(tiny_exp2):
    """'T = ' header cells in T order, tolerance titles in TOLS order, the regimes' row names."""
    Ts = TINY_EXP2.T2_list
    th = [f"{T:,}".replace(",", "{,}") for T in Ts]
    head = [f"$T = {th[0]}$"] + [f"${t}$" for t in th[1:]]
    r2 = tex_rows(tiny_exp2.tex)
    assert r2[1] == "Dynamics & Memory & Factor & " + " & ".join(head) + " & " + " & ".join(head) + r" \\"
    spans = [int(v) for v in re.findall(r"\\multicolumn\{(\d+)\}", r2[0])]
    titles = re.findall(r"Passes to (1\\% of sd|\$10\^\{-8\}\$ sd)", r2[0])
    assert titles == [r"1\% of sd", r"$10^{-8}$ sd"] and spans == [len(Ts), len(Ts)]
    assert ncols(tiny_exp2.tex) == 3 + 2 * len(Ts) and all(ln.count("&") == 2 + 2 * len(Ts) for ln in r2[1:])
    assert list(TOLS) == [1e-2, 1e-8]
    names = {"a_t <= 0.6": r"$a_t \le 0.6$", "a_t ~ 0.5": r"$a_t \approx 0.5$", "a_t ~ 0.9": r"$a_t \approx 0.9$",
             "a_t ~ 0.99": r"$a_t \approx 0.99$", "a_t ~ 0.999": r"$a_t \approx 0.999$", "random walk": "Random walk"}
    assert [ln.split(" & ")[0] for ln in r2[2:]] == [names[row["regime"]] for row in tiny_exp2.rows]
