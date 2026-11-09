"""The figures: every line, mark and legend entry against the values it stands for."""

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from toy_experiment.config import BAYES, C2, C_PRIOR, INDEP, ONE_ML, ONE_TRUE, REGIMES, ROB, TOLS, Z90
from toy_experiment.figures import (CURVE_NAME, INDEP_NAME, INF_NAME, ML_NAME, PRIOR_NAME, TUNED_NAME,
                                    figure_exp1)
from toy_experiment.passes import count_passes
from toy_experiment.simulate import simulate_selective

from .helpers import FABRICATED_CFG, SMALL_EXTRA, TINY_EXP2, capture_figures, full_result

REFERENCE_LINES = ((ONE_ML, ML_NAME), (INDEP, INDEP_NAME), (BAYES, "Bayes filter (reference)"))


def regime_label(label):  # figure 2's legend label of a regime, written out again
    return ("random walk ($a_t = 1$)" if label == "random walk"
            else "$" + label.replace("<=", r"\leq").replace("~", r"\approx") + "$")


def test_figure_1_shows_the_values_of_the_small_run(small_main_run, tmp_path):
    """The filters in (a); the curve, the c = inf point, the fixed and tuned c marks and the
    reference lines in (b) and (c)."""
    res = dict(small_main_run, fitted=replace(small_main_run["fitted"], c_rob=2.25))  # a tuned c distinct from C_PRIOR
    with capture_figures() as cap:
        figure_exp1(res, SMALL_EXTRA, tmp_path / "fig")
    fig = cap.figs[0]
    t0, t1 = max(SMALL_EXTRA.jump_t - 70, 0), min(SMALL_EXTRA.jump_t + 100, SMALL_EXTRA.T)
    lines_a = {ln.get_label(): ln for ln in fig.axes[0].get_lines()}
    for lab in (ONE_TRUE, ONE_ML, INDEP, ROB):
        assert lab in lines_a and np.array_equal(lines_a[lab].get_ydata(), res["example"]["fits"][lab][0][t0:t1])
    curve = {d["c"]: d for d in res["curve"]}
    finite = [d for d in res["curve"] if np.isfinite(d["c"])]
    for axb, key in zip(fig.axes[1:3], ("rmse", "delay")):
        ln = {line.get_label(): line for line in axb.get_lines()}
        assert all(name in ln for name in (CURVE_NAME, INF_NAME, PRIOR_NAME, TUNED_NAME))
        assert ln[INF_NAME].get_ydata()[0] == curve[np.inf][key]
        assert ln[PRIOR_NAME].get_xdata()[0] == C_PRIOR and ln[PRIOR_NAME].get_ydata()[0] == curve[C_PRIOR][key]
        assert ln[TUNED_NAME].get_xdata()[0] == 2.25 and ln[TUNED_NAME].get_ydata()[0] == curve[2.25][key]
        assert list(ln[CURVE_NAME].get_xdata()) == [d["c"] for d in finite]
        assert list(ln[CURVE_NAME].get_ydata()) == [d[key] for d in finite]
        for lab, name in REFERENCE_LINES:
            assert name in ln and ln[name].get_ydata()[0] == res["metrics"][lab][key][0]


@pytest.fixture(scope="module")
def fig1(tmp_path_factory):
    """Figure 1 of the fabricated result, with the calls that drew it."""
    res = full_result()
    with capture_figures() as cap:
        figure_exp1(res, FABRICATED_CFG, tmp_path_factory.mktemp("fig1") / "fig")
    return res, cap


def window():
    jt, T = FABRICATED_CFG.jump_t, FABRICATED_CFG.T
    return max(jt - 70, 0), min(jt + 100, T)


def test_figure_1a_pairs_each_legend_entry_with_the_line_it_names(fig1):
    res, cap = fig1
    t0, t1 = window()
    ax = cap.figs[0].axes[0]
    lines_a = {ln.get_label(): ln for ln in ax.get_lines()}
    leg = [c for c in cap.calls if c[0] == "Axes.legend" and c[1] is ax][-1]
    handles, labels = leg[2][0], leg[2][1]
    named = {INDEP_NAME: INDEP, "Kalman, $R = 1$": ONE_TRUE, ML_NAME: ONE_ML}
    for h, lab in zip(handles, labels):
        if lab in named:
            assert h is lines_a.get(named[lab]), lab
        elif lab.startswith("robust, tuned"):
            assert isinstance(h, tuple) and h[-1] is lines_a.get(ROB)
        elif lab == "true state $x_t$":
            assert np.array_equal(h.get_ydata(), res["example"]["x"][t0:t1])
        elif lab == "outlier beyond the axis":
            assert isinstance(h, tuple) and [m.get_marker() for m in h] == ["^", "v"]
    assert sorted(named) == sorted(lab for lab in labels if lab in named)


def test_figure_1a_band_edge_markers_and_ticks(fig1):
    """The band is m +- Z90 sqrt(P) of the robust filter; up markers at the top for readings
    above the axis, down markers at the bottom; no tick in the edge strips."""
    res, cap = fig1
    t0, t1 = window()
    ex = res["example"]
    ax = cap.figs[0].axes[0]
    m_r, P_r = ex["fits"][ROB]
    fb = [c for c in cap.calls if c[0] == "Axes.fill_between" and c[1] is ax][0]
    assert np.allclose(fb[2][1], (m_r - Z90 * np.sqrt(P_r))[t0:t1], rtol=0, atol=1e-12)
    assert np.allclose(fb[2][2], (m_r + Z90 * np.sqrt(P_r))[t0:t1], rtol=0, atol=1e-12)
    lo, hi = ax.get_ylim()
    pad = 0.05 / 1.1 * (hi - lo)
    yy, oo, tt = ex["y"][t0:t1], ex["outlier"][t0:t1], np.arange(t0, t1) + 1
    n_edge = 0
    for c in cap.calls:
        if c[0] == "Axes.scatter" and c[1] is ax and c[3].get("marker") in ("^", "v"):
            n_edge += 1
            mk, ys, xs = c[3]["marker"], np.asarray(c[2][1]), np.asarray(c[2][0])
            beyond = oo & ((yy > hi - pad) if mk == "^" else (yy < lo + pad))
            assert np.array_equal(xs, tt[beyond])
            assert np.all(ys > hi - pad) if mk == "^" else np.all(ys < lo + pad)
    assert n_edge == 2  # the fabricated example has a reading far beyond each edge
    ticks = ax.get_yticks()
    assert len(ticks) >= 3 and np.all((ticks >= lo + pad) & (ticks <= hi - pad))


def test_figure_1bc_legend_entries_and_the_dotted_connector(fig1):
    """Each legend entry is the line with that label, all seven present; the dotted connector
    runs from c = 15 to the c = inf point."""
    res, cap = fig1
    fl = [c for c in cap.calls if c[0] == "Figure.legend"][-1]
    hs, ls = fl[2][0], fl[2][1]
    assert all(lab == "" or h.get_label() == lab for h, lab in zip(hs, ls))
    assert sorted(lab for lab in ls if lab) == sorted([CURVE_NAME, PRIOR_NAME, TUNED_NAME, INF_NAME, ML_NAME,
                                                       INDEP_NAME, "Bayes filter (reference)"])
    curve = [d for d in res["curve"] if np.isfinite(d["c"])]
    inf_pt = res["curve"][-1]
    for axb, key in zip(cap.figs[0].axes[1:3], ("rmse", "delay")):
        dotted = [ln for ln in axb.get_lines() if ln.get_linestyle() == ":"]
        assert len(dotted) == 1 and list(dotted[0].get_ydata()) == [curve[-1][key], inf_pt[key]]


def test_figure_2_shows_each_regimes_median_passes_and_errors(tiny_exp2):
    """(b) and (c) show each regime's median passes at their own tolerance, at T up to the
    dodge; (a) shows the c = C2 runs' median errors."""
    fig2, Ts = tiny_exp2.fig, TINY_EXP2.T2_list
    for axb, tol in zip(fig2.axes[1:3], TOLS):
        for row, (label, *_) in zip(tiny_exp2.rows, REGIMES):
            ln = [line for line in axb.get_lines() if line.get_label() == regime_label(label)]
            rr = [row[f"T={T} c={C2:g}"] for T in Ts]
            med = [np.median(np.where(np.array(q["passes"][str(tol)]) < 0, q["maxpass"], q["passes"][str(tol)]))
                   for q in rr]
            assert len(ln) == 1 and list(ln[0].get_ydata()) == med
            assert np.all(np.abs(np.log10(np.asarray(ln[0].get_xdata()) / Ts)) <= 0.16)
    T = TINY_EXP2.T2_mid
    for label, a_med, cap, seed in REGIMES:
        _, y, a, Qs = simulate_selective(np.random.default_rng([2027, seed, T]), TINY_EXP2.n2, T, a_med, cap)
        cp = count_passes(y, Qs, a, C2, min(TINY_EXP2.maxpass, T + 2))
        e2 = np.maximum(np.median(cp["errs"], axis=1), 1e-13)
        assert any(np.array_equal(line.get_ydata(), e2) for line in fig2.axes[0].get_lines()), label


def test_figures_keep_their_details_legible(fig1, tiny_exp2):
    """1(a)'s legend wholly below the x-axis label, and its input-dependent entry marked 'tuned';
    in 1(b)/(c) the tuned-c diamond as large as the c = 2 square; in figure 2 every regime at its
    own, dodged x in (b) and (c), each regime's markers drawn again above every line at the same
    points, and diamonds shrunk by sqrt(2) to the other markers' size."""
    fig = fig1[1].figs[0]
    ax = fig.axes[0]
    canvas = FigureCanvasAgg(fig)  # closing the figure dropped its canvas
    canvas.draw()
    rend = canvas.get_renderer()
    leg_a = ax.get_legend()
    assert leg_a is not None
    assert leg_a.get_window_extent(rend).y1 <= ax.xaxis.label.get_window_extent(rend).y0 + 0.5
    assert any("input-dependent" in t.get_text() and "tuned" in t.get_text() for t in leg_a.get_texts())
    for axb in fig.axes[1:3]:
        marks = {ln.get_label(): ln for ln in axb.get_lines()}
        assert abs(marks[TUNED_NAME].get_markersize() * np.sqrt(2) - marks[PRIOR_NAME].get_markersize()) <= 1e-9
    fig2 = tiny_exp2.fig
    names = [regime_label(label) for label, *_ in REGIMES]
    for axb in fig2.axes[1:3]:
        by_name = {ln.get_label(): ln for ln in axb.get_lines()}
        xs = np.sort(np.array([np.asarray(by_name[n].get_xdata(), float) for n in names]), axis=0)
        assert np.all(xs[1:] / xs[:-1] > 10 ** 0.03)
    for j, axb in enumerate(fig2.axes[:3]):
        lns = axb.get_lines()
        regime = [ln for ln in lns if ln.get_linestyle() != "None" and ln.get_marker() not in ("None", "", None)]
        top = max(ln.get_zorder() for ln in lns if ln.get_linestyle() != "None")
        assert len(regime) == len(REGIMES)
        for ln in regime:
            twins = [o for o in lns if o.get_linestyle() == "None" and o.get_marker() == ln.get_marker()
                     and np.array_equal(o.get_xdata(), ln.get_xdata()) and np.array_equal(o.get_ydata(), ln.get_ydata())
                     and o.get_markevery() == ln.get_markevery() and o.get_zorder() > top]
            size = 2.6 if j == 0 else 3.6
            assert len(twins) == 1
            assert abs(ln.get_markersize() - (size / np.sqrt(2) if ln.get_marker() == "D" else size)) <= 1e-9
