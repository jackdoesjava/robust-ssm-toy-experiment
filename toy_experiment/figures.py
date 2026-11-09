"""Figures 1 and 2."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

from .config import BAYES, C2, C_PRIOR, INDEP, ONE_ML, ONE_TRUE, REGIMES, ROB, TOLS, Z90  # noqa: E402

# Chart style: categorical slots 1-3, an ordinal blue ramp, neutral inks, TrueType fonts
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "font.family": "sans-serif",
         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42}

CURVE_NAME = "robust, $R$ tuned for each $c$"  # legend labels of figure 1 (b) and (c)
INF_NAME = r"$c = \infty$: Kalman, one $R$ (forecast CRPS)"
PRIOR_NAME = f"$c = {C_PRIOR:g}$ fixed"
TUNED_NAME = "tuned $c$"
INDEP_NAME = "Kalman, input-dependent $R_t$, tuned"  # in all three panels
ML_NAME = "Kalman, one $R$ (max. likelihood)"
REF_LINES = ((ONE_ML, BLUE, "-", ML_NAME),  # the reference lines of figure 1 (b) and (c)
             (INDEP, INK2, (0, (4, 2)), INDEP_NAME),
             (BAYES, INK, "-", "Bayes filter (reference)"))
X_INF = 26.0  # where c = inf is drawn in figure 1 (b) and (c), after a gap

REGIME_COLOURS = RAMP + [MUTED]
REGIME_MARKERS = ["o", "s", "^", "D", "v", "P"]


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=INK2, length=2.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def regime_name(label):
    """A regime's label in figure 2's legend."""
    if label == "random walk":
        return "random walk ($a_t = 1$)"
    return "$" + label.replace("<=", r"\leq").replace("~", r"\approx") + "$"


def marker_size(marker, size):
    """matplotlib's "D" is sqrt(2) wider than the other markers at the same size."""
    return size / np.sqrt(2) if marker == "D" else size


# Figure 1
def figure_exp1(res, cfg, stem):
    """Figure 1: (a) the first test sequence around the jump; (b) and (c) the RMSE before the
    jump and the delay after it over the robust scale c. Saved as stem.pdf and stem.png."""
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(6.3, 4.75), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1])
    _example_panel(fig, fig.add_subplot(gs[0, :]), res["example"], cfg)
    panels = (("rmse", "(b) RMSE before the jump"), ("delay", "(c) delay after the jump (steps)"))
    for j, (key, title) in enumerate(panels):
        axb = fig.add_subplot(gs[1, j])
        _curve_panel(axb, res, key, title)
        if j == 0:
            handles, labels = axb.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
    # one legend for (b) and (c), below them, in three rows: the robust curve and its marks;
    # the reference lines; the c = inf point. Listed column by column, as matplotlib fills it.
    layout = [[CURVE_NAME, PRIOR_NAME, TUNED_NAME], [name for *_, name in REF_LINES], [INF_NAME, None, None]]
    blank = Line2D([], [], linestyle="none")
    cells = [layout[r][col] for col in range(3) for r in range(len(layout))]
    fig.legend([by_label[name] if name else blank for name in cells], [name or "" for name in cells],
               loc="outside lower center", ncol=3, frameon=False, columnspacing=1.4, handlelength=1.8)
    fig.draw_without_rendering()  # let the constrained layout settle, so the PDF has the same layout as the PNG
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _example_panel(fig, ax, ex, cfg):
    """Panel (a): readings, truth and four filters' means around the jump, the robust filter's band."""
    jt = cfg.jump_t
    t0, t1 = max(jt - 70, 0), min(jt + 100, cfg.T)
    t = np.arange(t0, t1) + 1
    x, y, o = ex["x"][t0:t1], ex["y"][t0:t1], ex["outlier"][t0:t1]
    shown = [ex["fits"][lab][0][t0:t1] for lab in (ONE_TRUE, ONE_ML, INDEP, ROB)]
    lo = min(x.min() - 3.5, min(s.min() for s in shown) - 0.5)
    hi = max(x.max() + 3.5, max(s.max() for s in shown) + 0.5)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad  # a strip at each edge for the markers of readings beyond the axis

    style_axes(ax)
    ax.axvline(jt + 0.5, color=AXIS, linewidth=0.8)
    ax.text(jt + 1.5, hi - 0.4, "real jump (+5)", color=INK2, va="top", fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    clean = ~o
    h_obs = ax.scatter(t[clean], y[clean], s=6, color=MUTED, linewidths=0, zorder=2)
    inside = o & (y >= lo + pad) & (y <= hi - pad)
    h_out = ax.scatter(t[inside], y[inside], s=16, marker="x", color=INK2, linewidths=0.9, zorder=6)
    for mask, edge, marker in ((o & (y > hi - pad), hi - pad / 2, "^"), (o & (y < lo + pad), lo + pad / 2, "v")):
        if mask.any():
            ax.scatter(t[mask], np.full(mask.sum(), edge), s=16, marker=marker, color=INK2, linewidths=0, zorder=6)
    m_r, P_r = ex["fits"][ROB]
    ax.fill_between(t, (m_r - Z90 * np.sqrt(P_r))[t0:t1], (m_r + Z90 * np.sqrt(P_r))[t0:t1],
                    color=AQUA, alpha=0.18, linewidth=0, zorder=1)
    h_true, = ax.plot(t, x, color=INK, linewidth=1.3, zorder=4)
    h_k1, = ax.plot(t, ex["fits"][ONE_TRUE][0][t0:t1], color=ORANGE, linewidth=1.2, linestyle=(0, (4, 1.5)), zorder=5,
                    label=ONE_TRUE)
    h_ml, = ax.plot(t, ex["fits"][ONE_ML][0][t0:t1], color=BLUE, linewidth=1.3, zorder=5, label=ONE_ML)
    h_in, = ax.plot(t, ex["fits"][INDEP][0][t0:t1], color=INK2, linewidth=1.0, linestyle=(0, (4, 2)), zorder=5,
                    label=INDEP)
    h_rob, = ax.plot(t, m_r[t0:t1], color=AQUA, linewidth=1.3, zorder=5, label=ROB)
    h_edge = (Line2D([], [], marker="^", linestyle="none", color=INK2, markersize=4),
              Line2D([], [], marker="v", linestyle="none", color=INK2, markersize=4))
    h_band = (Patch(facecolor=AQUA, alpha=0.18, edgecolor="none"), h_rob)
    # three rows (data; truth and the one-R Kalman filters; input-dependent and robust),
    # listed column by column because that is how matplotlib fills a legend
    handles = [h_obs, h_true, h_in, h_out, h_k1, h_band, h_edge, h_ml, Line2D([], [], linestyle="none")]
    labels = ["reading $y_t$", "true state $x_t$", INDEP_NAME, "outlier", "Kalman, $R = 1$",
              "robust, tuned (90% band)", "outlier beyond the axis", ML_NAME, ""]
    # anchored a fixed 22 pt below the axes, so the legend clears the x-axis label at any panel height
    ax.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0),
              bbox_transform=ax.transAxes + ScaledTranslation(0, -22 / 72, fig.dpi_scale_trans),
              columnspacing=0.8, handlelength=1.6, handletextpad=0.5,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0)})
    ax.set_ylim(lo, hi)
    # ticks (and their gridlines) well clear of the edge strips, so an off-axis marker never sits on a value
    ax.set_yticks([v for v in MaxNLocator(nbins=5).tick_values(lo + pad, hi - pad)
                   if lo + 1.5 * pad <= v <= hi - 1.5 * pad])
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("step $t$")
    ax.set_ylabel("state and readings")
    ax.set_title("(a) the first test sequence around the jump", loc="left")


def _curve_panel(axb, res, key, title):
    """Panel (b) or (c): metric `key` of the robust filter over c, with the reference lines."""
    curve = [d for d in res["curve"] if np.isfinite(d["c"])]
    inf_pt = next(d for d in res["curve"] if not np.isfinite(d["c"]))
    cs = np.array([d["c"] for d in curve])
    style_axes(axb)
    for label, col, ls, name in REF_LINES:
        axb.axhline(res["metrics"][label][key][0], color=col, linewidth=1.0, linestyle=ls, label=name, zorder=2)
    vals = [d[key] for d in curve]
    axb.plot(cs, vals, color=AQUA, linewidth=1.5, marker="o", markersize=3.2, label=CURVE_NAME, zorder=3)
    axb.plot([cs[-1], X_INF], [vals[-1], inf_pt[key]], color=AQUA, linewidth=1.0, linestyle=":", zorder=3)
    axb.plot([X_INF], [inf_pt[key]], marker="o", markersize=3.6, markerfacecolor="white", color=AQUA,
             linestyle="none", label=INF_NAME, zorder=4)
    for c_mark, marker, name in ((C_PRIOR, "s", PRIOR_NAME), (res["fitted"].c_rob, "D", TUNED_NAME)):
        x_mark = c_mark if np.isfinite(c_mark) else X_INF
        v = next(d[key] for d in res["curve"] if d["c"] == c_mark)
        axb.plot([x_mark], [v], marker=marker, markersize=marker_size(marker, 6),
                 markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.9, linestyle="none", label=name,
                 zorder=5)
    axb.set_xscale("log")
    ticks = [0.5, 1, 2, 4, 8, 15, X_INF]
    axb.set_xticks(ticks)
    axb.set_xticklabels(["0.5", "1", "2", "4", "8", "15", r"$\infty$"])
    axb.minorticks_off()
    axb.set_xlabel("robust scale $c$ (smaller = more robust)")
    axb.set_title(title, loc="left")
    top = max(max(vals), inf_pt[key], *(res["metrics"][lab][key][0] for lab, *_ in REF_LINES))
    if key == "rmse":
        bottom = min(min(vals), res["metrics"][BAYES][key][0])
        axb.set_ylim(max(0.0, bottom - 0.15 * (top - bottom)), top + 0.08 * (top - bottom))
    else:
        axb.set_ylim(0, 1.08 * top)
        axb.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


# Figure 2
def figure_exp2(rows, curves, cfg, stem):
    """Figure 2: (a) the median error after each pass at T = T2_mid; (b) and (c) the median
    passes to each tolerance over T. Saved as stem.pdf and stem.png."""
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.95), layout="constrained")
    _error_panel(axes[0], curves, cfg.T2_mid)
    titles = ("(b) passes to 1% of sd", "(c) passes to $10^{-8}$ sd")
    for axb, tol, title in zip(axes[1:], TOLS, titles):
        _passes_panel(axb, rows, cfg.T2_list, tol, title)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False, columnspacing=1.0,
               handlelength=1.6, handletextpad=0.4)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _error_panel(ax, curves, mid):
    """Panel (a): each regime's median error after each pass."""
    for i, ((label, *_), col, mk) in enumerate(zip(REGIMES, REGIME_COLOURS, REGIME_MARKERS)):
        e = np.maximum(curves[label], 1e-13)
        # markers on every third pass, starting one pass later for each regime, and drawn again above
        # every line, so curves that coincide still show each regime's markers
        ax.plot(np.arange(1, len(e) + 1), e, color=col, linewidth=1.3, marker=mk, markersize=marker_size(mk, 2.6),
                markevery=(i % 3, 3))
        ax.plot(np.arange(1, len(e) + 1), e, color=col, linestyle="none", marker=mk, markersize=marker_size(mk, 2.6),
                markevery=(i % 3, 3), zorder=3)
    style_axes(ax)
    kmax = len(max(curves.values(), key=len))
    for tol, x_text, ha, text in ((TOLS[0], kmax, "right", "1% of sd"), (TOLS[1], 1, "left", "$10^{-8}$ sd")):
        ax.axhline(tol, color=AXIS, linewidth=0.8)
        ax.text(x_text, tol * 2.5, text, color=INK2, fontsize=7.5, ha=ha)
    ax.set_yscale("log")
    ax.set_ylim(1e-12, None)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("pass $k$")
    ax.set_ylabel("largest error, in posterior sd\n(median over sequences)")
    ax.set_title("(a) error after each pass", loc="left")
    ax.text(0.97, 0.97, f"$T = {mid:,}$".replace(",", "{,}"), transform=ax.transAxes, ha="right", va="top",
            color=INK2, fontsize=7.5)


def _passes_panel(axb, rows, Ts, tol, title):
    """Panel (b) or (c): each regime's median passes to tolerance tol over the lengths Ts."""
    for i, ((label, *_), col, mk) in enumerate(zip(REGIMES, REGIME_COLOURS, REGIME_MARKERS)):
        r = next(rw for rw in rows if rw["regime"] == label)
        med = [np.median(np.where(np.array(r[f"T={T} c={C2:g}"]["passes"][str(tol)]) < 0,
                                  r[f"T={T} c={C2:g}"]["maxpass"],
                                  r[f"T={T} c={C2:g}"]["passes"][str(tol)])) for T in Ts]
        dodge = 10 ** (0.06 * (i - 2.5))  # sideways offset of about 2.6 pt per regime, so equal medians stay visible
        axb.plot(np.array(Ts) * dodge, med, color=col, linewidth=1.3, marker=mk, markersize=marker_size(mk, 3.6),
                 label=regime_name(label))
        # the markers again, above every regime's line, so no other line runs through them
        axb.plot(np.array(Ts) * dodge, med, color=col, linestyle="none", marker=mk, markersize=marker_size(mk, 3.6),
                 zorder=3)
    style_axes(axb)
    axb.set_xscale("log")
    axb.set_xticks(Ts)
    axb.set_xticklabels([f"{T:,}" for T in Ts])
    axb.minorticks_off()
    axb.yaxis.set_major_locator(MaxNLocator(integer=True))
    axb.set_xlabel("sequence length $T$")
    axb.set_ylabel("passes (median over sequences)")
    axb.set_title(title, loc="left")
    axb.set_ylim(bottom=0)
