#!/usr/bin/env python3
"""Publication figures, built from the committed results files.

Nothing is recomputed here and nothing is hand-entered: every figure reads
results/*.json, so a figure cannot drift away from the number it illustrates.
Run the experiments first, then this.

    python3 -m research.figures

Writes PDF (for typesetting) and PNG (for previewing) into paper/figures/.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from oracle import paths

RESULTS = paths.ROOT / "results"
OUT = paths.ROOT / "paper" / "figures"

# IEEE two-column geometry, in inches. Figures are drawn at the exact size they
# are placed, so LaTeX never rescales them and no font is silently shrunk.
COL_W = 3.45          # one column
FULL_W = 7.10         # both columns

# Validated categorical slots 1-3 (all-pairs clean under CVD simulation).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
GRID = "#e4e3df"


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "DejaVu Sans", "font.size": 7,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.frameon": False, "legend.fontsize": 6.5,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _load(stem):
    f = RESULTS / stem
    if not f.exists():
        print(f"  skip — {f.name} not found; run the experiment first")
        return None
    return json.loads(f.read_text())


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf / .png")


# ── Figure 1 — the overlap artefact ───────────────────────────────────
def fig_autocorrelation():
    """Overlapping returns look autocorrelated; the correlation dies at lag k."""
    panels = []
    for horizon in ("30m", "60m"):
        d = _load(f"ceiling_test_{horizon}.json")
        if not d:
            continue
        # Name the index explicitly; taking whichever block happens to be first
        # lets an unrelated re-run silently change the figure.
        blk = next((b for b in d if b["index"] == "BANKNIFTY"), d[0])
        panels.append((horizon, blk))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(FULL_W, 2.15), sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (horizon, blk) in zip(axes, panels):
        ac = blk["autocorrelation"]
        lags, k = ac["lags"], ac["horizon_bars"]

        ax.axhline(0, color=INK3, lw=0.8)
        ax.axvline(k, color=INK3, lw=0.8, ls=(0, (4, 3)))
        ax.plot(lags, ac["overlapping"]["acf"], "o-", color=ORANGE, lw=1.6, ms=4)
        ax.plot(lags, ac["non_overlapping"]["acf"], "s-", color=BLUE, lw=1.6, ms=4)

        ax.annotate("overlapping", (lags[0], ac["overlapping"]["acf"][0]),
                    xytext=(6, 4), textcoords="offset points",
                    color=ORANGE, fontsize=6.5, fontweight="bold")
        ax.annotate("non-overlapping", (lags[1], ac["non_overlapping"]["acf"][1]),
                    xytext=(6, -14), textcoords="offset points",
                    color=BLUE, fontsize=6.5, fontweight="bold")
        ax.annotate(f"lag = k = {k}", (k, 0.93), xytext=(5, 0),
                    textcoords="offset points", color=INK2, fontsize=6)

        ax.set_title(f"{blk['index']}, {horizon} horizon  (k = {k} bars)", loc="left")
        ax.set_xlabel("lag (bars)")
        ax.set_xticks(lags)
        ax.grid(axis="y")
        ax.set_ylim(-0.12, 1.02)

    axes[0].set_ylabel("autocorrelation of forward return")
    fig.tight_layout()
    _save(fig, "fig1_autocorrelation")


# ── Figure 2 — what a coinflip posts ──────────────────────────────────
def fig_zero_skill():
    """The session hit-rate distribution of a forecaster with no skill."""
    d = _load("overlap_simulation_30m.json")
    if not d:
        return
    key = next(k for k in d if k.startswith("per_bar_"))
    blk, index = d[key], key.replace("per_bar_", "")

    import numpy as np
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    # The measured distribution itself, as simulated — not a curve fitted to it.
    h = blk["histogram"]
    edges = np.array(h["bin_edges"])
    counts = np.array(h["counts"], dtype=float)
    counts /= counts.sum()
    centres = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    # Bars only. The multimodality is real, not noise: with roughly a dozen
    # independent decisions per session the hit-rate lands on near-discrete
    # values, which is the small effective sample made visible.
    ax.bar(centres, counts, width=width * 0.9, color=BLUE, alpha=0.55, lw=0)

    top = counts.max()
    ax.set_ylim(0, top * 1.34)
    for i, (thr, label) in enumerate(((0.60, "60%"), (0.71, "71%"), (0.85, "85%"))):
        p = blk[f"p_session_ge_{int(thr*100)}"]
        ax.axvline(thr, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
        rate = f"1 in {1/p:,.0f}" if p > 0 else "n/a"
        y = top * (1.31 - 0.13 * i)          # stagger so labels never collide
        ax.annotate(f"{label}: {rate}", (thr, y),
                    xytext=(3, 0), textcoords="offset points",
                    color=ORANGE, fontsize=6, fontweight="bold", va="top")

    ax.set_title(f"Zero-skill forecaster, scored every bar\n"
                 f"({index}, 30m, {blk['simulated_sessions']:,} simulated sessions)",
                 loc="left", fontsize=7)
    ax.set_xlabel("session hit-rate")
    ax.set_ylabel("density")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_yticks([])
    ax.grid(axis="x")
    ax.set_xlim(0.2, 0.9)

    ax.annotate(
        f"stated 95% CI ±{1.96*blk['stated_sd']*100:.0f} pts  ·  "
        f"realised ±{1.96*blk['realised_sd']*100:.0f} pts  ·  "
        f"{blk['understatement_factor']:.2f}× too narrow",
        (0.5, -0.30), xycoords="axes fraction", ha="center",
        color=INK2, fontsize=6.5)

    fig.tight_layout()
    _save(fig, "fig2_zero_skill_distribution")


# ── Figure 3 — required versus achieved ───────────────────────────────
def fig_breakeven():
    """The gap between the accuracy achievable and the accuracy required."""
    d = _load("cost_geometry.json")
    if not d:
        return

    fig, axes = plt.subplots(1, len(d), figsize=(FULL_W, 2.25), sharey=True)
    axes = [axes] if len(d) == 1 else list(axes)

    for ax, blk in zip(axes, d):
        rows = blk["horizons"]
        ys = range(len(rows))
        req = [min(r["breakeven_accuracy"], 1.15) for r in rows]
        ach = [r["achieved_accuracy"] for r in rows]

        for y, r_req in zip(ys, req):
            ax.plot([0.5, r_req], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
        ax.scatter(req, ys, color=ORANGE, s=30, zorder=3, label="required to break even")
        got = [(a, y) for a, y in zip(ach, ys) if a]
        if got:
            ax.scatter([a for a, _ in got], [y for _, y in got],
                       color=BLUE, s=30, zorder=3, label="best model achieved")

        ax.axvline(1.0, color=INK3, lw=0.8, ls=(0, (2, 3)))
        for y, r in zip(ys, rows):
            if r["breakeven_accuracy"] > 1.0:
                ax.annotate("impossible", (1.02, y), fontsize=6, va="center",
                            color=INK2, style="italic")

        ax.set_yticks(list(ys))
        ax.set_yticklabels([r["horizon"] for r in rows])
        ax.invert_yaxis()
        ax.set_title(f"{blk['index']}   ({blk['cost_bp']:.0f} bp round trip)", loc="left")
        ax.set_xlabel("directional accuracy")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlim(0.45, 1.2)
        ax.grid(axis="x")

    axes[0].set_ylabel("horizon")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig3_breakeven_gap")


# ── Figure 4 — every model family ─────────────────────────────────────
def fig_model_zoo():
    """Nine families with exact intervals, against chance and against noise."""
    d = _load("model_zoo_30m.json")
    if not d:
        return

    fig, axes = plt.subplots(1, len(d), figsize=(FULL_W, 2.6), sharey=True)
    axes = [axes] if len(d) == 1 else list(axes)

    for ax, blk in zip(axes, d):
        models = [m for m in blk["models"] if m["model"] != "majority-class baseline"][::-1]
        ys = range(len(models))

        ax.axvline(0.5, color=INK3, lw=1.0)
        for y, m in zip(ys, models):
            lo, hi = m["ci95"]
            ax.plot([lo, hi], [y, y], color=BLUE, lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter([m["accuracy"]], [y], color=BLUE, s=22, zorder=3)
            if m.get("shuffled_accuracy"):
                ax.scatter([m["shuffled_accuracy"]], [y], marker="|",
                           color=ORANGE, s=60, lw=1.5, zorder=3)

        ax.set_yticks(list(ys))
        ax.set_yticklabels([m["model"] for m in models], fontsize=6.5)
        ax.set_title(f"{blk['index']}   ({blk['bars']:,} bars, {blk['folds']} purged folds)",
                     loc="left")
        ax.set_xlabel("out-of-sample directional accuracy")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="x")
        ax.set_xlim(0.465, 0.535)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", lw=1.6, markersize=4,
               label="real labels, point estimate and 95% CI"),
        Line2D([], [], color=ORANGE, marker="|", lw=0, markersize=8, mew=1.5,
               label="shuffled-label control"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig4_model_zoo")


def main():
    _style()
    print("building figures from results/ …")
    fig_autocorrelation()
    fig_zero_skill()
    fig_breakeven()
    fig_model_zoo()


if __name__ == "__main__":
    main()
