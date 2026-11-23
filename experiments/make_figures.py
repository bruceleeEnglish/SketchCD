"""Figures and tables of the paper, generated from the raw records in results/.

    python experiments/make_figures.py                   (Python >= 3.12; matplotlib, numpy)
    SKETCHCD_OUT=<dir> python experiments/make_figures.py   (write into <dir> instead of paper_output/)

Datasets: the eight high-dimensional collections (GIST, FiQA, TREC-COVID, DBpedia-3S, CIFAR-10, CIFAR-100, SVHN,
Tiny-ImageNet), the two ten-million-vector collections (BIGANN-10M, DEEP-10M), and BIGANN-100M for the scale experiment.

Output (under paper_output/ by default):
  figures/*.pdf     dominance, tradeoff, qps, scaling, dims, sensitivity, coldstart
  tables/*.tex      main_rows, recallk_rows, scale100m_rows, breakdown_rows, starve_rows, exact_rows, ablation_rows, cdkm_rows
  preview/*.png     the figures as images
  numbers.txt       every number that the text of the paper quotes

Conventions of the plots: a restrained palette in the manner of database papers. SketchCD is the one strong colour, blue
#2166AC, with filled circles and a heavier line. Every competitor is thin, in a darker and less saturated hue, with
white-filled markers of its own shape: FullSuperKMeans terracotta #C0652F (SampleSuperKMeans: same hue, dashed),
HierSuperKMeans violet #6F5499, QuantSuperKMeans maroon #8E2A35, BatchedHartigan orchid #A45C93, e-means teal #3D8F7B
(EMeans-2: dashed), FaissKMeans grey #666666 (PCA-FaissKMeans: dashed). A dashed line always means "the cheaper
configuration of the method drawn solid in the same colour", and no two methods share a marker, so the figures can be read
without colour. The analysis datasets (all lines are SketchCD) use slate #3B4A5A, ochre #B98A2C, rose #B0557A.

Layout: the text under an axis is always the quantity on that axis. In the grids of small multiples (trade-off curves,
query throughput) the dataset is therefore the title above its panel, and the x and y labels are given once. The row chart
(dominance) is laid out like a table: names flush left, numbers flush right, no frame.

The scan budget of a dataset is set by the seven configurations FullSuperKMeans ... FaissKMeans. BatchedHartigan runs
without a single size-aware move are left out (DEEP-10M, see the comment where exp01_extra_baselines.jsonl is loaded).

Typeface: Linux Libertine, the body face of the paper. The script takes it from experiments/fonts/*.ttf (or from the folder
in SKETCHCD_FONTS) if present, otherwise converts it once from a TeX installation (kpsewhich), otherwise falls back to
DejaVu Sans.
"""
import json
import math
import os
import shutil
import statistics as st
import subprocess
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository
RES = os.environ.get("SKETCHCD_RESULTS", os.path.join(ROOT, "results"))
_o = os.environ.get("SKETCHCD_OUT", os.path.join(ROOT, "paper_output"))
FIG, TAB, PREV = (os.path.join(_o, x) for x in ("figures", "tables", "preview"))
NUMBERS = os.path.join(_o, "numbers.txt")
for _d in (FIG, TAB, PREV):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------- typeface: Biolinum from the TeX installation
# Matplotlib mis-measures the CFF outlines of the Libertine OTFs (giant glyph boxes), so they are converted once to
# TrueType outlines (fontTools, cubic -> quadratic) and cached next to this script.
FONTDIR = os.environ.get("SKETCHCD_FONTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts"))


def _otf_to_ttf(src, dst):
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable
    f = TTFont(src)
    order = f.getGlyphOrder()
    gs = f.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    for g in order:
        pen = TTGlyphPen(gs)
        gs[g].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
        glyf[g] = pen.glyph()
    f["glyf"], f["loca"] = glyf, newTable("loca")
    maxp = f["maxp"]
    maxp.tableVersion = 0x00010000
    for attr in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
                 "maxSizeOfInstructions", "maxComponentElements", "maxComponentDepth"):
        setattr(maxp, attr, 0)
    maxp.maxZones = 1
    del f["CFF "]
    if "VORG" in f:
        del f["VORG"]
    f.sfntVersion = "\x00\x01\x00\x00"
    f["head"].indexToLocFormat = 0
    post = f["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    f.save(dst)


def _register_biolinum():
    """Registers Linux Libertine (the serif body face of acmart; figure text then matches the paper) and Biolinum."""
    os.makedirs(FONTDIR, exist_ok=True)
    names = ("LinLibertine_R", "LinLibertine_RB", "LinLibertine_RI", "LinBiolinum_R", "LinBiolinum_RB", "LinBiolinum_RI")
    kp = shutil.which("kpsewhich")
    for nm in names:
        ttf = os.path.join(FONTDIR, nm + ".ttf")
        if not os.path.exists(ttf) and kp:
            r = subprocess.run([kp, nm + ".otf"], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                try:
                    _otf_to_ttf(r.stdout.strip(), ttf)
                except Exception as e:  # noqa: BLE001
                    print("font conversion failed:", e)
        if os.path.exists(ttf):
            fm.fontManager.addfont(ttf)
    return "Linux Libertine O" if os.path.exists(os.path.join(FONTDIR, names[0] + ".ttf")) else None


FONT = _register_biolinum() or "DejaVu Sans"

# ---------------------------------------------------------------- palette (validated; see docstring)
ACC = "#2166ac"      # SketchCD: the one strong colour
SKM = "#c0652f"      # FullSuperKMeans (and, dashed, SampleSuperKMeans): terracotta
HIC = "#6f5499"      # HierSuperKMeans: violet
QUC = "#8e2a35"      # QuantSuperKMeans: maroon
EMC = "#3d8f7b"      # e-means: teal
FAI = "#666666"      # FaissKMeans (and, dashed, PCA-FaissKMeans)
BHC = "#a45c93"      # BatchedHartigan: orchid
SAC = SKM
G1, G2, G3 = "#4a4946", "#7d7b75", "#aaa8a0"
NCC = G1                  # nearest-centroid variant of SketchCD (rule comparison only)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8C8C8C"
GRID, AXIS = "#e4e4e4", "#333333"
DS3 = {"gist": ("#3b4a5a", "o", "-"), "treccovid": ("#b98a2c", "s", "-"), "svhn": ("#b0557a", "^", "-")}  # analysis datasets
LW, LW_OURS, MEW = 0.8, 1.3, 0.65   # line widths (pt) of the competitors and of SketchCD; marker edge width

plt.rcParams.update({
    "font.family": FONT, "font.size": 7.8, "axes.labelsize": 7.8, "axes.titlesize": 7.8,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 7.2,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic", "mathtext.bf": FONT + ":bold",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.55, "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
    "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": False, "ytick.right": False,
    "xtick.major.width": 0.55, "ytick.major.width": 0.55,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": "-", "hatch.linewidth": 0.5,
    "axes.spines.top": True, "axes.spines.right": True, "axes.axisbelow": True,
    "lines.linewidth": LW, "lines.markersize": 3.0, "lines.markeredgewidth": MEW, "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "butt", "lines.dashed_pattern": (3.2, 1.6), "lines.dashdot_pattern": (4.6, 1.4, 1.0, 1.4),
    "lines.dotted_pattern": (1.0, 1.4),
    "legend.frameon": False, "legend.handlelength": 2.0, "legend.handletextpad": 0.5, "legend.columnspacing": 1.3,
    "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "figure.dpi": 150,
})
G = FuncFormatter(lambda v, _: f"{v:g}")
X_ = FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$")
KF = FuncFormatter(lambda v, _: f"{v / 1000:g}K" if v >= 1000 else f"{v:g}")

# SketchCD: filled circles, heavier line, on top. Competitors: thin lines, white-filled markers in the line colour, so
# that crossing curves stay readable. Dashed = the cheaper configuration of the method drawn solid in the same colour.
STYLE = {
    "SketchCD": dict(color=ACC, marker="o", ls="-", lw=LW_OURS, ms=3.3, z=8, filled=True),
    "SketchCD-Fast": dict(color="#8db4dd", marker="o", ls="-", lw=LW, ms=3.0, z=7, filled=True),
    "FullSuperKMeans": dict(color=SKM, marker="s", ls="-", lw=LW, ms=2.7, z=5),
    "SampleSuperKMeans": dict(color=SAC, marker="p", ls="--", lw=LW, ms=3.3, z=4),
    "HierSuperKMeans": dict(color=HIC, marker="^", ls="-", lw=LW, ms=3.1, z=4),
    "QuantSuperKMeans": dict(color=QUC, marker="D", ls="-", lw=LW, ms=2.5, z=4),
    "EMeans-2": dict(color=EMC, marker="v", ls="--", lw=LW, ms=3.1, z=5),
    "EMeans-5": dict(color=EMC, marker="v", ls="-", lw=LW, ms=3.1, z=5),
    "FaissKMeans": dict(color=FAI, marker="x", ls="-", lw=LW, ms=2.9, z=3),
    "PCA-FaissKMeans": dict(color=FAI, marker="x", ls="--", lw=LW, ms=2.9, z=3),
    "BatchedHartigan": dict(color=BHC, marker="h", ls="-", lw=LW, ms=3.1, z=4),
}
ORDER = ["FaissKMeans", "PCA-FaissKMeans", "EMeans-2", "EMeans-5", "FullSuperKMeans", "SampleSuperKMeans", "HierSuperKMeans",
         "QuantSuperKMeans", "BatchedHartigan", "SketchCD-Fast", "SketchCD"]
COMP = [m for m in ORDER if not m.startswith("SketchCD")]  # the nine competitor configurations
NEWB = ["BatchedHartigan", "PCA-FaissKMeans"]              # the two baselines of bench/run_extra_baselines.py
SEVEN = [m for m in COMP if m not in NEWB]                 # the seven configurations of the three systems: they set the scan
                                                           # budgets and are the ones with trade-off and throughput curves
LEGEND_ORDER = ["SketchCD", "FullSuperKMeans", "SampleSuperKMeans", "HierSuperKMeans", "QuantSuperKMeans", "EMeans-5", "EMeans-2",
                "FaissKMeans"]


def _mk(s):
    if s.get("filled"):
        return dict(markerfacecolor=s["color"], markeredgecolor="white", markeredgewidth=0.4)
    return dict(markerfacecolor="white", markeredgecolor=s["color"], markeredgewidth=MEW)


def ds_line(ax, ds, xs, ys, **kw):
    """One analysis dataset (every such line is SketchCD): its grey, its white-filled marker, its dash pattern."""
    col, mk, ls = TRIO_STYLE[ds]
    return ax.plot(xs, ys, color=col, marker=mk, linestyle=ls, linewidth=0.9, markersize=2.9, markerfacecolor="white",
                   markeredgecolor=col, markeredgewidth=MEW, **kw)


def ds_handle(ds):
    col, mk, ls = TRIO_STYLE[ds]
    return Line2D([0], [0], color=col, marker=mk, linestyle=ls, linewidth=0.9, markersize=2.9, markerfacecolor="white",
                  markeredgecolor=col, markeredgewidth=MEW, label=TITLE[ds])


RING = dict(linestyle="none", marker="o", markersize=6.6, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.55)  # "the default"


def plot_method(ax, m, xs, ys, **kw):
    s = STYLE[m]
    ax.plot(xs, ys, color=s["color"], marker=s["marker"], linestyle=s["ls"], linewidth=s["lw"], markersize=s["ms"],
            zorder=s["z"], label=kw.get("label", m), **_mk(s))


def handle(m, label=None):
    s = STYLE[m]
    return Line2D([0], [0], color=s["color"], marker=s["marker"], linestyle=s["ls"], linewidth=s["lw"], markersize=s["ms"],
                  label=label or m, **_mk(s))


def panel(ax, letter, x=-0.01, y=1.0):
    """Bold panel letter at the top left, outside the plotting area."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, fontweight="bold", color=INK)


def ygrid_only(ax):
    ax.grid(axis="x", visible=False)


# Row charts (one row per method or dataset): the figure is laid out like a table. The names stand flush left in a column
# of their own, so that their different lengths give a ragged right edge inside the figure and not a ragged left edge of
# the figure; a light band on every second row leads from the name to its marks; numbers stand flush right in a column.
BAND = "#f3f3f1"


def row_axes(fig, x0, x1, y0, y1):
    """Axes at the given position in inches, without a frame: only the x axis is drawn."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])
    ax.patch.set_visible(False)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, labelleft=False)
    ax.tick_params(axis="x", direction="out", length=2.2)
    ax.grid(axis="y", visible=False)
    return ax


def row_x(ax, x_in):
    """x position, in inches from the left edge of the figure, as an axes fraction of ax."""
    W = ax.figure.get_figwidth()
    p = ax.get_position()
    return (x_in / W - p.x0) / p.width


def row_text(ax, x_in, y, s, ha="left", **kw):
    """Text at x_in inches from the left edge of the figure, on row y of ax."""
    kw = dict(dict(va="center", fontsize=6.6, color=INK), **kw)
    return ax.text(row_x(ax, x_in), y, s, transform=ax.get_yaxis_transform(), ha=ha, clip_on=False, **kw)


def row_bands(ax, ys, x0_in=0.0, x1_in=None, height=1.0):
    """Light bands on the rows ys of ax, from x0_in to x1_in inches (default: the full width of the figure)."""
    x1_in = ax.figure.get_figwidth() if x1_in is None else x1_in
    a, b = row_x(ax, x0_in), row_x(ax, x1_in)
    for y in ys:
        ax.add_patch(Rectangle((a, y - height / 2), b - a, height, transform=ax.get_yaxis_transform(), facecolor=BAND,
                               edgecolor="none", clip_on=False, zorder=0))


GRID_L, GRID_R = 0.05, 0.997                              # left and right edge of the panels of a small-multiples grid
GRID_IN = dict(legend=0.18, title=0.125, ticks=0.125, xlabel=0.15, panel=0.45)  # heights in inches


def grid_fig(n, xlabel, ylabel):
    """Compact small multiples, five per row. The dataset is the title of its panel, directly above it; the text under
    an axis is always the quantity on that axis, so the x label and the y label are given once for the whole figure."""
    nc = 5
    nr = (n + nc - 1) // nc
    g = GRID_IN
    between = g["ticks"] + g["title"] + 0.03                # tick labels of the upper row, title of the lower row
    H = g["legend"] + nr * (g["title"] + g["panel"]) + (nr - 1) * (between - g["title"]) + g["ticks"] + g["xlabel"]
    fig, axes = plt.subplots(nr, nc, figsize=(7.1, H), squeeze=False)
    bottom, top = (g["ticks"] + g["xlabel"]) / H, 1 - (g["legend"] + g["title"]) / H
    fig.subplots_adjust(left=GRID_L, right=GRID_R, bottom=bottom, top=top, wspace=0.22, hspace=between / g["panel"])
    fig.supxlabel(xlabel, x=(GRID_L + GRID_R) / 2, y=0.0, va="bottom", fontsize=7.8)
    fig.supylabel(ylabel, x=0.004, y=(bottom + top) / 2, fontsize=7.8)
    for j in range(n, nr * nc):
        axes[j // nc, j % nc].set_visible(False)
    return fig, axes, nc


def subcap_grid(ax, j, ds):
    """Panel title "(a) GIST" directly above a panel of a small-multiples grid."""
    ax.set_title(f"({chr(97 + j)}) {SHORT[ds]}", fontsize=7.6, pad=2.6)
    ax.tick_params(labelsize=6.8)


def grid_legend(fig, handles):
    many = len(handles) > 6
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=((GRID_L + GRID_R) / 2, 1.0),
               fontsize=6.9 if many else 7.2, columnspacing=1.0 if many else 1.4, handlelength=1.9 if many else 2.2,
               handletextpad=0.35, borderaxespad=0, borderpad=0.15)


def log_time_axis(ax):
    ax.set_xscale("log")
    x0, x1 = ax.get_xlim()
    cand = [x for x in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if x0 <= x <= x1]
    ax.set_xticks(cand[::2] if len(cand) > 5 else cand)
    ax.xaxis.set_major_formatter(G)
    ax.xaxis.set_minor_formatter(NullFormatter())


def fmt_t(t):
    return f"{t:.2f}" if t < 10 else f"{t:.1f}" if t < 100 else f"{t:.0f}"


def padnum(s, n):
    """A number set in a centred column: invisible leading digits bring every entry of the column to the width of n characters."""
    return (r"\phantom{" + "0" * (n - len(s)) + "}" + s) if len(s) < n else s


def ours_row(cells):
    """A SketchCD row on a light-blue background, without colour overhang at the table edges (ICDE table style)."""
    return "\\oursL " + cells[0] + " & " + " & ".join("\\ours " + c for c in cells[1:-1]) + " & \\oursR " + cells[-1] + " \\\\"


def load(*names):
    rs = []
    for name in names:
        p = os.path.join(RES, name)
        if os.path.exists(p):
            rs += [json.loads(l) for l in open(p) if l.strip()]
    return rs


def mean(rs, f):
    return st.mean(f(r) for r in rs)


def gmean(xs):
    return math.exp(st.mean(math.log(x) for x in xs))


num = open(NUMBERS, "w", encoding="utf-8")


def out(*a):
    print(*a, file=num)


def wtab(name, text):
    """A LaTeX table body in tables/."""
    with open(os.path.join(TAB, name), "w") as f:
        f.write(text)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(PREV, name + ".png"), dpi=220)
    plt.close(fig)


TITLE = {"gist": "GIST", "fiqa": "FiQA", "treccovid": "TREC-COVID", "dbp3s": "DBpedia-3S", "cifar10": "CIFAR-10",
         "cifar100": "CIFAR-100", "svhn": "SVHN", "tinyimagenet": "Tiny-ImageNet", "bigann10m": "BIGANN-10M", "deep10m": "DEEP-10M"}
SHORT = dict(TITLE, tinyimagenet="Tiny-ImageNet")
HIGH = ["gist", "fiqa", "treccovid", "dbp3s", "cifar10", "cifar100", "svhn", "tinyimagenet"]
TEN = ["bigann10m", "deep10m"]
ALLD = HIGH + TEN  # the ten datasets of the paper
TRIO = ["gist", "treccovid", "svhn"]  # analysis experiments: one dataset of each kind
DIMS = {"gist": 960, "fiqa": 1024, "treccovid": 1024, "dbp3s": 1536, "cifar10": 3072, "cifar100": 3072, "svhn": 3072,
        "tinyimagenet": 12288, "bigann10m": 128, "deep10m": 96}
TRIO_STYLE = DS3
GEMM_MADD = 5.5e11  # multiply-adds per second (1100 GFLOPS) at which one exhaustive assignment pass is priced; see THROUGHPUT below


def dlabel(ds):
    d = DIMS[ds]
    return f"{d:,}".replace(",", "{,}") if d >= 10000 else str(d)


# ================================================================ main records (experiments/bench/run_methods.py)
MAIN = defaultdict(list)
for r in load("exp01_main.jsonl"):
    if r["dataset"] in ALLD:
        MAIN[(r["dataset"], r["method"])].append(r)
# the seed-42 runs of the 10M sets were made without the reference; the runs of the other seeds computed it (same K, same data)
JREF = {}
for (ds, m), rs in MAIN.items():
    for r in rs:
        if r.get("Jref"):
            JREF[ds] = r["Jref"]
for (ds, m), rs in MAIN.items():
    for r in rs:
        if not r.get("Jref") and ds in JREF:
            r["Jref"] = JREF[ds]
            r["excess"] = 100.0 * (r["wcss"] - JREF[ds]) / JREF[ds]

# ---- two more baselines in the same harness and session protocol (experiments/bench/run_extra_baselines.py), three seeds each.
# BatchedHartigan is the fork of SuperKMeans with the size-aware rule in its pruned pass. The library switches that pass
# off for d < 128 and then runs plain Lloyd iterations: on DEEP-10M (d = 96) no iteration makes a single size-aware move,
# and WCSS and CV equal those of FullSuperKMeans. Such a run is not BatchedHartigan and is left out.
# PCA-FaissKMeans exists only where SketchCD uses a sketch (d > 256).
BH_DROPPED, EXTRA_JREF = [], {}
for r in load("exp01_extra_baselines.jsonl"):
    if r["dataset"] in ALLD and r["method"] in NEWB:
        EXTRA_JREF[r["dataset"]] = r["Jref"]
        if r["method"] == "BatchedHartigan" and not any(r["parts"]["moves"]):
            BH_DROPPED.append(f"{r['dataset']} seed {r['seed']} ({r['build_s']:.0f}s, excess {r['excess']:.2f}, cv {r['cv']:.3f})")
            continue
        MAIN[(r["dataset"], r["method"])].append(r)


SCAN = ("0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1")


def _mean_comp_r10(ds, s, comp=None):
    v = [st.mean(r["R10_scan"][s] for r in MAIN[(ds, m)]) for m in (comp or SEVEN) if MAIN.get((ds, m))]
    return st.mean(v)


# Scan budget of each dataset: the smallest standard budget at which the competitors reach Recall@10 >= 90% on average.
# Datasets differ widely in difficulty (raw pixels need ~5%, the 10M collections exceed 98% at 1%), so one budget for
# all would compare some datasets at 65% recall and others in saturation.
BUD = {}
for _ds in ALLD:
    BUD[_ds] = next(s for s in SCAN if _mean_comp_r10(_ds, s) >= 0.90)


def bud(tag):
    for pre, ds in (("gist", "gist"), ("deep", "deep10m"), ("tiny", "tinyimagenet"), ("bigann", "bigann10m")):
        if tag.startswith(pre):
            return BUD[ds]
    return BUD[tag]


def pct(s):
    return f"{100 * float(s):g}"


def R(r, k=10, s=None):
    return r[f"R{k}_scan"][s or bud(r["dataset"])]


def cell(ds, m, f):
    rs = MAIN.get((ds, m))
    return mean(rs, f) if rs else None


def T(ds, m):
    return cell(ds, m, lambda r: r["build_s"])


def RC(ds, m, k=10, s=None):
    v = cell(ds, m, lambda r: R(r, k, s))
    return None if v is None else 100 * v


def EX(ds, m):
    return cell(ds, m, lambda r: r["excess"])


out("== SCAN BUDGETS: " + ", ".join(f"{ds} {pct(BUD[ds])}%" for ds in ALLD))
out("   (rule on the seven configurations of the three systems; with all nine it gives: " + ", ".join(
    f"{ds} {pct(next(s for s in SCAN if _mean_comp_r10(ds, s, COMP) >= 0.90))}%" for ds in ALLD) + ")")
out("   reference of the extra-baseline runs relative to the one of the main runs: " + ", ".join(
    f"{ds} {EXTRA_JREF[ds] / JREF[ds] - 1:+.1e}" for ds in ALLD if ds in EXTRA_JREF and ds in JREF))
out("== MAIN TABLE (mean of seeds; FaissKMeans seed 42): n | time s | R@10 % (sample sd over seeds) | WCSS excess % | cv | tiny % | R@100 % | path")
out("   BatchedHartigan runs left out (no size-aware move in any iteration, d < 128): " + ("; ".join(BH_DROPPED) or "none"))
for ds in ALLD:
    out(f"--- {ds}")
    for m in ORDER:
        rs = MAIN.get((ds, m))
        if rs:
            path = ",".join(sorted({r["parts"].get("path", "") for r in rs if isinstance(r.get("parts"), dict)}))
            sd = st.stdev([R(r) for r in rs]) if len(rs) > 1 else 0.0
            out(f"{m:18s} n={len(rs)} {T(ds, m):8.2f} {RC(ds, m):6.2f} (sd {100 * sd:.2f}) {EX(ds, m):7.2f} "
                f"cv {cell(ds, m, lambda r: r['cv']):.3f} tiny {100 * cell(ds, m, lambda r: r['tiny']):5.2f} R100 {RC(ds, m, 100):6.2f} {path}")


def speedups(dsl, title):
    out(f"\n== SPEEDUPS of SketchCD over {title}: geo-mean (range) | dR@10 mean (min) | d-excess mean (min) | R@10 not lower")
    for m in ORDER[:-1]:
        sp, dr, de, ok = [], [], [], 0
        for ds in dsl:
            if T(ds, m) is None or T(ds, "SketchCD") is None:
                continue
            sp.append(T(ds, m) / T(ds, "SketchCD"))
            dr.append(RC(ds, "SketchCD") - RC(ds, m))
            de.append(EX(ds, "SketchCD") - EX(ds, m))
            ok += dr[-1] >= 0
        if sp:
            out(f"{m:18s} {gmean(sp):6.2f}x ({min(sp):.1f}-{max(sp):.1f}) | {st.mean(dr):+.2f} ({min(dr):+.2f}) | "
                f"{st.mean(de):+.2f} ({max(de):+.2f}) | {ok}/{len(sp)}")


speedups(ALLD, "the ten datasets")
speedups(HIGH, "the eight high-dimensional datasets")
speedups(TEN, "the two 10M datasets")
out("\n-- datasets on which EMeans-5 has the lower WCSS: " + ", ".join(ds for ds in ALLD if EX(ds, "EMeans-5") is not None and EX(ds, "EMeans-5") < EX(ds, "SketchCD")))
out("-- CV of SketchCD: %.2f-%.2f; of EMeans-5: %.2f-%.2f" % (
    min(cell(ds, "SketchCD", lambda r: r["cv"]) for ds in ALLD), max(cell(ds, "SketchCD", lambda r: r["cv"]) for ds in ALLD),
    min(cell(ds, "EMeans-5", lambda r: r["cv"]) for ds in ALLD), max(cell(ds, "EMeans-5", lambda r: r["cv"]) for ds in ALLD)))
out("-- fastest competitor per dataset: " + "; ".join(
    f"{ds}: {min(COMP, key=lambda m: T(ds, m) or 1e9)} {min(T(ds, m) for m in COMP if T(ds, m)):.1f}s vs {T(ds, 'SketchCD'):.2f}s "
    f"({min(T(ds, m) for m in COMP if T(ds, m)) / T(ds, 'SketchCD'):.1f}x)" for ds in ALLD))
out("-- best competitor R@10 per dataset: " + "; ".join(
    f"{ds}: {max(COMP, key=lambda m: RC(ds, m) or -1)} {max(RC(ds, m) for m in COMP if RC(ds, m)):.2f} vs {RC(ds, 'SketchCD'):.2f}"
    for ds in ALLD))

# ---------------------------------------------------------------- returned centroids vs the means of the returned lists
# The WCSS excess of the paper uses the centroids that a method returns (an index routes its queries by them). The records
# of experiments/bench/centroid_gap.py give the part of a method's WCSS that the means of its lists would remove; in
# points of the excess this is 100 * gap / Jref. SketchCD, e-means, and PCA-FaissKMeans return the means of their lists.
GAP = defaultdict(list)
for r in load("exp01_centroid_gap.jsonl"):
    if r["dataset"] in ALLD and MAIN.get((r["dataset"], r["method"])):
        GAP[(r["dataset"], r["method"])].append(r)
if GAP:
    _dev = 0.0
    for (ds, m), gs in GAP.items():
        rec = {r["seed"]: r for r in MAIN[(ds, m)]}
        _dev = max([_dev] + [abs(g["wcss"] / rec[g["seed"]]["wcss"] - 1) for g in gs if g["seed"] in rec])
    out("\n== RETURNED CENTROIDS vs LIST MEANS (mean of seeds): points of WCSS excess that the means of the lists would remove "
        "(range over the datasets) | per dataset: excess -> excess with the means of the lists")
    out(f"   check: the WCSS recomputed from the saved builds equals that of the main records, max relative difference {_dev:.1e}")
    for m in ORDER:
        row = [(ds, st.mean(100 * g["gap"] / JREF[ds] for g in GAP[(ds, m)])) for ds in ALLD if GAP.get((ds, m))]
        if row:
            out(f"{m:18s} n={len(row):2d} {min(x for _, x in row):6.2f}-{max(x for _, x in row):5.2f} | "
                + " ".join(f"{ds} {EX(ds, m):.2f}->{EX(ds, m) - x:.2f}" for ds, x in row))
    for m in ("SampleSuperKMeans", "HierSuperKMeans", "QuantSuperKMeans"):
        v = [EX(ds, m) - st.mean(100 * g["gap"] / JREF[ds] for g in GAP[(ds, m)]) for ds in ALLD if GAP.get((ds, m))]
        out(f"   {m}: excess with the means of the lists {min(v):.2f}-{max(v):.2f} (as returned {min(EX(ds, m) for ds in ALLD if GAP.get((ds, m))):.2f}-"
            f"{max(EX(ds, m) for ds in ALLD if GAP.get((ds, m))):.2f}); SketchCD {min(EX(ds, 'SketchCD') for ds in ALLD):.2f}-{max(EX(ds, 'SketchCD') for ds in ALLD):.2f}")

# ---------------------------------------------------------------- Figure: advantage per competitor (box + dataset dots)
# One row per competitor; each dot is one dataset (circle: high-dimensional, diamond: 10M). Box = quartiles,
# whiskers = full range, centre line = median. Left: build-time speedup of SketchCD; right: its Recall@10 gain.
ADV_ROWS = ["FullSuperKMeans", "SampleSuperKMeans", "HierSuperKMeans", "QuantSuperKMeans", "BatchedHartigan", "EMeans-5", "EMeans-2",
            "FaissKMeans", "PCA-FaissKMeans"]
pts_all = []
NR_ = len(ADV_ROWS)
ADV_ROW, ADV_HEAD, ADV_BOTTOM = 0.15, 0.14, 0.29                                      # inches: a row, the header line, axis labels
ADV_H = ADV_BOTTOM + ADV_ROW * NR_ + ADV_HEAD
ADV_X = dict(name=0.03, L=(0.86, 1.89), numL=2.18, R=(2.26, 3.03), numR=3.3)           # inches from the left edge
fig = plt.figure(figsize=(3.33, ADV_H))
axL = row_axes(fig, *ADV_X["L"], ADV_BOTTOM, ADV_BOTTOM + ADV_ROW * NR_)
axR = row_axes(fig, *ADV_X["R"], ADV_BOTTOM, ADV_BOTTOM + ADV_ROW * NR_)
row_bands(axL, range(NR_ - 1, -1, -2))
rng = np.random.default_rng(3)
for i, m in enumerate(ADV_ROWS):
    y = NR_ - 1 - i
    col = STYLE[m]["color"]
    rows = [(ds, T(ds, m) / T(ds, "SketchCD"), RC(ds, "SketchCD") - RC(ds, m)) for ds in ALLD if T(ds, m) is not None]
    pts_all += [(sp, g) for _, sp, g in rows]
    for ax, k in ((axL, 1), (axR, 2)):
        vals = [r[k] for r in rows]
        bp = ax.boxplot([vals], positions=[y], vert=False, widths=0.54, whis=(0, 100), patch_artist=True, showfliers=False,
                        medianprops=dict(color=col, linewidth=1.6), whiskerprops=dict(color=col, linewidth=0.8),
                        capprops=dict(color=col, linewidth=0.8), boxprops=dict(facecolor=col + "26", edgecolor=col, linewidth=0.8))
        jit = rng.uniform(-0.17, 0.17, len(rows))
        for (ds, sp, g), j in zip(rows, jit):
            ax.plot([sp if k == 1 else g], [y + j], linestyle="none", marker="D" if ds in TEN else "o", markersize=2.7,
                    markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.4, zorder=5)
    row_text(axL, ADV_X["name"], y, m)
    row_text(axL, ADV_X["numL"], y, f"{gmean([r[1] for r in rows]):.1f}" + r"$\times$", ha="right")
    row_text(axR, ADV_X["numR"], y, f"{st.mean(r[2] for r in rows):+.1f}", ha="right")
# header line directly above the first row: the columns of a table (names, geometric-mean speedup, mean gain)
_yh = NR_ - 0.5 + 0.5 * ADV_HEAD / ADV_ROW
row_text(axL, ADV_X["name"], _yh, "Competitor", fontsize=6.0, color=INK2)
row_text(axL, ADV_X["numL"] - 0.13, _yh, "geo. mean", ha="center", fontsize=6.0, color=INK2)
row_text(axR, ADV_X["numR"] - 0.10, _yh, "mean", ha="center", fontsize=6.0, color=INK2)
axL.set_xscale("log")
axL.set_xlim(0.85, 105)
axL.set_xticks([1, 3, 10, 30, 100])
axL.xaxis.set_major_formatter(X_)
axL.xaxis.set_minor_locator(NullLocator())
axL.axvline(1, color=INK2, linewidth=0.8, zorder=2)
axL.set_xlabel("(a) Speedup of SketchCD")
axR.set_xscale("symlog", linthresh=1.0, linscale=1.2)
axR.set_xlim(-0.45, 8)
axR.set_xticks([0, 0.5, 1, 2, 5])
axR.xaxis.set_major_formatter(G)
axR.xaxis.set_minor_locator(NullLocator())
axR.axvline(0, color=INK2, linewidth=0.8, zorder=2)
axR.set_xlabel("(b) Recall@10 gain (points)")
for ax in (axL, axR):
    ax.set_yticks([])
    ax.set_ylim(-0.5, NR_ - 0.5)
# key of the two marks: under the names, beside the axis labels
fig.legend(handles=[Line2D([0], [0], linestyle="none", marker="o", markersize=3.2, markerfacecolor=MUTED, markeredgecolor="white",
                           label="high-dimensional"),
                    Line2D([0], [0], linestyle="none", marker="D", markersize=3.0, markerfacecolor=MUTED, markeredgecolor="white",
                           label="10M vectors")],
           loc="lower left", ncol=1, bbox_to_anchor=(0.0, 0.0), fontsize=6.2, handletextpad=0.1, handlelength=1.0, labelspacing=0.2,
           borderaxespad=0, borderpad=0.1)
save(fig, "dominance")
out("\n== DOMINANCE (advantage figure): points with speedup >= 1 and gain >= 0: %d of %d; min speedup %.2f; min gain %.2f" % (
    sum(x >= 1 and y >= 0 for x, y in pts_all), len(pts_all), min(x for x, _ in pts_all), min(y for _, y in pts_all)))
out("   speedups %.1f-%.1f; points below zero gain: %s" % (
    min(x for x, _ in pts_all), max(x for x, _ in pts_all),
    "; ".join(f"{m} on {ds} {RC(ds, 'SketchCD') - RC(ds, m):+.2f}" for m in ADV_ROWS for ds in ALLD
              if T(ds, m) is not None and RC(ds, "SketchCD") - RC(ds, m) < 0) or "none"))

# ---------------------------------------------------------------- Recall@k
out("\n== RECALL@k, mean over the ten datasets (%), and #datasets where SketchCD >= method")
for s in (None, "0.005", "0.01", "0.02"):
    out(f"-- scan {'per-dataset budget' if s is None else pct(s) + '%'}")
    for m in ORDER:
        vals, wins = [], []
        for k in (1, 10, 30, 50, 100):
            per = [RC(ds, m, k, s) for ds in ALLD if MAIN.get((ds, m))]
            ours = [RC(ds, "SketchCD", k, s) for ds in ALLD if MAIN.get((ds, m))]
            vals.append(st.mean(per))
            wins.append(sum(round(o, 2) >= round(p, 2) for o, p in zip(ours, per)))
        out(f"{m:18s} " + " ".join(f"R@{k}={v:6.2f}" for k, v in zip((1, 10, 30, 50, 100), vals)) + "  SketchCD>=: " + "/".join(map(str, wins)))
out("-- per dataset, competitors above SketchCD at the dataset's budget:")
for k in (1, 10, 30, 50, 100):
    worse = [(ds, m, RC(ds, m, k) - RC(ds, "SketchCD", k)) for ds in ALLD for m in COMP if MAIN.get((ds, m)) and RC(ds, m, k) > RC(ds, "SketchCD", k) + 0.005]
    out(f"R@{k}: " + ("; ".join(f"{ds} {m} +{g:.2f}" for ds, m, g in worse) if worse else "none"))

# ---------------------------------------------------------------- the two extra baselines, paired with SketchCD
out("\n== EXTRA BASELINES, paired with SketchCD on the datasets where they ran (mean of seeds)")
for m in NEWB:
    dsl = [ds for ds in ALLD if MAIN.get((ds, m))]
    out(f"-- {m}: {len(dsl)} datasets; missing: {', '.join(ds for ds in ALLD if ds not in dsl) or 'none'}")
    for ds in dsl:
        rs, ro = MAIN[(ds, m)], MAIN[(ds, "SketchCD")]
        out(f"{ds:12s} n={len(rs)} time {T(ds, m):8.2f}s ({T(ds, m) / T(ds, 'SketchCD'):5.1f}x) | R@10 {RC(ds, m):6.2f} (sd {100 * st.pstdev([R(r) for r in rs]):.2f}) "
            f"vs {RC(ds, 'SketchCD'):6.2f} (sd {100 * st.pstdev([R(r) for r in ro]):.2f}): {RC(ds, m) - RC(ds, 'SketchCD'):+.2f} | excess {EX(ds, m):5.2f} vs "
            f"{EX(ds, 'SketchCD'):5.2f} | cv {cell(ds, m, lambda r: r['cv']):.3f} vs {cell(ds, 'SketchCD', lambda r: r['cv']):.3f} | tiny % "
            f"{100 * cell(ds, m, lambda r: r['tiny']):5.2f} vs {100 * cell(ds, 'SketchCD', lambda r: r['tiny']):5.2f} | parts "
            + " ".join(f"{k}={st.mean(r['parts'][k] for r in rs):.2f}" for k in rs[0]["parts"] if k.endswith("_s")))
    sp = [T(ds, m) / T(ds, "SketchCD") for ds in dsl]
    out(f"   slower than SketchCD: geo-mean {gmean(sp):.2f}x ({min(sp):.1f}-{max(sp):.1f}); R@10 lower than SketchCD on "
        f"{sum(RC(ds, m) < RC(ds, 'SketchCD') for ds in dsl)}/{len(dsl)}; WCSS lower than SketchCD on: "
        + (", ".join(ds for ds in dsl if EX(ds, m) < EX(ds, "SketchCD")) or "none")
        + "; of these with lower R@10: " + (", ".join(ds for ds in dsl if EX(ds, m) < EX(ds, "SketchCD") and RC(ds, m) < RC(ds, "SketchCD")) or "none"))
    out("   excess %.2f-%.2f (SketchCD %.2f-%.2f on the same datasets); excess above SketchCD by %.1f-%.1f points" % (
        min(EX(ds, m) for ds in dsl), max(EX(ds, m) for ds in dsl), min(EX(ds, "SketchCD") for ds in dsl), max(EX(ds, "SketchCD") for ds in dsl),
        min(EX(ds, m) - EX(ds, "SketchCD") for ds in dsl), max(EX(ds, m) - EX(ds, "SketchCD") for ds in dsl)))
    out("   cv %.2f-%.2f (SketchCD %.2f-%.2f); max tiny %.1f%%" % (
        min(cell(ds, m, lambda r: r["cv"]) for ds in dsl), max(cell(ds, m, lambda r: r["cv"]) for ds in dsl),
        min(cell(ds, "SketchCD", lambda r: r["cv"]) for ds in dsl), max(cell(ds, "SketchCD", lambda r: r["cv"]) for ds in dsl),
        max(100 * cell(ds, m, lambda r: r["tiny"]) for ds in dsl)))
    for k in (1, 10, 30, 50, 100):
        a, b = st.mean(RC(ds, m, k) for ds in dsl), st.mean(RC(ds, "SketchCD", k) for ds in dsl)
        above = [f"{ds} {RC(ds, m, k) - RC(ds, 'SketchCD', k):+.2f}" for ds in dsl if RC(ds, m, k) > RC(ds, "SketchCD", k) + 0.005]
        out(f"   R@{k}: {a:.2f} vs SketchCD on the same datasets {b:.2f} (SketchCD {b - a:+.2f}); {m} above SketchCD on: {', '.join(above) or 'none'}")
    if m == "PCA-FaissKMeans":
        out("   sketch-normalised (the NumPy sketch of the baseline replaced by the sketching time of SketchCD): own sketch s | "
            "SketchCD sketch s | rest (cluster + assign + means) s | normalised total s (x SketchCD) | SketchCD after its sketch s")
        nx = []
        for ds in dsl:
            rs, ro = MAIN[(ds, m)], MAIN[(ds, "SketchCD")]
            own = st.mean(r["parts"]["pca_s"] + r["parts"]["proj_s"] for r in rs)
            rest = st.mean(r["parts"]["train_s"] + r["parts"]["assign_s"] + r["parts"]["means_s"] for r in rs)
            sk = st.mean((r["parts"]["basis_ms"] + r["parts"]["proj_ms"] + r["parts"]["ext_ms"]) / 1e3 for r in ro)
            nx.append((sk + rest) / T(ds, "SketchCD"))
            out(f"   {ds:12s} {own:6.2f} | {sk:6.2f} | {rest:6.2f} | {sk + rest:6.2f} ({nx[-1]:5.2f}x) | {T(ds, 'SketchCD') - sk:6.2f} "
                f"(rest of the baseline / rest of SketchCD: {rest / (T(ds, 'SketchCD') - sk):.2f}x)")
        out(f"   normalised: {min(nx):.2f}-{max(nx):.2f}x slower than SketchCD, geo-mean {gmean(nx):.2f}x; own sketch "
            f"{min(st.mean(r['parts']['pca_s'] + r['parts']['proj_s'] for r in MAIN[(ds, m)]) for ds in dsl):.2f}-"
            f"{max(st.mean(r['parts']['pca_s'] + r['parts']['proj_s'] for r in MAIN[(ds, m)]) for ds in dsl):.2f}s, "
            f"{min(100 * st.mean((r['parts']['pca_s'] + r['parts']['proj_s']) / r['build_s'] for r in MAIN[(ds, m)]) for ds in dsl):.0f}-"
            f"{max(100 * st.mean((r['parts']['pca_s'] + r['parts']['proj_s']) / r['build_s'] for r in MAIN[(ds, m)]) for ds in dsl):.0f}% of its build")
    for s_ in ("0.005", "0.01", "0.02"):  # fixed budgets, as in the Recall@k experiment
        d_ = [st.mean(RC(ds, "SketchCD", k, s_) for ds in dsl) - st.mean(RC(ds, m, k, s_) for ds in dsl) for k in (1, 10, 30, 50, 100)]
        out(f"   fixed {pct(s_)}% scanned, SketchCD minus {m} on the same datasets, R@1..R@100: " + " ".join(f"{x:+.2f}" for x in d_))
out("-- Fig. 1(a) points: geo-mean build time relative to SketchCD and mean Recall@10 over each method's datasets; paired = "
    "Recall@10 of SketchCD over the ten datasets minus its mean gain on the method's datasets")
_o10 = st.mean(RC(ds, "SketchCD") for ds in ALLD)
for m in COMP:
    dsl = [ds for ds in ALLD if MAIN.get((ds, m))]
    gain = st.mean(RC(ds, "SketchCD") - RC(ds, m) for ds in dsl)
    out(f"   {m:18s} {gmean([T(ds, m) / T(ds, 'SketchCD') for ds in dsl]):6.2f}x  R@10 {st.mean(RC(ds, m) for ds in dsl):.2f} ({len(dsl)} datasets)  "
        f"paired {_o10 - gain:.2f} (SketchCD {_o10:.2f}, gain {gain:+.2f})")

# ================================================================ Figure: query throughput (QPS vs recall)
QPS = defaultdict(dict)
for r in load("exp04_qps.jsonl"):
    QPS[r["dataset"]][r["method"]] = r["curve"]
QDS = [ds for ds in ALLD if QPS.get(ds)]
QM = ["FaissKMeans", "EMeans-5", "FullSuperKMeans", "SampleSuperKMeans", "HierSuperKMeans", "QuantSuperKMeans", "SketchCD"]
QXLIM = {"cifar10": (0.6, 0.95), "cifar100": (0.6, 0.95), "tinyimagenet": (0.55, 0.93), "gist": (0.75, 0.97),
         "fiqa": (0.85, 0.99), "dbp3s": (0.85, 0.99), "treccovid": (0.88, 0.99), "svhn": (0.88, 0.99),
         "bigann10m": (0.88, 0.995), "deep10m": (0.88, 0.995)}


def qps_at(c, k, tgt):
    xs = [p[f"R{k}"] for p in c]
    ys = [p["qps"] for p in c]
    for a_, b_ in zip(range(len(xs) - 1), range(1, len(xs))):
        if xs[a_] < tgt <= xs[b_]:
            w_ = (tgt - xs[a_]) / (xs[b_] - xs[a_])
            return math.exp(math.log(ys[a_]) + w_ * (math.log(ys[b_]) - math.log(ys[a_])))
    return None


if QDS:
    fig, axes, NC_ = grid_fig(len(QDS), "Recall@10 (%)", "Queries per second")
    for j, ds in enumerate(QDS):
        ax = axes[j // NC_, j % NC_]
        lo, hi = QXLIM.get(ds, (0.88, 0.99))
        ymin, ymax = float("inf"), 0
        for m in QM:
            c = QPS[ds].get(m)
            if not c:
                continue
            xs = [p["R10"] for p in c]
            ys = [p["qps"] for p in c]
            keep = [t for t, x in enumerate(xs) if lo - 0.03 <= x <= hi + 0.006]
            plot_method(ax, m, [100 * xs[t] for t in keep], [ys[t] for t in keep])
            inr = [ys[t] for t, x in enumerate(xs) if lo <= x <= hi]
            if inr:
                ymin, ymax = min(ymin, min(inr)), max(ymax, max(inr))
        ax.set_yscale("log")
        if ymax > 0:
            ax.set_ylim(ymin / 1.15, ymax * 1.15)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(100 * lo, 100 * hi)
        ax.xaxis.set_major_locator(MultipleLocator(10 if hi - lo > 0.25 else 5))
        ax.xaxis.set_major_formatter(G)
        subcap_grid(ax, j, ds)
    grid_legend(fig, [handle(m) for m in LEGEND_ORDER if m in QM])
    save(fig, "qps")
    out("\n== QPS at target recall (interpolated on log QPS), single thread; [ratio SketchCD / method]")
    RAT = defaultdict(list)
    TOP = defaultdict(list)
    for ds in QDS:
        for k, tgt in ((10, 0.90), (10, 0.95), (100, 0.90)):
            row, vals = [], {}
            ours = qps_at(QPS[ds]["SketchCD"], k, tgt) if QPS[ds].get("SketchCD") else None
            for m in ORDER:
                c = QPS[ds].get(m)
                if not c:
                    continue
                v = qps_at(c, k, tgt)
                if v:
                    vals[m] = v
                if v and ours and m != "SketchCD":
                    RAT[(k, tgt, m)].append(ours / v)
                row.append(f"{m}={v:.0f}" + (f"[{ours / v:.2f}]" if v and ours else "") if v else f"{m}=n/a")
            if vals:
                TOP[(k, tgt)].append((ds, max(vals, key=vals.get)))
            out(f"{ds} R@{k}={tgt}: " + " ".join(row))
    out("-- geo-mean QPS ratio SketchCD/method over the datasets")
    for key in sorted(RAT):
        out(f"R@{key[0]}={key[1]} {key[2]:18s} {gmean(RAT[key]):.3f} (min {min(RAT[key]):.2f}, max {max(RAT[key]):.2f}, n={len(RAT[key])})")
    for key, v in sorted(TOP.items()):
        out(f"-- highest QPS at R@{key[0]}={key[1]}: " + ", ".join(f"{ds}:{m}" for ds, m in v))

# ================================================================ Figure: starvation (tiny lists, CV)
UNW = defaultdict(list)
for r in load("exp09_nc_rule.jsonl"):
    if r["dataset"] in ALLD:
        UNW[r["dataset"]].append(dict(tiny=r["tiny"], cv=r["cv"], build_s=r["build_s"], R10=R(r)))
for r in load("exp11_ablation.jsonl"):  # GIST: the nearest-centroid variant of the ablation
    if r["method"] == "unweighted" and r["dataset"] in ALLD and not UNW.get(r["dataset"]):
        UNW[r["dataset"]].append(dict(tiny=r["tiny_lab"], cv=r["cv_lab"], build_s=r["build_s"], R10=r[f"R10@{BUD[r['dataset']]}_lab"]))
CS = [ds for ds in ALLD if UNW.get(ds)]
series = [  # (label, records, colour) for the numbers below
    ("SketchCD-NC (nearest-centroid rule)", lambda ds: UNW[ds], NCC),
    ("SketchCD (size-aware rule)", lambda ds: MAIN[(ds, "SketchCD")], ACC),
    ("SampleSuperKMeans", lambda ds: MAIN[(ds, "SampleSuperKMeans")], SKM),
    ("FullSuperKMeans", lambda ds: MAIN[(ds, "FullSuperKMeans")], SKM),
]
# (a) list-size distributions (exp09_list_sizes.jsonl: list size / mean size, seed 42)
# (b) controlled experiment: share of tiny lists as the mean list size n/K shrinks (exp09_list_size_sweep.jsonl: same data, K = sqrt(n)..64 sqrt(n))
LS = {(r["dataset"], r["method"]): r for r in load("exp09_list_sizes.jsonl")}
NK = load("exp09_list_size_sweep.jsonl")
BOXM = [("SketchCD", "SketchCD (size-aware rule)", ACC), ("SketchCD@weighted=0", "SketchCD-NC (nearest-centroid rule)", NCC),
        ("FullSuperKMeans", "FullSuperKMeans", SKM), ("EMeans-5", "EMeans-5", EMC)]
if CS and all((ds, m) in LS for ds in ALLD for m, _, _ in BOXM):
    # Table (Exp-9) at the default K: share of tiny lists and CV of the list sizes under the size-aware rule (SketchCD),
    # the nearest-centroid rule (SketchCD-NC), and FullSuperKMeans; recall change of SketchCD-NC. Mean of the seeds.
    def _f1(v):
        return f"{v:.1f}"

    srows = ["\\filltable{8}{%", "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{Tiny lists (\\%)} & \\multicolumn{3}{c}{CV of list sizes} & $\\Delta$R@10\\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-8}",
             "Dataset & \\sys & NC & Full & \\sys & NC & Full & NC\\\\", "\\midrule"]
    for ds in ALLD:
        sk_t, nc_t, fu_t = (mean(rs, lambda r: 100 * r["tiny"]) for rs in (MAIN[(ds, "SketchCD")], UNW[ds], MAIN[(ds, "FullSuperKMeans")]))
        sk_c, nc_c, fu_c = (mean(rs, lambda r: r["cv"]) for rs in (MAIN[(ds, "SketchCD")], UNW[ds], MAIN[(ds, "FullSuperKMeans")]))
        sk42 = [r for r in MAIN[(ds, "SketchCD")] if r["seed"] == 42]  # SketchCD-NC ran with seed 42 only
        dr = 100 * (mean(UNW[ds], lambda r: r["R10"]) - mean(sk42, R))
        out(f"STARVE-TABLE {ds}: dR10(NC - SketchCD seed 42) = {dr:+.2f}")
        if ds == TEN[0]:
            srows.append("\\midrule")
        srows.append(f"{SHORT[ds]} & \\ours {_f1(sk_t)} & {_f1(nc_t)} & {_f1(fu_t)} & \\ours {sk_c:.2f} & {nc_c:.2f} & {fu_c:.2f} & "
                     + f"{dr:+.2f}".replace("-", "$-$").replace("+", "$+$") + "\\\\")
    srows += ["\\bottomrule", "\\end{tabular}}"]
    wtab("starve_rows.tex", "\n".join(srows) + "\n")

    # Figure (Exp-9): the same data with more, shorter lists (single panel, single column)
    CS_H = 1.18                                             # inches; 0.37 for the axis labels, 0.32 for the legend
    fig, axb = plt.subplots(figsize=(3.33, CS_H))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.37 / CS_H, top=1 - 0.32 / CS_H)
    NKS = defaultdict(dict)
    for r in NK:
        NKS[(r["dataset"].split("-k")[0], r["method"])][r["N"] / r["K"]] = r
    NKD = [("gist", "o", "GIST", 250), ("deep10m", "s", "DEEP-10M", 790), ("bigann10m", "D", "BIGANN-10M", 791)]
    for ds, mk, lab, dflt in NKD:
        for m, col, lw in (("SketchCD@weighted=0", NCC, LW), ("SketchCD", ACC, LW_OURS)):
            d_ = NKS.get((ds, m))
            if not d_:
                continue
            xs = sorted(d_)
            ys = [100 * d_[x]["tiny"] for x in xs]
            axb.plot(xs, ys, color=col, marker=mk, linestyle="-", linewidth=lw, markersize=3.0,
                     markerfacecolor=col if m == "SketchCD" else "white", markeredgecolor="white" if m == "SketchCD" else col,
                     markeredgewidth=0.4 if m == "SketchCD" else MEW, zorder=6 if m == "SketchCD" else 5)
            if m == "SketchCD@weighted=0":
                xd = min(xs, key=lambda x: abs(x - dflt))
                axb.plot([xd], [100 * d_[xd]["tiny"]], zorder=7, **RING)
                axb.annotate(lab, (xs[0], ys[0]), xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                             fontsize=6.8, color=INK2)
    axb.text(8.5, 4.5, "size-aware rule (all three)", fontsize=6.8, color=ACC, va="bottom", ha="left")
    axb.tick_params(labelsize=7.0)
    axb.set_xscale("log")
    axb.xaxis.set_major_formatter(G)
    axb.xaxis.set_minor_formatter(NullFormatter())
    axb.set_xticks([30, 100, 300, 1000])
    axb.set_xlim(7, 1400)
    axb.set_ylim(-3, 64)
    axb.set_yticks([0, 20, 40, 60])
    axb.set_xlabel("Mean list size $n/K$ (vectors per list)", labelpad=1.5, fontsize=7.8)
    axb.set_ylabel("Tiny lists (%)", labelpad=1.5, fontsize=7.8)
    fig.legend(handles=[Line2D([0], [0], color=NCC, linewidth=LW, label="nearest-centroid rule (SketchCD-NC)"),
                        Line2D([0], [0], color=ACC, linewidth=LW_OURS, label="size-aware rule (SketchCD)"),
                        Line2D([0], [0], label=r"default $K=4\sqrt{n}$", **RING)],
               loc="lower center", bbox_to_anchor=(0.56, 1 - 0.32 / CS_H), ncol=2, fontsize=7.0, handletextpad=0.35,
               columnspacing=1.0, labelspacing=0.25, borderaxespad=0)
    save(fig, "coldstart")
    out("\n== LIST SIZES (relative to the mean, seed 42): q05 / q25 / median / q75 / q95 | tiny % | cv")
    for ds in ALLD:
        for m, lab, _ in BOXM:
            r = LS[(ds, m)]
            q = dict(zip(r["q"], r["rel_quantiles"]))
            out(f"{ds:12s} {m:20s} {q[0.05]:.2f} / {q[0.25]:.2f} / {q[0.5]:.2f} / {q[0.75]:.2f} / {q[0.95]:.2f} | "
                f"{100 * r['tiny']:5.2f} | {r['cv']:.3f} | empty {r['empty']}")
    out("\n== TINY LISTS vs MEAN LIST SIZE (same data, K varied; seed 42): n/K: tiny % / cv / R@10 % at the dataset's budget")
    for ds, *_ in NKD:
        for m in ("SketchCD@weighted=0", "SketchCD"):
            d_ = NKS.get((ds, m), {})
            out(f"{ds:10s} {m:20s} " + "  ".join(f"{x:.0f}: {100 * d_[x]['tiny']:.1f}/{d_[x]['cv']:.2f}/{100 * R(d_[x]):.2f}"
                                                  for x in sorted(d_)))
elif CS:  # fallback while the list-size data are missing: tiny lists as grouped bars
    fig, ax = plt.subplots(figsize=(3.33, 1.5))
    w = 0.2
    for s_, (lab, getter, col) in enumerate(series):
        vals = [mean(getter(ds), lambda r: 100.0 * r["tiny"]) for ds in CS]
        ax.bar([i + (s_ - 1.5) * w for i in range(len(CS))], vals, width=w * 0.9, color=col)
    ax.set_xticks(range(len(CS)))
    ax.set_xticklabels([SHORT[d] for d in CS], fontsize=5.6, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Tiny lists (%)")
    save(fig, "coldstart")
if CS:
    out("\n== STARVATION: tiny % / cv per dataset; SketchCD-NC time and R@10 vs SketchCD")
    for ds in CS:
        out(ds, " | ".join(f"{lab.split(' ')[0]}: {mean(g(ds), lambda r: 100 * r['tiny']):.2f} / {mean(g(ds), lambda r: r['cv']):.3f}"
                           for lab, g, _ in series),
            f"| NC {mean(UNW[ds], lambda r: r['build_s']):.2f}s R10 {100 * mean(UNW[ds], lambda r: r['R10']):.2f} "
            f"(dR {100 * mean(UNW[ds], lambda r: r['R10']) - RC(ds, 'SketchCD'):+.2f})")
    for lab, g, _ in series:
        tv = [mean(g(ds), lambda r: 100 * r["tiny"]) for ds in CS]
        cv = [mean(g(ds), lambda r: r["cv"]) for ds in CS]
        out(f"{lab}: tiny {min(tv):.2f}-{max(tv):.2f} %, cv {min(cv):.2f}-{max(cv):.2f}")
    for m in COMP:
        tv = [(cell(ds, m, lambda r: 100 * r["tiny"]), ds) for ds in ALLD if MAIN.get((ds, m))]
        out(f"{m}: max tiny {max(tv)[0]:.1f}% ({max(tv)[1]})")
    out("NC recall drop: " + ", ".join(f"{ds} {RC(ds, 'SketchCD') - 100 * mean(UNW[ds], lambda r: r['R10']):.2f}" for ds in CS))

# ================================================================ stage breakdown (a table), machine throughput, assignment pass
stages = [("sketch", ("basis_ms", "proj_ms", "ext_ms")), ("first pass", ("pass_ms",)), ("sketch sweeps", ("sweep_ms",)),
          ("verified", ("cent_ms", "xsweep_ms", "fnn_ms"))]
out("\n== STAGES (share %): sketch / first pass / sketch sweeps / verified+relabel | l ext pilot")
for ds in ALLD:
    rs = MAIN.get((ds, "SketchCD"))
    if rs:
        v = [mean(rs, lambda r, ks=ks: sum(r["parts"][k] for k in ks) / 1e3) for _, ks in stages]
        out(f"{ds:12s} " + " ".join(f"{100 * x / sum(v):5.1f}" for x in v) + f" | total {sum(v):.2f}s | l={rs[0]['parts']['l']} "
            f"ext={rs[0]['parts']['ext']} pilot={st.mean(r['parts']['pilot'] for r in rs):.1f}")
# Table: where the build time of SketchCD goes. One row per dataset: the build time (as in the main table) and the share
# of each of the four stages in the summed stage times (mean of the seeds). A dash stands for the sketch of the datasets
# that use none (d <= 256).
BRK = [(ds, [mean(MAIN[(ds, "SketchCD")], lambda r, ks=ks: sum(r["parts"][k] for k in ks) / 1e3) for _, ks in stages]) for ds in ALLD]
brows = [r"\filltable{6}{%", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", r"Dataset & Time (s) & Sketch & First pass & Sketch sweeps & Verified\\",
         r"\midrule"]
for ds, v in BRK:
    if ds == TEN[0]:
        brows.append(r"\midrule")
    brows.append(f"{SHORT[ds]} & {fmt_t(T(ds, 'SketchCD'))} & " + " & ".join("--" if x == 0 else padnum(f"{100 * x / sum(v):.1f}", 4) for x in v) + r"\\")
brows += [r"\bottomrule", r"\end{tabular}}"]
wtab("breakdown_rows.tex", "\n".join(brows) + "\n")
out("   stage shares over the eight high-dimensional datasets: sketch %.0f-%.0f, first pass + sketch sweeps %.0f-%.0f, verified %.0f-%.0f" % tuple(
    f(100 * g(v) / sum(v) for ds, v in BRK if ds in HIGH) for g in (lambda v: v[0], lambda v: v[1] + v[2], lambda v: v[3]) for f in (min, max)))

# Throughput of the machine (experiments/bench/throughput.cpp, records in exp00_throughput.jsonl): single-threaded BLAS
# calls of 20 workers on 256-row tiles, full-dimensional distances, and the read bandwidth. GEMM_MADD (1100 GFLOPS) prices
# the arithmetic of one exhaustive assignment pass (n K d multiply-adds); it lies near the top of what the assignment shapes reach.
THRU = load("exp00_throughput.jsonl")
if THRU:
    out("\n== THROUGHPUT of one socket (GFLOPS, 2 flops per multiply-add; best of the runs): dataset: projection / cells / sweep / assignment | distances")
    for ds in dict.fromkeys(r["dataset"] for r in THRU):
        g = {r["what"]: r["gflops"] for r in THRU if r["dataset"] == ds and "gflops" in r}
        out(f"{ds:12s} " + " / ".join(f"{g[w]:6.0f}" if w in g else "     -" for w in ("projection", "cells", "sweep", "assignment"))
            + f" | {g['distance']:6.0f}")

    def _rng(v):
        return f"{min(v):.0f}-{max(v):.0f}"

    _own = [r["gflops"] for r in THRU if r["what"] in ("projection", "cells", "sweep")]
    _asg = [r["gflops"] for r in THRU if r["what"] == "assignment"]
    _dlo = [r["gflops"] for r in THRU if r["what"] == "distance" and r["l"] < r["d"] <= 1536]   # sketched datasets only
    _dhi = [r["gflops"] for r in THRU if r["what"] == "distance" and r["l"] < r["d"] and r["d"] == 3072]
    _g = GEMM_MADD * 2 / 1e9
    out(f"   matrix multiplications of SketchCD (projection, cells, sweeps): {_rng(_own)} GFLOPS; of an exhaustive assignment pass: "
        f"{_rng(_asg)} GFLOPS (priced at {_g:.0f})")
    out(f"   full-dimensional distances: {_rng(_dlo)} GFLOPS for d <= 1536 (rho = {_g / max(_dlo):.1f}-{_g / min(_dlo):.1f} at {_g:.0f} GFLOPS), "
        f"{_rng(_dhi)} GFLOPS for d = 3072 (rho = {_g / max(_dhi):.1f}-{_g / min(_dhi):.1f})")
    # read bandwidth B of one socket: every pass over the vectors takes at least 4 n d / B seconds (Section 5.3)
    _rd = {r["dataset"]: r["gbps"] for r in THRU if r["what"] == "read"}
    if _rd:
        out(f"   read bandwidth: {min(_rd.values()):.1f}-{max(_rd.values()):.1f} GB/s; five passes over the vectors at 38 GB/s: "
            + "; ".join(f"{ds} {5 * 4 * rs[0]['N'] * rs[0]['d'] / 38e9:.2f}s of the {T(ds, 'SketchCD'):.2f}s build "
                        f"({100 * 5 * 4 * rs[0]['N'] * rs[0]['d'] / 38e9 / T(ds, 'SketchCD'):.0f}%)"
                        for ds in ("gist", "svhn") for rs in [MAIN[(ds, "SketchCD")]]))
out(f"\n== ONE EXHAUSTIVE ASSIGNMENT PASS at {GEMM_MADD:.2g} madd/s ({GEMM_MADD * 2 / 1e9:.0f} GFLOPS) vs SketchCD build time")
for ds in ALLD:
    rs = MAIN.get((ds, "SketchCD"))
    if rs:
        t = rs[0]["N"] * rs[0]["K"] * rs[0]["d"] / GEMM_MADD
        out(f"{ds:12s} pass {t:6.2f}s SketchCD {T(ds, 'SketchCD'):6.2f}s ratio {t / T(ds, 'SketchCD'):5.2f} | "
            f"input {rs[0]['N'] * rs[0]['d'] * 4 / 1e9:.2f} GB")

# ================================================================ trade-off curves
base = load("exp02_tradeoff_baselines.jsonl")  # a later run of a configuration replaces an earlier one
OURS_TAG = {"SketchCD@sweeps=0,xsweeps=0": "s0x0", "SketchCD@sweeps=2,xsweeps=0": "s2x0", "SketchCD@sweeps=4,xsweeps=1": "s4x1",
            "SketchCD": "s6x2", "SketchCD@sweeps=8,xsweeps=3": "s8x3", "SketchCD@sweeps=12,xsweeps=4": "s12x4",
            "SketchCD@sweeps=16,xsweeps=6": "s16x6"}
BY3 = {}  # (dataset, configuration) -> [point]; seed 42; a later run of the same configuration replaces an earlier one
NAN = float("nan")
for r in base:
    if r["seed"] == 42 and r["dataset"] in ALLD:
        J = r["wcss_lab"] if r["which"] == "lab" else r["wcss_nc"]
        Jr = r.get("Jref") or JREF.get(r["dataset"])
        BY3[(r["dataset"], r["method"])] = [dict(build_s=r["build_s"], R10=r.get(f"R10@{BUD[r['dataset']]}_{r['which']}", NAN),
                                                 ex=100.0 * (J - Jr) / Jr if Jr else NAN)]
for r in load("exp02_tradeoff_sketchcd.jsonl"):
    if r["method"] in OURS_TAG and r["seed"] == 42 and r["dataset"] in ALLD:
        Jr = r.get("Jref") or JREF.get(r["dataset"])
        BY3[(r["dataset"], "ours-" + OURS_TAG[r["method"]])] = [dict(build_s=r["build_s"], R10=R(r),
                                                                     ex=100.0 * (r["wcss"] - Jr) / Jr if Jr else NAN)]


def best(ds, m):
    """A SuperKMeans configuration with its faster assignment path (exact assign or the library's fast path)."""
    a, b = BY3.get((ds, m)), BY3.get((ds, m + "+fa"))
    if a and b:
        return a if a[0]["build_s"] <= b[0]["build_s"] else b
    return a or b


def pt(rs):
    return mean(rs, lambda r: r["build_s"]), 100 * mean(rs, lambda r: r["R10"]), mean(rs, lambda r: r["ex"])


OURS_CURVE = ["s0x0", "s2x0", "s4x1", "s6x2", "s8x3", "s12x4", "s16x6"]
FAMILIES = [
    ("SketchCD", "SketchCD", lambda ds: [BY3.get((ds, "ours-" + c)) for c in OURS_CURVE]),
    ("FullSuperKMeans", "FullSuperKMeans", lambda ds: [best(ds, f"skm-full-it{i}") for i in (1, 2, 3, 5, 10, 25)]),
    ("HierSuperKMeans", "HierSuperKMeans", lambda ds: [best(ds, m) for m in ("skm-hier", "skm-hier-r1", "skm-hier-r2")]),
    ("QuantSuperKMeans", "QuantSuperKMeans", lambda ds: [best(ds, "skm-rabitq-hier"), best(ds, "skm-rabitq-full")]),
    ("e-means", "EMeans-5", lambda ds: [BY3.get((ds, f"emeans-{e}ep")) for e in (1, 2, 3, 5)]),
    ("FaissKMeans", "FaissKMeans", lambda ds: [BY3.get((ds, f"faiss-it{i}")) for i in (2, 5, 10, 20)]),
]


def fam_points(ds, getter):
    return sorted(pt(rs) for rs in getter(ds) if rs)


def draw_families(ax, ds, ycol, ymin=None):
    for lab, m, getter in FAMILIES:
        pts = [p for p in fam_points(ds, getter) if p[ycol] == p[ycol]]
        if ymin is not None:
            pts = [p for p in pts if p[ycol] >= ymin]
        if pts:
            plot_method(ax, m, [p[0] for p in pts], [p[ycol] for p in pts])


def fam_handles():
    return [handle(m, lab) for lab, m, _ in FAMILIES]


def wcss_axis(ax):
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.45)
    ax.set_ylim(-0.12, 15)
    ax.set_yticks([0, 0.1, 1, 10])
    ax.yaxis.set_major_formatter(G)
    ax.axhline(0, color=AXIS, linewidth=0.7, zorder=1)


TD = [ds for ds in ALLD if BY3.get((ds, "ours-s6x2")) and BY3.get((ds, "faiss-it10"))]
out("\n== TRADE-OFF CURVES available for: " + ", ".join(TD))
TDR = [ds for ds in TD if all(p[1] == p[1] for _, _, g in FAMILIES for p in fam_points(ds, g))]
out("== RECALL TRADE-OFF CURVES complete (recall at the dataset's budget) for: " + ", ".join(TDR))
# One figure for both criteria (Exp-2): per dataset, a Recall@10 panel above a WCSS panel on the same time axis.
if TD:
    g, nc = GRID_IN, 5
    nr = (len(TD) + nc - 1) // nc
    PH, PGAP = 0.40, 0.035                                    # inches: one panel, the gap inside a pair
    pair = 2 * PH + PGAP
    step = pair + g["ticks"] + 0.03 + g["title"]              # from the top of one row of pairs to the top of the next
    H = g["legend"] + g["title"] + nr * pair + (nr - 1) * (step - pair) + g["ticks"] + g["xlabel"]
    fig = plt.figure(figsize=(7.1, H))
    aw = (GRID_R - GRID_L) / (nc + (nc - 1) * 0.22)           # width of a panel as a fraction of the figure
    for j, ds in enumerate(TD):
        r, c = divmod(j, nc)
        x0, ytop = GRID_L + c * aw * 1.22, H - g["legend"] - g["title"] - r * step
        axw = fig.add_axes([x0, (ytop - pair) / H, aw, PH / H])
        axr = fig.add_axes([x0, (ytop - PH) / H, aw, PH / H], sharex=axw)
        draw_families(axw, ds, 2)
        wcss_axis(axw)
        lowest = min(p[2] for _, _, g_ in FAMILIES for p in fam_points(ds, g_) if p[2] == p[2])
        if lowest < -0.08:
            axw.set_ylim(bottom=1.6 * lowest)
            axw.set_yticks([-0.1, 0, 0.1, 1, 10])
        if ds in TDR:
            lo = max(min(RC(ds, m) for m in SEVEN if MAIN.get((ds, m))) - 1.0, RC(ds, "SketchCD") - 4.0)  # omit points far below
            draw_families(axr, ds, 1, lo)
            axr.set_ylim(bottom=lo)
        log_time_axis(axw)
        subcap_grid(axr, j, ds)
        axr.tick_params(labelbottom=False)
        axw.tick_params(labelsize=6.8)
        if c == 0:
            axr.set_ylabel("R@10 (%)", fontsize=6.8, labelpad=2)
            axw.set_ylabel(r"$\Delta J$ (%)", fontsize=6.8, labelpad=2)
    fig.align_ylabels()
    fig.supxlabel("Build time (s)", x=(GRID_L + GRID_R) / 2, y=0.0, va="bottom", fontsize=7.8)
    grid_legend(fig, fam_handles())
    save(fig, "tradeoff")
if TDR:
    out("\n== TIME TO EQUAL RECALL (Recall@10 of default SketchCD at the dataset's budget, seed-42 curves)")
    for ds in TDR:
        t0, r0, _ = pt(BY3[(ds, "ours-s6x2")])
        line = [f"{ds}: SketchCD {t0:.2f}s at {r0:.2f}"]
        for lab, m, getter in FAMILIES[1:]:
            pts = fam_points(ds, getter)
            if not pts:
                continue
            t = None
            for (ta, ya, _), (tb, yb, _) in zip(pts, pts[1:]):
                if ya < r0 <= yb:
                    t = math.exp(math.log(ta) + (r0 - ya) / (yb - ya) * (math.log(tb) - math.log(ta)))
                    break
            if t is None and pts[0][1] >= r0:
                t = pts[0][0]
            line.append(f"{m}: " + (f"{t:.1f}s ({t / t0:.1f}x)" if t else f"never (best {max(p[1] for p in pts):.2f} at {pts[-1][0]:.0f}s)"))
        out(" | ".join(line))
if TD:

    def reach(pts, target, col, up):
        """Time at which a curve (sorted by time) first reaches target in column col (log-time interpolation)."""
        for (ta, *a), (tb, *b) in zip(pts, pts[1:]):
            ya, yb = a[col - 1], b[col - 1]
            if (ya < target <= yb) if up else (ya > target >= yb):
                w_ = (target - ya) / (yb - ya)
                return math.exp(math.log(ta) + w_ * (math.log(tb) - math.log(ta)))
        if pts and ((pts[0][col] >= target) if up else (pts[0][col] <= target)):
            return pts[0][0]
        return None

    for col, up, name in ((2, False, "WCSS"),):
        out(f"\n== TIME TO EQUAL QUALITY ({name} of default SketchCD, seed-42 curves)")
        for ds in TD:
            t0, r0, e0 = pt(BY3[(ds, "ours-s6x2")])
            if col == 2 and not math.isfinite(e0):
                continue
            line = [f"{ds}: SketchCD {t0:.2f}s at {(r0 if col == 1 else e0):.2f}"]
            for lab, m, getter in FAMILIES[1:]:
                pts = fam_points(ds, getter)
                if not pts or (col == 2 and not all(math.isfinite(p[2]) for p in pts)):
                    continue
                t = reach(pts, r0 if col == 1 else e0, col, up)
                bestv = max(p[col] for p in pts) if up else min(p[col] for p in pts)
                line.append(f"{m}: " + (f"{t:.1f}s ({t / t0:.1f}x)" if t else f"never (best {bestv:.2f} at {pts[-1][0]:.0f}s)"))
            out(" | ".join(line))
    out("\n== CURVE POINTS (time s / R@10 / excess) of SketchCD, and of every method on SVHN")
    for ds in TD:
        out(ds, " ".join(f"{p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}" for p in fam_points(ds, FAMILIES[0][2])))
    for lab, m, getter in FAMILIES:
        out("svhn", m, " ".join(f"{p[0]:.1f}/{p[1]:.2f}/{p[2]:.2f}" for p in fam_points("svhn", getter)))

# ================================================================ parameters of the bias model
PAR = {r["dataset"]: r for r in load("exp09_model_params.jsonl")}
if PAR:
    out("\n== MODEL PARAMETERS on converged partitions: sigma^2/Delta^2 = b*")
    for ds in ALLD:
        if ds in PAR:
            out(f"{ds:12s} b*={PAR[ds]['bstar']:.2f} sigma2={PAR[ds]['sigma2']:.4g} delta_nn2={PAR[ds]['delta_nn2']:.4g}")
bs_ = [PAR[ds]["bstar"] for ds in ALLD if ds in PAR]
blo, bhi = (min(bs_), max(bs_)) if bs_ else (2.88, 3.61)
bstar = st.median(bs_) if bs_ else 3.2
out(f"b* over the ten datasets: median {bstar:.2f}, range {blo:.2f}-{bhi:.2f}")


# ================================================================ bench/run_sketchcd_configs.py records (analysis experiments on the trio)
def last_by(rs, key):
    d = {}
    for r in rs:
        d[key(r)] = r  # a later record of the same key replaces an earlier one
    return d


def ex_lab(r):
    return 100.0 * (r["wcss_lab"] - r["Jref"]) / r["Jref"]


# ---- sketch dimension
DIMR = defaultdict(list)
for r in last_by(load("exp07_sketch_dim.jsonl"), lambda r: (r["dataset"], r["method"])).values():
    DIMR[r["dataset"]].append(r)
if DIMR:
    fig, axes = plt.subplots(1, 3, figsize=(3.33, 1.14), gridspec_kw=dict(wspace=0.62))
    out("\n== SKETCH DIMENSION (no extension, seed 42): l: dists/pt/sweep, total s, sketch s, verify s, excess %, R@10 %")
    for ds in TRIO:
        rs = sorted(DIMR.get(ds, []), key=lambda r: r["run"]["l"])
        if not rs:
            continue
        ls_ = [r["run"]["l"] for r in rs]
        ys3 = ([r["run"]["xevals_per_point"] for r in rs], [r["build_s"] for r in rs], [ex_lab(r) for r in rs])
        dflt = MAIN[(ds, "SketchCD")][0]["parts"]["l"]  # the sketch dimension that SketchCD's rule chooses
        for ax, ys in zip(axes, ys3):
            ds_line(ax, ds, ls_, ys)
            if dflt in ls_:
                ax.plot([dflt], [ys[ls_.index(dflt)]], zorder=7, **RING)
        t_ = ys3[1]
        out(f"{ds}: default l={dflt}; fastest l={ls_[t_.index(min(t_))]} ({min(t_):.2f}s vs {t_[ls_.index(dflt)]:.2f}s, "
            f"{100 * (1 - min(t_) / t_[ls_.index(dflt)]):.0f}% faster); excess there {ys3[2][t_.index(min(t_))]:.2f} vs {ys3[2][ls_.index(dflt)]:.2f}")
        for r in rs:
            u = r["run"]
            out(f"{ds:10s} l={u['l']:3d}: {u['xevals_per_point']:6.1f} {r['build_s']:6.2f} {(u['basis_ms'] + u['proj_ms']) / 1e3:5.2f} "
                f"{(u['xsweep_ms'] + u['fnn_ms']) / 1e3:5.2f} {ex_lab(r):6.2f} {100 * r[f"R10@{BUD[ds]}_lab"]:6.2f}")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([32, 128, 384])
        ax.xaxis.set_major_formatter(G)
        ax.set_xlabel(f"({'abc'[list(axes).index(ax)]}) Sketch dim. " + "$\\ell$", labelpad=1)
        ax.tick_params(labelsize=5.8)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(G)
    axes[0].set_ylabel("Full-dim. distances\nper vector and sweep", fontsize=6.6)
    axes[1].set_ylabel("Build time (s)", fontsize=6.6)
    axes[1].set_ylim(bottom=0)
    axes[2].set_ylabel("WCSS excess (%)", fontsize=6.6)
    axes[2].set_ylim(bottom=0)
    fig.legend(handles=[ds_handle(d) for d in TRIO] + [Line2D([0], [0], label="chosen by SketchCD", **RING)],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), columnspacing=0.9, handlelength=1.4, fontsize=6.0)
    save(fig, "dims")

# ---- sensitivity
SEN = defaultdict(list)
for r in last_by(load("exp08_sensitivity_quality.jsonl"), lambda r: (r["dataset"], r["method"], r["seed"])).values():
    SEN[(r["dataset"], r["method"])].append(r)
SENQ = last_by(load("exp08_sensitivity_times.jsonl"), lambda r: (r["dataset"], r["method"]))  # times: seed 42 on an idle machine
if SEN:
    PARAMS = [("Candidate lists $|N(p)|$", "c", [32, 64, 96, 128, 192, 256], 128, False),
              ("Over-relaxation $\\omega$", "w", [1.0, 1.3, 1.6, 1.9, 2.2], 1.6, False),
              ("Sketch sample size", "s", [4096, 8192, 16384, 32768, 65536], 16384, True),
              ("Cells $M/\\sqrt{K}$", "m", [0.25, 0.5, 1, 2, 4], 1, True)]
    fig, axes = plt.subplots(3, 4, figsize=(7.1, 1.56), sharey="row", gridspec_kw=dict(wspace=0.08, hspace=0.16))
    fig.subplots_adjust(left=0.075, right=0.997)
    DEF = {"c": "c128", "w": "w1.6", "s": "s16384", "m": "m1"}
    SIG = {}
    for ds in TRIO:
        sds = [st.pstdev([100 * r[f"R10@{BUD[ds]}_lab"] for r in rs]) for (d_, m_), rs in SEN.items() if d_ == ds and len(rs) > 1]
        SIG[ds] = st.mean(sds)
    sig = max(SIG.values())
    out("sensitivity: seed-to-seed sd of Recall@10 (points), mean over settings: " + ", ".join(f"{d} {v:.2f}" for d, v in SIG.items()))
    out("\n== SENSITIVITY (3 seeds): param value: time s / excess % / R@10 %")
    for j, (lab, pre, vals, dflt, logx) in enumerate(PARAMS):
        for ds in TRIO:
            xs, t, e, rr = [], [], [], []
            for v in vals:
                tag = f"w{v:.1f}" if pre == "w" else f"{pre}{v:g}" if pre == "m" else f"{pre}{v}"
                rs = SEN.get((ds, tag))
                if not rs:
                    continue
                xs.append(v)
                t.append(SENQ[(ds, tag)]["build_s"] if (ds, tag) in SENQ else mean(rs, lambda r: r["build_s"]))
                e.append(mean(rs, ex_lab))
                rr.append(100 * mean(rs, lambda r: r[f"R10@{BUD[ds]}_lab"]))
                out(f"{ds:10s} {tag:7s}: {t[-1]:.2f} / {e[-1]:.2f} / {rr[-1]:.2f} [{len(rs)}]")
            dtag = DEF[pre]
            t0 = SENQ[(ds, dtag)]["build_s"] if (ds, dtag) in SENQ else mean(SEN[(ds, dtag)], lambda r: r["build_s"])
            r0 = 100 * mean(SEN[(ds, dtag)], lambda r: r[f"R10@{BUD[ds]}_lab"])
            rel = ([x / t0 for x in t], e, [x - r0 for x in rr])
            for i, ys in enumerate(rel):
                ds_line(axes[i, j], ds, xs, ys)
        for i in range(3):
            ax = axes[i, j]
            ax.axvline(dflt, color="#9a9a9a", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
            if i != 1:
                ax.axhline(1.0 if i == 0 else 0.0, color="#9a9a9a", linewidth=0.5, zorder=1)
            if logx:
                ax.set_xscale("log", base=2)
            ax.set_xticks(vals)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$2^{{{round(math.log2(v))}}}$" if v >= 1024 else f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", labelsize=5.9)
            ax.tick_params(axis="y", labelsize=5.9)
            if i < 2:
                ax.tick_params(labelbottom=False)
        axes[2, j].set_xlabel(f"({'abcd'[j]}) " + lab, labelpad=1)
    axes[0, 0].set_ylim(0.7, 1.45)
    axes[0, 0].set_yticks([0.8, 1.0, 1.2, 1.4])
    axes[0, 0].yaxis.set_major_formatter(X_)
    axes[1, 0].set_ylim(0.3, 1.3)
    axes[1, 0].set_yticks([0.4, 0.8, 1.2])
    axes[2, 0].set_ylim(-2, 2)
    axes[2, 0].set_yticks([-2, -1, 0, 1, 2])
    for j in range(4):
        axes[2, j].axhspan(-2 * sig, 2 * sig, color="#ececec", zorder=0, linewidth=0)
    axes[0, 0].set_ylabel("Time\n(relative)", fontsize=6.4, labelpad=2)
    axes[1, 0].set_ylabel("$\\Delta J$\n(%)", fontsize=6.4, labelpad=2)
    axes[2, 0].set_ylabel("$\\Delta$R@10\n(points)", fontsize=6.4, labelpad=2)
    fig.align_ylabels(axes[:, 0])
    fig.legend(handles=[ds_handle(d) for d in TRIO] +
               [Line2D([0], [0], color="#9a9a9a", linewidth=0.6, linestyle=(0, (3, 2)), label="default setting"),
                Patch(facecolor="#ececec", edgecolor="none", label=r"seed noise of recall ($\pm2$ s.d.)")],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.915), columnspacing=1.6, fontsize=6.6)
    save(fig, "sensitivity")
    out("sensitivity times from " + ("the idle-machine rerun (seed 42)" if SENQ else "the three-seed runs"))
    for ds in TRIO:
        allr = [100 * mean(rs, lambda r: r[f"R10@{BUD[ds]}_lab"]) for (d_, m_), rs in SEN.items() if d_ == ds]
        if allr:
            out(f"{ds}: R@10 range over all settings {min(allr):.2f}-{max(allr):.2f} (spread {max(allr) - min(allr):.2f})")

# ---- exactness
EXA = last_by(load("exp10_exactness.jsonl"), lambda r: r["dataset"])
if EXA:
    out("\n== EXACTNESS (pruned vs all candidates, same sketch): labels differ / max centroid diff / l / ext / pilot / dists / stage s / speedup")
    lines = []
    for ds in HIGH:
        r = EXA.get(ds)
        if not r:
            continue
        sp = r["stage_all_s"] / r["stage_pruned_s"]
        out(f"{ds:12s} {r['labels_differ']} {r['max_cdiff']:.1e} l={r['l']} ext={r['ext']} pilot={r['pilot']:.1f} "
            f"{r['xev_pruned']:.1f}/{r['fev_pruned']:.1f} vs {r['xev_all']:.0f}/{r['fev_all']:.0f} | {r['stage_pruned_s']:.2f} vs "
            f"{r['stage_all_s']:.2f} ({sp:.1f}x) | total {r['total_pruned_s']:.2f} vs {r['total_all_s']:.2f} ({r['total_all_s'] / r['total_pruned_s']:.1f}x)")
        dist = padnum(f"{r['xev_pruned']:.1f}", 4) + " / " + padnum(f"{r['fev_pruned']:.1f}", 4)
        same = "yes" if r["labels_differ"] == 0 and r["max_cdiff"] == 0 else "NO"
        lines.append(f"{TITLE[ds]} & {r['l']} & {dist} & {r['stage_pruned_s']:.2f} & " + padnum(f"{r['stage_all_s']:.2f}", 5)
                     + " & " + padnum(f"{sp:.1f}", 4) + r"$\times$ & " + same + r"\\")
    wtab("exact_rows.tex",
        "\\filltable{7}{%\n\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "Dataset & $\\ell$ & Dist. & Pruned (s) & All (s) & Speedup & Identical\\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n")

# ---- ablation
ABL = last_by(load("exp11_ablation.jsonl"), lambda r: (r["dataset"], r["method"]))  # the run on an otherwise idle machine
if ABL:
    rows = [("\\sys{} (default)", "default"), ("\\sys-NC", "unweighted"), ("no cells (flat first pass)", "flat-first-pass"),
            ("no replays in cells", "no-replays"), ("no sketch sweeps", "no-sketch-sweeps"), ("no verified sweeps", "no-exact-sweeps"),
            ("no over-relaxation", "omega=1"), ("64 / 128 candidates", "cand64/128"), ("no final relabeling", "no-final-relabel"),
            ("no pruning", "no-pruning")]
    out("\n== ABLATION (seed 42): time s / excess % / R@10 % / cv")
    lines = []
    for lab, key in rows:
        row = [lab]
        for ds in TRIO:
            if key == "no-pruning":
                r0, e = ABL.get((ds, "default")), EXA.get(ds)
                if r0 and e:  # bit-identical output; time of the build that evaluates every candidate in full dimension
                    row.append(f"{e['total_all_s']:.2f} / {ex_lab(r0):.2f} / {100 * r0[f"R10@{BUD[ds]}_lab"]:.2f}")
                    out(f"{ds:10s} {key:18s} {e['total_all_s']:.2f} (pruned {e['total_pruned_s']:.2f})")
                else:
                    row.append("---")
                continue
            r = ABL.get((ds, key))
            if not r:
                row.append("---")
                continue
            row.append(f"{r['build_s']:.2f} / {ex_lab(r):.2f} / {100 * r[f"R10@{BUD[ds]}_lab"]:.2f}")
            out(f"{ds:10s} {key:18s} {r['build_s']:.2f} / {ex_lab(r):.2f} / {100 * r[f"R10@{BUD[ds]}_lab"]:.2f} / cv {r['cv_lab']:.3f} tiny {100 * r['tiny_lab']:.2f}")
        lines.append(ours_row(row) + "\n\\addlinespace[2pt]" if key == "default" else " & ".join(row) + "\\\\")
    wtab("ablation_rows.tex",
        "\\filltable{4}{%\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\nVariant & GIST & TREC-COVID & SVHN\\\\\n\\midrule\n"
        + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}}\n")

# ---- memory
MEM = load("exp06_memory.jsonl")
if MEM:
    out("\n== MEMORY: extra GB beyond the input vectors (peak RSS - n*d*4)")
    for r in MEM:
        out(f"{r['dataset']:10s} {r['method']:18s} data {r['data_gb']:.2f} GB extra {r['extra_gb']:.2f} GB ({100 * r['extra_gb'] / r['data_gb']:.0f}%)")

# ================================================================ Figure: scalability (K, n, d, threads)
KS = load("exp05_scale_k.jsonl")
NS = load("exp05_scale_n.jsonl")
DS_ = load("exp05_scale_d.jsonl")
THR = load("exp05_threads.jsonl")  # run on an otherwise idle machine
if KS or NS or DS_ or THR:
    # flat panels, as in the small-multiples grids; rows only as far apart as the labels need
    SC_H, SC_PANEL, SC_LEGEND, SC_LABELS = 1.82, 0.50, 0.30, 0.26   # inches: figure, panel, legend rows, ticks + x label
    fig, axes = plt.subplots(2, 2, figsize=(3.33, SC_H))
    fig.subplots_adjust(left=0.115, right=0.995, top=1 - SC_LEGEND / SC_H, bottom=SC_LABELS / SC_H, wspace=0.3,
                        hspace=(SC_H - SC_LEGEND - SC_LABELS - 2 * SC_PANEL) / SC_PANEL)
    axes = axes.ravel()
    SM = ["FullSuperKMeans", "EMeans-5", "HierSuperKMeans", "QuantSuperKMeans", "SketchCD"]

    def series_by(rs, xkey):
        d = defaultdict(dict)
        for r in rs:
            d[r["method"]][xkey(r)] = r
        return d

    # number of lists on GIST
    ax = axes[0]
    S = series_by(KS, lambda r: r["K"])
    out("\n== K SWEEP (GIST, seed 42): K: time s / R@10 %")
    for m in SM:
        if S.get(m):
            xs = sorted(S[m])
            plot_method(ax, m, xs, [S[m][k]["build_s"] for k in xs])
            out(f"{m:18s} " + " ".join(f"K={k}: {S[m][k]['build_s']:.2f}/{100 * R(S[m][k]):.2f}" for k in xs))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{round(math.log10(v))}}}$"))
    ax.set_yticks([1, 10, 100])
    ax.yaxis.set_major_formatter(G)
    ax.set_ylim(1, 300)
    ax.set_xlabel("(a) Lists $K$ (GIST)", labelpad=1)
    ax.set_ylabel("Build time (s)", labelpad=1)

    # number of vectors: prefixes of DEEP-10M (the 10M point is the main run, seed 42)
    ax = axes[1]
    nrs = NS + [r for m in SM for r in MAIN.get(("deep10m", m), []) if r["seed"] == 42]
    S = series_by(nrs, lambda r: r["N"])
    out("\n== n SCALING (DEEP-10M prefixes, seed 42): n: time s / R@10 %")
    for m in SM:
        if S.get(m):
            xs = sorted(S[m])
            plot_method(ax, m, xs, [S[m][n]["build_s"] for n in xs])
            out(f"{m:18s} " + " ".join(f"n={n}: {S[m][n]['build_s']:.2f}/{100 * R(S[m][n]):.2f}" for n in xs))
    if S.get("SketchCD"):
        xs = sorted(S["SketchCD"])
        ax.plot([xs[0], xs[-1]], [S["SketchCD"][xs[0]]["build_s"], S["SketchCD"][xs[0]]["build_s"] * xs[-1] / xs[0]], color=MUTED,
                linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{round(math.log10(v))}}}$"))
    ax.yaxis.set_major_formatter(G)
    ax.set_xlabel("(b) Vectors $n$ (DEEP-10M)", labelpad=1)

    # dimension: Tiny-ImageNet at 8x8 ... 64x64 pixels (the 64x64 point is the main run, seed 42)
    ax = axes[2]
    drs = DS_ + [r for m in SM for r in MAIN.get(("tinyimagenet", m), []) if r["seed"] == 42]
    S = series_by(drs, lambda r: r["d"])
    out("\n== d SCALING (Tiny-ImageNet resolutions, seed 42): d: time s / R@10 %")
    for m in SM:
        if S.get(m):
            xs = sorted(S[m])
            plot_method(ax, m, xs, [S[m][x]["build_s"] for x in xs])
            out(f"{m:18s} " + " ".join(f"d={x}: {S[m][x]['build_s']:.2f}/{100 * R(S[m][x]):.2f}" for x in xs))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([192, 768, 3072, 12288])
    ax.set_xticklabels(["192", "768", "3072", "12,288"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(G)
    ax.set_xlabel("(c) Dimension $d$ (Tiny-ImageNet)", labelpad=1)
    ax.set_ylabel("Build time (s)", labelpad=1)

    # threads
    ax = axes[3]
    S = defaultdict(dict)
    for r in last_by(THR, lambda r: (r["dataset"], r["method"])).values():
        S[r["dataset"]][int(r["method"][1:])] = r["build_s"]
    out("\n== THREADS (seed 42): t: time s (speedup)")
    for ds in TRIO:
        if S.get(ds) and 1 in S[ds]:
            t = sorted(S[ds])
            ds_line(ax, ds, t, [S[ds][1] / S[ds][k] for k in t], label=TITLE[ds])
            out(f"{ds:10s} " + " ".join(f"t={k}: {S[ds][k]:.2f} ({S[ds][1] / S[ds][k]:.1f}x)" for k in t))
    ax.plot([1, 20], [1, 20], color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(16.3, 12.6, "ideal", fontsize=5.8, color=INK2, ha="left", va="center")    # between the diagonal and the curves
    ax.set_xticks([1, 5, 10, 20])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_xlabel("(d) Threads", labelpad=1)
    ax.set_ylabel("Speedup", labelpad=1)
    for ax_ in axes:
        ax_.tick_params(labelsize=6.0)
        ax_.xaxis.set_minor_formatter(NullFormatter()) if ax_ is not axes[1] else None
    ax.set_ylim(0, 21)
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.3, labelspacing=0.1, handletextpad=0.4, borderaxespad=0.1, borderpad=0.15)
    fig.legend(handles=[handle(m) for m in ["SketchCD", "FullSuperKMeans", "HierSuperKMeans", "QuantSuperKMeans", "EMeans-5"]],
               loc="upper center", ncol=3, bbox_to_anchor=(0.555, 1.0), columnspacing=1.0, handlelength=2.0, fontsize=6.3,
               borderaxespad=0)
    save(fig, "scaling")

# ================================================================ LaTeX table bodies
TEXNAME = {m: m for m in ORDER}
TEXNAME["SketchCD"] = r"\sys{}"
TEXNAME["SketchCD-Fast"] = r"\sys-Fast"


def bold_if(s, cond):
    return r"\textbf{" + s + "}" if cond else s


def emit(m, row):
    return ours_row(row) if m.startswith("SketchCD") else " & ".join(row) + r" \\"


# ---- main table: two blocks of five datasets, the whole tabular spread over the text width by \filltable
# a cell without a run is a dash, and the table note of the paper gives the reason: QuantSuperKMeans does not support
# d > 4096; BatchedHartigan needs d >= 128 (see the load of exp01_extra_baselines.jsonl); PCA-FaissKMeans is FaissKMeans where SketchCD uses
# no sketch (d <= 256)
BLOCKS = [ALLD[:5], ALLD[5:]]
body = [r"\filltable{16}{%", r"\begin{tabular}{@{}l rrr rrr rrr rrr rrr@{}}", r"\toprule"]
for bi, BLOCK in enumerate(BLOCKS):
    body += ([r"\midrule"] if bi > 0 else []) + [
        " & " + " & ".join(r"\multicolumn{3}{c}{" + TITLE[x] + "}" for x in BLOCK) + r"\\",
        "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(BLOCK))),
        "Method" + r" & Time & R@10 & $\Delta J$" * len(BLOCK) + r"\\",
        r"\midrule"]
    for m in ORDER:
        row = [TEXNAME[m] + (r"$^\dagger$" if m == "FaissKMeans" else "")]
        for ds in BLOCK:
            if not MAIN.get((ds, m)):
                row += [r"\multicolumn{3}{c}{--}"]
                continue
            t, r10, ex = T(ds, m), RC(ds, m), EX(ds, m)
            cols = [x for x in ORDER if MAIN.get((ds, x))]
            row += [bold_if(fmt_t(t), t == min(T(ds, x) for x in cols)),
                    bold_if(f"{r10:.2f}", round(r10, 2) == round(max(RC(ds, x) for x in cols), 2)),
                    bold_if(f"{ex:.2f}".replace("-", "$-$"), round(ex, 2) == round(min(EX(ds, x) for x in cols), 2))]
        body.append(emit(m, row))
body += [r"\bottomrule", r"\end{tabular}}"]
wtab("main_rows.tex", "\n".join(body) + "\n")

# ---- Recall@k table: mean over the ten datasets at each dataset's scan budget
vals = {}
for m in ORDER:
    vals[m] = [st.mean(RC(ds, m, k) for ds in ALLD if MAIN.get((ds, m))) for k in (1, 10, 30, 50, 100)]
FULL10 = [m for m in ORDER if all(MAIN.get((ds, m)) for ds in ALLD)]
lines = [r"\filltable{6}{%", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", r"Method & R@1 & R@10 & R@30 & R@50 & R@100\\",
         r"\midrule"]
for m in ORDER:
    nds = sum(bool(MAIN.get((ds, m))) for ds in ALLD)  # a mean over fewer than ten datasets is marked and never set in bold
    row = [TEXNAME[m] + {10: "", 9: r"$^\ddagger$", 8: r"$^\S$"}[nds]] + \
          [bold_if(f"{v:.2f}", nds == 10 and round(v, 2) == round(max(vals[x][i] for x in FULL10), 2)) for i, v in enumerate(vals[m])]
    lines.append(emit(m, row))
lines += [r"\bottomrule", r"\end{tabular}}"]
wtab("recallk_rows.tex", "\n".join(lines) + "\n")

# ================================================================ exact coordinate descent (CDKM) on the analysis trio
# cdkm.cpp: K random vectors, nearest assignment, then sequential passes of the size-aware rule over all K lists in full
# dimension (every move committed before the next decision). CDKM: 10 passes, three seeds; CDKM@p: p passes, seed 42.
CD = defaultdict(list)
for r in load("exp01_extra_baselines.jsonl"):
    if r["method"].startswith("CDKM"):
        CD[(r["dataset"], r["method"])].append(r)
if CD:
    def cdm(ds, m, f):
        return mean(CD[(ds, m)], f)

    out("\n== EXACT COORDINATE DESCENT (CDKM): n | time s (x SketchCD) | WCSS excess % | R@10 % at the budget (vs SketchCD) | cv | tiny %")
    for ds in [d_ for d_ in ALLD if CD.get((d_, "CDKM"))]:
        out(f"--- {ds}: SketchCD {T(ds, 'SketchCD'):.2f}s / {EX(ds, 'SketchCD'):.2f} / {RC(ds, 'SketchCD'):.2f} / cv {cell(ds, 'SketchCD', lambda r: r['cv']):.3f} / "
            f"tiny {100 * cell(ds, 'SketchCD', lambda r: r['tiny']):.2f}; seed 42 alone: "
            + "; ".join(f"{r['build_s']:.2f}s / {r['excess']:.2f} / {100 * R(r):.2f}" for r in MAIN[(ds, 'SketchCD')] if r["seed"] == 42))
        for m in ("CDKM@1", "CDKM@2", "CDKM@3", "CDKM"):
            rs = CD.get((ds, m))
            if not rs:
                continue
            t = cdm(ds, m, lambda r: r["build_s"])
            out(f"{m:8s} n={len(rs)} seeds {','.join(str(r['seed']) for r in rs)} {t:8.2f}s ({t / T(ds, 'SketchCD'):5.1f}x) | {cdm(ds, m, lambda r: r['excess']):5.2f} | "
                f"{100 * cdm(ds, m, R):6.2f} ({100 * cdm(ds, m, R) - RC(ds, 'SketchCD'):+.2f}; sd {100 * st.pstdev([R(r) for r in rs]):.2f}) | cv {cdm(ds, m, lambda r: r['cv']):.3f} | "
                f"tiny {100 * cdm(ds, m, lambda r: r['tiny']):.2f} | R@100 {100 * cdm(ds, m, lambda r: R(r, 100)):.2f} ({100 * cdm(ds, m, lambda r: R(r, 100)) - RC(ds, 'SketchCD', 100):+.2f})")
        r = next(r for r in CD[(ds, "CDKM")] if r["seed"] == 42)
        o42 = next(o for o in MAIN[(ds, "SketchCD")] if o["seed"] == 42)
        out(f"         seed 42 against SketchCD of seed 42: time {r['build_s'] / o42['build_s']:.1f}x, R@10 {100 * (R(r) - R(o42)):+.2f}, "
            f"excess {r['excess']:.2f} vs {o42['excess']:.2f}, cv {r['cv']:.3f} vs {o42['cv']:.3f}")
        p = r["parts"]
        out(f"         seed 42: init {p['init_ms'] / 1e3:.1f}s, passes (s) " + " ".join(f"{x / 1e3:.1f}" for x in p["pass_ms"])
            + f" | moves (% of n) " + " ".join(f"{100 * x / r['N']:.1f}" for x in p["moves"])
            + f" | one nKd pass at peak GEMM throughput {r['N'] * r['K'] * r['d'] / GEMM_MADD:.1f}s")
    trio_cd = [ds for ds in TRIO if CD.get((ds, "CDKM"))]
    if trio_cd:
        f_ = lambda v: f"{min(v):.2f}-{max(v):.2f}"  # noqa: E731
        out("-- trio summary: CDKM (10 passes) slower than SketchCD by " + ", ".join(
            f"{ds} {cdm(ds, 'CDKM', lambda r: r['build_s']) / T(ds, 'SketchCD'):.0f}x" for ds in trio_cd)
            + " | one pass (CDKM@1) by " + ", ".join(f"{ds} {cdm(ds, 'CDKM@1', lambda r: r['build_s']) / T(ds, 'SketchCD'):.1f}x" for ds in trio_cd)
            + " | first pass alone / SketchCD build: " + ", ".join(
                f"{ds} {cdm(ds, 'CDKM', lambda r: r['parts']['pass_ms'][0]) / 1e3 / T(ds, 'SketchCD'):.1f}x" for ds in trio_cd))
        out("   excess: CDKM " + f_([cdm(ds, "CDKM", lambda r: r["excess"]) for ds in trio_cd]) + ", CDKM@1 "
            + f_([cdm(ds, "CDKM@1", lambda r: r["excess"]) for ds in trio_cd]) + ", CDKM@3 " + f_([cdm(ds, "CDKM@3", lambda r: r["excess"]) for ds in trio_cd])
            + ", SketchCD " + f_([EX(ds, "SketchCD") for ds in trio_cd]) + ", BatchedHartigan " + f_([EX(ds, "BatchedHartigan") for ds in trio_cd]))
        out("   cv: CDKM " + f_([cdm(ds, "CDKM", lambda r: r["cv"]) for ds in trio_cd]) + ", SketchCD "
            + f_([cell(ds, "SketchCD", lambda r: r["cv"]) for ds in trio_cd]) + " | BatchedHartigan slower than SketchCD by "
            + ", ".join(f"{ds} {T(ds, 'BatchedHartigan') / T(ds, 'SketchCD'):.0f}x" for ds in trio_cd)
            + " | CDKM slower than BatchedHartigan by " + ", ".join(
                f"{ds} {cdm(ds, 'CDKM', lambda r: r['build_s']) / T(ds, 'BatchedHartigan'):.1f}x" for ds in trio_cd))

        if ABL:
            out("   list balance (cv / tiny % / excess / R@10): CDKM (3 seeds) | BatchedHartigan (3 seeds) | SketchCD without cells "
                "(ablation, seed 42) | SketchCD (ablation default, seed 42)")
            for ds in trio_cd:
                fl, sk = ABL[(ds, "flat-first-pass")], ABL[(ds, "default")]
                out(f"   {ds:10s} {cdm(ds, 'CDKM', lambda r: r['cv']):.3f} / {100 * cdm(ds, 'CDKM', lambda r: r['tiny']):.2f} / "
                    f"{cdm(ds, 'CDKM', lambda r: r['excess']):.2f} / {100 * cdm(ds, 'CDKM', R):.2f} | "
                    f"{cell(ds, 'BatchedHartigan', lambda r: r['cv']):.3f} / {100 * cell(ds, 'BatchedHartigan', lambda r: r['tiny']):.2f} / "
                    f"{EX(ds, 'BatchedHartigan'):.2f} / {RC(ds, 'BatchedHartigan'):.2f} | "
                    + " | ".join(f"{r['cv_lab']:.3f} / {100 * r['tiny_lab']:.2f} / {ex_lab(r):.2f} / {100 * r[f'R10@{BUD[ds]}_lab']:.2f}" for r in (fl, sk)))

        def cd_cell(t, ex, r10):
            return f"{fmt_t(t)} / {ex:.2f} / {r10:.2f}"

        rows = [ours_row([r"\sys{}"] + [cd_cell(T(ds, "SketchCD"), EX(ds, "SketchCD"), RC(ds, "SketchCD")) for ds in trio_cd])
                + "\n\\addlinespace[2pt]"]
        for m, lab in (("CDKM@1", r"CDKM, 1 pass$^\dagger$"), ("CDKM@2", r"CDKM, 2 passes$^\dagger$"), ("CDKM@3", r"CDKM, 3 passes$^\dagger$"),
                       ("CDKM", "CDKM, 10 passes")):
            rows.append(" & ".join([lab] + [cd_cell(cdm(ds, m, lambda r: r["build_s"]), cdm(ds, m, lambda r: r["excess"]), 100 * cdm(ds, m, R))
                                            for ds in trio_cd]) + r"\\")
        rows.append(" & ".join(["BatchedHartigan"] + [cd_cell(T(ds, "BatchedHartigan"), EX(ds, "BatchedHartigan"), RC(ds, "BatchedHartigan"))
                                                      for ds in trio_cd]) + r"\\")
        wtab("cdkm_rows.tex", "\\filltable{4}{%\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\nMethod & " + " & ".join(TITLE[ds] for ds in trio_cd)
             + "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n")

# ================================================================ scale experiment: BIGANN-100M (one seed, K = 40,000)
S100 = {r["method"]: r for r in load("exp05_bigann100m.jsonl")}
FAILED100 = []
_fp = os.path.join(RES, "exp05_bigann100m_stopped.txt")
if os.path.exists(_fp):
    FAILED100 = [ln.rstrip("\r\n").split("\t") for ln in open(_fp) if ln.strip()]
if S100:
    o100 = S100["SketchCD"]
    c100 = [m for m in COMP if m in S100]
    b100 = next(s for s in SCAN if st.mean(S100[m]["R10_scan"][s] for m in c100 if m in SEVEN) >= 0.90)
    out(f"\n== SCALE EXPERIMENT ({o100['dataset']}: n={o100['N']} d={o100['d']} K={o100['K']}, mean list size {o100['N'] / o100['K']:.0f}; seed {o100['seed']})")
    out(f"-- scan budget by the rule of the paper (competitors of the seven that ran: {', '.join(m for m in c100 if m in SEVEN)}): {pct(b100)}% "
        f"(mean R@10 {100 * st.mean(S100[m]['R10_scan'][b100] for m in c100 if m in SEVEN):.2f}; with BatchedHartigan "
        f"{100 * st.mean(S100[m]['R10_scan'][b100] for m in c100):.2f})")
    out("-- method: time s (min; x SketchCD) | WCSS relative to SketchCD % | cv | max/mean | R@1, R@10, R@100 at the budget | R@10 at 0.2%, 0.5%, 1%")
    for m in ORDER:
        r = S100.get(m)
        if r:
            out(f"{m:18s} {r['build_s']:9.1f}s ({r['build_s'] / 60:6.1f} min; {r['build_s'] / o100['build_s']:6.2f}x) | {100 * (r['wcss'] / o100['wcss'] - 1):+.2f} | "
                f"cv {r['cv']:.3f} | {r['max_over_mean']:.1f} | " + " ".join(f"{100 * r[f'R{k}_scan'][b100]:.2f}" for k in (1, 10, 100))
                + " | " + " ".join(f"{100 * r['R10_scan'][s]:.2f}" for s in ("0.002", "0.005", "0.01"))
                + " | " + " ".join(f"{k}={v:.0f}" for k, v in r["parts"].items() if v))
    for ds_, m, why, wall in FAILED100:
        out(f"{m:18s} STOPPED after {wall}: {why}")
    best_c = max(c100, key=lambda m: S100[m]["R10_scan"][b100])
    out(f"-- most accurate competitor at the budget: {best_c}, R@10 {100 * (S100[best_c]['R10_scan'][b100] - o100['R10_scan'][b100]):+.2f} points against SketchCD at "
        f"{S100[best_c]['build_s'] / o100['build_s']:.0f}x the build time; at 0.2% {100 * (S100[best_c]['R10_scan']['0.002'] - o100['R10_scan']['0.002']):+.2f}, at 0.5% "
        f"{100 * (S100[best_c]['R10_scan']['0.005'] - o100['R10_scan']['0.005']):+.2f}; R@1 {100 * (S100[best_c]['R1_scan'][b100] - o100['R1_scan'][b100]):+.2f}, "
        f"R@100 {100 * (S100[best_c]['R100_scan'][b100] - o100['R100_scan'][b100]):+.2f} at the budget")
    out("-- competitors below SketchCD in R@10 at the budget: " + ", ".join(
        f"{m} {100 * (S100[m]['R10_scan'][b100] - o100['R10_scan'][b100]):+.2f}" for m in c100 if S100[m]["R10_scan"][b100] < o100["R10_scan"][b100]))
    out("-- fastest competitor that finished: %s, %.0fx" % (min(c100, key=lambda m: S100[m]["build_s"]),
                                                          min(S100[m]["build_s"] for m in c100) / o100["build_s"]))
    r10m = [r for r in MAIN[("bigann10m", "SketchCD")]]
    out(f"-- from BIGANN-10M to BIGANN-100M: SketchCD {T('bigann10m', 'SketchCD'):.2f}s (K={r10m[0]['K']}, mean of seeds; seed 42: "
        f"{next(r['build_s'] for r in r10m if r['seed'] == 42):.2f}s) -> {o100['build_s']:.1f}s: {o100['build_s'] / T('bigann10m', 'SketchCD'):.1f}x for "
        f"{o100['N'] / r10m[0]['N']:.0f}x the vectors and {o100['K'] / r10m[0]['K']:.2f}x the lists; stages (s): first pass "
        f"{mean(r10m, lambda r: r['parts']['pass_ms']) / 1e3:.2f} -> {o100['parts']['pass_ms'] / 1e3:.1f}, sketch sweeps "
        f"{mean(r10m, lambda r: r['parts']['sweep_ms']) / 1e3:.2f} -> {o100['parts']['sweep_ms'] / 1e3:.1f}, relabeling "
        f"{mean(r10m, lambda r: r['parts']['fnn_ms']) / 1e3:.2f} -> {o100['parts']['fnn_ms'] / 1e3:.1f}")
    for m in c100:
        if T("bigann10m", m):
            out(f"   {m:18s} {T('bigann10m', m):8.1f}s -> {S100[m]['build_s']:9.1f}s: {S100[m]['build_s'] / T('bigann10m', m):5.1f}x")
    out(f"-- one nKd pass at peak GEMM throughput: {o100['N'] * o100['K'] * o100['d'] / GEMM_MADD:.0f}s ({o100['N'] * o100['K'] * o100['d'] / GEMM_MADD / o100['build_s']:.1f}x the SketchCD build); "
        f"input {o100['N'] * o100['d'] * 4 / 1e9:.1f} GB")

    def c100cell(m, f, fmt, best):
        return bold_if(fmt(f(S100[m])), f(S100[m]) == best([f(S100[x]) for x in S100]))

    def f_min(t):
        return f"{t / 60:.1f}" if t < 600 else f"{t / 60:.0f}"

    def f_x(v):
        return (f"{v:.1f}" if v < 10 else f"{v:.0f}") + r"$\times$"

    rows100 = []
    for m in [x for x in ORDER if x in S100]:
        r = S100[m]
        row = [TEXNAME[m],
               c100cell(m, lambda r: r["build_s"], f_min, min),
               f_x(r["build_s"] / o100["build_s"]),
               c100cell(m, lambda r: round(100 * r["R10_scan"][b100], 2), lambda v: f"{v:.2f}", max),
               c100cell(m, lambda r: round(100 * r["R100_scan"][b100], 2), lambda v: f"{v:.2f}", max),
               c100cell(m, lambda r: round(100 * r["R10_scan"]["0.005"], 2), lambda v: f"{v:.2f}", max),
               c100cell(m, lambda r: round(100 * (r["wcss"] / o100["wcss"] - 1), 2), lambda v: f"{v:+.2f}".replace("-", "$-$").replace("+", "$+$") if v else "0.00", min),
               c100cell(m, lambda r: round(r["cv"], 2), lambda v: f"{v:.2f}", min)]
        rows100.append(emit(m, row))
    if FAILED100:  # the configurations that the watchdog stopped, in one row above SketchCD
        k_ = next(i for i, ln in enumerate(rows100) if ln.startswith("\\oursL"))
        rows100.insert(k_, r"SuperKMeans$^\ast$ & \multicolumn{7}{c}{out of memory} \\")
    wtab("scale100m_rows.tex", "\\filltable{8}{%\n\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
         " & \\multicolumn{2}{c}{Build time} & \\multicolumn{2}{c}{" + pct(b100) + "\\% scanned} & 0.5\\% scanned & & \\\\\n"
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\n"
         "Method & minutes & relative & R@10 & R@100 & R@10 & $\\Delta J_{\\mathrm{rel}}$ & CV\\\\\n\\midrule\n"
         + "\n".join(rows100) + "\n\\bottomrule\n\\end{tabular}}\n")



# ================================================================ numbers that answer the reviews (quoted in Sections 1, 4, 6)
out("\n== REVIEW NUMBERS")
# speedup over the configuration that SuperKMeans selects by itself: the sampled flat default up to 100,000 vectors,
# the hierarchical one above
LIB = {ds: ("SampleSuperKMeans" if MAIN[(ds, "SketchCD")][0]["N"] <= 100_000 else "HierSuperKMeans") for ds in ALLD}
_sp = [T(ds, LIB[ds]) / T(ds, "SketchCD") for ds in ALLD]
_g = [RC(ds, "SketchCD") - RC(ds, LIB[ds]) for ds in ALLD]
out(f"-- library-selected SuperKMeans configuration: speedup geo-mean {gmean(_sp):.2f}x ({min(_sp):.1f}-{max(_sp):.1f}); "
    f"R@10 gain {min(_g):+.2f} to {max(_g):+.2f}, mean {st.mean(_g):+.2f}")
# speedup within each seed (SketchCD of seed s against each competitor of seed s)
_pairs = []
for m in COMP:
    for ds in ALLD:
        own = {r["seed"]: r["build_s"] for r in MAIN.get((ds, "SketchCD"), [])}
        _pairs += [(r["build_s"] / own[r["seed"]], ds, m, r["seed"]) for r in MAIN.get((ds, m), []) if r["seed"] in own]
_pairs.sort()
out(f"-- smallest speedup within one seed: {_pairs[0][0]:.2f}x ({_pairs[0][2]}, {_pairs[0][1]}, seed {_pairs[0][3]})")


def _wilcoxon(d):
    """Exact two-sided Wilcoxon signed-rank test of paired differences (no ties among |d| in our data)."""
    d = [x for x in d if x != 0]
    n = len(d)
    rank = {i: k + 1 for k, i in enumerate(sorted(range(n), key=lambda i: abs(d[i])))}
    wneg = sum(rank[i] for i in range(n) if d[i] < 0)
    w = min(wneg, n * (n + 1) // 2 - wneg)
    cnt = [1] + [0] * (n * (n + 1) // 2)  # number of sign patterns with each rank sum
    for k in range(1, n + 1):
        for v in range(len(cnt) - 1, k - 1, -1):
            cnt[v] += cnt[v - k]
    return wneg, min(1.0, 2 * sum(cnt[: w + 1]) / 2 ** n), n


out("-- paired Wilcoxon signed-rank test over the datasets, R@10 of SketchCD minus R@10 of the method (mean of seeds):")
_ps = []
for m in COMP:
    d = [RC(ds, "SketchCD") - RC(ds, m) for ds in ALLD if MAIN.get((ds, m))]
    wneg, p_, n_ = _wilcoxon(d)
    _ps.append(p_)
    out(f"   {m:18s} n={n_:2d} higher on {sum(x > 0 for x in d):2d}  W-={wneg:2d}  p={p_:.4f}")
out(f"   largest p: {max(_ps):.4f}")
_sd = {}
for ds in ("cifar100", "fiqa"):
    for m in ["SketchCD"] + COMP:
        rs = MAIN.get((ds, m), [])
        if len(rs) > 1 and (m == "SketchCD" or RC(ds, m) > RC(ds, "SketchCD")):
            _sd[(ds, m)] = 100 * st.stdev([R(r) for r in rs])
out("-- sample sd of R@10 where a competitor is ahead, and of SketchCD there: " + ", ".join(
    f"{ds} {m} {v:.2f}" for (ds, m), v in _sd.items()) + f"  => {min(_sd.values()):.2f}-{max(_sd.values()):.2f}")
# vectors whose list is not that of their nearest returned centroid (measured for n <= 2M)
out("-- share of vectors outside the list of their nearest returned centroid (%, mean of seeds, range over datasets):")
for m in ORDER:
    v = [100 * st.mean(r["label_ne_nearest"] for r in MAIN[(ds, m)]) for ds in ALLD
         if MAIN.get((ds, m)) and all("label_ne_nearest" in r for r in MAIN[(ds, m)])]
    if v:
        out(f"   {m:18s} {min(v):.2f}-{max(v):.2f} over {len(v)} datasets")
# where the time of HierSuperKMeans goes at ten million vectors
for ds in TEN:
    rs = MAIN[(ds, "HierSuperKMeans")]
    tr = [r["parts"]["train_s"] for r in rs]
    asg = [r["build_s"] - r["parts"]["train_s"] for r in rs]
    out(f"-- HierSuperKMeans on {ds}: train {min(tr):.1f}-{max(tr):.1f}s, final assignment {min(asg):.1f}-{max(asg):.1f}s, "
        f"total {min(r['build_s'] for r in rs):.1f}-{max(r['build_s'] for r in rs):.1f}s")
# the BatchedHartigan build against the pip build of SuperKMeans on identical work: DEEP-10M, where the fork makes no
# size-aware move and its output equals FullSuperKMeans
_bh = {r["seed"]: r["build_s"] for r in load("exp01_extra_baselines.jsonl")
       if r["dataset"] == "deep10m" and r["method"] == "BatchedHartigan"}
_fu = {r["seed"]: r["build_s"] for r in MAIN[("deep10m", "FullSuperKMeans")]}
_rt = [_bh[s] / _fu[s] for s in sorted(_bh) if s in _fu]
out(f"-- BatchedHartigan without size-aware moves (deep10m) vs FullSuperKMeans: {min(_rt):.2f}-{max(_rt):.2f}x the time")
# singleton lists: lower bounds from the size quantiles of the seed-42 builds (a quantile of at most one vector)
out("-- share of singleton lists (lower bound from the quantiles q = 1, 5, 10, 25%), seed 42:")
for r in load("exp09_list_sizes.jsonl"):
    if r["method"] in ("SketchCD", "SketchCD@weighted=0", "FullSuperKMeans") and r["dataset"] in ALLD:
        mean_ = r["n"] / r["K"]
        s1 = max([q for q, v in zip(r["q"], r["rel_quantiles"]) if q > 0 and v * mean_ <= 1.0 + 1e-9], default=0.0)
        out(f"   {r['dataset']:12s} {r['method']:20s} >= {100 * s1:4.0f}%")
# thread scaling per stage from 10 to 20 threads
_thr = last_by(THR, lambda r: (r["dataset"], r["method"]))
for key, name in (("cent_ms", "full-dimensional means"), ("sweep_ms", "sketch sweeps"), ("xsweep_ms", "verified sweeps"),
                  ("fnn_ms", "final relabeling"), ("pass_ms", "first pass")):
    v = [_thr[(ds, "t10")]["run"][key] / _thr[(ds, "t20")]["run"][key] for ds in TRIO]
    out(f"-- 10 -> 20 threads, {name}: {min(v):.2f}-{max(v):.2f}x")
# the principal directions: largest time of the basis estimation over all records, by thread count
_bm = defaultdict(float)
for f in sorted(os.listdir(RES)):
    if f.endswith(".jsonl"):
        for r in load(f):
            u = r.get("run") or r.get("parts") or {}
            if isinstance(u, dict) and "basis_ms" in u:
                _bm[int(u.get("threads", 20) or 20)] = max(_bm[int(u.get("threads", 20) or 20)], u["basis_ms"] / 1e3)
out("-- basis estimation, largest time: " + ", ".join(f"{t} threads {v:.2f}s" for t, v in sorted(_bm.items())))
# sketch dimension: Recall@10 of the fastest sketch against the default one
for ds in TRIO:
    rs = sorted(DIMR.get(ds, []), key=lambda r: r["run"]["l"])
    if rs:
        f_ = min(rs, key=lambda r: r["build_s"])
        d_ = next(r for r in rs if r["run"]["l"] == MAIN[(ds, "SketchCD")][0]["parts"]["l"])
        out(f"-- {ds}: R@10 of the fastest sketch (l={f_['run']['l']}) {100 * f_[f'R10@{BUD[ds]}_lab']:.2f} vs default "
            f"(l={d_['run']['l']}) {100 * d_[f'R10@{BUD[ds]}_lab']:.2f}")


# ---- added after the final review
_hp = sorted(_ps)
_holm = max(min(1.0, (len(_hp) - k) * p_) for k, p_ in enumerate(_hp))
out(f"-- Holm-adjusted largest p of the nine Wilcoxon tests: {_holm:.4f}")
_sp9 = [T(ds, LIB[ds]) / T(ds, "SketchCD") for ds in ALLD if ds != "tinyimagenet"]
out(f"-- library-selected SuperKMeans configuration without Tiny-ImageNet: speedup geo-mean {gmean(_sp9):.2f}x")
_tb = {ds: 100 * st.mean(r["tiny"] for r in MAIN[(ds, "BatchedHartigan")]) for ds in ALLD if MAIN.get((ds, "BatchedHartigan"))}
_tf = {ds: 100 * st.mean(r["tiny"] for r in MAIN[(ds, "FullSuperKMeans")]) for ds in ALLD if MAIN.get((ds, "FullSuperKMeans"))}
out(f"-- tiny lists (%, mean of seeds): BatchedHartigan at most {max(_tb.values()):.2f}, FullSuperKMeans at most {max(_tf.values()):.2f}; "
    + ", ".join(f"{ds} {_tf[ds]:.1f}->{_tb[ds]:.2f}" for ds in _tb))
_paths = {ds: sorted({r["parts"].get("path") for r in MAIN[(ds, "QuantSuperKMeans")]}) for ds in ALLD if MAIN.get((ds, "QuantSuperKMeans"))}
out("-- QuantSuperKMeans assignment path per dataset: " + ", ".join(f"{ds} {'/'.join(v)}" for ds, v in _paths.items()))


num.close()
print("figures written to", FIG, "| tables to", TAB)
