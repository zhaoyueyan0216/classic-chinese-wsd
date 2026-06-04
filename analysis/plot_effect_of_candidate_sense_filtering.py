"""
plot_effect_of_candidate_sense_filtering.py
--------------------------------------------
Produces a two-panel figure:
  Left  — Accuracy vs. Candidate Sense Set Size
  Right — Accuracy vs. Normalised Position of Gold Sense

Both panels compare PureLLM and NoHistory (TopK filtering only).

Usage:
    python plot_effect_of_candidate_sense_filtering.py

Configuration:
    Edit the CONFIG section below.
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import pearsonr

# ── CONFIG ────────────────────────────────────────────────────────────────────

INPUT_FILE = "data_combined_updated.xlsx"

# ---- Panel A: Candidate Set Size ----
SIZE_BINS   = [0, 5, 10, 20, 30, 50, 200]
SIZE_LABELS = ["1–5", "6–10", "11–20", "21–30", "31–50", "51+"]

# ---- Panel B: Gold Sense Position ----
GOLD_RANK_COL       = "gold_rank_all"
CANDIDATE_COUNT_COL = "candidate_count"
N_RANK_BINS         = 5          # number of equal-width bins over [0, 1]

# ---- Methods (shared by both panels) ----
METHODS = [
    {
        "column":    "onlyllm_correct",
        "label":     "PureLLM (DeepSeek)",
        "color":     "#E15759",
        "marker":    "o",
        "trendline": True,          # draw dashed trendline in Panel B
    },
    {
        "column":    "nohistory_correct",
        "label":     "NoHistory (TopK only)",
        "color":     "#4E79A7",
        "marker":    "s",
        "trendline": False,
    },
]

# Show Pearson correlation annotation for these method columns (Panel B)
SHOW_CORRELATION = ["onlyllm_correct"]

FIGURE_SIZE = (15, 5.5)           # overall figure size (width, height)
Y_LIM_A     = (20, 90)            # y-axis limits for Panel A
Y_LIM_B     = (18, 82)            # y-axis limits for Panel B
SHOW_N      = True                # show sample count below x-axis

OUTPUT_PDF  = "effect_of_candidate_sense_filtering.pdf"
OUTPUT_PNG  = "effect_of_candidate_sense_filtering.png"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def smart_offsets(all_vals: list, gap_threshold: float = 6.0) -> list:
    """
    Return per-method, per-bin y-offsets (points) to avoid label overlap.
    all_vals : list of arrays, one per method.
    """
    n_methods = len(all_vals)
    n_bins    = len(all_vals[0])
    offsets   = []
    for m in range(n_methods):
        row = []
        for i in range(n_bins):
            v      = all_vals[m][i]
            others = [all_vals[j][i] for j in range(n_methods) if j != m]
            close  = any(abs(v - o) < gap_threshold for o in others)
            above  = all(v >= o for o in others)
            if close:
                row.append(+13 if above else -17)
            else:
                row.append(+11 if above else -15)
        offsets.append(row)
    return offsets


def draw_lines(ax, x, all_vals, offsets, methods,
               trendline=False, label_fontsize=8.5):
    """Plot lines, markers, and value labels onto ax."""
    for idx, m in enumerate(methods):
        vals = all_vals[idx]
        ax.plot(x, vals,
                marker=m["marker"], linewidth=2, markersize=7,
                color=m["color"], label=m["label"], zorder=3)
        if trendline and m.get("trendline"):
            z = np.polyfit(x, vals, 1)
            ax.plot(x, np.poly1d(z)(x),
                    linestyle="--", linewidth=1.4,
                    color=m["color"], alpha=0.5, zorder=2)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}%", (x[i], v),
                        textcoords="offset points",
                        xytext=(0, offsets[idx][i]),
                        ha="center", fontsize=label_fontsize,
                        color=m["color"], fontweight="bold")


def draw_sample_counts(ax, x, counts, y_bottom):
    for i, c in enumerate(counts):
        ax.annotate(f"n={c}", (x[i], y_bottom + 1),
                    ha="center", fontsize=7.5, color="gray")


def style_ax(ax, xtick_labels, xlabel, ylabel, title, y_lim,
             xtick_fontsize=10, xtick_rotation=0):
    x = np.arange(len(xtick_labels))
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=xtick_fontsize,
                       rotation=xtick_rotation,
                       ha="right" if xtick_rotation else "center")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(y_lim)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

    df = pd.read_excel(INPUT_FILE)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    fig.suptitle("Effect of Candidate Sense Filtering",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── Panel A: Candidate Set Size ───────────────────────────────────────────
    df["sense_group"] = pd.cut(
        df["candidate_count"], bins=SIZE_BINS, labels=SIZE_LABELS
    )
    grp_a = (
        df.groupby("sense_group", observed=True)
        .agg(
            count=("candidate_count", "count"),
            **{m["column"]: (m["column"], lambda x: (x == True).mean())
               for m in METHODS},
        )
        .reset_index()
    )

    x_a      = np.arange(len(grp_a))
    vals_a   = [grp_a[m["column"]].values * 100 for m in METHODS]
    off_a    = smart_offsets(vals_a)

    draw_lines(ax_a, x_a, vals_a, off_a, METHODS, trendline=False)
    if SHOW_N:
        draw_sample_counts(ax_a, x_a, grp_a["count"], Y_LIM_A[0])

    style_ax(ax_a,
             xtick_labels=SIZE_LABELS,
             xlabel="Number of Candidate Senses",
             ylabel="Accuracy (%)",
             title="(A) Accuracy vs. Candidate Set Size",
             y_lim=Y_LIM_A)

    # ── Panel B: Gold Sense Position ──────────────────────────────────────────
    df["norm_rank"] = (
        (df[GOLD_RANK_COL] - 1)
        / (df[CANDIDATE_COUNT_COL] - 1).clip(lower=1)
    ).clip(0, 1)

    rank_edges  = np.linspace(0, 1, N_RANK_BINS + 1)
    rank_labels = [
        f"{int(b*100)}–{int(rank_edges[i+1]*100)}%"
        for i, b in enumerate(rank_edges[:-1])
    ]
    df["rank_bin"] = pd.cut(
        df["norm_rank"], bins=rank_edges,
        labels=rank_labels, include_lowest=True
    )

    grp_b = (
        df.groupby("rank_bin", observed=True)
        .agg(
            count=("norm_rank", "count"),
            **{m["column"]: (m["column"], lambda x: (x == True).mean())
               for m in METHODS},
        )
        .reset_index()
    )

    x_b    = np.arange(len(grp_b))
    vals_b = [grp_b[m["column"]].values * 100 for m in METHODS]
    off_b  = smart_offsets(vals_b)

    draw_lines(ax_b, x_b, vals_b, off_b, METHODS, trendline=True)
    if SHOW_N:
        draw_sample_counts(ax_b, x_b, grp_b["count"], Y_LIM_B[0])

    # Pearson correlation annotation
    y_pos = 0.97
    for m in METHODS:
        if m["column"] in SHOW_CORRELATION:
            valid = df.dropna(subset=["norm_rank", m["column"]]).copy()
            valid["_int"] = valid[m["column"]].astype(int)
            r_val, p_val = pearsonr(valid["norm_rank"], valid["_int"])
            p_str = "$p < 0.0001$" if p_val < 0.0001 else f"$p = {p_val:.4f}$"
            ax_b.text(
                0.98, y_pos,
                f"{m['label']}: $r = {r_val:.3f}$, {p_str}",
                transform=ax_b.transAxes, ha="right", va="top",
                fontsize=9, color=m["color"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=m["color"], alpha=0.8),
            )
            y_pos -= 0.10

    style_ax(ax_b,
             xtick_labels=rank_labels,
             xlabel="Normalised Rank of Gold Sense  (0 = first, 1 = last)",
             ylabel="Accuracy (%)",
             title="(B) Accuracy vs. Gold Sense Position",
             y_lim=Y_LIM_B,
             xtick_fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PDF}, {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
