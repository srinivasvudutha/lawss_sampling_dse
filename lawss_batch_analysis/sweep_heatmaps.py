"""
sweep_heatmaps.py

Reads inlet_mega_sweep_results.csv and wake_mega_sweep_results.csv,
groups each by (I_U, T_INT), averages the nodes-sampled and mean
convergence time over the 10 trials per group, then produces two
heatmaps per dataset (4 figures total).

Adjust the INPUT_DIR / OUTPUT_DIR paths at the top if needed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_DIR  = "."          # folder containing the two CSV files
OUTPUT_DIR = "."          # folder where the four PNGs will be saved

INLET_CSV = os.path.join(INPUT_DIR, "inlet_mega_sweep_results.csv")
WAKE_CSV  = os.path.join(INPUT_DIR, "wake_mega_sweep_results.csv")

# ── Helper ─────────────────────────────────────────────────────────────────────

def build_pivot_tables(csv_path, nodes_col):
    """
    Load a sweep CSV and return two pivot tables:
        pivot_nodes  – mean nodes sampled   (rows = T_INT, cols = I_U)
        pivot_conv   – mean conv_time_mean_s (rows = T_INT, cols = I_U)
    """
    df = pd.read_csv(csv_path)

    grouped = df.groupby(["I_U", "T_INT"]).agg(
        avg_nodes    = (nodes_col,          "mean"),
        avg_conv_s   = ("conv_time_mean_s", "mean"),
    ).reset_index()

    pivot_nodes = grouped.pivot(index="T_INT", columns="I_U", values="avg_nodes")
    pivot_conv  = grouped.pivot(index="T_INT", columns="I_U", values="avg_conv_s")

    # Sort axes so they read low→high on the plots
    pivot_nodes = pivot_nodes.sort_index(ascending=False)   # T_INT top = 7 s
    pivot_conv  = pivot_conv.sort_index(ascending=False)

    return pivot_nodes, pivot_conv


def plot_heatmap(pivot, title, cbar_label, filename, fmt=".1f", cmap="viridis"):
    """
    Draw a single annotated heatmap and save it to OUTPUT_DIR.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="upper")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=12)

    # Axis ticks
    col_labels = [f"{v:.3g}" for v in pivot.columns]   # I_U values
    row_labels = [f"{v:.3g}" for v in pivot.index]     # T_INT values (high→low)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    ax.set_xlabel("Turbulence Intensity  I_U  (–)", fontsize=12)
    ax.set_ylabel("Integral Time Scale  T_INT  (s)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=14)

    # Cell annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                txt = "nan"
            else:
                txt = f"{val:{fmt}}"
            # pick contrasting text colour
            norm_val = (val - np.nanmin(pivot.values)) / (
                np.nanmax(pivot.values) - np.nanmin(pivot.values) + 1e-12
            )
            text_color = "white" if norm_val < 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=text_color)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


# ── Inlet ──────────────────────────────────────────────────────────────────────

print("Processing inlet …")
pivot_inlet_nodes, pivot_inlet_conv = build_pivot_tables(INLET_CSV, "nodes_completed")

fig_inlet_nodes = plot_heatmap(
    pivot_inlet_nodes,
    title      = "Inlet sweep – Mean nodes sampled per (I_U, T_INT)",
    cbar_label = "Mean nodes sampled",
    filename   = "inlet_heatmap_nodes.png",
    fmt        = ".1f",
    cmap       = "plasma",
)

fig_inlet_conv = plot_heatmap(
    pivot_inlet_conv,
    title      = "Inlet sweep – Mean convergence time per (I_U, T_INT)",
    cbar_label = "Mean convergence time (s)",
    filename   = "inlet_heatmap_conv_time.png",
    fmt        = ".2f",
    cmap       = "viridis",
)

# ── Wake ───────────────────────────────────────────────────────────────────────

print("Processing wake …")
pivot_wake_nodes, pivot_wake_conv = build_pivot_tables(WAKE_CSV, "nodes_measured")

fig_wake_nodes = plot_heatmap(
    pivot_wake_nodes,
    title      = "Wake sweep – Mean nodes sampled per (I_U, T_INT)",
    cbar_label = "Mean nodes sampled",
    filename   = "wake_heatmap_nodes.png",
    fmt        = ".1f",
    cmap       = "plasma",
)

fig_wake_conv = plot_heatmap(
    pivot_wake_conv,
    title      = "Wake sweep – Mean convergence time per (I_U, T_INT)",
    cbar_label = "Mean convergence time (s)",
    filename   = "wake_heatmap_conv_time.png",
    fmt        = ".2f",
    cmap       = "viridis",
)

print("Done. All four heatmaps saved.")
plt.show()