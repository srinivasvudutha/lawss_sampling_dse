import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# CONFIG
# ============================================================================

CSV_FILENAME = "wake_mega_sweep_results.csv"

NODES_COL = "nodes_measured"
TIME_COL = "conv_time_mean_s"

GROUP_X = "I_U"
GROUP_Y = "T_INT"


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, CSV_FILENAME)

OUTPUT_DIR = os.path.join(BASE_DIR, "heatmaps")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# LOAD CSV
# ============================================================================

def load_data():

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"\nCSV not found:\n{CSV_PATH}"
        )

    print(f"\nLoading:\n{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    required = [
        GROUP_X,
        GROUP_Y,
        NODES_COL,
        TIME_COL,
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"\nMissing columns:\n{missing}\n\n"
            f"Found:\n{list(df.columns)}"
        )

    print(f"Rows loaded: {len(df)}")

    return df


# ============================================================================
# BUILD PIVOTS
# ============================================================================

def build_heatmap_data(df):

    grouped = (
        df.groupby([GROUP_Y, GROUP_X])
        .agg(
            mean_nodes=(NODES_COL, "mean"),
            mean_conv=(TIME_COL, "mean"),
        )
        .reset_index()
    )

    nodes = grouped.pivot(
        index=GROUP_Y,
        columns=GROUP_X,
        values="mean_nodes"
    )

    conv = grouped.pivot(
        index=GROUP_Y,
        columns=GROUP_X,
        values="mean_conv"
    )

    nodes = nodes.sort_index(ascending=False)
    conv = conv.sort_index(ascending=False)

    return nodes, conv


# ============================================================================
# DRAW
# ============================================================================

def save_heatmap(
    pivot,
    title,
    colorbar,
    filename,
    cmap,
    decimals,
):

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    image = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=cmap,
        origin="upper",
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
    )

    cbar.set_label(
        colorbar
    )

    ax.set_xticks(
        range(len(pivot.columns))
    )

    ax.set_xticklabels(
        [f"{x:.3g}" for x in pivot.columns],
        rotation=45,
    )

    ax.set_yticks(
        range(len(pivot.index))
    )

    ax.set_yticklabels(
        [f"{y:.3g}" for y in pivot.index]
    )

    ax.set_xlabel("I_U")
    ax.set_ylabel("T_INT")
    ax.set_title(title)

    values = pivot.values

    low = np.nanmin(values)
    high = np.nanmax(values)

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):

            val = values[r, c]

            if np.isnan(val):
                text = "-"
            else:
                text = f"{val:.{decimals}f}"

            if np.isnan(val):
                color = "black"
            else:
                norm = (val - low) / (high - low + 1e-9)
                color = "white" if norm < 0.4 else "black"

            ax.text(
                c,
                r,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    fig.tight_layout()

    out = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved → {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    df = load_data()

    nodes, conv = build_heatmap_data(df)

    save_heatmap(
        nodes,
        title="Mean Nodes Sampled",
        colorbar="Nodes",
        filename="heatmap_nodes.png",
        cmap="plasma",
        decimals=1,
    )

    save_heatmap(
        conv,
        title="Mean Convergence Time",
        colorbar="Seconds",
        filename="heatmap_conv_time.png",
        cmap="viridis",
        decimals=2,
    )

    print("\nDone.")
    print(f"Output folder:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()