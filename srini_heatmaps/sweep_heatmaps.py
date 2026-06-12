from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap specifications
# ─────────────────────────────────────────────────────────────────────────────

HEATMAP_SPECS = [
    {
        "column":   "total_survey_time_s",
        "title":    "Mean Total Simulation Time",
        "label":    "Mean Total Simulation Time (s)",
        "filename": "heatmap_mean_total_sim_time",
        "cmap":     "viridis",
        "fmt":      ".1f",
    },
    {
        "column":   "conv_time_max_s",
        "title":    "Mean Maximum Convergence Time",
        "label":    "Mean Max Convergence Time (s)",
        "filename": "heatmap_mean_max_conv_time",
        "cmap":     "plasma",
        "fmt":      ".1f",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Core plotting
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Aggregate *column* by mean over all seeds, then pivot to a
    (I_U × T_INT) matrix suitable for sns.heatmap.

    Rows    → I_U   (turbulence intensity),  ascending top→bottom
    Columns → T_INT (integral time scale),   ascending left→right
    """
    agg = (
        df.groupby(["I_U", "T_INT"])[column]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot(index="I_U", columns="T_INT", values=column)
    # Sort axes so the heatmap reads naturally
    pivot = pivot.sort_index(ascending=False)          # I_U high → low top→bottom
    pivot = pivot.sort_index(axis=1, ascending=True)   # T_INT low → high left→right
    return pivot


def _make_heatmap(
    pivot: pd.DataFrame,
    title: str,
    cbar_label: str,
    cmap: str,
    fmt: str,
    out_path: Path,
    trial_counts: Optional[pd.DataFrame] = None,
) -> None:
    """Render and save a single annotated heatmap."""

    fig, ax = plt.subplots(figsize=(11, 7))

    # ── seaborn heatmap ──────────────────────────────────────────────────────
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.85},
    )

    # ── axes labels & ticks ──────────────────────────────────────────────────
    ax.set_xlabel("Integral Time Scale  $T_{int}$ (s)", fontsize=12, labelpad=8)
    ax.set_ylabel("Turbulence Intensity  $I_u$", fontsize=12, labelpad=8)

    # Format tick labels
    x_labels = [f"{float(t):.1f}" for t in pivot.columns]
    y_labels = [f"{float(i):.2f}" for i in pivot.index]
    ax.set_xticklabels(x_labels, rotation=0, fontsize=9)
    ax.set_yticklabels(y_labels, rotation=0, fontsize=9)

    # ── title ────────────────────────────────────────────────────────────────
    n_trials_str = ""
    if trial_counts is not None and not trial_counts.empty:
        counts = trial_counts.values.flatten()
        counts = counts[~np.isnan(counts.astype(float))]
        if len(counts) > 0:
            mn, mx = int(counts.min()), int(counts.max())
            n_trials_str = (
                f"  [n = {mn} trial{'s' if mn != 1 else ''} per cell]"
                if mn == mx
                else f"  [n = {mn}–{mx} trials per cell]"
            )

    ax.set_title(
        f"{title}{n_trials_str}",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Trial-count pivot (shown in title)
# ─────────────────────────────────────────────────────────────────────────────

def _trial_count_pivot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = (
        df.groupby(["I_U", "T_INT"])[column]
        .count()
        .reset_index()
        .rename(columns={column: "n"})
    )
    pivot = counts.pivot(index="I_U", columns="T_INT", values="n")
    pivot = pivot.sort_index(ascending=False).sort_index(axis=1, ascending=True)
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_csv(directory: str, csv_name: Optional[str]) -> Path:
    """
    Locate the CSV to load.

    If *csv_name* is given, look for that file inside *directory*.
    Otherwise, look for exactly one .csv file in *directory* and use it
    automatically; error if there are zero or more than one.
    """
    d = Path(directory)
    if not d.is_dir():
        sys.exit(f"ERROR: directory not found: {d}")

    if csv_name:
        p = d / csv_name
        if not p.exists():
            sys.exit(f"ERROR: CSV file not found: {p}")
        return p

    # Auto-detect
    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        sys.exit(f"ERROR: no .csv files found in {d}")
    if len(csvs) > 1:
        names = "\n    ".join(str(c.name) for c in csvs)
        sys.exit(
            f"ERROR: multiple CSV files found in {d} — specify one with --csv:\n    {names}"
        )
    print(f"  Auto-detected CSV: {csvs[0].name}")
    return csvs[0]


def generate_heatmaps(
    directory: str,
    csv_name: Optional[str] = None,
    out_dir: Optional[str] = None,
    fmt: str = "png",
) -> None:
    csv_path = _resolve_csv(directory, csv_name)

    # Default: save figures alongside the CSV
    out_dir = Path(out_dir) if out_dir else csv_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    # ── basic validation ─────────────────────────────────────────────────────
    required = {"I_U", "T_INT"} | {spec["column"] for spec in HEATMAP_SPECS}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing required columns: {missing}")

    # Drop rows flagged as errors (non-null 'error' column if present)
    if "error" in df.columns:
        n_before = len(df)
        df = df[df["error"].isna()].copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"  ⚠  Dropped {n_dropped} error row(s) before plotting.")

    print(
        f"  Grid detected: {df['I_U'].nunique()} I_U values × "
        f"{df['T_INT'].nunique()} T_INT values  "
        f"({len(df)} total rows after filtering)\n"
    )

    # ── render each heatmap ──────────────────────────────────────────────────
    for spec in HEATMAP_SPECS:
        col = spec["column"]

        # Skip if the column is entirely NaN
        if df[col].isna().all():
            print(f"  ⚠  Skipping '{col}' — all values are NaN.")
            continue

        pivot = _pivot_mean(df, col)
        counts = _trial_count_pivot(df, col)

        out_path = out_dir / f"{spec['filename']}.{fmt}"

        _make_heatmap(
            pivot=pivot,
            title=spec["title"],
            cbar_label=spec["label"],
            cmap=spec["cmap"],
            fmt=spec["fmt"],
            out_path=out_path,
            trial_counts=counts,
        )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sweep_heatmaps.py",
        description=(
            "Generate I_U × T_INT heatmaps from a LAWSS mega-sweep CSV.\n"
            "Produces:\n"
            "  • Mean total simulation time per (I_U, T_INT) cell\n"
            "  • Mean maximum convergence time per (I_U, T_INT) cell\n\n"
            "Examples:\n"
            "  # directory contains one CSV — auto-detected:\n"
            "  python sweep_heatmaps.py srini_heatmaps/\n\n"
            "  # directory contains multiple CSVs — name the one you want:\n"
            "  python sweep_heatmaps.py srini_heatmaps/ --csv inlet_mega_sweep_results.csv\n\n"
            "  # save figures somewhere else:\n"
            "  python sweep_heatmaps.py srini_heatmaps/ --out-dir figures/ --fmt pdf"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "directory",
        metavar="DIR",
        help="Directory that contains the sweep results CSV.",
    )
    parser.add_argument(
        "--csv", "-c",
        default=None,
        metavar="FILENAME",
        help=(
            "Name of the CSV file inside DIR (e.g. inlet_mega_sweep_results.csv). "
            "If omitted and DIR contains exactly one .csv, that file is used automatically."
        ),
    )
    parser.add_argument(
        "--out-dir", "-o",
        default=None,
        metavar="OUTDIR",
        help="Directory to write the output figures (created if absent). Default: same as DIR.",
    )
    parser.add_argument(
        "--fmt", "-f",
        default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Output figure format. Default: png.",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    generate_heatmaps(
        directory=args.directory,
        csv_name=args.csv,
        out_dir=args.out_dir,
        fmt=args.fmt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())