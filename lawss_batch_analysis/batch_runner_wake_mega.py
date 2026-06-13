from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Default CLI values
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_TRIALS:   int = 10
DEFAULT_SEED_START: int = 0
DEFAULT_OUTPUT:     str = "wake_mega_sweep_results.csv"
DEFAULT_WORKERS:    int = 10

# Parameter Sweep Grids
I_U_GRID = np.linspace(0.25, 0.40, 16)
T_INT_GRID = np.linspace(3.0, 7.0, 9)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess initialiser
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = "1"


# ─────────────────────────────────────────────────────────────────────────────
# Core worker function
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(args: tuple) -> Dict:
    seed, i_u, t_int = args
    import time as _time
    import numpy as _np

    import os as _os
    import sys as _sys

    _os.environ["LAWSS_I_U"] = str(i_u)
    _os.environ["LAWSS_T_INT"] = str(t_int)
    _sys.modules.pop('lawss_stage3_wake_mega', None)

    try:
        from lawss_stage3_wake_mega import (
            Environment,
            BATTERY_CAPACITY_S,
            DT_KIN,
            N_TARGETS,
        )
    except ImportError as exc:
        return {
            "I_U":               i_u,
            "T_INT":             t_int,
            "seed":              seed,
            "nodes_measured":    0,
            "n_targets":         0,
            "total_assignments": 0,
            "sim_elapsed_s":     float("nan"),
            "total_survey_time_s": float("nan"),
            "fleet_distance_m":  float("nan"),
            "conv_time_min_s":   float("nan"),
            "conv_time_max_s":   float("nan"),
            "conv_time_mean_s":  float("nan"),
            "wall_time_s":       0.0,
            "var_Ubar_min":      float("nan"),
            "var_Ubar_max":      float("nan"),
            "var_Ubar_mean":     float("nan"),
            "var_Ubar_median":   float("nan"),
            "var_Ubar_std":      float("nan"),
            "sigma_Ubar_mean":   float("nan"),
            "ci_rel_mean":       float("nan"),
            "ci_rel_max":        float("nan"),
            "error":             f"Import failed: {exc}",
        }

    wall_start = _time.perf_counter()

    try:
        env          = Environment(seed=seed)
        budget_ticks = int(BATTERY_CAPACITY_S / DT_KIN)

        n_drones     = len(env.drones)
        prev_pos     = _np.array(
            [d.position for d in env.drones], dtype=float
        )
        fleet_dist_m = 0.0

        for _ in range(budget_ticks):
            env.step()

            curr_pos = _np.array(
                [d.position for d in env.drones], dtype=float
            )
            fleet_dist_m += float(
                _np.sum(_np.linalg.norm(curr_pos - prev_pos, axis=1))
            )
            prev_pos[:] = curr_pos

            if env.all_measured:
                break

        wall_elapsed = _time.perf_counter() - wall_start

        conv_times = env.sampling_times
        if conv_times:
            c_min = float(_np.min(conv_times))
            c_max = float(_np.max(conv_times))
            c_mean = float(_np.mean(conv_times))
        else:
            c_min = c_max = c_mean = float("nan")

        # total_survey_time_s: sim-clock time at which the last node was
        # confirmed measured (or NaN if budget exhausted before completion).
        total_survey_time_s = round(float(env.elapsed_s), 3) if env.all_measured else float("nan")

        # GPR heteroscedastic noise variance summary across completed nodes.
        # var_Ubar = 2·σ²_u·T_int/T  [m²/s²] — the σ²_n(x_i) diagonal entries.
        ns = env.noise_variance_summary()
        vb_min    = round(ns.get("var_Ubar_min",    float("nan")), 6)
        vb_max    = round(ns.get("var_Ubar_max",    float("nan")), 6)
        vb_mean   = round(ns.get("var_Ubar_mean",   float("nan")), 6)
        vb_median = round(ns.get("var_Ubar_median", float("nan")), 6)
        vb_std    = round(ns.get("var_Ubar_std",    float("nan")), 6)
        sig_mean  = round(ns.get("sigma_Ubar_mean", float("nan")), 5)
        ci_mean   = round(ns.get("ci_rel_mean",     float("nan")), 4)
        ci_max    = round(ns.get("ci_rel_max",      float("nan")), 4)

        return {
            "I_U":               i_u,
            "T_INT":             t_int,
            "seed":              seed,
            "nodes_measured":    int(env.n_measured),
            "n_targets":         int(N_TARGETS),
            "total_assignments": int(env._assignments_made),
            "sim_elapsed_s":     round(float(env.elapsed_s),    3),
            "total_survey_time_s": total_survey_time_s,
            "fleet_distance_m":  round(float(fleet_dist_m),     2),
            "conv_time_min_s":   round(c_min, 2),
            "conv_time_max_s":   round(c_max, 2),
            "conv_time_mean_s":  round(c_mean, 2),
            "wall_time_s":       round(float(wall_elapsed),     2),
            # ── GPR noise variance columns ─────────────────────────────────────
            "var_Ubar_min":      vb_min,
            "var_Ubar_max":      vb_max,
            "var_Ubar_mean":     vb_mean,
            "var_Ubar_median":   vb_median,
            "var_Ubar_std":      vb_std,
            "sigma_Ubar_mean":   sig_mean,
            "ci_rel_mean":       ci_mean,
            "ci_rel_max":        ci_max,
            "error":             None,
        }

    except Exception:
        wall_elapsed = _time.perf_counter() - wall_start
        return {
            "I_U":               i_u,
            "T_INT":             t_int,
            "seed":              seed,
            "nodes_measured":    0,
            "n_targets":         int(N_TARGETS) if 'N_TARGETS' in dir() else 0,
            "total_assignments": 0,
            "sim_elapsed_s":     float("nan"),
            "total_survey_time_s": float("nan"),
            "fleet_distance_m":  float("nan"),
            "conv_time_min_s":   float("nan"),
            "conv_time_max_s":   float("nan"),
            "conv_time_mean_s":  float("nan"),
            "wall_time_s":       round(float(wall_elapsed), 2),
            "var_Ubar_min":      float("nan"),
            "var_Ubar_max":      float("nan"),
            "var_Ubar_mean":     float("nan"),
            "var_Ubar_median":   float("nan"),
            "var_Ubar_std":      float("nan"),
            "sigma_Ubar_mean":   float("nan"),
            "ci_rel_mean":       float("nan"),
            "ci_rel_max":        float("nan"),
            "error":             traceback.format_exc(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Terminal formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

_BAR_WIDTH = 86


def _hr(char: str = "─") -> str:
    return char * _BAR_WIDTH


def _fmt_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _print_header(total_trials: int, workers: int, output: str) -> None:
    print()
    print(_hr("═"))
    print("  LAWSS  ·  Mega-Scale Wake Grid Search Harness")
    print(f"  Grid: I_U [{len(I_U_GRID)}] x T_INT [{len(T_INT_GRID)}]  -> "
          f"{len(I_U_GRID) * len(T_INT_GRID)} combinations")
    print(_hr("─"))
    print(f"  Total Trials : {total_trials}")
    print(f"  Workers      : {workers}  (logical CPUs on this host: {os.cpu_count()})")
    print(f"  Output       : {output}")
    print(_hr("═"))
    print()


def _print_trial(idx: int, total: int, result: Dict) -> None:
    i_u       = result["I_U"]
    t_int     = result["T_INT"]
    seed      = result["seed"]
    nodes     = result["nodes_measured"]
    n_targets = result.get("n_targets", nodes)
    assn      = result["total_assignments"]
    sim_t     = result["sim_elapsed_s"]
    cmean     = result["conv_time_mean_s"]
    fdist     = result.get("fleet_distance_m", float("nan"))
    wall      = result["wall_time_s"]
    err       = result["error"]

    w = len(str(total))
    if err is not None:
        tag    = "✗"
        detail = f"IU={i_u:.2f} T={t_int:<3} seed={seed:<5}  ERROR: {err.splitlines()[-1][:50]}"
    else:
        tag    = "✓"
        dist_str = f"{fdist:.0f}m" if math.isfinite(fdist) else "—"
        detail = (
            f"IU={i_u:.2f} T={t_int:<3} seed={seed:<5}  "
            f"nodes={nodes:>3}/{n_targets}  "
            f"assigns={assn:>4}  "
            f"dist={dist_str:<8}  "
            f"sim={_fmt_duration(sim_t):<10}  "
            f"mean_conv={cmean:>6.1f}s  "
            f"wall={wall:.1f}s"
        )
    print(f"  {tag}  [{idx:>{w}}/{total}]  {detail}", flush=True)


def _print_summary(df: pd.DataFrame, n_failed: int, total_wall: float) -> None:
    ok   = df[df["error"].isna()].copy()
    n_ok = len(ok)

    metrics = [
        ("Nodes Measured",    "nodes_measured",      ".1f", "nodes"),
        ("Total Assignments", "total_assignments",   ".1f", ""),
        ("Total Survey Time", "total_survey_time_s", ".1f", "s"),
        ("Fleet Distance",    "fleet_distance_m",    ".1f", "m"),
        ("Conv Time Min",     "conv_time_min_s",     ".1f", "s"),
        ("Conv Time Max",     "conv_time_max_s",     ".1f", "s"),
        ("Conv Time Mean",    "conv_time_mean_s",    ".1f", "s"),
        ("Sim Duration",      "sim_elapsed_s",       ".1f", "s"),
        ("Wall Time / Trial", "wall_time_s",         ".1f", "s"),
        ("var_Ubar mean",     "var_Ubar_mean",       ".6f", "m2/s2"),
        ("var_Ubar max",      "var_Ubar_max",        ".6f", "m2/s2"),
        ("sigma_Ubar mean",   "sigma_Ubar_mean",     ".5f", "m/s"),
        ("CI_rel mean",       "ci_rel_mean",         ".4f", ""),
        ("CI_rel max",        "ci_rel_max",          ".4f", ""),
    ]

    label_w = 22
    val_w   = 10

    def _row(label: str, col: str, fmt: str, unit: str) -> str:
        s = ok[col].dropna() if col in ok.columns else pd.Series([], dtype=float)
        if s.empty:
            vals = ["—"] * 5
        else:
            vals = [f"{v:{fmt}}" for v in [
                s.min(), s.max(), s.mean(),
                s.median(),
                s.std(ddof=1) if len(s) > 1 else 0.0,
            ]]
        cells = "  ".join(v.rjust(val_w) for v in vals)
        unit_tag = f"  [{unit}]" if unit else ""
        return f"  {label:<{label_w}}  {cells}{unit_tag}"

    hdr = "  ".join(h.rjust(val_w) for h in ["Min", "Max", "Mean", "Median", "Std Dev"])

    print()
    print(_hr("═"))
    print(f"  SUMMARY — Mega-Scale Wake Grid Search  |  "
          f"{n_ok}/{n_ok + n_failed} trials succeeded")
    print(_hr("─"))
    print(f"  {'Metric':<{label_w}}  {hdr}")
    print(_hr("─"))
    for m in metrics:
        print(_row(*m))
    print(_hr("─"))
    print(_hr("═"))

    # Highlighted single-line callout for the headline metric
    survey_s = ok["total_survey_time_s"].dropna()
    if not survey_s.empty:
        n_complete      = len(survey_s)
        mean_s          = survey_s.mean()
        std_s           = survey_s.std(ddof=1) if len(survey_s) > 1 else 0.0
        completion_rate = 100.0 * n_complete / n_ok if n_ok else 0.0
        print(
            f"\n  ★  Mean time to sample all nodes : "
            f"{mean_s:.1f} s  (±{std_s:.1f} s)  "
            f"over {n_complete} completed trials  "
            f"[{completion_rate:.0f}% completion rate]"
        )
    else:
        print("\n  ★  Mean time to sample all nodes : — (no trials completed all nodes)")

    if n_failed:
        print(f"\n  ⚠  {n_failed} trial(s) failed — "
              "see 'error' column in the CSV for details.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    n_trials:   int  = DEFAULT_N_TRIALS,
    seed_start: int  = DEFAULT_SEED_START,
    workers:    int  = DEFAULT_WORKERS,
    output:     str  = DEFAULT_OUTPUT,
    quiet:      bool = False,
) -> pd.DataFrame:
    
    tasks = []
    for i_u, t_int in itertools.product(I_U_GRID, T_INT_GRID):
        for s in range(seed_start, seed_start + n_trials):
            tasks.append((s, i_u, t_int))
            
    total_trials = len(tasks)

    if not quiet:
        _print_header(total_trials, workers, output)

    results:  List[Dict] = []
    n_done    = 0
    n_failed  = 0
    wall_t0   = time.perf_counter()

    effective_workers = min(workers, total_trials)

    with ProcessPoolExecutor(
        max_workers = effective_workers,
        initializer = _worker_init,
    ) as pool:

        future_to_args: Dict[Future, tuple] = {
            pool.submit(_run_trial, task): task
            for task in tasks
        }

        for future in as_completed(future_to_args):
            n_done += 1
            try:
                result = future.result()
            except Exception:
                task = future_to_args[future]
                result = {
                    "I_U":               task[1],
                    "T_INT":             task[2],
                    "seed":              task[0],
                    "nodes_measured":    0,
                    "n_targets":         0,
                    "total_assignments": 0,
                    "sim_elapsed_s":     float("nan"),
                    "total_survey_time_s": float("nan"),
                    "conv_time_min_s":   float("nan"),
                    "conv_time_max_s":   float("nan"),
                    "conv_time_mean_s":  float("nan"),
                    "wall_time_s":       float("nan"),
                    "var_Ubar_min":      float("nan"),
                    "var_Ubar_max":      float("nan"),
                    "var_Ubar_mean":     float("nan"),
                    "var_Ubar_median":   float("nan"),
                    "var_Ubar_std":      float("nan"),
                    "sigma_Ubar_mean":   float("nan"),
                    "ci_rel_mean":       float("nan"),
                    "ci_rel_max":        float("nan"),
                    "error":             traceback.format_exc(),
                }

            if result["error"] is not None:
                n_failed += 1

            results.append(result)

            if not quiet:
                _print_trial(n_done, total_trials, result)

    df = pd.DataFrame(results, columns=[
        "I_U",
        "T_INT",
        "seed",
        "nodes_measured",
        "n_targets",
        "total_assignments",
        "sim_elapsed_s",
        "total_survey_time_s",
        "fleet_distance_m",
        "conv_time_min_s",
        "conv_time_max_s",
        "conv_time_mean_s",
        "wall_time_s",
        "var_Ubar_min",
        "var_Ubar_max",
        "var_Ubar_mean",
        "var_Ubar_median",
        "var_Ubar_std",
        "sigma_Ubar_mean",
        "ci_rel_mean",
        "ci_rel_max",
        "error",
    ])
    df = df.sort_values(by=["I_U", "T_INT", "seed"]).reset_index(drop=True)

    try:
        df.to_csv(output, index=False, float_format="%.6g")
        if not quiet:
            print(f"\n  Results written → {os.path.abspath(output)}")
    except OSError as exc:
        print(f"  WARNING: could not write CSV: {exc}", file=sys.stderr)

    total_wall = time.perf_counter() - wall_t0
    if not quiet:
        _print_summary(df, n_failed, total_wall)
        sim_sum = df["sim_elapsed_s"].dropna().sum()
        if total_wall > 0:
            speedup = sim_sum / total_wall
            print(f"  Total wall time      : {_fmt_duration(total_wall)}  ({total_wall:.1f} s)")
            print(f"  Aggregate sim time   : {sim_sum:.0f} s  "
                  f"·  effective speed-up : {speedup:.1f}×")
        print()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog            = "batch_runner_wake_mega.py",
        description     = "Grid Search HPC harness for LAWSS Wake Mega-Scale",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-trials", "-n",
        type    = int,
        default = DEFAULT_N_TRIALS,
        metavar = "N",
        help    = "Number of trials per parameter combination.",
    )
    p.add_argument(
        "--seed-start", "-s",
        type    = int,
        default = DEFAULT_SEED_START,
        metavar = "SEED",
        help    = "First RNG seed.",
    )
    p.add_argument(
        "--workers", "-w",
        type    = int,
        default = DEFAULT_WORKERS,
        metavar = "W",
        help    = "Parallel worker processes.",
    )
    p.add_argument(
        "--output", "-o",
        type    = str,
        default = DEFAULT_OUTPUT,
        metavar = "FILE",
        help    = "Output CSV path.",
    )
    p.add_argument(
        "--quiet", "-q",
        action  = "store_true",
        help    = "Suppress per-trial lines.",
    )
    p.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Print resolved configuration and exit.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.dry_run:
        print("  Dry-run — no trials executed.")
        return 0

    run_batch(
        n_trials   = args.n_trials,
        seed_start = args.seed_start,
        workers    = args.workers,
        output     = args.output,
        quiet      = args.quiet,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())