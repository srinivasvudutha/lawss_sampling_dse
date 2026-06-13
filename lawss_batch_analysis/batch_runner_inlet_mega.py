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
DEFAULT_OUTPUT:     str = "inlet_mega_sweep_results.csv"
DEFAULT_WORKERS:    int = 10

# Parameter Sweep Grids
I_U_GRID = np.linspace(0.05, 0.20, 16)
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
    
    # Inject variables and force module cache reload to re-evaluate constants
    _os.environ["LAWSS_I_U"] = str(i_u)
    _os.environ["LAWSS_T_INT"] = str(t_int)
    _sys.modules.pop('lawss_stage3_inlet_mega', None)

    try:
        from lawss_stage3_inlet_mega import (
            Environment,
            BATTERY_CAPACITY_S,
            DT_KIN,
        )
    except ImportError as exc:
        return {
            "I_U":                  i_u,
            "T_INT":                t_int,
            "seed":                 seed,
            "nodes_completed":      0,
            "sim_duration_s":       float("nan"),
            "total_survey_time_s":  float("nan"),
            "fleet_distance_m":     float("nan"),
            "conv_time_min_s":      float("nan"),
            "conv_time_max_s":      float("nan"),
            "conv_time_mean_s":     float("nan"),
            "wall_time_s":          0.0,
            "error":                f"Import failed: {exc}",
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

        tick = 0
        while tick < budget_ticks:
            env.step()
            tick += 1

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
        # confirmed measured (or battery ran out if not all measured).
        # This equals env.elapsed_s at loop exit and represents the total
        # mission duration needed to cover all sampled target points.
        total_survey_time_s = round(float(env.elapsed_s), 3) if env.all_measured else float("nan")

        return {
            "I_U":                  i_u,
            "T_INT":                t_int,
            "seed":                 seed,
            "nodes_completed":      int(env.n_measured),
            "sim_duration_s":       round(float(env.elapsed_s),  3),
            "total_survey_time_s":  total_survey_time_s,
            "fleet_distance_m":     round(float(fleet_dist_m),   2),
            "conv_time_min_s":      round(c_min, 2),
            "conv_time_max_s":      round(c_max, 2),
            "conv_time_mean_s":     round(c_mean, 2),
            "wall_time_s":          round(float(wall_elapsed),   2),
            "error":                None,
        }

    except Exception:
        wall_elapsed = _time.perf_counter() - wall_start
        return {
            "I_U":                  i_u,
            "T_INT":                t_int,
            "seed":                 seed,
            "nodes_completed":      0,
            "sim_duration_s":       float("nan"),
            "total_survey_time_s":  float("nan"),
            "fleet_distance_m":     float("nan"),
            "conv_time_min_s":      float("nan"),
            "conv_time_max_s":      float("nan"),
            "conv_time_mean_s":     float("nan"),
            "wall_time_s":          round(float(wall_elapsed), 2),
            "error":                traceback.format_exc(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Terminal output helpers
# ─────────────────────────────────────────────────────────────────────────────

_BAR_WIDTH = 86


def _hr(char: str = "─") -> str:
    return char * _BAR_WIDTH


def _print_header(total_trials: int, workers: int, output: str) -> None:
    print()
    print(_hr("═"))
    print("  LAWSS  ·  Inlet Mega-Scale Grid Search Harness")
    print(f"  Grid: I_U [{len(I_U_GRID)}] x T_INT [{len(T_INT_GRID)}]  -> {len(I_U_GRID)*len(T_INT_GRID)} combinations")
    print(_hr("─"))
    print(f"  Total Trials : {total_trials}")
    print(f"  Workers      : {workers}  (logical CPUs available: {os.cpu_count()})")
    print(f"  Output       : {output}")
    print(_hr("═"))
    print()


def _fmt_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _print_trial_result(
    idx: int,
    total: int,
    result: Dict,
) -> None:
    i_u   = result["I_U"]
    t_int = result["T_INT"]
    seed  = result["seed"]
    nodes = result["nodes_completed"]
    sim_t = result["sim_duration_s"]
    cmean = result["conv_time_mean_s"]
    wtick = result["wall_time_s"]
    err   = result["error"]

    w = len(str(total))
    if err is not None:
        tag    = "✗"
        detail = f"IU={i_u:.2f} T={t_int:<3} seed={seed:<5}  ERROR: {err.splitlines()[-1][:50]}"
    else:
        tag    = "✓"
        detail = (
            f"IU={i_u:.2f} T={t_int:<3} seed={seed:<5}  "
            f"nodes={nodes:>4}  "
            f"sim={_fmt_duration(sim_t):<10}  "
            f"mean_conv={cmean:>6.1f}s  "
            f"({wtick:.1f}s wall)"
        )
    print(f"  {tag}  [{idx:>{w}}/{total}]  {detail}", flush=True)


def _print_summary(df: pd.DataFrame, n_failed: int) -> None:
    ok   = df[df["error"].isna()].copy()
    n_ok = len(ok)

    metrics = [
        ("Nodes Completed",    "nodes_completed",     ".1f", "nodes"),
        ("Total Survey Time",  "total_survey_time_s", ".1f", "s"),
        ("Conv Time Min",      "conv_time_min_s",     ".1f", "s"),
        ("Conv Time Max",      "conv_time_max_s",     ".1f", "s"),
        ("Conv Time Mean",     "conv_time_mean_s",    ".1f", "s"),
        ("Sim Duration",       "sim_duration_s",      ".1f", "s"),
        ("Fleet Distance",     "fleet_distance_m",    ".1f", "m"),
        ("Wall Time / Trial",  "wall_time_s",         ".1f", "s"),
    ]

    label_w = 22
    val_w   = 10

    def _row(label: str, col: str, fmt: str, unit: str) -> str:
        series = ok[col].dropna() if n_ok > 0 and col in ok.columns else pd.Series([], dtype=float)
        if series.empty:
            vals = ["—"] * 5
        else:
            vals = [f"{v:{fmt}}" for v in [
                series.min(), series.max(), series.mean(),
                series.median(),
                series.std(ddof=1) if len(series) > 1 else 0.0,
            ]]
        cells = "  ".join(v.rjust(val_w) for v in vals)
        return f"  {label:<{label_w}}  {cells}  [{unit}]"

    hdr = "  ".join(h.rjust(val_w) for h in ["Min", "Max", "Mean", "Median", "Std Dev"])

    print()
    print(_hr("═"))
    print(f"  Summary  ·  Inlet Mega-Scale Grid Search")
    print(f"  Successful trials : {n_ok} / {n_ok + n_failed}")
    print(_hr("─"))
    print(f"  {'Metric':<{label_w}}  {hdr}")
    print(_hr("─"))
    for m in metrics:
        print(_row(*m))
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
              "see the 'error' column in the CSV for details.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main batch-runner logic
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

    results: List[Dict] = []
    n_done   = 0
    n_failed = 0
    wall_t0  = time.perf_counter()

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
                    "I_U":                  task[1],
                    "T_INT":                task[2],
                    "seed":                 task[0],
                    "nodes_completed":      0,
                    "sim_duration_s":       float("nan"),
                    "total_survey_time_s":  float("nan"),
                    "fleet_distance_m":     float("nan"),
                    "conv_time_min_s":      float("nan"),
                    "conv_time_max_s":      float("nan"),
                    "conv_time_mean_s":     float("nan"),
                    "wall_time_s":          float("nan"),
                    "error":                traceback.format_exc(),
                }

            if result["error"] is not None:
                n_failed += 1

            results.append(result)

            if not quiet:
                _print_trial_result(
                    idx    = n_done,
                    total  = total_trials,
                    result = result,
                )

    df = pd.DataFrame(results, columns=[
        "I_U",
        "T_INT",
        "seed",
        "nodes_completed",
        "sim_duration_s",
        "total_survey_time_s",
        "fleet_distance_m",
        "conv_time_min_s",
        "conv_time_max_s",
        "conv_time_mean_s",
        "wall_time_s",
        "error",
    ])
    df = df.sort_values(by=["I_U", "T_INT", "seed"]).reset_index(drop=True)

    try:
        df.to_csv(output, index=False, float_format="%.3f")
        if not quiet:
            print(f"  Results written → {os.path.abspath(output)}")
    except OSError as exc:
        print(f"  WARNING: could not write CSV: {exc}", file=sys.stderr)

    total_wall = time.perf_counter() - wall_t0
    if not quiet:
        _print_summary(df, n_failed)
        print(f"  Total wall time : {_fmt_duration(total_wall)}  ({total_wall:.1f} s)")
        sim_sum = df["sim_duration_s"].dropna().sum()
        if total_wall > 0:
            speedup = sim_sum / total_wall
            print(f"  Aggregate sim time : {sim_sum:.0f} s  "
                  f"·  effective speed-up : {speedup:.1f}×")
        print()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog            = "batch_runner_inlet_mega.py",
        description     = "Grid Search HPC harness for LAWSS Inlet Mega-Scale",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials", "-n",
        type    = int,
        default = DEFAULT_N_TRIALS,
        metavar = "N",
        help    = "Trials per parameter combination.",
    )
    parser.add_argument(
        "--seed-start", "-s",
        type    = int,
        default = DEFAULT_SEED_START,
        metavar = "SEED",
        help    = "Starting RNG seed.",
    )
    parser.add_argument(
        "--workers", "-w",
        type    = int,
        default = DEFAULT_WORKERS,
        metavar = "W",
        help    = "Number of parallel worker processes.",
    )
    parser.add_argument(
        "--output", "-o",
        type    = str,
        default = DEFAULT_OUTPUT,
        metavar = "FILE",
        help    = "Output CSV path.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action  = "store_true",
        help    = "Suppress per-trial output; print only the final summary.",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Print resolved configuration and exit without running.",
    )
    return parser


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