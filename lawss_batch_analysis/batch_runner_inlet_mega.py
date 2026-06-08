from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Default CLI values
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_TRIALS:   int = 100
DEFAULT_SEED_START: int = 0
DEFAULT_OUTPUT:     str = "inlet_mega_batch_results.csv"
# Leave one core free for the OS; each worker builds 20 MPC solvers so
# memory pressure rises quickly — users on tight RAM may lower this.
DEFAULT_WORKERS:    int = max(1, (os.cpu_count() or 2) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess initialiser  — single-threaded BLAS/OpenMP pinning
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init() -> None:
    """
    Run once in each worker process before any trial begins.

    Explicitly pins BLAS, OpenMP, and MKL thread counts to 1 so that
    N worker processes do not each expand into BLAS_THREADS threads,
    which would severely oversubscribe the CPU on HPC nodes.

    Must be applied BEFORE any NumPy / CasADi import in the worker.
    """
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
# Core worker function  (module-level — required for pickling with spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(seed: int) -> Dict:
    """
    Execute one complete LAWSS Inlet Mega-Scale simulation trial.
    Motion and arrival behaviour come from lawss_stage3_inlet_mega's
    dense-cost, proximity-braking CasADi MPC backend.

    All imports are local to this function so each subprocess loads its own
    independent copy of lawss_stage3_inlet_mega, keeping CasADi's global
    solver state isolated between workers.

    Parameters
    ----------
    seed : int
        RNG seed passed to Environment(seed=seed).

    Returns
    -------
    dict
        Keys: seed, nodes_completed, sim_duration_s, fleet_distance_m,
              wall_time_s, error.
        ``error`` is None on success; a string traceback on failure.
        Metric fields are NaN on failure so the DataFrame stays typed.
    """
    import time as _time
    import numpy as _np

    # Isolated import — each subprocess gets its own module instance.
    # This prevents any shared global state (CasADi IPOPT warm-start,
    # NumPy RNG state, etc.) between parallel workers.
    try:
        from lawss_stage3_inlet_mega import (
            Environment,
            BATTERY_CAPACITY_S,
            DT,
        )
    except ImportError as exc:
        return {
            "seed":             seed,
            "nodes_completed":  0,
            "sim_duration_s":   float("nan"),
            "fleet_distance_m": float("nan"),
            "wall_time_s":      0.0,
            "error":            f"Import failed: {exc}",
        }

    wall_start = _time.perf_counter()

    try:
        env          = Environment(seed=seed)
        budget_ticks = int(BATTERY_CAPACITY_S / DT)

        # Fleet odometry: pre-snapshot positions so the first step's
        # displacement is captured correctly.
        n_drones     = len(env.drones)
        prev_pos     = _np.array(
            [d.position for d in env.drones], dtype=float
        )                                    # (N_DRONES, 3)
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
            prev_pos[:] = curr_pos   # reuse buffer

            if env.all_measured:
                break

        wall_elapsed = _time.perf_counter() - wall_start

        return {
            "seed":             seed,
            "nodes_completed":  int(env.n_measured),
            "sim_duration_s":   round(float(env.elapsed_s),  3),
            "fleet_distance_m": round(float(fleet_dist_m),   2),
            "wall_time_s":      round(float(wall_elapsed),   2),
            "error":            None,
        }

    except Exception:  # noqa: BLE001 — intentional catch-all in workers
        wall_elapsed = _time.perf_counter() - wall_start
        return {
            "seed":             seed,
            "nodes_completed":  0,
            "sim_duration_s":   float("nan"),
            "fleet_distance_m": float("nan"),
            "wall_time_s":      round(float(wall_elapsed), 2),
            "error":            traceback.format_exc(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Terminal output helpers
# ─────────────────────────────────────────────────────────────────────────────

_BAR_WIDTH = 76


def _hr(char: str = "─") -> str:
    return char * _BAR_WIDTH


def _print_header(n_trials: int, workers: int, seed_start: int,
                  output: str) -> None:
    print()
    print(_hr("═"))
    print("  LAWSS  ·  Inlet Mega-Scale Batch Harness  (20 Drones, 250 Nodes)")
    print("  Backend: lawss_stage3_inlet_mega.py  "
          "(dense-cost proximity-braking CasADi MPC)")
    print(_hr("─"))
    print(f"  Trials    : {n_trials}")
    print(f"  Seeds     : {seed_start} … {seed_start + n_trials - 1}")
    print(f"  Workers   : {workers}  "
          f"(logical CPUs available: {os.cpu_count()})")
    print(f"  Output    : {output}")
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
    elapsed_wall: float,
) -> None:
    seed  = result["seed"]
    nodes = result["nodes_completed"]
    sim_t = result["sim_duration_s"]
    dist  = result["fleet_distance_m"]
    wtick = result["wall_time_s"]
    err   = result["error"]

    w = len(str(total))
    if err is not None:
        tag    = "✗"
        detail = f"seed={seed:<5}  ERROR: {err.splitlines()[-1][:60]}"
    else:
        tag    = "✓"
        detail = (
            f"seed={seed:<5}  "
            f"nodes={nodes:>4}  "
            f"sim={_fmt_duration(sim_t):<10}  "
            f"dist={dist:>10.1f} m  "
            f"({wtick:.1f}s wall)"
        )
    print(f"  {tag}  [{idx:>{w}}/{total}]  {detail}", flush=True)


def _print_summary(df: pd.DataFrame, n_failed: int) -> None:
    ok   = df[df["error"].isna()].copy()
    n_ok = len(ok)

    metrics = [
        ("Nodes Completed",   "nodes_completed",  ".1f", "nodes"),
        ("Sim Duration",      "sim_duration_s",   ".1f", "s"),
        ("Fleet Distance",    "fleet_distance_m", ".1f", "m"),
        ("Wall Time / Trial", "wall_time_s",      ".1f", "s"),
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
    print(f"  Summary  ·  Inlet Mega-Scale (20 Drones / 250 Nodes)")
    print(f"  Successful trials : {n_ok} / {n_ok + n_failed}")
    print(_hr("─"))
    print(f"  {'Metric':<{label_w}}  {hdr}")
    print(_hr("─"))
    for m in metrics:
        print(_row(*m))
    print(_hr("═"))

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
    """
    Run ``n_trials`` inlet mega-scale trials in parallel and return a DataFrame.

    Parameters
    ----------
    n_trials : int
        Number of independent trials to run.
    seed_start : int
        Seeds are ``range(seed_start, seed_start + n_trials)``.
    workers : int
        Number of parallel worker processes.
    output : str
        Path to the output CSV file.
    quiet : bool
        Suppress per-trial progress lines; only show the final summary.

    Returns
    -------
    pd.DataFrame
        Columns: seed, nodes_completed, sim_duration_s, fleet_distance_m,
                 wall_time_s, error.
    """
    seeds = list(range(seed_start, seed_start + n_trials))

    if not quiet:
        _print_header(n_trials, workers, seed_start, output)

    results: List[Dict] = []
    n_done   = 0
    n_failed = 0
    wall_t0  = time.perf_counter()

    effective_workers = min(workers, n_trials)

    with ProcessPoolExecutor(
        max_workers = effective_workers,
        initializer = _worker_init,
    ) as pool:

        future_to_seed: Dict[Future, int] = {
            pool.submit(_run_trial, s): s
            for s in seeds
        }

        for future in as_completed(future_to_seed):
            n_done += 1
            try:
                result = future.result()
            except Exception:  # noqa: BLE001
                seed = future_to_seed[future]
                result = {
                    "seed":             seed,
                    "nodes_completed":  0,
                    "sim_duration_s":   float("nan"),
                    "fleet_distance_m": float("nan"),
                    "wall_time_s":      float("nan"),
                    "error":            traceback.format_exc(),
                }

            if result["error"] is not None:
                n_failed += 1

            results.append(result)

            if not quiet:
                _print_trial_result(
                    idx          = n_done,
                    total        = n_trials,
                    result       = result,
                    elapsed_wall = time.perf_counter() - wall_t0,
                )

    df = pd.DataFrame(results, columns=[
        "seed",
        "nodes_completed",
        "sim_duration_s",
        "fleet_distance_m",
        "wall_time_s",
        "error",
    ])
    df = df.sort_values("seed").reset_index(drop=True)

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
        description     = (
            "Headless parallel batch harness for LAWSS Inlet Mega-Scale "
            "(20 Drones, 250 Nodes) using the updated dense-cost "
            "proximity-braking MPC backend."
        ),
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials", "-n",
        type    = int,
        default = DEFAULT_N_TRIALS,
        metavar = "N",
        help    = "Total number of simulation trials to run.",
    )
    parser.add_argument(
        "--seed-start", "-s",
        type    = int,
        default = DEFAULT_SEED_START,
        metavar = "SEED",
        help    = "Starting RNG seed.  Seeds are range(seed_start, seed_start+N).",
    )
    parser.add_argument(
        "--workers", "-w",
        type    = int,
        default = DEFAULT_WORKERS,
        metavar = "W",
        help    = (
            "Number of parallel worker processes.  "
            f"Defaults to cpu_count-1 ({DEFAULT_WORKERS} on this machine).  "
            "Each worker builds 20 MPC solvers — lower this on memory-constrained nodes."
        ),
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
        help    = "Print resolved configuration and exit without running any trials.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.n_trials < 1:
        parser.error("--n-trials must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.seed_start < 0:
        parser.error("--seed-start must be >= 0")

    if args.dry_run:
        print()
        print(_hr("═"))
        print("  Dry-run — resolved configuration (no trials will be executed)")
        print(_hr("─"))
        print("  backend    : lawss_stage3_inlet_mega.py "
              "(dense-cost proximity-braking MPC)")
        print(f"  n_trials   : {args.n_trials}")
        print(f"  seed_start : {args.seed_start}")
        print(f"  seeds      : {args.seed_start} … "
              f"{args.seed_start + args.n_trials - 1}")
        print(f"  workers    : {args.workers}")
        print(f"  output     : {os.path.abspath(args.output)}")
        print(_hr("═"))
        print()
        return 0

    run_batch(
        n_trials   = args.n_trials,
        seed_start = args.seed_start,
        workers    = args.workers,
        output     = args.output,
        quiet      = args.quiet,
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Guard — essential for ProcessPoolExecutor on Windows / macOS (spawn context)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())
