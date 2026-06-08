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
DEFAULT_OUTPUT:     str = "wake_mega_batch_results.csv"
# On a cluster node, leave 1 core for the OS / job scheduler.
DEFAULT_WORKERS:    int = max(1, (os.cpu_count() or 2) - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess initialiser  ── MANDATORY for HPC cluster stability ──
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init() -> None:
    """
    Pin all BLAS / OpenMP thread pools to a single thread per worker process.

    Each trial already runs in its own OS process; intra-process thread pools
    add no useful parallelism and would oversubscribe cluster nodes, causing
    CPU thrashing, memory contention, and potential OOM kills.

    This must be called BEFORE any NumPy, CasADi, or SciPy import inside
    the worker — hence its placement in the ProcessPoolExecutor initializer
    rather than at module import time.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "GOTO_NUM_THREADS",
    ):
        os.environ[var] = "1"


# ─────────────────────────────────────────────────────────────────────────────
# Core worker function  ── module-level so ProcessPoolExecutor can pickle it ──
# ─────────────────────────────────────────────────────────────────────────────

def _run_trial(seed: int) -> Dict:
    """
    Execute one complete LAWSS mega-scale wake simulation trial.
    Motion and arrival behaviour come from lawss_stage3_wake_mega's
    dense-cost, proximity-braking CasADi MPC backend.

    All imports are LOCAL to this function.  This is intentional and critical:
    CasADi maintains global solver state (JIT caches, plugin registries).
    Importing the backend module at the top level of the worker process and
    sharing it across trials would corrupt that state.  Each call to
    _run_trial() imports a fresh copy in its own address space.

    Parameters
    ----------
    seed : int
        RNG seed forwarded to Environment(seed=seed).

    Returns
    -------
    dict with keys:
        seed              — the RNG seed used
        nodes_measured    — number of targets successfully sampled
        total_assignments — total dispatcher assignments issued
        sim_elapsed_s     — simulated clock time at termination  [s]
        wall_time_s       — real wall-clock time for this trial  [s]
        error             — None on success; traceback string on failure
    Metric fields are NaN on failure to keep the DataFrame uniformly typed.
    """
    import time as _time
    import numpy as _np

    # ── Isolated import of the mega-scale backend ─────────────────────────────
    try:
        from lawss_stage3_wake_mega import (
            Environment,
            BATTERY_CAPACITY_S,
            DT,
        )
    except ImportError as exc:
        return {
            "seed":             seed,
            "nodes_measured":   0,
            "total_assignments": 0,
            "sim_elapsed_s":    float("nan"),
            "wall_time_s":      0.0,
            "error":            f"Import failed: {exc}",
        }

    wall_start = _time.perf_counter()

    try:
        env          = Environment(seed=seed)
        budget_ticks = int(BATTERY_CAPACITY_S / DT)

        for _ in range(budget_ticks):
            env.step()
            if env.all_measured:
                break

        wall_elapsed = _time.perf_counter() - wall_start

        return {
            "seed":              seed,
            "nodes_measured":    int(env.n_measured),
            "total_assignments": int(env._assignments_made),
            "sim_elapsed_s":     round(float(env.elapsed_s),    3),
            "wall_time_s":       round(float(wall_elapsed),     2),
            "error":             None,
        }

    except Exception:
        wall_elapsed = _time.perf_counter() - wall_start
        return {
            "seed":              seed,
            "nodes_measured":    0,
            "total_assignments": 0,
            "sim_elapsed_s":     float("nan"),
            "wall_time_s":       round(float(wall_elapsed), 2),
            "error":             traceback.format_exc(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Terminal formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

_BAR_WIDTH = 76


def _hr(char: str = "─") -> str:
    return char * _BAR_WIDTH


def _fmt_duration(seconds: float) -> str:
    """Format seconds as  Xm Ys."""
    if not math.isfinite(seconds):
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _print_header(n_trials: int, workers: int, seed_start: int,
                  output: str) -> None:
    print()
    print(_hr("═"))
    print("  LAWSS  ·  Mega-Scale Wake Batch Harness")
    print("  Backend: lawss_stage3_wake_mega.py  "
          "(50 drones, 600 nodes, dense-cost proximity-braking CasADi MPC)")
    print(_hr("─"))
    print(f"  Trials    : {n_trials}")
    print(f"  Seeds     : {seed_start} … {seed_start + n_trials - 1}")
    print(f"  Workers   : {workers}  "
          f"(logical CPUs on this host: {os.cpu_count()})")
    print(f"  Output    : {output}")
    print(_hr("═"))
    print()


def _print_trial(idx: int, total: int, result: Dict) -> None:
    seed   = result["seed"]
    nodes  = result["nodes_measured"]
    assn   = result["total_assignments"]
    sim_t  = result["sim_elapsed_s"]
    wall   = result["wall_time_s"]
    err    = result["error"]

    w = len(str(total))
    if err is not None:
        tag    = "✗"
        detail = f"seed={seed:<5}  ERROR: {err.splitlines()[-1][:60]}"
    else:
        tag    = "✓"
        detail = (
            f"seed={seed:<5}  "
            f"nodes={nodes:>3}/600  "
            f"assigns={assn:>4}  "
            f"sim={_fmt_duration(sim_t):<10}  "
            f"wall={wall:.1f}s"
        )
    print(f"  {tag}  [{idx:>{w}}/{total}]  {detail}", flush=True)


def _print_summary(df: pd.DataFrame, n_failed: int,
                   total_wall: float) -> None:
    ok   = df[df["error"].isna()].copy()
    n_ok = len(ok)

    metrics = [
        ("Nodes Measured",    "nodes_measured",    ".1f", "nodes"),
        ("Total Assignments", "total_assignments", ".1f", ""),
        ("Sim Duration",      "sim_elapsed_s",     ".1f", "s"),
        ("Wall Time / Trial", "wall_time_s",       ".1f", "s"),
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
    print(f"  SUMMARY — Mega-Scale Wake  |  "
          f"{n_ok}/{n_ok + n_failed} trials succeeded")
    print(_hr("─"))
    print(f"  {'Metric':<{label_w}}  {hdr}")
    print(_hr("─"))
    for m in metrics:
        print(_row(*m))
    print(_hr("─"))
    print(f"  Total wall time  : {_fmt_duration(total_wall)}  ({total_wall:.1f} s)")

    sim_sum = ok["sim_elapsed_s"].dropna().sum() if n_ok else 0.0
    if total_wall > 0 and sim_sum > 0:
        print(f"  Aggregate sim    : {sim_sum:.0f} s  "
              f"|  speed-up : {sim_sum / total_wall:.1f}×")

    print(_hr("═"))
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
    """
    Run ``n_trials`` mega-scale wake simulations in parallel.

    Each trial is isolated in its own subprocess (ProcessPoolExecutor) so
    CasADi JIT state and BLAS thread pools never leak between trials.

    Parameters
    ----------
    n_trials   : number of independent seeds to simulate
    seed_start : first seed; remaining seeds are seed_start+1, +2, …
    workers    : parallel OS processes (default: cpu_count - 1)
    output     : path for the CSV results file
    quiet      : suppress per-trial progress lines

    Returns
    -------
    pd.DataFrame — one row per trial, columns:
        seed, nodes_measured, total_assignments, sim_elapsed_s,
        wall_time_s, error
    """
    seeds = list(range(seed_start, seed_start + n_trials))

    if not quiet:
        _print_header(n_trials, workers, seed_start, output)

    results:  List[Dict] = []
    n_done    = 0
    n_failed  = 0
    wall_t0   = time.perf_counter()

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
            except Exception:
                seed = future_to_seed[future]
                result = {
                    "seed":              seed,
                    "nodes_measured":    0,
                    "total_assignments": 0,
                    "sim_elapsed_s":     float("nan"),
                    "wall_time_s":       float("nan"),
                    "error":             traceback.format_exc(),
                }

            if result["error"] is not None:
                n_failed += 1

            results.append(result)

            if not quiet:
                _print_trial(n_done, n_trials, result)

    # ── Assemble and sort DataFrame ───────────────────────────────────────────
    df = pd.DataFrame(results, columns=[
        "seed",
        "nodes_measured",
        "total_assignments",
        "sim_elapsed_s",
        "wall_time_s",
        "error",
    ])
    df = df.sort_values("seed").reset_index(drop=True)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    try:
        df.to_csv(output, index=False, float_format="%.3f")
        if not quiet:
            print(f"\n  Results written → {os.path.abspath(output)}")
    except OSError as exc:
        print(f"  WARNING: could not write CSV: {exc}", file=sys.stderr)

    # ── Print summary table ───────────────────────────────────────────────────
    total_wall = time.perf_counter() - wall_t0
    if not quiet:
        _print_summary(df, n_failed, total_wall)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog            = "batch_runner_wake_mega.py",
        description     = (
            "Parallel batch harness for the LAWSS mega-scale wake simulation "
            "(50 drones, 600 nodes, updated dense-cost proximity-braking MPC).  "
            "Designed for DelftBlue and similar HPC clusters."
        ),
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-trials", "-n",
        type    = int,
        default = DEFAULT_N_TRIALS,
        metavar = "N",
        help    = "Number of independent simulation trials.",
    )
    p.add_argument(
        "--seed-start", "-s",
        type    = int,
        default = DEFAULT_SEED_START,
        metavar = "SEED",
        help    = "First RNG seed; remaining seeds are SEED+1, …, SEED+N-1.",
    )
    p.add_argument(
        "--workers", "-w",
        type    = int,
        default = DEFAULT_WORKERS,
        metavar = "W",
        help    = (
            "Parallel worker processes.  On a cluster, set to the number of "
            f"cores allocated to your job.  Host default: {DEFAULT_WORKERS}."
        ),
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
        help    = "Suppress per-trial lines; print only the final summary.",
    )
    p.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Print resolved configuration and exit (no trials run).",
    )
    return p


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
        print("  Dry-run — resolved configuration")
        print(_hr("─"))
        print("  Backend    : lawss_stage3_wake_mega.py "
              "(50 drones, 600 nodes, dense-cost proximity-braking MPC)")
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
# Guard — required for ProcessPoolExecutor on macOS / Windows (spawn context)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())
