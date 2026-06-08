from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Physical & statistical constants  (unchanged from Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

FS: float = 10.0
DT: float = 1.0 / FS

U_MEAN: float   = 12.0
I_U: float      = 0.10
T_INT: float    = 5.0

SIGMA_U: float   = I_U * U_MEAN
PHI: float       = np.exp(-DT / T_INT)
SIGMA_EPS: float = SIGMA_U * np.sqrt(1 - PHI**2)

EPSILON_CI: float = 0.05
DELTA_STAB: float = 0.01
Z_SCORE: float    = 1.96
N_EFF_MIN: int    = 10

EMA_ALPHA: float  = 0.05
LAG1_WARMUP: int  = 20

BURNIN_SAMPLES: int = int(5.0 * T_INT * FS)   # 250 samples
STAB_WIN: int       = int(5.0 * T_INT * FS)   # 250 samples

BATTERY_CAPACITY_S: float = 420.0
TRANSIT_SPEED_MAX: float  = 5.0               # placeholder; replaced in Stage 3

N_TARGETS_X: int    = 10
TARGET_SPACING: float = 5.0
TARGET_Y: float      = 0.0
TARGET_Z: float      = 10.0

N_DRONES: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class DroneState(Enum):
    IDLE     = auto()
    TRANSIT  = auto()
    SAMPLING = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Online statistics accumulator  (unchanged from Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WelfordState:
    """
    All mutable accumulators for one measurement run.
    Four-accumulator unbiased lag-1 estimator mirrors inlet_sampler.py.
    """
    wf_n: int      = 0
    wf_mean: float = 0.0
    wf_M2: float   = 0.0

    L_n: int     = 0
    L_mux: float = 0.0
    L_muy: float = 0.0
    L_mxy: float = 0.0
    L_mx2: float = 0.0

    T_int_est: float = T_INT

    mean_hist: np.ndarray = field(
        default_factory=lambda: np.full(STAB_WIN, np.nan))
    hist_ptr: int = 0

    def reset(self) -> None:
        self.wf_n = 0; self.wf_mean = 0.0; self.wf_M2 = 0.0
        self.L_n  = 0
        self.L_mux = self.L_muy = self.L_mxy = self.L_mx2 = 0.0
        self.T_int_est = T_INT
        self.mean_hist[:] = np.nan
        self.hist_ptr = 0


# ─────────────────────────────────────────────────────────────────────────────
# Drone
# ─────────────────────────────────────────────────────────────────────────────

class Drone:
    """
    Single drone.  Stage 2 additions:
      • completed_node_id  — set in _finish_run(), cleared by Environment
                             after polling.  Allows the Environment to detect
                             exactly one new completion per drone per poll
                             cycle without any callback.
    """

    def __init__(self, drone_id: int, position: np.ndarray,
                 rng: np.random.Generator):
        self.id       = drone_id
        self.position = position.astype(float).copy()
        self.velocity = np.zeros(3)
        self.rng      = rng

        self.state: DroneState = DroneState.IDLE

        self.target_position: Optional[np.ndarray] = None
        self.target_node_id:  Optional[int]        = None

        # ── NEW: one-shot completion signal for the Environment ──────────────
        # Set to the node_id when _finish_run() fires; Environment reads it
        # once then resets it to None.  Guarantees exactly-once delivery
        # without a callback.
        self.completed_node_id: Optional[int] = None

        self.battery_remaining_s: float = BATTERY_CAPACITY_S
        self.battery_depleted: bool     = False

        self.stats: WelfordState = WelfordState()
        self.samples_this_run: int = 0

        _buf = int(BATTERY_CAPACITY_S * FS)
        self.hist_raw_velocity: np.ndarray = np.full(_buf, np.nan)
        self.hist_running_mean: np.ndarray = np.full(_buf, np.nan)
        self.hist_ci_rel: np.ndarray       = np.full(_buf, np.nan)
        self.hist_T_int: np.ndarray        = np.full(_buf, np.nan)
        self._hist_ptr: int                = 0

        self._u_prime: float = 0.0

        self.last_result: Optional[dict] = None

    # ── Assignment (called by dispatcher only) ───────────────────────────────

    def assign_target(self, target_position: np.ndarray,
                      target_node_id: int) -> None:
        """Transition IDLE → TRANSIT with a new target."""
        self.target_position = target_position.copy()
        self.target_node_id  = target_node_id
        self.state           = DroneState.TRANSIT

    # ── Stage 3 stub ─────────────────────────────────────────────────────────

    def update_trajectory(self) -> None:
        """
        Stage 3 (CasADI MPC): replaced with a constant-speed linear step.
        The MPC will enforce v_max, a_max, and inter-drone separation.
        """
        if self.target_position is None:
            return
        diff = self.target_position - self.position
        dist = np.linalg.norm(diff)
        if dist < 0.05:
            self.position = self.target_position.copy()
            self.velocity = np.zeros(3)
            self._begin_sampling()
        else:
            step = min(TRANSIT_SPEED_MAX * DT, dist)
            self.position += (diff / dist) * step

    # ── Sampling internals ───────────────────────────────────────────────────

    def _begin_sampling(self) -> None:
        self.state            = DroneState.SAMPLING
        self.samples_this_run = 0
        self.stats.reset()
        self._u_prime         = 0.0

    def _generate_sample(self) -> float:
        eps           = self.rng.normal(0.0, SIGMA_EPS)
        self._u_prime = PHI * self._u_prime + eps
        return U_MEAN + self._u_prime

    def _update_welford(self, u: float, u_prev: float) -> None:
        s = self.stats
        n = self.samples_this_run

        s.wf_n   += 1
        d         = u - s.wf_mean
        s.wf_mean += d / s.wf_n
        s.wf_M2  += d * (u - s.wf_mean)

        if n > 0:
            s.L_n   += 1
            s.L_mux += (u        - s.L_mux) / s.L_n
            s.L_muy += (u_prev   - s.L_muy) / s.L_n
            s.L_mxy += (u*u_prev - s.L_mxy) / s.L_n
            s.L_mx2 += (u*u      - s.L_mx2) / s.L_n

            if s.L_n > LAG1_WARMUP:
                cov_lag    = s.L_mxy - s.L_mux * s.L_muy
                var_lag    = max(s.L_mx2 - s.L_mux**2, 1e-12)
                rho1       = np.clip(cov_lag / var_lag, 1e-6, 1.0 - 1e-6)
                T_new      = -DT / np.log(rho1)
                s.T_int_est = EMA_ALPHA * T_new + (1.0 - EMA_ALPHA) * s.T_int_est

        s.mean_hist[s.hist_ptr % STAB_WIN] = s.wf_mean
        s.hist_ptr += 1

    def _check_stopping_conditions(self) -> tuple[bool, bool, bool]:
        s         = self.stats
        current_T = s.wf_n * DT
        var_u     = s.wf_M2 / max(s.wf_n - 1, 1)

        sigma_Ubar = np.sqrt(max(2.0 * var_u * s.T_int_est / current_T, 0.0))

        N_eff = current_T / (2.0 * s.T_int_est)
        cond3 = N_eff >= N_EFF_MIN

        ci_rel = Z_SCORE * sigma_Ubar / max(abs(s.wf_mean), 1e-9)
        cond1  = ci_rel < EPSILON_CI

        oldest_ptr = s.hist_ptr % STAB_WIN
        oldest     = s.mean_hist[oldest_ptr]
        if np.isfinite(oldest):
            stab  = abs(s.wf_mean - oldest) / max(abs(s.wf_mean), 1e-9)
            cond2 = stab < DELTA_STAB
        else:
            cond2 = False

        return cond1, cond2, cond3

    def _record_history(self, u: float, ci_rel: float) -> None:
        ptr = self._hist_ptr % len(self.hist_raw_velocity)
        self.hist_raw_velocity[ptr] = u
        self.hist_running_mean[ptr] = self.stats.wf_mean
        self.hist_ci_rel[ptr]       = ci_rel
        self.hist_T_int[ptr]        = self.stats.T_int_est
        self._hist_ptr += 1

    def _finish_run(self) -> None:
        """
        Store results, signal the Environment via completed_node_id,
        and return to IDLE.

        completed_node_id is written here and cleared by the Environment
        in _collect_completions() after being read.  The Drone does NOT
        clear it itself — that would race with the Environment's poll.
        """
        s = self.stats
        self.last_result = {
            "node_id":   self.target_node_id,
            "position":  self.position.copy(),
            "mean_est":  s.wf_mean,
            "T_int_est": s.T_int_est,
            "n_samples": self.samples_this_run,
            "elapsed_s": self.samples_this_run * DT,
        }
        # Signal the Environment — exactly-once, no callback needed.
        self.completed_node_id = self.target_node_id

        self.target_position = None
        self.target_node_id  = None
        self.state           = DroneState.IDLE

    # ── Main per-tick update ──────────────────────────────────────────────────

    def tick(self, u_prev_store: list) -> None:
        if self.battery_depleted:
            return

        self.battery_remaining_s -= DT
        if self.battery_remaining_s <= 0.0:
            self.battery_remaining_s = 0.0
            self.battery_depleted    = True
            self.state               = DroneState.IDLE
            return

        if self.state == DroneState.IDLE:
            pass   # dispatcher will act on this drone next

        elif self.state == DroneState.TRANSIT:
            self.update_trajectory()

        elif self.state == DroneState.SAMPLING:
            u      = self._generate_sample()
            u_prev = u_prev_store[0]

            self._update_welford(u, u_prev)
            self.samples_this_run += 1
            u_prev_store[0] = u

            s          = self.stats
            var_u      = s.wf_M2 / max(s.wf_n - 1, 1)
            current_T  = s.wf_n * DT
            sigma_Ubar = np.sqrt(max(2.0 * var_u * s.T_int_est / current_T, 0.0))
            ci_rel_now = Z_SCORE * sigma_Ubar / max(abs(s.wf_mean), 1e-9)
            self._record_history(u, ci_rel_now)

            if self.samples_this_run < BURNIN_SAMPLES:
                return

            cond1, cond2, cond3 = self._check_stopping_conditions()
            if cond1 and cond2 and cond3:
                self._finish_run()


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class Environment:
    """
    Stage 2: owns the Hungarian dispatcher and all node-state bookkeeping.

    Node state matrix
    ──────────────────
    target_measured[i]  True once a drone has completed sampling node i.
                        Permanent — never flipped back to False.
    target_locked[i]    True while a drone is in TRANSIT or SAMPLING at i.
                        Set by dispatch(); cleared by _collect_completions().

    A node is "available for assignment" iff:
        not target_measured[i] and not target_locked[i]
    """

    def __init__(self, seed: int = 42):
        self.tick_count: int  = 0
        self.elapsed_s: float = 0.0

        master_rng  = np.random.default_rng(seed)
        drone_seeds = master_rng.integers(0, 2**31, size=N_DRONES)

        # Target grid
        self.target_positions: np.ndarray = np.array([
            [i * TARGET_SPACING, TARGET_Y, TARGET_Z]
            for i in range(N_TARGETS_X)
        ], dtype=float)

        # Node state arrays — both False at construction
        self.target_measured: np.ndarray = np.zeros(N_TARGETS_X, dtype=bool)
        self.target_locked:   np.ndarray = np.zeros(N_TARGETS_X, dtype=bool)

        # Drone fleet — all start at the origin, ground level
        self.drones: list[Drone] = [
            Drone(
                drone_id=i,
                position=np.zeros(3),
                rng=np.random.default_rng(int(drone_seeds[i])),
            )
            for i in range(N_DRONES)
        ]

        self._u_prev: list[list[float]] = [[U_MEAN] for _ in range(N_DRONES)]

        # Dispatch statistics (useful for smoke-test audit)
        self._dispatch_calls:      int = 0
        self._assignments_made:    int = 0

    # ── Completion polling ────────────────────────────────────────────────────

    def _collect_completions(self) -> None:
        """
        Scan every drone for a pending completed_node_id.

        When found:
          1. Mark that node as measured (permanent).
          2. Unlock it so dispatch() stops treating it as occupied.
          3. Clear completed_node_id on the drone (consumed).

        Called at the top of step(), before dispatch(), so the assignment
        pool is always up-to-date before the Hungarian solver runs.
        """
        for drone in self.drones:
            node_id = drone.completed_node_id
            if node_id is None:
                continue
            # Validate — guard against any future logic bug producing bad ids
            if 0 <= node_id < N_TARGETS_X:
                self.target_measured[node_id] = True
                self.target_locked[node_id]   = False
            drone.completed_node_id = None   # consume the signal

    # ── Hungarian dispatcher ──────────────────────────────────────────────────

    def dispatch(self) -> None:
        """
        Assign idle drones to unmeasured, unlocked nodes via the
        Hungarian algorithm.

        Fast-path: returns immediately (< 1 µs) when there is nothing to do.

        Algorithm
        ─────────
        1. Collect idle drone indices.
        2. Collect free node indices  (not measured AND not locked).
        3. Build an (n_idle × n_free) Euclidean distance cost matrix.
        4. Solve with linear_sum_assignment (minimises total distance).
        5. For each (drone_row, node_col) pair: call assign_target()
           and immediately lock the node.

        Cost metric: Euclidean distance in 3-D.  For Stage 3 this will
        be replaced by estimated flight time from the MPC, but the
        dispatcher interface stays identical.
        """
        self._dispatch_calls += 1

        # ── 1. Idle drones ────────────────────────────────────────────────────
        idle_ids = [
            d.id for d in self.drones
            if d.state == DroneState.IDLE and not d.battery_depleted
        ]
        if not idle_ids:
            return   # fast-path: nothing to assign

        # ── 2. Free nodes ─────────────────────────────────────────────────────
        free_node_ids = [
            i for i in range(N_TARGETS_X)
            if not self.target_measured[i] and not self.target_locked[i]
        ]
        if not free_node_ids:
            return   # fast-path: no nodes available

        # ── 3. Cost matrix  (n_idle × n_free) ────────────────────────────────
        idle_positions = np.array(
            [self.drones[i].position for i in idle_ids])          # (n_idle, 3)
        node_positions = self.target_positions[free_node_ids]     # (n_free, 3)

        # Broadcasting: diff shape (n_idle, n_free, 3) → norm → (n_idle, n_free)
        diff        = idle_positions[:, np.newaxis, :] - node_positions[np.newaxis, :, :]
        cost_matrix = np.linalg.norm(diff, axis=-1)               # (n_idle, n_free)

        # ── 4. Solve ──────────────────────────────────────────────────────────
        # linear_sum_assignment always assigns min(n_idle, n_free) pairs,
        # covering whichever side is the binding constraint.
        drone_rows, node_cols = linear_sum_assignment(cost_matrix)

        # ── 5. Execute assignments ────────────────────────────────────────────
        for dr, nc in zip(drone_rows, node_cols):
            drone_id  = idle_ids[dr]
            node_id   = free_node_ids[nc]
            drone     = self.drones[drone_id]

            drone.assign_target(
                target_position=self.target_positions[node_id],
                target_node_id=node_id,
            )
            self.target_locked[node_id] = True   # block until completion
            self._assignments_made += 1

    # ── Simulation clock ──────────────────────────────────────────────────────

    def step(self) -> None:
        """
        One 10 Hz tick:
          1. Advance every drone (physics + sampling + statistics).
          2. Collect any new completions (updates measured/locked arrays).
          3. Dispatch idle drones to free nodes.
          4. Increment clock.
        """
        for i, drone in enumerate(self.drones):
            drone.tick(self._u_prev[i])

        self._collect_completions()   # must precede dispatch()
        self.dispatch()

        self.tick_count += 1
        self.elapsed_s  += DT

    def run(self, duration_s: float) -> None:
        """Blocking loop — headless testing only; replaced by FuncAnimation."""
        for _ in range(int(duration_s / DT)):
            self.step()

    # ── Properties for easy external inspection ───────────────────────────────

    @property
    def n_measured(self) -> int:
        return int(self.target_measured.sum())

    @property
    def all_measured(self) -> bool:
        return bool(self.target_measured.all())

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self, show_completed: bool = True) -> str:
        lines = [
            f"t = {self.elapsed_s:.1f} s  |  tick {self.tick_count}  |  "
            f"nodes measured: {self.n_measured}/{N_TARGETS_X}  |  "
            f"assignments so far: {self._assignments_made}",
            "",
            f"  {'ID':>3}  {'State':>10}  {'Node':>5}  "
            f"{'Samples':>8}  {'Mean est':>9}  {'T_int':>7}  "
            f"{'Battery':>7}  {'Locked?':>7}",
            "  " + "─" * 68,
        ]

        for d in self.drones:
            s     = d.stats
            node  = str(d.target_node_id) if d.target_node_id is not None else "—"
            mean  = f"{s.wf_mean:.3f}" if s.wf_n > 0 else "—"
            tint  = f"{s.T_int_est:.2f}" if s.L_n > LAG1_WARMUP else "prior"
            batt  = f"{d.battery_remaining_s:.0f}s"

            # Show whether the drone's last completed node was locked
            locked_flag = ""
            if d.last_result is not None:
                nid = d.last_result["node_id"]
                locked_flag = "Y" if self.target_locked[nid] else "N"

            lines.append(
                f"  {d.id:>3}  {d.state.name:>10}  {node:>5}  "
                f"{d.samples_this_run:>8}  {mean:>9}  {tint:>7}  "
                f"{batt:>7}  {locked_flag:>7}"
            )

        if show_completed:
            completed = [d for d in self.drones if d.last_result is not None]
            if completed:
                lines += ["", "  Completed runs:"]
                # Sort by elapsed time so the table is readable
                for d in sorted(completed,
                                 key=lambda x: x.last_result["elapsed_s"]):
                    r   = d.last_result
                    err = abs(r["mean_est"] - U_MEAN) / U_MEAN * 100
                    lines.append(
                        f"    Drone {d.id:>2}  node {r['node_id']:>2}  "
                        f"mean={r['mean_est']:.4f} m/s  "
                        f"err={err:.2f}%  "
                        f"T={r['elapsed_s']:.1f}s  "
                        f"T_int_est={r['T_int_est']:.2f}s"
                    )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Verify Stage 2 correctness without _demo_assign_all().

    Assertions checked
    ───────────────────
    A. All 10 nodes are measured before the 7-minute battery expires.
    B. No node is ever simultaneously measured by two drones
       (double-assignment prevention).
    C. target_locked[i] is False for every measured node at simulation end.
    D. total assignments == N_DRONES (one per node, no re-assignments needed
       given 10 drones and 10 nodes in this simple scenario).
    """
    print("=" * 70)
    print("  LAWSS Stage 2 — Smoke Test  (Hungarian Dispatcher)")
    print(f"  {N_DRONES} drones  |  {N_TARGETS_X} nodes  |  "
          f"U_mean={U_MEAN} m/s  I_u={I_U}  T_int={T_INT} s")
    print(f"  ε_ci={EPSILON_CI*100:.0f}%  δ_stab={DELTA_STAB*100:.0f}%  "
          f"Z={Z_SCORE}  N_eff_min={N_EFF_MIN}")
    print(f"  Transit speed (stub): {TRANSIT_SPEED_MAX} m/s")
    print("=" * 70)

    env = Environment(seed=0)

    snapshot_times: set[float] = {60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0}
    t0_wall = time.perf_counter()

    early_stop_t: Optional[float] = None

    for _ in range(int(420 / DT)):
        env.step()

        t = round(env.elapsed_s, 1)
        if t in snapshot_times:
            snapshot_times.discard(t)
            print(f"\n{'─' * 70}")
            print(env.summary())

        if env.all_measured and early_stop_t is None:
            early_stop_t = env.elapsed_s

    wall = time.perf_counter() - t0_wall

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  Simulation complete.")
    print(f"  Wall time  : {wall:.3f} s  ({420/wall:.0f}× real-time)")
    if early_stop_t:
        print(f"  All nodes measured at t = {early_stop_t:.1f} s "
              f"({early_stop_t/60:.2f} min)  — {420-early_stop_t:.1f}s before budget.")
    else:
        print(f"  WARNING: not all nodes measured within 420 s budget.")
    print(f"  Total assignments : {env._assignments_made}")
    print(f"{'═' * 70}")
    print(env.summary(show_completed=True))

    # ── Assertions ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Assertions:")

    # A: all nodes measured
    assert env.all_measured, \
        f"FAIL A: only {env.n_measured}/{N_TARGETS_X} nodes measured"
    print("  A — All nodes measured within budget          ✓")

    # B: no node double-measured (each node appears at most once in last_results)
    measured_nodes = [
        d.last_result["node_id"]
        for d in env.drones
        if d.last_result is not None
    ]
    assert len(measured_nodes) == len(set(measured_nodes)), \
        f"FAIL B: duplicate node measurements: {measured_nodes}"
    print("  B — No node measured twice (no double-assign)  ✓")

    # C: no measured node still locked
    still_locked = [
        i for i in range(N_TARGETS_X)
        if env.target_measured[i] and env.target_locked[i]
    ]
    assert not still_locked, \
        f"FAIL C: nodes {still_locked} are measured but still locked"
    print("  C — All measured nodes correctly unlocked      ✓")

    # D: assignment count equals N_TARGETS_X (one pass covers all nodes)
    assert env._assignments_made == N_TARGETS_X, \
        (f"FAIL D: expected {N_TARGETS_X} assignments, "
         f"got {env._assignments_made}")
    print(f"  D — Exactly {N_TARGETS_X} assignments made (one per node)  ✓")

    print(f"{'─' * 70}")
    print("  All assertions passed.")


if __name__ == "__main__":
    _smoke_test()
