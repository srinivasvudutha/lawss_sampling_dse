from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Physical & statistical constants  (mirror inlet_sampler.py defaults)
# ─────────────────────────────────────────────────────────────────────────────

# Simulation clock
FS: float = 10.0          # sampling / physics rate [Hz]
DT: float = 1.0 / FS     # timestep [s]

# Atmospheric boundary-layer signal
U_MEAN: float   = 12.0   # mean inlet wind speed [m/s]
I_U: float      = 0.10   # turbulence intensity σ_u / U [-]
T_INT: float    = 5.0    # integral time scale [s]

# Derived AR(1) coefficients
SIGMA_U: float     = I_U * U_MEAN
PHI: float         = np.exp(-DT / T_INT)          # AR(1) autocorrelation coeff
SIGMA_EPS: float   = SIGMA_U * np.sqrt(1 - PHI**2)

# Stopping-criterion thresholds  (inlet_sampler.py values)
EPSILON_CI: float  = 0.05    # Cond 1: CI half-width / mean < 5 %
DELTA_STAB: float  = 0.01    # Cond 2: running-mean drift < 1 %
Z_SCORE: float     = 1.96    # 95 % confidence
N_EFF_MIN: int     = 10      # Cond 3: min effective independent samples

EMA_ALPHA: float   = 0.05    # EMA smoothing for T_int estimate
LAG1_WARMUP: int   = 20      # lag-1 accumulator needs this many pairs first

# Burn-in: must collect 5 × T_INT seconds before any condition fires
BURNIN_SAMPLES: int = int(5.0 * T_INT * FS)   # 250 samples at 10 Hz

# Stability circular buffer: one slot per sample over 5 × T_INT window
STAB_WIN: int = int(5.0 * T_INT * FS)         # 250 samples

# Battery model  (14-min cycle, 2 × 7-min measurement iterations)
BATTERY_CAPACITY_S: float  = 420.0   # 7 min per measurement pass [s]
TRANSIT_SPEED_MAX: float   = 5.0     # placeholder max speed [m/s]

# Target grid geometry  (10 × 1 inlet plane, 5 m spacing, z = 10 m)
N_TARGETS_X: int = 10
TARGET_SPACING: float = 5.0          # [m]
TARGET_Y: float = 0.0                # inlet plane y-coordinate
TARGET_Z: float = 10.0              # measurement height [m]

# Swarm
N_DRONES: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Drone state machine
# ─────────────────────────────────────────────────────────────────────────────

class DroneState(Enum):
    IDLE     = auto()   # awaiting target assignment
    TRANSIT  = auto()   # flying toward assigned target  (CasADI MPC — Stage 3)
    SAMPLING = auto()   # hovering and collecting data


# ─────────────────────────────────────────────────────────────────────────────
# Online statistics accumulator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WelfordState:
    """
    Holds all mutable accumulators for one drone's current measurement run.

    Lag-1 estimator uses four separate running means — identical to
    inlet_sampler.py's L_mux / L_muy / L_mxy / L_mx2 approach — so
    we compute an unbiased online covariance:

        cov  = E[u_k * u_{k-1}] - E[u_k] * E[u_{k-1}]
        var  = E[u_k^2]         - E[u_k]^2

    This avoids the biased shortcut that breaks in wake flows (Stage 2+).
    """
    # Welford grand mean / variance
    wf_n: int     = 0
    wf_mean: float = 0.0
    wf_M2: float  = 0.0          # running sum of squared deviations

    # Lag-1 pair accumulators (online means over lag pairs)
    L_n: int     = 0
    L_mux: float = 0.0           # E[u_k]
    L_muy: float = 0.0           # E[u_{k-1}]
    L_mxy: float = 0.0           # E[u_k * u_{k-1}]
    L_mx2: float = 0.0           # E[u_k^2]

    # EMA-smoothed T_int estimate  (seeded with true T_INT as prior)
    T_int_est: float = T_INT

    # Stability circular buffer
    mean_hist: np.ndarray = field(
        default_factory=lambda: np.full(STAB_WIN, np.nan))
    hist_ptr: int = 0

    def reset(self) -> None:
        """Clear all accumulators for the next measurement run."""
        self.wf_n    = 0
        self.wf_mean = 0.0
        self.wf_M2   = 0.0
        self.L_n     = 0
        self.L_mux   = self.L_muy = self.L_mxy = self.L_mx2 = 0.0
        self.T_int_est = T_INT
        self.mean_hist[:] = np.nan
        self.hist_ptr = 0


# ─────────────────────────────────────────────────────────────────────────────
# Drone
# ─────────────────────────────────────────────────────────────────────────────

class Drone:
    """
    Single drone in the LAWSS swarm.

    Responsibilities
    ─────────────────
    - Maintain state machine (IDLE → TRANSIT → SAMPLING → IDLE).
    - In SAMPLING: ingest one velocity sample per 10 Hz tick,
      run online Welford + lag-1 statistics, evaluate three stopping
      conditions exactly as in inlet_sampler.py.
    - Track battery usage and flag when a recharge cycle is needed.

    Stage stubs
    ────────────
    - update_trajectory()  left empty for Stage 3 (CasADI MPC).
    - assign_target()      left empty for Stage 2 (Hungarian dispatcher).
    """

    def __init__(self, drone_id: int, position: np.ndarray, rng: np.random.Generator):
        self.id: int             = drone_id
        self.position: np.ndarray = position.astype(float).copy()   # [x, y, z]
        self.velocity: np.ndarray = np.zeros(3)                      # [vx, vy, vz]
        self.rng: np.random.Generator = rng

        self.state: DroneState = DroneState.IDLE

        # Target assignment
        self.target_position: Optional[np.ndarray] = None
        self.target_node_id:  Optional[int]         = None

        # Battery  (seconds of hover-equivalent remaining)
        self.battery_remaining_s: float = BATTERY_CAPACITY_S
        self.battery_depleted: bool     = False

        # Per-run statistics
        self.stats: WelfordState = WelfordState()
        self.samples_this_run: int = 0

        # History for plotting (ring buffer, 420 s × 10 Hz = 4200 entries)
        _buf = int(BATTERY_CAPACITY_S * FS)
        self.hist_raw_velocity: np.ndarray  = np.full(_buf, np.nan)
        self.hist_running_mean: np.ndarray  = np.full(_buf, np.nan)
        self.hist_ci_rel: np.ndarray        = np.full(_buf, np.nan)
        self.hist_T_int: np.ndarray         = np.full(_buf, np.nan)
        self._hist_ptr: int                 = 0

        # AR(1) state for this drone's local turbulence realisation
        self._u_prime: float = 0.0   # previous fluctuation value

        # Results from the completed run
        self.last_result: Optional[dict] = None

    # ── Stage stubs ──────────────────────────────────────────────────────────

    def assign_target(self, target_position: np.ndarray, target_node_id: int) -> None:
        """
        Stage 2 (Hungarian Dispatcher): assign a measurement node and
        transition from IDLE to TRANSIT.
        Called by the Environment dispatcher; do not call directly.
        """
        # Placeholder — dispatcher logic added in Stage 2.
        self.target_position = target_position.copy()
        self.target_node_id  = target_node_id
        self.state           = DroneState.TRANSIT

    def update_trajectory(self) -> None:
        """
        Stage 3 (CasADI MPC): compute one-step optimal control action,
        advance self.position and self.velocity toward self.target_position
        while respecting v_max, a_max, and inter-drone collision constraints.
        Called every tick while state == TRANSIT.
        """
        # Temporary placeholder: teleport to target for testing.
        if self.target_position is not None:
            diff = self.target_position - self.position
            dist = np.linalg.norm(diff)
            if dist < 0.05:
                self.position = self.target_position.copy()
                self.velocity = np.zeros(3)
                self._begin_sampling()
            else:
                step = min(TRANSIT_SPEED_MAX * DT, dist)
                self.position += (diff / dist) * step

    # ── Core sampling logic ──────────────────────────────────────────────────

    def _begin_sampling(self) -> None:
        """Called the moment the drone arrives at its target node."""
        self.state           = DroneState.SAMPLING
        self.samples_this_run = 0
        self.stats.reset()
        self._u_prime         = 0.0

    def _generate_sample(self) -> float:
        """
        Advance the AR(1) process by one step and return u(t).

        u(t) = U_mean + u'(t)
        u'(t) = phi * u'(t-1) + eps,   eps ~ N(0, sigma_eps)

        Using the drone's own RNG so each drone gets an independent
        turbulence realisation (statistically identical distribution,
        different noise sequence).
        """
        eps           = self.rng.normal(0.0, SIGMA_EPS)
        self._u_prime = PHI * self._u_prime + eps
        return U_MEAN + self._u_prime

    def _update_welford(self, u: float, u_prev: float) -> None:
        """
        Ingest one sample into all online accumulators.
        Mirrors the inner loop body of inlet_sampler.py exactly.
        """
        s = self.stats
        n = self.samples_this_run   # 0-indexed sample count at entry

        # ── Welford grand mean / variance ────────────────────────────────────
        s.wf_n    += 1
        d          = u - s.wf_mean
        s.wf_mean += d / s.wf_n
        s.wf_M2   += d * (u - s.wf_mean)     # note: uses updated mean (Welford)

        # ── Lag-1 pair accumulators (requires at least one previous sample) ──
        if n > 0:
            s.L_n   += 1
            s.L_mux += (u        - s.L_mux) / s.L_n
            s.L_muy += (u_prev   - s.L_muy) / s.L_n
            s.L_mxy += (u*u_prev - s.L_mxy) / s.L_n
            s.L_mx2 += (u*u      - s.L_mx2) / s.L_n

            if s.L_n > LAG1_WARMUP:
                cov_lag = s.L_mxy - s.L_mux * s.L_muy
                var_lag = max(s.L_mx2 - s.L_mux**2, 1e-12)
                rho1    = np.clip(cov_lag / var_lag, 1e-6, 1.0 - 1e-6)
                T_new   = -DT / np.log(rho1)
                # EMA smoothing: damps single-sample noise spikes
                s.T_int_est = EMA_ALPHA * T_new + (1.0 - EMA_ALPHA) * s.T_int_est

        # ── Stability circular buffer ────────────────────────────────────────
        s.mean_hist[s.hist_ptr % STAB_WIN] = s.wf_mean
        s.hist_ptr += 1

    def _check_stopping_conditions(self) -> tuple[bool, bool, bool]:
        """
        Evaluate the three inlet stopping conditions.
        Returns (cond1, cond2, cond3).
        Matches Step 5 of inlet_sampler.py; must be called after burn-in.
        """
        s           = self.stats
        current_T   = s.wf_n * DT                # elapsed sampling time [s]

        # Sample variance (unbiased)
        var_u       = s.wf_M2 / max(s.wf_n - 1, 1)

        # Standard error of the time-averaged mean
        # σ_Ū² = 2 σ_u² T_int / T   (Lenschow et al. 1994)
        sigma_Ubar  = np.sqrt(max(2.0 * var_u * s.T_int_est / current_T, 0.0))

        # ── Condition 3: effective independent samples ────────────────────────
        N_eff = current_T / (2.0 * s.T_int_est)
        cond3 = N_eff >= N_EFF_MIN

        # ── Condition 1: 95% CI half-width relative to estimated mean ─────────
        ci_rel = Z_SCORE * sigma_Ubar / max(abs(s.wf_mean), 1e-9)
        cond1  = ci_rel < EPSILON_CI

        # ── Condition 2: mean drift over 5 × T_int stability window ───────────
        oldest_ptr = s.hist_ptr % STAB_WIN   # oldest entry in circular buffer
        oldest     = s.mean_hist[oldest_ptr]
        if np.isfinite(oldest):
            stab  = abs(s.wf_mean - oldest) / max(abs(s.wf_mean), 1e-9)
            cond2 = stab < DELTA_STAB
        else:
            cond2 = False   # buffer not yet full — cannot evaluate

        return cond1, cond2, cond3

    def _record_history(self, u: float, ci_rel: float) -> None:
        """Write current-tick scalars into the rolling plot buffer."""
        ptr = self._hist_ptr % len(self.hist_raw_velocity)
        self.hist_raw_velocity[ptr] = u
        self.hist_running_mean[ptr] = self.stats.wf_mean
        self.hist_ci_rel[ptr]       = ci_rel
        self.hist_T_int[ptr]        = self.stats.T_int_est
        self._hist_ptr += 1

    def _finish_run(self) -> None:
        """
        Record results and return drone to IDLE to await re-assignment.
        """
        s = self.stats
        self.last_result = {
            "node_id":    self.target_node_id,
            "position":   self.position.copy(),
            "mean_est":   s.wf_mean,
            "T_int_est":  s.T_int_est,
            "n_samples":  self.samples_this_run,
            "elapsed_s":  self.samples_this_run * DT,
        }
        self.target_position = None
        self.target_node_id  = None
        self.state           = DroneState.IDLE

    # ── Main per-tick update ─────────────────────────────────────────────────

    def tick(self, u_prev_store: list) -> None:
        """
        Advance the drone by one 10 Hz timestep.

        Parameters
        ----------
        u_prev_store : mutable list of length 1 holding the previous
                       velocity sample for this drone (needed by lag-1).
                       The Environment owns and passes this per-drone.
        """
        if self.battery_depleted:
            return

        self.battery_remaining_s -= DT
        if self.battery_remaining_s <= 0.0:
            self.battery_remaining_s = 0.0
            self.battery_depleted    = True
            self.state               = DroneState.IDLE
            return

        if self.state == DroneState.IDLE:
            # Nothing to do until Stage 2 assigns a target.
            pass

        elif self.state == DroneState.TRANSIT:
            self.update_trajectory()   # Stage 3 MPC placeholder

        elif self.state == DroneState.SAMPLING:
            # Generate a new velocity sample
            u      = self._generate_sample()
            u_prev = u_prev_store[0]

            # Update all online accumulators
            self._update_welford(u, u_prev)
            self.samples_this_run += 1
            u_prev_store[0] = u    # hand back to Environment for next tick

            # Compute ci_rel for history (re-derive cheaply from current state)
            s           = self.stats
            var_u       = s.wf_M2 / max(s.wf_n - 1, 1)
            current_T   = s.wf_n * DT
            sigma_Ubar  = np.sqrt(max(2.0 * var_u * s.T_int_est / current_T, 0.0))
            ci_rel_now  = Z_SCORE * sigma_Ubar / max(abs(s.wf_mean), 1e-9)
            self._record_history(u, ci_rel_now)

            # Enforce burn-in before evaluating stopping conditions
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
    Owns the simulation clock, the target node grid, and the drone swarm.

    Responsibilities
    ─────────────────
    - Build the 10-node inlet target grid.
    - Instantiate N_DRONES Drone objects with independent RNGs.
    - Advance the simulation one 10 Hz tick at a time via step().
    - Expose state for the Stage 4 dashboard.

    Design constraints
    ──────────────────
    - step() must complete in << 100 ms so it never blocks the UI thread.
    - All randomness is seeded through a single master RNG, then split
      into per-drone child generators for reproducibility.
    """

    def __init__(self, seed: int = 42):
        self.tick_count: int = 0
        self.elapsed_s: float = 0.0

        # Master RNG → per-drone child generators
        master_rng = np.random.default_rng(seed)
        drone_seeds = master_rng.integers(0, 2**31, size=N_DRONES)

        # Target grid: N_TARGETS_X nodes equally spaced in x, fixed y and z
        self.target_positions: np.ndarray = np.array([
            [i * TARGET_SPACING, TARGET_Y, TARGET_Z]
            for i in range(N_TARGETS_X)
        ], dtype=float)
        self.target_measured: np.ndarray = np.zeros(N_TARGETS_X, dtype=bool)

        # Drone fleet: start clustered at x=0, z=0 (ground level, pre-takeoff)
        self.drones: list[Drone] = []
        for i in range(N_DRONES):
            pos = np.array([0.0, 0.0, 0.0], dtype=float)
            rng = np.random.default_rng(int(drone_seeds[i]))
            self.drones.append(Drone(drone_id=i, position=pos, rng=rng))

        # Per-drone previous-sample store (lag-1 needs u[t-1])
        # Each element is a mutable list[float] of length 1 so the Drone
        # can write back to it without allocating.
        self._u_prev: list[list[float]] = [[U_MEAN] for _ in range(N_DRONES)]

        # Simple demo: immediately assign each drone to a unique target.
        # Stage 2 will replace this with the Hungarian dispatcher.
        self._demo_assign_all()

    # ── Demo assignment (replaced by dispatcher in Stage 2) ─────────────────

    def _demo_assign_all(self) -> None:
        """
        One-shot initial assignment: drone i → target node i.
        Shows that the pipeline reaches SAMPLING without needing Stage 2.
        Replaced entirely by Stage 2's dispatcher.
        """
        for i, drone in enumerate(self.drones):
            if i < len(self.target_positions):
                drone.assign_target(
                    target_position=self.target_positions[i],
                    target_node_id=i,
                )

    # ── Dispatcher stub (Stage 2) ────────────────────────────────────────────

    def dispatch(self) -> None:
        """
        Stage 2 (Hungarian Dispatcher): find all IDLE drones and
        all un-measured nodes, then solve the assignment problem to
        pair them optimally and call drone.assign_target().
        """
        pass   # implemented in Stage 2

    # ── Simulation clock ─────────────────────────────────────────────────────

    def step(self) -> None:
        """
        Advance the entire simulation by one 10 Hz tick.

        Call order per tick:
          1. Tick every drone (physics + sampling + statistics).
          2. Run the dispatcher for any newly IDLE drones (Stage 2).
          3. Increment global clock.
        """
        for i, drone in enumerate(self.drones):
            drone.tick(self._u_prev[i])

        self.dispatch()   # no-op until Stage 2

        self.tick_count += 1
        self.elapsed_s  += DT

    def run(self, duration_s: float) -> None:
        """
        Blocking simulation loop — for headless testing only.
        Stage 4 will replace this with a non-blocking FuncAnimation callback.
        """
        n_steps = int(duration_s / DT)
        for _ in range(n_steps):
            self.step()

    # ── Reporting helpers ────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"t = {self.elapsed_s:.1f} s  (tick {self.tick_count})",
            f"{'ID':>3}  {'State':>10}  {'Node':>5}  "
            f"{'Samples':>8}  {'Mean est':>10}  {'T_int est':>10}  "
            f"{'Battery':>8}",
            "-" * 65,
        ]
        for d in self.drones:
            s    = d.stats
            node = str(d.target_node_id) if d.target_node_id is not None else "—"
            mean = f"{s.wf_mean:.3f}" if s.wf_n > 0 else "—"
            tint = f"{s.T_int_est:.2f}" if s.L_n > LAG1_WARMUP else "prior"
            batt = f"{d.battery_remaining_s:.0f} s"
            lines.append(
                f"{d.id:>3}  {d.state.name:>10}  {node:>5}  "
                f"{d.samples_this_run:>8}  {mean:>10}  {tint:>10}  {batt:>8}"
            )
        # Completed runs
        done = [d for d in self.drones if d.last_result is not None and
                d.state == DroneState.IDLE]
        if done:
            lines.append("\nCompleted runs:")
            for d in done:
                r = d.last_result
                err = abs(r["mean_est"] - U_MEAN) / U_MEAN * 100
                lines.append(
                    f"  Drone {d.id}: node {r['node_id']}  "
                    f"mean={r['mean_est']:.4f} m/s  "
                    f"err={err:.2f}%  "
                    f"T={r['elapsed_s']:.1f}s  "
                    f"T_int_est={r['T_int_est']:.2f}s"
                )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Run the environment for 420 s (one full 7-min battery pass) and print
    a status snapshot every 60 s, then show final summary.

    Expected: all 10 drones should complete their sampling run well within
    420 s given I_u=10% and T_int=5 s.  Typical stop times from
    inlet_sampler.py are 70–150 s.
    """
    print("=" * 65)
    print("  LAWSS Stage 1 — Smoke Test")
    print(f"  U_mean={U_MEAN} m/s  I_u={I_U}  T_int={T_INT} s  fs={FS} Hz")
    print(f"  ε_ci={EPSILON_CI*100:.0f}%  δ_stab={DELTA_STAB*100:.0f}%  "
          f"Z={Z_SCORE}  N_eff_min={N_EFF_MIN}")
    print(f"  Burn-in = {BURNIN_SAMPLES} samples ({BURNIN_SAMPLES*DT:.0f} s)")
    print("=" * 65)

    env = Environment(seed=0)

    snapshot_times = {60, 120, 180, 240, 300, 360, 420}
    t0_wall = time.perf_counter()

    for step_i in range(int(420 / DT)):
        env.step()
        if round(env.elapsed_s, 1) in snapshot_times:
            snapshot_times.discard(round(env.elapsed_s, 1))
            print(f"\n{'─'*65}")
            print(env.summary())

    wall = time.perf_counter() - t0_wall
    print(f"\n{'═'*65}")
    print(f"  Simulation complete.  Wall time: {wall:.2f} s  "
          f"({420/wall:.0f}× real-time)")
    print(f"{'═'*65}")
    print(env.summary())


if __name__ == "__main__":
    _smoke_test()
