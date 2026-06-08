from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Physical & statistical constants  (inlet physics — UNCHANGED from Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

FS: float = 10.0
DT: float = 1.0 / FS

U_MEAN: float   = 12.0
I_U: float      = 0.10
T_INT: float    = 5.0

SIGMA_U: float   = I_U * U_MEAN
PHI: float       = np.exp(-DT / T_INT)
SIGMA_EPS: float = SIGMA_U * np.sqrt(1.0 - PHI**2)

EPSILON_CI: float = 0.05
DELTA_STAB: float = 0.01
Z_SCORE: float    = 1.96
N_EFF_MIN: int    = 10

EMA_ALPHA: float  = 0.05
LAG1_WARMUP: int  = 20

BURNIN_SAMPLES: int = int(5.0 * T_INT * FS)   # 250 samples
STAB_WIN: int       = int(5.0 * T_INT * FS)   # 250 samples

# ─────────────────────────────────────────────────────────────────────────────
# Scale-up constants
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_CAPACITY_S: float = 1080.0   # 18-min battery (same as inlet stage 3)

N_DRONES:  int = 20    # SCALE-UP: 20 drones
N_TARGETS: int = 250   # SCALE-UP: 250 random 3-D nodes

MAX_SAMPLING_TIME_S: float = 180.0   # 3-minute hard cap per run (unchanged)

# ─────────────────────────────────────────────────────────────────────────────
# MPC constants
# ─────────────────────────────────────────────────────────────────────────────

MPC_N:   int   = 20
V_MAX:   float = 14.0
A_MAX:   float = 2.0
D_MIN:   float = 3.0
N_OBS:   int   = N_DRONES - 1
MASS:    float = 3.645
Q_STAGE: float = 1.0
Q_TERM:  float = 100.0
Q_VEL:       float = 5.0
Q_VEL_PROX:  float = 5.0
R_CTRL:      float = 0.1

ARRIVAL_DIST:  float = 0.5
ARRIVAL_SPEED: float = 0.3
RTH_THRESHOLD: float = 0.07

_OBS_SENTINEL = np.array([1e6, 1e6, 1e6], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class DroneState(Enum):
    IDLE     = auto()
    TRANSIT  = auto()
    SAMPLING = auto()
    RTH      = auto()


# ─────────────────────────────────────────────────────────────────────────────
# WelfordState  — inlet lag-1 variant (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WelfordState:
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
# Drone  — identical logic to lawss_stage3_inlet.py; N_OBS updated to 19
# ─────────────────────────────────────────────────────────────────────────────

class Drone:
    def __init__(self, drone_id: int, position: np.ndarray,
                 rng: np.random.Generator):
        self.id       = drone_id
        self.position = position.astype(float).copy()
        self.velocity = np.zeros(3, dtype=float)
        self.rng      = rng

        self.home_position: np.ndarray = position.astype(float).copy()

        self.state: DroneState = DroneState.IDLE

        self.target_position:   Optional[np.ndarray] = None
        self.target_node_id:    Optional[int]        = None
        self.completed_node_id: Optional[int]        = None
        self.abandoned_node_id: Optional[int]        = None

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

        self._warm_X: Optional[np.ndarray] = None
        self._warm_U: Optional[np.ndarray] = None
        self._last_solve_ok: bool = True
        self._build_mpc()

    # ── MPC construction  ─────────────────────────────────────────────────────

    def _build_mpc(self) -> None:
        opti = ca.Opti()

        opti.solver(
            'ipopt',
            {'expand': True, 'print_time': 0},
            {
                'max_iter':              200,
                'tol':                   1e-3,
                'print_level':           0,
                'sb':                    'yes',
                'warm_start_init_point': 'yes',
            },
        )

        X = opti.variable(6, MPC_N + 1)
        U = opti.variable(3, MPC_N)

        p_init   = opti.parameter(6)
        p_target = opti.parameter(3)
        p_obs_p  = opti.parameter(3, N_OBS)
        p_obs_v  = opti.parameter(3, N_OBS)
        p_dist   = opti.parameter()
        p_v_bound = opti.parameter()          # dynamic velocity bound [m/s]

        # Initial state constraint
        opti.subject_to(X[:, 0] == p_init)

        # Horizon constraints
        for k in range(MPC_N):
            pk = X[:3, k]
            vk = X[3:, k]
            ak = U[:, k]

            opti.subject_to(X[:3, k + 1] == pk + vk * DT + 0.5 * ak * DT**2)
            opti.subject_to(X[3:, k + 1] == vk + ak * DT)

            opti.subject_to(opti.bounded(-p_v_bound, X[3:, k], p_v_bound))
            opti.subject_to(opti.bounded(-A_MAX, U[:, k],  A_MAX))

            if k > 0:
                for j in range(N_OBS):
                    p_obs_k = p_obs_p[:, j] + p_obs_v[:, j] * (k * DT)
                    opti.subject_to(ca.sumsqr(X[:3, k] - p_obs_k) >= D_MIN**2)

        # Terminal step: velocity bound + collision avoidance
        opti.subject_to(opti.bounded(-p_v_bound, X[3:, MPC_N], p_v_bound))
        for j in range(N_OBS):
            p_obs_N = p_obs_p[:, j] + p_obs_v[:, j] * (MPC_N * DT)
            opti.subject_to(ca.sumsqr(X[:3, MPC_N] - p_obs_N) >= D_MIN**2)

        prox_factor = Q_VEL_PROX / ca.fmax(p_dist, Q_VEL_PROX)

        cost = ca.MX(0)
        for k in range(MPC_N):
            cost += Q_STAGE * ca.sumsqr(X[:3, k] - p_target)
            cost += Q_VEL * prox_factor * ca.sumsqr(X[3:, k])
            force = MASS * U[:, k]
            cost += R_CTRL * ca.sumsqr(force)
        cost += Q_TERM * ca.sumsqr(X[:3, MPC_N] - p_target)
        cost += Q_VEL  * prox_factor * ca.sumsqr(X[3:, MPC_N])
        opti.minimize(cost)

        self._opti     = opti
        self._mpc_X    = X
        self._mpc_U    = U
        self._p_init   = p_init
        self._p_target = p_target
        self._p_obs_p  = p_obs_p
        self._p_obs_v  = p_obs_v
        self._p_dist   = p_dist
        self._p_v_bound = p_v_bound

    # ── MPC solve  ────────────────────────────────────────────────────────────

    def _solve_mpc(self, neighbours: List[Tuple[np.ndarray, np.ndarray]],) -> np.ndarray:
        state6 = np.hstack([self.position, self.velocity])
        self._opti.set_value(self._p_init,   state6)
        self._opti.set_value(self._p_target, self.target_position)

        dist_to_target = float(np.linalg.norm(self.position - self.target_position))
        self._opti.set_value(self._p_dist, dist_to_target)

        # Kinematic braking envelope: v_safe = sqrt(2 * a_max * d)
        # Buffer of 0.5 m distance and 0.5 m/s speed slack prevents the
        # constraint from becoming infeasible in the final approach.
        safe_v = float(np.sqrt(2.0 * A_MAX * max(0.0, dist_to_target - 0.5)))
        dyn_v_bound = min(V_MAX, safe_v + 0.5)
        self._opti.set_value(self._p_v_bound, dyn_v_bound)

        obs_p = np.empty((3, N_OBS), dtype=float)
        obs_v = np.zeros((3, N_OBS), dtype=float)
        for j in range(N_OBS):
            if j < len(neighbours):
                obs_p[:, j] = neighbours[j][0]
                obs_v[:, j] = neighbours[j][1]
            else:
                obs_p[:, j] = _OBS_SENTINEL
        self._opti.set_value(self._p_obs_p, obs_p)
        self._opti.set_value(self._p_obs_v, obs_v)

        if self._warm_X is not None and self._warm_U is not None:
            X_init = np.hstack([self._warm_X[:, 1:], self._warm_X[:, -1:]])
            U_init = np.hstack([self._warm_U[:, 1:], self._warm_U[:, -1:]])
        else:
            direction = self.target_position - self.position
            dist      = float(np.linalg.norm(direction))
            if dist > 1e-3:
                unit     = direction / dist
                cruise_v = unit * min(dyn_v_bound * 0.9, dist / (MPC_N * DT))
            else:
                unit     = np.zeros(3)
                cruise_v = np.zeros(3)
            X_init = np.zeros((6, MPC_N + 1))
            X_init[:3, 0] = self.position
            X_init[3:, 0] = self.velocity
            for k in range(1, MPC_N + 1):
                X_init[:3, k] = self.position + cruise_v * k * DT
                X_init[3:, k] = cruise_v
            U_init = np.zeros((3, MPC_N))
        self._opti.set_initial(self._mpc_X, X_init)
        self._opti.set_initial(self._mpc_U, U_init)

        try:
            sol = self._opti.solve()
            X_val = np.array(sol.value(self._mpc_X))
            U_val = np.array(sol.value(self._mpc_U))
            self._last_solve_ok = True
        except Exception:
            try:
                X_val = np.array(self._opti.debug.value(self._mpc_X))
                U_val = np.array(self._opti.debug.value(self._mpc_U))
            except Exception:
                self._last_solve_ok = False
                return np.zeros(3)
            self._last_solve_ok = False

        self._warm_X = X_val
        self._warm_U = U_val

        return np.clip(U_val[:, 0], -A_MAX, A_MAX)

    # ── update_trajectory  ────────────────────────────────────────────────────

    def update_trajectory(
        self,
        neighbours: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        if self.target_position is None:
            return
        if neighbours is None:
            neighbours = []

        accel = self._solve_mpc(neighbours)
        self.position = (self.position
                         + self.velocity * DT
                         + 0.5 * accel * DT**2)
        self.velocity = np.clip(self.velocity + accel * DT, -V_MAX, V_MAX)

        dist  = float(np.linalg.norm(self.position - self.target_position))
        speed = float(np.linalg.norm(self.velocity))
        if dist < ARRIVAL_DIST and speed < ARRIVAL_SPEED:
            self.position = self.target_position.copy()
            self.velocity = np.zeros(3)
            if self.state == DroneState.TRANSIT:
                self._begin_sampling()
            elif self.state == DroneState.RTH:
                self.state            = DroneState.IDLE
                self.battery_depleted = True
                self.target_position  = None

    # ── Assignment & sampling lifecycle  ─────────────────────────────────────

    def assign_target(self, target_position: np.ndarray,
                      target_node_id: int) -> None:
        self.target_position = target_position.copy()
        self.target_node_id  = target_node_id
        self.state           = DroneState.TRANSIT
        self._warm_X         = None
        self._warm_U         = None

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
                cov_lag     = s.L_mxy - s.L_mux * s.L_muy
                var_lag     = max(s.L_mx2 - s.L_mux**2, 1e-12)
                rho1        = np.clip(cov_lag / var_lag, 1e-6, 1.0 - 1e-6)
                T_new       = -DT / np.log(rho1)
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
        s = self.stats
        self.last_result = {
            "node_id":   self.target_node_id,
            "position":  self.position.copy(),
            "mean_est":  s.wf_mean,
            "T_int_est": s.T_int_est,
            "n_samples": self.samples_this_run,
            "elapsed_s": self.samples_this_run * DT,
        }
        self.completed_node_id = self.target_node_id
        self.target_position   = None
        self.target_node_id    = None
        self.state             = DroneState.IDLE

    def tick(
        self,
        u_prev_store: list,
        neighbours: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        if self.battery_depleted:
            return

        self.battery_remaining_s -= DT
        if self.battery_remaining_s <= 0.0:
            self.battery_remaining_s = 0.0
            self.battery_depleted    = True
            self.state               = DroneState.IDLE
            return

        # RTH trigger
        if (self.battery_remaining_s <= BATTERY_CAPACITY_S * RTH_THRESHOLD
                and self.state not in (DroneState.RTH, DroneState.IDLE)):
            if self.target_node_id is not None:
                self.abandoned_node_id = self.target_node_id
                self.target_node_id    = None
            self.state           = DroneState.RTH
            self.target_position = self.home_position.copy()
            self._warm_X         = None
            self._warm_U         = None

        if self.state == DroneState.IDLE:
            pass

        elif self.state in (DroneState.TRANSIT, DroneState.RTH):
            self.update_trajectory(neighbours=neighbours)

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

            # 3-minute hard cap
            if self.samples_this_run * DT >= MAX_SAMPLING_TIME_S:
                self._finish_run()
                return

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
    Mega-scale inlet environment: 20 drones, 250 random 3-D nodes.

    Drone spawn layout:  4 rows × 5 columns, 4 m spacing in X and Y.
    This gives a minimum inter-drone separation of 4.0 m > D_MIN = 3.0 m,
    so the MPC collision constraints are satisfied from tick 0.

    Grid positions:
        col 0..4  →  x = 0, 4, 8, 12, 16
        row 0..3  →  y = 0, 4, 8, 12
        z = 0 (ground, pre-takeoff)

    All other logic (Hungarian dispatcher, RTH unlock, neighbour snapshot)
    is identical to lawss_stage3_inlet.py.
    """

    # Spawn grid parameters
    _SPAWN_COLS:    int   = 5
    _SPAWN_ROWS:    int   = 4
    _SPAWN_SPACING: float = 4.0   # [m] — must be > D_MIN = 3.0

    def __init__(self, seed: int = 42):
        self.tick_count: int  = 0
        self.elapsed_s: float = 0.0

        master_rng  = np.random.default_rng(seed)
        drone_seeds = master_rng.integers(0, 2**31, size=N_DRONES)

        # 250 random targets uniformly distributed in [0, 100]³
        self.target_positions: np.ndarray = master_rng.uniform(
            0.0, 100.0, size=(N_TARGETS, 3)
        ).astype(float)

        self.target_measured: np.ndarray = np.zeros(N_TARGETS, dtype=bool)
        self.target_locked:   np.ndarray = np.zeros(N_TARGETS, dtype=bool)

        # 4 × 5 grid spawn positions on the ground plane
        spawn_positions = []
        for row in range(self._SPAWN_ROWS):
            for col in range(self._SPAWN_COLS):
                spawn_positions.append(np.array([
                    col * self._SPAWN_SPACING,
                    row * self._SPAWN_SPACING,
                    0.0,
                ], dtype=float))
        # Safety check: exactly N_DRONES = 20 positions
        assert len(spawn_positions) == N_DRONES, (
            f"Spawn grid mismatch: {len(spawn_positions)} positions "
            f"for {N_DRONES} drones"
        )

        self.drones: list[Drone] = [
            Drone(
                drone_id=i,
                position=spawn_positions[i],
                rng=np.random.default_rng(int(drone_seeds[i])),
            )
            for i in range(N_DRONES)
        ]

        self._u_prev: list[list[float]] = [[U_MEAN] for _ in range(N_DRONES)]
        self._dispatch_calls:   int = 0
        self._assignments_made: int = 0

    # ── Completion polling  ───────────────────────────────────────────────────

    def _collect_completions(self) -> None:
        for drone in self.drones:
            # Unlock abandoned nodes (RTH triggered mid-task)
            if drone.abandoned_node_id is not None:
                node_id = drone.abandoned_node_id
                if 0 <= node_id < N_TARGETS:
                    if not self.target_measured[node_id]:
                        self.target_locked[node_id] = False
                drone.abandoned_node_id = None

            node_id = drone.completed_node_id
            if node_id is None:
                continue
            if 0 <= node_id < N_TARGETS:
                self.target_measured[node_id] = True
                self.target_locked[node_id]   = False
            drone.completed_node_id = None

    # ── Hungarian dispatcher  ─────────────────────────────────────────────────

    def dispatch(self) -> None:
        self._dispatch_calls += 1

        idle_ids = [
            d.id for d in self.drones
            if d.state == DroneState.IDLE and not d.battery_depleted
        ]
        if not idle_ids:
            return

        free_node_ids = [
            i for i in range(N_TARGETS)
            if not self.target_measured[i] and not self.target_locked[i]
        ]
        if not free_node_ids:
            return

        idle_positions = np.array([self.drones[i].position for i in idle_ids])
        node_positions = self.target_positions[free_node_ids]

        diff        = idle_positions[:, np.newaxis, :] - node_positions[np.newaxis, :, :]
        cost_matrix = np.linalg.norm(diff, axis=-1)

        drone_rows, node_cols = linear_sum_assignment(cost_matrix)

        for dr, nc in zip(drone_rows, node_cols):
            drone_id = idle_ids[dr]
            node_id  = free_node_ids[nc]
            self.drones[drone_id].assign_target(
                target_position=self.target_positions[node_id],
                target_node_id=node_id,
            )
            self.target_locked[node_id] = True
            self._assignments_made += 1

    # ── Simulation clock  ─────────────────────────────────────────────────────

    def step(self) -> None:
        """
        One 10 Hz tick:
          1. Snapshot all (position, velocity) pairs BEFORE any drone moves.
          2. Advance each drone with its N_DRONES-1 neighbour snapshot.
          3. Collect completions, dispatch.
          4. Advance clock.
        """
        snapshot: List[Tuple[np.ndarray, np.ndarray]] = [
            (d.position.copy(), d.velocity.copy()) for d in self.drones
        ]

        for i, drone in enumerate(self.drones):
            neighbours = [
                snap for j, (snap, d) in enumerate(zip(snapshot, self.drones))
                if j != i and d.state in (DroneState.TRANSIT, DroneState.RTH)
            ]
            drone.tick(self._u_prev[i], neighbours=neighbours)

        self._collect_completions()
        self.dispatch()

        self.tick_count += 1
        self.elapsed_s  += DT

    def run(self, duration_s: float) -> None:
        """Blocking loop — headless testing only."""
        for _ in range(int(duration_s / DT)):
            self.step()

    # ── Properties  ───────────────────────────────────────────────────────────

    @property
    def n_measured(self) -> int:
        return int(self.target_measured.sum())

    @property
    def all_measured(self) -> bool:
        return bool(self.target_measured.all())

    # ── Reporting  ────────────────────────────────────────────────────────────

    def summary(self, show_completed: bool = False) -> str:
        lines = [
            f"t = {self.elapsed_s:.1f} s  |  tick {self.tick_count}  |  "
            f"nodes measured: {self.n_measured}/{N_TARGETS}  |  "
            f"assignments so far: {self._assignments_made}",
            "",
            f"  {'ID':>3}  {'State':>10}  {'Node':>5}  "
            f"{'Samples':>8}  {'Mean est':>9}  {'T_int':>7}  "
            f"{'Battery':>7}  {'MPC':>5}",
            "  " + "─" * 68,
        ]

        for d in self.drones:
            s      = d.stats
            node   = str(d.target_node_id) if d.target_node_id is not None else "—"
            mean   = f"{s.wf_mean:.3f}" if s.wf_n > 0 else "—"
            tint   = f"{s.T_int_est:.2f}" if s.L_n > LAG1_WARMUP else "prior"
            batt   = f"{d.battery_remaining_s:.0f}s"
            mpc_ok = "ok" if d._last_solve_ok else "fb"
            lines.append(
                f"  {d.id:>3}  {d.state.name:>10}  {node:>5}  "
                f"{d.samples_this_run:>8}  {mean:>9}  {tint:>7}  "
                f"{batt:>7}  {mpc_ok:>5}"
            )

        if show_completed:
            completed = [d for d in self.drones if d.last_result is not None]
            if completed:
                lines += ["", "  Completed runs (most recent per drone):"]
                for d in sorted(completed, key=lambda x: x.last_result["elapsed_s"]):
                    r   = d.last_result
                    err = abs(r["mean_est"] - U_MEAN) / U_MEAN * 100
                    lines.append(
                        f"    Drone {d.id:>2}  node {r['node_id']:>3}  "
                        f"mean={r['mean_est']:.4f} m/s  "
                        f"err={err:.2f}%  "
                        f"T={r['elapsed_s']:.1f}s"
                    )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Quick correctness check — runs a short burst, prints a snapshot,
    then verifies basic invariants.

    Full 18-min runs are handled by batch_runner_inlet_mega.py.
    """
    print("=" * 72)
    print("  LAWSS Inlet Mega-Scale — Smoke Test")
    print(f"  {N_DRONES} drones  |  {N_TARGETS} random 3-D nodes in [0,100]³")
    print(f"  Battery: {BATTERY_CAPACITY_S:.0f} s ({BATTERY_CAPACITY_S/60:.0f} min)  "
          f"|  RTH threshold: {RTH_THRESHOLD*100:.0f}%  "
          f"|  Hard cap: {MAX_SAMPLING_TIME_S:.0f} s")
    print(f"  MPC horizon N={MPC_N}, N_OBS={N_OBS}, d_min={D_MIN} m")
    print(f"  Spawn grid: {Environment._SPAWN_ROWS}×{Environment._SPAWN_COLS}, "
          f"spacing={Environment._SPAWN_SPACING} m")
    print()
    print("  Building MPC solvers …")
    t_build = time.perf_counter()
    env = Environment(seed=0)
    print(f"  Build time: {time.perf_counter() - t_build:.2f} s\n")

    # Run 120 s (short validation burst)
    t0 = time.perf_counter()
    for _ in range(int(120 / DT)):
        env.step()
    wall = time.perf_counter() - t0

    print(env.summary())
    print(f"\n  Wall time for 120 s sim: {wall:.2f} s  "
          f"({120/wall:.0f}× real-time)")

    # Invariant checks
    print(f"\n  Nodes measured: {env.n_measured}")
    still_locked_and_measured = [
        i for i in range(N_TARGETS)
        if env.target_measured[i] and env.target_locked[i]
    ]
    assert not still_locked_and_measured, \
        f"Measured nodes still locked: {still_locked_and_measured}"
    print("  Invariant: no measured node still locked  ✓")
    print("=" * 72)


if __name__ == "__main__":
    _smoke_test()
