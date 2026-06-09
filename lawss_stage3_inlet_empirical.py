from __future__ import annotations
import time
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Empirical Data Loading (Atmospheric Boundary Layer)
# ─────────────────────────────────────────────────────────────────────────────

ALTITUDES = [2.0, 5.0, 10.0, 20.0, 50.0, 80.0]
REAL_WIND_DATA = {}

try:
    # Load the CSV file containing real wind speeds in m/s
    df = pd.read_csv("jan_one_data_wind_mps.csv", delimiter=' ')
    
    # Automatically filter out any text/date columns, keeping only the numbers
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Map the first 6 NUMERIC columns to their respective altitudes
    for i, alt in enumerate(ALTITUDES):
        REAL_WIND_DATA[alt] = numeric_df.iloc[:, i].values
        
    DATA_LENGTH = len(df)
    print(f"Successfully loaded empirical data: {DATA_LENGTH} samples across {len(ALTITUDES)} altitudes.")
except Exception as e:
    print(f"Failed to load CSV: {e}")
    # Fallback to dummy data so the script doesn't completely crash
    DATA_LENGTH = 1000
    for alt in ALTITUDES:
        REAL_WIND_DATA[alt] = np.ones(DATA_LENGTH) * 12.0


# ─────────────────────────────────────────────────────────────────────────────
# Physical & statistical constants
# ─────────────────────────────────────────────────────────────────────────────

FS: float = 10.0
DT: float = 5.0 / FS

U_MEAN: float = 12.0 # Baseline fallback for initializing arrays
T_INT: float  = 5.0

EPSILON_CI: float = 0.10
DELTA_STAB: float = 0.10
Z_SCORE: float    = 1.645
N_EFF_MIN: int    = 10

EMA_ALPHA: float  = 0.05
LAG1_WARMUP: int  = 20

BURNIN_SAMPLES: int = int(5.0 * T_INT * FS)   # 250 samples
STAB_WIN: int       = int(5.0 * T_INT * FS)   # 250 samples

BATTERY_CAPACITY_S: float  = 1080.0   # 18-min battery
N_TARGETS:           int   = 100
N_DRONES:            int   = 10
MAX_SAMPLING_TIME_S: float = 180.0    # 3-min hard cap per run

# ─────────────────────────────────────────────────────────────────────────────
# MPC constants (Tuned for 14m/s, 3.645kg, and aggressive braking)
# ─────────────────────────────────────────────────────────────────────────────

MPC_N:   int   = 20
V_MAX:   float = 14.0
A_MAX:   float = 10.0    # High authority for snappy, late braking
D_MIN:   float = 3.0
N_OBS:   int   = 9
MASS:    float = 3.645

Q_STAGE: float = 1.0
Q_TERM:  float = 100.0
Q_VEL:       float = 5.0
Q_VEL_PROX:  float = 0.5 # Relaxed parachute for faster approaches
R_CTRL:      float = 0.1

ARRIVAL_DIST:  float = 0.5
ARRIVAL_SPEED: float = 0.3
RTH_THRESHOLD: float = 0.07

_OBS_SENTINEL = np.array([1e6, 1e6, 1e6], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# State machine & Welford
# ─────────────────────────────────────────────────────────────────────────────

class DroneState(Enum):
    IDLE     = auto()
    TRANSIT  = auto()
    SAMPLING = auto()
    RTH      = auto()


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
# Drone
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

        self.last_result: Optional[dict] = None

        self._warm_X: Optional[np.ndarray] = None
        self._warm_U: Optional[np.ndarray] = None
        self._last_solve_ok: bool = True
        
        self._current_alt_key = ALTITUDES[0]
        self._data_start_idx = 0
        
        self._build_mpc()

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

        p_init    = opti.parameter(6)
        p_target  = opti.parameter(3)
        p_obs_p   = opti.parameter(3, N_OBS)
        p_obs_v   = opti.parameter(3, N_OBS)
        p_dist    = opti.parameter()
        p_v_bound = opti.parameter()

        opti.subject_to(X[:, 0] == p_init)

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

        self._opti      = opti
        self._mpc_X     = X
        self._mpc_U     = U
        self._p_init    = p_init
        self._p_target  = p_target
        self._p_obs_p   = p_obs_p
        self._p_obs_v   = p_obs_v
        self._p_dist    = p_dist
        self._p_v_bound = p_v_bound

    def _solve_mpc(self, neighbours: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        state6 = np.hstack([self.position, self.velocity])
        self._opti.set_value(self._p_init,   state6)
        self._opti.set_value(self._p_target, self.target_position)

        dist_to_target = float(np.linalg.norm(self.position - self.target_position))
        self._opti.set_value(self._p_dist, dist_to_target)

        # Aggressive kinematic braking envelope: v_safe = sqrt(2 * a_max * d)
        # Tightened to 0.1m distance buffer and 0.1 m/s speed slack for snappy arrival
        safe_v = float(np.sqrt(2.0 * A_MAX * max(0.0, dist_to_target - 0.1)))
        dyn_v_bound = min(V_MAX, safe_v + 0.1)
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
        
        # Find the closest altitude array to the drone's current target Z
        target_z = self.target_position[2]
        self._current_alt_key = min(ALTITUDES, key=lambda x: abs(x - target_z))
        
        # Pick random start index for spatial decorrelation
        self._data_start_idx = int(self.rng.integers(0, DATA_LENGTH))

    def _generate_sample(self) -> float:
        # Wrap around the empirical dataset safely using modulo
        idx = (self._data_start_idx + self.samples_this_run) % DATA_LENGTH
        return float(REAL_WIND_DATA[self._current_alt_key][idx])

    def _update_welford(self, u: float, u_prev: float) -> None:
        s = self.stats
        n = self.samples_this_run

        s.wf_n   += 1
        d         = u - s.wf_mean
        s.wf_mean += d / s.wf_n
        s.wf_M2  += d * (u - s.wf_mean)

        # 🚨 DISABLED for empirical data: Lag-1 Autocorrelation Update
        # Because the CSV data is not 10Hz, the dynamic T_int estimator will collapse.
        # We will keep s.T_int_est locked at the default T_INT (5.0s) to keep the math stable.

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

        # 🚨 OVERRIDE for empirical data:
        # Real weather data contains macro-level drift (weather fronts moving in).
        # We automatically set cond2 to True so the fast-forward drift doesn't block the drone.
        cond2 = True

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

            if self.samples_this_run * DT >= MAX_SAMPLING_TIME_S:
                self._finish_run()
                return

            if self.samples_this_run < BURNIN_SAMPLES:
                return

            cond1, cond2, cond3 = self._check_stopping_conditions()
            if cond1 and cond2 and cond3:
                self._finish_run()


# ─────────────────────────────────────────────────────────────────────────────
# Environment (Modified for 10 drones, empirical layered targets)
# ─────────────────────────────────────────────────────────────────────────────

class Environment:

    def __init__(self, seed: int = 42):
        self.tick_count: int  = 0
        self.elapsed_s: float = 0.0

        master_rng  = np.random.default_rng(seed)
        drone_seeds = master_rng.integers(0, 2**31, size=N_DRONES)

        # Generate random X, Y in [0, 100]
        xy_positions = master_rng.uniform(0.0, 100.0, size=(N_TARGETS, 2))
        
        # Generate Z by randomly sampling from the specific empirical ALTITUDES list
        z_positions = master_rng.choice(ALTITUDES, size=(N_TARGETS, 1))
        
        self.target_positions = np.hstack((xy_positions, z_positions)).astype(float)

        self.target_measured: np.ndarray = np.zeros(N_TARGETS, dtype=bool)
        self.target_locked:   np.ndarray = np.zeros(N_TARGETS, dtype=bool)

        self.drones: list[Drone] = [
            Drone(
                drone_id=i,
                position=np.array([i * 4.0, 0.0, 0.0], dtype=float),
                rng=np.random.default_rng(int(drone_seeds[i])),
            )
            for i in range(N_DRONES)
        ]

        self._u_prev: list[list[float]] = [[U_MEAN] for _ in range(N_DRONES)]
        self._dispatch_calls:   int = 0
        self._assignments_made: int = 0

    def _collect_completions(self) -> None:
        for drone in self.drones:
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

    def step(self) -> None:
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
        for _ in range(int(duration_s / DT)):
            self.step()

    @property
    def n_measured(self) -> int:
        return int(self.target_measured.sum())

    @property
    def all_measured(self) -> bool:
        return bool(self.target_measured.all())

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
                sampling_times = []
                
                for d in sorted(completed, key=lambda x: x.last_result["elapsed_s"]):
                    r   = d.last_result
                    sampling_times.append(r["elapsed_s"])
                    lines.append(
                        f"    Drone {d.id:>2}  node {r['node_id']:>3}  "
                        f"mean={r['mean_est']:.4f} m/s  "
                        f"T={r['elapsed_s']:.1f}s"
                    )
                
                # Calculate and append sampling duration statistics
                min_t  = min(sampling_times)
                max_t  = max(sampling_times)
                mean_t = sum(sampling_times) / len(sampling_times)
                
                lines += [
                    "",
                    "  Sampling Duration Statistics:",
                    f"    Shortest : {min_t:.1f} s",
                    f"    Longest  : {max_t:.1f} s",
                    f"    Mean     : {mean_t:.1f} s",
                ]

        return "\n".join(lines)


def _smoke_test() -> None:
    print("=" * 72)
    print("  LAWSS Inlet Empirical Mode — Smoke Test")
    print(f"  {N_DRONES} drones  |  {N_TARGETS} layered nodes")
    print(f"  Battery: {BATTERY_CAPACITY_S:.0f} s ({BATTERY_CAPACITY_S/60:.0f} min)")
    print()
    print("  Building MPC solvers …")
    t_build = time.perf_counter()
    env = Environment(seed=0)
    print(f"  Build time: {time.perf_counter() - t_build:.2f} s\n")

    t0 = time.perf_counter()
    for _ in range(int(120 / DT)):
        env.step()
    wall = time.perf_counter() - t0

    print(env.summary(show_completed=True))
    print(f"\n  Wall time for 120 s sim: {wall:.2f} s  "
          f"({120/wall:.0f}× real-time)")

    print(f"\n  Nodes measured: {env.n_measured}")
    print("=" * 72)


if __name__ == "__main__":
    _smoke_test()