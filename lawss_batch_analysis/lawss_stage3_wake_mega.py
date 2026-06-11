from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import casadi as ca
import numpy as np
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Simulation clock
# ─────────────────────────────────────────────────────────────────────────────

FS_SAMPLE: float = 10.0
DT_SAMPLE: float = 1.0 / FS_SAMPLE
DT_KIN: float    = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Wake signal parameters  (injected via os.environ for sweeps)
# ─────────────────────────────────────────────────────────────────────────────

U_INF:   float = 15.0    # free-stream wind speed         [m/s]
U_WAKE:  float = 9.0     # mean velocity in the wake          [m/s]  (~0.6·U_inf)

# INJECTED VARIABLES FOR HPC SWEEP
I_U_RAND:   float = float(os.environ.get("LAWSS_I_U", "0.22"))
I_U_SHED:   float = float(os.environ.get("LAWSS_I_U", "0.15"))
T_INT_RAND: float = float(os.environ.get("LAWSS_T_INT", "5.0"))

D:  float = 20.0          # characteristic building dimension [m]
ST: float = 0.10          # Strouhal number                   [-]

# Derived signal parameters
SIGMA_RAND: float = I_U_RAND * U_WAKE
SIGMA_SHED: float = I_U_SHED * U_WAKE
F_SHED:     float = ST * U_INF / D                   # vortex shedding freq [Hz]
T_SHED:     float = 1.0 / F_SHED                     # shedding period      [s]

# AR(1) coefficients for the random turbulence component
PHI_RAND:   float = np.exp(-DT_SAMPLE / T_INT_RAND)
SIGMA_EPS:  float = SIGMA_RAND * np.sqrt(1.0 - PHI_RAND**2)

# Effective integral time scale (random variance fraction of total)
ALPHA_FRAC:   float = SIGMA_RAND**2 / (SIGMA_RAND**2 + SIGMA_SHED**2)
T_INT_EFF:    float = ALPHA_FRAC * T_INT_RAND        # what ACF estimator converges to

# ─────────────────────────────────────────────────────────────────────────────
# Stopping-criterion thresholds  (relaxed for wake — UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

EPSILON_CI:  float = 0.10    # Cond 1: Z·σ_Ū/Ū  < 10 %
DELTA_STAB:  float = 0.05    # Cond 2: mean drift < 2 %
Z_SCORE:     float = 1.645   # 90 % confidence
N_EFF_MIN:   int   = 15      # Cond 3: min independent samples
N_SHED_MIN:  int   = 5       # Cond 4: min complete shedding cycles

# ─────────────────────────────────────────────────────────────────────────────
# ACF estimator settings  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

N_ACF_UPDATE:  int   = 50
ACF_EMA_ALPHA: float = 0.3

ACF_BUF_LEN:  int = max(int(4.0 * T_SHED * FS_SAMPLE), 400)
ACF_MAX_LAG:  int = max(int(1.5 * T_SHED * FS_SAMPLE), 150)

BURNIN_SAMPLES: int = max(
    int(5.0 * T_INT_EFF * FS_SAMPLE),
    int(N_SHED_MIN * T_SHED * FS_SAMPLE),
)

STAB_WIN: int = max(int(8.0 * T_INT_EFF * FS_SAMPLE), 2)

PERSIST_SAMPLES: int = int(T_SHED * FS_SAMPLE)

# ─────────────────────────────────────────────────────────────────────────────
# Fleet & operational constants  ── SCALED UP FOR MEGA RUN ──
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_CAPACITY_S:  float = 1080.0
N_DRONES:  int = 70
N_TARGETS: int = 70
MAX_SAMPLING_TIME_S: float = 300.0

# ─────────────────────────────────────────────────────────────────────────────
# MPC constants  ── N_OBS MUST equal N_DRONES - 1 ──
# ─────────────────────────────────────────────────────────────────────────────

MPC_N:    int   = 20
V_MAX:    float = 14.0    # max velocity per axis [m/s]
A_MAX:    float = 8.0
D_MIN:    float = 3.0
N_OBS:    int   = N_DRONES - 1   # Dynamically sizes based on N_DRONES
MASS:     float = 3.645   # [kg] Drone mass
Q_STAGE:  float = 1.0
Q_TERM:   float = 100.0
Q_VEL:       float = 5.0
Q_VEL_PROX:  float = 5.0
R_CTRL:      float = 0.1

ARRIVAL_DIST:  float = 0.5
ARRIVAL_SPEED: float = 0.3
RTH_THRESHOLD: float = 0.07

_OBS_SENTINEL = np.array([1e6, 1e6, 1e6], dtype=float)

# ─────────────────────────────────────────────────────────────────────────────
# Spawn grid layout (Dynamically factored to equal N_DRONES)
# ─────────────────────────────────────────────────────────────────────────────

_SPAWN_COLS: int   = 2
_SPAWN_ROWS: int   = N_DRONES // _SPAWN_COLS
_SPAWN_STEP: float = 4.0   # metres between adjacent drones in both axes

assert _SPAWN_ROWS * _SPAWN_COLS == N_DRONES, (
    f"Spawn grid layout ({_SPAWN_ROWS}x{_SPAWN_COLS}) must equal N_DRONES ({N_DRONES})."
)


def _spawn_position(drone_id: int) -> np.ndarray:
    """Return the (x, y, 0) ground-level spawn position for drone_id."""
    row = drone_id // _SPAWN_COLS
    col = drone_id  % _SPAWN_COLS
    return np.array([col * _SPAWN_STEP,
                     row * _SPAWN_STEP,
                     0.0], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# ACF zero-crossing integrator
# ─────────────────────────────────────────────────────────────────────────────

def acf_zero_crossing_T_int(buf: np.ndarray, dt: float, max_lag: int) -> float:
    n = len(buf)
    if n < max_lag + 2:
        return np.nan

    x = buf - buf.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.nan

    lags = np.arange(max_lag + 1)
    acf  = np.array([np.mean(x[:n - k] * x[k:]) for k in lags]) / var

    sign_changes = np.where(np.diff(np.sign(acf)))[0]
    if len(sign_changes) == 0:
        return np.nan

    k0       = sign_changes[0]
    tau_grid = lags[:k0 + 1] * dt
    T_est    = float(np.trapezoid(acf[:k0 + 1], tau_grid))

    return max(T_est, dt)


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class DroneState(Enum):
    IDLE     = auto()
    TRANSIT  = auto()
    SAMPLING = auto()
    RTH      = auto()


# ─────────────────────────────────────────────────────────────────────────────
# WelfordState
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WelfordState:
    wf_n: int      = 0
    wf_mean: float = 0.0
    wf_M2: float   = 0.0

    acf_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(ACF_BUF_LEN, dtype=float))
    buf_idx: int = 0

    T_int_eff_cur: float = T_INT_EFF

    stab_hist: np.ndarray = field(
        default_factory=lambda: np.full(
            int(MAX_SAMPLING_TIME_S * FS_SAMPLE) + STAB_WIN + 10, np.nan))
    stab_hist_ptr: int = 0

    drift_hist: np.ndarray = field(
        default_factory=lambda: np.full(PERSIST_SAMPLES, np.inf))
    drift_ptr: int = 0

    def reset(self) -> None:
        self.wf_n = 0
        self.wf_mean = 0.0
        self.wf_M2 = 0.0
        self.acf_buffer[:] = 0.0
        self.buf_idx = 0
        self.T_int_eff_cur = T_INT_EFF
        self.stab_hist[:] = np.nan
        self.stab_hist_ptr = 0
        self.drift_hist[:] = np.inf
        self.drift_ptr = 0


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

        self.phase0: float = rng.uniform(0.0, 2.0 * np.pi)

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

        _buf = int(BATTERY_CAPACITY_S * FS_SAMPLE)
        self.hist_raw_velocity: np.ndarray = np.full(_buf, np.nan)
        self.hist_running_mean: np.ndarray = np.full(_buf, np.nan)
        self.hist_ci_rel: np.ndarray       = np.full(_buf, np.nan)
        self.hist_T_eff: np.ndarray        = np.full(_buf, np.nan)
        self._hist_ptr: int                = 0

        self._u_rand_prime: float = 0.0

        self.last_result: Optional[dict] = None

        self._warm_X: Optional[np.ndarray] = None
        self._warm_U: Optional[np.ndarray] = None
        self._last_solve_ok: bool = True
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

        p_init   = opti.parameter(6)
        p_target = opti.parameter(3)
        p_obs_p  = opti.parameter(3, N_OBS)
        p_obs_v  = opti.parameter(3, N_OBS)
        p_dist   = opti.parameter()
        p_v_bound = opti.parameter()

        opti.subject_to(X[:, 0] == p_init)

        for k in range(MPC_N):
            pk = X[:3, k]
            vk = X[3:, k]
            ak = U[:, k]

            opti.subject_to(X[:3, k + 1] == pk + vk * DT_KIN + 0.5 * ak * DT_KIN**2)
            opti.subject_to(X[3:, k + 1] == vk + ak * DT_KIN)

            opti.subject_to(opti.bounded(-p_v_bound, X[3:, k], p_v_bound))
            opti.subject_to(opti.bounded(-A_MAX, U[:, k],  A_MAX))

            if k > 0:
                for j in range(N_OBS):
                    p_obs_k = p_obs_p[:, j] + p_obs_v[:, j] * (k * DT_KIN)
                    opti.subject_to(ca.sumsqr(X[:3, k] - p_obs_k) >= D_MIN**2)

        opti.subject_to(opti.bounded(-p_v_bound, X[3:, MPC_N], p_v_bound))
        for j in range(N_OBS):
            p_obs_N = p_obs_p[:, j] + p_obs_v[:, j] * (MPC_N * DT_KIN)
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

    def _solve_mpc(self, neighbours: List[Tuple[np.ndarray, np.ndarray]],) -> np.ndarray:
        state6 = np.hstack([self.position, self.velocity])
        self._opti.set_value(self._p_init,   state6)
        self._opti.set_value(self._p_target, self.target_position)

        dist_to_target = float(np.linalg.norm(self.position - self.target_position))
        self._opti.set_value(self._p_dist, dist_to_target)

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
                cruise_v = unit * min(dyn_v_bound * 0.9, dist / (MPC_N * DT_KIN))
            else:
                unit     = np.zeros(3)
                cruise_v = np.zeros(3)
            X_init = np.zeros((6, MPC_N + 1))
            X_init[:3, 0] = self.position
            X_init[3:, 0] = self.velocity
            for k in range(1, MPC_N + 1):
                X_init[:3, k] = self.position + cruise_v * k * DT_KIN
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
                         + self.velocity * DT_KIN
                         + 0.5 * accel * DT_KIN**2)
        self.velocity = np.clip(self.velocity + accel * DT_KIN, -V_MAX, V_MAX)

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
        self._u_rand_prime    = 0.0

    def _generate_sample(self) -> float:
        eps                = self.rng.normal(0.0, SIGMA_EPS)
        self._u_rand_prime = PHI_RAND * self._u_rand_prime + eps
        u_rand = self._u_rand_prime

        t_local = self.samples_this_run * DT_SAMPLE
        u_shed  = SIGMA_RAND * np.sqrt(2.0) * np.sin(
            2.0 * np.pi * F_SHED * t_local + self.phase0
        )
        return U_WAKE + u_rand + u_shed

    def _update_welford(self, u: float) -> None:
        s = self.stats

        s.wf_n   += 1
        d         = u - s.wf_mean
        s.wf_mean += d / s.wf_n
        s.wf_M2  += d * (u - s.wf_mean)

        s.stab_hist[s.stab_hist_ptr] = s.wf_mean
        s.stab_hist_ptr += 1

        s.acf_buffer[s.buf_idx % ACF_BUF_LEN] = u
        s.buf_idx += 1

        if (s.buf_idx >= ACF_BUF_LEN
                and self.samples_this_run > 0
                and self.samples_this_run % N_ACF_UPDATE == 0):
            start   = s.buf_idx % ACF_BUF_LEN
            ordered = np.concatenate([s.acf_buffer[start:], s.acf_buffer[:start]])
            T_new   = acf_zero_crossing_T_int(ordered, DT_SAMPLE, ACF_MAX_LAG)
            if np.isfinite(T_new) and T_new > 0.0:
                s.T_int_eff_cur = (
                    ACF_EMA_ALPHA * T_new
                    + (1.0 - ACF_EMA_ALPHA) * s.T_int_eff_cur
                )

    def _check_stopping_conditions(self) -> tuple[bool, bool, bool, bool]:
        s         = self.stats
        current_T = s.wf_n * DT_SAMPLE
        var_u     = s.wf_M2 / max(s.wf_n - 1, 1)

        sigma_Ubar = np.sqrt(
            max(2.0 * var_u * s.T_int_eff_cur / current_T, 0.0))
        ci_rel = Z_SCORE * sigma_Ubar / max(abs(s.wf_mean), 1e-9)
        cond1  = ci_rel < EPSILON_CI

        n_hist = s.stab_hist_ptr
        oldest = s.stab_hist[n_hist - STAB_WIN] if n_hist >= STAB_WIN else np.nan
        drift  = (abs(s.wf_mean - oldest) / max(abs(s.wf_mean), 1e-9)
                  if np.isfinite(oldest) else np.inf)

        s.drift_hist[s.drift_ptr % PERSIST_SAMPLES] = drift
        s.drift_ptr += 1
        cond2 = bool(np.all(s.drift_hist < DELTA_STAB))

        N_eff = current_T / (2.0 * s.T_int_eff_cur)
        cond3 = N_eff >= N_EFF_MIN

        cond4 = (current_T / T_SHED) >= N_SHED_MIN

        return cond1, cond2, cond3, cond4

    def _record_history(self, u: float, ci_rel: float) -> None:
        ptr = self._hist_ptr % len(self.hist_raw_velocity)
        self.hist_raw_velocity[ptr] = u
        self.hist_running_mean[ptr] = self.stats.wf_mean
        self.hist_ci_rel[ptr]       = ci_rel
        self.hist_T_eff[ptr]        = self.stats.T_int_eff_cur
        self._hist_ptr += 1

    def _finish_run(self) -> None:
        s = self.stats
        self.last_result = {
            "node_id":       self.target_node_id,
            "position":      self.position.copy(),
            "mean_est":      s.wf_mean,
            "T_int_eff_est": s.T_int_eff_cur,
            "n_samples":     self.samples_this_run,
            "elapsed_s":     self.samples_this_run * DT_SAMPLE,
        }
        self.completed_node_id = self.target_node_id
        self.target_position   = None
        self.target_node_id    = None
        self.state             = DroneState.IDLE

    def tick(
        self,
        neighbours: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        if self.battery_depleted:
            return

        self.battery_remaining_s -= DT_KIN
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
            # Sub-step the sampling to generate 10Hz measurements throughout the 1s kinematic interval
            num_samples = int(DT_KIN / DT_SAMPLE)
            for _ in range(num_samples):
                u = self._generate_sample()
                self._update_welford(u)
                self.samples_this_run += 1

                s          = self.stats
                var_u      = s.wf_M2 / max(s.wf_n - 1, 1)
                current_T  = s.wf_n * DT_SAMPLE
                sigma_Ubar = np.sqrt(
                    max(2.0 * var_u * s.T_int_eff_cur / current_T, 0.0))
                ci_rel_now = Z_SCORE * sigma_Ubar / max(abs(s.wf_mean), 1e-9)
                self._record_history(u, ci_rel_now)

                if self.samples_this_run * DT_SAMPLE >= MAX_SAMPLING_TIME_S:
                    self._finish_run()
                    break

                if self.samples_this_run < BURNIN_SAMPLES:
                    continue

                cond1, cond2, cond3, cond4 = self._check_stopping_conditions()
                if cond1 and cond2 and cond3 and cond4:
                    self._finish_run()
                    break


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class Environment:
    """
    Mega-scale wake Environment.
    """

    def __init__(self, seed: int = 42):
        self.tick_count: int  = 0
        self.elapsed_s: float = 0.0
        self.sampling_times: List[float] = []

        master_rng  = np.random.default_rng(seed)
        drone_seeds = master_rng.integers(0, 2**31, size=N_DRONES)

        self.target_positions: np.ndarray = master_rng.uniform(
            0.0, 100.0, size=(N_TARGETS, 3)
        ).astype(float)

        self.target_measured: np.ndarray = np.zeros(N_TARGETS, dtype=bool)
        self.target_locked:   np.ndarray = np.zeros(N_TARGETS, dtype=bool)

        self.drones: list[Drone] = [
            Drone(
                drone_id=i,
                position=_spawn_position(i),
                rng=np.random.default_rng(int(drone_seeds[i])),
            )
            for i in range(N_DRONES)
        ]

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
                self.sampling_times.append(drone.last_result["elapsed_s"])
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
        diff           = (idle_positions[:, np.newaxis, :]
                          - node_positions[np.newaxis, :, :])
        cost_matrix    = np.linalg.norm(diff, axis=-1)
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
            drone.tick(neighbours=neighbours)

        self._collect_completions()
        self.dispatch()
        self.tick_count += 1
        self.elapsed_s  += DT_KIN

    def run(self, duration_s: float) -> None:
        for _ in range(int(duration_s / DT_KIN)):
            self.step()

    @property
    def n_measured(self) -> int:
        return int(self.target_measured.sum())

    @property
    def all_measured(self) -> bool:
        return bool(self.target_measured.all())

    def summary(self) -> str:
        n_idle     = sum(1 for d in self.drones if d.state == DroneState.IDLE)
        n_transit  = sum(1 for d in self.drones if d.state == DroneState.TRANSIT)
        n_sampling = sum(1 for d in self.drones if d.state == DroneState.SAMPLING)
        n_rth      = sum(1 for d in self.drones if d.state == DroneState.RTH)
        n_depleted = sum(1 for d in self.drones if d.battery_depleted)
        return (
            f"t={self.elapsed_s:7.1f}s  "
            f"nodes={self.n_measured:>3}/{N_TARGETS}  "
            f"assigns={self._assignments_made:>4}  "
            f"[IDLE={n_idle} TRANSIT={n_transit} "
            f"SAMP={n_sampling} RTH={n_rth} DEP={n_depleted}]"
        )


def _smoke_test() -> None:
    print("=" * 72)
    print("  LAWSS Stage 3 Wake MEGA — Smoke Test")
    print(f"  {N_DRONES} drones  |  {N_TARGETS} random 3-D nodes in [0,100]³")
    print(f"  Spawn grid: {_SPAWN_ROWS} rows × {_SPAWN_COLS} cols, "
          f"{_SPAWN_STEP} m step")
    print(f"  Battery: {BATTERY_CAPACITY_S:.0f} s ({BATTERY_CAPACITY_S/60:.0f} min)  "
          f"|  RTH threshold: {RTH_THRESHOLD*100:.0f}%  "
          f"|  Hard cap: {MAX_SAMPLING_TIME_S:.0f} s")
    print(f"  N_OBS = {N_OBS}  (N_DRONES - 1)")
    print()
    print(f"  Building {N_DRONES} MPC solvers …")

    t0 = time.perf_counter()
    env = Environment(seed=0)
    print(f"  Build time: {time.perf_counter() - t0:.2f} s\n")

    budget_ticks = int(BATTERY_CAPACITY_S / DT_KIN)
    report_every = int(60.0 / DT_KIN)
    wall_start   = time.perf_counter()
    early_stop   = None

    for tick in range(budget_ticks):
        env.step()
        if env.all_measured and early_stop is None:
            early_stop = env.elapsed_s
        if tick % report_every == 0 or env.all_measured:
            wall = time.perf_counter() - wall_start
            print(f"  {env.summary()}  wall={wall:.1f}s")
            if env.all_measured:
                break

    total_wall = time.perf_counter() - wall_start
    print(f"\n  Total wall time: {total_wall:.1f} s")
    if early_stop:
        print(f"  All {N_TARGETS} nodes measured at t = {early_stop:.1f} s "
              f"({early_stop / 60:.2f} min)")
    else:
        print(f"  {env.n_measured}/{N_TARGETS} nodes measured within budget.")
    print("=" * 72)


if __name__ == "__main__":
    _smoke_test()