from __future__ import annotations
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

FS: float = 10.0
DT: float = 1.0 / FS

# ─────────────────────────────────────────────────────────────────────────────
# Wake signal parameters  (mirrors wake_sampler.py CONFIG)
# ─────────────────────────────────────────────────────────────────────────────

U_INF:    float = 15.0    # free-stream wind speed            [m/s]
U_WAKE:   float = 9.0     # mean velocity in the wake         [m/s]  (~0.6·U_inf)
I_U_RAND: float = 0.22    # random turbulence intensity       [-]
I_U_SHED: float = 0.15    # periodic shedding intensity       [-]
T_INT_RAND: float = 5.0   # true random integral time scale   [s]

D:  float = 20.0          # characteristic building dimension [m]
ST: float = 0.10          # Strouhal number                   [-]

# Derived signal parameters
SIGMA_RAND: float = I_U_RAND * U_WAKE
SIGMA_SHED: float = I_U_SHED * U_WAKE
F_SHED:     float = ST * U_INF / D                   # vortex shedding freq [Hz]
T_SHED:     float = 1.0 / F_SHED                     # shedding period      [s]

# AR(1) coefficients for the random turbulence component
PHI_RAND:   float = np.exp(-DT / T_INT_RAND)
SIGMA_EPS:  float = SIGMA_RAND * np.sqrt(1.0 - PHI_RAND**2)

# Effective integral time scale (random variance fraction of total)
ALPHA_FRAC:   float = SIGMA_RAND**2 / (SIGMA_RAND**2 + SIGMA_SHED**2)
T_INT_EFF:    float = ALPHA_FRAC * T_INT_RAND        # what ACF estimator converges to

# ─────────────────────────────────────────────────────────────────────────────
# Stopping-criterion thresholds  (relaxed for wake, per design rationale)
# ─────────────────────────────────────────────────────────────────────────────

EPSILON_CI:  float = 0.10    # Cond 1: Z·σ_Ū/Ū  < 10 %     (inlet: 5 %)
DELTA_STAB:  float = 0.02    # Cond 2: mean drift < 2 %    (inlet: 1 %)
Z_SCORE:     float = 1.645   # 90 % confidence             (inlet: 1.96)
N_EFF_MIN:   int   = 15      # Cond 3: min independent samples (inlet: 10)
N_SHED_MIN:  int   = 5       # Cond 4 (NEW): min complete shedding cycles

# ─────────────────────────────────────────────────────────────────────────────
# ACF estimator settings
# ─────────────────────────────────────────────────────────────────────────────

N_ACF_UPDATE: int  = 50      # re-estimate ACF every N samples
ACF_EMA_ALPHA: float = 0.3   # EMA weight for new ACF estimate
                              # (higher than inlet EMA because ACF is stable)

# Circular buffer: holds 4 shedding periods; lag search up to 1.5 periods
ACF_BUF_LEN:  int = max(int(4.0 * T_SHED * FS), 400)
ACF_MAX_LAG:  int = max(int(1.5 * T_SHED * FS), 150)

# Burn-in: wait at least 5 × T_int_eff OR 5 shedding cycles before any check
BURNIN_SAMPLES: int = max(
    int(5.0 * T_INT_EFF * FS),
    int(N_SHED_MIN * T_SHED * FS),
)

# Stability look-back window: 8 × T_int_eff (wider than inlet's 5 × T_int)
STAB_WIN: int = max(int(8.0 * T_INT_EFF * FS), 2)

# Persistence filter: Cond 2 must hold continuously for 1 full shedding cycle
PERSIST_SAMPLES: int = int(T_SHED * FS)

# ─────────────────────────────────────────────────────────────────────────────
# Fleet & operational constants  (UNCHANGED from Stage 3 inlet)
# ─────────────────────────────────────────────────────────────────────────────

BATTERY_CAPACITY_S:  float = 1080.0   # 18-min battery
N_TARGETS:           int   = 100      # 100 random 3-D nodes
N_DRONES:            int   = 10
MAX_SAMPLING_TIME_S: float = 180.0    # 3-min hard cap per run

# ─────────────────────────────────────────────────────────────────────────────
# MPC constants  (UNCHANGED)
# ─────────────────────────────────────────────────────────────────────────────

MPC_N:    int   = 20
V_MAX:    float = 14.0    # max velocity per axis [m/s]
A_MAX:    float = 5.0
D_MIN:    float = 3.0
N_OBS:    int   = 9
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
# ACF zero-crossing integrator  (standalone helper; mirrors wake_sampler.py)
# ─────────────────────────────────────────────────────────────────────────────

def acf_zero_crossing_T_int(buf: np.ndarray, dt: float, max_lag: int) -> float:
    """
    Estimate the integral time scale T_int by integrating the sample
    autocorrelation function from lag 0 up to its first zero-crossing.

    Why not lag-1?
    ──────────────
    In the wake the ACF is a damped cosine, not a pure exponential:
        ρ(τ) = α·exp(−τ/T_int_random) + (1−α)·cos(2πf_shed τ)
    The cosine keeps ρ₁ ≈ 1, so the lag-1 estimator T = −Δt/ln(ρ₁)
    returns ~90 s (a ~27× overestimate).  Integrating only up to the
    first zero-crossing captures the exponential lobe and ignores the
    runaway periodic contribution.

    Parameters
    ----------
    buf     : 1-D array of velocity samples (already time-ordered)
    dt      : timestep [s]
    max_lag : maximum lag index to search (should ≥ T_shed/4 · fs)

    Returns
    -------
    T_int_est : float [s], or np.nan if insufficient data / flat signal
    """
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

    k0 = sign_changes[0]
    # Sub-sample linear interpolation to the exact zero crossing
    # frac = acf[k0] / (acf[k0] - acf[k0 + 1])  # not used directly

    tau_grid = lags[:k0 + 1] * dt
    T_est    = float(np.trapezoid(acf[:k0 + 1], tau_grid))

    return max(T_est, dt)


# ─────────────────────────────────────────────────────────────────────────────
# State machine  (UNCHANGED from Stage 3 inlet)
# ─────────────────────────────────────────────────────────────────────────────

class DroneState(Enum):
    IDLE     = auto()
    TRANSIT  = auto()
    SAMPLING = auto()
    RTH      = auto()


# ─────────────────────────────────────────────────────────────────────────────
# WelfordState  — wake variant
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WelfordState:
    """
    All mutable accumulators for one measurement run.

    Changes vs. inlet WelfordState:
      • Lag-1 accumulators (L_n, L_mux, L_muy, L_mxy, L_mx2) removed —
        they are unreliable in the wake due to vortex shedding.
      • acf_buffer / buf_idx: circular buffer for the ACF estimator.
      • T_int_eff_cur: current online estimate of the effective integral
        time scale; initialised to the theoretical value T_INT_EFF.
      • stab_hist: full-length history array for the stability look-back
        and the 1-cycle persistence filter.
      • stab_hist_ptr: rolling pointer into stab_hist.
    """
    wf_n: int      = 0
    wf_mean: float = 0.0
    wf_M2: float   = 0.0

    # ACF circular buffer
    acf_buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(ACF_BUF_LEN, dtype=float))
    buf_idx: int = 0

    # Online T_int_eff estimate (ACF zero-crossing integrator)
    T_int_eff_cur: float = T_INT_EFF

    # Stability look-back: sized to MAX_SAMPLING_TIME_S so we can always
    # look back STAB_WIN samples without index arithmetic issues.
    stab_hist: np.ndarray = field(
        default_factory=lambda: np.full(
            int(MAX_SAMPLING_TIME_S * FS) + STAB_WIN + 10, np.nan))
    stab_hist_ptr: int = 0

    # Persistence filter: track last PERSIST_SAMPLES values of the drift metric
    # (stored as a flat array; we only care whether ALL are < DELTA_STAB)
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
    """
    Stage 3 Wake variant.

    Identical to the inlet Stage 3 Drone in all MPC / RTH / fleet logic.
    Only the sampling physics (signal generation, Welford update, stopping
    conditions) differ.
    """

    def __init__(self, drone_id: int, position: np.ndarray,
                 rng: np.random.Generator):
        self.id       = drone_id
        self.position = position.astype(float).copy()
        self.velocity = np.zeros(3, dtype=float)
        self.rng      = rng

        # Each drone gets a random initial shedding phase so their signals
        # are not coherent (realistic: drones are at different spatial positions)
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

        # History arrays for the live dashboard
        _buf = int(BATTERY_CAPACITY_S * FS)
        self.hist_raw_velocity: np.ndarray = np.full(_buf, np.nan)
        self.hist_running_mean: np.ndarray = np.full(_buf, np.nan)
        self.hist_ci_rel: np.ndarray       = np.full(_buf, np.nan)
        self.hist_T_eff: np.ndarray        = np.full(_buf, np.nan)  # replaces hist_T_int
        self._hist_ptr: int                = 0

        # AR(1) random-turbulence state (reset each run)
        self._u_rand_prime: float = 0.0

        self.last_result: Optional[dict] = None

        # MPC internals
        self._warm_X: Optional[np.ndarray] = None
        self._warm_U: Optional[np.ndarray] = None
        self._last_solve_ok: bool = True
        self._build_mpc()

    # ── MPC construction  (IDENTICAL to inlet Stage 3) ───────────────────────

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

    # ── MPC solve  (IDENTICAL to inlet Stage 3) ───────────────────────────────

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
                cruise_v = unit * min(V_MAX * 0.5, dist / (MPC_N * DT))
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

    # ── update_trajectory  (IDENTICAL to inlet Stage 3) ──────────────────────

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

    # ── Assignment & sampling lifecycle ──────────────────────────────────────

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
        self._u_rand_prime    = 0.0   # reset AR(1) random turbulence state

    # ── Wake signal generation  (MODIFIED) ───────────────────────────────────

    def _generate_sample(self) -> float:
        """
        Generate one velocity sample from the two-component wake signal:

            u(t) = U_wake + u_rand(t) + u_shed(t)

        where:
            u_rand(t) ~ AR(1) broadband turbulence
            u_shed(t) = σ_rand·√2 · sin(2π f_shed t + phase0)

        The shedding amplitude matches wake_sampler.py (σ_rand·√2 so that
        the RMS of the sine equals σ_rand, keeping I_u_shed = I_u_rand
        in the original script).

        Time for the sine wave is derived from the sample counter so that
        the shedding phase evolves smoothly across each run (it resets at
        the start of each run together with stats, which is correct because
        the drone has moved to a new spatial location).
        """
        # AR(1) random turbulence
        eps                = self.rng.normal(0.0, SIGMA_EPS)
        self._u_rand_prime = PHI_RAND * self._u_rand_prime + eps
        u_rand = self._u_rand_prime

        # Quasi-periodic vortex shedding
        t_local = self.samples_this_run * DT
        u_shed  = SIGMA_RAND * np.sqrt(2.0) * np.sin(
            2.0 * np.pi * F_SHED * t_local + self.phase0
        )

        return U_WAKE + u_rand + u_shed

    # ── Welford + ACF update  (MODIFIED — no lag-1) ──────────────────────────

    def _update_welford(self, u: float) -> None:
        """
        Update Welford running mean/variance and, every N_ACF_UPDATE samples,
        re-estimate T_int_eff from the ACF zero-crossing integrator.

        The circular buffer is written in a ring (buf_idx mod ACF_BUF_LEN).
        When the buffer is full, we unroll it into chronological order before
        passing to acf_zero_crossing_T_int().
        """
        s = self.stats

        # ── Welford ──────────────────────────────────────────────────────────
        s.wf_n   += 1
        d         = u - s.wf_mean
        s.wf_mean += d / s.wf_n
        s.wf_M2  += d * (u - s.wf_mean)

        # ── Stability history (full-length; needed for look-back + persist) ───
        s.stab_hist[s.stab_hist_ptr] = s.wf_mean
        s.stab_hist_ptr += 1  # never wraps — sized to MAX_SAMPLING_TIME_S

        # ── ACF circular buffer ───────────────────────────────────────────────
        s.acf_buffer[s.buf_idx % ACF_BUF_LEN] = u
        s.buf_idx += 1

        # Periodic ACF re-estimation (once buffer is full)
        if (s.buf_idx >= ACF_BUF_LEN
                and self.samples_this_run > 0
                and self.samples_this_run % N_ACF_UPDATE == 0):
            start   = s.buf_idx % ACF_BUF_LEN
            ordered = np.concatenate([s.acf_buffer[start:], s.acf_buffer[:start]])
            T_new   = acf_zero_crossing_T_int(ordered, DT, ACF_MAX_LAG)
            if np.isfinite(T_new) and T_new > 0.0:
                s.T_int_eff_cur = (
                    ACF_EMA_ALPHA * T_new
                    + (1.0 - ACF_EMA_ALPHA) * s.T_int_eff_cur
                )

    # ── Stopping conditions  (MODIFIED — 4 conditions) ───────────────────────

    def _check_stopping_conditions(self) -> tuple[bool, bool, bool, bool]:
        """
        Returns (cond1, cond2, cond3, cond4).

        Cond 1: Z·σ_Ū/Ū  < ε_ci           (relaxed: 90% CI, ε = 10%)
        Cond 2: running-mean drift < δ_stab, sustained for ≥ 1 T_shed
        Cond 3: N_eff ≥ N_eff_min
        Cond 4: elapsed cycles ≥ N_shed_min

        Cond 2 persistence filter: the drift metric must be below δ_stab
        continuously for at least PERSIST_SAMPLES (≈ 1 shedding cycle)
        before Cond 2 is declared True.  This prevents the algorithm from
        stopping on a transient trough in the drift between shedding bursts.
        """
        s         = self.stats
        current_T = s.wf_n * DT
        var_u     = s.wf_M2 / max(s.wf_n - 1, 1)

        # ── Cond 1: CI width ─────────────────────────────────────────────────
        sigma_Ubar = np.sqrt(
            max(2.0 * var_u * s.T_int_eff_cur / current_T, 0.0))
        ci_rel = Z_SCORE * sigma_Ubar / max(abs(s.wf_mean), 1e-9)
        cond1  = ci_rel < EPSILON_CI

        # ── Cond 2: stability + 1-cycle persistence filter ───────────────────
        n_hist = s.stab_hist_ptr
        if n_hist >= STAB_WIN:
            oldest = s.stab_hist[n_hist - STAB_WIN]
        else:
            oldest = np.nan

        if np.isfinite(oldest):
            drift = abs(s.wf_mean - oldest) / max(abs(s.wf_mean), 1e-9)
        else:
            drift = np.inf

        # Write current drift into rolling persistence buffer
        s.drift_hist[s.drift_ptr % PERSIST_SAMPLES] = drift
        s.drift_ptr += 1

        # Cond 2 True only if the ENTIRE persistence window is below threshold
        # (the buffer is initialised to inf so it can't trigger prematurely)
        cond2 = bool(np.all(s.drift_hist < DELTA_STAB))

        # ── Cond 3: effective independent samples ────────────────────────────
        N_eff = current_T / (2.0 * s.T_int_eff_cur)
        cond3 = N_eff >= N_EFF_MIN

        # ── Cond 4: minimum shedding cycles flushed ──────────────────────────
        n_cycles = current_T / T_SHED
        cond4    = n_cycles >= N_SHED_MIN

        return cond1, cond2, cond3, cond4

    # ── History recording ─────────────────────────────────────────────────────

    def _record_history(self, u: float, ci_rel: float) -> None:
        ptr = self._hist_ptr % len(self.hist_raw_velocity)
        self.hist_raw_velocity[ptr] = u
        self.hist_running_mean[ptr] = self.stats.wf_mean
        self.hist_ci_rel[ptr]       = ci_rel
        self.hist_T_eff[ptr]        = self.stats.T_int_eff_cur
        self._hist_ptr += 1

    # ── Run finalisation ──────────────────────────────────────────────────────

    def _finish_run(self) -> None:
        s = self.stats
        self.last_result = {
            "node_id":       self.target_node_id,
            "position":      self.position.copy(),
            "mean_est":      s.wf_mean,
            "T_int_eff_est": s.T_int_eff_cur,
            "n_samples":     self.samples_this_run,
            "elapsed_s":     self.samples_this_run * DT,
        }
        self.completed_node_id = self.target_node_id
        self.target_position   = None
        self.target_node_id    = None
        self.state             = DroneState.IDLE

    # ── Main tick  (one 10 Hz step) ───────────────────────────────────────────

    def tick(
        self,
        neighbours: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        """
        One 10 Hz tick.

        Note: no previous-sample accumulator is needed here — the wake
        signal generator is self-contained (AR(1) state is stored in
        self._u_rand_prime).
        """
        if self.battery_depleted:
            return

        self.battery_remaining_s -= DT
        if self.battery_remaining_s <= 0.0:
            self.battery_remaining_s = 0.0
            self.battery_depleted    = True
            self.state               = DroneState.IDLE
            return

        # RTH trigger — battery below threshold
        if (self.battery_remaining_s <= BATTERY_CAPACITY_S * RTH_THRESHOLD
                and self.state not in (DroneState.RTH, DroneState.IDLE)):
            if self.target_node_id is not None:
                self.abandoned_node_id = self.target_node_id
                self.target_node_id    = None
            self.state           = DroneState.RTH
            self.target_position = self.home_position.copy()
            self._warm_X         = None
            self._warm_U         = None

        # State dispatch
        if self.state == DroneState.IDLE:
            pass

        elif self.state in (DroneState.TRANSIT, DroneState.RTH):
            self.update_trajectory(neighbours=neighbours)

        elif self.state == DroneState.SAMPLING:
            u = self._generate_sample()
            self._update_welford(u)
            self.samples_this_run += 1

            s          = self.stats
            var_u      = s.wf_M2 / max(s.wf_n - 1, 1)
            current_T  = s.wf_n * DT
            sigma_Ubar = np.sqrt(
                max(2.0 * var_u * s.T_int_eff_cur / current_T, 0.0))
            ci_rel_now = Z_SCORE * sigma_Ubar / max(abs(s.wf_mean), 1e-9)
            self._record_history(u, ci_rel_now)

            # 3-minute hard cap
            if self.samples_this_run * DT >= MAX_SAMPLING_TIME_S:
                self._finish_run()
                return

            # Burn-in gate
            if self.samples_this_run < BURNIN_SAMPLES:
                return

            cond1, cond2, cond3, cond4 = self._check_stopping_conditions()
            if cond1 and cond2 and cond3 and cond4:
                self._finish_run()


# ─────────────────────────────────────────────────────────────────────────────
# Environment  (IDENTICAL logic to Stage 3 inlet; only drone type changes)
# ─────────────────────────────────────────────────────────────────────────────

class Environment:
    """
    Stage 3 Wake Environment.

    Fleet logic — Hungarian dispatcher, RTH unlock, neighbour snapshot —
    is IDENTICAL to the inlet Stage 3 Environment.  The only difference
    is that Drone objects use wake-physics signal generation; the AR(1)+shed
    model is self-contained so no previous-sample accumulator is needed.
    """

    def __init__(self, seed: int = 42):
        self.tick_count: int  = 0
        self.elapsed_s: float = 0.0

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
                position=np.array([i * 4.0, 0.0, 0.0], dtype=float),
                rng=np.random.default_rng(int(drone_seeds[i])),
            )
            for i in range(N_DRONES)
        ]

        self._dispatch_calls:   int = 0
        self._assignments_made: int = 0

    # ── Completion polling ────────────────────────────────────────────────────

    def _collect_completions(self) -> None:
        for drone in self.drones:
            # Unlock nodes abandoned by RTH
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

    # ── Hungarian dispatcher  (UNCHANGED) ────────────────────────────────────

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

        idle_positions  = np.array([self.drones[i].position for i in idle_ids])
        node_positions  = self.target_positions[free_node_ids]
        diff            = (idle_positions[:, np.newaxis, :]
                           - node_positions[np.newaxis, :, :])
        cost_matrix     = np.linalg.norm(diff, axis=-1)
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

    # ── Simulation clock ──────────────────────────────────────────────────────

    def step(self) -> None:
        """
        One 10 Hz tick:
          1. Snapshot positions/velocities BEFORE any drone moves.
          2. Tick every drone with its neighbours' snapshot.
          3. Collect completions, run dispatcher.
          4. Advance clock.
        """
        snapshot: List[Tuple[np.ndarray, np.ndarray]] = [
            (d.position.copy(), d.velocity.copy()) for d in self.drones
        ]

        for i, drone in enumerate(self.drones):
            # FIX: only include actively-moving drones as MPC obstacles.
            # IDLE and SAMPLING drones are stationary — including them causes
            # the MPC to route around ghost obstacles at already-vacated nodes.
            neighbours = [
                snap for j, (snap, d) in enumerate(zip(snapshot, self.drones))
                if j != i and d.state in (DroneState.TRANSIT, DroneState.RTH)
            ]
            drone.tick(neighbours=neighbours)

        self._collect_completions()
        self.dispatch()
        self.tick_count += 1
        self.elapsed_s  += DT

    def run(self, duration_s: float) -> None:
        """Blocking loop — headless testing only."""
        for _ in range(int(duration_s / DT)):
            self.step()

    # ── Properties ───────────────────────────────────────────────────────────

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
            f"nodes measured: {self.n_measured}/{N_TARGETS}  |  "
            f"assignments so far: {self._assignments_made}",
            "",
            f"  {'ID':>3}  {'State':>10}  {'Node':>5}  "
            f"{'Samples':>8}  {'Mean est':>9}  {'T_eff':>7}  "
            f"{'Battery':>7}  {'MPC':>5}",
            "  " + "─" * 68,
        ]

        for d in self.drones:
            s    = d.stats
            node = str(d.target_node_id) if d.target_node_id is not None else "—"
            mean = f"{s.wf_mean:.3f}" if s.wf_n > 0 else "—"
            teff = f"{s.T_int_eff_cur:.2f}" if s.buf_idx >= ACF_BUF_LEN else "prior"
            batt = f"{d.battery_remaining_s:.0f}s"
            mpc_ok = "ok" if d._last_solve_ok else "fb"
            lines.append(
                f"  {d.id:>3}  {d.state.name:>10}  {node:>5}  "
                f"{d.samples_this_run:>8}  {mean:>9}  {teff:>7}  "
                f"{batt:>7}  {mpc_ok:>5}"
            )

        if show_completed:
            completed = [d for d in self.drones if d.last_result is not None]
            if completed:
                lines += ["", "  Completed runs:"]
                for d in sorted(completed,
                                 key=lambda x: x.last_result["elapsed_s"]):
                    r   = d.last_result
                    err = abs(r["mean_est"] - U_WAKE) / U_WAKE * 100
                    lines.append(
                        f"    Drone {d.id:>2}  node {r['node_id']:>3}  "
                        f"mean={r['mean_est']:.4f} m/s  "
                        f"err={err:.2f}%  "
                        f"T={r['elapsed_s']:.1f}s  "
                        f"T_eff_est={r['T_int_eff_est']:.2f}s"
                    )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Verify Stage 3 Wake correctness.

    Assertions
    ──────────
    A. All 100 nodes measured before the 18-minute battery expires.
    B. No node measured by two drones (double-assignment prevention).
    C. target_locked[i] is False for every measured node at end.
    D. Each drone arrived within ARRIVAL_DIST of its target node.
    """
    print("=" * 72)
    print("  LAWSS Stage 3 Wake — Smoke Test  (CasADI MPC + Wake Physics)")
    print(f"  {N_DRONES} drones  |  {N_TARGETS} random 3-D nodes in [0,100]³")
    print(f"  Battery: {BATTERY_CAPACITY_S:.0f} s ({BATTERY_CAPACITY_S/60:.0f} min)  "
          f"|  RTH threshold: {RTH_THRESHOLD*100:.0f}%  "
          f"|  Hard cap: {MAX_SAMPLING_TIME_S:.0f} s")
    print()
    print("  Wake signal parameters:")
    print(f"    U_inf={U_INF} m/s  U_wake={U_WAKE} m/s  "
          f"I_u_rand={I_U_RAND}  I_u_shed={I_U_SHED}")
    print(f"    D={D} m  St={ST}  f_shed={F_SHED:.4f} Hz  T_shed={T_SHED:.2f} s")
    print(f"    T_int_rand={T_INT_RAND} s  T_int_eff={T_INT_EFF:.3f} s  "
          f"alpha_frac={ALPHA_FRAC:.3f}")
    print()
    print("  Stopping thresholds (relaxed for wake):")
    print(f"    ε_ci={EPSILON_CI*100:.0f}%  δ_stab={DELTA_STAB*100:.0f}%  "
          f"Z={Z_SCORE}  N_eff_min={N_EFF_MIN}  N_shed_min={N_SHED_MIN}")
    print(f"    Burn-in: {BURNIN_SAMPLES} samples ({BURNIN_SAMPLES*DT:.1f} s)  "
          f"ACF buf: {ACF_BUF_LEN} samples  "
          f"Persist: {PERSIST_SAMPLES} samples ({PERSIST_SAMPLES*DT:.1f} s)")
    print()
    print("  Building MPC solvers …")

    t_build = time.perf_counter()
    env = Environment(seed=0)
    print(f"  Build time: {time.perf_counter() - t_build:.2f} s\n")

    snapshot_times: set[float] = {
        60.0, 120.0, 180.0, 300.0, 420.0, 600.0, 780.0, 900.0, 1080.0,
    }
    t0_wall    = time.perf_counter()
    early_stop = None

    budget_ticks = int(BATTERY_CAPACITY_S / DT)
    for _ in range(budget_ticks):
        env.step()

        t = round(env.elapsed_s, 1)
        if t in snapshot_times:
            snapshot_times.discard(t)
            print(f"\n{'─' * 72}")
            print(env.summary())

        if env.all_measured and early_stop is None:
            early_stop = env.elapsed_s

    wall = time.perf_counter() - t0_wall

    print(f"\n{'═' * 72}")
    print(f"  Simulation complete.")
    print(f"  Wall time  : {wall:.2f} s  ({BATTERY_CAPACITY_S/wall:.0f}× real-time)")
    if early_stop:
        print(f"  All nodes measured at t = {early_stop:.1f} s "
              f"({early_stop/60:.2f} min)  — "
              f"{BATTERY_CAPACITY_S - early_stop:.1f} s before budget.")
    else:
        print(f"  WARNING: not all nodes measured within "
              f"{BATTERY_CAPACITY_S:.0f} s budget.")
    print(f"  Total assignments: {env._assignments_made}")
    print(f"{'═' * 72}")
    print(env.summary(show_completed=True))

    # ── Assertions ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print("  Assertions:")

    assert env.all_measured, \
        f"FAIL A: only {env.n_measured}/{N_TARGETS} nodes measured"
    print("  A — All nodes measured within budget               ✓")

    measured_nodes = [
        d.last_result["node_id"]
        for d in env.drones if d.last_result is not None
    ]
    assert len(measured_nodes) == len(set(measured_nodes)), \
        f"FAIL B: duplicate node IDs {measured_nodes}"
    print("  B — No node measured twice (no double-assign)      ✓")

    still_locked = [i for i in range(N_TARGETS)
                    if env.target_measured[i] and env.target_locked[i]]
    assert not still_locked, \
        f"FAIL C: measured nodes still locked: {still_locked}"
    print("  C — All measured nodes correctly unlocked          ✓")

    for d in env.drones:
        if d.last_result is None:
            continue
        r    = d.last_result
        npos = env.target_positions[r["node_id"]]
        dist = float(np.linalg.norm(np.array(r["position"]) - npos))
        assert dist <= ARRIVAL_DIST + 1e-6, \
            (f"FAIL D: drone {d.id} finished {dist:.3f} m from node "
             f"{r['node_id']} (threshold {ARRIVAL_DIST} m)")
    print(f"  D — All drones arrived within {ARRIVAL_DIST} m of target     ✓")

    print(f"{'─' * 72}")
    print("  All assertions passed.")


if __name__ == "__main__":
    _smoke_test()
