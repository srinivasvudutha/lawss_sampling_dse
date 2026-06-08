from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ── Simulation import (must precede any Qt initialisation) ───────────────────
try:
    from lawss_stage3_wake import (
        Environment,
        DroneState,
        N_DRONES,
        N_TARGETS,
        U_WAKE,
        DT,
        BATTERY_CAPACITY_S,
        ACF_BUF_LEN,
    )
except ModuleNotFoundError:
    print(
        "ERROR: lawss_stage3_wake.py not found. "
        "Place it in the same directory as this script."
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard constants
# ─────────────────────────────────────────────────────────────────────────────

STEPS_PER_FRAME: int  = 1        # physics ticks per GUI refresh cycle
GUI_TIMER_MS:    int  = 16       # ~60 Hz GUI poll (ms)

TRAIL_LEN:        int   = 80     # position history depth per drone
PLOT_WINDOW_S:    float = 60.0   # seconds of signal shown in 2-D panels
PLOT_WIN_SAMPLES: int   = int(PLOT_WINDOW_S / DT)

ENV_LIM: float = 100.0           # cubic domain side length [m]

# ── Colour palette (dark theme) ───────────────────────────────────────────────
BG_DARK = (18,  20,  28)
FG_TEXT = (210, 215, 230)

# Per-drone colours as float RGBA (0–1) and integer RGB (0–255)
DRONE_COLORS_F: List[tuple] = [
    (0.18, 0.60, 1.00, 1.0),   # 0  azure
    (1.00, 0.32, 0.32, 1.0),   # 1  coral
    (0.18, 0.88, 0.70, 1.0),   # 2  teal
    (1.00, 0.65, 0.10, 1.0),   # 3  amber
    (0.80, 0.35, 1.00, 1.0),   # 4  violet
    (0.30, 0.85, 1.00, 1.0),   # 5  sky
    (0.35, 1.00, 0.55, 1.0),   # 6  mint
    (1.00, 0.30, 0.65, 1.0),   # 7  hot pink
    (0.90, 0.90, 0.15, 1.0),   # 8  lime
    (0.65, 0.50, 1.00, 1.0),   # 9  lavender
]
DRONE_COLORS_I = [tuple(int(x * 255) for x in c[:3]) for c in DRONE_COLORS_F]

# Node status colours as float RGBA arrays
COL_NODE_FREE     = np.array([0.55, 0.55, 0.65, 0.55], dtype=np.float32)
COL_NODE_LOCKED   = np.array([1.00, 0.58, 0.08, 0.90], dtype=np.float32)
COL_NODE_MEASURED = np.array([0.20, 0.90, 0.40, 0.95], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe snapshot dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DroneSnapshot:
    id:               int
    position:         np.ndarray          # (3,)  copy
    state:            DroneState
    target_position:  Optional[np.ndarray]
    batt_pct:         float
    wf_mean:          float
    T_int_eff_est:    float
    acf_n:            int
    samples_this_run: int
    target_node_id:   Optional[int]
    hist_raw:         np.ndarray          # (PLOT_WIN_SAMPLES,)
    hist_mean:        np.ndarray
    hist_ci:          np.ndarray
    hist_T_eff:       np.ndarray


@dataclass
class EnvSnapshot:
    elapsed_s:        float
    tick_count:       int
    n_measured:       int
    assignments_made: int
    target_measured:  np.ndarray          # (N_TARGETS,) bool
    target_locked:    np.ndarray
    target_positions: np.ndarray          # (N_TARGETS, 3)
    drones:           List[DroneSnapshot]
    min_sep_m:        float
    n_rth:            int


def _build_snapshot(env: Environment) -> EnvSnapshot:
    """
    Deep-copy the entire mutable environment into an immutable snapshot.
    Called on the simulation thread — every array is .copy()'d so the
    GUI thread never aliases live simulation data.
    """
    drones  = env.drones
    buf_len = len(drones[0].hist_raw_velocity)

    # Minimum inter-drone separation (O(N²), N=10, negligible cost)
    positions = np.array([d.position for d in drones])
    diff      = positions[:, None, :] - positions[None, :, :]
    dist_sq   = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_sep = float(np.sqrt(dist_sq.min()))
    n_rth   = sum(1 for d in drones if d.state == DroneState.RTH)

    drone_snaps: List[DroneSnapshot] = []
    for d in drones:
        ptr       = d._hist_ptr % buf_len
        # Unroll circular buffer so index 0 = oldest sample
        raw_full  = np.roll(d.hist_raw_velocity, -ptr)
        mean_full = np.roll(d.hist_running_mean, -ptr)
        ci_full   = np.roll(d.hist_ci_rel,       -ptr)
        t_eff_full = np.roll(d.hist_T_eff,        -ptr)

        drone_snaps.append(DroneSnapshot(
            id               = d.id,
            position         = d.position.copy(),
            state            = d.state,
            target_position  = (d.target_position.copy()
                                if d.target_position is not None else None),
            batt_pct         = 100.0 * d.battery_remaining_s / BATTERY_CAPACITY_S,
            wf_mean          = d.stats.wf_mean,
            T_int_eff_est    = d.stats.T_int_eff_cur,
            acf_n            = d.stats.buf_idx,
            samples_this_run = d.samples_this_run,
            target_node_id   = d.target_node_id,
            hist_raw         = raw_full [-PLOT_WIN_SAMPLES:].copy(),
            hist_mean        = mean_full[-PLOT_WIN_SAMPLES:].copy(),
            hist_ci          = ci_full  [-PLOT_WIN_SAMPLES:].copy(),
            hist_T_eff       = t_eff_full[-PLOT_WIN_SAMPLES:].copy(),
        ))

    return EnvSnapshot(
        elapsed_s        = env.elapsed_s,
        tick_count       = env.tick_count,
        n_measured       = env.n_measured,
        assignments_made = env._assignments_made,
        target_measured  = env.target_measured.copy(),
        target_locked    = env.target_locked.copy(),
        target_positions = env.target_positions.copy(),
        drones           = drone_snaps,
        min_sep_m        = min_sep,
        n_rth            = n_rth,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Background simulation worker  (plain threading.Thread — no Qt dependency)
# ─────────────────────────────────────────────────────────────────────────────

class SimWorker(threading.Thread):
    """
    Runs Environment.step() at real-time pace in a daemon thread.

    After every STEPS_PER_FRAME ticks it pushes an EnvSnapshot into
    `out_queue` (maxsize=4 prevents unbounded memory growth if the GUI
    lags).  The GUI thread drains the queue on its own timer tick.
    """

    def __init__(
        self,
        env: Environment,
        out_queue: "queue.Queue[EnvSnapshot]",
        realtime_pacing: bool = True,
    ):
        super().__init__(daemon=True, name="SimWorker")
        self._env             = env
        self._queue           = out_queue
        self._realtime_pacing = realtime_pacing
        self._stop_flag       = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def run(self) -> None:
        env           = self._env
        budget_ticks  = int(BATTERY_CAPACITY_S / DT)
        step_wall     = DT * STEPS_PER_FRAME        # real-time budget per batch
        total_ticks   = 0

        while not self._stop_flag.is_set() and total_ticks < budget_ticks:
            t0 = time.perf_counter()

            for _ in range(STEPS_PER_FRAME):
                if self._stop_flag.is_set():
                    return
                env.step()
                total_ticks += 1

            snap = _build_snapshot(env)
            # Non-blocking put: drop oldest if full so the GUI is never stalled
            try:
                self._queue.put_nowait(snap)
            except queue.Full:
                try:
                    self._queue.get_nowait()   # discard oldest
                except queue.Empty:
                    pass
                self._queue.put_nowait(snap)

            # Pace to real time for the live GUI. Export paths can disable this.
            if self._realtime_pacing:
                elapsed = time.perf_counter() - t0
                remaining = step_wall - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        # Push one final snapshot so the GUI shows the completed state
        if not self._stop_flag.is_set():
            try:
                self._queue.put(self._build_snapshot_safe(), timeout=1.0)
            except Exception:
                pass

    def _build_snapshot_safe(self) -> EnvSnapshot:
        try:
            return _build_snapshot(self._env)
        except Exception:
            # If env is in a weird state just return what we have
            return _build_snapshot(self._env)


# ─────────────────────────────────────────────────────────────────────────────
# GPU-accelerated 3-D scene
# ─────────────────────────────────────────────────────────────────────────────

class Scene3D:
    """
    Manages all pyqtgraph.opengl GL items inside a single GLViewWidget.
    All update() calls happen on the GUI thread from already-copied snapshots.
    """

    def __init__(self, gl_widget) -> None:
        import pyqtgraph.opengl as gl

        self._w = gl_widget
        self._w.setBackgroundColor(BG_DARK)
        self._w.setCameraPosition(distance=220, elevation=22, azimuth=-55)

        # Reference grid on the ground plane
        grid = gl.GLGridItem()
        grid.setSize(ENV_LIM, ENV_LIM)
        grid.setSpacing(10, 10)
        grid.translate(ENV_LIM / 2, ENV_LIM / 2, 0)
        grid.setColor((55, 60, 85, 70))
        self._w.addItem(grid)

        # ── Axis lines (X → origin → Y → origin → Z) ─────────────────────────
        # gl.GLAxisItem() is not used here: its paint method hard-codes RGB
        # primaries (red/green/blue) with no way to override the colour, so it
        # clashes with the off-white theme.  Instead a single line-strip trace
        # draws all three semi-axes in one consistent off-white colour.
        _axis_pts = np.array([
            [ENV_LIM, 0.0,     0.0    ],   # +X tip
            [0.0,     0.0,     0.0    ],   # origin
            [0.0,     ENV_LIM, 0.0    ],   # +Y tip
            [0.0,     0.0,     0.0    ],   # origin (return)
            [0.0,     0.0,     ENV_LIM],   # +Z tip
        ], dtype=np.float32)
        axis_item = gl.GLLinePlotItem(
            pos      = _axis_pts,
            color    = (0.80, 0.82, 0.88, 0.8),
            width    = 2.0,
            mode     = 'line_strip',
        )
        self._w.addItem(axis_item)

        # ── Target node cloud ─────────────────────────────────────────────────
        self._node_scatter = gl.GLScatterPlotItem(
            pos   = np.zeros((N_TARGETS, 3), dtype=np.float32),
            color = np.tile(COL_NODE_FREE, (N_TARGETS, 1)),
            size  = 6.0,
            pxMode= True,
        )
        self._w.addItem(self._node_scatter)

        # ── Drone cloud (single item; colour/size arrays mutated each frame) ──
        self._drone_pos    = np.zeros((N_DRONES, 3),  dtype=np.float32)
        self._drone_colors = np.array(DRONE_COLORS_F, dtype=np.float32)
        self._drone_scatter = gl.GLScatterPlotItem(
            pos   = self._drone_pos,
            color = self._drone_colors,
            size  = 10.0,
            pxMode= True,
        )
        self._w.addItem(self._drone_scatter)

        # ── Per-drone trails (GLLinePlotItem, line_strip) ─────────────────────
        self._trails: List[deque] = [deque(maxlen=TRAIL_LEN) for _ in range(N_DRONES)]
        self._trail_items: List   = []
        for i in range(N_DRONES):
            r, g, b = DRONE_COLORS_I[i]
            item = gl.GLLinePlotItem(
                pos      = np.zeros((2, 3), dtype=np.float32),
                color    = (r / 255, g / 255, b / 255, 0.35),
                width    = 1.5,
                antialias= True,
                mode     = 'line_strip',
            )
            self._w.addItem(item)
            self._trail_items.append(item)

        # ── Per-drone assignment lines (drone → target) ───────────────────────
        self._assign_items: List = []
        for i in range(N_DRONES):
            r, g, b = DRONE_COLORS_I[i]
            item = gl.GLLinePlotItem(
                pos      = np.zeros((2, 3), dtype=np.float32),
                color    = (r / 255, g / 255, b / 255, 0.22),
                width    = 1.0,
                antialias= False,
                mode     = 'lines',
            )
            item.setVisible(False)
            self._w.addItem(item)
            self._assign_items.append(item)

        self._nodes_set = False   # positions only need uploading once

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, snap: EnvSnapshot) -> None:
        # Upload target positions on the very first snapshot
        if not self._nodes_set:
            self._node_scatter.setData(pos=snap.target_positions.astype(np.float32))
            self._nodes_set = True

        # Target node colours
        node_colors = np.empty((N_TARGETS, 4), dtype=np.float32)
        for n in range(N_TARGETS):
            if snap.target_measured[n]:
                node_colors[n] = COL_NODE_MEASURED
            elif snap.target_locked[n]:
                node_colors[n] = COL_NODE_LOCKED
            else:
                node_colors[n] = COL_NODE_FREE
        self._node_scatter.setData(color=node_colors)

        # Drone positions + state-dependent appearance
        sizes        = np.full(N_DRONES, 9.0,  dtype=np.float32)
        drone_colors = np.array(DRONE_COLORS_F, dtype=np.float32)

        for ds in snap.drones:
            i = ds.id
            p = ds.position.astype(np.float32)
            self._drone_pos[i] = p
            self._trails[i].append(p.copy())

            if ds.state == DroneState.IDLE:
                drone_colors[i, 3] = 0.28
                sizes[i] = 6.0
            elif ds.state == DroneState.RTH:
                drone_colors[i] = np.array([1.0, 0.12, 0.12, 1.0], dtype=np.float32)
                sizes[i] = 12.0
            elif ds.state == DroneState.TRANSIT:
                drone_colors[i, 3] = 0.78
                sizes[i] = 9.0
            else:   # SAMPLING
                drone_colors[i, 3] = 1.0
                sizes[i] = 13.0

        self._drone_scatter.setData(pos=self._drone_pos,
                                    color=drone_colors,
                                    size=sizes)

        # Trail lines
        for i in range(N_DRONES):
            trail = self._trails[i]
            if len(trail) >= 2:
                arr = np.array(trail, dtype=np.float32)
                self._trail_items[i].setData(pos=arr)
            self._trail_items[i].setVisible(len(trail) >= 2)

        # Assignment / RTH lines
        for i, ds in enumerate(snap.drones):
            active = (ds.state in (DroneState.TRANSIT, DroneState.RTH)
                      and ds.target_position is not None)
            if active:
                seg   = np.array([ds.position, ds.target_position], dtype=np.float32)
                r, g, b = DRONE_COLORS_I[i]
                if ds.state == DroneState.RTH:
                    col = (1.0, 0.08, 0.08, 0.60)
                else:
                    col = (r / 255, g / 255, b / 255, 0.28)
                self._assign_items[i].setData(pos=seg, color=col)
                self._assign_items[i].setVisible(True)
            else:
                self._assign_items[i].setVisible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Per-drone 2-D velocity panel
# ─────────────────────────────────────────────────────────────────────────────

class DronePanel:
    """
    Wraps a single pyqtgraph PlotItem and owns four PlotDataItem curves:
      • raw velocity  (thin, translucent)
      • running mean  (thick, opaque)
      • CI upper / lower (thin dashed)
    Plus a horizontal InfiniteLine at U_WAKE for reference.
    """

    def __init__(self, plot_item, drone_id: int) -> None:
        import pyqtgraph as pg

        self._pi  = plot_item
        self._id  = drone_id
        r, g, b   = DRONE_COLORS_I[drone_id]
        self._rgb = (r, g, b)

        # Shared x-axis array (time in seconds, oldest → newest)
        self._x = np.linspace(0.0, PLOT_WINDOW_S, PLOT_WIN_SAMPLES,
                               dtype=np.float32)
        nan_y   = np.full(PLOT_WIN_SAMPLES, np.nan, dtype=np.float32)

        # U_WAKE reference — InfiniteLine is the correct pyqtgraph API
        ref_line = pg.InfiniteLine(
            pos   = float(U_WAKE),
            angle = 0,
            pen   = pg.mkPen((100, 100, 115, 110), width=1.0),
        )
        self._pi.addItem(ref_line)

        # Raw velocity (thin + semi-transparent)
        self._raw_curve = self._pi.plot(
            self._x, nan_y.copy(),
            pen=pg.mkPen((r, g, b, 65), width=0.8),
        )

        # Running mean (thick + opaque)
        self._mean_curve = self._pi.plot(
            self._x, nan_y.copy(),
            pen=pg.mkPen((r, g, b, 215), width=2.0),
        )

        # CI upper bound (thin dashed)
        self._ci_up = self._pi.plot(
            self._x, nan_y.copy(),
            pen=pg.mkPen((r, g, b, 85), width=0.8,
                         style=pg.QtCore.Qt.PenStyle.DashLine),
        )

        # CI lower bound (thin dashed)
        self._ci_dn = self._pi.plot(
            self._x, nan_y.copy(),
            pen=pg.mkPen((r, g, b, 85), width=0.8,
                         style=pg.QtCore.Qt.PenStyle.DashLine),
        )

        # Axis / display cosmetics
        self._pi.setYRange(U_WAKE - 5.0, U_WAKE + 5.0, padding=0)
        self._pi.setXRange(0.0, PLOT_WINDOW_S, padding=0)
        self._pi.getAxis('left').setWidth(30)
        self._pi.getAxis('left').setStyle(tickTextOffset=2)
        
        # Configure X-axis (bottom) to show 10-second intervals
        bottom_axis = self._pi.getAxis('bottom')
        bottom_axis.setStyle(showValues=True, tickTextOffset=2)
        bottom_axis.setHeight(15) # Give the numbers room to breathe
        
        # Force major ticks every 10 seconds up to PLOT_WINDOW_S (60s)
        ticks = [(i, str(i)) for i in range(0, int(PLOT_WINDOW_S) + 1, 10)]
        bottom_axis.setTicks([ticks]) 
        
        # Turn on the vertical grid lines so the 10s intervals are highly visible
        self._pi.showGrid(x=True, y=True, alpha=0.12)

        # Performance: auto-downsample + clip off-screen data
        for curve in (self._raw_curve, self._mean_curve,
                      self._ci_up, self._ci_dn):
            curve.setDownsampling(auto=True, method='subsample')
            curve.setClipToView(True)

        # Initial title
        self._set_title(DroneState.IDLE, None, 100.0, 0, 0.0)

    def _set_title(self, state: DroneState, node_id: Optional[int],
                   batt: float, acf_n: int, t_eff: float) -> None:
        r, g, b   = self._rgb
        node_str  = f"→n{node_id}" if node_id is not None else ""
        t_str     = f"T_eff={t_eff:.1f}s" if acf_n >= ACF_BUF_LEN else "T_eff=prior"
        self._pi.setTitle(
            f"<span style='color:rgb({r},{g},{b});font-family:monospace;"
            f"font-size:8pt;'>"
            f"D{self._id} {node_str} | {state.name} | {t_str} | "
            f"batt={batt:.0f}%</span>"
        )

    def update(self, ds: DroneSnapshot) -> None:
        raw  = ds.hist_raw.astype(np.float32)
        mean = ds.hist_mean.astype(np.float32)
        ci   = ds.hist_ci.astype(np.float32)

        ci_abs = ci * np.abs(mean)

        self._raw_curve.setData(self._x, raw)
        self._mean_curve.setData(self._x, mean)
        self._ci_up.setData(self._x, mean + ci_abs)
        self._ci_dn.setData(self._x, mean - ci_abs)

        self._set_title(ds.state, ds.target_node_id,
                        ds.batt_pct, ds.acf_n, ds.T_int_eff_est)


# ─────────────────────────────────────────────────────────────────────────────
# HUD status bar
# ─────────────────────────────────────────────────────────────────────────────

class HUDLabel:
    def __init__(self, qlabel) -> None:
        self._label = qlabel
        self._label.setStyleSheet(
            "QLabel {"
            "  color: rgb(200,210,230);"
            "  background: rgba(18,20,28,210);"
            "  font-family: 'Courier New', monospace;"
            "  font-size: 11pt;"
            "  padding: 6px 14px;"
            "  border-bottom: 1px solid rgba(75,85,125,130);"
            "}"
        )
        self._label.setText("LAWSS  |  Initialising …")

    def update(self, snap: EnvSnapshot) -> None:
        rth_part = (
            f"  |  <span style='color:#ff4444;'>RTH: {snap.n_rth}</span>"
            if snap.n_rth > 0 else ""
        )
        sep_col = (
            "#ff5050" if snap.min_sep_m < 4.0 else
            "#ffcc00" if snap.min_sep_m < 6.0 else
            "#55ee88"
        )
        self._label.setText(
            f"LAWSS Stage 4"
            f"  |  <b>t = {snap.elapsed_s:7.1f} s</b>"
            f"  ({snap.elapsed_s / 60:.2f} min)"
            f"  |  nodes: <b>{snap.n_measured}/{N_TARGETS}</b>"
            f"  |  assigns: {snap.assignments_made}"
            f"  |  Min Sep: <span style='color:{sep_col};'>"
            f"{snap.min_sep_m:5.2f} m</span>"
            f"{rth_part}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main window  (assembled only when Qt is available)
# ─────────────────────────────────────────────────────────────────────────────

def _build_main_window(env: Environment):
    """
    Construct and return the MainWindow object.
    All Qt imports happen inside this function so that importing this
    module (for headless tests) never touches Qt.
    """
    try:
        from PySide6.QtWidgets import (
            QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        )
        from PySide6.QtCore import Qt, QTimer
    except ImportError:
        from PyQt6.QtWidgets import ( # type: ignore
            QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        )
        from PyQt6.QtCore import Qt, QTimer # type: ignore

    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    # Global pyqtgraph settings (must be called before any widget creation)
    pg.setConfigOptions(antialias=True, useOpenGL=True, enableExperimental=True)
    pg.setConfigOption('background', BG_DARK)
    pg.setConfigOption('foreground', FG_TEXT)

    # ── Shared data channel ───────────────────────────────────────────────────
    snap_queue: "queue.Queue[EnvSnapshot]" = queue.Queue(maxsize=4)

    # ── Simulation thread ─────────────────────────────────────────────────────
    worker = SimWorker(env, snap_queue)

    # ── Qt window ─────────────────────────────────────────────────────────────
    win = QMainWindow()
    win.setWindowTitle("LAWSS Stage 4  —  Drone Swarm Dashboard")
    win.resize(1600, 900)
    win.setStyleSheet(
        f"QMainWindow {{ background-color: rgb{BG_DARK}; }}"
        f"QWidget      {{ background-color: rgb{BG_DARK}; }}"
    )

    central = QWidget()
    win.setCentralWidget(central)
    root = QVBoxLayout(central)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    # HUD bar (full width, top of window)
    hud_label = QLabel()
    hud_label.setTextFormat(Qt.TextFormat.RichText)
    hud = HUDLabel(hud_label)
    root.addWidget(hud_label)

    # Horizontal split: 3-D view (left) | 2-D grid (right)
    content        = QWidget()
    content_layout = QHBoxLayout(content)
    content_layout.setContentsMargins(4, 4, 4, 4)
    content_layout.setSpacing(6)
    root.addWidget(content, stretch=1)

    # ── 3-D GLViewWidget ──────────────────────────────────────────────────────
    gl_view = gl.GLViewWidget()
    gl_view.setMinimumWidth(620)
    content_layout.addWidget(gl_view, stretch=55)
    scene3d = Scene3D(gl_view)

    # ── 2-D panel grid  (GraphicsLayoutWidget → 5 rows × 2 cols) ─────────────
    #
    # GraphicsLayoutWidget is the correct container for a grid of PlotItems.
    # Each cell is obtained via glw.addPlot(row=r, col=c) which returns a
    # PlotItem (not a PlotWidget), so DronePanel receives the right object type.
    #
    glw = pg.GraphicsLayoutWidget()
    glw.setBackground(BG_DARK)
    glw.setMinimumWidth(480)
    content_layout.addWidget(glw, stretch=45)

    drone_panels: List[DronePanel] = []
    for row in range(5):
        for col in range(2):
            drone_id = row * 2 + col
            # addPlot returns a PlotItem — exactly what DronePanel expects
            pi = glw.addPlot(row=row, col=col)
            pi.setContentsMargins(1, 1, 1, 1)
            drone_panels.append(DronePanel(pi, drone_id))

    # ── GUI timer — drains the snapshot queue on the GUI thread ───────────────
    def _on_timer() -> None:
        snap = None
        # Drain all queued snapshots; display only the latest
        while True:
            try:
                snap = snap_queue.get_nowait()
            except queue.Empty:
                break
        if snap is not None:
            hud.update(snap)
            scene3d.update(snap)
            for panel, ds in zip(drone_panels, snap.drones):
                panel.update(ds)

    timer = QTimer(win)
    timer.setInterval(GUI_TIMER_MS)
    timer.timeout.connect(_on_timer)

    # ── Window close handler ──────────────────────────────────────────────────
    def _on_close(event) -> None:
        timer.stop()
        worker.stop()
        event.accept()

    win.closeEvent = _on_close   # type: ignore[method-assign]

    # ── Start everything ──────────────────────────────────────────────────────
    worker.start()
    timer.start()
    win.show()

    return win, worker, timer   # keep references alive for the event loop


# ─────────────────────────────────────────────────────────────────────────────
# Live dashboard entry point
# ─────────────────────────────────────────────────────────────────────────────

class Dashboard:
    """
    Owns the LAWSS simulation dashboard in either live-GUI or headless
    video-export mode.
    """

    def __init__(self, seed: int = 0) -> None:
        print("Building MPC solvers ... (one cold-start per drone)")
        t0 = time.perf_counter()
        self.env = Environment(seed=seed)
        print(f"Build time: {time.perf_counter() - t0:.2f} s\n")

    def run(
        self,
        export_video: bool = False,
        filename: str = "LAWSS_Simulation.mp4",
        fps: int = 60,
        dpi: int = 200,
    ) -> None:
        if export_video:
            self.export_video(filename=filename, fps=fps, dpi=dpi)
            return

        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            from PyQt6.QtWidgets import QApplication # type: ignore

        app = QApplication.instance() or QApplication(sys.argv)
        win, worker, timer = _build_main_window(self.env)

        ret = app.exec()
        worker.stop()
        sys.exit(ret)

    def export_video(
        self,
        filename: str = "LAWSS_Simulation.mp4",
        fps: int = 60,
        dpi: int = 200,
    ) -> None:
        """
        Render the dashboard to an MP4 using Matplotlib's FFMpegWriter.

        This path does not create a Qt application or open the live GUI. The
        simulation advances as fast as CPU/rendering allows; no time.sleep()
        pacing is used.
        """
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib import animation

        if fps <= 0:
            raise ValueError("fps must be a positive integer")
        if dpi <= 0:
            raise ValueError("dpi must be a positive integer")
        ffmpeg_path = self._ensure_ffmpeg_writer(animation)

        fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor=self._mpl_rgb(BG_DARK))
        gs = fig.add_gridspec(
            6, 4,
            height_ratios=[0.34, 1, 1, 1, 1, 1],
            width_ratios=[1.45, 1.45, 1.0, 1.0],
            left=0.025,
            right=0.985,
            top=0.965,
            bottom=0.045,
            wspace=0.22,
            hspace=0.55,
        )

        hud_ax = fig.add_subplot(gs[0, :])
        scene_ax = fig.add_subplot(gs[1:, :2], projection="3d")
        panel_axes = [
            fig.add_subplot(gs[row + 1, col + 2])
            for row in range(5)
            for col in range(2)
        ]
        trails: List[deque] = [deque(maxlen=TRAIL_LEN) for _ in range(N_DRONES)]
        x_axis = np.linspace(0.0, PLOT_WINDOW_S, PLOT_WIN_SAMPLES, dtype=np.float32)

        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=8000,
            metadata={
                "title": "LAWSS Stage 4 Simulation",
                "artist": "LAWSS Dashboard",
            },
            extra_args=[
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "18",
            ],
        )

        budget_frames = int(np.ceil(BATTERY_CAPACITY_S / (DT * STEPS_PER_FRAME)))
        width_px = int(fig.get_figwidth() * dpi)
        height_px = int(fig.get_figheight() * dpi)
        print(
            f"Exporting {filename} at {fps} fps "
            f"({budget_frames} max frames, {width_px}x{height_px}, dpi={dpi})"
        )
        print("Using Matplotlib FFMpegWriter with writer='ffmpeg' / libx264.")
        if ffmpeg_path:
            print(f"Using bundled ffmpeg executable: {ffmpeg_path}")

        try:
            with writer.saving(fig, filename, dpi=dpi):
                for frame in range(budget_frames):
                    for _ in range(STEPS_PER_FRAME):
                        if self.env.elapsed_s >= BATTERY_CAPACITY_S:
                            break
                        self.env.step()

                    snap = _build_snapshot(self.env)
                    self._draw_video_frame(
                        fig=fig,
                        hud_ax=hud_ax,
                        scene_ax=scene_ax,
                        panel_axes=panel_axes,
                        trails=trails,
                        x_axis=x_axis,
                        snap=snap,
                    )
                    writer.grab_frame(facecolor=fig.get_facecolor())

                    if frame % max(fps, 1) == 0:
                        print(
                            f"  frame {frame:>5}/{budget_frames} | "
                            f"t={snap.elapsed_s:7.1f}s | "
                            f"nodes={snap.n_measured}/{N_TARGETS}"
                        )

                    if self.env.all_measured or self.env.elapsed_s >= BATTERY_CAPACITY_S:
                        break
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg was not found. Install ffmpeg and ensure the ffmpeg "
                "executable is on PATH, then rerun with --export-video."
            ) from exc
        finally:
            plt.close(fig)

        print(f"Video export complete: {filename}")

    @staticmethod
    def _mpl_rgb(rgb: tuple) -> tuple:
        return tuple(channel / 255.0 for channel in rgb)

    @staticmethod
    def _ensure_ffmpeg_writer(animation_module) -> Optional[str]:
        """
        Ensure Matplotlib can locate a real ffmpeg executable.

        The PyPI packages named "ffmpeg" / "python-ffmpeg" are wrappers and do
        not necessarily install ffmpeg.exe. imageio-ffmpeg, if present, bundles
        a working binary that Matplotlib can use via animation.ffmpeg_path.
        """
        import matplotlib as mpl

        if animation_module.writers.is_available("ffmpeg"):
            return None

        try:
            import imageio_ffmpeg
        except ImportError:
            imageio_ffmpeg = None

        if imageio_ffmpeg is not None:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
            if animation_module.writers.is_available("ffmpeg"):
                return ffmpeg_exe

        raise RuntimeError(
            "Matplotlib cannot find a real ffmpeg executable. "
            "Note: `pip install ffmpeg` installs a Python wrapper, not ffmpeg.exe. "
            "Install a binary with `pip install imageio-ffmpeg`, or install system "
            "ffmpeg and put ffmpeg.exe on PATH. You can also set "
            "matplotlib.rcParams['animation.ffmpeg_path'] to the full ffmpeg.exe path."
        )

    @staticmethod
    def _format_hud(snap: EnvSnapshot) -> str:
        return (
            f"LAWSS Stage 4  |  t = {snap.elapsed_s:7.1f} s "
            f"({snap.elapsed_s / 60:.2f} min)  |  "
            f"nodes: {snap.n_measured}/{N_TARGETS}  |  "
            f"assigns: {snap.assignments_made}  |  "
            f"Min Sep: {snap.min_sep_m:5.2f} m  |  RTH: {snap.n_rth}"
        )

    def _style_axis_2d(self, ax) -> None:
        bg = self._mpl_rgb(BG_DARK)
        fg = self._mpl_rgb(FG_TEXT)
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_color((0.30, 0.34, 0.48, 0.7))
            spine.set_linewidth(0.6)
        ax.tick_params(colors=fg, labelsize=6, length=2, pad=1)
        ax.grid(True, color=(0.55, 0.60, 0.78, 0.12), linewidth=0.5)
        ax.set_xlim(0.0, PLOT_WINDOW_S)
        ax.set_ylim(U_WAKE - 5.0, U_WAKE + 5.0)
        ax.set_xticks(np.arange(0, int(PLOT_WINDOW_S) + 1, 10))
        ax.axhline(U_WAKE, color=(0.50, 0.50, 0.58, 0.45), linewidth=0.7)

    def _style_axis_3d(self, ax) -> None:
        bg = self._mpl_rgb(BG_DARK)
        fg = self._mpl_rgb(FG_TEXT)
        ax.set_facecolor(bg)
        ax.set_xlim(0.0, ENV_LIM)
        ax.set_ylim(0.0, ENV_LIM)
        ax.set_zlim(0.0, ENV_LIM)
        ax.view_init(elev=22, azim=-55)
        ax.set_box_aspect((1, 1, 1))
        ax.tick_params(colors=fg, labelsize=7, pad=0)
        ax.xaxis.pane.set_facecolor((*bg, 1.0))
        ax.yaxis.pane.set_facecolor((*bg, 1.0))
        ax.zaxis.pane.set_facecolor((*bg, 1.0))
        ax.xaxis.pane.set_edgecolor((0.30, 0.34, 0.48, 0.55))
        ax.yaxis.pane.set_edgecolor((0.30, 0.34, 0.48, 0.55))
        ax.zaxis.pane.set_edgecolor((0.30, 0.34, 0.48, 0.55))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0.30, 0.34, 0.48, 0.25)
            axis._axinfo["grid"]["linewidth"] = 0.5
        ax.set_xlabel("X", color=fg, labelpad=-8)
        ax.set_ylabel("Y", color=fg, labelpad=-8)
        ax.set_zlabel("Z", color=fg, labelpad=-8)

    def _draw_video_frame(
        self,
        fig,
        hud_ax,
        scene_ax,
        panel_axes: List,
        trails: List[deque],
        x_axis: np.ndarray,
        snap: EnvSnapshot,
    ) -> None:
        bg = self._mpl_rgb(BG_DARK)
        fg = self._mpl_rgb(FG_TEXT)

        hud_ax.clear()
        hud_ax.set_facecolor(bg)
        hud_ax.set_xticks([])
        hud_ax.set_yticks([])
        for spine in hud_ax.spines.values():
            spine.set_visible(False)
        hud_ax.text(
            0.01, 0.5, self._format_hud(snap),
            color=fg,
            fontsize=12,
            fontfamily="monospace",
            va="center",
            ha="left",
            transform=hud_ax.transAxes,
        )

        scene_ax.clear()
        self._style_axis_3d(scene_ax)

        node_colors = []
        for n in range(N_TARGETS):
            if snap.target_measured[n]:
                node_colors.append(COL_NODE_MEASURED)
            elif snap.target_locked[n]:
                node_colors.append(COL_NODE_LOCKED)
            else:
                node_colors.append(COL_NODE_FREE)
        node_colors = np.array(node_colors)
        scene_ax.scatter(
            snap.target_positions[:, 0],
            snap.target_positions[:, 1],
            snap.target_positions[:, 2],
            s=12,
            c=node_colors,
            depthshade=False,
        )

        drone_positions = np.array([ds.position for ds in snap.drones])
        drone_colors = np.array(DRONE_COLORS_F, dtype=float)
        drone_sizes = np.full(N_DRONES, 38.0)
        for ds in snap.drones:
            trails[ds.id].append(ds.position.copy())
            if ds.state == DroneState.IDLE:
                drone_colors[ds.id, 3] = 0.30
                drone_sizes[ds.id] = 22.0
            elif ds.state == DroneState.RTH:
                drone_colors[ds.id] = np.array([1.0, 0.12, 0.12, 1.0])
                drone_sizes[ds.id] = 55.0
            elif ds.state == DroneState.SAMPLING:
                drone_sizes[ds.id] = 60.0

        scene_ax.scatter(
            drone_positions[:, 0],
            drone_positions[:, 1],
            drone_positions[:, 2],
            s=drone_sizes,
            c=drone_colors,
            depthshade=False,
            edgecolors=(0.95, 0.96, 1.0, 0.28),
            linewidths=0.4,
        )

        for i, trail in enumerate(trails):
            if len(trail) >= 2:
                pts = np.array(trail)
                r, g, b = DRONE_COLORS_F[i][:3]
                scene_ax.plot(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    color=(r, g, b, 0.35),
                    linewidth=1.0,
                )

        for i, ds in enumerate(snap.drones):
            active = (
                ds.state in (DroneState.TRANSIT, DroneState.RTH)
                and ds.target_position is not None
            )
            if active:
                r, g, b = DRONE_COLORS_F[i][:3]
                color = (1.0, 0.08, 0.08, 0.60) if ds.state == DroneState.RTH else (r, g, b, 0.28)
                seg = np.array([ds.position, ds.target_position])
                scene_ax.plot(
                    seg[:, 0], seg[:, 1], seg[:, 2],
                    color=color,
                    linewidth=0.9,
                )

        for ax, ds in zip(panel_axes, snap.drones):
            ax.clear()
            self._style_axis_2d(ax)
            r, g, b, _ = DRONE_COLORS_F[ds.id]
            raw = ds.hist_raw.astype(np.float32)
            mean = ds.hist_mean.astype(np.float32)
            ci = ds.hist_ci.astype(np.float32)
            ci_abs = ci * np.abs(mean)
            node_str = f"->n{ds.target_node_id}" if ds.target_node_id is not None else ""
            t_str = f"T_eff={ds.T_int_eff_est:.1f}s" if ds.acf_n >= ACF_BUF_LEN else "T_eff=prior"

            ax.plot(x_axis, raw, color=(r, g, b, 0.25), linewidth=0.45)
            ax.plot(x_axis, mean, color=(r, g, b, 0.95), linewidth=1.2)
            ax.plot(x_axis, mean + ci_abs, color=(r, g, b, 0.40), linewidth=0.55, linestyle="--")
            ax.plot(x_axis, mean - ci_abs, color=(r, g, b, 0.40), linewidth=0.55, linestyle="--")
            ax.set_title(
                f"D{ds.id} {node_str} | {ds.state.name} | {t_str} | batt={ds.batt_pct:.0f}%",
                color=(r, g, b, 1.0),
                fontsize=6.5,
                fontfamily="monospace",
                pad=2,
            )

        fig.canvas.draw_idle()


def run_dashboard(seed: int = 0) -> None:
    Dashboard(seed=seed).run()


def export_dashboard_video(
    seed: int = 0,
    filename: str = "LAWSS_Simulation.mp4",
    fps: int = 60,
    dpi: int = 200,
) -> None:
    Dashboard(seed=seed).export_video(filename=filename, fps=fps, dpi=dpi)


def run_dashboard_legacy(seed: int = 0) -> None:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        from PyQt6.QtWidgets import QApplication # type: ignore

    app = QApplication.instance() or QApplication(sys.argv)

    print("Building MPC solvers … (one cold-start per drone)")
    t0 = time.perf_counter()
    env = Environment(seed=seed)
    print(f"Build time: {time.perf_counter() - t0:.2f} s\n")

    win, worker, timer = _build_main_window(env)

    ret = app.exec()
    worker.stop()
    sys.exit(ret)


# ─────────────────────────────────────────────────────────────────────────────
# Headless smoke test  (no Qt / OpenGL required)
# ─────────────────────────────────────────────────────────────────────────────

def _headless_smoke_test(n_frames: int = 30) -> None:
    """
    Verify Stage 4 correctness without a display.

    Assertions
    ──────────
    A. _build_snapshot() returns a valid EnvSnapshot every frame.
    B. Physics advances by exactly STEPS_PER_FRAME × DT per call.
    C. The simulation runs to battery budget/completion and measures nodes.
    D. No exception during snapshot construction.
    E. All history arrays have length == PLOT_WIN_SAMPLES.
    F. Wake velocity histories are centered on U_WAKE when samples exist.
    """
    print("=" * 70)
    print("  LAWSS Stage 4  PyQtGraph — Headless Smoke Test")
    print(f"  Simulating {n_frames} frames  "
          f"({n_frames * STEPS_PER_FRAME * DT:.1f} s physics)")
    print(f"  Battery budget: {BATTERY_CAPACITY_S:.0f} s  |  Targets: {N_TARGETS}")
    print(f"  STEPS_PER_FRAME={STEPS_PER_FRAME}  DT={DT}")
    print("=" * 70)

    print("  Building MPC solvers …")
    t_build = time.perf_counter()
    env = Environment(seed=0)
    print(f"  Build time: {time.perf_counter() - t_build:.2f} s\n")

    t_start = time.perf_counter()

    for frame in range(n_frames):
        t_before = env.elapsed_s

        for _ in range(STEPS_PER_FRAME):
            env.step()

        t_after = env.elapsed_s

        # B: physics tick check
        expected = STEPS_PER_FRAME * DT
        actual   = t_after - t_before
        assert abs(actual - expected) < 1e-9, (
            f"Frame {frame}: physics advanced {actual:.6f}s, expected {expected:.6f}s"
        )

        # A + D: snapshot construction
        try:
            snap = _build_snapshot(env)
        except Exception as exc:
            raise AssertionError(
                f"Frame {frame}: _build_snapshot() raised {exc}"
            ) from exc

        # A: type and shape checks
        assert isinstance(snap, EnvSnapshot),                    "FAIL A: wrong type"
        assert len(snap.drones)              == N_DRONES,        "FAIL A: drone count"
        assert snap.target_positions.shape[0] == N_TARGETS,     "FAIL A: target count"

        # E: history array lengths
        for ds in snap.drones:
            assert len(ds.hist_raw)  == PLOT_WIN_SAMPLES, "FAIL E: hist_raw length"
            assert len(ds.hist_mean) == PLOT_WIN_SAMPLES, "FAIL E: hist_mean length"
            assert len(ds.hist_ci)   == PLOT_WIN_SAMPLES, "FAIL E: hist_ci length"
            assert len(ds.hist_T_eff) == PLOT_WIN_SAMPLES, "FAIL E: hist_T_eff length"

            finite_mean = ds.hist_mean[np.isfinite(ds.hist_mean)]
            if finite_mean.size >= 10:
                mean_err = abs(float(np.nanmean(finite_mean)) - U_WAKE)
                assert mean_err < 5.0, (
                    f"FAIL F: D{ds.id} running mean is {mean_err:.2f} m/s "
                    f"from U_WAKE={U_WAKE:.2f}"
                )

        if frame % 5 == 0 or frame == n_frames - 1:
            n_rth = sum(1 for d in snap.drones if d.state == DroneState.RTH)
            print(
                f"  frame {frame:>3}  |  t={snap.elapsed_s:6.2f}s"
                f"  |  nodes {snap.n_measured}/{N_TARGETS}"
                f"  |  RTH={n_rth}"
                f"  |  min_sep={snap.min_sep_m:.2f}m"
            )

    wall = time.perf_counter() - t_start

    # C: run to completion or battery budget
    print("\n  Running to battery budget / completion …")
    budget_ticks = int(BATTERY_CAPACITY_S / DT)
    ticks_done   = n_frames * STEPS_PER_FRAME

    for _ in range(budget_ticks - ticks_done):
        if env.all_measured:
            break
        env.step()

    assert env.elapsed_s <= BATTERY_CAPACITY_S + DT, (
        f"FAIL C: elapsed time {env.elapsed_s:.1f}s exceeded "
        f"battery budget {BATTERY_CAPACITY_S:.1f}s"
    )
    assert env.n_measured > 0, (
        f"FAIL C: no nodes measured after {BATTERY_CAPACITY_S:.0f} s"
    )

    print(f"\n  Smoke test complete.  Wall time for {n_frames} frames: {wall:.2f} s")
    print(
        f"  Nodes measured at t = {env.elapsed_s:.1f} s: "
        f"{env.n_measured}/{N_TARGETS}"
    )
    print()
    print("  Assertions:")
    print("  A — _build_snapshot() returns valid EnvSnapshot every frame   OK")
    print("  B — Physics advances by exactly STEPS_PER_FRAME×DT            OK")
    print("  C — Simulation runs to budget/completion and measures nodes     OK")
    print("  D — No snapshot construction exception in any frame            OK")
    print(f"  E — All history arrays have length == {PLOT_WIN_SAMPLES}          OK")
    print("  F — Sampled wake histories are centered on U_WAKE               OK")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LAWSS Stage 4  PyQtGraph Dashboard / Headless Test"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run headless smoke test instead of live dashboard",
    )
    parser.add_argument(
        "--frames", type=int, default=30,
        help="Number of frames for headless test (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the Environment (default: 0)",
    )
    parser.add_argument(
        "--export-video", action="store_true",
        help="Export a headless MP4 video instead of opening the live GUI",
    )
    parser.add_argument(
        "--video-filename", type=str, default="LAWSS_Simulation.mp4",
        help="Output MP4 filename for --export-video (default: LAWSS_Simulation.mp4)",
    )
    parser.add_argument(
        "--video-fps", type=int, default=60,
        help="Output frame rate for --export-video (default: 60)",
    )
    parser.add_argument(
        "--video-dpi", type=int, default=200,
        help="Output DPI for --export-video (default: 200, produces 3200x1800)",
    )
    args = parser.parse_args()

    if args.headless:
        _headless_smoke_test(n_frames=args.frames)
    elif args.export_video:
        Dashboard(seed=args.seed).run(
            export_video=True,
            filename=args.video_filename,
            fps=args.video_fps,
            dpi=args.video_dpi,
        )
    else:
        run_dashboard(seed=args.seed)
