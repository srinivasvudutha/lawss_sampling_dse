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
    from lawss_stage3_inlet import (
        Environment,
        DroneState,
        N_DRONES,
        N_TARGETS,
        U_MEAN,
        DT,
        BATTERY_CAPACITY_S,
        LAG1_WARMUP,
    )
except ModuleNotFoundError:
    print(
        "ERROR: lawss_stage3_inlet.py not found. "
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

# Maximum number of panels shown in the 2-D dashboard grid (5 rows × 2 cols)
MAX_DASHBOARD_PANELS: int = 10

def _make_drone_colors(n: int) -> tuple:
    """
    Generate `n` visually distinct RGBA colours using the golden-ratio HSV
    method.  Returns (colors_f, colors_i) where:
      colors_f – list of (r, g, b, 1.0) float tuples  (0–1 range)
      colors_i – list of (r, g, b)       int   tuples  (0–255 range)
    Works for any n ≥ 1, including n > 10.
    """
    import colorsys
    # Start with 10 hand-picked colours so the first 10 drones look great,
    # then fall back to the golden-ratio generator for extras.
    _fixed = [
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
    colors_f = list(_fixed[:n])
    # Golden-ratio fill for any drone beyond index 9
    golden = 0.618033988749895
    hue = 0.61  # start hue for extras
    for _ in range(n - len(colors_f)):
        hue = (hue + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors_f.append((r, g, b, 1.0))
    colors_i = [tuple(int(x * 255) for x in c[:3]) for c in colors_f]
    return colors_f, colors_i

# Build palettes sized to N_DRONES at import time
DRONE_COLORS_F, DRONE_COLORS_I = _make_drone_colors(N_DRONES)

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
    T_int_est:        float
    L_n:              int
    samples_this_run: int
    target_node_id:   Optional[int]
    hist_raw:         np.ndarray          # (PLOT_WIN_SAMPLES,)
    hist_mean:        np.ndarray
    hist_ci:          np.ndarray


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

        drone_snaps.append(DroneSnapshot(
            id               = d.id,
            position         = d.position.copy(),
            state            = d.state,
            target_position  = (d.target_position.copy()
                                if d.target_position is not None else None),
            batt_pct         = 100.0 * d.battery_remaining_s / BATTERY_CAPACITY_S,
            wf_mean          = d.stats.wf_mean,
            T_int_est        = d.stats.T_int_est,
            L_n              = d.stats.L_n,
            samples_this_run = d.samples_this_run,
            target_node_id   = d.target_node_id,
            hist_raw         = raw_full [-PLOT_WIN_SAMPLES:].copy(),
            hist_mean        = mean_full[-PLOT_WIN_SAMPLES:].copy(),
            hist_ci          = ci_full  [-PLOT_WIN_SAMPLES:].copy(),
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

    def __init__(self, env: Environment, out_queue: "queue.Queue[EnvSnapshot]"):
        super().__init__(daemon=True, name="SimWorker")
        self._env       = env
        self._queue     = out_queue
        self._stop_flag = threading.Event()

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

            # Pace to real time
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
        drone_colors = np.array(DRONE_COLORS_F, dtype=np.float32)  # (N_DRONES, 4)

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
    Plus a horizontal InfiniteLine at U_MEAN for reference.
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

        # U_MEAN reference — InfiniteLine is the correct pyqtgraph API
        ref_line = pg.InfiniteLine(
            pos   = float(U_MEAN),
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
        self._pi.setYRange(U_MEAN - 4.5, U_MEAN + 4.5, padding=0)
        self._pi.setXRange(0.0, PLOT_WINDOW_S, padding=0)
        self._pi.getAxis('left').setWidth(30)
        self._pi.getAxis('left').setStyle(tickTextOffset=2)
        self._pi.getAxis('bottom').setStyle(showValues=False)
        self._pi.showGrid(x=False, y=True, alpha=0.12)
        self._pi.setMenuEnabled(False)
        self._pi.setMouseEnabled(x=False, y=False)

        # Performance: auto-downsample + clip off-screen data
        for curve in (self._raw_curve, self._mean_curve,
                      self._ci_up, self._ci_dn):
            curve.setDownsampling(auto=True, method='subsample')
            curve.setClipToView(True)

        # Initial title
        self._set_title(DroneState.IDLE, None, 100.0, 0, 0.0)

    def _set_title(self, state: DroneState, node_id: Optional[int],
                   batt: float, l_n: int, t_int: float) -> None:
        r, g, b   = self._rgb
        node_str  = f"→n{node_id}" if node_id is not None else ""
        t_str     = f"T={t_int:.1f}s" if l_n > LAG1_WARMUP else "T=prior"
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
                        ds.batt_pct, ds.L_n, ds.T_int_est)


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
        from PyQt6.QtWidgets import (
            QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        )
        from PyQt6.QtCore import Qt, QTimer

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

    # Build exactly MAX_DASHBOARD_PANELS (10) plots regardless of N_DRONES.
    # Each panel is assigned to drone IDs 0..min(N_DRONES,10)-1; if
    # N_DRONES < 10, only the first N_DRONES panels are populated.
    n_panels = min(MAX_DASHBOARD_PANELS, N_DRONES)
    drone_panels: List[DronePanel] = []
    for row in range(5):
        for col in range(2):
            drone_id = row * 2 + col
            pi = glw.addPlot(row=row, col=col)
            pi.setContentsMargins(1, 1, 1, 1)
            if drone_id < n_panels:
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
            # Only update the panels that were created (first n_panels drones)
            for panel, ds in zip(drone_panels, snap.drones[:n_panels]):
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

def run_dashboard(seed: int = 0) -> None:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        from PyQt6.QtWidgets import QApplication

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
    C. All N_TARGETS nodes are measured within BATTERY_CAPACITY_S.
    D. No exception during snapshot construction.
    E. All history arrays have length == PLOT_WIN_SAMPLES.
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

        if frame % 5 == 0 or frame == n_frames - 1:
            n_rth = sum(1 for d in snap.drones if d.state == DroneState.RTH)
            print(
                f"  frame {frame:>3}  |  t={snap.elapsed_s:6.2f}s"
                f"  |  nodes {snap.n_measured}/{N_TARGETS}"
                f"  |  RTH={n_rth}"
                f"  |  min_sep={snap.min_sep_m:.2f}m"
            )

    wall = time.perf_counter() - t_start

    # C: run to completion within battery budget
    print("\n  Running to completion …")
    budget_ticks = int(BATTERY_CAPACITY_S / DT)
    ticks_done   = n_frames * STEPS_PER_FRAME

    for _ in range(budget_ticks - ticks_done):
        if env.all_measured:
            break
        env.step()

    assert env.all_measured, (
        f"FAIL C: only {env.n_measured}/{N_TARGETS} nodes measured "
        f"after {BATTERY_CAPACITY_S:.0f} s"
    )

    print(f"\n  Smoke test complete.  Wall time for {n_frames} frames: {wall:.2f} s")
    print(f"  All nodes measured at t = {env.elapsed_s:.1f} s")
    print()
    print("  Assertions:")
    print("  A — _build_snapshot() returns valid EnvSnapshot every frame   ✓")
    print("  B — Physics advances by exactly STEPS_PER_FRAME×DT            ✓")
    print(f"  C — All {N_TARGETS} nodes measured within battery budget        ✓")
    print("  D — No snapshot construction exception in any frame            ✓")
    print(f"  E — All history arrays have length == {PLOT_WIN_SAMPLES}          ✓")
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
    args = parser.parse_args()

    if args.headless:
        _headless_smoke_test(n_frames=args.frames)
    else:
        run_dashboard(seed=args.seed)