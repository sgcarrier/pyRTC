# app_gui_qt_mockable.py
"""
PyQt5 GUI (viewer + controls) with an optional --mock mode so you can develop
and test without connecting to a device or shared memory.

Usage:
  # Mock mode (no SHM/pyRTC needed)
  python app_gui_qt_mockable.py --mock --fps 30 --shape 480x640

  # Real SHM mode (requires pyRTC)
  python app_gui_qt_mockable.py <shm_name> --fps 30 [--vmin ...] [--vmax ...]
"""

from __future__ import annotations

import sys
import argparse
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QFrame,
    QSizePolicy,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm


# -----------------------------
# Application state container
# -----------------------------
@dataclass
class AppState:
    value_a: float = 0.0
    value_b: float = 0.0
    items: Dict[str, bool] = field(
        default_factory=lambda: {
            "Item A": False,
            "Item B": False,
            "Item C": False,
            "Item D": False,
        }
    )


# -----------------------------
# Base viewer widget (Matplotlib image)
# -----------------------------
class BaseViewer(QWidget):
    infoUpdated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 5), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.log = False
        self.LinearNorm = None
        self.vmin = None
        self.vmax = None

    def toggle_log(self):
        self.log = not self.log
        if self.LinearNorm is None:
            return
        if self.log:
            self.im.set_norm(LogNorm(vmin=max(1e-2, (self.vmax or 1e0)/1000), vmax=max(1e-2, self.vmax or 1)))
        else:
            self.im.set_norm(self.LinearNorm)
        self.canvas.draw_idle()


# -----------------------------
# Mock viewer (no SHM required)
# -----------------------------
class MockViewer(BaseViewer):
    def __init__(self, fps: int = 30, shape: Tuple[int, int] = (480, 640), parent=None):
        super().__init__(parent)
        self.shape = shape
        self.t0 = time.time()
        frame0 = self._generate_frame(0.0)
        self.vmin, self.vmax = float(frame0.min()), float(frame0.max())
        self.im = self.axes.imshow(frame0, cmap='inferno', interpolation='nearest', origin='upper', vmin=self.vmin, vmax=self.vmax)
        self.LinearNorm = self.im.norm
        self.cbar = self.figure.colorbar(self.im, ax=self.axes)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(max(1, int(1000 // max(1, fps))))
        self._last_frames = 0
        self._last_time = self.t0
        self._counter = 0

    def _generate_frame(self, t: float) -> np.ndarray:
        # Moving Gaussian blob + waves + noise for a lively mock feed
        h, w = self.shape
        y = np.linspace(-1, 1, h)[:, None]
        x = np.linspace(-1, 1, w)[None, :]
        cx = 0.5 * math.sin(0.7 * t)
        cy = 0.5 * math.cos(0.9 * t)
        r2 = (x - cx)**2 + (y - cy)**2
        blob = np.exp(-r2 * 8.0)
        waves = 0.2 * (np.sin(10*x + 2*t) + np.cos(10*y - 1.5*t))
        noise = 0.05 * np.random.randn(h, w)
        frame = blob + waves + noise
        frame -= frame.min()
        frame /= (frame.max() + 1e-9)
        return frame.astype(np.float32)

    def _tick(self):
        self._counter += 1
        t = time.time() - self.t0
        frame = self._generate_frame(t)
        self.vmin, self.vmax = float(frame.min()), float(frame.max())
        self.im.set_data(frame)
        self.im.set_clim(self.vmin, self.vmax)
        self.cbar.update_normal(self.im)
        self.canvas.draw_idle()

        # FPS estimate
        dt = t - self._last_time
        if dt >= 0.5:
            fps = (self._counter - self._last_frames) / dt
            self._last_frames = self._counter
            self._last_time = t
            self.infoUpdated.emit(f"{fps:.2f} FPS | vmin={self.vmin:.3g} vmax={self.vmax:.3g}")


# -----------------------------
# Real SHM viewer (optional import of pyRTC)
# -----------------------------
class SHMViewer(BaseViewer):
    def __init__(self, shm_name: str, fps: int = 30, static_vmin: Optional[float] = None, static_vmax: Optional[float] = None, parent=None):
        super().__init__(parent)
        # Lazy import so mock mode works without pyRTC installed
        from pyRTC.Pipeline import ImageSHM
        from pyRTC.utils import float_to_dtype
        import numpy as np

        self.metadata = ImageSHM(shm_name + "_meta", (ImageSHM.METADATA_SIZE,), np.float64)
        md = self.metadata.read_noblock()
        shm_w, shm_h = int(md[4]), int(md[5])
        shm_h = max(1, shm_h)
        shm_w = max(1, shm_w)
        shm_dtype = float_to_dtype(md[3])
        self.shm = ImageSHM(shm_name, (shm_w, shm_h), shm_dtype)

        frame0 = self.shm.read_noblock_safe()
        self.vmin, self.vmax = float(frame0.min()), float(frame0.max())
        if static_vmin is not None:
            self.vmin = float(static_vmin)
        if static_vmax is not None:
            self.vmax = float(static_vmax)

        aspect = None
        ASPECTCAP = 10
        if shm_w / shm_h < 1/ASPECTCAP or shm_w / shm_h > ASPECTCAP:
            aspect = 'auto'

        self.im = self.axes.imshow(frame0, cmap='inferno', interpolation='nearest', origin='upper', aspect=aspect, vmin=self.vmin, vmax=self.vmax)
        self.LinearNorm = self.im.norm
        self.cbar = self.figure.colorbar(self.im, ax=self.axes)

        self._old_count = 0
        self._old_time = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_view)
        self._timer.start(max(1, int(1000 // max(1, fps))))

    def _update_view(self):
        frame = self.shm.read_noblock()
        md = self.metadata.read_noblock()
        new_count, new_time = md[0], md[1]
        if isinstance(frame, np.ndarray):
            vmin, vmax = float(frame.min()), float(frame.max())
            if self.log:
                vmin, vmax = max(vmin, vmax/1e3), max(1e-2, vmax)
            self.vmin, self.vmax = vmin, vmax
            self.im.set_data(frame)
            self.im.set_clim(self.vmin, self.vmax)
            self.cbar.update_normal(self.im)
            self.canvas.draw_idle()
        if new_time > self._old_time:
            fps = (new_count - self._old_count) / (new_time - self._old_time)
            speed_text = f"{fps:.2f} FPS"
        else:
            speed_text = "PAUSED"
        self._old_count, self._old_time = new_count, new_time
        self.infoUpdated.emit(f"{speed_text} | vmin={self.vmin:.3g} vmax={self.vmax:.3g}")

    def closeEvent(self, event):
        try:
            self.shm.close()
        except Exception:
            pass
        event.accept()


# -----------------------------
# Controls panel
# -----------------------------
class ControlsPanel(QWidget):
    valueChanged = pyqtSignal(str, float)
    itemToggled = pyqtSignal(str, bool)

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_value_group('value_a', 'Set value A:'))
        layout.addWidget(self._build_value_group('value_b', 'Set value B:'))
        items_box = QGroupBox('Items')
        items_layout = QVBoxLayout(items_box)
        for name in self.state.items.keys():
            items_layout.addLayout(self._build_item_row(name))
        layout.addWidget(items_box)
        layout.addStretch(1)

    def _build_value_group(self, var_name: str, label: str) -> QGroupBox:
        g = QGroupBox(label)
        grid = QGridLayout(g)
        cur_label = QLabel('Current:')
        cur_value = QLabel(str(getattr(self.state, var_name)))
        entry = QLineEdit()
        submit = QPushButton('Submit')
        grid.addWidget(cur_label, 0, 0)
        grid.addWidget(cur_value, 0, 1)
        grid.addWidget(entry, 1, 0, 1, 2)
        grid.addWidget(submit, 1, 2)
        def do_submit():
            txt = entry.text().strip()
            if not txt:
                return
            try:
                val = float(txt)
            except ValueError:
                cur_value.setText(f'Invalid: {txt}')
                return
            setattr(self.state, var_name, val)
            cur_value.setText(str(val))
            self.valueChanged.emit(var_name, val)
        submit.clicked.connect(do_submit)
        return g

    def _build_item_row(self, name: str) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(name)
        status = QLabel('OFF')
        start_btn = QPushButton('Start')
        stop_btn = QPushButton('Stop')
        status.setMinimumWidth(30)
        def refresh():
            status.setText('ON' if self.state.items[name] else 'OFF')
        def start():
            self.state.items[name] = True
            refresh()
            self.itemToggled.emit(name, True)
        def stop():
            self.state.items[name] = False
            refresh()
            self.itemToggled.emit(name, False)
        start_btn.clicked.connect(start)
        stop_btn.clicked.connect(stop)
        row.addWidget(label, 1)
        row.addWidget(status)
        row.addWidget(start_btn)
        row.addWidget(stop_btn)
        return row


# -----------------------------
# Main window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self, viewer: BaseViewer, state: AppState | None = None):
        super().__init__()
        self.setWindowTitle('App GUI')
        self.resize(1200, 800)
        self.state = state or AppState()
        central = QWidget(self)
        self.setCentralWidget(central)
        hbox = QHBoxLayout(central)
        hbox.setContentsMargins(8, 8, 8, 8)
        hbox.setSpacing(12)

        # Left: viewer + info label
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        self.viewer = viewer
        left_v.addWidget(self.viewer, 1)
        self.info_label = QLabel('Waiting for data…')
        self.info_label.setStyleSheet('color: #333;')
        left_v.addWidget(self.info_label)
        hbox.addWidget(left, 2)

        # Right: controls
        self.controls = ControlsPanel(self.state)
        hbox.addWidget(self.controls, 1)

        # Hook signals
        self.viewer.infoUpdated.connect(self._on_viewer_info)
        self.controls.valueChanged.connect(self._on_value_changed)
        self.controls.itemToggled.connect(self._on_item_toggled)

        # Optional: add a Toggle Log button quickly
        btn = QPushButton('Toggle Log Scale')
        btn.clicked.connect(self.viewer.toggle_log)
        self.controls.layout().addWidget(btn)

    def _on_viewer_info(self, text: str):
        items_text = ', '.join(f"{k}:{'ON' if v else 'OFF'}" for k, v in self.state.items.items())
        self.info_label.setText(f"{text} | A={self.state.value_a} B={self.state.value_b} | {items_text}")

    def _on_value_changed(self, name: str, value: float):
        # TODO: Hook backend
        pass

    def _on_item_toggled(self, item: str, on: bool):
        # TODO: Hook backend
        pass


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description='GUI with mock or SHM viewer')
    p.add_argument('shm_name', nargs='?', help='Shared memory base name (omit in --mock)')
    p.add_argument('--mock', action='store_true', help='Run without SHM; use synthetic frames')
    p.add_argument('--fps', type=int, default=30, help='Frames per second')
    p.add_argument('--vmin', type=float, default=None, help='Static vmin (SHM only)')
    p.add_argument('--vmax', type=float, default=None, help='Static vmax (SHM only)')
    p.add_argument('--shape', type=str, default='480x640', help='HxW for mock frames, e.g., 480x640')
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv or sys.argv[1:])
    app = QApplication(sys.argv)

    if args.mock:
        try:
            h, w = map(int, args.shape.lower().split('x'))
        except Exception:
            h, w = 480, 640
        viewer = MockViewer(fps=args.fps, shape=(h, w))
    else:
        if not args.shm_name:
            print('Error: specify <shm_name> or use --mock', file=sys.stderr)
            return 2
        viewer = SHMViewer(shm_name=args.shm_name, fps=args.fps, static_vmin=args.vmin, static_vmax=args.vmax)

    win = MainWindow(viewer)
    win.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
