"""Main window: owns the view state and ties the canvas, rendering and controls together."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QAction, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QProgressDialog, QPushButton, QRadioButton, QSizePolicy, QSlider, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from .canvas import Canvas
from .colormap import ColorMapper
from .engine import FrameCache, RenderWorker, Renderer
from .indicators import INDICATORS
from .state import MAX_C_RES, ViewState, W, H, C, T

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_DIR, "videoOutput")
DIGIT_DIMS = {Qt.Key_1: W, Qt.Key_2: H, Qt.Key_3: C, Qt.Key_4: T}


@dataclass
class Settings:
    dynamic_downscale: bool = True
    preview_factor: int = 3          # preview renders at 1/3 resolution...
    res_mode: str = "window"         # "window" follows the canvas size; "custom" uses custom_w/h
    quality: int = 100               # percent of the canvas size (logical pixels), in window mode
    custom_w: int = 960
    custom_h: int = 720
    display_mode: str = "fit"        # "fit" scales to the window; "actual" shows 1 sample per screen pixel
    cache_mb: int = 1024
    fps: int = 30


class MainWindow(QMainWindow):
    def __init__(self, mapping: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("WHCT")
        self.resize(1400, 900)
        self.settings = Settings()
        self.state = ViewState()
        self.renderer = Renderer()
        self.cache = FrameCache(self.settings.cache_mb)
        self.worker = RenderWorker(self.renderer)
        self.worker.finished.connect(self._on_render_finished)
        self.mapper = ColorMapper()
        if mapping:
            self.mapper.set_mapping(mapping)

        self.values_on = False
        self.tumble_lock = False
        self.held_dims = set()
        self.playing = False
        self._job_id = 0
        self._shown_job = -1
        self._playback_wait: Optional[int] = None
        self._first_resize_done = False
        self._undo: List[tuple] = []
        self._redo: List[tuple] = []
        self._gesture_depth = 0
        self._last_ms = 0.0
        self._current_index: Optional[np.ndarray] = None

        self.canvas = Canvas(self)
        self._build_ui()

        self.idle_timer = QTimer(self, singleShot=True, interval=220)
        self.idle_timer.timeout.connect(lambda: self.schedule_render(interactive=False))
        self.wheel_timer = QTimer(self, singleShot=True, interval=400)
        self.wheel_timer.timeout.connect(self.end_gesture)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance)
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas, 1)

        bar = QWidget(central)
        bar.setObjectName("bar")
        b = QHBoxLayout(bar)
        b.setContentsMargins(10, 6, 10, 6)
        b.setSpacing(8)

        self.play_btn = QPushButton("Play", bar)
        self.play_btn.setToolTip("Play / pause (Space)")
        self.play_btn.clicked.connect(self.toggle_play)
        self.loop_cb = QCheckBox("Loop", bar)
        self.loop_cb.setChecked(True)
        self.bounce_cb = QCheckBox("Bounce", bar)
        self.bounce_cb.setToolTip("Play forward then backward: 29, 30, 31, 30, 29, ..., 2, 1, 2, 3, ...")
        self.frame_slider = QSlider(Qt.Horizontal, bar)
        self.frame_slider.setMinimumWidth(220)
        self.frame_slider.valueChanged.connect(self._on_slider)
        self.frame_label = QLabel(bar)
        self.frame_label.setFixedWidth(170)

        self.shape_combo = QComboBox(bar)
        self.shape_combo.addItems(list(INDICATORS))
        self.shape_combo.currentTextChanged.connect(self._on_shape)
        self.map_combo = QComboBox(bar)
        self.map_combo.addItems(self.mapper.names)
        self.map_combo.setCurrentText(self.mapper.name)
        self.map_combo.currentTextChanged.connect(self._on_mapping)
        self.prec_spin = QSpinBox(bar)
        self.prec_spin.setRange(1, 100000)
        self.prec_spin.setValue(self.state.precision)
        self.prec_spin.setKeyboardTracking(False)
        self.prec_spin.valueChanged.connect(self._on_precision)

        self.save_btn = QToolButton(bar)
        self.save_btn.setText("Save")
        self.save_btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.save_btn)
        menu.addAction("Save MP4…", self.save_mp4)
        menu.addAction("Save GIF…", self.save_gif)
        menu.addAction("Save current time slice…", self.save_slice)
        self.save_btn.setMenu(menu)
        self.cfg_btn = QToolButton(bar)
        self.cfg_btn.setText("Configuration")
        self.cfg_btn.setPopupMode(QToolButton.InstantPopup)
        cmenu = QMenu(self.cfg_btn)
        cmenu.addAction("Save configuration…", self.save_config)
        cmenu.addAction("Load configuration…", self.load_config)
        self.cfg_btn.setMenu(cmenu)
        opt_btn = QPushButton("Options…", bar)
        opt_btn.clicked.connect(self.open_options)
        self.status = QLabel(bar)
        self.status.setObjectName("status")
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # long text must not widen the window

        for wdg in (self.play_btn, self.loop_cb, self.bounce_cb):
            b.addWidget(wdg)
        b.addWidget(self.frame_slider, 2)
        b.addWidget(self.frame_label)
        b.addSpacing(10)
        b.addWidget(QLabel("Shape", bar))
        b.addWidget(self.shape_combo)
        b.addWidget(QLabel("Color", bar))
        b.addWidget(self.map_combo)
        b.addWidget(QLabel("Precision", bar))
        b.addWidget(self.prec_spin)
        b.addWidget(self.save_btn)
        b.addWidget(self.cfg_btn)
        b.addWidget(opt_btn)
        b.addWidget(self.status, 1)
        for wdg in bar.findChildren(QWidget):
            if isinstance(wdg, (QPushButton, QCheckBox, QToolButton, QSlider, QComboBox)):
                wdg.setFocusPolicy(Qt.NoFocus)  # keep Space and the 1-4 keys for navigation
        lay.addWidget(bar)
        self.setCentralWidget(central)
        self._sync_slider()

    # ------------------------------------------------------------------ geometry
    def render_size(self):
        """Full-quality W x H sample counts."""
        return self.state.res[W], self.state.res[H]

    def on_canvas_resized(self):
        if self.settings.res_mode != "window":
            return
        w, h = self.canvas.width(), self.canvas.height()
        nw = max(16, int(w * self.settings.quality / 100))
        nh = max(16, int(h * self.settings.quality / 100))
        old_w, old_h = self.state.res[W], self.state.res[H]
        self.state.res[W], self.state.res[H] = nw, nh
        if not self._first_resize_done:
            self._first_resize_done = True
            self.state.reset_view()
        else:
            # Keep the math size of a sample, so resizing reveals more instead of stretching.
            self.state.spans[W] *= nw / max(1, old_w)
            self.state.spans[H] *= nh / max(1, old_h)
        self.schedule_render(interactive=True)

    # ------------------------------------------------------------------ navigation (called by the canvas)
    def begin_gesture(self):
        if self._gesture_depth == 0:
            self._gesture_snap = self.state.nav_snapshot()
            if self.playing:
                self.toggle_play()
        self._gesture_depth += 1

    def end_gesture(self):
        self._gesture_depth = max(0, self._gesture_depth - 1)
        if self._gesture_depth == 0:
            # Only gestures that actually moved the view become an undo step.
            snap = getattr(self, "_gesture_snap", None)
            now = self.state.nav_snapshot()
            if snap is not None and not all(np.allclose(a, b) for a, b in zip(snap, now)):
                self._undo.append(snap)
                del self._undo[:-200]
                self._redo.clear()
            self._gesture_snap = None
            self.schedule_render(interactive=False)

    def _changed(self):
        self.canvas.refresh_value_boxes()
        self.canvas.update()
        self._update_frame_label()
        self.schedule_render(interactive=True)

    def pan(self, du):
        self.state.pan(du)
        self._changed()

    def zoom(self, dims, factor, anchor_u):
        if not self.wheel_timer.isActive():
            self.begin_gesture()
        self.wheel_timer.start()
        f = np.ones(4)
        for k in dims:
            f[k] = factor
        self.state.scale(f, anchor_u)
        self._changed()

    def rotate_plane(self, i, k, theta):
        self.state.rotate_plane(i, k, theta)
        self._changed()

    def set_span(self, k, value):
        self.state.set_span(k, value)
        self._changed()

    def set_component(self, j, c, value):
        self.state.set_component(j, c, value)
        self._changed()

    def edit_component(self, j, c, value):
        self.begin_gesture()
        self.set_component(j, c, value)
        self.end_gesture()

    def edit_span(self, k, value):
        if value > 0:
            self.begin_gesture()
            self.set_span(k, value)
            self.end_gesture()

    def reset_view(self):
        self.begin_gesture()
        self.state.reset_view()
        self.canvas.rot.reset_camera()
        self._changed()
        self.end_gesture()

    def reset_rotation(self):
        self.begin_gesture()
        self.state.reset_rotation()
        self._changed()
        self.end_gesture()

    def reset_spans(self):
        self.begin_gesture()
        self.state.reset_spans()
        self._changed()
        self.end_gesture()

    def undo(self):
        if self._undo:
            self._redo.append(self.state.nav_snapshot())
            self.state.restore_nav(self._undo.pop())
            self._changed()
            self.schedule_render(interactive=False)

    def redo(self):
        if self._redo:
            self._undo.append(self.state.nav_snapshot())
            self.state.restore_nav(self._redo.pop())
            self._changed()
            self.schedule_render(interactive=False)

    # ------------------------------------------------------------------ rendering
    def _job(self, kind: str, frame: int) -> dict:
        st = self.state
        res = list(st.res)
        prec = st.precision
        if kind == "preview":
            pts = res[W] * res[H] * res[C]
            f = max(self.settings.preview_factor, int(np.ceil(np.sqrt(pts / self.renderer.preview_points))))
            res[W] = max(8, res[W] // f)
            res[H] = max(8, res[H] // f)
            prec = max(8, prec // 2)
        self._job_id += 1
        return {
            "id": self._job_id, "kind": kind, "frame": frame, "state": st.copy(),
            "res": res, "precision": prec, "sig": st.signature(),
        }

    def schedule_render(self, interactive: bool = False):
        st = self.state
        sig = st.signature()
        cached = self.cache.get(sig, st.frame)
        if cached is not None:
            self.idle_timer.stop()
            self._show(cached)
            return
        if interactive and self.settings.dynamic_downscale:
            self.worker.submit(self._job("preview", st.frame))
            self.idle_timer.start()
        else:
            self.idle_timer.stop()
            self.worker.submit(self._job("full", st.frame))
        self._update_status()

    def _on_render_finished(self, res: dict):
        if res.get("error") is not None:
            self.status.setText(f"Render error: {res['error']}")
            return
        self._last_ms = res["ms"]
        img = res["image"]
        if res["kind"] == "full" and res["sig"] == self.state.signature():
            self.cache.put(res["sig"], res["frame"], img)
        if self._playback_wait is not None and res["id"] == self._playback_wait:
            self._playback_wait = None
            if res["sig"] == self.state.signature():
                self.state.frame = res["frame"]
                self._sync_slider()
                self._show(img)
            return
        if res["id"] >= self._shown_job and res["frame"] == self.state.frame:
            self._shown_job = res["id"]
            self._show(img)
        self._update_status()

    def _show(self, index: np.ndarray):
        self._current_index = index
        self.canvas.set_image(self.mapper.to_rgb(index), index)
        self._update_status()

    # ------------------------------------------------------------------ playback and frames
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self.play_timer.start(int(1000 / max(1, self.settings.fps)))
        else:
            self.play_timer.stop()
            self._playback_wait = None

    def _next_frame(self) -> Optional[int]:
        """Next frame to play, or None to stop. Bounce reverses at each end without repeating it."""
        st = self.state
        n = st.res[T]
        if n <= 1:
            return None
        if not self.bounce_cb.isChecked():
            self._dir = 1
            nxt = st.frame + 1
            if nxt >= n:
                return 0 if self.loop_cb.isChecked() else None
            return nxt
        d = getattr(self, "_dir", 1)
        if st.frame + d >= n or st.frame + d < 0:
            d = -d
            if d == 1 and not self.loop_cb.isChecked():
                self._dir = 1
                return None  # one forward-and-back pass finished
        self._dir = d
        return st.frame + d

    def _advance(self):
        if self._playback_wait is not None:
            return  # still rendering the next frame
        st = self.state
        nxt = self._next_frame()
        if nxt is None:
            self.toggle_play()
            return
        cached = self.cache.get(st.signature(), nxt)
        if cached is not None:
            st.frame = nxt
            self._sync_slider()
            self._show(cached)
        else:
            job = self._job("full", nxt)
            self._playback_wait = job["id"]
            self.worker.submit(job)

    def step_frame(self, delta: int):
        st = self.state
        st.frame = int(np.clip(st.frame + delta, 0, st.res[T] - 1))
        self._sync_slider()
        self._frame_changed()

    def _on_slider(self, v: int):
        if v != self.state.frame:
            self.state.frame = v
            self._frame_changed(interactive=self.frame_slider.isSliderDown())

    def _frame_changed(self, interactive: bool = False):
        # The pivot moves with the current frame; values boxes and the probe follow.
        self._update_frame_label()
        self.canvas.refresh_value_boxes()
        self.schedule_render(interactive=interactive)

    def _sync_slider(self):
        st = self.state
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, st.res[T] - 1))
        self.frame_slider.setValue(st.frame)
        self.frame_slider.blockSignals(False)
        self._update_frame_label()

    def _update_frame_label(self):
        st = self.state
        self.frame_label.setText(f"frame {st.frame + 1}/{st.res[T]}   T {st.t_value():+.3f}")

    def _update_status(self):
        st = self.state
        n = len(self.cache.frames) if self.cache.sig == st.signature() else 0
        mb = self.cache.nbytes / 2 ** 20
        dev = self.renderer.device_name if self.renderer.device.type == "cuda" else "CPU"
        self.status.setText(
            f"{dev} · {self._last_ms:.0f} ms · {st.res[W]}×{st.res[H]}×{st.res[C]}×{st.res[T]}"
            f" · cache {n}/{st.res[T]} frames, {mb:.0f}/{self.settings.cache_mb} MB"
        )

    # ------------------------------------------------------------------ bar controls
    def _on_shape(self, name: str):
        self.state.indicator = name
        self.reset_view()

    def _on_mapping(self, name: str):
        self.mapper.set_mapping(name)
        if self._current_index is not None:
            self._show(self._current_index)

    def _on_precision(self, v: int):
        self.state.precision = int(v)
        self.schedule_render()

    # ------------------------------------------------------------------ keys
    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if ev.type() not in (QEvent.KeyPress, QEvent.KeyRelease) or not self.isActiveWindow():
            return False
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QAbstractSpinBox)):
            return False  # typing in a field
        key = ev.key()
        if ev.type() == QEvent.KeyRelease:
            if key in DIGIT_DIMS and not ev.isAutoRepeat():
                self.held_dims.discard(DIGIT_DIMS[key])
                return True
            return False
        mods = ev.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        if key in DIGIT_DIMS:
            self.held_dims.add(DIGIT_DIMS[key])
            return True
        if ev.isAutoRepeat() and key not in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Z, Qt.Key_Y):
            return key in (Qt.Key_Space, Qt.Key_P, Qt.Key_V, Qt.Key_Home)
        if key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_P and not ctrl:
            self.canvas.probe_on = not self.canvas.probe_on
            self.canvas.update()
        elif key == Qt.Key_V and not ctrl:
            self.values_on = not self.values_on
            self.canvas.refresh_value_boxes()
            self.canvas.update()
        elif key == Qt.Key_Home:
            self.reset_view()
        elif key == Qt.Key_Left:
            self.step_frame(-1)
        elif key == Qt.Key_Right:
            self.step_frame(1)
        elif key == Qt.Key_Escape:
            self.canvas.collapse_widgets()
        elif ctrl and key == Qt.Key_Z:
            self.redo() if mods & Qt.ShiftModifier else self.undo()
        elif ctrl and key == Qt.Key_Y:
            self.redo()
        else:
            return False
        return True

    def focusOutEvent(self, e):
        self.held_dims.clear()
        super().focusOutEvent(e)

    def changeEvent(self, e):
        if e.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self.held_dims.clear()
        super().changeEvent(e)

    # ------------------------------------------------------------------ options
    def open_options(self):
        dlg = OptionsDialog(self)
        dlg.exec()

    def apply_settings(self, new: Settings, c_res: int, t_res: int):
        old = self.settings
        self.settings = new
        self.cache.set_budget(new.cache_mb)
        st = self.state
        st.res[C] = int(np.clip(c_res, 1, MAX_C_RES))
        st.set_t_res(t_res)
        if new.res_mode == "custom":
            st.res[W], st.res[H] = new.custom_w, new.custom_h
        else:
            self.on_canvas_resized()
        if self.playing:
            self.play_timer.start(int(1000 / max(1, new.fps)))
        self._sync_slider()
        self.canvas.update()
        self.schedule_render()

    # ------------------------------------------------------------------ configuration files
    def save_config(self):
        import json
        from dataclasses import asdict
        default = os.path.join(REPO_DIR, "config.whct")
        path, _ = QFileDialog.getSaveFileName(self, "Save configuration", default, "WHCT configuration (*.whct)")
        if not path:
            return
        data = {
            "whct_config": 1,
            "view": self.state.to_dict(),
            "settings": asdict(self.settings),
            "color_mapping": self.mapper.name,
            "loop": self.loop_cb.isChecked(),
            "bounce": self.bounce_cb.isChecked(),
            "rotation_widget_camera": [self.canvas.rot.yaw, self.canvas.rot.pitch],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        self.status.setText(f"Saved configuration {path}")

    def load_config(self):
        import json
        path, _ = QFileDialog.getOpenFileName(self, "Load configuration", REPO_DIR, "WHCT configuration (*.whct)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            view = data["view"]
            if view["indicator"] not in INDICATORS:
                raise ValueError(f"unknown shape {view['indicator']}")
            fields = {k: v for k, v in data.get("settings", {}).items() if k in Settings.__dataclass_fields__}
            new_settings = Settings(**fields)
            self.state.load_dict(view)
        except Exception as exc:
            QMessageBox.critical(self, "Load configuration", f"Could not load this file:\n{exc}")
            return
        if self.playing:
            self.toggle_play()
        self.settings = new_settings
        self.cache.set_budget(new_settings.cache_mb)
        self.cache.clear()
        mapping = data.get("color_mapping")
        if mapping in self.mapper.names:
            self.mapper.set_mapping(mapping)
        self.loop_cb.setChecked(bool(data.get("loop", True)))
        self.bounce_cb.setChecked(bool(data.get("bounce", False)))
        cam = data.get("rotation_widget_camera")
        if cam:
            self.canvas.rot.yaw, self.canvas.rot.pitch = cam
            self.canvas.rot._update_camera()
        for combo, text in ((self.shape_combo, self.state.indicator), (self.map_combo, self.mapper.name)):
            combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(False)
        self.prec_spin.blockSignals(True)
        self.prec_spin.setValue(self.state.precision)
        self.prec_spin.blockSignals(False)
        self._undo.clear()
        self._redo.clear()
        self._sync_slider()
        self.canvas.refresh_value_boxes()
        self.canvas.update()
        self.schedule_render()
        self.status.setText(f"Loaded configuration {path}")

    # ------------------------------------------------------------------ saving
    def _render_full(self, frame: int) -> Optional[np.ndarray]:
        sig = self.state.signature()
        img = self.cache.get(sig, frame)
        if img is None:
            img = self.renderer.render(self.state, frame, self.state.res, self.state.precision)
            self.cache.put(sig, frame, img)
        return img

    def save_slice(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        default = os.path.join(OUTPUT_DIR, f"{self.state.indicator}_frame{self.state.frame + 1}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save current time slice", default, "PNG image (*.png)")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            img = self._render_full(self.state.frame)
            cv2.imwrite(path, cv2.cvtColor(self.mapper.to_rgb(img), cv2.COLOR_RGB2BGR))
        finally:
            QApplication.restoreOverrideCursor()
        self.status.setText(f"Saved {path}")

    def save_mp4(self):
        if self.playing:
            self.toggle_play()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        default = os.path.join(OUTPUT_DIR, f"{self.state.indicator}.mp4")
        path, _ = QFileDialog.getSaveFileName(self, "Save MP4", default, "MP4 video (*.mp4)")
        if not path:
            return
        st = self.state
        n = st.res[T]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.settings.fps, (st.res[W], st.res[H]))
        if not writer.isOpened():
            QMessageBox.critical(self, "Save MP4", f"Could not open a video writer for:\n{path}")
            return
        prog = QProgressDialog("Rendering frames…", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        try:
            for f in range(n):
                if prog.wasCanceled():
                    break
                prog.setValue(f)
                QApplication.processEvents()
                img = self._render_full(f)
                writer.write(cv2.cvtColor(self.mapper.to_rgb(img), cv2.COLOR_RGB2BGR))
            prog.setValue(n)
        finally:
            writer.release()
        self._update_status()
        self.status.setText(f"Saved {path}")

    def save_gif(self):
        from PIL import Image
        if self.playing:
            self.toggle_play()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        default = os.path.join(OUTPUT_DIR, f"{self.state.indicator}.gif")
        path, _ = QFileDialog.getSaveFileName(self, "Save GIF", default, "GIF image (*.gif)")
        if not path:
            return
        n = self.state.res[T]
        # With Bounce on, the GIF goes forward and back (without repeating the end frames).
        order = list(range(n))
        if self.bounce_cb.isChecked() and n > 2:
            order += list(range(n - 2, 0, -1))
        prog = QProgressDialog("Rendering frames…", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        rendered = {}
        try:
            for f in range(n):
                if prog.wasCanceled():
                    return
                prog.setValue(f)
                QApplication.processEvents()
                img = Image.fromarray(self.mapper.to_rgb(self._render_full(f)))
                rendered[f] = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            prog.setValue(n)
            frames = [rendered[f] for f in order]
            duration = max(20, int(round(1000 / max(1, self.settings.fps) / 10)) * 10)  # GIF timing is in 10 ms steps
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False)
        except Exception as exc:
            QMessageBox.critical(self, "Save GIF", f"Could not save the GIF:\n{exc}")
            return
        self._update_status()
        self.status.setText(f"Saved {path} ({os.path.getsize(path) / 2 ** 20:.1f} MB)")

    def closeEvent(self, e):
        self.worker.stop()
        super().closeEvent(e)


class OptionsDialog(QDialog):
    def __init__(self, win: MainWindow):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Options")
        s = win.settings
        st = win.state
        form = QVBoxLayout(self)

        perf = QGroupBox("Speed", self)
        pf = QFormLayout(perf)
        self.dyn = QCheckBox("Dynamic downscale (lower resolution and precision while you move)", perf)
        self.dyn.setChecked(s.dynamic_downscale)
        pf.addRow(self.dyn)
        self.cache = QSpinBox(perf)
        self.cache.setRange(64, 262144)
        self.cache.setSingleStep(256)
        self.cache.setSuffix(" MB")
        self.cache.setValue(s.cache_mb)
        pf.addRow("Frame cache limit", self.cache)
        form.addWidget(perf)

        res = QGroupBox("Resolution", self)
        rf = QFormLayout(res)
        self.mode_window = QRadioButton("Match the window", res)
        self.mode_custom = QRadioButton("Custom", res)
        grp = QButtonGroup(self)
        grp.addButton(self.mode_window)
        grp.addButton(self.mode_custom)
        (self.mode_window if s.res_mode == "window" else self.mode_custom).setChecked(True)
        self.quality = QSpinBox(res)
        self.quality.setRange(10, 400)
        self.quality.setSuffix(" %")
        self.quality.setValue(s.quality)
        row1 = QHBoxLayout()
        row1.addWidget(self.mode_window)
        row1.addWidget(QLabel("quality", res))
        row1.addWidget(self.quality)
        rf.addRow(row1)
        self.cw = QSpinBox(res)
        self.cw.setRange(16, 16384)
        self.cw.setValue(s.custom_w if s.res_mode == "custom" else st.res[W])
        self.ch = QSpinBox(res)
        self.ch.setRange(16, 16384)
        self.ch.setValue(s.custom_h if s.res_mode == "custom" else st.res[H])
        row2 = QHBoxLayout()
        row2.addWidget(self.mode_custom)
        row2.addWidget(QLabel("W", res))
        row2.addWidget(self.cw)
        # Link button (like Inkscape's): while linked, changing W or H scales the other to keep the ratio.
        self.link = QToolButton(res)
        self.link.setCheckable(True)
        self.link.setChecked(True)
        self.link.setMinimumWidth(62)
        self.link.toggled.connect(self._on_link_toggled)
        row2.addWidget(self.link)
        row2.addWidget(QLabel("H", res))
        row2.addWidget(self.ch)
        rf.addRow(row2)
        self._ratio = self.cw.value() / max(1, self.ch.value())
        self._on_link_toggled(True)
        self.cw.valueChanged.connect(lambda v: self._follow(self.cw, self.ch, v, True))
        self.ch.valueChanged.connect(lambda v: self._follow(self.ch, self.cw, v, False))
        self.cres = QSpinBox(res)
        self.cres.setRange(1, MAX_C_RES)
        self.cres.setValue(st.res[C])
        rf.addRow("C slices", self.cres)
        self.tres = QSpinBox(res)
        self.tres.setRange(1, 100000)
        self.tres.setValue(st.res[T])
        rf.addRow("T frames", self.tres)
        form.addWidget(res)

        disp = QGroupBox("Display", self)
        df = QFormLayout(disp)
        self.fit = QRadioButton("Fit to window", disp)
        self.actual = QRadioButton("Actual pixels (one sample per screen pixel)", disp)
        dgrp = QButtonGroup(self)
        dgrp.addButton(self.fit)
        dgrp.addButton(self.actual)
        (self.fit if s.display_mode == "fit" else self.actual).setChecked(True)
        df.addRow(self.fit)
        df.addRow(self.actual)
        self.fps = QSpinBox(disp)
        self.fps.setRange(1, 240)
        self.fps.setValue(s.fps)
        df.addRow("Playback fps", self.fps)
        form.addWidget(disp)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply)
        form.addWidget(buttons)

    def _on_link_toggled(self, linked: bool):
        self.link.setText("Linked" if linked else "Unlinked")
        self.link.setToolTip("Linked: W and H change together, keeping their ratio" if linked
                             else "Unlinked: W and H change independently")
        if linked:
            self._ratio = self.cw.value() / max(1, self.ch.value())  # lock the current ratio

    def _follow(self, src: QSpinBox, dst: QSpinBox, value: int, src_is_w: bool):
        if not self.link.isChecked():
            return
        other = value / self._ratio if src_is_w else value * self._ratio
        dst.blockSignals(True)
        dst.setValue(int(round(other)))
        dst.blockSignals(False)

    def _apply(self):
        s = Settings(
            dynamic_downscale=self.dyn.isChecked(),
            preview_factor=self.win.settings.preview_factor,
            res_mode="window" if self.mode_window.isChecked() else "custom",
            quality=self.quality.value(),
            custom_w=self.cw.value(),
            custom_h=self.ch.value(),
            display_mode="fit" if self.fit.isChecked() else "actual",
            cache_mb=self.cache.value(),
            fps=self.fps.value(),
        )
        self.win.apply_settings(s, self.cres.value(), self.tres.value())

    def _ok(self):
        self._apply()
        self.accept()
