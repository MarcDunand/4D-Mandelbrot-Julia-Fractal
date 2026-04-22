import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from PySide6.QtCore import Qt, QTimer, QSize, Signal, QPoint
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)


# -----------------------------------------------------------------------------
# Indicator functions
# -----------------------------------------------------------------------------

def get_xyzt_coords(grids):
    x = torch.real(grids["X"])
    y = torch.real(grids["Y"])
    z = torch.real(grids["H"])
    w = torch.real(grids["T"])
    return x, y, z, w



def julia_mandelbrot_indicator(grids, parameterization, prec, device):
    z = grids[parameterization[0]] + 1j * grids[parameterization[1]]
    c = grids[parameterization[2]] + 1j * grids[parameterization[3]]

    mask = torch.ones_like(z, dtype=torch.bool, device=device)

    for _ in range(prec):
        z_next = z**2 + c
        divergence = z_next.abs() > 2
        mask = mask & ~divergence
        z = torch.where(mask, z_next, z)

        if not mask.any():
            break

    return mask



def tesseract_indicator(grids, parameterization, prec, device):
    x, y, z, w = get_xyzt_coords(grids)
    half_side = 0.5
    return (
        (torch.abs(x) <= half_side)
        & (torch.abs(y) <= half_side)
        & (torch.abs(z) <= half_side)
        & (torch.abs(w) <= half_side)
    )



def hypersphere_indicator(grids, parameterization, prec, device):
    x, y, z, w = get_xyzt_coords(grids)
    radius = 0.5
    radius_sq = radius * radius
    return (x * x + y * y + z * z + w * w) <= radius_sq



def cross_polytope_indicator(grids, parameterization, prec, device):
    x, y, z, w = get_xyzt_coords(grids)
    r = 0.8
    return (torch.abs(x) + torch.abs(y) + torch.abs(z) + torch.abs(w)) <= r



def torus_4d_indicator(grids, parameterization, prec, device):
    x, y, z, w = get_xyzt_coords(grids)
    R = 0.5
    r = 0.15
    r3 = torch.sqrt(x * x + y * y + z * z)
    return ((r3 - R) ** 2 + w * w) <= (r * r)



def shell_indicator(grids, parameterization, prec, device):
    x, y, z, w = get_xyzt_coords(grids)
    r1 = 0.35
    r2 = 0.5
    dist_sq = x * x + y * y + z * z + w * w
    return (dist_sq >= r1 * r1) & (dist_sq <= r2 * r2)



def sphere_minus_cube_indicator(grids, parameterization, prec, device):
    x, y, z, w = get_xyzt_coords(grids)
    sphere_r = 0.5
    cube_half = 0.3

    sphere = (x * x + y * y + z * z + w * w) <= (sphere_r * sphere_r)
    cube = (
        (torch.abs(x) <= cube_half)
        & (torch.abs(y) <= cube_half)
        & (torch.abs(z) <= cube_half)
        & (torch.abs(w) <= cube_half)
    )
    return sphere & ~cube



def sine_field_4d_indicator(grids, parameterization, prec, device):
    x, y, z, w = get_xyzt_coords(grids)
    threshold = 0.0
    return (torch.sin(x) + torch.sin(y) + torch.sin(z) + torch.sin(w)) > threshold



def gyroid_4d_shell_indicator(grids, parameterization, prec, device):
    x, y, z, w = get_xyzt_coords(grids)
    eps = 0.25
    field = (
        torch.sin(x) * torch.cos(y)
        + torch.sin(y) * torch.cos(z)
        + torch.sin(z) * torch.cos(w)
        + torch.sin(w) * torch.cos(x)
    )
    return torch.abs(field) < eps


INDICATORS = {
    "julia_mandelbrot": julia_mandelbrot_indicator,
    "tesseract": tesseract_indicator,
    "hypersphere": hypersphere_indicator,
    "4D_diamond": cross_polytope_indicator,
    "4D_torus": torus_4d_indicator,
    "hollow_hypersphere": shell_indicator,
    "hypersphere_minus_tesseract": sphere_minus_cube_indicator,
    "4D_sine_field": sine_field_4d_indicator,
    "4D_gyroid_shell": gyroid_4d_shell_indicator,
}


# -----------------------------------------------------------------------------
# Data model and engine
# -----------------------------------------------------------------------------

DEFAULT_PARAMETERIZATION = "HTYX"


@dataclass
class ViewerState:
    indicator: str = "julia_mandelbrot"

    Ymin: float = -1.0
    Ymax: float = 1.0
    Yres: int = 500

    Xmin: float = -1.0
    Xmax: float = 1.0
    Xres: int = 500

    Tmin: float = -1.0
    Tmax: float = 1.0
    Tres: int = 30

    Cmin: float = -1.0
    Cmax: float = 1.0
    Cres: int = 24

    precision: int = 30
    snapping_enabled: bool = False

    XYrot: float = 0.0
    XTrot: float = 0.0
    XHrot: float = 0.0
    YTrot: float = 0.0
    YHrot: float = 0.0
    HTrot: float = 0.0

    current_frame: int = 0


class WHCTEngine:
    def __init__(self, mapping_path: Optional[str]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = ViewerState()
        self.mapping_path = mapping_path
        self._lut_cache: Dict[Tuple[int, str], torch.Tensor] = {}
        self.lut: Optional[torch.Tensor] = None
        if mapping_path is not None:
            self.lut = self.get_lut(24, mapping_path=mapping_path)

        self.last_index_img: Optional[np.ndarray] = None
        self.last_base_bgr: Optional[np.ndarray] = None
        self.last_lut_bgr: Optional[np.ndarray] = None

    def get_lut(self, bits: int, mapping_path: str) -> torch.Tensor:
        key = (bits, mapping_path)
        if key in self._lut_cache:
            return self._lut_cache[key]

        base, _ = os.path.splitext(mapping_path)
        lut_path = base + ".pt"

        if os.path.exists(lut_path):
            lut = torch.load(lut_path, map_location=self.device)
        else:
            data = np.load(mapping_path, allow_pickle=True)
            boid_to_rgb = data["boid_to_rgb"]
            side = int(data["side"])
            n = boid_to_rgb.shape[0]
            capacity = 1 << bits

            if n != capacity:
                raise ValueError(
                    f"Mapping size N={n} does not match 2**bits={capacity} for bits={bits}"
                )

            x = boid_to_rgb[:, 0].astype(np.int64)
            y = boid_to_rgb[:, 1].astype(np.int64)
            z = boid_to_rgb[:, 2].astype(np.int64)

            if side == 256:
                r = x
                g = y
                b = z
            else:
                denom = max(1, side - 1)
                r = (x * 255) // denom
                g = (y * 255) // denom
                b = (z * 255) // denom

            lut_np = ((r << 16) | (g << 8) | b).astype(np.int32)
            lut = torch.from_numpy(lut_np).to(self.device)
            torch.save(lut.cpu(), lut_path)

        self._lut_cache[key] = lut
        return lut

    def evaluate_indicator(self, grids) -> torch.Tensor:
        indicator_fn = INDICATORS[self.state.indicator]
        mask = indicator_fn(
            grids,
            DEFAULT_PARAMETERIZATION,
            self.state.precision,
            self.device,
        )
        if mask.dtype != torch.bool:
            raise TypeError(f"Indicator '{self.state.indicator}' must return a boolean tensor.")
        return mask

    def generate_frame(self, t: float):
        s = self.state

        rotational_planes = {
            "XY": s.XYrot,
            "XT": s.XTrot,
            "XH": s.XHrot,
            "YT": s.YTrot,
            "YH": s.YHrot,
            "HT": s.HTrot,
        }

        ymin, ymax = s.Ymax * -1, s.Ymin * -1
        y_values = torch.linspace(ymin, ymax, s.Yres, device=self.device)
        x_values = torch.linspace(s.Xmin, s.Xmax, s.Xres, device=self.device)
        h_values = torch.linspace(s.Cmin, s.Cmax, s.Cres, device=self.device)

        y_grid, x_grid, h_grid = torch.meshgrid(y_values, x_values, h_values, indexing="ij")
        t_grid = torch.full_like(x_grid, t)
        grids = {"H": h_grid, "X": x_grid, "Y": y_grid, "T": t_grid}

        for plane, rot in rotational_planes.items():
            if abs(rot) > 0.001:
                axis1 = plane[0]
                axis2 = plane[1]
                tensor_rot = torch.tensor(rot, device=self.device)
                cos_t, sin_t = torch.cos(tensor_rot), torch.sin(tensor_rot)

                axis1_original = grids[axis1].clone()
                axis2_original = torch.real(grids[axis2])
                grids[axis1] = axis1_original * cos_t - axis2_original * sin_t
                grids[axis2] = axis1_original * sin_t + axis2_original * cos_t

        mask = self.evaluate_indicator(grids)
        weights = 2 ** torch.arange(s.Cres, device=self.device, dtype=torch.int32)
        index_image = torch.sum(mask.int() * weights, dim=-1)

        if self.lut is not None:
            recolored_image = self.lut[index_image]
            return index_image.cpu().numpy(), recolored_image.cpu().numpy()
        return index_image.cpu().numpy(), None

    @staticmethod
    def packed_int_to_bgr(packed: np.ndarray) -> np.ndarray:
        image = np.zeros((packed.shape[0], packed.shape[1], 3), dtype=np.uint8)
        image[..., 2] = (packed >> 16) & 0xFF
        image[..., 1] = (packed >> 8) & 0xFF
        image[..., 0] = packed & 0xFF
        return image

    def current_t(self) -> float:
        s = self.state
        if s.Tres <= 0:
            return s.Tmin
        return s.Tmin + (s.Tmax - s.Tmin) * (s.current_frame / s.Tres)

    def render_current_base_images(self):
        t = self.current_t()
        index_img, lut_img = self.generate_frame(t)

        base_bgr = self.packed_int_to_bgr(index_img)
        lut_bgr = self.packed_int_to_bgr(lut_img) if lut_img is not None else None

        self.last_index_img = index_img
        self.last_base_bgr = base_bgr
        self.last_lut_bgr = lut_bgr
        return base_bgr, lut_bgr


# -----------------------------------------------------------------------------
# Overlay drawing
# -----------------------------------------------------------------------------

class HoverOverlay:
    def __init__(self):
        self.overlay_enabled = True
        self.hover_inside = False
        self.hover_x = 0
        self.hover_y = 0

    def set_hover(self, pos: Optional[Tuple[int, int]]):
        if pos is None:
            self.hover_inside = False
            return
        self.hover_inside = True
        self.hover_x, self.hover_y = pos

    def toggle(self):
        self.overlay_enabled = not self.overlay_enabled

    def draw(self, frame_bgr: np.ndarray, index_img: Optional[np.ndarray], bits: int) -> np.ndarray:
        if index_img is None or not self.overlay_enabled or not self.hover_inside:
            return frame_bgr

        h_idx, w_idx = index_img.shape[:2]
        x, y = int(self.hover_x), int(self.hover_y)
        if x < 0 or x >= w_idx or y < 0 or y >= h_idx:
            return frame_bgr

        val = int(index_img[y, x])
        occupancy = [(val >> i) & 1 for i in range(bits)]

        img_h, img_w, _ = frame_bgr.shape
        bar_width = 12
        bar_height = 6
        gap = 2
        padding = 4
        margin = 10

        total_height = bits * bar_height + (bits - 1) * gap
        box_width = bar_width + 2 * padding
        box_height = total_height + 2 * padding

        box_x = x + margin
        box_y = y + margin

        if box_x + box_width > img_w:
            box_x = x - margin - box_width
        if box_y + box_height > img_h:
            box_y = y - margin - box_height

        box_x = max(0, box_x)
        box_y = max(0, box_y)

        overlay = frame_bgr.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (0, 0, 0),
            thickness=-1,
        )
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, dst=frame_bgr)

        start_y = box_y + padding
        for i in range(bits):
            bit_index = bits - 1 - i
            filled = occupancy[bit_index] == 1

            y0 = start_y + i * (bar_height + gap)
            y1 = y0 + bar_height - 1
            x0 = box_x + padding
            x1 = x0 + bar_width - 1

            color_fill = (0, 255, 0) if filled else (40, 40, 40)
            color_border = (220, 220, 220)

            if filled:
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color_fill, thickness=-1)
            else:
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color_fill, thickness=1)

            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color_border, thickness=1)

        cv2.circle(frame_bgr, (x, y), 3, (255, 255, 255), thickness=1)
        return frame_bgr


# -----------------------------------------------------------------------------
# Viewport widget
# -----------------------------------------------------------------------------

class RenderViewport(QLabel):
    imagePositionChanged = Signal(object)  # None or (x, y)
    dataPanRequested = Signal(float, float)  # dx, dy in display-pixel units
    dataRotateRequested = Signal(float, float)  # dx, dy in display-pixel units

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.setText(title)

        self._source_qimage: Optional[QImage] = None
        self._display_rect = None
        self._cached_pixmap: Optional[QPixmap] = None
        self._is_right_dragging = False
        self._is_left_dragging = False
        self._last_drag_pos: Optional[QPoint] = None

    def set_numpy_bgr_image(self, image_bgr: Optional[np.ndarray]):
        if image_bgr is None:
            self._source_qimage = None
            self._cached_pixmap = None
            self._display_rect = None
            self.setText("No image")
            self.setPixmap(QPixmap())
            return

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._source_qimage = qimg
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._source_qimage is None:
            return

        pixmap = QPixmap.fromImage(self._source_qimage)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._cached_pixmap = scaled
        self.setPixmap(scaled)

        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._display_rect = (x, y, scaled.width(), scaled.height())

    def _map_widget_to_image(self, pos):
        if self._source_qimage is None or self._display_rect is None:
            return None

        x0, y0, dw, dh = self._display_rect
        px = pos.x()
        py = pos.y()

        if px < x0 or py < y0 or px >= x0 + dw or py >= y0 + dh:
            return None

        iw = self._source_qimage.width()
        ih = self._source_qimage.height()

        ix = int((px - x0) * iw / max(1, dw))
        iy = int((py - y0) * ih / max(1, dh))

        ix = max(0, min(iw - 1, ix))
        iy = max(0, min(ih - 1, iy))
        return ix, iy

    def _is_inside_display_rect(self, pos: QPoint) -> bool:
        if self._display_rect is None:
            return False
        x0, y0, dw, dh = self._display_rect
        return x0 <= pos.x() < x0 + dw and y0 <= pos.y() < y0 + dh

    def mousePressEvent(self, event):
        point = event.position().toPoint()

        if event.button() == Qt.LeftButton and self._is_inside_display_rect(point):
            self._is_left_dragging = True
            self._last_drag_pos = point
            self.imagePositionChanged.emit(None)
            self.setCursor(Qt.SizeAllCursor)
            event.accept()
            return

        if event.button() == Qt.RightButton and self._is_inside_display_rect(point):
            self._is_right_dragging = True
            self._last_drag_pos = point
            self.imagePositionChanged.emit(None)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = event.position().toPoint()
        if self._is_left_dragging and self._last_drag_pos is not None:
            delta = point - self._last_drag_pos
            self._last_drag_pos = point
            if delta.x() != 0 or delta.y() != 0:
                self.dataRotateRequested.emit(float(delta.x()), float(delta.y()))
            event.accept()
            return

        if self._is_right_dragging and self._last_drag_pos is not None:
            delta = point - self._last_drag_pos
            self._last_drag_pos = point
            if delta.x() != 0 or delta.y() != 0:
                self.dataPanRequested.emit(float(delta.x()), float(delta.y()))
            event.accept()
            return

        self.imagePositionChanged.emit(self._map_widget_to_image(point))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_left_dragging:
            self._is_left_dragging = False
            self._last_drag_pos = None
            self.unsetCursor()
            self.imagePositionChanged.emit(self._map_widget_to_image(event.position().toPoint()))
            event.accept()
            return

        if event.button() == Qt.RightButton and self._is_right_dragging:
            self._is_right_dragging = False
            self._last_drag_pos = None
            self.unsetCursor()
            self.imagePositionChanged.emit(self._map_widget_to_image(event.position().toPoint()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if not self._is_right_dragging and not self._is_left_dragging:
            self.imagePositionChanged.emit(None)
        super().leaveEvent(event)


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------

class WHCTMainWindow(QMainWindow):
    def __init__(self, mapping_path: Optional[str]):
        super().__init__()
        self.engine = WHCTEngine(mapping_path)
        self.overlay = HoverOverlay()
        self.is_paused = True
        self.is_rendering = False

        self.setWindowTitle("WHCT Viewer (PySide6)")
        self.resize(1500, 900)

        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(1)
        self.animation_timer.timeout.connect(self._advance_animation)

        self.live_update_timer = QTimer(self)
        self.live_update_timer.setSingleShot(True)
        self.live_update_timer.setInterval(100)
        self.live_update_timer.timeout.connect(self._commit_live_ui_update)

        self._build_ui()
        self._load_state_into_ui()
        self._render_and_show()
        self._update_status()

    # ---------- UI construction ----------
    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal, self)
        root_layout.addWidget(splitter)

        controls_scroll = QScrollArea(self)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(360)
        controls_scroll.setMaximumWidth(500)

        controls_host = QWidget()
        controls_scroll.setWidget(controls_host)
        controls_layout = QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_layout.setSpacing(10)

        splitter.addWidget(controls_scroll)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self.viewport = RenderViewport("Visualization")
        self.viewport.imagePositionChanged.connect(self._on_hover_changed)
        self.viewport.dataPanRequested.connect(self._on_viewport_pan)
        self.viewport.dataRotateRequested.connect(self._on_viewport_rotate)
        right_layout.addWidget(self.viewport, stretch=1)

        playback_panel = QWidget(self)
        playback_layout = QVBoxLayout(playback_panel)
        playback_layout.setContentsMargins(0, 0, 0, 0)
        playback_layout.setSpacing(6)

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.valueChanged.connect(self.set_frame)
        playback_layout.addWidget(self.progress_slider)

        controls_row = QWidget()
        controls_row_layout = QHBoxLayout(controls_row)
        controls_row_layout.setContentsMargins(0, 0, 0, 0)
        controls_row_layout.setSpacing(8)

        self.pause_button = QPushButton("Pause")
        self.play_button = QPushButton("Play")
        self.update_button = QPushButton("Update")
        self.save_button = QPushButton("Save Animation")

        self.pause_button.clicked.connect(self.pause_animation)
        self.play_button.clicked.connect(self.start_animation)
        self.update_button.clicked.connect(self.update_animation)
        self.save_button.clicked.connect(self.save_animation)

        for button in (self.pause_button, self.play_button, self.update_button, self.save_button):
            controls_row_layout.addWidget(button)

        playback_layout.addWidget(controls_row)
        right_layout.addWidget(playback_panel, stretch=0)

        self.lut_group = QGroupBox("LUT Preview")
        lut_layout = QVBoxLayout(self.lut_group)
        self.lut_viewport = RenderViewport("No LUT")
        self.lut_viewport.setMinimumHeight(180)
        lut_layout.addWidget(self.lut_viewport)
        self.lut_group.setVisible(self.engine.lut is not None)
        right_layout.addWidget(self.lut_group, stretch=0)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Top controls
        viewer_group = QGroupBox("Viewer")
        viewer_layout = QFormLayout(viewer_group)

        self.indicator_combo = QComboBox()
        self.indicator_combo.addItems(list(INDICATORS.keys()))
        self.indicator_combo.currentIndexChanged.connect(self._on_indicator_changed)
        viewer_layout.addRow("indicator", self.indicator_combo)

        controls_layout.addWidget(viewer_group)

        # Dimension controls
        self.dimension_controls = {}
        dim_group = QGroupBox("Dimensions")
        dim_grid = QGridLayout(dim_group)
        dim_grid.addWidget(QLabel("Axis"), 0, 0)
        dim_grid.addWidget(QLabel("Min"), 0, 1)
        dim_grid.addWidget(QLabel("Max"), 0, 2)
        dim_grid.addWidget(QLabel("Res"), 0, 3)

        for row, key in enumerate(["Y", "X", "T", "C"], start=1):
            dim_grid.addWidget(QLabel(key), row, 0)
            min_spin = QDoubleSpinBox()
            max_spin = QDoubleSpinBox()
            res_spin = QSpinBox()
            res_spin.setRange(1, 100000)
            res_spin.setValue(1)
            res_spin.setKeyboardTracking(False)

            for spin in (min_spin, max_spin):
                spin.setRange(-1_000_000.0, 1_000_000.0)
                spin.setDecimals(6)
                spin.setSingleStep(0.1)
                spin.setKeyboardTracking(False)

            for spin in (min_spin, max_spin, res_spin):
                spin.valueChanged.connect(self._schedule_live_ui_update)

            self.dimension_controls[key] = (min_spin, max_spin, res_spin)
            dim_grid.addWidget(min_spin, row, 1)
            dim_grid.addWidget(max_spin, row, 2)
            dim_grid.addWidget(res_spin, row, 3)

        controls_layout.addWidget(dim_group)

        # Rotation sliders
        rot_group = QGroupBox("Rotations (degrees)")
        rot_layout = QGridLayout(rot_group)
        self.rotation_sliders = {}
        labels = [
            ("XY", "XYrot"),
            ("XC", "XHrot"),
            ("XT", "XTrot"),
            ("YC", "YHrot"),
            ("YT", "YTrot"),
            ("CT", "HTrot"),
        ]
        for row, (label, attr) in enumerate(labels):
            rot_layout.addWidget(QLabel(f"{label} rotation"), row, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 359)
            slider.valueChanged.connect(lambda value, a=attr: self._set_rotation(a, value))
            self.rotation_sliders[attr] = slider
            rot_layout.addWidget(slider, row, 1)
        controls_layout.addWidget(rot_group)

        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)

        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(1, 100000)
        self.precision_spin.setValue(30)
        self.precision_spin.setKeyboardTracking(False)
        self.precision_spin.valueChanged.connect(self._schedule_live_ui_update)
        settings_layout.addRow("precision", self.precision_spin)

        self.snapping_checkbox = QCheckBox()
        self.snapping_checkbox.stateChanged.connect(self._on_snapping_changed)
        settings_layout.addRow("snap rotations", self.snapping_checkbox)

        controls_layout.addWidget(settings_group)

        controls_layout.addStretch(1)

        self.setStatusBar(QStatusBar(self))

        toggle_overlay = QAction("Toggle Hover Overlay", self)
        toggle_overlay.setShortcut(QKeySequence(Qt.Key_Space))
        toggle_overlay.triggered.connect(self.toggle_overlay)
        self.addAction(toggle_overlay)

    def _load_state_into_ui(self):
        s = self.engine.state
        self.indicator_combo.blockSignals(True)
        index = self.indicator_combo.findText(s.indicator)
        if index >= 0:
            self.indicator_combo.setCurrentIndex(index)
        self.indicator_combo.blockSignals(False)

        self.precision_spin.setValue(s.precision)
        self.snapping_checkbox.setChecked(s.snapping_enabled)

        for key in ["Y", "X", "T", "C"]:
            min_spin, max_spin, res_spin = self.dimension_controls[key]
            min_spin.setValue(getattr(s, f"{key}min"))
            max_spin.setValue(getattr(s, f"{key}max"))
            res_spin.setValue(getattr(s, f"{key}res"))

        self.progress_slider.blockSignals(True)
        self.progress_slider.setRange(0, max(0, s.Tres - 1))
        self.progress_slider.setValue(s.current_frame)
        self.progress_slider.blockSignals(False)

        for attr, slider in self.rotation_sliders.items():
            slider.blockSignals(True)
            slider.setValue(int(np.degrees(getattr(s, attr))) % 360)
            slider.blockSignals(False)

    # ---------- State sync ----------
    def _apply_ui_to_state(self):
        s = self.engine.state
        old_tres = s.Tres

        s.indicator = self.indicator_combo.currentText()
        s.precision = self.precision_spin.value()
        s.snapping_enabled = self.snapping_checkbox.isChecked()

        for key in ["Y", "X", "T", "C"]:
            min_spin, max_spin, res_spin = self.dimension_controls[key]
            setattr(s, f"{key}min", float(min_spin.value()))
            setattr(s, f"{key}max", float(max_spin.value()))
            setattr(s, f"{key}res", int(res_spin.value()))

        if s.Tres != old_tres:
            if old_tres > 0:
                s.current_frame = int(s.current_frame * (s.Tres / old_tres))
            s.current_frame = max(0, min(s.current_frame, max(0, s.Tres - 1)))
            self.progress_slider.blockSignals(True)
            self.progress_slider.setRange(0, max(0, s.Tres - 1))
            self.progress_slider.setValue(s.current_frame)
            self.progress_slider.blockSignals(False)

    def _schedule_live_ui_update(self, *args):
        self.live_update_timer.start()

    def _commit_live_ui_update(self):
        try:
            self._apply_ui_to_state()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        if self.is_paused:
            self._render_and_show()
        else:
            self._update_status()

    def _on_snapping_changed(self):
        self.engine.state.snapping_enabled = self.snapping_checkbox.isChecked()
        self._apply_snapping_to_all_rotations()
        if self.is_paused:
            self._render_and_show()
        else:
            self._update_status()

    def _snap_angle_rad(self, angle_rad: float) -> float:
        s = self.engine.state
        if not s.snapping_enabled:
            return angle_rad
        angle_deg = np.degrees(angle_rad)
        snapped_deg = round(angle_deg / 15.0) * 15.0
        if abs(angle_deg - snapped_deg) <= 7.5:
            return np.deg2rad(snapped_deg)
        return angle_rad

    def _apply_snapping_to_all_rotations(self):
        for attr in ("XYrot", "XTrot", "XHrot", "YTrot", "YHrot", "HTrot"):
            snapped = self._snap_angle_rad(getattr(self.engine.state, attr))
            setattr(self.engine.state, attr, snapped)

        for attr, slider in self.rotation_sliders.items():
            slider_value = int(round(np.degrees(getattr(self.engine.state, attr)))) % 360
            slider.blockSignals(True)
            slider.setValue(slider_value)
            slider.blockSignals(False)

    def _set_rotation(self, attr: str, value: int):
        setattr(self.engine.state, attr, float(value) * np.pi / 180.0)
        self._apply_snapping_to_all_rotations()
        if self.is_paused:
            self._render_and_show()

    def _on_indicator_changed(self):
        self.engine.state.indicator = self.indicator_combo.currentText()
        if self.is_paused:
            self._render_and_show()
        self._update_status()

    # ---------- Rendering ----------
    def _render_and_show(self):
        if self.is_rendering:
            return

        self.is_rendering = True
        try:
            base_bgr, lut_bgr = self.engine.render_current_base_images()
            display_bgr = base_bgr.copy()
            display_bgr = self.overlay.draw(display_bgr, self.engine.last_index_img, self.engine.state.Cres)
            self.viewport.set_numpy_bgr_image(display_bgr)
            self.lut_viewport.set_numpy_bgr_image(lut_bgr)
            self.lut_group.setVisible(lut_bgr is not None)
            self._update_status()
        finally:
            self.is_rendering = False

    def _rerender_overlay_only(self):
        if self.engine.last_base_bgr is None:
            return
        display_bgr = self.engine.last_base_bgr.copy()
        display_bgr = self.overlay.draw(display_bgr, self.engine.last_index_img, self.engine.state.Cres)
        self.viewport.set_numpy_bgr_image(display_bgr)

    # ---------- Interaction ----------
    def _on_hover_changed(self, pos):
        self.overlay.set_hover(pos)
        self._rerender_overlay_only()

    def _on_viewport_pan(self, dx_pixels: float, dy_pixels: float):
        if self.viewport._display_rect is None:
            return

        try:
            self._apply_ui_to_state()
        except Exception:
            return

        if not self.is_paused:
            self.pause_animation()

        _, _, display_width, display_height = self.viewport._display_rect
        if display_width <= 0 or display_height <= 0:
            return

        s = self.engine.state
        x_span = s.Xmax - s.Xmin
        y_span = s.Ymax - s.Ymin

        dx_data = -(dx_pixels / display_width) * x_span
        dy_data = (dy_pixels / display_height) * y_span

        s.Xmin += dx_data
        s.Xmax += dx_data
        s.Ymin += dy_data
        s.Ymax += dy_data

        x_min_spin, x_max_spin, _ = self.dimension_controls["X"]
        y_min_spin, y_max_spin, _ = self.dimension_controls["Y"]

        for spin, value in (
            (x_min_spin, s.Xmin),
            (x_max_spin, s.Xmax),
            (y_min_spin, s.Ymin),
            (y_max_spin, s.Ymax),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        self._render_and_show()

    def _on_viewport_rotate(self, dx_pixels: float, dy_pixels: float):
        try:
            self._apply_ui_to_state()
        except Exception:
            return

        if not self.is_paused:
            self.pause_animation()

        degrees_per_pixel = 0.5
        s = self.engine.state
        s.XHrot += np.deg2rad(dx_pixels * degrees_per_pixel)
        s.YHrot += np.deg2rad(dy_pixels * degrees_per_pixel)

        self._apply_snapping_to_all_rotations()
        self._render_and_show()

    def toggle_overlay(self):
        self.overlay.toggle()
        self._rerender_overlay_only()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.pause_animation()
            return
        super().keyPressEvent(event)

    # ---------- Animation ----------
    def start_animation(self):
        try:
            self._apply_ui_to_state()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        self.is_paused = False
        self.animation_timer.start()
        self._update_status()

    def pause_animation(self):
        self.is_paused = True
        self.animation_timer.stop()
        self._update_status()

    def update_animation(self):
        self.live_update_timer.stop()
        try:
            self._apply_ui_to_state()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        self.is_paused = True
        self.animation_timer.stop()
        self._render_and_show()
        self._update_status()

    def _advance_animation(self):
        self.live_update_timer.stop()
        try:
            self._apply_ui_to_state()
        except Exception as exc:
            self.pause_animation()
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        if self.is_paused:
            return

        self._render_and_show()
        s = self.engine.state
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(s.current_frame)
        self.progress_slider.blockSignals(False)

        s.current_frame += 1
        if s.current_frame >= s.Tres:
            s.current_frame = max(0, s.Tres - 1)
            self.pause_animation()

    def set_frame(self, value: int):
        self.engine.state.current_frame = int(value)
        if self.is_paused:
            self._render_and_show()

    # ---------- Save ----------
    def save_animation(self):
        try:
            self._apply_ui_to_state()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        s = self.engine.state
        os.makedirs("./videoOutput", exist_ok=True)
        video_filename = "./videoOutput/fractal_animation.mp4"
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (s.Xres, s.Yres))

        original_frame = s.current_frame
        try:
            for frame_idx in tqdm(range(s.Tres), desc="Rendering animation"):
                QApplication.processEvents()
                s.current_frame = frame_idx
                base_bgr, lut_bgr = self.engine.render_current_base_images()
                frame_bgr = lut_bgr if lut_bgr is not None else base_bgr
                video_writer.write(frame_bgr)
        finally:
            video_writer.release()
            s.current_frame = original_frame
            self._render_and_show()

        QMessageBox.information(self, "Saved", f"Animation saved to:\n{os.path.abspath(video_filename)}")

    # ---------- Misc ----------
    def _update_status(self):
        s = self.engine.state
        state_text = "paused" if self.is_paused else "playing"
        self.statusBar().showMessage(
            f"device: {self.engine.device} | indicator: {s.indicator} | frame: {s.current_frame + 1}/{max(1, s.Tres)} | {state_text}"
        )



def main(mapping_path: Optional[str] = None):
    app = QApplication(sys.argv)
    window = WHCTMainWindow(mapping_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Optional .npz mapping file. If provided, a corresponding .pt LUT is loaded/created and used.",
    )
    args = ap.parse_args()
    main(mapping_path=args.mapping)
