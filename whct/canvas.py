"""The main view: the rendered image plus the two floating widgets, in one OpenGL canvas.

On the image: left-drag pans W/H, right-drag pans T (horizontal) and C (vertical),
scroll zooms W and H about the mouse (hold 1-4 to pick W, H, C, T instead).
Widgets sit side by side in the top-right corner. Click one to expand it; it stays
expanded until you click somewhere else. In full screen (compact) the collapsed
widgets shrink to tiny icon squares.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QPushButton

from .colormap import bit_positions
from .gizmos import MATH_COLORS, RotationGizmo, ScaleGizmo, dim_panel, paint_mini, paint_panel, plane_name
from .state import DISPLAY_AXES, MATH_AXES, W, H, C, T
from .valuefields import ValueBox


def fmt_num(x: float) -> str:
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
        return f"{x:.3g}".replace("-", "−")
    return f"{x:.4g}".replace("-", "−")


def fmt_part(x: float) -> str:
    v = 0.0 if abs(x) < 0.0005 else x
    return f"{v:.3f}".replace("-", "−")


class Canvas(QOpenGLWidget):
    def __init__(self, ctl, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        self.rot = RotationGizmo()
        self.scl = ScaleGizmo()
        self._rgb: Optional[np.ndarray] = None
        self._index: Optional[np.ndarray] = None
        self._tex = None
        self._tex_dirty = False
        self._drag: Optional[dict] = None
        self.probe_on = False
        self._mouse: Optional[QPointF] = None
        self.boxes: Dict[str, ValueBox] = {}
        self.compact = False  # full screen: collapsed widgets are tiny squares

    # ----- image -----
    def set_image(self, rgb: np.ndarray, index: np.ndarray) -> None:
        self._rgb = np.ascontiguousarray(rgb)
        self._index = index
        self._tex_dirty = True
        self.update()

    def image_rect(self) -> QRectF:
        iw, ih = self.ctl.render_size()
        cw, ch = self.width(), self.height()
        if self.ctl.settings.display_mode == "actual":
            dpr = self.devicePixelRatioF()
            w, h = iw / dpr, ih / dpr
        else:
            s = min(cw / max(1, iw), ch / max(1, ih))
            w, h = iw * s, ih * s
        return QRectF((cw - w) / 2, (ch - h) / 2, w, h)

    def pos_to_u(self, pos: QPointF) -> np.ndarray:
        r = self.image_rect()
        return np.array([
            (pos.x() - r.left()) / max(1.0, r.width()) - 0.5,
            0.5 - (pos.y() - r.top()) / max(1.0, r.height()),
            0.0,
            self.ctl.state.u_t(),
        ])

    # ----- widget layout -----
    def layout_widgets(self) -> None:
        side = float(np.clip(round(self.width() / 12), 90, 170))
        margin = 12.0
        x_right = self.width() - margin
        for g in (self.scl, self.rot):  # scale on the far right, rotation to its left
            s = side * 2 if g.expanded else (30.0 if self.compact else side)
            g.rect = QRectF(x_right - s, margin, s, s)
            x_right -= s + 8
        self.refresh_value_boxes()

    def collapse_widgets(self) -> None:
        if self.rot.expanded or self.scl.expanded:
            self.rot.expanded = self.scl.expanded = False
            self.layout_widgets()
            self.update()

    def set_compact(self, on: bool) -> None:
        self.compact = on
        self.rot.expanded = self.scl.expanded = False
        self.layout_widgets()
        self.update()

    def _is_mini(self, g) -> bool:
        return self.compact and not g.expanded

    def _expand(self, which: str) -> None:
        self.rot.expanded = which == "rot"
        self.scl.expanded = which == "scale"
        self.layout_widgets()
        self.update()

    def _widget_at(self, pos: QPointF) -> Optional[str]:
        if self.rot.rect.contains(pos):
            return "rot"
        if self.scl.rect.contains(pos):
            return "scale"
        return None

    def resizeGL(self, w: int, h: int) -> None:
        self.layout_widgets()
        self.ctl.on_canvas_resized()

    def device_size(self):
        dpr = self.devicePixelRatioF()
        return int(round(self.width() * dpr)), int(round(self.height() * dpr))

    # ----- painting -----
    def initializeGL(self) -> None:
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

    def paintGL(self) -> None:
        dpr = self.devicePixelRatioF()
        fb_w, fb_h = int(self.width() * dpr), int(self.height() * dpr)
        st = self.ctl.state
        p = QPainter(self)
        p.beginNativePainting()
        self._reset_gl()
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self._draw_image()
        self._restore_gl(fb_w, fb_h)
        p.endNativePainting()

        paint_panel(p, self.rot.rect, self.rot.expanded)
        paint_panel(p, self.scl.rect, self.scl.expanded)

        if self._is_mini(self.rot):
            paint_mini(p, self.rot.rect, "rot")
        else:
            p.beginNativePainting()
            self._reset_gl()
            self.rot.paint_gl(st, dpr, fb_h)
            self._restore_gl(fb_w, fb_h)
            p.endNativePainting()
            self.rot.paint_labels(p, st)
        if self._is_mini(self.scl):
            paint_mini(p, self.scl.rect, "scale")
        else:
            self.scl.paint(p, st)
        if self.ctl.values_on:
            for g in (self.rot, self.scl):
                if g.expanded:
                    dim_panel(p, g.rect)
        if self.probe_on and self._drag is None:
            self._paint_probe(p)
        p.end()

    def _reset_gl(self) -> None:
        gl.glUseProgram(0)
        for i in range(3):
            try:
                gl.glDisableVertexAttribArray(i)
            except Exception:
                pass
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glDisable(gl.GL_STENCIL_TEST)

    def _restore_gl(self, fb_w: int, fb_h: int) -> None:
        # Hand the state QPainter expects back after fixed-function drawing.
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def _draw_image(self) -> None:
        if self._rgb is None:
            return
        if self._tex is None:
            self._tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex)
        nearest = self.ctl.settings.display_mode == "actual"
        filt = gl.GL_NEAREST if nearest else gl.GL_LINEAR
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filt)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filt)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        if self._tex_dirty:
            h, w = self._rgb.shape[:2]
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB8, w, h, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, self._rgb)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)  # Qt's glyph cache expects the default
            self._tex_dirty = False
        r = self.image_rect()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width(), self.height(), 0, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0); gl.glVertex2f(r.left(), r.top())
        gl.glTexCoord2f(1, 0); gl.glVertex2f(r.right(), r.top())
        gl.glTexCoord2f(1, 1); gl.glVertex2f(r.right(), r.bottom())
        gl.glTexCoord2f(0, 1); gl.glVertex2f(r.left(), r.bottom())
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def _paint_probe(self, p: QPainter) -> None:
        if self._index is None or self._mouse is None or self._widget_at(self._mouse):
            return
        r = self.image_rect()
        if not r.contains(self._mouse):
            return
        ih, iw = self._index.shape
        ix = int((self._mouse.x() - r.left()) / r.width() * iw)
        iy = int((self._mouse.y() - r.top()) / r.height() * ih)
        ix, iy = min(max(ix, 0), iw - 1), min(max(iy, 0), ih - 1)
        val = int(self._index[iy, ix])
        n_c = self.ctl.state.res[C]
        pos = bit_positions(n_c)
        filled = [(val >> int(pos[i])) & 1 for i in range(n_c)]
        bar_w, bar_h, gap, pad = 14, max(3, min(7, int(160 / max(1, n_c)))), 2, 5
        box_h = n_c * bar_h + (n_c - 1) * gap + 2 * pad + 14
        box_w = bar_w + 2 * pad + 26
        x = self._mouse.x() + 14
        y = self._mouse.y() + 14
        if x + box_w > self.width():
            x = self._mouse.x() - 14 - box_w
        if y + box_h > self.height():
            y = self._mouse.y() - 14 - box_h
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 60), 1))
        p.setBrush(QColor(14, 16, 19, 215))
        p.drawRoundedRect(QRectF(x, y, box_w, box_h), 6, 6)
        p.setFont(QFont("Consolas", 7))
        p.setPen(QColor(163, 170, 183))
        p.drawText(QPointF(x + pad, y + pad + 8), f"{sum(filled)}/{n_c}")
        for i in range(n_c):
            k = n_c - 1 - i  # high C at the top
            by = y + pad + 14 + i * (bar_h + gap)
            rect = QRectF(x + pad, by, bar_w, bar_h)
            p.setPen(QPen(QColor(200, 204, 212), 1))
            p.setBrush(QColor(91, 214, 154) if filled[k] else QColor(40, 44, 52))
            p.drawRect(rect)
        p.setPen(QPen(QColor(255, 255, 255), 1.2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(self._mouse, 4, 4)
        p.restore()

    # ----- mouse -----
    def mousePressEvent(self, e):
        pos = e.position()
        self.setFocus()
        target = self._widget_at(pos)
        ctrl = bool(e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)) or self.ctl.tumble_lock
        st = self.ctl.state
        self._drag = None
        if target is not None:
            g = self.rot if target == "rot" else self.scl
            if not g.expanded:
                self._drag = {"kind": "expand", "target": target, "start": pos}
                return
            if target == "rot":
                if ctrl:
                    self._drag = {"kind": "tumble", "last": pos}
                elif not self.ctl.values_on and e.button() == Qt.LeftButton:
                    h = self.rot.hit_plane(pos)
                    if h is not None:
                        self.ctl.begin_gesture()
                        self._drag = {"kind": "plane", "h": h, "last": pos}
            elif not self.ctl.values_on and e.button() == Qt.LeftButton:
                k = self.scl.hit_handle(st, pos)
                if k is not None:
                    self.ctl.begin_gesture()
                    self.scl.begin_drag(st)
                    self._drag = {"kind": "span", "k": k, "last": pos}
            return
        self.collapse_widgets()  # clicking off the widgets collapses them
        if e.button() in (Qt.LeftButton, Qt.RightButton):
            self.ctl.begin_gesture()
            self._drag = {"kind": "pan_wh" if e.button() == Qt.LeftButton else "pan_tc", "last": pos,
                          "start": pos, "applied": QPointF(0, 0)}
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        pos = e.position()
        self._mouse = pos
        d = self._drag
        if d is None:
            target = self._widget_at(pos)
            g = {"rot": self.rot, "scale": self.scl}.get(target)
            if g is not None and not g.expanded:
                self.setCursor(Qt.PointingHandCursor)  # a click expands it
            else:
                self.unsetCursor()
            self.ctl.canvas_hover(pos)
            if self.probe_on:
                self.update()
            return
        if d["kind"] == "expand":
            if (pos - d["start"]).manhattanLength() > 5:
                self._drag = None
            return
        dx, dy = pos.x() - d["last"].x(), pos.y() - d["last"].y()
        d["last"] = pos
        if d["kind"] in ("pan_wh", "pan_tc"):
            # Holding Ctrl: a drag that is close to horizontal or vertical snaps fully to that axis.
            tx, ty = pos.x() - d["start"].x(), pos.y() - d["start"].y()
            if e.modifiers() & Qt.ControlModifier:
                ang = math.degrees(math.atan2(abs(ty), abs(tx)))
                if ang < 20:
                    ty = 0.0
                elif ang > 70:
                    tx = 0.0
            dx, dy = tx - d["applied"].x(), ty - d["applied"].y()
            d["applied"] = QPointF(tx, ty)
        if d["kind"] == "tumble":
            self.rot.tumble(dx, dy)
            self.refresh_value_boxes()
            self.update()
        elif d["kind"] == "plane":
            t = d["h"]["tangent"]
            self.ctl.rotate_plane(d["h"]["i"], d["h"]["k"], (dx * t[0] + dy * t[1]) * 0.012)
        elif d["kind"] == "span":
            k = d["k"]
            self.ctl.set_span(k, self.scl.dragged_span(self.ctl.state, k, dx, dy))
        elif d["kind"] in ("pan_wh", "pan_tc"):
            r = self.image_rect()
            fx, fy = dx / max(1.0, r.width()), dy / max(1.0, r.height())
            du = np.zeros(4)
            if d["kind"] == "pan_wh":
                du[W], du[H] = -fx, fy
            else:
                du[T], du[C] = -fx, fy
            self.ctl.pan(du)

    def mouseReleaseEvent(self, e):
        d = self._drag
        self._drag = None
        self.unsetCursor()
        if d is None:
            return
        if d["kind"] == "expand":
            self._expand(d["target"])
        elif d["kind"] == "span":
            self.scl.end_drag()
            self.ctl.end_gesture()
        elif d["kind"] in ("plane", "pan_wh", "pan_tc"):
            self.ctl.end_gesture()
        self.update()

    def leaveEvent(self, e):
        self._mouse = None
        self.update()

    def wheelEvent(self, e):
        pos = e.position()
        if self._widget_at(pos):
            return
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        dims = sorted(self.ctl.held_dims) or [W, H]
        self.ctl.zoom(dims, 1.15 ** (-steps), self.pos_to_u(pos))

    # ----- values mode fields -----
    def refresh_value_boxes(self) -> None:
        st = self.ctl.state
        wanted = set()
        if self.ctl.values_on and self.rot.expanded:
            for j, spot in self.rot.label_spots(st):
                key = f"v{j}"
                wanted.add(key)
                box = self.boxes.get(key) or self._make_vector_box(j)
                for c, f in enumerate(box.fields):
                    f.set_display(fmt_part(st.R[j, c]))
                box.place_centered(spot.x(), spot.y() + 14)
            # Plane "turn by" chips sit in a strip under the widget so they don't cover the arrows.
            r = self.rot.rect
            for n, h in enumerate(self.rot.plane_handles()):
                key = f"p{h['i']}{h['k']}"
                wanted.add(key)
                box = self.boxes.get(key) or self._make_plane_box(h["i"], h["k"])
                row, col = divmod(n, 3)
                box.place_centered(r.left() + r.width() * (col + 0.5) / 3, r.bottom() + 18 + row * 30)
        if self.ctl.values_on and self.scl.expanded:
            for k, spot in self.scl.label_spots(st):
                key = f"s{k}"
                wanted.add(key)
                box = self.boxes.get(key) or self._make_span_box(k)
                box.fields[0].set_display(fmt_num(st.spans[k]))
                box.place_centered(spot.x(), spot.y())
        for key in list(self.boxes):
            if key not in wanted:
                self.boxes.pop(key).deleteLater()
        self._update_reset_buttons()

    def _update_reset_buttons(self) -> None:
        if not hasattr(self, "reset_btns"):
            self.reset_btns = {}
            for name, g, fn in (("rot", self.rot, self.ctl.reset_rotation), ("scale", self.scl, self.ctl.reset_spans)):
                b = QPushButton("Reset", self)
                b.setFocusPolicy(Qt.NoFocus)
                b.setStyleSheet("QPushButton { background: #2B3039; color: #E7E9EE; border: 1px solid #4E5563;"
                                " border-radius: 5px; padding: 2px 8px; } QPushButton:hover { background: #3A4150; }")
                b.clicked.connect(fn)
                b.hide()
                self.reset_btns[name] = (b, g)
        for b, g in self.reset_btns.values():
            show = self.ctl.values_on and g.expanded
            b.setVisible(show)
            if show:
                b.adjustSize()
                b.move(int(g.rect.right() - b.width() - 6), int(g.rect.bottom() - b.height() - 6))
                b.raise_()

    def _add_box(self, key, box):
        self.boxes[key] = box
        box.show()
        return box

    def _make_vector_box(self, j: int) -> ValueBox:
        color = "#%02x%02x%02x" % MATH_COLORS[j]
        box = ValueBox(self, MATH_AXES[j], color, list(DISPLAY_AXES), minimized=True)
        for c, f in enumerate(box.fields):
            f.committed.connect(lambda v, j=j, c=c: self.ctl.edit_component(j, c, v))
            f.scrubStarted.connect(self.ctl.begin_gesture)
            f.scrubbed.connect(lambda dx, j=j, c=c: self.ctl.set_component(j, c, self.ctl.state.R[j, c] + dx * 0.005))
            f.scrubEnded.connect(self.ctl.end_gesture)
        return self._add_box(f"v{j}", box)

    def _make_plane_box(self, i: int, k: int) -> ValueBox:
        box = ValueBox(self, plane_name(i, k), "#A3AAB7", ["turn °"], minimized=True)
        f = box.fields[0]
        f.setPlaceholderText("°")
        acc = {"deg": 0.0}

        def commit(v):
            self.ctl.begin_gesture()
            self.ctl.rotate_plane(i, k, math.radians(v))
            self.ctl.end_gesture()
            f.setText("")

        def scrub(dx):
            step = dx * 0.5
            acc["deg"] += step
            self.ctl.rotate_plane(i, k, math.radians(step))
            f.setText(("+" if acc["deg"] >= 0 else "−") + f"{abs(acc['deg']):.0f}°")

        def start():
            acc["deg"] = 0.0
            self.ctl.begin_gesture()

        def end():
            self.ctl.end_gesture()
            f.setText("")

        f.committed.connect(commit)
        f.scrubStarted.connect(start)
        f.scrubbed.connect(scrub)
        f.scrubEnded.connect(end)
        return self._add_box(f"p{i}{k}", box)

    def _make_span_box(self, k: int) -> ValueBox:
        box = ValueBox(self, DISPLAY_AXES[k], "#E7E9EE", [DISPLAY_AXES[k]], minimized=False)
        f = box.fields[0]
        f.committed.connect(lambda v, k=k: self.ctl.edit_span(k, v))
        f.scrubStarted.connect(self.ctl.begin_gesture)
        f.scrubbed.connect(lambda dx, k=k: self.ctl.set_span(k, self.ctl.state.spans[k] * math.exp(dx * 0.01)))
        f.scrubEnded.connect(self.ctl.end_gesture)
        return self._add_box(f"s{k}", box)
