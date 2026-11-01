"""The three floating navigation widgets.

PanGizmo: four sliders, one per physical dimension. Drag a knob left or right to
move the view along W, H, C or T; it springs back to the middle on release, and the
number beside it is the view's position along that dimension. Panning any other way
(dragging the image, scrubbing a value) moves the knobs the same way.

RotationGizmo: a real 3D axis frame (W, H, C, with the corner r standing for T)
drawn with OpenGL. Each math axis is an arrow: its direction is its W/H/C mix and
its length is how much of it lies outside T. Six plane arrows turn the view.

ScaleGizmo: a tesseract drawn as two joined cubes. The four edges from vertex v
are the W, H, C and T spans; dragging one scales that dimension.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from .state import DISPLAY_AXES, MATH_AXES, ViewState

MATH_COLORS = [(255, 159, 90), (79, 209, 194), (130, 174, 255), (199, 169, 255)]  # x, y, z, w
INK = QColor(231, 233, 238)
MUTED = QColor(163, 170, 183)
FAINT = QColor(110, 118, 132)
PANEL = QColor(20, 23, 28, 225)
PANEL_EDGE = QColor(255, 255, 255, 40)
PANEL_EDGE_ACTIVE = QColor(255, 255, 255, 140)


def plane_name(i: int, k: int) -> str:
    """WH, WC, HC for the arcs; WT, HT, CT for the short arrows at r."""
    return DISPLAY_AXES[k] + "T" if i == 3 else DISPLAY_AXES[i] + DISPLAY_AXES[k]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _e3(k, s=1.0):
    e = np.zeros(3)
    e[k] = s
    return e


def paint_panel(p: QPainter, rect: QRectF, active: bool) -> None:
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(PANEL_EDGE_ACTIVE if active else PANEL_EDGE, 1.2))
    p.setBrush(PANEL)
    p.drawRoundedRect(rect, 8, 8)
    p.restore()


def paint_mini(p: QPainter, rect: QRectF, kind: str) -> None:
    """Icon for a widget shrunk to a tiny square (full screen): a turn arrow or two joined squares."""
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(MUTED, 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    c, r = rect.center(), rect.width() * 0.26
    if kind == "rot":
        p.drawArc(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), 90 * 16, 270 * 16)
        tip = QPointF(c.x(), c.y() - r)  # arrowhead where the arc ends, pointing clockwise
        p.drawLine(tip, QPointF(tip.x() - r * 0.5, tip.y() - r * 0.4))
        p.drawLine(tip, QPointF(tip.x() - r * 0.5, tip.y() + r * 0.4))
    elif kind == "pan":
        for n, (y, x) in enumerate(((-0.75, 0.35), (0.0, -0.45), (0.75, 0.1))):
            yy = c.y() + y * r
            p.drawLine(QPointF(c.x() - r * 1.15, yy), QPointF(c.x() + r * 1.15, yy))
            p.setBrush(MUTED)
            p.drawEllipse(QPointF(c.x() + x * r, yy), r * 0.24, r * 0.24)
            p.setBrush(Qt.NoBrush)
    else:
        s, o = r * 1.3, r * 0.55
        a, b = QRectF(c.x() - o - s / 2, c.y() + o - s / 2, s, s), QRectF(c.x() + o - s / 2, c.y() - o - s / 2, s, s)
        p.drawRect(a)
        p.drawRect(b)
        for pa, pb in ((a.topLeft(), b.topLeft()), (a.topRight(), b.topRight()),
                       (a.bottomLeft(), b.bottomLeft()), (a.bottomRight(), b.bottomRight())):
            p.drawLine(pa, pb)
    p.restore()


def dim_panel(p: QPainter, rect: QRectF) -> None:
    p.save()
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(20, 23, 28, 170))
    p.drawRoundedRect(rect, 8, 8)
    p.restore()


# ----------------------------------------------------------------------------------
# Rotation widget
# ----------------------------------------------------------------------------------

class RotationGizmo:
    EXTENT = 1.5  # half-width of the visible 3D region, in axis units
    YAW0 = math.atan2(0.6, 1.0)
    PITCH0 = math.asin(0.5 / math.sqrt(0.36 + 0.25 + 1.0))
    PLANES_3D = [(0, 1), (0, 2), (1, 2)]  # WH, WC, HC arcs

    def __init__(self):
        self.rect = QRectF()
        self.expanded = False
        self.yaw, self.pitch = self.YAW0, self.PITCH0
        self._update_camera()

    # ----- camera -----
    def reset_camera(self):
        self.yaw, self.pitch = self.YAW0, self.PITCH0
        self._update_camera()

    def tumble(self, dx: float, dy: float):
        self.yaw -= dx * 0.01
        self.pitch = float(np.clip(self.pitch + dy * 0.01, -1.45, 1.45))
        self._update_camera()

    def _update_camera(self):
        cp = math.cos(self.pitch)
        self.cam_e = np.array([cp * math.sin(self.yaw), math.sin(self.pitch), cp * math.cos(self.yaw)])
        self.cam_r = _unit(np.array([self.cam_e[2], 0.0, -self.cam_e[0]]))
        self.cam_u = np.cross(self.cam_e, self.cam_r)

    def scale_px(self) -> float:
        return self.rect.width() / 2.0 / self.EXTENT

    def project(self, p3) -> QPointF:
        s = self.scale_px()
        c = self.rect.center()
        return QPointF(c.x() + s * float(p3 @ self.cam_r), c.y() - s * float(p3 @ self.cam_u))

    # ----- plane arrows -----
    def plane_handles(self) -> List[dict]:
        """Screen geometry of the six plane arrows: points, drag tangent and label spot."""
        handles = []
        for i, k in self.PLANES_3D:
            a0, a1 = 0.16, math.pi / 2 - 0.16
            pts3 = [0.6 * (math.cos(t) * _e3(i) + math.sin(t) * _e3(k)) for t in np.linspace(a0, a1, 21)]
            pts = [self.project(p) for p in pts3]
            tan = np.array([pts[11].x() - pts[9].x(), pts[11].y() - pts[9].y()])
            label = self.project(0.75 * math.sqrt(0.5) * (_e3(i) + _e3(k)))
            handles.append({"i": i, "k": k, "pts3": pts3, "pts": pts, "tangent": _unit(tan), "label": label})
        for k in range(3):
            off = 0.08 * _unit(np.cross(_e3(k), self.cam_e))
            a3, b3 = 0.13 * _e3(k) + off, 0.38 * _e3(k) + off
            a, b = self.project(a3), self.project(b3)
            tan = _unit(np.array([b.x() - a.x(), b.y() - a.y()]))
            label = self.project(0.255 * _e3(k) + 2.2 * off)
            handles.append({"i": 3, "k": k, "pts3": [a3, b3], "pts": [a, b], "tangent": tan, "label": label})
        return handles

    def hit_plane(self, pos: QPointF) -> Optional[dict]:
        best, best_d = None, 11.0
        for h in self.plane_handles():
            pts = h["pts"]
            for p0, p1 in zip(pts, pts[1:]):
                d = _dist_to_segment(pos, p0, p1)
                if d < best_d:
                    best, best_d = h, d
        return best

    def label_spots(self, st: ViewState) -> List[Tuple[int, QPointF]]:
        """Where each math axis label sits, for the values fields."""
        spots = []
        ring = 0
        for j in range(4):
            tip = st.R[j, :3]
            P = self.project(tip)
            O = self.rect.center()
            d = np.array([P.x() - O.x(), P.y() - O.y()])
            if np.linalg.norm(tip) > 0.06 and np.linalg.norm(d) > 6:
                d = _unit(d)
                spots.append((j, QPointF(P.x() + d[0] * 13 - d[1] * 9, P.y() + d[1] * 13 + d[0] * 9)))
            else:
                ang = -0.9 + j * 0.8
                rr = 12 + ring * 6
                ring += 1
                spots.append((j, QPointF(O.x() + math.cos(ang) * (rr + 14), O.y() + math.sin(ang) * (rr + 14))))
        return spots

    # ----- OpenGL drawing -----
    def paint_gl(self, st: ViewState, dpr: float, fb_height: int) -> None:
        r = self.rect
        x, y = int(r.left() * dpr), int(fb_height - r.bottom() * dpr)
        w, h = int(r.width() * dpr), int(r.height() * dpr)
        gl.glUseProgram(0)
        gl.glViewport(x, y, w, h)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(x, y, w, h)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        e = self.EXTENT
        gl.glOrtho(-e, e, -e, e, -10, 10)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        # Light fixed to the camera, from upper left.
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (-0.35, 0.6, 1.0, 0.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (0.32, 0.32, 0.34, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (0.85, 0.85, 0.85, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (0.35, 0.35, 0.35, 1.0))
        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (0.1, 0.1, 0.1, 1.0))
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, (0.4, 0.4, 0.4, 1.0))
        gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, 24.0)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_NORMALIZE)
        view = np.eye(4)
        view[0, :3], view[1, :3], view[2, :3] = self.cam_r, self.cam_u, self.cam_e
        gl.glLoadMatrixd(np.ascontiguousarray(view.T))

        small = not self.expanded
        # Axes: solid positive halves, dashed negative halves.
        gl.glEnable(gl.GL_LIGHTING)
        for k in range(3):
            _cylinder(np.zeros(3), 1.08 * _e3(k), 0.012, (0.62, 0.66, 0.72, 1.0))
        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineStipple(2, 0x3333)
        gl.glLineWidth(1.2)
        gl.glColor4f(0.45, 0.48, 0.54, 1.0)
        gl.glBegin(gl.GL_LINES)
        for k in range(3):
            gl.glVertex3d(0, 0, 0)
            gl.glVertex3d(*(-0.62 * _e3(k)))
        gl.glEnd()
        gl.glDisable(gl.GL_LINE_STIPPLE)

        # Plane arrows (hidden when collapsed to keep the small view clean).
        if not small:
            gl.glLineWidth(2.0)
            for h in self.plane_handles():
                pts3 = h["pts3"]
                gl.glDisable(gl.GL_LIGHTING)
                gl.glColor4f(0.55, 0.58, 0.64, 0.95)
                gl.glBegin(gl.GL_LINE_STRIP)
                for p in pts3:
                    gl.glVertex3d(*p)
                gl.glEnd()
                gl.glEnable(gl.GL_LIGHTING)
                d_end = _unit(pts3[-1] - pts3[-2])
                _cone(pts3[-1] - 0.07 * d_end, pts3[-1] + 0.03 * d_end, 0.035, (0.6, 0.63, 0.69, 1.0))
                if h["i"] == 3:
                    d_start = _unit(pts3[0] - pts3[1])
                    _cone(pts3[0] - 0.07 * d_start, pts3[0] + 0.03 * d_start, 0.035, (0.6, 0.63, 0.69, 1.0))

        # Origin r.
        gl.glEnable(gl.GL_LIGHTING)
        _sphere(np.zeros(3), 0.035, (0.9, 0.92, 0.95, 1.0))

        # Math arrows.
        ring = 0
        for j in range(4):
            v = st.R[j]
            tip, t = v[:3], v[3]
            length = float(np.linalg.norm(tip))
            col = tuple(c / 255 for c in MATH_COLORS[j]) + (1.0,)
            if length > 0.06:
                d = tip / length
                head = min(0.17, 0.45 * length)
                base = tip - head * d
                gl.glEnable(gl.GL_LIGHTING)
                if length - head > 0.01:
                    _cylinder(np.zeros(3), base, 0.03 if not small else 0.04, col)
                _cone(base, tip, 0.075 if not small else 0.09, col, wire=t < -0.005)
                if abs(tip[1]) > 0.04:  # drop line to the W-C floor
                    gl.glDisable(gl.GL_LIGHTING)
                    gl.glEnable(gl.GL_LINE_STIPPLE)
                    gl.glLineStipple(1, 0x3333)
                    gl.glLineWidth(1.0)
                    gl.glColor4f(col[0], col[1], col[2], 0.6)
                    foot = np.array([tip[0], 0.0, tip[2]])
                    gl.glBegin(gl.GL_LINES)
                    gl.glVertex3d(*tip)
                    gl.glVertex3d(*foot)
                    gl.glEnd()
                    gl.glDisable(gl.GL_LINE_STIPPLE)
                    gl.glEnable(gl.GL_LIGHTING)
                    _sphere(foot, 0.018, col)
            else:
                # Lies (almost) entirely in T: a ring around r, facing the camera.
                gl.glDisable(gl.GL_LIGHTING)
                gl.glLineWidth(2.5)
                gl.glColor4f(*col)
                rad = 0.09 + 0.05 * ring
                ring += 1
                gl.glBegin(gl.GL_LINE_LOOP)
                for a in np.linspace(0, 2 * math.pi, 40, endpoint=False):
                    p = rad * (math.cos(a) * self.cam_r + math.sin(a) * self.cam_u)
                    gl.glVertex3d(*p)
                gl.glEnd()

        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glLineWidth(1.0)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()

    def paint_labels(self, p: QPainter, st: ViewState) -> None:
        small = not self.expanded
        p.save()
        p.setRenderHint(QPainter.TextAntialiasing)
        mono = QFont("Consolas", 8 if small else 10)
        mono.setBold(True)
        p.setFont(mono)
        for k in range(3):
            pt = self.project(1.34 * _e3(k))
            _text(p, pt, DISPLAY_AXES[k], INK)
            if not small:
                f2 = QFont("Consolas", 8)
                p.setFont(f2)
                _text(p, self.project(-0.74 * _e3(k)), "−" + DISPLAY_AXES[k], FAINT)
                p.setFont(mono)
        O = self.rect.center()
        if not small:
            _text(p, QPointF(O.x() - 16, O.y() + 16), "T", INK)
            f2 = QFont("Consolas", 8)
            p.setFont(f2)
            for h in self.plane_handles():
                _text(p, h["label"], plane_name(h["i"], h["k"]), MUTED)
            p.setFont(mono)
        for j, spot in self.label_spots(st):
            _text(p, spot, MATH_AXES[j], QColor(*MATH_COLORS[j]))
        p.restore()


# ----------------------------------------------------------------------------------
# Pan widget
# ----------------------------------------------------------------------------------

class PanGizmo:
    """Four spring-back sliders. One track width moves the view by one span of that dimension."""

    def __init__(self):
        self.rect = QRectF()
        self.expanded = False
        self.travel = np.zeros(4)  # this gesture's pan so far, in spans; the knobs show it

    def _rows(self):
        """Per dimension: (y, track left x, track right x)."""
        r = self.rect
        pad = 14.0 if self.expanded else 8.0
        label_w = 20.0 if self.expanded else 14.0
        readout_w = 64.0 if self.expanded else 0.0
        x0, x1 = r.left() + pad + label_w, r.right() - pad - readout_w
        step = (r.height() - 2 * pad) / 4
        return [(r.top() + pad + step * (k + 0.5), x0, x1) for k in range(4)]

    def track_width(self) -> float:
        _, x0, x1 = self._rows()[0]
        return max(1.0, x1 - x0)

    def hit_row(self, pos: QPointF) -> Optional[int]:
        rows = self._rows()
        half = (rows[1][0] - rows[0][0]) / 2
        for k, (y, x0, x1) in enumerate(rows):
            if abs(pos.y() - y) <= half and x0 - 12 <= pos.x() <= x1 + 12:
                return k
        return None

    def label_spots(self) -> List[Tuple[int, QPointF]]:
        """Where values-mode fields go: over each track."""
        return [(k, QPointF((x0 + x1) / 2 + 20, y)) for k, (y, x0, x1) in enumerate(self._rows())]

    def paint(self, p: QPainter, st: ViewState) -> None:
        small = not self.expanded
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont("Consolas", 10 if not small else 8)
        font.setBold(True)
        num_font = QFont("Consolas", 9)
        for k, (y, x0, x1) in enumerate(self._rows()):
            mid = (x0 + x1) / 2
            p.setPen(QPen(FAINT, 2.0, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
            p.setPen(QPen(FAINT, 1.2))
            p.drawLine(QPointF(mid, y - 5), QPointF(mid, y + 5))
            kx = float(np.clip(mid + self.travel[k] * (x1 - x0), x0, x1))
            p.setPen(Qt.NoPen)
            p.setBrush(INK)
            p.drawEllipse(QPointF(kx, y), 6.0 if not small else 4.0, 6.0 if not small else 4.0)
            p.setFont(font)
            _text(p, QPointF(x0 - (12 if not small else 8), y), DISPLAY_AXES[k], INK)
            if not small:
                p.setFont(num_font)
                fm = p.fontMetrics()
                s = _fmt(st.position(k))
                p.setPen(MUTED)
                p.drawText(QPointF(self.rect.right() - 12 - fm.horizontalAdvance(s), y + fm.ascent() / 2 - 1), s)
        p.restore()


def _fmt(x: float) -> str:
    if x != 0 and (abs(x) >= 1e4 or abs(x) < 1e-3):
        return f"{x:+.3g}".replace("-", "−")
    return f"{x:+.4g}".replace("-", "−")


# ----------------------------------------------------------------------------------
# Scale widget
# ----------------------------------------------------------------------------------

class ScaleGizmo:
    # Screen directions (y down) of the four physical dimensions. T runs along the
    # back-right C-W diagonal, so at equal spans the two cubes touch edge to edge.
    _DW, _DH, _DC = np.array([1.0, 0.0]), np.array([0.0, -1.0]), np.array([-0.56, 0.5])
    DIRS = [_DW, _DH, _DC, _DW - _DC]
    GAMMA = 1.0 / 3.0  # cube-root compression: 1000:1 shows as 10:1

    def __init__(self):
        self.rect = QRectF()
        self.expanded = False
        self._frozen = None  # (g, scale, origin) held during a drag so the drawing doesn't jump

    def _lengths(self, spans, g):
        return [float(np.clip((spans[k] / g) ** self.GAMMA, 0.08, 4.0)) for k in range(4)]

    def _layout(self, st: ViewState):
        if self._frozen is not None:
            g, scale, origin = self._frozen
        else:
            g = float(np.prod(st.spans) ** 0.25)
        lens = self._lengths(st.spans, g)
        verts = []
        for b in range(16):
            v = np.zeros(2)
            for k in range(4):
                if b >> k & 1:
                    v += lens[k] * self.DIRS[k]
            verts.append(v)
        if self._frozen is None:
            vs = np.array(verts)
            lo, hi = vs.min(axis=0), vs.max(axis=0)
            pad = 22 if self.expanded else 10
            avail_w = max(10.0, self.rect.width() - 2 * pad)
            avail_h = max(10.0, self.rect.height() - 2 * pad)
            scale = min(avail_w / max(1e-6, hi[0] - lo[0]), avail_h / max(1e-6, hi[1] - lo[1]))
            mid = (lo + hi) / 2
            c = self.rect.center()
            origin = np.array([c.x(), c.y()]) - mid * scale
        pts = [QPointF(*(origin + v * scale)) for v in verts]
        return g, scale, origin, lens, pts

    def handle_points(self, st: ViewState) -> List[QPointF]:
        _, _, _, _, pts = self._layout(st)
        return [pts[1 << k] for k in range(4)]

    def label_spots(self, st: ViewState) -> List[Tuple[int, QPointF]]:
        _, _, _, _, pts = self._layout(st)
        spots = []
        for k in range(4):
            d = _unit(self.DIRS[k])
            q = pts[1 << k]
            spots.append((k, QPointF(q.x() + d[0] * 30, q.y() + d[1] * 30)))
        return spots

    def hit_handle(self, st: ViewState, pos: QPointF) -> Optional[int]:
        best, best_d = None, 12.0
        for k, q in enumerate(self.handle_points(st)):
            d = math.hypot(pos.x() - q.x(), pos.y() - q.y())
            if d < best_d:
                best, best_d = k, d
        return best

    def begin_drag(self, st: ViewState) -> None:
        g, scale, origin, _, _ = self._layout(st)
        self._frozen = (g, scale, origin)

    def end_drag(self) -> None:
        self._frozen = None

    def dragged_span(self, st: ViewState, k: int, dx: float, dy: float) -> float:
        """New span for dimension k after moving its handle by (dx, dy) screen pixels."""
        g, scale, _ = self._frozen
        d = self.DIRS[k]
        along = (dx * d[0] + dy * d[1]) / np.linalg.norm(d)
        length = (st.spans[k] / g) ** self.GAMMA + along / (scale * np.linalg.norm(d))
        length = max(0.05, length)
        return g * length ** (1.0 / self.GAMMA)

    def paint(self, p: QPainter, st: ViewState) -> None:
        _, _, _, _, pts = self._layout(st)
        small = not self.expanded
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        tiers = {"second": [], "conn": [], "main": []}
        for b in range(1, 16):
            for k in range(4):
                if not b >> k & 1:
                    q = b | 1 << k
                    if k == 3:
                        tiers["conn"].append((b, q))
                    elif b & 8:
                        tiers["second"].append((b, q))
                    else:
                        tiers["main"].append((b, q))
        # Faintest to boldest: T connectors, second cube, main cube (control edges drawn after).
        for name, alpha, width in (("conn", 0.10, 1.1), ("second", 0.24, 1.2), ("main", 0.45, 1.4)):
            col = QColor(INK)
            col.setAlphaF(alpha)
            p.setPen(QPen(col, width))
            for a, b in tiers[name]:
                p.drawLine(pts[a], pts[b])
        p.setPen(QPen(INK, 2.0 if not small else 1.6, Qt.SolidLine, Qt.RoundCap))
        for k in range(4):
            p.drawLine(pts[0], pts[1 << k])
        p.setPen(Qt.NoPen)
        p.setBrush(INK)
        for k in range(4):
            p.drawEllipse(pts[1 << k], 4.5 if not small else 3.0, 4.5 if not small else 3.0)
        p.drawEllipse(pts[0], 3.0, 3.0)
        font = QFont("Consolas", 10 if not small else 8)
        font.setBold(True)
        p.setFont(font)
        for k in range(4):
            d = _unit(self.DIRS[k])
            q = pts[1 << k]
            _text(p, QPointF(q.x() + d[0] * (16 if not small else 10), q.y() + d[1] * (16 if not small else 10)), DISPLAY_AXES[k], INK)
        if not small:
            _text(p, QPointF(pts[0].x() - 10, pts[0].y() + 12), "v", MUTED)
        p.restore()


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------

def _text(p: QPainter, pt: QPointF, s: str, color: QColor) -> None:
    fm = p.fontMetrics()
    w = fm.horizontalAdvance(s)
    h = fm.ascent()
    x, y = pt.x() - w / 2, pt.y() + h / 2 - 1
    path_pen = QPen(QColor(20, 23, 28, 230), 3.5)
    p.setPen(path_pen)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        p.drawText(QPointF(x + dx, y + dy), s)
    p.setPen(color)
    p.drawText(QPointF(x, y), s)


def _dist_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((p.x() - ax) * dx + (p.y() - ay) * dy) / L2))
    return math.hypot(p.x() - (ax + t * dx), p.y() - (ay + t * dy))


def _frame(axis):
    a = _unit(axis)
    helper = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _unit(np.cross(a, helper))
    v = np.cross(a, u)
    return a, u, v


def _cylinder(p0, p1, r, color, n=18):
    a, u, v = _frame(p1 - p0)
    gl.glColor4f(*color)
    gl.glBegin(gl.GL_QUAD_STRIP)
    for i in range(n + 1):
        ang = 2 * math.pi * i / n
        d = math.cos(ang) * u + math.sin(ang) * v
        gl.glNormal3d(*d)
        gl.glVertex3d(*(p0 + r * d))
        gl.glVertex3d(*(p1 + r * d))
    gl.glEnd()


def _cone(base, tip, r, color, n=24, wire=False):
    axis = tip - base
    height = float(np.linalg.norm(axis))
    a, u, v = _frame(axis)
    ring = [math.cos(2 * math.pi * i / n) * u + math.sin(2 * math.pi * i / n) * v for i in range(n)]
    if wire:
        # Hollow head: an outline cone, so a negative T part reads as "open".
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor4f(*color)
        gl.glLineWidth(1.6)
        gl.glBegin(gl.GL_LINE_LOOP)
        for d in ring:
            gl.glVertex3d(*(base + r * d))
        gl.glEnd()
        gl.glBegin(gl.GL_LINES)
        for d in ring[::3]:
            gl.glVertex3d(*(base + r * d))
            gl.glVertex3d(*tip)
        gl.glEnd()
        gl.glEnable(gl.GL_LIGHTING)
        return
    gl.glColor4f(*color)
    gl.glBegin(gl.GL_TRIANGLES)
    for i in range(n):
        d0, d1 = ring[i], ring[(i + 1) % n]
        n0 = _unit(d0 * height + a * r)
        n1 = _unit(d1 * height + a * r)
        gl.glNormal3d(*_unit(n0 + n1))
        gl.glVertex3d(*tip)
        gl.glNormal3d(*n0)
        gl.glVertex3d(*(base + r * d0))
        gl.glNormal3d(*n1)
        gl.glVertex3d(*(base + r * d1))
    gl.glEnd()
    gl.glBegin(gl.GL_TRIANGLE_FAN)
    gl.glNormal3d(*(-a))
    gl.glVertex3d(*base)
    for d in ring + ring[:1]:
        gl.glVertex3d(*(base + r * d))
    gl.glEnd()


def _sphere(c, r, color, n=12):
    gl.glColor4f(*color)
    for i in range(n):
        t0 = math.pi * i / n - math.pi / 2
        t1 = math.pi * (i + 1) / n - math.pi / 2
        gl.glBegin(gl.GL_QUAD_STRIP)
        for j in range(2 * n + 1):
            ph = 2 * math.pi * j / (2 * n)
            for t in (t0, t1):
                d = np.array([math.cos(t) * math.cos(ph), math.sin(t), math.cos(t) * math.sin(ph)])
                gl.glNormal3d(*d)
                gl.glVertex3d(*(c + r * d))
        gl.glEnd()
