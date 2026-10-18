"""View state: where the four physical display dimensions sit inside the math space.

Physical display dimensions (uppercase): W screen width, H screen height,
C color, T time. Math dimensions (lowercase): x, y, z, w.

A display sample at normalized position u = (uW, uH, uC, uT), each in [-1/2, 1/2],
shows the math point

    p = center + R @ (spans * u)

- center: the math point at the middle of the view (panning changes it)
- spans:  how much math distance each physical dimension covers (scaling changes it)
- R:      a 4D rotation. Row j holds math axis j's parts along W, H, C, T, which is
          what the rotation widget draws. Column k is the math direction of display
          axis k. Rotating changes it.

Rotation and widget zooms pivot on the middle of the visible W, H and C at the
current T, so the frame you are looking at stays put.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

DISPLAY_AXES = ("W", "H", "C", "T")
MATH_AXES = ("x", "y", "z", "w")
W, H, C, T = range(4)

# Default alignment from the sketch: x on W, z on H, y on C, w on T.
DEFAULT_R = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # x
        [0.0, 0.0, 1.0, 0.0],  # y
        [0.0, 1.0, 0.0, 0.0],  # z
        [0.0, 0.0, 0.0, 1.0],  # w
    ]
)

# Per-shape home view: center (x, y, z, w) and spans (W, H, C, T).
_GENERIC_HOME = ((0.0, 0.0, 0.0, 0.0), (2.0, 2.0, 2.0, 2.0))
HOME_VIEWS = {
    "julia_mandelbrot": ((-0.75, 0.0, 0.0, 0.0), (3.0, 3.0, 2.0, 2.0)),
    "4D_sine_field": ((0.0, 0.0, 0.0, 0.0), (8.0, 8.0, 6.0, 6.0)),
    "4D_gyroid_shell": ((0.0, 0.0, 0.0, 0.0), (8.0, 8.0, 6.0, 6.0)),
}

MAX_C_RES = 24  # color slices are packed into a 24-bit color


def plane_rotation(i: int, k: int, theta: float) -> np.ndarray:
    """4x4 rotation in display plane (i, k). Positive theta turns axis i toward axis k."""
    g = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    g[i, i] = c
    g[k, k] = c
    g[i, k] = s
    g[k, i] = -s
    return g


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v.copy()


@dataclass
class ViewState:
    indicator: str = "julia_mandelbrot"
    center: np.ndarray = field(default_factory=lambda: np.array(HOME_VIEWS["julia_mandelbrot"][0]))
    spans: np.ndarray = field(default_factory=lambda: np.array(HOME_VIEWS["julia_mandelbrot"][1]))
    R: np.ndarray = field(default_factory=lambda: DEFAULT_R.copy())
    res: list = field(default_factory=lambda: [640, 480, 24, 31])  # samples along W, H, C, T
    frame: int = 15  # middle frame: T = 0
    precision: int = 30

    # ----- copies and identity -----
    def copy(self) -> "ViewState":
        return copy.deepcopy(self)

    def nav_snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.center.copy(), self.spans.copy(), self.R.copy()

    def restore_nav(self, snap) -> None:
        self.center, self.spans, self.R = (a.copy() for a in snap)

    def signature(self, res: Optional[Sequence[int]] = None, precision: Optional[int] = None) -> tuple:
        """Everything that changes a rendered frame except the frame index and color mapping."""
        r = tuple(res) if res is not None else tuple(self.res)
        return (
            self.indicator,
            int(precision if precision is not None else self.precision),
            tuple(np.round(self.center, 12)),
            tuple(np.round(self.spans, 12)),
            tuple(np.round(self.R.ravel(), 12)),
            r,
        )

    # ----- sample positions -----
    def u_t(self, frame: Optional[int] = None) -> float:
        f = self.frame if frame is None else frame
        n = self.res[T]
        return 0.0 if n <= 1 else f / (n - 1) - 0.5

    def t_value(self, frame: Optional[int] = None) -> float:
        """Position of a frame along the display T axis, in math units from the center."""
        return self.u_t(frame) * self.spans[T]

    def pivot_u(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, self.u_t()])

    def point(self, u: np.ndarray) -> np.ndarray:
        return self.center + self.R @ (self.spans * u)

    # ----- navigation -----
    def pan(self, du: np.ndarray) -> None:
        """Move the view by du in normalized display units."""
        self.center = self.center + self.R @ (self.spans * du)

    def scale(self, factors: np.ndarray, anchor_u: np.ndarray) -> None:
        """Multiply spans by factors, keeping the math point at anchor_u fixed on the display."""
        new = np.clip(self.spans * factors, 1e-12, 1e12)
        self.center = self.center + self.R @ ((self.spans - new) * anchor_u)
        self.spans = new

    def set_span(self, k: int, value: float, anchor_u: Optional[np.ndarray] = None) -> None:
        f = np.ones(4)
        f[k] = max(1e-12, value) / self.spans[k]
        self.scale(f, self.pivot_u() if anchor_u is None else anchor_u)

    def _set_R(self, new_R: np.ndarray, first: int = 0) -> None:
        pivot_u = self.pivot_u()
        pivot = self.point(pivot_u)
        self.R = new_R
        self._reortho(first)
        self.center = pivot - self.R @ (self.spans * pivot_u)

    def rotate_plane(self, i: int, k: int, theta: float) -> None:
        """Turn the view in display plane (i, k) about the pivot. Math arrows move from i toward k."""
        self._set_R(self.R @ plane_rotation(i, k, theta))

    def set_component(self, j: int, c: int, a: float) -> None:
        """Set math axis j's part along display axis c to exactly a.

        The other three parts of that axis rescale to keep unit length, and the
        smallest rotation that makes this happen is applied to every axis.
        """
        a = float(np.clip(a, -1.0, 1.0))
        old = self.R[j].copy()
        target = old.copy()
        target[c] = 0.0
        rest = np.linalg.norm(target)
        r = np.sqrt(max(0.0, 1.0 - a * a))
        if rest > 1e-9:
            target *= r / rest
        else:
            # The other parts are all zero: put the remainder on the display axis
            # least used by the other math axes.
            coverage = [
                sum(self.R[q, i] ** 2 for q in range(4) if q != j) if i != c else np.inf
                for i in range(4)
            ]
            target[int(np.argmin(coverage))] = r
        target[c] = a
        R = self._minimal_rotation(old, target)
        R[j] = target
        self._set_R(R, first=j)
        self.R[j, c] = a
        self._reortho(first=j)

    def _minimal_rotation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return R with the smallest 4D rotation taking unit a to unit b applied to every row."""
        a, b = _unit(a), _unit(b)
        cos_t = float(np.clip(a @ b, -1.0, 1.0))
        if cos_t > 1 - 1e-12:
            return self.R.copy()
        v = b - cos_t * a
        n = np.linalg.norm(v)
        if n < 1e-12:
            return self.R.copy()
        v /= n
        theta = np.arccos(cos_t)
        sn, cs = np.sin(theta), np.cos(theta) - 1.0
        rows = []
        for m in self.R:
            pa, pv = m @ a, m @ v
            rows.append(m + sn * (pa * v - pv * a) + cs * (pa * a + pv * v))
        return np.array(rows)

    def _reortho(self, first: int = 0) -> None:
        order = [first] + [i for i in range(4) if i != first]
        done = []
        for i in order:
            r = self.R[i].copy()
            for j in done:
                r -= (r @ self.R[j]) * self.R[j]
            self.R[i] = _unit(r)
            done.append(i)

    def reset_view(self) -> None:
        """Home view for the current shape. H span follows the W:H sample ratio so pixels start square."""
        center, spans = HOME_VIEWS.get(self.indicator, _GENERIC_HOME)
        self.center = np.array(center, dtype=float)
        self.spans = np.array(spans, dtype=float)
        self.spans[H] = self.spans[W] * self.res[H] / max(1, self.res[W])
        self.R = DEFAULT_R.copy()

    def reset_rotation(self) -> None:
        """Back to the default alignment, about the pivot so the view doesn't jump."""
        self._set_R(DEFAULT_R.copy())

    def reset_spans(self) -> None:
        """Home spans for this shape (H follows the W:H sample ratio), zooming about the pivot."""
        spans = np.array(HOME_VIEWS.get(self.indicator, _GENERIC_HOME)[1], dtype=float)
        spans[H] = spans[W] * self.res[H] / max(1, self.res[W])
        self.scale(spans / self.spans, self.pivot_u())

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "center": [float(v) for v in self.center],
            "spans": [float(v) for v in self.spans],
            "R": [[float(v) for v in row] for row in self.R],
            "res": [int(v) for v in self.res],
            "frame": int(self.frame),
            "precision": int(self.precision),
        }

    def load_dict(self, d: dict) -> None:
        self.indicator = d["indicator"]
        self.center = np.array(d["center"], dtype=float)
        self.spans = np.array(d["spans"], dtype=float)
        self.R = np.array(d["R"], dtype=float)
        self._reortho(0)
        self.res = [int(v) for v in d["res"]]
        self.precision = int(d["precision"])
        self.frame = int(np.clip(d["frame"], 0, self.res[T] - 1))

    def set_t_res(self, n: int) -> None:
        n = max(1, int(n))
        old = self.res[T]
        if old > 1 and n > 1:
            self.frame = int(round(self.frame * (n - 1) / (old - 1)))
        self.res[T] = n
        self.frame = int(np.clip(self.frame, 0, n - 1))
