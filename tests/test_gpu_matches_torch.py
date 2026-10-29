"""The shader engine must give the same packed bits as the PyTorch engine.

    python -m pytest tests

float32 on the GPU (fused multiply-add, approximate division and sin/cos) rounds a
little differently from PyTorch, so a point lying right on a shape's boundary can
flip. The tests allow a small fraction of differing bits instead of exact equality.
Views are nudged off the home center so grid points don't land exactly on a
boundary (|x| + |z| = 0.8 for the diamond, sin x = -sin z for the sine field), where
the result is a coin toss.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whct.colormap import bit_positions  # noqa: E402
from whct.engine import Renderer  # noqa: E402
from whct.indicators import GLSL, INDICATORS  # noqa: E402
from whct.state import ViewState, W, H, C, T  # noqa: E402

MAX_BIT_MISMATCH = 1e-3  # fraction of all W x H x C samples
NUDGE = np.array([3.1e-4, 1.7e-4, 2.3e-4, 1.3e-4])


@pytest.fixture(scope="module")
def engines():
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841
    from whct.gpu import GpuRenderer
    try:
        gpu = GpuRenderer()
    except RuntimeError as exc:
        pytest.skip(f"no OpenGL 3.3: {exc}")
    yield Renderer(), gpu
    gpu.stop()


def make_view(shape, n_c=24, rotate=(), center=None, spans=None, res=(160, 120)):
    st = ViewState(indicator=shape)
    st.res = [res[0], res[1], n_c, 31]
    st.reset_view()
    st.center = st.center + NUDGE * st.spans.max()
    if center is not None:
        st.center = np.array(center, dtype=float)
    if spans is not None:
        st.spans = np.array(spans, dtype=float)
    for i, k, theta in rotate:
        st.rotate_plane(i, k, theta)
    return st


def bit_mismatch(a, b, n_c):
    diff = np.bitwise_xor(a, b)
    flipped = np.unpackbits(diff.view(np.uint8)).sum()
    return flipped / (a.size * n_c)


def compare(engines, st, frame=None, precision=None):
    torch_r, gpu = engines
    frame = st.frame if frame is None else frame
    precision = st.precision if precision is None else precision
    ref = torch_r.render(st, frame, st.res, precision)
    got = gpu.render(st, frame, st.res, precision)
    assert got.shape == ref.shape == (st.res[H], st.res[W])
    assert got.dtype == np.uint32
    mismatch = bit_mismatch(ref, got, st.res[C])
    assert mismatch <= MAX_BIT_MISMATCH, f"{mismatch:.4%} of bits differ"
    return ref, got


def test_every_shape_has_glsl():
    assert set(GLSL) == set(INDICATORS)


@pytest.mark.parametrize("shape", list(INDICATORS))
@pytest.mark.parametrize("n_c", [1, 7, 24])
def test_home_view(engines, shape, n_c):
    ref, _ = compare(engines, make_view(shape, n_c))
    assert ref.any(), "view shows nothing; the test would prove little"


@pytest.mark.parametrize("shape", list(INDICATORS))
@pytest.mark.parametrize("frame", [0, 7, 30])
def test_rotated_view_and_frames(engines, shape, frame):
    st = make_view(shape, 7, rotate=[(W, C, 0.4), (H, T, 0.7), (C, T, -0.3)])
    compare(engines, st, frame=frame)


@pytest.mark.parametrize("precision", [1, 30, 300])
def test_julia_precision(engines, precision):
    compare(engines, make_view("julia_mandelbrot"), precision=precision)


def julia_float64(st, precision):
    """Reference Julia-Mandelbrot bits in float64, as engine.Renderer samples them."""
    n_w, n_h, n_c = st.res[W], st.res[H], st.res[C]
    u_w = ((np.arange(n_w) + 0.5) / n_w - 0.5)[None, :, None]
    u_h = (0.5 - (np.arange(n_h) + 0.5) / n_h)[:, None, None]
    u_c = (np.arange(n_c) / (n_c - 1) - 0.5 if n_c > 1 else np.zeros(1))[None, None, :]
    coef = st.R * st.spans[None, :]
    x, y, z, w = (st.center[j] + coef[j, T] * st.u_t() + coef[j, W] * u_w + coef[j, H] * u_h + coef[j, C] * u_c
                  for j in range(4))
    zr, zi = y, w
    inside = np.ones(zr.shape, dtype=bool)
    for _ in range(precision):
        nzr, nzi = zr * zr - zi * zi + x, 2.0 * zr * zi + z
        inside &= nzr * nzr + nzi * nzi <= 4.0
        zr, zi = np.where(inside, nzr, zr), np.where(inside, nzi, zi)
    return (inside * (2 ** bit_positions(n_c))).sum(axis=-1).astype(np.uint32)


def test_julia_zoomed_boundary(engines):
    # Zoomed in on the boundary, float32 itself is the limit: both engines miss the
    # float64 answer by ~0.4% of bits, differently. The shader must be no worse.
    torch_r, gpu = engines
    st = make_view("julia_mandelbrot", 24, center=(-0.745, 0.0, 0.11, 0.0), spans=(0.01, 0.0075, 0.01, 0.01))
    exact = julia_float64(st, 200)
    err_torch = bit_mismatch(torch_r.render(st, st.frame, st.res, 200), exact, 24)
    err_gpu = bit_mismatch(gpu.render(st, st.frame, st.res, 200), exact, 24)
    assert err_gpu <= 1.25 * err_torch + MAX_BIT_MISMATCH, f"shader {err_gpu:.4%}, PyTorch {err_torch:.4%}"


def test_julia_wide_rotated(engines):
    st = make_view("julia_mandelbrot", 24, rotate=[(W, T, 1.1), (H, C, 0.5)], res=(320, 240))
    compare(engines, st, frame=22, precision=50)


def test_row_zero_is_top(engines):
    # Shift the hypersphere up: only the top half of the image may light up.
    st = make_view("hypersphere", 1, center=(0.0, 0.0, -0.6, 0.0))
    _, got = compare(engines, st)
    assert got[: st.res[H] // 2].any() and not got[st.res[H] // 2 + 1 :].any()


def test_cancel_returns_none(engines):
    _, gpu = engines
    st = make_view("julia_mandelbrot")
    assert gpu.render(st, st.frame, st.res, 30, cancel=lambda: True) is None


def test_odd_sizes_after_larger_frame(engines):
    # The render target only grows; a smaller frame afterwards must use the right corner.
    compare(engines, make_view("4D_torus", 24, res=(301, 203)))
    compare(engines, make_view("4D_torus", 5, res=(37, 211)))
