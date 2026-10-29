"""Shader engine: the same renders as engine.Renderer, computed in one OpenGL fragment shader.

One fragment per W x H sample loops over the C slices, runs the shape's GLSL
inside_shape() (indicators.GLSL) with its own early exit, and packs the inside bits
into a GL_R32UI texture that is read back as the usual (H, W) uint32 array.

A GL context only works on the thread it belongs to, and renders come from both the
render worker and the GUI thread (export), so the context lives on its own thread
and render() hands each job to it. The frame is drawn in tiles that take ~50 ms
each, with cancel() checked between them; Windows resets the GPU driver if one
draw runs for more than ~2 s.
"""

from __future__ import annotations

import atexit
import queue
import sys
import threading
import time
from typing import Callable, Optional, Sequence

import numpy as np
from OpenGL import GL as gl
from PySide6.QtCore import QCoreApplication, QThread
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat

from .colormap import bit_positions
from .indicators import GLSL
from .state import MAX_C_RES, ViewState, W, H, C, T

VERTEX = """#version 330
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Sample positions and the order of the sums match engine.Renderer, so both engines
# agree except for rounding right on a shape's boundary.
FRAGMENT = """#version 330
uniform vec4 uBase;                 // center + T direction * u_t
uniform vec4 uCoefW, uCoefH, uCoefC;  // math step per unit of display axis W, H, C
uniform vec2 uGrid;                 // n_w, n_h
uniform int uNC;
uniform int uBitPos[%(max_c)d];
uniform int uPrec;
layout(location = 0) out uint outBits;
%(shape)s
void main() {
    vec2 ij = floor(gl_FragCoord.xy);  // texel row 0 = top row, as in engine.py
    float uw = (ij.x + 0.5) / uGrid.x - 0.5;
    float uh = 0.5 - (ij.y + 0.5) / uGrid.y;
    vec4 base = uBase + uCoefW * uw + uCoefH * uh;
    uint bits = 0u;
    for (int s = 0; s < uNC; s++) {
        float uc = uNC > 1 ? float(s) / float(uNC - 1) - 0.5 : 0.0;
        if (inside_shape(base + uCoefC * uc, uPrec)) bits |= 1u << uint(uBitPos[s]);
    }
    outBits = bits;
}
"""

TILE_SECONDS = 0.05      # aim for tiles this long...
MAX_TILE_SECONDS = 0.5   # ...and never longer than this, even if every point runs all iterations


def _log(msg: str) -> None:
    print(f"[whct shader] {msg}", file=sys.stderr)


class _GlThread(QThread):
    """Runs submitted functions with the GL context current."""

    def __init__(self, ctx: QOpenGLContext, surface: QOffscreenSurface):
        super().__init__()
        self.ctx = ctx
        self.surface = surface
        self.jobs: "queue.Queue" = queue.Queue()

    def run(self) -> None:
        if not self.ctx.makeCurrent(self.surface):
            self._drain(RuntimeError("could not make the OpenGL context current"))
            return
        while True:
            job = self.jobs.get()
            if job is None:
                break
            fn, args, box, done = job
            try:
                box["result"] = fn(*args)
            except BaseException as exc:
                box["error"] = exc
            done.set()
        self.ctx.doneCurrent()
        self.ctx.moveToThread(QCoreApplication.instance().thread())

    def _drain(self, exc: Exception) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            job[2]["error"] = exc
            job[3].set()


class GpuRenderer:
    """Same render() as engine.Renderer. Raises RuntimeError if OpenGL 3.3 is unavailable.

    Create it on the GUI thread (Qt requires that for the offscreen surface), after
    the QGuiApplication exists.
    """

    preview_points = 12_000_000

    def __init__(self):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
        self._surface = QOffscreenSurface()
        self._surface.setFormat(fmt)
        self._surface.create()
        ctx = QOpenGLContext()
        ctx.setFormat(fmt)
        if not ctx.create():
            raise RuntimeError("could not create an OpenGL context")
        got = ctx.format()
        if (got.majorVersion(), got.minorVersion()) < (3, 3):
            raise RuntimeError(f"OpenGL {got.majorVersion()}.{got.minorVersion()}; the shader engine needs 3.3")
        self._ctx = ctx
        self._thread = _GlThread(ctx, self._surface)
        ctx.moveToThread(self._thread)
        self._thread.start()
        self._lock = threading.Lock()
        self._programs = {}  # indicator -> (program, uniform locations), or None if it failed to compile
        self._fbo = self._tex = self._vao = None
        self._size = (0, 0)
        self._iters_per_second = None  # measured on the first render
        try:
            self.device_name, self._max_size = self._call(self._init_gl)
            for indicator in GLSL:  # up front, so supports() never waits behind a render
                self._call(self._program, indicator)
        except Exception:
            self.stop()
            raise
        atexit.register(self.stop)

    # ----- public -----
    def supports(self, indicator: str, res: Sequence[int]) -> bool:
        if indicator not in GLSL or int(res[C]) > MAX_C_RES:
            return False
        if int(res[W]) > self._max_size or int(res[H]) > self._max_size:
            return False
        return self._programs.get(indicator) is not None

    def render(
        self,
        st: ViewState,
        frame: int,
        res: Sequence[int],
        precision: int,
        cancel: Callable[[], bool] = lambda: False,
    ) -> Optional[np.ndarray]:
        """Return an (H, W) uint32 image of packed C bits, or None if cancelled."""
        with self._lock:
            return self._call(self._render, st, frame, res, precision, cancel)

    def stop(self) -> None:
        if self._thread.isRunning():
            self._thread.jobs.put(None)
            self._thread.wait()

    # ----- plumbing -----
    def _call(self, fn, *args):
        if not self._thread.isRunning():
            raise RuntimeError("the shader engine has stopped")
        box, done = {}, threading.Event()
        self._thread.jobs.put((fn, args, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["result"]

    # ----- everything below runs on the GL thread -----
    def _init_gl(self):
        name = gl.glGetString(gl.GL_RENDERER).decode(errors="replace")
        for junk in ("(R)", "(TM)", "(tm)"):
            name = name.replace(junk, "")
        name = " ".join(name.split())
        max_size = min(int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)), *(int(v) for v in gl.glGetIntegerv(gl.GL_MAX_VIEWPORT_DIMS)))
        self._vao = gl.glGenVertexArrays(1)
        self._fbo = gl.glGenFramebuffers(1)
        self._tex = gl.glGenTextures(1)
        return name, max_size

    def _program(self, indicator: str):
        if indicator in self._programs:
            return self._programs[indicator]
        try:
            prog = _link(VERTEX, FRAGMENT % {"max_c": MAX_C_RES, "shape": GLSL[indicator]})
            names = ("uBase", "uCoefW", "uCoefH", "uCoefC", "uGrid", "uNC", "uBitPos", "uPrec")
            entry = (prog, {n: gl.glGetUniformLocation(prog, n) for n in names})
        except Exception as exc:
            _log(f"{indicator}: {exc}; using PyTorch for this shape")
            entry = None
        self._programs[indicator] = entry
        return entry

    def _ensure_target(self, n_w: int, n_h: int) -> None:
        """Grow-only R32UI render target; a frame uses its bottom-left n_w x n_h corner."""
        if n_w <= self._size[0] and n_h <= self._size[1]:
            return
        w, h = max(n_w, self._size[0]), max(n_h, self._size[1])
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32UI, w, h, 0, gl.GL_RED_INTEGER, gl.GL_UNSIGNED_INT, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._tex, 0)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"render target incomplete (status 0x{int(status):x})")
        self._size = (w, h)

    def _measure_speed(self) -> float:
        """Iterations per second when every point runs all of them (c = z0 = 0 never escapes)."""
        entry = self._program("julia_mandelbrot")
        if entry is None:
            return 1e8  # a slow GPU, so the tiles stay short
        prog, loc = entry
        n, n_c, prec = 64, MAX_C_RES, 200
        self._ensure_target(n, n)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glBindVertexArray(self._vao)
        gl.glUseProgram(prog)
        for name in ("uBase", "uCoefW", "uCoefH", "uCoefC"):
            gl.glUniform4f(loc[name], 0.0, 0.0, 0.0, 0.0)
        gl.glUniform2f(loc["uGrid"], float(n), float(n))
        gl.glUniform1i(loc["uNC"], n_c)
        gl.glUniform1iv(loc["uBitPos"], MAX_C_RES, np.arange(MAX_C_RES, dtype=np.int32))
        gl.glUniform1i(loc["uPrec"], prec)
        gl.glViewport(0, 0, n, n)
        best = 0.0
        for _ in range(3):  # the first draw includes driver warm-up
            t0 = time.perf_counter()
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            gl.glFinish()
            best = max(best, n * n * n_c * prec / max(1e-4, time.perf_counter() - t0))
        return best

    def _render(self, st, frame, res, precision, cancel):
        n_w, n_h, n_c = int(res[W]), int(res[H]), int(res[C])
        entry = self._program(st.indicator)
        if entry is None:
            raise RuntimeError(f"no shader for {st.indicator}")
        prog, loc = entry
        coef = st.R * st.spans[None, :]  # coef[j, k]: math axis j per unit of display axis k
        base = st.center + coef[:, T] * st.u_t(frame)
        bitpos = np.zeros(MAX_C_RES, dtype=np.int32)
        bitpos[:n_c] = bit_positions(n_c)

        self._ensure_target(n_w, n_h)
        if self._iters_per_second is None:
            self._iters_per_second = self._measure_speed()
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glBindVertexArray(self._vao)
        gl.glUseProgram(prog)
        gl.glUniform4f(loc["uBase"], *(float(v) for v in base))
        for name, k in (("uCoefW", W), ("uCoefH", H), ("uCoefC", C)):
            gl.glUniform4f(loc[name], *(float(v) for v in coef[:, k]))
        gl.glUniform2f(loc["uGrid"], float(n_w), float(n_h))
        gl.glUniform1i(loc["uNC"], n_c)
        gl.glUniform1iv(loc["uBitPos"], MAX_C_RES, bitpos)
        gl.glUniform1i(loc["uPrec"], int(precision))

        # Tiles of W x H cells. A cell costs at most n_c * precision iterations (a
        # Julia-Mandelbrot point that never escapes); the tile size starts from that
        # worst case, then follows the measured speed, but never exceeds the worst case
        # for MAX_TILE_SECONDS.
        worst = n_c * max(1, int(precision)) / self._iters_per_second  # seconds per cell
        cap = max(1, int(MAX_TILE_SECONDS / worst))
        cells = max(1, int(TILE_SECONDS / worst))
        y = 0
        while y < n_h:
            rows = min(n_h - y, max(1, cells // n_w))
            x = 0
            while x < n_w:
                if cancel():
                    return None
                cols = min(n_w - x, n_w if cells >= n_w else cells)
                t0 = time.perf_counter()
                gl.glViewport(x, y, cols, rows)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
                gl.glFinish()
                dt = max(1e-4, time.perf_counter() - t0)
                cells = max(1, min(cap, cells * 8, int(cols * rows / dt * TILE_SECONDS)))
                x += cols
            y += rows

        out = np.empty((n_h, n_w), dtype=np.uint32)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
        gl.glReadPixels(0, 0, n_w, n_h, gl.GL_RED_INTEGER, gl.GL_UNSIGNED_INT, out)
        gl.glUseProgram(0)
        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return out


def _compile(kind, src: str):
    sh = gl.glCreateShader(kind)
    gl.glShaderSource(sh, src)
    gl.glCompileShader(sh)
    if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
        log = gl.glGetShaderInfoLog(sh)
        gl.glDeleteShader(sh)
        raise RuntimeError("shader compile failed: " + (log.decode(errors="replace") if isinstance(log, bytes) else str(log)))
    return sh


def _link(vert: str, frag: str):
    vs, fs = _compile(gl.GL_VERTEX_SHADER, vert), _compile(gl.GL_FRAGMENT_SHADER, frag)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        log = gl.glGetProgramInfoLog(prog)
        gl.glDeleteProgram(prog)
        raise RuntimeError("shader link failed: " + (log.decode(errors="replace") if isinstance(log, bytes) else str(log)))
    return prog
