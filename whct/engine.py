"""Rendering: sample the 4D view, run the indicator, pack the C column into bits.

Rendering runs on a background thread so the window stays responsive. A newer
request cancels an older one between row chunks. Finished full-quality frames are
kept in a size-limited cache so playback can loop from memory.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections import OrderedDict
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PySide6.QtCore import QObject, Signal

from .colormap import bit_positions
from .indicators import INDICATORS
from .state import ViewState, W, H, C, T


class Renderer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        # Points evaluated per chunk; bounds memory use for large W x H x C.
        self.chunk_points = 16_000_000 if self.device.type == "cuda" else 2_000_000
        # Target size of a dynamic-downscale preview, in sampled points.
        self.preview_points = 12_000_000 if self.device.type == "cuda" else 1_200_000
        self.lock = threading.Lock()

    def render(
        self,
        st: ViewState,
        frame: int,
        res: Sequence[int],
        precision: int,
        cancel: Callable[[], bool] = lambda: False,
    ) -> Optional[np.ndarray]:
        """Return an (H, W) uint32 image of packed C bits, or None if cancelled."""
        with self.lock:
            return self._render(st, frame, res, precision, cancel)

    def _render(self, st, frame, res, precision, cancel):
        dev = self.device
        n_w, n_h, n_c = int(res[W]), int(res[H]), int(res[C])
        f32 = torch.float32
        u_w = (torch.arange(n_w, device=dev, dtype=f32) + 0.5) / n_w - 0.5
        u_h = 0.5 - (torch.arange(n_h, device=dev, dtype=f32) + 0.5) / n_h
        if n_c > 1:
            u_c = torch.arange(n_c, device=dev, dtype=f32) / (n_c - 1) - 0.5
        else:
            u_c = torch.zeros(1, device=dev, dtype=f32)
        u_t = st.u_t(frame)
        weights = torch.tensor(2 ** bit_positions(n_c), device=dev, dtype=torch.int32)
        coef = st.R * st.spans[None, :]  # coef[j, k]: math axis j per unit of display axis k
        fn = INDICATORS[st.indicator]

        out = np.empty((n_h, n_w), dtype=np.uint32)
        rows = max(1, self.chunk_points // max(1, n_w * n_c))
        with torch.no_grad():
            for r0 in range(0, n_h, rows):
                if cancel():
                    return None
                uh = u_h[r0 : r0 + rows]
                coords = []
                for j in range(4):
                    base = float(st.center[j] + coef[j, T] * u_t)
                    coords.append(
                        base
                        + float(coef[j, W]) * u_w[None, :, None]
                        + float(coef[j, H]) * uh[:, None, None]
                        + float(coef[j, C]) * u_c[None, None, :]
                    )
                mask = fn(*coords, precision)
                packed = (mask.to(torch.int32) * weights).sum(dim=-1, dtype=torch.int32)
                out[r0 : r0 + rows] = packed.cpu().numpy().astype(np.uint32)
        return out


class EngineSwitch:
    """Sends each render to the shader engine (gpu.py) when it can take it, else to PyTorch.

    If the shader engine fails at runtime it is logged and switched off for the session.
    """

    def __init__(self, torch_renderer: Renderer, gpu=None):
        self.torch = torch_renderer
        self.gpu = gpu
        self.use_shader = True

    def engine_for(self, indicator: str, res: Sequence[int]):
        if self.use_shader and self.gpu is not None and self.gpu.supports(indicator, res):
            return self.gpu
        return self.torch

    def label(self, indicator: str, res: Sequence[int]) -> str:
        if self.engine_for(indicator, res) is self.gpu:
            return f"Shader · {self.gpu.device_name}"
        return f"PyTorch · {self.torch.device_name}"

    def render(self, st, frame, res, precision, cancel=lambda: False):
        engine = self.engine_for(st.indicator, res)
        if engine is self.gpu:
            try:
                return engine.render(st, frame, res, precision, cancel)
            except Exception:
                traceback.print_exc()
                print("[whct shader] failed; using PyTorch from now on", file=sys.stderr)
                self.gpu = None
        return self.torch.render(st, frame, res, precision, cancel)

    def stop(self) -> None:
        if self.gpu is not None:
            self.gpu.stop()


class FrameCache:
    """Full-quality frames for the current view only, limited to a memory budget.

    Frames are stored as packed bits (4 bytes per pixel), so changing the color
    mapping does not invalidate them. Any change to the view clears the cache.
    """

    def __init__(self, budget_mb: int = 1024):
        self.budget = budget_mb * 1024 * 1024
        self.sig = None
        self.frames: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.nbytes = 0

    def set_budget(self, budget_mb: int) -> None:
        self.budget = max(16, int(budget_mb)) * 1024 * 1024
        self._evict()

    def clear(self) -> None:
        self.frames.clear()
        self.nbytes = 0

    def get(self, sig, frame: int) -> Optional[np.ndarray]:
        if sig != self.sig or frame not in self.frames:
            return None
        self.frames.move_to_end(frame)
        return self.frames[frame]

    def put(self, sig, frame: int, image: np.ndarray) -> None:
        if sig != self.sig:
            self.clear()
            self.sig = sig
        if image.nbytes > self.budget:
            return
        if frame in self.frames:
            self.nbytes -= self.frames.pop(frame).nbytes
        self.frames[frame] = image
        self.nbytes += image.nbytes
        self._evict()

    def _evict(self) -> None:
        while self.nbytes > self.budget and self.frames:
            _, old = self.frames.popitem(last=False)
            self.nbytes -= old.nbytes


class RenderWorker(QObject):
    """Renders the most recent request on a background thread.

    Only one request waits at a time (the newest). A running full-quality render is
    cancelled by any newer request; a running preview always finishes, so something
    shows up even on a slow machine.
    """

    finished = Signal(object)

    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer
        self._cond = threading.Condition()
        self._pending = None
        self._stop = False
        self.busy = False
        self._thread = threading.Thread(target=self._run, name="whct-render", daemon=True)
        self._thread.start()

    def submit(self, job: dict) -> None:
        with self._cond:
            self._pending = job
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job, self._pending = self._pending, None
                self.busy = True
            t0 = time.perf_counter()
            try:
                image = self.renderer.render(
                    job["state"], job["frame"], job["res"], job["precision"],
                    cancel=lambda: self._stop or (self._pending is not None and job["kind"] == "full"),
                )
                error = None
            except Exception as exc:  # report to the window instead of killing the thread
                image, error = None, exc
            self.busy = False
            if image is None and error is None:
                continue
            self.finished.emit({**job, "image": image, "error": error, "ms": (time.perf_counter() - t0) * 1000})
