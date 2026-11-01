"""Render a video that moves steadily from one saved view (.whct) to another.

    python -m whct.tween before.whct after.whct out.mp4 [--seconds 10] [--ease 2] [--hold 1] [--fps 30] [--size 1920 1080]

Each video frame is one T slice. Between the two views:
- the rotation turns at a constant rate along the shortest 4D path (a geodesic),
- the middle of the visible slice (the view center moved to the saved frame) moves in a straight line,
- the C span changes linearly; W and H spans too, with H following the output aspect
  ratio so pixels stay square.
Shape, precision, C slices and color mapping come from the first file.

Needs ffmpeg on PATH for an H.264 MP4 that PowerPoint and browsers play; without it,
falls back to OpenCV's MPEG-4 writer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time

import numpy as np

from .colormap import ColorMapper
from .state import ViewState, W, H, C, T


def load_view(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    st = ViewState()
    st.load_dict(data["view"])
    return st, data


def slice_center(st: ViewState) -> np.ndarray:
    """The math point in the middle of the visible slice (W, H, C middle, current frame)."""
    return st.point(st.pivot_u())


def rotation_path(R0: np.ndarray, R1: np.ndarray):
    """t -> R0 @ (R0^T R1)^t: constant-speed rotation from R0 (t=0) to R1 (t=1)."""
    Q = R0.T @ R1
    vals, vecs = np.linalg.eig(Q)
    angles = np.angle(vals)
    if np.any(np.abs(np.abs(angles) - np.pi) < 1e-6):
        raise ValueError("the two rotations differ by a half turn in some plane; the shortest path is ambiguous")
    inv = np.linalg.inv(vecs)

    def at(t: float) -> np.ndarray:
        Qt = (vecs @ np.diag(np.exp(1j * angles * t)) @ inv).real
        u, _, vt = np.linalg.svd(R0 @ Qt)  # snap back to an exact rotation
        return u @ vt

    return at


def eased(t: float, a: float) -> float:
    """Progress 0..1 at time t (0..1): speeds up over the first a, slows down over the last a.

    The speed ramps smoothly (a half cosine) from 0 to full and back, and is constant
    in between, so the motion never jerks.
    """
    if a <= 0:
        return t
    a = min(a, 0.5)
    v = 1.0 / (1.0 - a)  # full speed, so the whole move still covers exactly 0..1

    def ramp(x):  # distance covered x into a ramp
        return v * (x / 2 - a / (2 * np.pi) * np.sin(np.pi * x / a))

    if t < a:
        return ramp(t)
    if t > 1 - a:
        return 1.0 - ramp(1 - t)
    return ramp(a) + v * (t - a)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("output")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--ease", type=float, default=0.0, help="seconds to speed up at the start and slow down at the end")
    ap.add_argument("--hold", type=float, default=0.0, help="seconds to hold the first and last frames")
    ap.add_argument("--fps", type=int, default=None, help="default: the first file's playback fps")
    ap.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="default: the first file's W x H (rounded to even numbers)")
    ap.add_argument("--engine", choices=("shader", "torch"), default="shader")
    args = ap.parse_args(argv)

    st0, data0 = load_view(args.start)
    st1, _ = load_view(args.end)
    if st0.indicator != st1.indicator:
        sys.exit(f"the two views show different shapes ({st0.indicator}, {st1.indicator})")
    fps = args.fps or int(data0.get("settings", {}).get("fps", 30))
    n_w, n_h = args.size or (st0.res[W], st0.res[H])
    n_w, n_h = n_w + n_w % 2, n_h + n_h % 2  # H.264 needs even sizes
    n_frames = max(2, int(round(args.seconds * fps)))
    n_hold = int(round(args.hold * fps))
    precision, n_c = st0.precision, st0.res[C]

    mapper = ColorMapper()
    mapping = data0.get("color_mapping")
    if mapping in mapper.names:
        mapper.set_mapping(mapping)

    if args.engine == "shader":
        from PySide6.QtGui import QGuiApplication
        app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841
        from .gpu import GpuRenderer
        renderer = GpuRenderer()
    else:
        from .engine import Renderer
        renderer = Renderer()

    p0, p1 = slice_center(st0), slice_center(st1)
    rot = rotation_path(st0.R, st1.R)
    s0, s1 = st0.spans.copy(), st1.spans.copy()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{n_w}x{n_h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.output],
            stdin=subprocess.PIPE,
        )
        write = lambda rgb: proc.stdin.write(rgb.tobytes())  # noqa: E731
    else:
        import cv2
        print("ffmpeg not found; writing MPEG-4 with OpenCV (PowerPoint may not play it)")
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (n_w, n_h))
        write = lambda rgb: writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))  # noqa: E731

    t_start = time.perf_counter()
    try:
        for i in range(n_frames):
            t = eased(i / (n_frames - 1), args.ease / args.seconds)
            st = ViewState(indicator=st0.indicator)
            st.R = rot(t)
            st.spans = (1 - t) * s0 + t * s1
            st.spans[H] = st.spans[W] * n_h / n_w  # square pixels at the output size
            st.res = [n_w, n_h, n_c, 1]  # one T slice, centered on the interpolated point
            st.frame = 0
            st.center = (1 - t) * p0 + t * p1
            rgb = mapper.to_rgb(renderer.render(st, 0, st.res, precision))
            for _ in range(1 + (n_hold if i in (0, n_frames - 1) else 0)):
                write(rgb)
            done = time.perf_counter() - t_start
            print(f"\rframe {i + 1}/{n_frames}  {done:.0f} s elapsed", end="", flush=True)
    finally:
        if ffmpeg:
            proc.stdin.close()
            proc.wait()
        else:
            writer.release()
        if hasattr(renderer, "stop"):
            renderer.stop()
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
