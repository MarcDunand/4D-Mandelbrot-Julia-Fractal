"""Time a full-quality render with each engine.

    python tests/bench_render.py                  # both engines, default view
    python tests/bench_render.py --engine torch   # only the PyTorch engine
    python tests/bench_render.py --res 640 480 24 --precision 30 100 --repeat 3

Prints the median time per frame and the speedup of the shader over PyTorch.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whct.state import ViewState  # noqa: E402


def time_render(renderer, st, precision, repeat):
    renderer.render(st, st.frame, st.res, precision)  # warm-up (shader compile, allocations)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        renderer.render(st, st.frame, st.res, precision)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("both", "torch", "shader"), default="both")
    ap.add_argument("--shape", default="julia_mandelbrot")
    ap.add_argument("--res", type=int, nargs=3, default=(640, 480, 24), metavar=("W", "H", "C"))
    ap.add_argument("--precision", type=int, nargs="+", default=(30,))
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    st = ViewState(indicator=args.shape)
    st.res[:3] = list(args.res)
    st.reset_view()

    engines = {}
    if args.engine in ("both", "torch"):
        from whct.engine import Renderer
        r = Renderer()
        engines["torch"] = (r, f"PyTorch on {r.device_name}")
    if args.engine in ("both", "shader"):
        from PySide6.QtGui import QGuiApplication
        app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841
        from whct.gpu import GpuRenderer
        g = GpuRenderer()
        engines["shader"] = (g, f"Shader on {g.device_name}")

    w, h, c = args.res
    print(f"{args.shape}  {w}x{h}x{c}  ({w * h * c / 1e6:.1f} M points)  median of {args.repeat}")
    for prec in args.precision:
        results = {}
        for key, (renderer, label) in engines.items():
            results[key] = time_render(renderer, st, prec, args.repeat)
            print(f"  precision {prec:>6}  {label:<40} {results[key] * 1000:9.1f} ms")
        if len(results) == 2:
            print(f"  precision {prec:>6}  speedup {results['torch'] / results['shader']:.1f}x")
    for renderer, _ in engines.values():
        if hasattr(renderer, "stop"):
            renderer.stop()


if __name__ == "__main__":
    main()
