# WHCT: a 4D Julia–Mandelbrot viewer

WHCT lets you see and navigate a four-dimensional shape, mainly the 4D Julia–Mandelbrot set, on an ordinary screen. Nothing is lost along the way: every point of the 4D region you are looking at is represented somewhere in the picture.

The name comes from the four physical things the viewer uses to show four dimensions:

| | Physical dimension | How you see it |
|---|---|---|
| **W** | screen width | left to right |
| **H** | screen height | bottom to top |
| **C** | screen color | the color of each pixel |
| **T** | time | the frames of an animation |

These four physical dimensions hold the four **mathematical** dimensions of the shape, written **x, y, z, w**. Keep the cases apart: uppercase **W** is screen width; lowercase **w** is a math axis. Navigating in WHCT means changing how the math dimensions sit inside the physical ones: sliding them (pan), stretching them (scale) and turning them (rotate).

Older files and the research paper use earlier names for the same idea (XYTC, WHCS).

---

## Contents

- [How the picture works](#how-the-picture-works)
- [Quick start (Windows)](#quick-start-windows)
- [Using the viewer](#using-the-viewer)
  - [Moving around on the image](#moving-around-on-the-image)
  - [The scale widget (tesseract)](#the-scale-widget-tesseract)
  - [The rotation widget](#the-rotation-widget)
  - [Values mode](#values-mode)
  - [Time, playback and saving](#time-playback-and-saving)
  - [Controls at a glance](#controls-at-a-glance)
- [Shapes](#shapes)
- [Color mappings](#color-mappings)
- [Performance](#performance)
- [Other tools in this repo](#other-tools-in-this-repo)
- [Repository layout](#repository-layout)
- [Adding a new shape](#adding-a-new-shape)
- [Troubleshooting](#troubleshooting)
- [Research](#research)

---

## How the picture works

### Width and height

Each pixel is one point on a 2D grid, as in any image.

### Color holds a third dimension

Behind every pixel, WHCT samples a short column of points along **C**, 24 by default. Each sample is either inside the shape or outside it, which gives a 24-bit on/off pattern per pixel. A **color mapping** then turns each possible pattern into its own color. No two patterns share a color, so the pixel's color tells you exactly which parts of its C column are inside the shape.

With the default mapping (`centered_bgr`):

- low-C slices push the color toward blue, middle slices toward green, and high-C slices toward red
- the more slices that are inside, the brighter the color
- white means the whole column is inside, and black means none of it is

Turn on the probe (**P**) to see the exact on/off column under the mouse.

### Time holds the fourth

The animation steps through **T**. Each frame is one slice along T, and the slider below the image picks which slice you see.

### What the viewer keeps track of

Internally, every sample maps to a math point:

```
p = center + R · (spans × u)
```

- **u** is where the sample sits on the display: its column (W), row (H), color slice (C) and frame (T), each running from −½ to ½.
- **spans** is how much math distance each physical dimension covers. **Scaling** changes this.
- **R** is a 4D rotation, saying which math directions each physical dimension points along. **Rotating** changes this.
- **center** is the math point in the middle of the view. **Panning** changes this.

Each navigation tool changes only its own part, so they never interfere. Rotations and widget zooms pivot on the middle of the visible W, H and C, at the current frame, so the slice in front of you stays put.

---

## Quick start (Windows)

You need **Windows**, **Python 3.11** (`py -3.11` should work in a terminal) and, ideally, an **NVIDIA GPU**. The viewer runs on a CPU too, but much more slowly.

**1. Build the environment.** From the repository root, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates `WHCT_env\`. If it finds an NVIDIA GPU it installs the CUDA build of PyTorch; otherwise it installs the CPU build. Then it installs `requirements.txt` and reports whether CUDA is working. Pass `-Cuda` or `-Cpu` to force one build.

**2. Activate it** (once per terminal window):

```powershell
.\WHCT_env\Scripts\Activate.ps1
```

If PowerShell refuses to run scripts, first run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. Your prompt should now start with `(WHCT_env)`.

**3. Generate the default color mapping** (once). Mapping files are large, so they aren't stored in git:

```powershell
python generateMapping\generateMapping.py --mapping-mode centered_bgr --output generateMapping\centered_bgr_mapping.npz
```

Without a mapping file the viewer still works, using the plain "raw bits" coloring. See [Color mappings](#color-mappings) for the other mappings.

**4. Run the viewer** from the repository root:

```powershell
python -m whct
```

The status line at the bottom names the device in use: your GPU, or "CPU". To check the environment at any time, run `.\doctor.ps1`.

---

## Using the viewer

The window is almost all image. Two small navigation widgets float side by side in the top-right corner, and a slim bar along the bottom holds playback, the shape and color pickers, precision, and the Save, Configuration and Options menus.

The interface is still evolving, so this section describes how things work rather than every button.

### Moving around on the image

- **Left-drag** pans across the screen, along **W** (horizontal) and **H** (vertical).
- **Right-drag** pans through the other two dimensions: horizontal moves along **T**, vertical moves along **C**.
- Holding **Ctrl** while panning snaps a drag that is close to horizontal or vertical onto that axis, so only one dimension changes.
- **Scroll** zooms in and out of **W** and **H** around the mouse, like a normal 2D zoom.
- **Hold 1, 2, 3 or 4 while scrolling** to zoom **W, H, C or T** instead. Keys combine: 1+2 is the same as no keys, and 1+2+3+4 zooms all four. Zooming in T keeps the current frame fixed.

Panning follows the physical axes. After a rotation, a W pan moves through whatever mix of math axes W currently holds, but on screen the image always slides the way your hand moves.

The image itself never rotates the view; rotation only happens through the rotation widget.

### The scale widget (tesseract)

The right-hand widget is a tesseract (4D cube) drawn as two cubes with their matching corners joined. It shows the **shape of the 4D window you are viewing**.

- The four bold edges leaving the corner **v** are the **W, H, C and T** spans: W points right, H up, C down-left, and T along the back-right diagonal.
- **Drag a bold edge toward v** to shrink that dimension's window (zoom in along it), or away from v to enlarge it (zoom out). Widget zooms happen about the center of the view.
- The rest of the drawing gives the shape context. The main cube (holding W, H and C) is drawn lighter, the second cube lighter still, and the eight connectors along T are faintest.
- Edge lengths show **proportions**, compressed by a cube root so extreme ratios stay readable: a window 1000× wider than it is tall draws as 10:1. Absolute zoom is shown as numbers in values mode.

### The rotation widget

The left-hand widget is a small 3D scene, rendered in real 3D with lighting:

- Three axes show **W**, **H** and **C**. The corner where they meet, **r**, stands for **T**.
- Four colored arrows show the math axes **x, y, z, w**. An arrow's **direction** shows how that axis is split across W, H and C. Its **length** shows how much of it is *outside* T: full length means none of it is in T, and zero length means it lies entirely in T.
- In the default view x points along W, z along H, y along C, and w sits at r with zero length, because w is entirely time.
- A **filled** arrowhead means the axis leans toward +T, and a **hollow** one means it leans toward −T. A **ring** around r means the axis lies entirely in T. Length alone can't show that sign.
- Faint dashed half-axes labeled −W, −H and −C show where arrows go when they point the negative way. Dotted drop lines to the W–C floor help judge each arrow's height.

**To rotate**, drag one of the six **plane arrows**:

- the three curved arcs turn the **WH**, **WC** and **HC** planes
- the three short double arrows at r turn **WT**, **HT** and **CT**, moving axes into or out of time

The math arrows follow as you drag. They can't be dragged directly.

**Ctrl-drag** anywhere on the widget tumbles the 3D scene to look at it from another side. This changes only your view of the widget, not the 4D view.

The orientation is stored as a full 4D rotation matrix, not as six angles. The same orientation can be reached by turning in different orders, and a six-angle readout would depend on that order, so the plane arrows turn the view *by* an amount rather than *to* an angle.

**Both widgets** start small. Click one to expand it to four times its area. It stays expanded until you click somewhere else.

### Values mode

Press **V** to show the numbers behind an expanded widget. Everything else in the widget dims, and the numbers become editable:

- **Rotation widget:** each math arrow's parts along W, H, C and T. The boxes start minimized; click one to open it. A strip of "turn by" fields under the widget rotates each plane by a typed amount.
- **Scale widget:** the span of each dimension.

Click a field to type in it, or press and drag left or right over it to change the value smoothly. A typed rotation value is kept exactly, and the arrow's other parts adjust around it. A **Reset** button returns the widget to its default (home spans, or the default alignment) without moving the slice you're looking at.

### Time, playback and saving

- **Space** plays and pauses. The frame counter under the image (frame 26 / 31) is editable: type a frame number to jump there, or a new total to change the number of T frames. **Loop** repeats the sweep. **Bounce** plays forward and back (…29, 30, 31, 30, 29…).
- Frames rendered at full quality are kept in memory, so the second time through plays smoothly. The cache has a size limit (Options) and is cleared whenever the view changes.
- **Save** offers an MP4 video, a GIF, or the current time slice as a PNG.
- **Configuration** saves or loads a small `.whct` file holding the current view, shape, resolution, color mapping and options, so you can return to an exact setup later.
- **Options** covers:
  - the GPU shader engine (on by default) and dynamic downscale (off by default)
  - render resolution in all four dimensions: match the window, or a custom W×H with a linked-ratio toggle, plus the number of C slices and T frames
  - fit-to-window or actual-pixels display
  - the cache limit
  - playback speed

### Controls at a glance

| Input | Action |
|---|---|
| Left-drag on image | pan W / H |
| Right-drag on image | pan T (horizontal) / C (vertical) |
| Ctrl while panning | snap to one axis |
| Scroll | zoom W and H about the mouse |
| Hold 1 / 2 / 3 / 4 + scroll | zoom W / H / C / T |
| Click a widget | expand it (click elsewhere to collapse) |
| Drag a plane arrow | rotate the 4D view |
| Ctrl-drag on rotation widget | tumble the widget's 3D view |
| Drag a tesseract edge | scale one dimension |
| Space | play / pause |
| ← / → | previous / next frame |
| P | probe: show the C column under the mouse |
| V | values mode |
| F | full screen: the widgets shrink to icons (click one to expand it) and the bar appears when the mouse reaches the bottom edge. F or Esc leaves |
| Home | reset the view for the current shape |
| Ctrl+Z / Ctrl+Y | undo / redo navigation |
| Esc | collapse the widgets |

---

## Shapes

Pick the shape from the bottom bar. Each has its own home view (**Home** returns to it).

| Shape | What it is |
|---|---|
| `julia_mandelbrot` | The 4D Julia–Mandelbrot set (default) |
| `tesseract` | A 4D cube |
| `hypersphere` | A solid 4D ball |
| `4D_diamond` | The 4D cross-polytope: \|x\|+\|y\|+\|z\|+\|w\| ≤ 0.8 |
| `4D_torus` | A torus extended into 4D |
| `hollow_hypersphere` | A thick 4D spherical shell |
| `hypersphere_minus_tesseract` | A 4D ball with a cube carved out |
| `4D_sine_field` | Where sin x + sin y + sin z + sin w > 0 |
| `4D_gyroid_shell` | A thin shell around a 4D gyroid surface |

**The Julia–Mandelbrot axes.** The set iterates z → z² + c, which involves four real numbers. WHCT uses:

- **x** = real part of *c*
- **z** = imaginary part of *c*
- **y** = real part of the starting *z₀*
- **w** = imaginary part of *z₀*

In the default view the screen therefore shows the familiar Mandelbrot plane, real axis horizontal, at the middle frame (T = 0, so z₀ has no imaginary part). Moving through C and T, or rotating them onto the screen, shows the Julia-set side of the shape.

**Precision** (bottom bar) is the number of iterations used to decide whether a point is inside. Higher is more detailed and slower.

---

## Color mappings

A color mapping assigns a unique RGB color to each of the 2²⁴ possible on/off patterns of a pixel's C column. `generateMapping/generateMapping.py` builds them:

| Mode | Idea |
|---|---|
| `centered_bgr` | **Recommended default.** A direct bit shuffle: low-C slices drive blue, middle slices green (centered), high-C slices red. Fixes the blue bias of plain RGB. |
| `layered_popcount` | Groups patterns by how many slices are inside, so brightness tracks the amount of shape in the column. |
| `rgb_bytes_generalized` | Splits the bit pattern into three chunks and uses them as R, G and B, then refines neighbors on a CUDA GPU if one is available. |
| `SPC` | Arranges patterns by three features (how many slices are inside, where they sit, how often inside/outside switches), then refines neighbors on a CUDA GPU if one is available. |

Generate one with, for example:

```powershell
python generateMapping\generateMapping.py --mapping-mode layered_popcount --output generateMapping\layered_popcount_mapping.npz
```

The viewer lists every mapping file it finds in `generateMapping\`, plus "raw bits", in the **Color** menu. The first time a mapping is used, a faster `.pt` copy (about 64 MB) is saved next to it. Switching mappings recolors instantly without re-rendering.

With fewer than 24 C slices, the slices are spread across the 24 color bits, so the colors still span the whole mapping.

---

## Performance

- **GPU shader.** Frames are computed in an OpenGL fragment shader on any GPU (Intel, AMD or NVIDIA), with one early exit per point. On an Intel Iris Xe laptop a 640×480×24 Julia-Mandelbrot frame takes about 7 ms, against about 1 s for PyTorch on the CPU. Without OpenGL 3.3, or with the shader turned off in Options, rendering falls back to PyTorch (CUDA when available, otherwise the CPU). `python tests/bench_render.py` times both engines; `python -m pytest tests` checks they agree.
- **Dynamic downscale** (Options, off by default). While you are moving, the viewer renders a quick preview at lower resolution and precision, then the full image once you stop.
- **Background rendering.** Rendering runs off the main thread, so the window stays responsive, and a newer request cancels an unfinished full render.
- **Cost.** Each frame evaluates W × H × C points (a 1200×700 window with 24 slices is about 20 million). Any change that touches T invalidates every frame, so playback re-renders and re-caches the sweep.
- **Memory.** Cached frames take 4 bytes per pixel. The cache limit (1 GB by default) stops long or high-resolution sweeps from filling memory.
- **Zoom depth.** Coordinates are 32-bit floats, so extremely deep zooms (beyond roughly a millionth of the starting size) lose precision and turn blocky.

---

## Other tools in this repo

### MandelCompiler (scripted animations)

`MandelCompiler/MandelCompiler.py` renders an MP4 from a small command file of keyframes, such as the gyroid animations in `MandelCompiler/commandSets/`. It drives the **previous** viewer, `WHCS_PySide6.py`, and uses that viewer's six-angle rotations and min/max window format:

```powershell
python MandelCompiler\MandelCompiler.py MandelCompiler\commandSets\gyroid_basic.txt --mapping generateMapping\centered_bgr_mapping.npz --display 1
```

Output goes to `MandelCompiler\outputs\` unless `--output` is given. The command format is documented at the top of the script.

A new command system built around the WHCT navigation model is planned to replace it.

### Previous viewers

- `WHCS_PySide6.py` is the previous PySide6 viewer, with six rotation sliders and min/max fields. MandelCompiler still uses it. Run it with `python WHCS_PySide6.py [--mapping path.npz]`. Its axis conventions differ from the new viewer's.
- `legacy/` holds the earlier tkinter/OpenCV viewers (`JuliaMandel.py`, `WHCS.py`), early experiments and the original notebook. They are kept for reference and are not maintained.

---

## Repository layout

```
whct/                    The viewer (python -m whct)
  state.py               View state: center, spans, rotation; pan/scale/rotate math
  indicators.py          The 4D shapes (PyTorch and GLSL versions)
  engine.py              PyTorch rendering, engine switch, background worker, frame cache
  gpu.py                 Shader rendering (OpenGL fragment shader on its own thread)
  colormap.py            Color mappings and bit packing
  gizmos.py              Rotation widget (OpenGL) and scale widget
  valuefields.py         Editable number fields for values mode
  canvas.py              The image view and mouse interaction
  window.py              Main window, playback, saving, options, shortcuts
  __main__.py            Entry point
generateMapping/         Color-mapping generator (+ generated .npz/.pt files, not in git)
MandelCompiler/          Command-file animation renderer for the previous viewer
WHCS_PySide6.py          Previous viewer
legacy/                  Older viewers and experiments
Research/                Paper drafts, study materials, figures and animations
setup.ps1                Builds WHCT_env (CUDA or CPU PyTorch)
doctor.ps1               Checks the environment
requirements.txt         Python dependencies (PyTorch is installed separately)
videoOutput/             Default folder for saved videos and images (not in git)
```

---

## Adding a new shape

A shape is a function that takes the math coordinates as PyTorch tensors and returns a boolean tensor, True where the point is inside:

```python
# whct/indicators.py
def my_shape(x, y, z, w, precision):
    return (x * x + y * y + z * z + w * w) <= 0.25

INDICATORS["my_shape"] = my_shape
```

Use tensor operations only, so it runs on the GPU. `precision` is available for iterative shapes. To give the shape its own home view, add an entry to `HOME_VIEWS` in `whct/state.py`: a center (x, y, z, w) and spans (W, H, C, T). It then appears in the Shape menu.

---

## Troubleshooting

**`activate` isn't recognized in PowerShell.** Run it with a path from the repository root: `.\WHCT_env\Scripts\Activate.ps1`. The `.bat` version is for cmd.exe.

**`No module named whct`.** Run `python -m whct` from the repository root, not from inside `WHCT_env\Scripts`.

**The status line says CPU on a machine with an NVIDIA GPU.** PyTorch was installed without CUDA. Rerun `.\setup.ps1 -Cuda`, or reinstall manually:

```powershell
python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

The CUDA 12.8 build needs a recent NVIDIA driver. Don't add `torch` to `requirements.txt`: a plain `pip install torch` gets the CPU build.

**Only "raw bits" in the Color menu.** No mapping files have been generated yet. See step 3 of the quick start.

**Everything is very slow.** The status bar probably says "PyTorch · CPU": the shader engine is off or unavailable (see the console for why). Turn on the shader or dynamic downscale in Options, lower the resolution in Options, or reduce C slices and precision.

**The widgets look wrong or the window is blank.** The viewer needs OpenGL 2.1 or newer. Update your graphics driver.

---

## Research

`Research/` contains the paper draft *XYTC: Lossless and Complete Representation of Four-dimensional Data Within Two-dimensional Displays*, study materials, and the figures and animations behind the explanations (`Research/4D_CSO/`).
