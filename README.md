# 4D Julia--Mandelbrot Visualizer

A Windows-only 4D visualization tool for exploring the **4D
Julia--Mandelbrot fractal**.

The visualizer encodes four dimensions as: - **X / Y**: two spatial
dimensions on screen - **Color**: a third dimension mapped into color -
**Time**: a fourth dimension shown across time slices

The system supports free navigation in 4D, saving visualizations, and
writing scripts that automatically navigate the space and export videos
of those navigations.

------------------------------------------------------------------------

## Requirements

### Operating System

-   **Windows only**

### Python

-   **Python 3.11**
-   Use the Windows Python Launcher (`py`)

### Hardware

-   Runs on CPU
-   **Strongly recommended:** NVIDIA GPU with CUDA for significantly
    faster performance
-   If CUDA is unavailable, the program automatically falls back to CPU

------------------------------------------------------------------------

## Setup (Windows)

Clone the repository and enter it:

``` powershell
git clone <REPO_URL>
cd <REPO_FOLDER>
```

Create and activate a virtual environment:

``` powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

``` powershell
pip install -r requirements.txt
```

(Optional but recommended) Run the environment diagnostic script:

``` powershell
.\doctor.ps1
```

------------------------------------------------------------------------

## Run

From the repository root:

``` powershell
python .\JuliaMandel.py
```

**First successful run:** the visualizer window opens and runs normally.

------------------------------------------------------------------------

## CUDA / GPU Notes

To check whether PyTorch can access CUDA:

``` powershell
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

If `cuda available` is `False`, the program will still run using CPU
(slower).

------------------------------------------------------------------------

## Repository Structure (High Level)

Important files and folders at the top level:

-   `JuliaMandel.py` --- main entry point\
-   `ColorConverter.py` --- local color selection / conversion
    utilities\
-   `ColorInspector.py` --- UI component for inspecting the color
    dimension\
-   `generateMapping/` --- mapping and LUT generation utilities\
-   `MandelCompiler.py`, `mandelV2.py`, `raysMandel.py` --- supporting
    modules / experiments\
-   `MandelbrotVis.ipynb` --- Jupyter notebook (optional / exploratory)

Common output directories (gitignored):

-   `videoOutput/` --- rendered videos\
-   `coolOutputs/` --- saved visualizations or exports

------------------------------------------------------------------------

## Notes

-   This project assumes it is run from the repository root.
-   There are no required configuration files or command-line arguments.
-   The virtual environment (`venv/`) is intentionally not tracked in
    git.

------------------------------------------------------------------------

## Troubleshooting

### Python not found

Use the Python launcher instead of `python`:

``` powershell
py --version
```

### Import errors for `ColorConverter` or `ColorInspector`

These are **local project files** and must remain in the repository.
Ensure they are not moved without updating imports.

### Torch install issues

If dependency installation fails around `torch`, the issue is usually
related to CUDA or wheel compatibility. Capture the full error output
and resolve by selecting the appropriate PyTorch build (CPU-only or
CUDA).

------------------------------------------------------------------------

## License

Internal / research use. Add a license if external distribution is
planned.
