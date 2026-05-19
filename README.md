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

## Setup on Windows with NVIDIA GPU

This project uses PyTorch with CUDA for GPU acceleration. Do **not** install `torch` from the normal `requirements.txt` flow, because that can install the CPU-only version of PyTorch.

### 1. Clone the repo

```powershell
git clone <your-repo-url>
cd 4D-Mandelbrot-Julia-Fractal
```

### 2. Create a virtual environment

Use Python 3.11 explicitly:

```powershell
& "C:\Program Files\Python311\python.exe" -m venv WHCT_env
```

Activate the environment:

```powershell
.\WHCT_env\Scripts\activate
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\WHCT_env\Scripts\activate
```

After activation, the terminal prompt should start with something like:

```text
(WHCT_env) PS C:\Users\Owner\Documents\dev\4D-Mandelbrot-Julia-Fractal>
```

Confirm that Python is coming from the virtual environment:

```powershell
python -c "import sys; print(sys.executable)"
```

Expected output should point somewhere inside the project folder, for example:

```text
C:\Users\Owner\Documents\dev\4D-Mandelbrot-Julia-Fractal\WHCT_env\Scripts\python.exe
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 4. Install CUDA PyTorch first

For an NVIDIA GPU, install the CUDA 12.8 PyTorch build:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify that CUDA is working:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Expected output should include:

```text
+cu128
True
NVIDIA ...
```

If the output says `+cpu`, `False`, or `None`, then PyTorch was installed incorrectly for GPU use.

### 5. Install the rest of the dependencies

Install the rest of the project requirements:

```powershell
python -m pip install -r requirements.txt
```

After installing requirements, check CUDA again:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If CUDA was accidentally replaced by CPU PyTorch, reinstall CUDA PyTorch:

```powershell
python -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then verify again:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### 6. Run the project

```powershell
python .\WHCS_PySide6.py
```

The app should report that it is using `cuda` rather than `cpu`.

## Important dependency note

Do **not** include normal `torch`, `torchvision`, or `torchaudio` lines in `requirements.txt` unless you specifically want the CPU build.

Remove lines like these from `requirements.txt`:

```text
torch==2.10.0
torchvision==...
torchaudio==...
```

Instead, install CUDA PyTorch separately before installing the rest of the requirements.

A useful comment to keep near the top of `requirements.txt` is:

```text
# Install CUDA PyTorch separately before this file:
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#
# Do not list torch/torchvision/torchaudio here unless you know you want the CPU build.
```

## GitHub note

Do **not** commit the virtual environment folder to GitHub. Add these lines to `.gitignore`:

```gitignore
WHCT_env/
.venv/
venv/
env/
__pycache__/
*.py[cod]
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
