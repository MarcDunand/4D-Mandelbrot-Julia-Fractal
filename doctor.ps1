# doctor.ps1: quick environment check. Run from the repo root with WHCT_env activated.
$ErrorActionPreference = "Stop"

Write-Host "== Python ==" -ForegroundColor Cyan
python --version
python -c "import sys; print('exe:', sys.executable)"

Write-Host "`n== Core imports ==" -ForegroundColor Cyan
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import cv2; print('opencv', cv2.__version__)"
python -c "import PySide6; print('PySide6', PySide6.__version__)"
python -c "import OpenGL; print('PyOpenGL', OpenGL.__version__)"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

Write-Host "`n== WHCT app ==" -ForegroundColor Cyan
python -c "import whct.window; print('whct ok')"

Write-Host "`nDoctor checks passed. Start the viewer with: python -m whct" -ForegroundColor Green
