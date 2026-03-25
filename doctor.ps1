# scripts/doctor.ps1
$ErrorActionPreference = "Stop"

Write-Host "== Python ==" -ForegroundColor Cyan
python --version
pip --version
python -c "import sys; print('exe:', sys.executable)"

Write-Host "`n== Core imports ==" -ForegroundColor Cyan
python -c "import numpy; print('numpy ok')"
python -c "import cv2; print('opencv ok')"
python -c "import torch; print('torch ok'); print('torch cuda:', torch.cuda.is_available())"
python -c "import matplotlib; print('matplotlib ok')"
python -c "from joblib import Parallel, delayed; print('joblib ok')"

Write-Host "`n== Local modules ==" -ForegroundColor Cyan
python -c "import ColorConverter; print('ColorConverter ok')"
python -c "import ColorInspector; print('ColorInspector ok')"

Write-Host "`nDoctor checks passed." -ForegroundColor Green