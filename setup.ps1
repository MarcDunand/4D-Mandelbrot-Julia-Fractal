# setup.ps1
# Builds the WHCT_env virtual environment. Installs CUDA PyTorch when an NVIDIA
# GPU is present, otherwise CPU PyTorch. Run from the repository root:
#     powershell -ExecutionPolicy Bypass -File .\setup.ps1
# Pass -Cpu to force the CPU build, or -Cuda to force the CUDA build.

param(
    [switch]$Cpu,
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$EnvDir = "WHCT_env"
$Python = Join-Path $EnvDir "Scripts\python.exe"
$CudaIndex = "https://download.pytorch.org/whl/cu128"
$CpuIndex = "https://download.pytorch.org/whl/cpu"

function Test-NvidiaGpu {
    $gpus = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name
    return [bool]($gpus | Where-Object { $_ -match "NVIDIA" })
}

if ($Cuda) { $UseCuda = $true }
elseif ($Cpu) { $UseCuda = $false }
else { $UseCuda = Test-NvidiaGpu }

if (-not (Test-Path $Python)) {
    Write-Host "== Creating $EnvDir with Python 3.11 ==" -ForegroundColor Cyan
    py -3.11 -m venv $EnvDir
    if ($LASTEXITCODE -ne 0) { throw "Could not create venv. Is Python 3.11 installed (py -3.11)?" }
}

& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

if ($UseCuda) {
    Write-Host "== NVIDIA GPU: installing CUDA PyTorch ==" -ForegroundColor Cyan
    & $Python -m pip install --force-reinstall torch --index-url $CudaIndex
}
else {
    Write-Host "== No NVIDIA GPU: installing CPU PyTorch ==" -ForegroundColor Cyan
    & $Python -m pip install torch --index-url $CpuIndex
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed." }

Write-Host "== Installing requirements.txt ==" -ForegroundColor Cyan
& $Python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements.txt install failed." }

Write-Host "== Verifying ==" -ForegroundColor Cyan
& $Python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
$CudaOk = (& $Python -c "import torch; print(torch.cuda.is_available())").Trim() -eq "True"

if ($UseCuda -and -not $CudaOk) {
    Write-Host "WARNING: NVIDIA GPU found but CUDA is not available to PyTorch. Update the NVIDIA driver (needs CUDA 12.8 support) and rerun." -ForegroundColor Yellow
    exit 1
}

Write-Host "`nSetup done. Activate with: .\$EnvDir\Scripts\activate" -ForegroundColor Green
