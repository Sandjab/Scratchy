# Scratchy Quick Start Installation Script (Windows)
# Requires Python 3.12 to be installed

param(
    [string]$VenvName = ".venv",
    [switch]$SkipOptional,
    [switch]$IncludeZTurbo
)

$ErrorActionPreference = "Stop"

Write-Host "=== Scratchy Installation ===" -ForegroundColor Cyan

# Check Python 3.12
Write-Host "`nChecking Python 3.12..." -ForegroundColor Yellow
try {
    $pythonVersion = py -3.12 --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python 3.12 not found. Please install Python 3.12." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`nCreating virtual environment '$VenvName'..." -ForegroundColor Yellow
if (Test-Path $VenvName) {
    Write-Host "Virtual environment already exists. Delete it first if you want a fresh install." -ForegroundColor Red
    exit 1
}
py -3.12 -m venv $VenvName
Write-Host "Virtual environment created." -ForegroundColor Green

# Get pip path
$pipPath = Join-Path $VenvName "Scripts\pip.exe"

# Install PyTorch
Write-Host "`nInstalling PyTorch 2.9.1 with CUDA 12.8..." -ForegroundColor Yellow
& $pipPath install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install xformers
Write-Host "`nInstalling xformers..." -ForegroundColor Yellow
& $pipPath install xformers==0.0.33.post2 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Install triton-windows
Write-Host "`nInstalling triton-windows..." -ForegroundColor Yellow
& $pipPath install triton-windows
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Optional: flash-attention and sageattention
if (-not $SkipOptional) {
    Write-Host "`nInstalling flash-attention..." -ForegroundColor Yellow
    & $pipPath install https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.3+cu128torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`nInstalling sageattention..." -ForegroundColor Yellow
    & $pipPath install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.9.0.post3-cp39-abi3-win_amd64.whl
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "`nSkipping optional packages (flash-attention, sageattention)." -ForegroundColor Yellow
}

# Install Scratchy
Write-Host "`nInstalling Scratchy..." -ForegroundColor Yellow
& $pipPath install -e .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Optional: diffusers from source for Z-Image-Turbo
if ($IncludeZTurbo) {
    Write-Host "`nInstalling diffusers from source (for Z-Image-Turbo)..." -ForegroundColor Yellow
    & $pipPath install git+https://github.com/huggingface/diffusers
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# Verify installation
Write-Host "`n=== Verifying Installation ===" -ForegroundColor Cyan
$pythonPath = Join-Path $VenvName "Scripts\python.exe"
& $pythonPath check.py

Write-Host "`n=== Installation Complete ===" -ForegroundColor Green
Write-Host "Activate the environment with: $VenvName\Scripts\activate" -ForegroundColor Cyan
