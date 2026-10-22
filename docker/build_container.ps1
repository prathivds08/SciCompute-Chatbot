# ==============================================================================
# Build and Verify DeePMD Analysis Sandbox Docker Image
# ==============================================================================

Write-Host ">>> Checking Docker daemon availability..." -ForegroundColor Cyan

try {
    $dockerVersion = & docker version --format '{{.Server.Version}}' 2>$null
    if (-not $dockerVersion) {
        Write-Host "ERROR: Docker daemon is not running or docker CLI is not installed." -ForegroundColor Red
        Write-Host "Please start Docker Desktop and ensure Docker CLI is available on your PATH." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Docker daemon is online (version $dockerVersion)." -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to invoke docker CLI. Please ensure Docker Desktop is installed and running." -ForegroundColor Red
    exit 1
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Write-Host ">>> Building Docker image 'deepmd-analysis-sandbox:latest' from $repoRoot..." -ForegroundColor Cyan

& docker build -t deepmd-analysis-sandbox:latest -f "$PSScriptRoot/Dockerfile.deepmd" "$repoRoot"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker build failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ">>> Testing container execution and Python environment..." -ForegroundColor Cyan
& docker run --rm deepmd-analysis-sandbox:latest python -c "import numpy, scipy, matplotlib, ase; print('Container health check passed! ASE:', ase.__version__)"

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: deepmd-analysis-sandbox:latest is ready for safe isolated execution!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Health check failed inside container." -ForegroundColor Red
}
