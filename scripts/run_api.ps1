# Always run from D: project .venv — never install into global / AppData (C:).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Missing .venv. Create with: python -m venv .venv"
    exit 1
}
if (-not $env:SECURECOATING_ENV) { $env:SECURECOATING_ENV = "development" }
if (-not $env:SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO) { $env:SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO = "true" }
if (-not $env:SECURECOATING_ENABLE_SENSOR_SIMULATION) { $env:SECURECOATING_ENABLE_SENSOR_SIMULATION = "true" }
& $py -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
