# Always run from D: project .venv — never install into global / AppData (C:).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Missing .venv. Create with: python -m venv .venv"
    exit 1
}
& $py -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
