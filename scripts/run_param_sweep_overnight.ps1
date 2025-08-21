# Run parameter sweeps vs Stockfish (explicit settings) overnight
# Uses $PSScriptRoot to locate repo root and activate venv

$ErrorActionPreference = 'Stop'
$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptDir
$Activate = Join-Path $RepoRoot 'env\Scripts\Activate.ps1'
if (-not (Test-Path $Activate)) { throw "Virtual env not found: $Activate" }
& $Activate
Set-Location $RepoRoot

# Ensure logs dir
New-Item -ItemType Directory -Path .\logs -ErrorAction SilentlyContinue | Out-Null

# Run sweep with default set of strategies, 2 games per side per config
python .\scripts\param_sweep_weak_sf.py --games-per-side 2 --out-dir logs

Write-Host "Parameter sweep completed. See logs\\param-sweep-*.csv and summary MD."
