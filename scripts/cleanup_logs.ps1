param(
    [switch]$Execute
)

# Cleanup plan: keep master artifacts, sweep results, tournament summaries, and AB-vs-SF PGNs.
# Dry-run by default: shows what would be removed. Use -Execute to actually delete.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo  = Split-Path -Parent $root
$logs  = Join-Path $repo 'logs'

if (-not (Test-Path $logs)) {
    Write-Host "Logs folder not found: $logs" -ForegroundColor Yellow
    exit 1
}

# Keep list (files/folders)
$keepNames = @(
    'MASTER_REPORT.md',
    'PROGRESS.md',
    'sweep_500ms_elo1320',
    'tournament-*.md',
    'pgn_ab_sf_*'
)

# Remove patterns (files) known to be exploratory or temporary
$removeFilePatterns = @(
    '*smoke*.csv'
)

# Helper to test whether a path should be kept
function ShouldKeep($item) {
    $name = Split-Path $item -Leaf
    foreach ($pat in $keepNames) {
        if ($name -like $pat) { return $true }
    }
    # Also keep anything inside the sweep folder and AB-vs-SF PGN folders
    if ($item -like (Join-Path $logs 'sweep_500ms_elo1320*')) { return $true }
    if ($item -like (Join-Path $logs 'pgn_ab_sf_*')) { return $true }
    return $false
}

$toRemove = @()

# 1) Remove files matching remove patterns, unless kept
Get-ChildItem -Path $logs -Recurse -File | ForEach-Object {
    $file = $_.FullName
    if (ShouldKeep $file) { return }
    foreach ($pat in $removeFilePatterns) {
        if ($_.Name -like $pat) { $toRemove += $file; break }
    }
}

# 2) Remove empty PGN directories not referenced (generic pgn/ within logs)
Get-ChildItem -Path $logs -Directory | Where-Object { $_.Name -like 'pgn' } | ForEach-Object {
    if (-not (ShouldKeep $_.FullName)) { $toRemove += $_.FullName }
}

# 3) Report
if ($toRemove.Count -eq 0) {
    Write-Host 'Nothing to remove based on current rules.' -ForegroundColor Green
    exit 0
}

Write-Host "Items to remove ($($toRemove.Count)):" -ForegroundColor Cyan
$toRemove | ForEach-Object { Write-Host " - $_" }

if ($Execute) {
    Write-Host 'Executing removal...' -ForegroundColor Yellow
    foreach ($p in $toRemove) {
        if (Test-Path $p) {
            try {
                if ((Get-Item $p).PSIsContainer) {
                    Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction Stop
                } else {
                    Remove-Item -LiteralPath $p -Force -ErrorAction Stop
                }
                Write-Host "Removed: $p" -ForegroundColor DarkGray
            } catch {
                Write-Host "Failed to remove: $p -> $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
    Write-Host 'Cleanup complete.' -ForegroundColor Green
} else {
    Write-Host 'Dry run only. Re-run with -Execute to apply.' -ForegroundColor Yellow
}
