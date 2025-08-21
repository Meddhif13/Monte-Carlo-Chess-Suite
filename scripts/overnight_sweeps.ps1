<#
Activate venv and run a series of Elo-based sweeps overnight.
This script is robust to where it's launched from by using $PSScriptRoot.
Logs: Markdown in .\logs\*.md, CSVs in .\logs\*.csv, PGNs in .\pgn
#>

$ErrorActionPreference = 'Stop'

# Compute repo root relative to script location
$ScriptDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $ScriptDir

# Activate environment
$Activate = Join-Path $RepoRoot 'env\Scripts\Activate.ps1'
if (-not (Test-Path $Activate)) {
	throw "Virtual env not found at: $Activate"
}
& $Activate

# Ensure we run from repo root so relative paths work
Set-Location $RepoRoot

# Common knobs
$PLAYOUTS_FAST = 80
$PLAYOUTS_MED = 120
$GAMES = 6                 # per side per Elo step
$ELO_START = 200
$ELO_END = 2000
$ELO_STEP = 200
$MAX_STEPS = 160
$RESIGN_THR = 1200
$RESIGN_MOVES = 4
$SF_THREADS = 2
$SF_HASH = 256

# Toggle: use depth-based sweep against Stockfish with explicit settings
$USE_DEPTH_SWEEP = $true    # set $false to use Elo-based sweep
$DEPTH_START = 1
$DEPTH_END = 2
$SF_SKILL = 0               # Stockfish skill level
$SF_MOVETIME_MS = 0         # keep 0 (depth-limited) or set tiny like 10

# Ensure dirs exist
New-Item -ItemType Directory -Path .\logs -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path .\pgn -ErrorAction SilentlyContinue | Out-Null

if ($USE_DEPTH_SWEEP) {
	# 1) PUTC_GRAVE tuned (fast) vs Stockfish (skill 0, depth 1-2)
	python .\tournament.py --ssf-sweep --strategies PUTC_GRAVE --playouts $PLAYOUTS_FAST --ssf-games-per-side $GAMES --ssf-depth-start $DEPTH_START --ssf-depth-end $DEPTH_END --rollout-steps 0 --use-eval-leaf --ab-leaf-depth 1 --use-eval-priors --c-puct 1.2 --grave-K 150 --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads 1 --sf-hash-mb 16 --sf-skill $SF_SKILL --log-md .\logs\depth-sweep-putc_grave.md --log-csv .\logs\depth-sweep-putc_grave.csv --pgn-dir .\pgn

	# 2) PUTC tuned (fast)
	python .\tournament.py --ssf-sweep --strategies PUTC --playouts $PLAYOUTS_FAST --ssf-games-per-side $GAMES --ssf-depth-start $DEPTH_START --ssf-depth-end $DEPTH_END --rollout-steps 0 --use-eval-leaf --ab-leaf-depth 1 --use-eval-priors --c-puct 1.2 --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads 1 --sf-hash-mb 16 --sf-skill $SF_SKILL --log-md .\logs\depth-sweep-putc.md --log-csv .\logs\depth-sweep-putc.csv --pgn-dir .\pgn

	# 3) PUTC_RAVE (medium)
	python .\tournament.py --ssf-sweep --strategies PUTC_RAVE --playouts $PLAYOUTS_MED --ssf-games-per-side $GAMES --ssf-depth-start $DEPTH_START --ssf-depth-end $DEPTH_END --rollout-steps 0 --use-eval-leaf --ab-leaf-depth 1 --use-eval-priors --c-puct 1.1 --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads 1 --sf-hash-mb 16 --sf-skill $SF_SKILL --log-md .\logs\depth-sweep-putc_rave.md --log-csv .\logs\depth-sweep-putc_rave.csv --pgn-dir .\pgn

	# 4) UCT_RAVE (medium)
	python .\tournament.py --ssf-sweep --strategies UCT_RAVE --playouts $PLAYOUTS_MED --ssf-games-per-side $GAMES --ssf-depth-start $DEPTH_START --ssf-depth-end $DEPTH_END --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads 1 --sf-hash-mb 16 --sf-skill $SF_SKILL --log-md .\logs\depth-sweep-uct_rave.md --log-csv .\logs\depth-sweep-uct_rave.csv --pgn-dir .\pgn

	# 5) UCT (medium)
	python .\tournament.py --ssf-sweep --strategies UCT --playouts $PLAYOUTS_MED --ssf-games-per-side $GAMES --ssf-depth-start $DEPTH_START --ssf-depth-end $DEPTH_END --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads 1 --sf-hash-mb 16 --sf-skill $SF_SKILL --log-md .\logs\depth-sweep-uct.md --log-csv .\logs\depth-sweep-uct.csv --pgn-dir .\pgn

	# 6) UCB pushed (try to maximize)
	python .\tournament.py --ssf-sweep --strategies UCB --playouts 300 --ssf-games-per-side $GAMES --ssf-depth-start $DEPTH_START --ssf-depth-end $DEPTH_END --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --ucb-c 1.4 --sf-threads 1 --sf-hash-mb 16 --sf-skill $SF_SKILL --log-md .\logs\depth-sweep-ucb.md --log-csv .\logs\depth-sweep-ucb.csv --pgn-dir .\pgn
}
else {
	# Elo-based sweeps (note: some Stockfish builds clamp minimum Elo; use explicit depth/skill if needed)
	python .\tournament.py --ssf-elo-sweep --strategies PUTC_GRAVE --playouts $PLAYOUTS_FAST --ssf-games-per-side $GAMES --ssf-elo-start $ELO_START --ssf-elo-end $ELO_END --ssf-elo-step $ELO_STEP --rollout-steps 0 --use-eval-leaf --ab-leaf-depth 1 --use-eval-priors --c-puct 1.2 --grave-K 150 --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads $SF_THREADS --sf-hash-mb $SF_HASH --sf-skill $SF_SKILL --log-md .\logs\elo-sweep-putc_grave.md --log-csv .\logs\elo-sweep-putc_grave.csv --pgn-dir .\pgn
	python .\tournament.py --ssf-elo-sweep --strategies PUTC --playouts $PLAYOUTS_FAST --ssf-games-per-side $GAMES --ssf-elo-start $ELO_START --ssf-elo-end $ELO_END --ssf-elo-step $ELO_STEP --rollout-steps 0 --use-eval-leaf --ab-leaf-depth 1 --use-eval-priors --c-puct 1.2 --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads $SF_THREADS --sf-hash-mb $SF_HASH --sf-skill $SF_SKILL --log-md .\logs\elo-sweep-putc.md --log-csv .\logs\elo-sweep-putc.csv --pgn-dir .\pgn
	python .\tournament.py --ssf-elo-sweep --strategies PUTC_RAVE --playouts $PLAYOUTS_MED --ssf-games-per-side $GAMES --ssf-elo-start $ELO_START --ssf-elo-end $ELO_END --ssf-elo-step $ELO_STEP --rollout-steps 0 --use-eval-leaf --ab-leaf-depth 1 --use-eval-priors --c-puct 1.1 --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads $SF_THREADS --sf-hash-mb $SF_HASH --sf-skill $SF_SKILL --log-md .\logs\elo-sweep-putc_rave.md --log-csv .\logs\elo-sweep-putc_rave.csv --pgn-dir .\pgn
	python .\tournament.py --ssf-elo-sweep --strategies UCT_RAVE --playouts $PLAYOUTS_MED --ssf-games-per-side $GAMES --ssf-elo-start $ELO_START --ssf-elo-end $ELO_END --ssf-elo-step $ELO_STEP --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads $SF_THREADS --sf-hash-mb $SF_HASH --sf-skill $SF_SKILL --log-md .\logs\elo-sweep-uct_rave.md --log-csv .\logs\elo-sweep-uct_rave.csv --pgn-dir .\pgn
	python .\tournament.py --ssf-elo-sweep --strategies UCT --playouts $PLAYOUTS_MED --ssf-games-per-side $GAMES --ssf-elo-start $ELO_START --ssf-elo-end $ELO_END --ssf-elo-step $ELO_STEP --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --sf-threads $SF_THREADS --sf-hash-mb $SF_HASH --sf-skill $SF_SKILL --log-md .\logs\elo-sweep-uct.md --log-csv .\logs\elo-sweep-uct.csv --pgn-dir .\pgn
	python .\tournament.py --ssf-elo-sweep --strategies UCB --playouts 300 --ssf-games-per-side $GAMES --ssf-elo-start 600 --ssf-elo-end 1600 --ssf-elo-step 200 --max-steps $MAX_STEPS --eval-adjudicate --eval-threshold 200 --resign-threshold $RESIGN_THR --resign-moves $RESIGN_MOVES --ucb-c 1.4 --sf-threads $SF_THREADS --sf-hash-mb $SF_HASH --sf-skill $SF_SKILL --log-md .\logs\elo-sweep-ucb.md --log-csv .\logs\elo-sweep-ucb.csv --pgn-dir .\pgn
}

Write-Host "All sweeps queued. Check .\\logs for summaries and CSVs."
