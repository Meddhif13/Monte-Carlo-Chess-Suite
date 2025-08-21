# PowerShell: minimal report reproduction suite
param(
  [string]$Py = "python",
  [string]$Csv = "logs/experiments.csv",
  [string]$Pgn = "logs/pgn"
)

# Clean optional: Uncomment to reset logs
# if (Test-Path $Csv) { Remove-Item $Csv }

# 1) Head-to-head baseline (may be drawish at low playouts)
& $Py tournament.py --strategies PUTC_GRAVE PUTC --games 8 --playouts 40 --max-steps 300 --processes 1 --seed 0 --use-eval-priors --use-eval-leaf --rollout-steps 0 --grave-K 100 --eval-adjudicate --eval-threshold 120 --random-openings 6 --resign-threshold 300 --resign-moves 4 --pgn-dir $Pgn --log-csv $Csv

# 2) Absolute baseline vs UNIFORM
& $Py tournament.py --strategies PUTC_GRAVE UNIFORM --games 8 --playouts 25 --max-steps 200 --processes 1 --seed 1 --use-eval-priors --use-eval-leaf --rollout-steps 0 --grave-K 100 --eval-adjudicate --eval-threshold 120 --random-openings 4 --resign-threshold 250 --resign-moves 3 --pgn-dir $Pgn --log-csv $Csv

# 3) Versus UCB (harder baseline) with AB leaves for PUTC_GRAVE
& $Py tournament.py --strategies PUTC_GRAVE UCB --games 8 --playouts 40 --max-steps 300 --processes 1 --seed 2 --use-eval-priors --use-eval-leaf --rollout-steps 0 --grave-K 100 --ab-leaf-depth 1 --eval-adjudicate --eval-threshold 120 --random-openings 6 --resign-threshold 300 --resign-moves 4 --pgn-dir $Pgn --log-csv $Csv

# Aggregate summary (stdout)
& $Py scripts/aggregate_results.py --csv $Csv --subject PUTC_GRAVE
