This folder contains experiment artifacts:

- PROGRESS.md: human-readable change log and experiment notes.
- experiments.csv: per-game CSV logs (timestamp, players, params, result, perf metrics like moves, duration, playouts/s).
- pgn/: per-pairing PGN game logs.

Tips:
- Aggregate results: `python scripts/aggregate_results.py --csv logs/experiments.csv --subject PUTC_GRAVE`
- To increase decisiveness: use `--max-steps`, `--eval-adjudicate --eval-threshold`, `--random-openings`, or `--resign-threshold --resign-moves`.
