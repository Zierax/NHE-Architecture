# analysis/ — read-only analysis of committed result files

No model needed. Reproduce every table in `../results/NUMBERS.md`:

- `stats.py strict|battery|significance` — strict scoring, Africa battery, static-arm stats
- `sweep.py thresholds|windows` — offline threshold/window sweeps (validated 1:1 vs live)
