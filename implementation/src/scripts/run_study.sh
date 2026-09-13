#!/usr/bin/env bash
# The case study (energy_tests.md §8.6), config/study.yaml: RecServe, the timetable
# and the dynamic policy (the OLT's cost broadcast on the PON), on predictable
# traffic (BurstGPT's conversation log as recorded) and unpredictable traffic (the
# same with unforeseen surges and dips).
#   main          surge factor 1 (predictable), 1.5, 2, 3, 5               x 3 seeds = 15 runs
#   sensitivity   factors 1 and 3, each with one change: average accounting; a 5x
#                 cheaper ONU; 1x2000, 10x200 or 100x20 households; statistics
#                 learned per household                                     x 3 seeds = 36 runs
#   real bursts   BurstGPT's all traffic (API included), as recorded         x 3 seeds = 3 runs
# Runs already finished are skipped, so the script can be restarted. Automatic sleep
# is blocked while it runs; closing the lid still suspends the laptop.
#   bash src/scripts/run_study.sh [parallel jobs]
set -euo pipefail
cd "$(dirname "$0")/../.."                        # implementation/
OUT=results/energy_tests/study
mkdir -p "$OUT"
PY=${PYTHON:-.venv/bin/python}
[ -f data/load_traces/burstgpt_hourly.csv ] || "$PY" src/scripts/prepare_load_traces.py
INHIBIT=()
command -v systemd-inhibit > /dev/null && INHIBIT=(systemd-inhibit --what=idle:sleep --who="tcc study" --why="simulation runs")

jobs() {
  for seed in 7 8 9; do
    for f in 1 1.5 2 3 5; do echo "main_surge$f $seed --surge-factor $f"; done
    for f in 1 3; do
      echo "average_surge$f $seed --surge-factor $f --accounting average"
      echo "onu0.2_surge$f $seed --surge-factor $f --onu-scale 0.2"
      echo "hh1x2000_surge$f $seed --surge-factor $f --households 1 --per-day 2000"
      echo "hh10x200_surge$f $seed --surge-factor $f --households 10 --per-day 200"
      echo "hh100x20_surge$f $seed --surge-factor $f --households 100 --per-day 20"
      echo "perhousehold_surge$f $seed --surge-factor $f --no-shared-stats"
    done
    echo "alltraffic $seed --load-trace all"
  done
}

jobs | "${INHIBIT[@]}" xargs -P "${1:-10}" -L 1 bash -c '
  tag=$0 seed=$1; shift
  name='"$OUT"'/study_${tag}_seed${seed}
  [ -f "$name.json" ] && exit 0
  '"$PY"' src/scripts/sim_piggyback.py --config config/study.yaml --seed "$seed" "$@" \
      --out "$name.csv" > "$name.txt" 2>&1
  echo "done $name"'
