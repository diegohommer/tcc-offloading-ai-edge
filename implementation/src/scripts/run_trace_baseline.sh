#!/usr/bin/env bash
# The trace-driven baseline (energy_tests.md §8.5): BurstGPT's load replayed at the OLT,
# configs and schedule calibrated on the first 30 days, policies run on the 31 after.
#   conversation trace: 2 accountings x 2 ONU costs x 4 household rates x 3 seeds = 48 runs
#   all traffic:        2 accountings x 2 ONU costs x 40 households at 50/day x 3 seeds = 12 runs
# ~5 min a run; -P 10 on 12 cores takes about half an hour.
#   bash src/scripts/run_trace_baseline.sh [parallel jobs]
set -euo pipefail
cd "$(dirname "$0")/../.."                        # implementation/
OUT=results/energy_tests/trace
mkdir -p "$OUT"
PY=${PYTHON:-.venv/bin/python}
[ -f data/load_traces/burstgpt_hourly.csv ] || "$PY" src/scripts/prepare_load_traces.py

jobs() {
  for acct in average marginal; do
    for onu in 1 0.2; do
      for seed in 7 8 9; do
        for hh in 100x20 40x50 10x200 1x2000; do
          echo "conversation $acct $onu ${hh%x*} ${hh#*x} $seed"
        done
        echo "all $acct $onu 40 50 $seed"
      done
    done
  done
}

jobs | xargs -P "${1:-10}" -L 1 bash -c '
  trace=$0 acct=$1 onu=$2 hh=$3 pd=$4 seed=$5
  name='"$OUT"'/sim_trace_${trace}_${acct}_onu${onu}_hh${hh}x${pd}_seed${seed}
  [ -f "$name.json" ] && exit 0
  '"$PY"' src/scripts/sim_piggyback.py --load-trace "$trace" --accounting "$acct" --onu-scale "$onu" \
      --households "$hh" --per-day "$pd" --seed "$seed" --peak-loads 2,4,8,16,32 \
      --policies stepwise,skip_onu,static,schedule,piggyback,oracle \
      --report window --rate-alpha 0.3 --no-hourly --out "$name.csv" > "$name.txt" 2>&1
  echo "done $name"'
