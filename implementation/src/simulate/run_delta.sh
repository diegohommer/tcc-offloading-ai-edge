#!/usr/bin/env bash
# The margin delta varied alone: the main scenario at 10,000 homes, seeds 7-9,
# RecServe and broadcast, into results/delta/. delta = 0 is the study itself
# (results/study/parts/study_main_seed*_n10000).
cd "$(dirname "$0")/../.."                       # implementation/
for delta in 0.05 0.1 0.2 0.3; do for seed in 7 8 9; do echo "$delta $seed"; done; done |
  nice -n 19 xargs -P 5 -L 1 bash -c '
    out=results/delta/delta$0_seed$1
    [ -f "$out.json" ] && exit 0
    .venv/bin/python src/simulate/simulate.py --config config/study.yaml --subscribers 10000 \
        --seed "$1" --policies recserve,broadcast --delta "$0" --out "$out.csv" > "$out.txt" 2>&1 \
      && echo "done $out" || echo "FAILED $out"'
