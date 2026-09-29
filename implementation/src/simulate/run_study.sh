#!/usr/bin/env bash
# The case study (energy_tests.md §8.6): all 78 runs behind the thesis's simulation
# results, config/study.yaml. RecServe, the static tables (static_day, static_hour),
# the broadcast policy (the OLT's cost broadcast on the PON), the oracle and the
# controls, on predictable traffic (BurstGPT's conversation log as recorded) and
# unpredictable traffic (the same with unforeseen surges and dips).
#   main          surge factor 1 (predictable), 1.5, 2, 3, 5               x 3 seeds = 15 runs
#   sensitivity   factors 1 and 3, each with one change: average accounting; a 2x or
#                 5x cheaper ONU; the OLT's energy x1.07 (Latin America's PUE, 1.65)
#                 or x1.75 (about where a busy OLT stops beating the phone); the
#                 households' queries flat over the day instead of following the
#                 OLT's load; 1x2000, 10x200 or 100x20 households; statistics
#                 learned per household                                     x 3 seeds = 60 runs
#   real bursts   BurstGPT's all traffic (API included), as recorded         x 3 seeds = 3 runs
# Runs already finished are skipped, so the script can be restarted. While it runs,
# nothing suspends the laptop: neither idle time (GNOME's and logind's) nor closing
# the lid.
#   bash src/simulate/run_study.sh [parallel jobs] [cores]   # e.g. 6 jobs pinned to cores 4-9
# Jobs run at the lowest CPU priority; with [cores] they are also confined to those
# cores (taskset), so the rest of the machine stays free. On this laptop (Core 5
# 120U) cores 0-3 are the performance threads and 4-11 the efficiency cores; the
# efficiency cores run cooler. A thermal guard pauses every run when the CPU package
# reaches TEMP_PAUSE degrees C (default 85) and resumes them at TEMP_RESUME (70).
# ~2-3 h at 6 jobs on performance cores, longer on efficiency cores.
# Then: python src/analyze/summarize_study.py -> results/study/SUMMARY.md
set -euo pipefail
cd "$(dirname "$0")/../.."                        # implementation/
OUT=results/study
mkdir -p "$OUT"
PY=${PYTHON:-.venv/bin/python}
[ -f data/load_traces/burstgpt_hourly.csv ] || "$PY" src/simulate/prepare_load_traces.py
INHIBIT=()
command -v systemd-inhibit > /dev/null && INHIBIT=(systemd-inhibit --what=idle:sleep:handle-lid-switch \
  --mode=block --who="tcc study" --why="simulation runs")
command -v gnome-session-inhibit > /dev/null && [ -n "${DBUS_SESSION_BUS_ADDRESS:-}" ] && \
  INHIBIT=(gnome-session-inhibit --inhibit suspend:idle --reason "tcc study" "${INHIBIT[@]}")

# Thermal guard: pause every run (SIGSTOP) when the CPU package is too hot, resume
# (SIGCONT) once it has cooled down.
TEMP_PAUSE=${TEMP_PAUSE:-85}
TEMP_RESUME=${TEMP_RESUME:-70}
ZONE=$(grep -l x86_pkg_temp /sys/class/thermal/thermal_zone*/type 2>/dev/null | head -1 || true)
# Only the Python simulations: the inhibitors' command lines hold the same text.
SIMS='^[^ ]*python src/simulate/simulate[.]py --config config/study'
guard() {
  local paused=0 t
  while sleep 10; do
    t=$(( $(cat "${ZONE%/type}/temp") / 1000 ))
    if [ "$paused" = 0 ] && [ "$t" -ge "$TEMP_PAUSE" ]; then
      pkill -STOP -f "$SIMS" || true
      paused=1; echo "[$(date +%H:%M:%S)] ${t}C: runs paused" >&2
    elif [ "$paused" = 1 ] && [ "$t" -le "$TEMP_RESUME" ]; then
      pkill -CONT -f "$SIMS" || true
      paused=0; echo "[$(date +%H:%M:%S)] ${t}C: runs resumed" >&2
    fi
  done
}
if [ -n "$ZONE" ]; then
  guard & GUARD=$!
  trap 'kill $GUARD 2>/dev/null; pkill -CONT -f "$SIMS" || true' EXIT
fi

jobs() {
  for seed in 7 8 9; do
    for f in 1 1.5 2 3 5; do echo "main_surge$f $seed --surge-factor $f"; done
    for f in 1 3; do
      echo "average_surge$f $seed --surge-factor $f --accounting average"
      echo "onu0.5_surge$f $seed --surge-factor $f --onu-scale 0.5"
      echo "onu0.2_surge$f $seed --surge-factor $f --onu-scale 0.2"
      echo "olt1.07_surge$f $seed --surge-factor $f --olt-scale 1.0714"
      echo "olt1.75_surge$f $seed --surge-factor $f --olt-scale 1.75"
      echo "flat_surge$f $seed --surge-factor $f --household-shape flat"
      echo "hh1x2000_surge$f $seed --surge-factor $f --households 1 --per-day 2000"
      echo "hh10x200_surge$f $seed --surge-factor $f --households 10 --per-day 200"
      echo "hh100x20_surge$f $seed --surge-factor $f --households 100 --per-day 20"
      echo "perhousehold_surge$f $seed --surge-factor $f --no-shared-stats"
    done
    echo "alltraffic $seed --load-trace all"
  done
}

PIN=()
[ -n "${2:-}" ] && PIN=(taskset -c "$2")
jobs | "${INHIBIT[@]}" "${PIN[@]}" nice -n 19 xargs -P "${1:-6}" -L 1 bash -c '
  tag=$0 seed=$1; shift
  name='"$OUT"'/study_${tag}_seed${seed}
  [ -f "$name.json" ] && exit 0
  '"$PY"' src/simulate/simulate.py --config config/study.yaml --seed "$seed" "$@" \
      --out "$name.csv" > "$name.txt" 2>&1
  echo "done $name"'
