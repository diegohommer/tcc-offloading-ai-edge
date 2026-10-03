# Energy-aware LLM offloading over a passive optical network

Undergraduate thesis (UFRGS, Computer Science). Advisor Prof. Dr. Gabriel Luca Nazar,
co-advisor Prof. Dr. Dennis Giovani Balreira.

## The question

A household's LLM query can be answered in three places along a fibre network: on the
user's device (a 1B model), on the home's ONU with an AI accelerator (a 1.5B model), or
on the operator's OLT, a GPU shared by every home on the PON (a 7B model).

RecServe climbs this ladder one tier at a time. A tier answers, and passes the query up
only when it is not confident. The OLT batches many homes' queries together, so the
busier it is, the cheaper each query gets, and at busy hours it can be cheaper than the
tiers below it. This work keeps RecServe's tiers and its escalation test and adds a
choice of *where* a query goes, by energy. It asks how much that saves at the same
accuracy, and where the tiers should learn the OLT's cost from: nowhere (RecServe), a
table set in advance, or a figure the OLT broadcasts on the PON.

```
phone (1B) ──> ONU (1.5B) ──> OLT (7B, batched GPU)
    ^              ^                 │
    └──────────────┴── broadcast: "a query costs X J here right now"
```

## How a query is simulated

1. **Households send queries** in conversations whose shape comes from BurstGPT's
   conversation log, at a rate set by the population (`households.py`).
2. **The query climbs the cascade** (`cascade.py`). Every tier's answer to every GSM8K
   question was recorded once, so the simulator replays answers instead of running
   models. A tier escalates when its answer's confidence falls below the beta-quantile
   of its recent confidences, as RecServe does.
3. **Energy-aware policies also decide where it goes** (`routing.py`): run here,
   forward to a cheaper tier above, or skip a tier on escalation, by the lowest expected
   energy to an answer.
4. **The OLT batches continuously** (`olt_server.py`). A query joins as soon as one of
   the 64 slots is free and leaves when its answer is done. Speed and energy follow the
   batch moment by moment, read off the measured curve.
5. **The answer travels back down** with the energy rates of the tiers it crossed, and
   the household learns from them.

The policies differ only in where they get the OLT's cost from:

| Policy | The OLT's cost | Role |
|---|---|---|
| `recserve` | not used: always one tier up | baseline |
| `recserve_no_onu` | not used: phone, then OLT | control: is a saving just dropping the ONU? |
| `static_day` | one rate for the day, observed over a month of RecServe | configuration |
| `static_hour` | one rate per hour, weekday or weekend | timetable |
| `broadcast` | the OLT's report over its last 5 minutes, sent on the PON every 10 s | the proposal |
| `oracle` | the same mean with no broadcast delay | reference |
| `piggyback` | the same report, heard only on the household's own answers | ablation |
| `stale_low`, `stale_high` | `static_day` observed on a population 4x smaller or larger | stale configuration |

## Layout

```
energy_tests.md            methods, measurements, decisions and their sources
thesis/latex/              the thesis (synced with Overleaf, see CLAUDE.md)
implementation/
  config/                  energy_sources.yaml (published figures), simulation.yaml, study.yaml
  data/load_traces/        conversation shapes measured in BurstGPT [1] (prepare_load_traces.py)
  src/measure/             runs on Modal GPUs: the OLT's energy curve, every tier's answers
  src/energy/three_tier.py the energy model: tier rates, the OLT curve, the boundary conversion
  src/simulate/            the simulator (entry point simulate.py) and run_study.sh
  src/analyze/             tables and checks
  tests/                   pytest, one file per module
  results/
    measurements/          the OLT's energy sweeps (run3 is the reference) and every tier's
                           1,319 GSM8K answers with their logprobs, with the terminal logs
    tier_energy.md         tier-level tables, written by src/analyze/tier_energy.py
    study/                 the case study's runs and SUMMARY.md, written by run_study.sh
```

Every file under `results/` is written by a script, never by hand.

## Running

```bash
cd implementation
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest                                    # tests, ~2 s
.venv/bin/python src/simulate/simulate.py --config config/study.yaml \
    --subscribers 10000 --policies recserve,static_hour,broadcast   # one run
bash src/simulate/run_study.sh 6                              # the case study
.venv/bin/python src/analyze/summarize_study.py               # its tables
.venv/bin/python src/analyze/tier_energy.py                   # tier tables
```

Every setting is explained in `config/simulation.yaml` and can be overridden by a flag
of the same name. One-off runs write to `results/adhoc/`.

The measurements cost GPU time and are already in `results/measurements/`. To redo
them: `modal run src/measure/measure_gpu_energy.py` and `modal run
src/measure/collect_answers.py`.

The thesis builds with `latexmk -pdf tcc.tex` in `thesis/latex/` (class `infufrgs`,
options `[cic,dipl,english]`, ABNT author-date citations).

[1] Wang et al. *BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems.*
KDD 2025. https://github.com/HPMLL/BurstGPT (CC BY 4.0).
