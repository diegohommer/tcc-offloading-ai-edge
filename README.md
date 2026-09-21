# TCC: energy-aware LLM offloading over a passive optical network

Undergraduate thesis (UFRGS, Computer Science), advisor Prof. Dr. Gabriel Luca
Nazar, co-advisor Prof. Dr. Dennis Giovani Balreira.

## The question in one paragraph

A home's LLM queries can be answered at three places along a fibre (PON) network:
the user's **device** (a light device running a 1B model, priced as a phone), the
home's **ONU** with an AI accelerator (a 1.5B model on a Hailo-10H NPU), or the
operator's **OLT** in the central office, a GPU shared by many homes (a 7B model). **RecServe** is a published method that climbs this ladder one step
at a time: a tier answers, and passes the query up only if it isn't confident, so
queries are answered as low as possible and the shared links carry little.
The OLT batches many homes' queries together, so the busier it is, the cheaper
each query becomes, and at busy hours it can undercut the tiers below it. This
thesis keeps RecServe's three tiers and its escalation rule, and adds a **dynamic**
choice of where a query goes: **if the tiers knew what the OLT costs right now,
how much energy could they save, at the same accuracy, and at what cost in latency
and PON traffic?** And where should that knowledge come from: nowhere (RecServe),
a table set in advance (per day or per hour), or a live figure the OLT
**broadcasts** on the PON to every ONU?

```
   phone (1B)  ──>  ONU (1.5B)  ──>  OLT (7B, batched GPU)
       ^               ^                  │
       └───────────────┴── broadcast: "a query costs X J here right now"
```

## How the code is organised: four stages

```
 1. MEASURE (once, on rented GPUs)        2. ENERGY MODEL                 3. SIMULATE                      4. ANALYZE
 src/measure/                             src/energy/three_tier.py        src/simulate/                    src/analyze/
   measure_gpu_energy.py ── OLT J/token ─┐  tier prices per query,         replays the recorded answers     tier_energy.py  → results/tier_energy.md
   collect_answers.py ──── every answer ─┼─ OLT curve vs batch size, ───>  under each policy, on real     summarize_study.py → results/study/SUMMARY.md
 config/energy_sources.yaml ─ phone, ONU ┘  boundary factors              OLT load (BurstGPT)             check_*.py (sanity checks)
   (published measurements)                                               → results/study/
```

1. **Measure.** Two Modal scripts measured what we can't take from the literature:
   the OLT's energy per token at every batch size, and every tier's answer to
   every GSM8K question. They cost money and are already done; the results are in
   `implementation/results/measurements/`.
2. **Energy model.** One module turns those measurements and two published figures
   (phone, ONU) into joules per query, on one common energy boundary.
3. **Simulate.** Households' queries flow through the cascade over a month of real
   traffic. Every answer is replayed from the recording, so no model runs. Each
   policy is scored on accuracy and energy.
4. **Analyze.** Scripts turn the measurements and the simulation runs into the
   thesis tables.

## The policies you can run

All policies use RecServe's own test to decide *whether* a query escalates. The
energy-aware ones (every row below the first two) also decide *where* it goes, by
the lowest expected energy to an answer, and differ **only** in where they get the
OLT's cost from.

| Policy (`--policies`) | Where the OLT's cost comes from | Role |
|---|---|---|
| `recserve` | nowhere: always one tier up | the baseline, RecServe as published |
| `recserve_no_onu` | nowhere: phone → OLT only | control only (the proposal keeps the ONU): is a saving just dropping it? |
| `static_day` | one fixed rate for the whole day, learned from past days | a configuration set once |
| `static_hour` | one rate per hour of day (weekday or weekend), learned from past days | a timetable |
| `broadcast` | the OLT's own average over its last 5 minutes, broadcast on the PON every 10 s | **the proposal** |
| `oracle` | the true current cost | the most any live signal could do |
| `piggyback` | the OLT's 5-minute average, heard only on the home's own answers | ablation: why broadcast |
| `stale_low`, `stale_high` | `static_day` set at ¼ or 4× the real load | a configuration gone stale |

## Folder and file guide

```
README.md                      this file
energy_tests.md                the methodology and results log: every decision, its source, every result
                               (the thesis's methodology and results chapters are written from it)
tcc_politica_energia_desenho.md   the energy-policy design notes (Portuguese)
thesis/
  latex/                       the thesis itself (tcc.tex, the UFRGS class, bibliography); see its README
  papers/                      local copies of cited papers (not committed)
implementation/
  requirements.txt             Python packages
  config/
    energy_sources.yaml        the phone's and the ONU's published energy figures and the OLT's
                               boundary factors, each with its exact source (paper, table)
    simulation.yaml            every simulator setting, with what it does and where its value comes from
    study.yaml                 the case study's settings (extends simulation.yaml)
  data/load_traces/            the OLT's load: BurstGPT's requests per hour (61 days) and how much
                               real traffic drifts around a timetable; see its README
  src/
    measure/                   STAGE 1: runs on Modal GPUs
      measure_gpu_energy.py    the OLT's energy vs batch size (NVML energy counter, vLLM, one L4)
      collect_answers.py       every tier answers every GSM8K test question; logprobs kept
      gsm8k.py                 the GSM8K task: prompt template and answer scoring
    energy/                    STAGE 2
      three_tier.py            the energy model: phone and ONU prices, the OLT's curve (average and
                               marginal), the boundary conversion, loading the recorded answers
    simulate/                  STAGE 3: the simulator
      simulate.py              the entry point: reads the settings, runs every policy at every load and
                               beta, writes the results
      olt_load.py              the OLT's load over time: BurstGPT replayed, unforeseen surges, and the
                               static_day / static_hour tables learned from past days
      olt_energy.py            what a query truly costs (average or marginal accounting), and the OLT's
                               5-minute report that the broadcast and piggyback policies hear
      routing.py               the decision rules: RecServe's confidence window and the energy-aware
                               "cheapest expected route" rule
      cascade.py               one run: one policy serving the whole query stream
      frontier.py              comparing policies at equal accuracy
      run_study.sh             runs the whole case study (78 runs), at low priority, optionally on chosen cores
      prepare_load_traces.py   builds data/load_traces/ from the public BurstGPT trace
    analyze/                   STAGE 4
      tier_energy.py           every tier-level table (costs, crossovers, marginal cost, answers)
      summarize_study.py       the case study's tables
      check_confidence.py      checks the recorded answers' confidence scores before routing on them
      check_beta_windows.py    checks that skipping tiers does not disturb RecServe's thresholds
  results/                     every output; see its README
    measurements/              stage 1's data and logs
    tier_energy.md             tier-level tables
    study/                     the case study's 78 runs and SUMMARY.md
```

## Running it

Setup, once:

```bash
cd implementation
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Run any set of policies (the case study's 40 households over BurstGPT's month of
traffic, OLT peak load 8, every RecServe beta; about 1–2 min):

```bash
.venv/bin/python src/simulate/simulate.py --config config/study.yaml \
    --policies recserve,static_day,static_hour,broadcast,oracle --peak-loads 8
```

Every setting can be changed the same way (`--help` lists them; `config/simulation.yaml`
explains each one). Results go to `results/adhoc/`.

Reproduce every number in the thesis:

| Step | Command (from `implementation/`) | Time | Writes |
|---|---|---|---|
| tier tables | `.venv/bin/python src/analyze/tier_energy.py` | ~5 s | `results/tier_energy.md` |
| answer checks | `.venv/bin/python src/analyze/check_confidence.py results/measurements/<answers>.raw.jsonl` | ~5 s | printed |
| the case study | `bash src/simulate/run_study.sh 6 0-5` (6 jobs pinned to cores 0–5, lowest priority; keep the laptop awake) | ~2–3 h | `results/study/` |
| its tables | `.venv/bin/python src/analyze/summarize_study.py` | ~10 s | `results/study/SUMMARY.md` |
| threshold check | `.venv/bin/python src/analyze/check_beta_windows.py` | a few min | printed |
| load traces | `.venv/bin/python src/simulate/prepare_load_traces.py --azure` (streams the 1.1 GB Azure trace once) | minutes | `data/load_traces/` |
| measurements | `.venv/bin/modal run src/measure/measure_gpu_energy.py`, `... collect_answers.py` (a Modal account, paid GPU time) | ~15–20 min each | `results/measurements/` |

## Where each thesis number comes from

| Number | Table | Computed by | From |
|---|---|---|---|
| OLT energy per token at batch 1–64 (falls 51×); two runs agree within 1.4% | `results/tier_energy.md` §1–2 | `analyze/tier_energy.py` | `measurements/gpu_energy_*_run{1,2}.json` |
| Phone 15.7 J, ONU 243 J (91 J marginal) per query | `tier_energy.md` §3 | `energy/three_tier.py: published_rates` | `config/energy_sources.yaml` + answer lengths |
| OLT per query at each boundary (× 2.47 whole system) | `tier_energy.md` §4 | `three_tier.py: olt_factor, boundary` | Google's shares, PUE 1.54 |
| OLT cheaper than the ONU from batch ≈ 5; never cheaper than the phone | `tier_energy.md` §5 | `tier_energy.py: crossing` | §3 and §4 |
| One more query: ~9 J on a busy OLT, ~820 J on an idle one | `tier_energy.md` §6 | `three_tier.py: OltCurve.marginal_rates` | the batch curve |
| Accuracy 0.47 / 0.69 / 0.92; confidence separates right from wrong | `tier_energy.md` §7 | `tier_energy.py`, `check_confidence.py` | `measurements/gsm8k_*.raw.jsonl` |
| Traffic shape (46× day swing, drift σ ≈ 0.6) | `data/load_traces/drift_fit.json` | `simulate/prepare_load_traces.py` | BurstGPT |
| Every policy's energy at equal accuracy; savings over RecServe and over the static tables; surges; latency, PON traffic and accuracy delivered; the sensitivities | `results/study/SUMMARY.md` | `analyze/summarize_study.py` | `results/study/*.json` ← `simulate/run_study.sh` |
| RecServe's thresholds keep their meaning when tiers are skipped | printed | `analyze/check_beta_windows.py` | the simulator |

## History

On 2026-09-19 the repo was cut down to what the thesis uses. Removed: the first
harness (RecServe's sentiment-classification code and a 4-tier classifier cascade),
an Ollama-based collection, a laptop energy measurement (Intel RAPL), an early
energy-policy sweep, the results web page, and the simulation runs of the
exploration that led to the case study (energy_tests.md §8.2–8.5). All of it is
in git at the tag **`pre-cleanup`**:

```bash
git checkout pre-cleanup            # the repo as it was; `git checkout main` to come back
```
