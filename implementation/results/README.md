# Results

Everything the thesis's numbers come from. Nothing here is edited by hand: each
file is written by the script named beside it (paths from `implementation/`).

## `measurements/`: the raw data (measured once, on rented GPUs)

| File | What | Written by |
|---|---|---|
| `gpu_energy_qwen2.5-7b-instruct_l4x1_run2.json` | **The OLT's energy**: Qwen2.5-7B (fp8) on one L4, energy per prompt token and per generated token at batch 1, 2, 4 … 64, gross and net of idle. The reference run | `src/measure/measure_gpu_energy.py` |
| `gpu_energy_qwen2.5-7b-instruct_l4x1_run1.json` | the same sweep, run before the prefill fix: its decode figures replicate run 2 within 1.4%; its prefill figures are unreliable (energy_tests.md §3.4) | same |
| `olt_sweep_run{1,2}.log` / `.txt` | the two sweeps' terminal output (`.log` raw, `.txt` cleaned) | same |
| `gsm8k_zeroshot_user-onu-olt_n1319_<UTC>.raw.jsonl` | **Every tier's answer** to all 1,319 GSM8K test questions, every token's logprob kept. Its phone and OLT rows are used | `src/measure/collect_answers.py` |
| `gsm8k_zeroshot_onu_n1319_<UTC>.raw.jsonl` | the ONU's answers again, in Q4_K_M (the format its energy source measured). They replace the ONU rows of the file above | same, `TIERS=onu` |
| `gsm8k_zeroshot_user-onu-olt_n20_smoke.raw.jsonl` | the 20-question smoke test run before the full collection | same, `LIMIT=20` |
| `collect_*.log` / `.txt` | the collections' terminal output | same |

## `tier_energy.md`: every tier-level number

The OLT's batch curve, the replication, the phone's and the ONU's cost per query,
the OLT's cost per query at each energy boundary, the batch from which the OLT is
cheaper, the marginal cost, and the answers table (energy_tests.md §3, §4, §7,
§8.3). Written by `src/analyze/tier_energy.py` from `measurements/`.

## `study/`: the case study (energy_tests.md §8.6)

| File | What | Written by |
|---|---|---|
| `study_<scenario>_seed<n>.csv` | one row per (OLT peak load, RecServe beta, policy): accuracy, J per query, latency, PON traffic, where queries ended up | `src/simulate/run_study.sh` → `simulate.py` |
| `study_<scenario>_seed<n>.json` | the same, plus each policy's accuracy–energy frontier, the run's settings and its surge events | same |
| `study_<scenario>_seed<n>.txt` | the run's printed output | same |
| `SUMMARY.md` | **the case study's tables**: savings at equal accuracy, mean over seeds 7, 8, 9 | `src/analyze/summarize_study.py` |

Scenarios (each run three times, seeds 7, 8 and 9):

- `main_surge1`: BurstGPT's conversation traffic as recorded (predictable).
- `main_surge1.5` … `main_surge5`: the same with unforeseen surges and dips of that size (unpredictable).
- `alltraffic`: BurstGPT's API-inclusive traffic (real bursts).
- Sensitivity runs, each changing one setting at surge factors 1 and 3:
  - `average_`: average accounting;
  - `onu0.5_`, `onu0.2_`: a 2× or 5× cheaper ONU;
  - `olt1.07_`, `olt1.75_`: the OLT's energy ×1.07 (a site PUE of 1.65) or ×1.75 (about where a busy OLT stops beating the phone);
  - `flat_`: the households' queries flat over the day, independent of the OLT's load;
  - `hh<households>x<queries a day>_`: other household sizes;
  - `perhousehold_`: question statistics learned per household.

## `adhoc/` (not committed)

Default output of a one-off `simulate.py` run without `--out`.
