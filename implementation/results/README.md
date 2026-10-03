# Results

Every file here is written by a script; none is edited by hand. Paths from
`implementation/`.

## `measurements/`

Measured once, on rented GPUs (`src/measure/`).

| File | What |
|---|---|
| `gpu_energy_qwen2.5-7b-instruct_l4x1_run3.json` | the OLT's energy per token at batch 1 to 64, gross and net of idle: the reference sweep |
| `gpu_energy_..._run1.json`, `..._run2.json` | two earlier sweeps, kept as the replication |
| `olt_sweep_run*.log` / `.txt` | the sweeps' terminal output, raw and cleaned |
| `gsm8k_zeroshot_user-onu-olt_n1319_<UTC>.raw.jsonl` | every tier's answer to the 1,319 GSM8K questions, every token's logprob; its user and OLT rows are used |
| `gsm8k_zeroshot_onu_n1319_<UTC>.raw.jsonl` | the ONU's answers in Q4_K_M, its energy source's format |
| `collect_*.log` / `.txt` | the collections' terminal output |

## `tier_energy.md`

The tier-level tables: the OLT's batch curve, the replication, each tier's cost per query,
the boundary conversion, the crossovers and the answers. Written by
`src/analyze/tier_energy.py`.

## `study/`

The case study, written by `src/simulate/run_study.sh`: one CSV, JSON and printout per
scenario and seed, and `SUMMARY.md` from `src/analyze/summarize_study.py`.

## `adhoc/` (not committed)

Output of one-off `simulate.py` runs.
