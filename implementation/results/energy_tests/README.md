# Energy tests: raw outputs and logs

Everything produced by the three-tier energy tests, kept verbatim. See
`../../../energy_tests.md` for methodology and results.

| File | What |
|---|---|
| `olt_sweep_run1.txt` / `.log` | output of the first full OLT sweep (Qwen2.5-7B fp8, one L4). Prefill figures in this run are unreliable (counter quantization, see energy_tests.md §3.4); decode is valid. |
| `olt_sweep_run2.txt` / `.log` | output of the second OLT sweep, with prefill repeated to >= 3 s. The reference run. |
| `gpu_energy_qwen2.5-7b-instruct_l4x1_run1.json` | machine-written results of run 1 |
| `gpu_energy_qwen2.5-7b-instruct_l4x1_run2.json` | machine-written results of run 2 |
| `collect_smoke_n20.txt` / `.log` | output of the 20-question smoke test of answer collection (all three tiers) |
| `gsm8k_zeroshot_user-onu-olt_n20_smoke.raw.jsonl` | its answers: one record per (tier, question), every token logprob kept |
| `collect_full_n1319_<UTC>.log` | output of the full collection, 1,319 questions per tier |
| `gsm8k_zeroshot_user-onu-olt_n1319_<UTC>.raw.jsonl` | its answers, same schema |
| `three_tiers_measured.html` | the results page (charts, hover tooltips, data tables), generated from the files above by `src/scripts/make_energy_artifact.py`; regenerate it after any new measurement |
| `collect_onu_q4km_n1319_<UTC>.log` / `.txt` | output of the ONU re-collection at Q4_K_M, after the ONU's hardware became the Orin Nano Super |
| `gsm8k_zeroshot_onu_n1319_<UTC>.raw.jsonl` | its answers; supersede the ONU rows of the 3-tier file (which were AWQ) |
| `sim_piggyback_system_<UTC>.csv` / `.json` / `.txt` | piggyback simulator (`src/scripts/sim_piggyback.py`) on the measured data, OLT at the whole-system boundary: one CSV row per (peak load, beta, policy); the JSON adds each policy's accuracy–energy frontier, per-hour routing and the inputs used; the `.txt` is the printed output. The reference run (energy_tests.md §8). |
| `sim_piggyback_system_seed8.*`, `_seed9.*`, `sim_piggyback_system_onu0.2_seed8.*`, `_seed9.*` | the reference run and the 0.2× ONU run repeated with two more random seeds, to measure noise |
| `sim_piggyback_gpu_<UTC>.*` | the same with the OLT at the GPU card only, for comparison |
| `sim_piggyback_system_onu<scale>_<UTC>.*` | the same with the ONU's energy scaled by 0.05, 0.1, 0.2 or 0.5: where a live signal starts to pay |
| `sim_piggyback_cpu_prefill.csv`, `_npu_prefill.csv`, `_delta0.1.csv` | an earlier version of the simulator over the older 5-shot trace (Gemma-2-9B OLT, estimated batch curve). Superseded; kept as a record. |

Each log exists twice. The `.log` is the raw terminal output, colour codes and
spinner redraws included; the repo ignores `*.log` everywhere else, but
`.gitignore` makes an exception for this folder, so these are committed. The
`.txt` is the same output cleaned of those redraws, small and readable.

From 2026-09-11 on, the Modal scripts write here directly with a UTC timestamp
in every file name, so no run overwrites another.
