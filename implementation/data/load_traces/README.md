# Load traces

`burstgpt_sessions.json` holds the shape of a conversation measured in BurstGPT's
conversation log (`BurstGPT_3.csv`): how many messages a burst holds, the pauses between
them, and how many bursts open in each hour of a weekday and of a weekend day. The
household generator (`src/simulate/households.py`) resamples it.

Written by `src/simulate/prepare_load_traces.py`, which downloads the raw trace once into
`implementation/.cache/load_traces/` (ignored by git).

Source, CC BY 4.0: Wang et al. *BurstGPT: A Real-world Workload Dataset to Optimize LLM
Serving Systems.* KDD 2025. https://github.com/HPMLL/BurstGPT
