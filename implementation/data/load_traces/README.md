# Load traces

Hourly request counts derived from two public LLM-serving traces, used as the
OLT's load in the piggyback simulation (`sim_piggyback.py --load-trace`,
energy_tests.md §8.5). Written by `src/scripts/prepare_load_traces.py`, which
downloads the raw traces into `implementation/.cache/load_traces/` (ignored by
git) and keeps only these aggregates.

| File | What |
|---|---|
| `burstgpt_hourly.csv` | BurstGPT v1.1, `BurstGPT_1.csv` (61 days, 1.43 M requests): requests per hour since the trace's first midnight, for `conversation` (Conversation log), `api` (API log) and `all` |
| `azure2024_conv_hourly.csv` | Azure LLM inference trace 2024, conversation service: requests per UTC hour, the 6 whole days from 2024-05-12 |
| `drift_fit.json` | per trace: daily shape, day-to-day variation, and the drift around a time-of-day schedule (log-sd net of Poisson noise, correlation time), in-sample and with the schedule fitted on the first 30 days |

Sources, both CC BY 4.0:

- Wang et al. *BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving
  Systems.* KDD 2025. https://github.com/HPMLL/BurstGPT
- Stojkovic et al. *DynamoLLM: Designing LLM Inference Clusters for Performance
  and Energy Efficiency.* HPCA 2025. Azure LLM Inference Dataset 2024,
  https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
