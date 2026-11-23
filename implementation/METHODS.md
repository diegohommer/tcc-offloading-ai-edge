# Methods and measurements

What was measured, how, and where each number comes from. The thesis's methodology
chapter gives the full argument; results are in `results/study/SUMMARY.md`.

## 1. The question

The case study is a three-tier cascade, **phone → ONU → OLT**, following the tiers of
Pakpahan and Hwang's PON architecture [2], with RecServe's quantile test deciding whether a
query escalates [1].

The phone and the ONU serve one household and run one query at a time, so their cost per
query is fixed. The OLT serves every home on the PON, so its batch grows with load and its
cost per query falls. When that cost drops below a lower tier's, the cheapest place to
answer depends on the moment, and a tier that knows the OLT's current cost can route on it.

## 2. The tiers

| Tier | Model | Precision | Hardware | Energy from |
|---|---|---|---|---|
| Phone | Llama-3.2-1B-Instruct | GGUF Q4_K_M | Snapdragon 8 Elite Gen 5 (OnePlus 15) | Cai et al. [9] |
| ONU | Qwen2.5-1.5B-Instruct | GGUF Q4_K_M | Raspberry Pi 5 + Hailo-10H NPU | Cloud to Edge [12], checked against [19] |
| OLT | Qwen2.5-7B-Instruct | fp8 | NVIDIA L4 | measured here (§3) |

- **Model sizes follow Pakpahan and Hwang** where they give one: tiny models on user
  devices, 7–13B in the fog tier where the OLT sits [2]. The ONU's 1.5B is about the
  largest an AI home gateway runs comfortably (§5).
- **Each tier's answers and energy come from the same model at the same precision**, since
  quantisation changes the answers and so the confidence the cascade routes on.

## 3. OLT energy sweep

### 3.1 Method

| Decision | Choice | Following |
|---|---|---|
| Instrument | NVML's cumulative energy counter, read before and after each pass | [3, 4]; power sampling misses part of the runtime [6] |
| Boundary | The GPU card; converted to the whole system in §7 | [3] |
| Engine | vLLM 0.21.0, offline `generate`, prefix caching off | [3]; see §3.4 |
| Unit | J per prompt token and per generated token, separately | [5]; the phases batch differently [7] |
| Phase split | A 1-token pass (prefill) and a full pass; decode is the difference | [5] |
| Batch | Static batches of 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 | [5, 7] |
| Output length | 300 tokens per sequence (`ignore_eos`), so a batch keeps its size | [8] |
| Repeats | One warm-up, 5 repeats, median; short passes repeated to span ≥ 3 s | [6] |
| Idle | 10 s with the model loaded; rates reported gross and net of idle | |

The reference sweep (`onecard`, `measure_gpu_energy_one_card.py`) runs the static sweep and
the continuous-batching check (§3.5) back to back on the same card, so both share one idle
power. Two separate containers had once landed on cards whose idle differed by 8 W.

### 3.2 Results

Idle 32.1 W. Full table in `results/tier_energy.md`.

| Batch | Power W | Generated tok/s | Prompt J/tok (net) | Generated J/tok (net) |
|---|---|---|---|---|
| 1 | 71.9 | 29 | 0.033 (0.018) | 2.464 (1.365) |
| 2 | 71.6 | 58 | 0.039 (0.022) | 1.228 (0.677) |
| 4 | 71.8 | 115 | 0.024 (0.013) | 0.618 (0.342) |
| 8 | 72.0 | 227 | 0.017 (0.010) | 0.313 (0.174) |
| 16 | 72.1 | 438 | 0.015 (0.008) | 0.160 (0.089) |
| 32 | 71.8 | 811 | 0.015 (0.008) | 0.084 (0.046) |
| 64 | 72.0 | 1,394 | 0.014 (0.008) | 0.047 (0.026) |

- **Decode energy falls 52× from batch 1 to 64 at constant power.** The L4 sits at its
  72 W limit from batch 1; decode is memory-bound, so extra sequences share the weight loads.
- **Prefill gains far less**, settling near 0.015 J per token from batch 16.
- **Replicated**: three earlier sweeps on other cards agree on decode within 2.1%.

### 3.3 Where the OLT undercuts the lower tiers

At the whole-system boundary (§7) and the OLT's mean lengths (122 prompt, 251 generated
tokens), a query costs 1,523 J alone and 33 J in a batch of 64. The OLT is cheaper than the
ONU (246 J) from a batch of about 6.4 and never cheaper than the phone (15.7 J) within 64.
Judged by the GPU card alone, as most benchmarks do, the ONU crossover would fall at 2.6.

### 3.4 Two measurement problems, fixed

- **Prefix caching zeroed prefill**: the warm-up filled vLLM's cache [11]. Now disabled.
- **The energy counter is too coarse for one short pass**: a batch-1 prefill read 6.75 J
  once and 0 J twice. Repeating the pass until it spans 3 s fixed it.

### 3.5 Continuous batching: validation, criterion fixed before the data

The sweep times static batches; the simulator's OLT batches continuously (§8.3).
`measure_gpu_energy_continuous.py` measures the same setup on vLLM's async engine with the
OLT's own GSM8K questions: fixed concurrency 1–64 (three 90 s windows each), and 15 Poisson
runs of 5 minutes (loads 1, 2, 4, 8, 16 in flight, three seeds each).

**Criterion, written 2026-10-05 before the full run:**

1. *The simulator reproduces the GPU*: replaying each Poisson run's arrivals, predicted
   energy and mean latency are within ±10% of the measured ones at every load (net of
   idle under marginal accounting, gross under average).
2. *The measurement is steady*: at each concurrency, the median lies within its three
   windows' range and the SM clock does not drop between windows.

*Amendment, same day, before the data*: the net comparison subtracts the measured card's
idle on both sides (as first written, each side used a different card's idle; that figure
is still reported).

**Outcome** (`results/continuous_batching.md`):

- *Criterion 1 is met*: net energy within −0.6% to +2.7%, gross within 0.0% to +2.4%,
  latency within −2.4% to −0.3%, every load and seed.
- *Criterion 2 is not met as written.* The energy windows agree within 0.5%, but the clock
  test fails at most concurrencies: an L4 at its power limit moves its clock continuously,
  so a single reading per window measures power management, not throttling.
- *Found after the data*: fixed-concurrency energy per token is within −1.5% to +2.9% of
  the static sweep at the same batch size.

An earlier version of the simulator charged a batch net + (n − 1) × slope, a straight
line, and failed criterion 1 at load 16 (+14%). It now charges the net energy measured at
each batch size, which is what met the criterion.

## 4. Answers

All three tiers answered GSM8K's 1,319 test questions, zero-shot, temperature 0, up to 512
tokens, each with its chat template. Every token's logprob is stored, so any confidence can
be recomputed (`src/analyze/check_confidence.py` checks them).

| Tier | Accuracy | Mean confidence | Prompt tokens | Generated tokens | Hit 512 | Welch t, exp(mean) | Welch t, exp(min) |
|---|---|---|---|---|---|---|---|
| Phone | 0.472 | 0.867 | 125.7 | 184.9 | 1.1% | +10.70 | +12.57 |
| ONU | 0.688 | 0.900 | 122.0 | 276.0 | 2.4% | +10.64 | +11.11 |
| OLT | 0.917 | 0.937 | 122.0 | 251.4 | 0.5% | +7.84 | +10.19 |

Accuracy rises tier by tier, and confidence separates right from wrong answers on every
tier. The simulator uses RecServe's exp(mean logprob).

## 5. The ONU

A Raspberry Pi 5 with a Hailo-10H NPU (40 TOPS INT4, 8 GB) running Qwen2.5-1.5B in 4-bit.
Today's ONUs have 32–315 MB of memory [18], too little for a 1.1 GB model, so this tier
stands for the AI home gateways operators have started to ship [16, 17].

- **0.88 J per generated token at 6.34 tok/s** [12], whole board at the plug, idle
  included. Tummalapalli et al. measured 0.27 J above idle on the same board [19]; 0.88 J
  minus 3.5 W of idle gives 0.33 J, so the two agree.
- **Prompt**: 200 tok/s [19], at the working power of 5.58 W, so 0.028 J per token
  (0.010 J above idle).
- **Per query** (122 prompt, 276 generated tokens): **246 J**, or **92 J** above idle.
  An answer takes about 44 s.

## 6. The phone

Cai et al. measured a OnePlus 15 with Qualcomm's power telemetry [9]: 0.016 J per prompt
token and 0.074 J per generated token on the CPU, **15.7 J** per query (126 prompt, 185
generated tokens). The telemetry covers the chip, not memory, display or radio, so the
figure is a **lower bound**. That can only widen the phone's lead, so it cannot flip a
result.

## 7. One boundary for all three tiers

Tiers priced from different sources must share one boundary, or the comparison measures
the boundaries. The common one is everything drawn to answer the query [13].

| Tier | What its source counts | Conversion |
|---|---|---|
| Phone | the chip | none (a lower bound, §6) |
| ONU | the whole board at the plug | none |
| OLT | the GPU card | × 1.586 for host and spare machines, × PUE 1.54: **× 2.44** |

Google's median prompt is 58% accelerators, 24% host CPU and memory, 10% spare machines and
8% building [13], so the server factor is (58 + 24 + 10) / 58 = 1.586. The building uses the
industry-average PUE of 1.54 [14]. The same factor applies under both accountings.
Sensitivity runs use × 1.0 (card alone), × 1.72 (Google's own accounting) and × 3.35 (a
host drawing as much as its card).

## 8. The simulator

An event-driven simulation of one OLT and its households over a 28-day calibration month
and a 28-day test month. No model runs: every answer is the one recorded in §4.

### 8.1 Traffic (`households.py`, `prepare_load_traces.py`)

- **Households**: 1,000 to 20,000, each with a Poisson(1) number of weekly LLM users.
- **Activity**: log-normal per user, mean 1, log-sd 1.5, so the busiest 10% send 59% of
  messages, as in WildChat [20].
- **Volume**: 3.6 messages per user per day (ChatGPT, July 2025 [21]), in conversations of
  2.4 messages on average.
- **Shape**: conversation starts follow WildChat's average day in each user's local time,
  weekday or weekend; length and pauses are resampled from WildChat conversations. Users
  who sent one message are dropped as likely automated.
- **Follow-ups** are sent a pause after the previous answer and carry the earlier turns as
  context.

**Drift.** A busyness factor shared by all households multiplies each hour's expected
starts: $z_k = \phi z_{k-1} + \varepsilon_k$, $F_k = e^{z_k - \sigma^2/2}$, with
$\phi = e^{-1/\tau}$ and $\varepsilon_k \sim N(0, \sigma^2(1-\phi^2))$, so $F$ has mean 1,
log-sd σ and correlation time τ (a log-Gaussian Cox process). σ and τ are measured on
BurstGPT [10] (110 days, one session = one conversation): hourly counts against their
weekday/weekend timetable, keeping hours with ≥ 20 expected starts, with
$E[(N/\lambda - 1)^2] = (e^{\sigma^2} - 1) + 1/\lambda$ removing counting noise.

| Scenario | Weekly level removed | σ | τ | One sd of the factor |
|---|---|---|---|---|
| No drift (`main`) | – | 0 | – | ×1 |
| Mild drift (`burst_week`) | yes | 0.253 | 2.64 h | ×0.75 to ×1.25 |
| Strong drift (`burst_all`) | no | 0.477 | 9.36 h | ×0.55 to ×1.44 |

Most of the strong drift is BurstGPT's growth over 110 days, recast as random nine-hour
swings, so it is a stress case rather than an estimate. The calibration month drifts too.

### 8.2 The cascade and the policy (`cascade.py`, `routing.py`)

A tier escalates when its answer's confidence is below the β-quantile of its last 1,000
confidences [1], one window per household. The phone and the ONU take their published
speeds; each household's ONU serves one query at a time.

WattLadder adds two decisions by expected energy to an answer,
$C_j = E_j + p_j\,C_{\text{next}}$ with $p_j = \beta$ and
$E_j = r^p_j P + r^g_j \bar L_j$ (published mean answer lengths):

- **On arrival**, forward to the cheapest tier above if
  $C_{k^*} < (1-\delta)(E_i + p_i C_{k^*})$.
- **On escalation**, skip to $k^*$ instead of $i+1$ if $C_{k^*} < (1-\delta)\,C_{i+1}$.

δ = 0 in the main runs; δ = 1 reproduces confidence-only escalation exactly
(`run_delta.sh` sweeps between).

### 8.3 The OLT (`olt_server.py`)

Continuous batching with 64 slots; a query that finds them all busy waits. Each running
sequence advances at the current batch's measured speed and pays its energy per token, so
a query that starts alone pays the lonely rate until others join. Tests check hand-worked
cases and that the joules charged sum to what the OLT spent. The OLT's prefill time and
network delay are left out.

### 8.4 Energy accounting

- **Marginal** (main): a batch of n draws the net-of-idle energy measured at that size,
  shared equally (§3.5). A query alone costs about 840 J, one in a batch of 64 about 18 J.
  The ONU drops its idle draw; the phone is unchanged.
- **Average** (sensitivity): each query pays its share of the gross energy [13].

What the tiers **route on** is separate (`--signal`):

- **added** (main): what one more query adds above idle,
  $r_J = b\,r_{\text{busy}} + (1-b)\,r_{\text{idle}}$, where b is the share of the last
  5 minutes the OLT was busy. $r_{\text{idle}}$ is the net batch-1 rate (≈ 840 J per
  query); $r_{\text{busy}}$ is the least-squares slope of net batch energy over batches
  1–64 (≈ 5 J per query).
- **share** (sensitivity): the mean cost per token of the OLT's recent work.

The two WattLadder strategies differ only in where $r_J$ comes from:

| Policy | Source of the OLT's cost |
|---|---|
| `broadcast` | the live report, sent on the PON every 10 s |
| `static_hour` (timetable) | the same formula over each hour of the calibration month, weekday or weekend, relearned while running itself (`calibration: self`) |
| `piggyback` | the report stamped on the household's own answers only (ablation) |

### 8.5 Comparing policies

Each policy runs at β = 0.1 … 0.9. Pareto-dominated points are dropped and energy is
interpolated linearly at the target accuracy (0.80 in the main results). Every run also
reports latency and RecServe's communication burden [1], read at the same point.

## 9. The case study

Settings in `config/study.yaml`, runs from `src/simulate/run_study.sh`, tables from
`src/analyze/summarize_study.py`, figures from `src/analyze/plot_study.py`. Three seeds
(7, 8, 9) per run.

- **Policies**: confidence-only escalation (`recserve`), the same without the ONU
  (`recserve_no_onu`), WattLadder with the timetable (`static_hour`) and the broadcast
  (`broadcast`), `piggyback`, and each tier alone (`*_alone`).
- **Scenarios**: no, mild and strong drift at 1,000–20,000 households; five sensitivity
  runs without drift up to 10,000 (OLT factor × 1.0, × 1.72, × 3.35; ONU at half energy;
  average accounting; routing on the share).
- **δ sweep** at 10,000 households: `src/simulate/run_delta.sh`.

Results: `results/study/SUMMARY.md` and Chapter 4 of the thesis.

## 10. Limitations

- **The OLT's whole-system figure is converted, not metered** (§7).
- **The phone figure is a lower bound** (§6); it can only widen the phone's lead.
- **The ONU source is thin**: one short prompt, five runs, power mode not stated [12].
- **One model per tier and one task**, GSM8K zero-shot; the phone and the ONU are
  different model families, so part of the accuracy step between them is the family.
- **Drift comes from BurstGPT**, an organisation's users rather than households.
- **Left out**: transport energy, the OLT's prefill time, network delay, and the cost of
  relaying the broadcast. The PON's downstream multicast exists [15], but the software to
  send and relay the cost would be new.

## References

1. Wu et al. *Recursive Offloading for LLM Serving in Multi-tier Networks* (RecServe). arXiv:2505.16502.
2. Pakpahan and Hwang. *Enabling Software-Defined Tiered LLM Inference Continuum on Passive Optical Network.* IEEE Access, vol. 14, 2026. doi:10.1109/ACCESS.2026.3651558.
3. Chung et al. *The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization.* arXiv:2505.06371.
4. ML.ENERGY. *Measuring GPU Energy: Best Practices.* https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/
5. Niu et al. *TokenPowerBench: Benchmarking the Power Consumption of LLM Inference.* AAAI 2026. arXiv:2512.03024.
6. Yang et al. *Part-time Power Measurements: nvidia-smi's Lack of Attention.* arXiv:2312.02741.
7. Delavande, Pierrard and Luccioni. *Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use.* arXiv:2601.22362.
8. Caravaca, Cuevas and Cuevas. *From Prompts to Power: Measuring the Energy Footprint of LLM Inference.* arXiv:2511.05597.
9. Cai et al. *Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference.* arXiv:2607.05475.
10. Wang et al. *BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems.* KDD 2025. arXiv:2401.17644.
11. vLLM documentation. *Automatic Prefix Caching.* https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
12. *Cloud to Edge: Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers.* arXiv:2604.24785.
13. Elsworth et al. (Google). *Measuring the environmental impact of delivering AI at Google Scale.* arXiv:2508.15734.
14. Uptime Institute. *Global Data Center Survey 2025* (average annual PUE 1.54).
15. Cisco. *Understand GPON Technology*; ITU-T G.984.3 and G.9807.1.
16. ZTE. *ZTE unveils AI-powered home network solutions at MWC Barcelona 2025* (4 TOPS NPU).
17. *ZTE released AI FTTR solution, empowering home network security* (6 TOPS NPU). The Register, 2026-06-29.
18. hack-gpon.org, ONT hardware pages: Huawei HN8010Ts, Huawei HG8010H, ZTE F6645P.
19. Tummalapalli et al. *LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load.* arXiv:2603.23640.
20. Zhao et al. *WildChat: 1M ChatGPT Interaction Logs in the Wild.* ICLR 2024.
21. Chatterji et al. *How People Use ChatGPT.* NBER Working Paper 34255, 2025.
