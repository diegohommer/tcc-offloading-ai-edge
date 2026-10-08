# Methods and measurements

What was measured, how, and why each choice was made. The thesis's methodology chapter is
written from this file. Simulation results go in `implementation/results/study/SUMMARY.md`
once the case study has been run.

## 1. The question

The case study is a three-tier cascade, **user → ONU → OLT**, following the tiers of
Pakpahan and Hwang's PON architecture [2], with RecServe's confidence rule deciding whether
a query escalates [1].

A user device or an ONU serves one household and runs one query at a time, so its cost per
query is fixed. The OLT serves every home on the PON, so its batch grows with load and its
cost per query falls. When that cost drops below a lower tier's, the cheapest place to
answer depends on the moment, and a tier that knows the OLT's current cost can route on it.

This needs, per tier, the energy per prompt token and per generated token, and for the OLT
both as a function of batch size.

## 2. The tiers

| Tier | Model | Precision | Hardware | Energy from |
|---|---|---|---|---|
| User | Llama-3.2-1B-Instruct | GGUF Q4_K_M | Snapdragon 8 Elite Gen 5 phone | Cai et al. [9] |
| ONU | Qwen2.5-1.5B-Instruct | GGUF Q4_K_M | Raspberry Pi 5 + Hailo-10H NPU | Cloud to Edge [12], checked against [20] |
| OLT | Qwen2.5-7B-Instruct | fp8 | NVIDIA L4 | measured here (§3) |

- **Sizes follow Pakpahan and Hwang** where they give one: tiny models on user devices and
  7–13B in the fog tier where the OLT sits [2]. The ONU's 1.5B is our choice, about the
  largest model an AI home gateway runs comfortably (§5). The OLT is at the bottom of its
  band on purpose, the configuration where batching is most likely to pay.
- **Each tier's answers and energy come from the same model at the same precision**, since
  quantisation changes the answers and so the confidence the cascade routes on.
- **Every tier answered every question** on Modal (§4), so any route can be replayed.

## 3. OLT energy sweep

### 3.1 Method

| Decision | Choice | Following |
|---|---|---|
| Instrument | NVML's cumulative energy counter, read before and after each pass | ML.ENERGY [3, 4]; power sampling misses part of the runtime on recent GPUs [6] |
| Boundary | The GPU card with its memory; converted to the whole system in §7 | ML.ENERGY [3] |
| Engine | vLLM, offline `generate` | ML.ENERGY [3] |
| Unit | J per prompt token (prefill) and per generated token (decode), separately | TokenPowerBench [5]; the phases respond to batching differently [7] |
| Phase split | Each trial runs a 1-token pass (prefill) and a full pass; decode is the difference | Phase accounting as in [5] |
| Batch | Static batches of 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 | Batch sweeps as in [5, 7] |
| Output length | 300 tokens per sequence (`ignore_eos`) | [8]; per-query costs use each tier's real answer lengths |
| Repeats | One discarded warm-up, 5 repeats, median reported | [6] |
| Short passes | The prefill pass repeats until it spans ≥ 3 s | [6]; see §3.4 |
| Prefix caching | Off | vLLM enables it by default [11]; see §3.4 |
| Idle power | 10 s with the model loaded; figures reported gross and net of idle | |

### 3.2 Results

Run 3, the reference sweep. Idle 30.5 W. Per query means 96 prompt and 300 generated
tokens; marginal is the energy one more query added between adjacent batch sizes.

| Batch | Prefill J/token (net) | Decode J/token | Tokens/s | J/query | Marginal J/query |
|---|---|---|---|---|---|
| 1 | 0.0339 (0.0196) | 2.465 | 29 | 740 | 740 |
| 2 | 0.0390 (0.0226) | 1.239 | 58 | 374 | 6.9 |
| 4 | 0.0239 (0.0137) | 0.625 | 115 | 189 | 7.1 |
| 8 | 0.0183 (0.0106) | 0.316 | 225 | 96 | 3.5 |
| 16 | 0.0155 (0.0089) | 0.161 | 433 | 50 | 3.6 |
| 32 | 0.0154 (0.0088) | 0.085 | 801 | 27 | 3.5 |
| 64 | 0.0151 (0.0087) | 0.048 | 1,367 | 16 | 4.1 |

The full table, every batch size, is in `implementation/results/tier_energy.md`.

- **Decode energy falls 51× from batch 1 to 64 at nearly constant power.** The L4 draws
  about 72 W from batch 1, and throughput almost doubles with every doubling of the batch
  up to 16, because decode is memory-bound and extra sequences reuse the same weight loads.
- **Prefill gets cheaper up to about batch 12**, then stays near 0.015 J per token.
- **One more query on a busy OLT costs about 4 J** at the card, against 16–50 J on average.
- **Replicated.** Two earlier sweeps agree with run 3 on decode within 1.2%.
- This matches the literature: energy per token falls roughly logarithmically with batch
  size, and decode benefits most [5, 7].

### 3.3 Where the OLT undercuts the lower tiers

At the whole-system boundary (§7) and each tier's real answer lengths (§4), the OLT
answers a query more cheaply than the ONU (243 J) from a batch of about 7. It never
undercuts the user device (15.7 J) within batch 64, where it still costs 34 J. Judged by
its GPU card alone, as most benchmarks do, the ONU crossover would fall at 2.6.

### 3.4 Two measurement problems, fixed

- **Prefix caching zeroed prefill.** The warm-up filled vLLM's cache, so measured trials
  skipped most of the prefill. It is now disabled.
- **The energy counter is too coarse for one short pass.** A batch-1 prefill read 6.75 J
  once and 0 J twice: the counter steps in about 0.1 s of energy. Repeating the pass until
  it spans 3 s fixed it. Decode passes last 10–14 s and were never affected.

### 3.5 Continuous batching: validation, criterion fixed before the data

The sweep times static batches, while the simulator runs the OLT with continuous batching
(§8.3). `src/measure/measure_gpu_energy_continuous.py` measures the same GPU, engine,
precision and settings the way a live OLT runs, on vLLM's async engine with the OLT's own
GSM8K questions at their recorded answer lengths:

- **Fixed concurrency**: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48 and 64 requests always in
  flight; after a 30 s warm-up, three back-to-back 90 s windows, median reported.
- **Poisson arrivals**: loads of 1, 2, 4, 8 and 16 requests in flight on average, three
  arrival seeds each, 5 minutes of arrivals; every request's arrival, finish and tokens.

**Criterion, written on 2026-10-05 before the full run:**

1. *The simulator reproduces the GPU.* Each Poisson run's measured arrivals are replayed
   through the simulator's OLT, with the static curve at the card boundary. Its predicted
   energy is within ±10% of the energy measured over the run, and its mean request latency
   (finish minus arrival) within ±10% of the measured mean, at every load. Energy is
   compared net of the run's own idle power under marginal accounting (net + (n − 1) ×
   slope per step, the study's model), and gross under average accounting.
2. *The measurement is steady.* At every fixed concurrency, the reported median lies within
   the range of its three windows, and the GPU's SM clock does not drop between windows.

*Amendment, 2026-10-05, before the full run's data:* the net comparison of criterion 1
takes both sides against the idle power of the card that was measured. As first written,
the measured side subtracted that card's idle (35.6 W in the smoke run) and the simulator
its net rates from the sweep's card (30.5 W), so the two sides subtracted different
baselines. The gross curve does not depend on the card, since the L4 runs at its power
limit, so the simulator's net rates are rebuilt from the same gross curve minus the
measured card's idle. The as-written figure is reported beside it.

Fixed-concurrency energy per token is also reported against the static sweep at the same
batch size, for the record. If criterion 1 fails at some load, that is reported as found,
and the simulator is rerun on the continuous curve. A 5-minute smoke run (concurrency 1 and
8, one load) only checked the pipeline: 2.473 J per generated token at concurrency 1 against
the sweep's 2.465, and 0.322 against 0.316 at 8.

**Outcome (full run, 2026-10-05; `results/continuous_batching.md`): neither criterion is
met as written.**

- *Criterion 1 fails at load 16 under marginal accounting.* The simulator overstates net
  energy by 13.5–14.2% there, and by 6.3–7.1% at load 8. Every other comparison passes,
  at every load and seed: gross energy within +1.0% to +2.1%, mean latency within −1.8% to
  −0.2%, net energy within +1.9% to +3.6% at loads 1–4.
- *Criterion 2 fails on its clock test at concurrency 4, 32, 48 and 64.* The three energy
  windows of every level agree within 0.73%, and the card stayed at its 72 W limit and
  76–77 °C. The clock check read one instant at each window's edge, and an L4 at its power
  limit moves its clock continuously to stay under it, so a single reading of 975–1,230 MHz
  is power management, not throttling. The clock test was the wrong instrument; the
  energy windows show the measurement was steady.

*Found after the data, and labelled as such:*

- Continuous batching costs what the static sweep says. Compared like for like (the sweep's
  decode energy plus its prefill energy at the window's mix of prompt and generated tokens),
  every concurrency from 1 to 64 is within −1.4% to +3.2%.
- The marginal failure comes from the straight-line marginal model (net + (n − 1) × slope),
  not from batching. Reading the net energy off the measured curve at each batch size
  instead, the replay is within +1.4% to +3.0% at every load and seed. The straight line
  overcharges a busy OLT, increasingly with load, so it errs against the proposal.

**The fix, and the check rerun on the same data.** The simulator now charges a batch the
net energy measured at its size (§8.4), as the remedy above provides. Rerun on the same
fifteen Poisson runs, criterion 1 is met at every load and seed: net energy within +1.4% to
+3.0%, gross within +1.0% to +2.1%, mean latency within −1.8% to −0.2%. Criterion 2 stays
not met by its clock test, as written.

## 4. Answers

All three tiers answered GSM8K's 1,319 test questions, zero-shot, at temperature 0, up to
512 tokens, with each model's chat template. Every token's logprob is stored, so any
confidence can be recomputed. `src/analyze/check_confidence.py` checks one logprob per
token, all ≤ 0, and that the stored confidence equals exp(mean logprob).

| Tier | Accuracy | Mean confidence | Prompt tokens | Generated tokens | Hit 512 tokens | Welch t, exp(mean) | Welch t, exp(min) |
|---|---|---|---|---|---|---|---|
| User | 0.472 | 0.867 | 125.7 | 184.9 | 1.1% | +10.70 | +12.57 |
| ONU | 0.688 | 0.900 | 122.0 | 276.0 | 2.4% | +10.64 | +11.11 |
| OLT | 0.917 | 0.937 | 122.0 | 251.4 | 0.5% | +7.84 | +10.19 |

- **Accuracy rises tier by tier**, so skipping a tier never lowers accuracy.
- **Confidence separates right from wrong answers on every tier.** exp(min logprob) is the
  stronger signal on all three; the simulator uses RecServe's exp(mean) by default.

## 5. The ONU

A Raspberry Pi 5 with a Hailo-10H NPU (40 TOPS INT4, 8 GB, under 5 W) running
Qwen2.5-1.5B in 4-bit. Pakpahan and Hwang only say ONUs carry "hardware accelerators" for
"quantized models" [2]. Today's ONUs have 32–315 MB of memory [18], too little for a 1.1 GB
model, so this tier stands for the AI home gateways operators have started to ship, with
4–6 TOPS NPUs [16, 17]. A chip of that class runs Qwen2.5-1.5B at about 10–23 tokens/s [19].

**Energy: 0.88 J per generated token at 6.34 tokens/s** (Cloud to Edge [12], Table 3), a
meter at the power input, whole board, idle included, 100 generated tokens, 5 runs.
Tummalapalli et al. [20] measured the same model on the same hardware at 6.91 tokens/s and
0.27 J per token above idle; the all-in 0.88 J minus 3.5 W of idle gives 0.33 J, so the two
agree.

- Per query: 0.88 J × 276 tokens = **243 J**. The figure includes its own prefill.
- Under marginal accounting the idle draw comes off: 0.33 J per token, **91 J** per query.
- It is slow: a 276-token answer takes about 44 s.

## 6. The user device

The user tier stands for a light device running a ~1B model, the "tiny-LLMs" of Pakpahan
and Hwang's customer tier [2]. It is priced as a phone because careful measurements exist:
Cai et al. [9] measured a OnePlus 15 with Qualcomm's power telemetry, at 0.074 J per
generated token and 0.016 J per prompt token on the CPU, 15.7 J per query.

The telemetry covers the SoC, not memory, display or radio, so the figure is a **lower
bound**. That can only widen the phone's lead over the OLT, so it cannot flip a result.
Flagship phones run ~3B models [21, 22] that are more accurate than the ONU [23]; the ONU
tier is for devices weaker than the gateway.

## 7. One boundary for all three tiers

Tiers priced from different sources must share one boundary, or the comparison measures
the boundaries. The common one is everything drawn to answer the query, Google's
per-prompt accounting [13].

| Tier | What its source counts | Conversion |
|---|---|---|
| User | the SoC | none (a lower bound, §6) |
| ONU | the whole board at the plug | none (a home device has no PUE) |
| OLT | the GPU card | × 1.60 for host and idle capacity, × PUE 1.54: **× 2.47** |

Google's median prompt is 58% accelerators, 25% host CPU and memory, 10% idle machines and
8% building overhead [13], so the server factor is (58 + 25 + 10) / 58 = 1.60. The building
uses the industry-average PUE of 1.54 [14]. Under marginal accounting the idle machines are
not something a query adds, so the factor is × 2.20. A central office holding one 72 W L4
likely spends more on its host than Google's large servers, so the OLT figure is if
anything low.

## 8. The simulator

### 8.1 Households and queries

Households open conversations through the day. The number of messages in a conversation,
the pauses between them, and when conversations open (by hour, weekday or weekend) are
resampled from BurstGPT's conversation log [10], split into bursts at pauses over 10
minutes. How much a household sends comes from ChatGPT's consumer figures: about 3.6
messages per weekly active user a day. Each message asks a random GSM8K question.

Without more, every weekday follows the same average day and only Poisson counting noise
departs from it (±2–4% an hour at 10,000 households), so a timetable knows almost
everything. Real load drifts: whole hours and days run busier or quieter than the
timetable. Each hour's rate of new conversations is therefore multiplied by a factor that
all households share, log-normal with mean 1, whose log follows an AR(1) process with
log-sd σ and correlation time τ: a doubly stochastic (Cox) Poisson process, so the
month's volume stays the same. σ and τ are measured on BurstGPT's conversation starts,
net of counting noise, over its busy hours: against one hour × weekday/weekend timetable,
σ = 0.46 with τ = 12.3 h; with each week's level taken out, leaving surges within a week,
σ = 0.24 with τ = 3.2 h (`prepare_load_traces.py`). The calibration month drifts too, so
the timetable learns an average that includes the drift.

### 8.2 The cascade

Every tier's recorded answer is replayed. A tier escalates when its answer's confidence is
below the beta-quantile of its last 1,000 confidences, RecServe's rule [1]. Lower tiers
answer at once and are charged their published energy; the OLT answers in its own time
(§8.3).

Energy-aware policies also choose where a query goes, by expected energy to an answer:
C(j) = E_j + p_j · C(j+1), where p_j is how often a query that reached tier j went further.
On arrival a tier runs the query or forwards it to a cheaper tier above; on escalation it
takes the next tier or skips to a cheaper one. Answer lengths and escalation rates are
learned from the answers coming back (EWMA), pooled across households in the study; each
household knows its own devices' costs. The policies differ only in where the OLT's cost
comes from (see the README).

### 8.3 The OLT: continuous batching

The OLT runs continuous batching, as vLLM does. A query joins as soon as one of 64 slots is
free (the largest batch measured) and leaves when its answer is done; one that finds every
slot busy waits. Every running sequence advances at the per-sequence speed of the current
batch, read off the measured curve, and pays that batch's energy per token, so a query
that starts alone pays the lonely rate until others join. A query reaches the OLT only
after the seconds the tiers below spent on it, so the batch it meets is whoever is
genuinely running then. The tests check this against hand-worked cases and check that the
joules charged to queries sum to what the OLT spent.

The OLT reports what a query costs, from its last 5 minutes (§8.4 says what that is under
each accounting). `broadcast` hears the report as sent on the PON every 10 s; `oracle`
reads it with no delay; `piggyback` hears it only on its own household's answers.

### 8.4 Energy accounting

- **Average**: a query pays its share of the batch's energy, the measured J per token at
  that batch, as Google's per-prompt accounting does [13].
- **Marginal**: only what queries add to the network. A batch of n draws the net-of-idle
  energy the sweep measured at that size, shared equally, which matches the GPU under
  continuous batching within 3% (§3.5). The first query on an idle OLT costs about 790 J
  at the whole-system boundary. The ONU drops its idle draw (§5); the phone is unchanged.

What the OLT reports, and what packets and the static tables carry, follows the
accounting. Under average accounting it is the mean cost per token of its recent work, the
share a query pays. Under marginal accounting it is what one more query would add: the
slope while the OLT is busy, the net-of-idle rate while it is idle, weighted by the share
of the last 5 minutes it spent busy. The slope is the least-squares slope of the batch's
energy against its size over batches 1–64, about 9 J for a query on a busy OLT: a smooth
estimate, where differences between adjacent measured batches are too noisy to route on.
Routing on the average instead would make a busy OLT look tens of times dearer than joining
it really is.

The study uses marginal accounting, the energy a routing decision changes, with average
accounting as a sensitivity run.

### 8.5 Comparing policies

Each policy is run across RecServe's beta from 0.1 to 0.9, which traces an accuracy–energy
curve. Policies are compared by the energy they need to reach the same accuracy, read off
each curve's Pareto points. Every run also reports latency, the bytes crossing the PON,
and how far the OLT cost a query was routed on was from what the OLT itself reported when
the query reached it.

## 9. The case study

Settings in `implementation/config/study.yaml`, runs from `src/simulate/run_study.sh`,
tables from `src/analyze/summarize_study.py`. 1,000 to 20,000 households, three seeds, a
month of calibration (when the static tables are observed over plain RecServe) before a
month of test. The main runs follow the average day; two more add BurstGPT's drift
(§8.1), within each week only and with whole weeks departing too. Sensitivity runs change
one setting each: average accounting, a 2× or 5× cheaper ONU, the OLT's energy × 1.07 or
× 1.75, two active users per household, and question statistics learned per household.
Figures from `src/analyze/plot_study.py` (`implementation/results/study/figures/`): the traffic, the beta knob, reading at equal accuracy, where queries are answered, the batching validation, and the savings.

### 9.1 Two timetables

A timetable is only as good as the traffic it was learned from. Observed during a month
of plain RecServe, the OLT is idle more often than once a static policy sends it more
work, which it then makes cheaper. The table makes the OLT look dear and the timetable
under-uses it. In the main, `burst_week` and `burst_all` runs the static policies are
therefore also relearned at each beta while running themselves, three times, as an
operator would keep refreshing a table (`--calibration self`, reported as
`static_hour_self`). Both versions are kept. The first is a table built before
energy-aware routing is deployed; the second is the fairest timetable, the one a live
signal has to beat. The sensitivity runs have only the first.

### 9.2 Results

J per query at 0.80 accuracy, mean of three seeds, and the saving over RecServe at the same
accuracy.

| Households | RecServe | Timetable under RecServe | Timetable relearned | Broadcast |
|---|---|---|---|---|
| 5,000 | 300.2 J | 254.1 J (15%) | 226.2 J (25%) | 217.8 J (27%) |
| 10,000 | 226.2 J | 159.0 J (30%) | 135.6 J (40%) | 129.8 J (43%) |
| 20,000 | 167.2 J | 86.1 J (49%) | 76.8 J (54%) | 71.9 J (57%) |

The broadcast's saving over the relearned timetable, at 0.80 accuracy:

| Load | 5,000 | 10,000 | 20,000 |
|---|---|---|---|
| Average days (`main`) | +3.7% | +4.3% | +6.3% |
| Drift within a week (`burst_week`) | +4.4% | +6.8% | +7.2% |
| All of BurstGPT's drift (`burst_all`) | +9.7% | +12.7% | +11.1% |

- **Routing by energy is the main saving.** Every energy-aware policy uses 25–57% less
  energy than RecServe at the same accuracy from 5,000 households up. Below that the OLT
  is rarely busy enough to batch, and nothing gains more than about 6%.
- **How the OLT's cost is learned matters.** A timetable learned under RecServe gives up
  9–10 points of that saving at 5,000 and 10,000 households. Relearned from its own
  traffic it recovers most of them.
- **The broadcast adds a little on top of the best timetable, more the less predictable
  the load.** About 4–6% on average days and 10–13% with BurstGPT's drift, positive in all
  27 seed and household combinations (+3.0% to +17.3%). The oracle, the same report with no
  broadcast delay, is within 0.4% of the broadcast.
- **The household's own answers are not enough.** Piggyback trails the relearned timetable
  by 13–22% at 5,000 households and up: a household hears the OLT too rarely.
- **Under average accounting live information is worth almost nothing.** The broadcast is
  within 2% of the timetable learned under RecServe, since a query's share of the batch
  changes smoothly with load.
- **The OLT can run out of slots.** With two users per household at 20,000 households,
  more than 1% of arrivals wait for one of its 64 slots, and the broadcast, which sends the
  most there, trails the timetable. That cell is indicative only.

`SUMMARY.md` has every scenario, both accuracy targets, latency and PON traffic.

## 10. Limitations

- **The OLT's whole-system figure is converted, not metered** (§7).
- **The user tier is a lower bound** (§6), which can only widen its lead.
- **The ONU source is thin**: one short prompt, five runs, power mode not stated [12].
- **The energy curve comes from static batches** with fixed-length answers; the
  simulator applies it to continuous batching.
- **Prefix caching off** during the sweep; production servers keep it on.
- **One model per tier, one task**: GSM8K zero-shot.
- **The phone and the ONU are different model families** (Llama, Qwen), so part of the
  accuracy step between them is the family.
- **No transport energy**, and the ONU-to-device relay of the broadcast is taken as free.
- **One confidence window per tier, shared by all households.** A real ONU keeps its own,
  and one that is skipped most of the day would fill it slowly.
- **The broadcast is not standardized.** The PON's downstream multicast exists [15], but
  the OLT and ONU software to send and relay the cost would be new.

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
19. Turing Pi. *RK3588 LLM Benchmarks: GGUF Quantization Compared*; *Running LLMs on RK3588 NPU with RKLLM*.
20. Tummalapalli et al. *LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load.* arXiv:2603.23640.
21. Apple Machine Learning Research. *Introducing Apple's On-Device and Server Foundation Models.*
22. Gemini Team, Google. *Gemini: A Family of Highly Capable Multimodal Models.* arXiv:2312.11805.
23. Meta. *Llama 3.2 model card* (GSM8K: 1B 44.4, 3B 77.7).
