# Energy measurements — three-tier case study

**Status (2026-09-11).** OLT energy sweep: done, run twice, replicated.
Answer collection for all three tiers: done, 1,319 questions each, every
confidence check passing. ONU and user tiers: priced
from published measurements, assessed in §5 and §6. The ONU is a Jetson
Orin Nano Super, a realistic home device, priced from Cloud to Edge [17]; its
answers were re-collected in the same 4-bit format. All three tiers are compared
at one common, whole-system boundary (§7). A first-party ONU measurement is
optional, if a board becomes available.

**Headline.** Counted as whole systems, the OLT answers a query more cheaply than
the ONU once it batches about 5 queries (≈ 3.6 at a best-case PUE). It never
undercuts the user device within batch 64. In simulation (§8), piggybacked
energy reports route as well as an oracle, but with the ONU as published they
save nothing over simply dropping the ONU. They pay (5–16%) only with an ONU
below about 60 J per query, at a busy OLT.

This document records the tests behind the three-tier case study: what was
measured, every methodological decision with the reference it follows, and the
results. It is written to be lifted into the thesis methodology chapter.

---

## 1. What is being measured, and why

The case study is a three-tier cascade, **user → ONU → OLT**, following the
tiers of Pakpahan and Hwang's PON architecture [2] with RecServe's confidence
rule deciding whether a query escalates [1].

The question behind the measurements: **can batching make a higher, shared tier
cheaper per query than a lower, dedicated one?** A user device or an ONU serves
one household and runs one query at a time, so its cost per query is fixed. The
OLT aggregates many ONUs, so its batch grows with load and its cost per query
falls. If that cost falls below the ONU's, the cheapest place to answer a query
depends on the time of day, and a lower tier that learns the OLT's current cost
(the piggyback mechanism) can route on it.

That requires three numbers per tier, measured comparably: energy per input
token (prefill), energy per output token (decode), and for the OLT, both as a
function of batch size.

## 2. The tiers

| Tier | Model | Precision | Hardware | Energy from | Answers from |
|---|---|---|---|---|---|
| User | Llama-3.2-1B-Instruct | GGUF Q4_K_M | Snapdragon 8 Elite Gen 5 | Cai et al. [10] (literature) | Modal, this work |
| ONU | Qwen2.5-1.5B-Instruct | GGUF Q4_K_M | Jetson Orin Nano Super 8GB | Cloud to Edge [17] (literature, §5) | Modal, this work |
| OLT | Qwen2.5-7B-Instruct | fp8 | NVIDIA L4 | **first-party (§3)** | Modal, this work |

Choices that hold across tiers:

- **Sizes sit inside Pakpahan's bands** [2]: a tiny model on the user device, 0.5–7B
  at the ONU, 7–13B at the fog/OLT. The OLT is at the bottom of its band on
  purpose: a smaller model is the easiest for batching to make cheap, so this is
  the configuration most likely to show the effect, and it is stated as such.
- **Each tier's answers and energy come from the same model at the same
  precision.** Quantisation changes the weights, hence the answers, hence the
  confidence the cascade routes on; pricing one checkpoint's answers with
  another's energy would join two different models. The user and ONU tiers
  match approximately: each uses the same 4-bit format as its energy source
  ([10], [17]), but runs through vLLM rather than llama.cpp or Ollama, and the
  user tier's exact 4-bit scheme in [10] is not stated.
- **Hardware is in each tier's declared class**: phone SoC, Jetson-class
  accelerator, and an L4, which is inside the fog hardware class of the
  project's energy table (T4 / L4 / A30 / A2 / A16 / L40S).

## 3. OLT energy sweep

### 3.1 Methodology

| Decision | Choice | Following |
|---|---|---|
| Instrument | NVML cumulative energy counter, `nvmlDeviceGetTotalEnergyConsumption`, read before and after each pass | The ML.ENERGY Benchmark [3] and its measurement guide [4]. Power *sampling* is avoided: on recent GPUs the built-in sensor samples only part of the runtime [6]. |
| Boundary | Measured at the GPU card, its own memory included. Converted to the whole system (host, idle capacity, building) for the comparison with other tiers, in §7 | Measured as ML.ENERGY [3] does, which justifies it by GPUs being 50–70% of provisioned server power. TokenPowerBench [5] also counts CPU and memory; Google's per-prompt method [18] counts everything, and is what §7 converts to. |
| Serving engine | vLLM 0.21, offline `generate` | ML.ENERGY [3] measures on vLLM. |
| Unit | Joules per input token for prefill, per output token for decode, reported separately | TokenPowerBench [5]; Solovyeva and Castor [8]; Delavande, Pierrard and Luccioni [7], who show the two phases respond to batching differently. |
| Phase split | Each trial runs the same prompts twice: a 1-token pass (the prefill forward pass) and a full pass. Prefill = first pass; decode = difference, over the remaining output tokens. | Phase-level accounting as in [5]; [5] tags power samples by phase instead. With static batches prefill happens first, so the difference is a close approximation. |
| Batch | Static batches of 1, 2, 4, 8, 16, 32, 64 prompts | Batch sweeps as in TokenPowerBench [5] and [7]. ML.ENERGY instead reports one saturated steady state [3]; the question here is how cost *varies* with batch, which one point cannot show. |
| Output length | Fixed at 300 tokens per sequence (`ignore_eos`) | Caravaca, Cuevas and Cuevas's 300-output-token profile [9]. Fixing it keeps J/token comparable across batches; per-query cost uses each tier's real answer lengths (§4). |
| Prompts | GSM8K test questions in the cascade's zero-shot template, ~93 tokens | The workload the case study runs; prefill cost depends on prompt length. |
| Warm-up and repeats | One discarded warm-up per batch size; 3 measured repeats; median reported, spread kept | Warm-up and repetition follow [6]. ML.ENERGY reports single runs without error bars [3]. |
| Short passes | The prefill pass is repeated back to back until the measured window spans ≥ 3 s, then averaged | [6] recommends minimum durations for short kernels. Found necessary here: see §3.4. |
| Prefix caching | Disabled | vLLM enables it by default [13]; left on, the warm-up would cache every prompt and trials would skip prefill. See §3.4. |
| Idle power | Measured for 10 s with the model loaded; every figure reported gross and net of idle | Neither [3] nor [5] subtracts idle. Both are kept so the comparison with other tiers can use whichever boundary they share. |
| Model and precision | Qwen2.5-7B-Instruct, fp8 (native on Ada) | Bottom of the 7–13B fog band [2]; fp8 leaves room in 24 GB for the KV cache that batching needs. |

### 3.2 Results

Run 2, with the prefill fix (idle 28.7 W). Decode spread is the min–max across
the three repeats.

| Batch | Prefill J/in-tok (net) | Decode J/out-tok [spread] | Tokens/s | Avg J/query | Marginal J/query |
|---|---|---|---|---|---|
| 1 | 0.0336 (0.0203) | 2.441 [2.441–2.442] | 29 | 733 | 733 |
| 2 | 0.0380 (0.0226) | 1.228 [1.228–1.240] | 58 | 370 | 7.3 |
| 4 | 0.0239 (0.0142) | 0.625 [0.619–0.625] | 115 | 189 | 7.1 |
| 8 | 0.0173 (0.0104) | 0.313 [0.313–0.313] | 226 | 95 | 1.8 |
| 16 | 0.0150 (0.0090) | 0.160 [0.159–0.160] | 437 | 49 | 3.6 |
| 32 | 0.0149 (0.0089) | 0.085 [0.085–0.085] | 807 | 27 | 4.1 |
| 64 | 0.0143 (0.0085) | 0.048 [0.047–0.048] | 1,387 | 16 | 4.5 |

Per query here means 93 input and 300 output tokens. Marginal is the slope of
batch energy between adjacent batch sizes: the extra energy one more query adds.

What it shows:

- **Decode energy fell 51× over the sweep at constant power.** The L4 draws its
  full 72 W from batch 1, so energy per token is 72 W divided by throughput, and
  throughput doubled almost exactly with every batch doubling up to batch 16
  (memory-bound decode: extra sequences ride along on weight loads already being
  paid for). Beyond 16 it slows to ~1.7× per doubling; the curve bends but does
  not flatten by batch 64.
- **Prefill gets cheaper only up to batch ~16, then stays flat** at ~0.015
  J/in-token. Small batches amortise per-call overhead; beyond that prefill is
  compute-bound and scales linearly, consistent with [7].
- **One more query costs ~4 J once the OLT is busy**, against an average of 16–95
  J per query in the same range. Small-batch marginals (7.3, 7.1, 1.8) are
  differences of two noisy totals and should not be read individually.
- **Replicated.** Run 1 (same configuration, before the prefill fix) agrees on
  decode within ±1.4% at every batch size.

### 3.3 Where the OLT undercuts the lower tiers

Per query, at each tier's real answer lengths from the collection (§4): ~122
prompt tokens; 185, 276 and 251 generated by user, ONU and OLT. The primary
column is the common, whole-system boundary of §7; the others show how much the
answer depends on that choice.

OLT energy per query:

| Batch | GPU card | GPU, net of idle | **Whole system, PUE 1.54** | Whole system, PUE 1.09 |
|---|---|---|---|---|
| 1 | 617.8 J | 370.6 J | **1,525.6 J** | 1,079.8 J |
| 2 | 313.4 | 187.9 | **773.9** | 547.8 |
| 4 | 160.0 | 96.5 | **395.0** | 279.6 |
| 8 | 80.9 | 48.5 | **199.7** | 141.3 |
| 16 | 42.1 | 25.3 | **104.0** | 73.6 |
| 32 | 23.2 | 13.9 | **57.2** | 40.5 |
| 64 | 13.8 | 8.3 | **34.0** | 24.1 |

Batch size from which the OLT is cheaper:

| Lower tier | Its cost per query | **Whole system, PUE 1.54** | Whole system, PUE 1.09 | GPU card only | GPU, net of idle |
|---|---|---|---|---|---|
| ONU (1.11 J × 276 tokens) | 306.4 J | **≈ 5.2** | ≈ 3.6 | ≈ 2.0 | ≈ 1.2 |
| User | 15.7 J | **not within 64** | not within 64 | ≈ 54 | ≈ 28 |

- **The ONU is undercut at modest load.** At the common boundary the OLT beats
  it from a batch of about 5. Judging the OLT by its GPU card alone, as most
  benchmarks do, would put the crossover at 2 and overstate the case for
  offloading by a factor of about 2.5.
- **The user device is never undercut.** Its per-token cost is the lowest in the
  cascade and it answers most briefly (185 tokens). Even the GPU-only OLT needs
  batch ≈ 54, and at the whole-system boundary the OLT's 34 J at batch 64 is
  still twice the phone's 15.7 J. Since the phone's figure is a lower bound (§6),
  that conclusion holds.
- **One more query on a busy OLT costs ~10 J** at the whole-system boundary
  (~4 J at the GPU), against 306 J on the ONU.

At the OLT, the inversion depends on load. Batch ≈ arrival rate × service time
(Little's law). A sequence decodes at ~28 tokens/s up to batch 16, so an OLT
answer takes ~9 s, and a batch of 5 needs about one query arriving every two
seconds across the PON. Load varies by day: BurstGPT's regional traffic is 11×
lower at night than at the afternoon peak [12]. So the cheapest tier for an ONU
query changes with the hour. This is the regime in which a live cost signal is
worth having.

### 3.4 Two measurement problems found and fixed

- **Prefix caching zeroed prefill.** vLLM caches prompt prefixes by default [13].
  The discarded warm-up filled the cache, so measured trials skipped most of the
  prefill. Disabled explicitly. An earlier partial run (batches 1–4) was affected;
  its decode figures are unaffected in practice.
- **The energy counter is too coarse for one short pass.** In the first run,
  batch-1 prefill read 6.75 J in one trial (= 72 W × 0.094 s) and exactly 0 J in
  the next two: the counter advances in steps of roughly 0.1 s of energy, longer
  than a batch-1 prefill. Fixed by repeating the prefill pass back to back until
  it spans ≥ 3 s, then averaging (66 repeats at batch 1, 3 at batch 64). Decode
  passes last 10–14 s and were never affected.

## 4. Answer collection

All three tiers answer every one of GSM8K's 1,319 test questions, so any
escalation policy can be replayed offline without re-running a model.

| Decision | Choice | Following |
|---|---|---|
| Exhaustive matrix | Every tier answers every query | The project's existing collection design; a single cascade trace cannot say what a skipped tier would have answered. |
| Where | Modal L4s, one container per tier, in parallel | Answers depend on model and precision, not the chip; energy is priced on each tier's own hardware separately. |
| Decoding | Temperature 0, up to 512 tokens, each model's chat template | Deterministic comparison across tiers. |
| Confidence | RecServe's exp(mean token logprob) stored; every token logprob kept | RecServe [1] specifies normalised perplexity for generation; the design doc (§13.2) found exp(min logprob) separates correct from wrong better, so both stay recomputable. Minimum-logprob deferral is also studied by Gupta et al. [14]. |
| Scale | n = 1,319 | The design doc flags n = 200 as thin. |
| Stop condition | None: generation ends at the model's end-of-sequence token or 512 tokens | Every generated token then has a logprob, so confidence and energy count the same tokens. |

**Checking the confidence before routing on it.** `src/scripts/check_confidence.py`
verifies, per tier, that every generated token has exactly one logprob, all
logprobs are ≤ 0, and the stored confidence equals exp(mean token logprob)
recomputed from the raw values. It also reports how many answers hit the token
limit, and whether confidence separates correct from wrong answers (Welch t) for
both exp(mean) and exp(min).

**Smoke test (20 questions per tier): all checks pass.** Every tier, including
the llama.cpp-format user model, returns one logprob per generated token, and
stored confidence matches the formula exactly. Accuracy rises tier by tier
(0.35 → 0.50 → 0.90). Separation is weak at this size (t = 0.6–2.2), as expected
with 20 samples; the full run decides it.

Collection prompts are ~122 tokens against the energy sweep's ~93, because the
collection applies each model's chat template and the sweep sent the raw prompt.
Prefill cost per token is nearly flat over that range, so the per-token rates
transfer; per-query costs use the collection's real token counts.

**What the check found in the older 5-shot trace** (`lb_full.raw.jsonl`):

- *Token counts do not line up in the two Ollama tiers* (58 of 200 user records,
  86 of 200 ONU). The collector stopped generation at `"Question:"` or a blank
  line, as the 5-shot harness does, and deliberately trimmed logprobs at the stop.
  The 1–8 extra tokens in `tokens_gen` are the stop sequence itself. Confidence is
  therefore computed over exactly the answer, which is correct, and the extra
  tokens really were generated, so counting them for energy is also correct.
- *exp(min logprob) does not separate correct from wrong there* for the user and
  ONU tiers (t = +0.76 and −0.30), while RecServe's exp(mean) does strongly
  (t = +6.6 and +10.4). The design doc's §13.2 found the opposite on an earlier
  zero-shot trace. The same 1B model appears in both, so the better confidence
  definition may depend on the prompt format; the full zero-shot collection
  tests it again.

**Full collection: 1,319 questions per tier, all checks pass.**

| Tier | Accuracy | Mean confidence | Prompt tokens | Generated tokens | Hit 512-token limit | Welch t, exp(mean) | Welch t, exp(min) |
|---|---|---|---|---|---|---|---|
| User | 0.472 | 0.867 | 125.7 | 184.9 | 14 (1.1%) | +10.70 | **+12.57** |
| ONU (Q4_K_M) | 0.688 | 0.900 | 122.0 | 276.0 | 32 (2.4%) | +10.64 | **+11.11** |
| OLT | 0.917 | 0.937 | 122.0 | 251.4 | 7 (0.5%) | +7.84 | **+10.19** |
| *ONU (AWQ, superseded)* | *0.644* | *0.880* | *122.0* | *245.4* | *23 (1.7%)* | *+7.90* | *+10.52* |

The ONU was collected twice. The AWQ run matched the AGX Orin energy source;
when the ONU became the Orin Nano Super (§5), its answers were re-collected in
Q4_K_M, the format that board's energy was measured in, and those replace the
AWQ rows. The Q4_K_M model is slightly more accurate and answers at greater
length (276 tokens against 245), and the per-query cost uses its length.

- **Accuracy rises tier by tier**, so skipping a tier never lowers accuracy (safe
  by dominance). Zero-shot with each model's chat template is far more accurate
  than the older 5-shot trace (0.12 and 0.29 for user and ONU there).
- **Confidence separates correct from wrong answers on every tier**, and
  **exp(min logprob) is the stronger signal on all three.** That replicates the
  design doc's §13.2, which was also measured on zero-shot prompts. The older
  5-shot trace showed the opposite (see above). The better confidence definition
  therefore depends on the prompt format, so it should be chosen per workload
  and reported as a finding.
- **Truncation is small**: at most 2.4% of answers hit the 512-token limit.

The answers files are `gsm8k_zeroshot_user-onu-olt_n1319_<UTC>.raw.jsonl`
(31 MB; its user and OLT rows are used) and
`gsm8k_zeroshot_onu_n1319_<UTC>.raw.jsonl` (13 MB, the ONU), every token
logprob included.

## 5. ONU hardware and energy

**The ONU is a Jetson Orin Nano Super 8GB.** Earlier drafts priced the ONU on a
Jetson AGX Orin 64GB, because that is what EdgeReasoning [11] measured. It is not
a realistic ONU: a ~$2,000 developer module drawing 15–60 W, in a device class
where an ONU draws ~4 W and costs tens of dollars. The reference architecture
only asks for a "Jetson-class accelerator" [2]. The Orin Nano Super is the
realistic reading: $249, 7–25 W, 67 TOPS, and it runs the ONU's 1.5B model in
4-bit. No ISP ships LLM accelerators in ONUs today in any country; the tier is
forward-looking, so the defensible choice is the cheapest board that runs it.

**Energy: Cloud to Edge [17], 1.11 J per generated token** for Qwen2.5-1.5B in
Q4_K_M on the Orin Nano Super's GPU (9.37 tokens/s).

| Aspect | What the source did |
|---|---|
| Meter | Mecheer JK-PM07 at the board's power input |
| Boundary | whole board; idle **not** subtracted |
| Engine | Ollama (llama.cpp), Q4_K_M |
| Workload | one fixed ~12-token prompt, 100 generated tokens, 5 runs |
| Metric | mean power × inference time ÷ generated tokens, so one all-in figure per generated token |
| Power mode | not stated |

Consequences for this case study:

- **Answers at the same precision.** The ONU's answers were re-collected in the
  same Q4_K_M format (§4), so energy and answers describe the same 4-bit
  weights. They are loaded by vLLM's GGUF loader rather than Ollama, the same
  approximation the user tier makes.
- **Per-query cost = 1.11 J × generated tokens**, 306 J at the ONU's mean of
  276 tokens. The figure already contains its own prompt processing, so no
  separate prefill term is added. The case study's prompts (~122 tokens) are
  longer than the source's (~12), so a few percent of prefill energy is not
  captured.
- **Already at the system boundary.** A meter at the plug counts the whole
  board, idle included, which is the boundary §7 puts every tier on. It is a
  home device, so no building overhead applies.

**What the earlier AGX Orin analysis found, kept as a result.** EdgeReasoning's
released code [16] shows its decode energy integrates one rail, `VDD_GPU_SOC`
(GPU plus SoC), from the Jetson's built-in INA3221 sensors. Those sensors
under-read true board power by a nearly constant ~3.1 W plus 2% on the AGX Orin
(true ≈ 1.02 × internal + 3,115 mW) [15]. So that figure was a lower bound at a
narrower boundary than the OLT's. The two ONU pictures bracket the boundary
question from opposite sides.

**First-party measurement, if a board becomes available.** Whole board with a
USB-C power meter (the source's method), the same Q4_K_M file, and the protocol
of §3: warm-up, repeats, and windows long enough for the meter. Chameleon Cloud's
CHI@Edge testbed has three Orin Nanos but requires a faculty-led project.
CloudJetson lists the Orin Nano as coming soon. The board itself costs $249.

## 6. User tier — literature, and why

Measured first-party would be better, but no rentable phone service reports
energy. Qualcomm AI Hub runs models on real Snapdragon devices in the cloud but
reports latency, memory and compute-unit placement, not power, and profiles
single compiled passes rather than token-by-token generation. The user tier
therefore keeps Cai et al.'s figures [10]: measured on a real OnePlus 15 with
Qualcomm's own power telemetry, CPU decode 0.074 J/token, CPU prefill 0.016
J/in-token.

The source is sound on its own terms. It keeps the device below 28 °C in
airplane mode with the screen off, discards a warm-up, repeats each measurement
at least three times, and separates prefill from decode. Its telemetry reports
energy for the whole SoC; for CPU-only inference that is the CPU cluster's, so
the figure is the whole chip. It leaves out off-chip memory, display and radio,
so it is a **lower bound** on the phone's draw. A home-made measurement
would be worse, not better: Android's battery counters report the whole
phone, screen and radio included, and an external meter means opening the
device.

**Does the lower bound matter?** Only in one direction. Understating the phone
makes the OLT look worse against it, and the OLT already never undercuts it
(§3.3); a fuller boundary would widen the gap. It cannot flip a result.

## 7. One boundary for all three tiers

Tiers priced from different sources have to be put on one boundary before they
are compared, or the comparison measures the boundaries rather than the tiers.
Before this section they were not: the ONU counted its whole board, the OLT only
its GPU card, and the phone only its chip, so the choice of boundary alone moved
the ONU crossover by a factor of about 2.5.

**The standard.** Google's per-prompt methodology [18] is the most complete
published accounting of a production LLM query. It separates four parts: the
active accelerators, the host CPU and memory, idle machines held for load
spikes, and data-centre overhead. It finds the accelerator-only view undercounts
its median prompt by 2.4× (0.10 Wh against 0.24 Wh). The common boundary here is
the comprehensive one: **everything drawn to answer the query**.

| Tier | Source | What it counts | Conversion to the common boundary |
|---|---|---|---|
| User | Cai et al. [10] | whole SoC (in CPU-only inference, the CPU cluster) | none; off-chip memory, display and radio are missing, so a lower bound (§6) |
| ONU | Cloud to Edge [17] | whole board at its power input, idle included | none; already at the boundary, and a home device has no PUE |
| OLT | this work (§3) | GPU card, its memory included | × 1.60 for host and idle capacity, × PUE 1.54: **× 2.47** |

**The OLT factor.** Google's shares of a median prompt are 58% accelerators, 25%
host CPU and memory, 10% idle machines and 8% overhead at a fleet PUE of 1.09
[18]. The server-side factor is (58 + 25 + 10) / 58 = 1.60. The building is
applied separately at the site's own PUE, because an OLT's central office is
not a hyperscale data centre: the primary value is the industry average of
1.54 [19], and Google's 1.09 is kept as the best case. Together, × 2.47
(× 1.75 at the best case).

**Why literature for two tiers and first-party for one.** The OLT's quantity is
a batch curve; no paper publishes one for this model on this GPU with a phase
split, so it had to be measured (§3). The user and ONU tiers run at batch 1 on
fixed devices, where published, peer-measured figures exist, and better
instruments than any available here: Qualcomm's power telemetry for the phone,
a meter at the board's plug for the Jetson (§5, §6). Each source's boundary is
stated and converted once, in this section.

**The conversion is conservative for the OLT.** Google's shares come from large
multi-accelerator hosts. A central-office server holding one 72 W L4 likely
spends a larger share on its host, so the whole-system OLT figure is if anything
still low, and the ONU crossover errs early rather than late.

**Verdict per tier.**

- **User: keep.** Cai et al. is the right source and needs no conversion. It is a
  lower bound, and that cannot change the result.
- **ONU: keep the switch to the Orin Nano Super.** Cloud to Edge measures it at
  the boundary this study needs. Its weak points are a short fixed prompt, five
  runs and an unstated power mode; a first-party measurement would fix those.
- **OLT: keep first-party.** Convert it with the factors above; report the
  GPU-only curve beside it, since that is how most benchmarks report.

## 8. Piggyback simulation

**What is tested.** RecServe decides whether a query escalates. The piggyback
mechanism adds where it goes. Every answer travelling back down carries the
energy rates each tier just ran at (J per prompt token, J per generated token)
and how many tokens it generated, so the tiers below learn the OLT's current cost
with no extra messages. A tier uses it twice: on arrival (answer here, or forward
straight to a tier above) and on escalation (next tier up, or skip to a later
one), picking the lowest expected cost to completion. The question: **does a
live report of the OLT's energy let the lower tiers route more cheaply than a
fixed chain, at the same accuracy?**

### 8.1 Methodology

| Decision | Choice | Following |
|---|---|---|
| Answers | Replay of the recorded 1,319-question collection (§4). Every tier answered every query, so any route is replayed without re-running a model | Exhaustive matrix (§4) |
| Escalation | RecServe's β-quantile rule, unchanged: escalate when confidence (exp(mean logprob)) falls below the β-quantile of the tier's last 1,000 confidences; β = 0.1 … 0.9 | RecServe [1], which recommends windows of 300–1,000 |
| Routing rule | Expected cost to completion, C(j) = E_j + p_j · C(j+1), where p_j is how often a query that reached j went further. p_j, answer lengths and rates are learned from the packets (EWMA, α = 0.05) after 20 reports per tier. No skip margin (δ = 0) | This work |
| Energy | User and ONU fixed (§5, §6). OLT from the measured batch curve (§3), log-log interpolated, at the whole-system boundary, × 2.47 (§7). A GPU-card run for comparison | §3, §7 |
| OLT load | Exogenous: the OLT serves many PONs, so the cascade's own queries are a small part of its traffic. Daily shape from BurstGPT's hourly arrivals, scaled so the busiest hour has 0.5–32 queries in service | BurstGPT [12] |
| OLT batch | An arriving query meets a Poisson number of others in service, so its batch is 1 + Poisson(load). Each packet therefore reports a noisy rate | In an M/G/∞ system occupancy is Poisson, and Poisson arrivals see time averages [21] |
| Load and arrivals | Arrival rate = load / service time, with ~9 s per 251-token answer: peak load 8 ≈ 3,200 arrivals an hour | Little's law [20] |
| Query stream | 6,006 queries at the user tier over 3 days, hour shares from BurstGPT, questions drawn at random | |
| Comparison | Each policy's accuracy–energy frontier over β (Pareto points only), J/query read off at equal accuracy (0.70 and 0.80) | Iso-accuracy comparison: skipping to a more accurate tier changes accuracy too |
| Controls | Three-tier RecServe; a fixed two-tier chain with the ONU dropped; a static configuration holding the OLT's day-average rate, calibrated on the true load, and stale versions calibrated at ¼ and 4× the load; an oracle that knows the current expected rate | The last four share piggyback's routing rule and differ only in which OLT rate they see |
| Noise | Common random numbers (every policy meets the same batches); the main run and the 0.2× ONU run repeated with two more seeds | |

### 8.2 Results

**At the published ONU cost the live signal adds nothing, because the ONU is not
worth using.** J per query at 0.80 accuracy, ONU at 306 J:

| OLT peak load (arrivals/h at peak) | RecServe, 3 tiers | ONU dropped | Static | Piggyback | Oracle |
|---|---|---|---|---|---|
| 2 (828) | 695 | 577 | 578 | 579 | 580 |
| 4 (1,640) | 542 | 391 | 392 | 393 | 392 |
| 8 (3,221) | 415 | 236 | 237 | 237 | 237 |
| 16 (6,216) | 336 | 136 | 138 | 137 | 137 |
| 32 (11,485) | 290 | 80 | 80 | 77 | 77 |

Every cost-aware policy saves 17–74% against three-tier RecServe, but dropping
the ONU saves the same. At 306 J for 0.688 accuracy the ONU is dominated once the
OLT is at all busy: an escalated query costs less at the OLT and is answered
better there. Even at peak load 0.5, where an OLT query costs 1,200–1,500 J,
dropping the ONU is cheaper at 0.80 accuracy (850 J against 919 J); only at 0.70
does the three-tier chain win, narrowly (531 J against 549 J).

**The live signal pays when the ONU is cheap.** Energy piggyback saves over the
better of the two fixed chains (three-tier RecServe, ONU dropped), at 0.80
accuracy, with the ONU's energy scaled down from its published 1.11 J/token to
stand in for a more efficient accelerator:

| ONU per query | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 306 J (published) | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | +4% |
| 153 J (× 0.5) | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | +5% |
| 61 J (× 0.2) | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | +6% | +5% | +8% |
| 31 J (× 0.1) | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | +6% | +12% |
| 15 J (× 0.05) | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 | +7% | +16% |

"≈ 0" is within ±3%. Columns are the OLT's peak load, in queries in service.

- **It pays in one regime:** an ONU cheap enough to be the right place for a
  query at night, when the OLT is idle and dear, and the wrong one at the daily
  peak. Then the cheapest route changes with the hour, and only a live signal
  follows it. At a 61 J ONU, peak load 16 and β 0.6, piggyback answers 24–28%
  of queries at the ONU between 02:00 and 10:00 and almost none from midday on,
  matching the oracle hour by hour. The static configuration skips the ONU almost all day.
- **The packets carry enough.** Piggyback is within 3% of the oracle in every
  configuration, and within 1% in three out of four, although the rate it
  learns is noisy (mean error 4–61% by configuration, median 10%): the decision
  only needs to know which side of the ONU's cost the OLT is on.
- **A stale configuration is the real risk of going without it.** A static
  configuration matches piggyback only when calibrated on the load it then meets.
  Calibrated at a quarter or four times the load, it costs up to 35% more
  (61 J ONU, peak load 32, 0.70 accuracy: 74.5 J against 55.0 J). At the
  published ONU, calibrated too low, it stops skipping the ONU: +27% at peak
  load 4, 0.70 accuracy (321 J against 253 J).
- **Robust across seeds.** At the 61 J ONU, loads 8–32 and 0.80 accuracy,
  piggyback beats both fixed chains in all three seeds (load 16: 129 / 129 / 127 J,
  against 136 / 137 / 133 J with the ONU dropped and 138 / 137 / 133 J static).
  At the published ONU the policies stay within ±3 J of each other, with no
  consistent sign.
- **At the GPU-card boundary** the OLT looks cheaper still and the ONU is
  dominated at every load. Piggyback matches the fixed chains up to load 8 and
  gains 7–18% over them at loads 16–32, where the OLT's peak cost (23–42 J)
  approaches the user device's 15.7 J. At the whole-system boundary that gain
  disappears.

**What this means for the case study.** The mechanism works: it learns the OLT's
cost from the answers already flowing back and routes as well as an oracle. What
it is worth depends on the middle tier. With the Orin Nano Super as published,
the better design is a two-tier chain, and no live signal is needed to find
that. For routing on live energy to matter, the ONU has to cost less than about
60 J per query (≈ 0.25 J per generated token), for example an NPU-class
accelerator, or the AGX Orin's 54 J from §5.

## 9. Limitations

- **The OLT's whole-system figure is converted, not measured.** Host, idle and
  building are added with Google's shares [18] and an industry-average PUE
  [19], not metered on a real central-office server. The conversion likely
  understates a single-GPU server (§7), which makes the ONU crossover early.
- **The user tier is a lower bound.** The phone's memory, display and radio are
  not counted [10]. This can only widen the user tier's lead (§6).
- **The ONU source is thin.** One short fixed prompt, five runs, power mode not
  stated [17].
- **Static batches, fixed-length answers.** Clean curves, but a live OLT sees
  random arrivals and a fluctuating batch [3].
- **Prefix caching off.** Needed to isolate prefill; production servers keep it
  on, which would make repeated prompt prefixes cheaper.
- **Rented cloud GPU.** The counter covers only the physical GPU the container is
  given, which assumes Modal does not share the card.
- **One model per tier, one task.** GSM8K zero-shot; results may differ on other
  tasks and prompt lengths.

- **The piggyback results are simulated.** OLT load is set, not measured; the
  cascade's own queries do not add to it; transport energy and latency are left
  out; the grid in §8.2 is one seed per cell (three for two rows).

## 10. Reproducing

```bash
cd implementation
.venv/bin/modal run src/modal_apps/measure_gpu_energy.py     # OLT sweep, ~15 min on one L4
.venv/bin/modal run src/modal_apps/collect_answers.py        # all tiers' answers, 1,319 queries
.venv/bin/python src/scripts/check_confidence.py results/energy_tests/<answers>.raw.jsonl
.venv/bin/python src/scripts/sim_piggyback.py                # piggyback simulation (§8), ~30 s
.venv/bin/python src/scripts/sim_piggyback.py --boundary gpu
for s in 0.05 0.1 0.2 0.5; do .venv/bin/python src/scripts/sim_piggyback.py --onu-scale $s; done
.venv/bin/python src/scripts/make_energy_artifact.py         # results page, from the files above
```

Every output and log is kept in `implementation/results/energy_tests/` (see its
README). The scripts write there with a UTC timestamp in each file name, so a
rerun never overwrites an earlier result. Raw terminal logs (`*.log`) are
committed here through an exception to the repo's global `*.log` ignore rule; a
cleaned copy of each (`*.txt`) sits beside it.

## References

1. Wu et al. *Recursive Offloading for LLM Serving in Multi-tier Networks* (RecServe). arXiv:2505.16502.
2. Pakpahan and Hwang. *Enabling Software-Defined Tiered LLM Inference Continuum on Passive Optical Network.* IEEE Access, vol. 14, 2026. doi:10.1109/ACCESS.2026.3651558.
3. Chung et al. *The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization.* arXiv:2505.06371. https://arxiv.org/abs/2505.06371
4. ML.ENERGY. *Measuring GPU Energy: Best Practices.* https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/
5. *TokenPowerBench: Benchmarking the Power Consumption of LLM Inference.* arXiv:2512.03024. https://arxiv.org/abs/2512.03024
6. Yang et al. *Part-time Power Measurements: nvidia-smi's Lack of Attention* (SC'24: *Accurate and Convenient Energy Measurements for GPUs*). arXiv:2312.02741. https://arxiv.org/abs/2312.02741
7. Delavande, Pierrard and Luccioni. *Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use.* arXiv:2601.22362. https://arxiv.org/abs/2601.22362
8. Solovyeva and Castor. *Towards Green AI: Decoding the Energy of LLM Inference in Software Development.* (local: `thesis/papers/TowardsGreenLLM.pdf`)
9. Caravaca, Cuevas and Cuevas. *From Prompts to Power: Measuring the Energy Footprint of LLM Inference.* arXiv:2511.05597.
10. Cai et al. *Is Your NPU Ready for LLMs? Dissecting the Hidden Efficiency Bottlenecks in Mobile LLM Inference.* arXiv:2607.05475.
11. Kubwimana and Huang. *EdgeReasoning: Characterizing Reasoning LLM Deployment on Edge GPUs.* arXiv:2511.01866.
12. Wang et al. *BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems.* KDD 2025. arXiv:2401.17644.
13. vLLM documentation. *Automatic Prefix Caching.* https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
14. Gupta et al. *Language Model Cascades: Token-level Uncertainty and Beyond.* ICLR 2024. arXiv:2404.10136.
15. *Accurate Calibration of Power Measurements from Internal Power Sensors on NVIDIA Jetson Devices.* arXiv:2306.13107. https://arxiv.org/abs/2306.13107
16. EdgeReasoning artifact code: `edge-inference/edgereasoning` v0.1.0 (tegrastats parsing, fitted power model: `eval/tegra/mmlu/src/telemetry/telemetry_proc.py`, `validation/decode_parameters.json`) and `edge-inference/token2metrics` (decode energy from `vdd_gpu_soc_current_mw`: `decodenergy/empirical_data.py`). https://github.com/edge-inference/edgereasoning
17. *Cloud to Edge: Benchmarking LLM Inference On Hardware-Accelerated Single-Board Computers.* arXiv:2604.24785. https://arxiv.org/abs/2604.24785
18. Elsworth et al. (Google). *Measuring the environmental impact of delivering AI at Google Scale.* arXiv:2508.15734, 2025. https://arxiv.org/abs/2508.15734
19. Uptime Institute. *Global Data Center Survey 2025* (average annual PUE 1.54). https://uptimeinstitute.com/
20. Little. *A Proof for the Queuing Formula: L = λW.* Operations Research 9(3), 1961.
21. Wolff. *Poisson Arrivals See Time Averages.* Operations Research 30(2), 1982.
