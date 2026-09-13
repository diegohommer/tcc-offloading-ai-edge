# Energy measurements — three-tier case study

**Status (2026-09-13).** OLT energy sweep: done, run twice, replicated.
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
energy reports let the cascade switch between three tiers, two, or straight to
the OLT, hour by hour, without configuration. Against the fixed chains they save
5–17% with an ONU below about 60 J per query, and nothing with the ONU as
published, where dropping the ONU is as good. Against a time-of-day schedule, the
harder baseline, they gain only when the load departs from the typical day: up
to 5% when each query is charged its share of the OLT's energy, up to 12% when it
is charged the energy it adds (§8.3), with the OLT reporting its own recent mean.
On real load (§8.5: BurstGPT replayed, households learning on their own, no tier
charged for being on), routing by energy still beats RecServe's fixed chain,
by up to 59%. But the live signal beats a weekday-aware schedule only on bursty
traffic (by up to 27%). On human chat traffic the schedule is within 11% of
perfect information, and the present piggyback trails it.
The case study (§8.6) compares three policies on predictable and on
unpredictable traffic (the same, with unforeseen surges and dips). Energy-aware
routing, with a timetable or live, saves 31–93% over RecServe. On predictable
traffic the timetable is enough. On unpredictable traffic the OLT's cost,
broadcast to every ONU on the PON, adds up to 13–29% over the timetable, within
about three points of perfect information.

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

**What is new and what is not.** Carrying server state back on responses is an
established technique: C3 [22] piggybacks each replica's queue length and
service time on its responses so that clients can pick replicas without
probing. What this case study adds is the quantity and the setting: energy
rates, which *fall* with load because of batching, reported across an LLM
cascade over a PON, where they decide how far a query travels rather than which
replica serves it.

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
| Controls | Three-tier RecServe; a fixed two-tier chain with the ONU dropped; a static configuration holding the OLT's day-average rate, calibrated on the true load, and stale versions calibrated at ¼ and 4× the load; a **time-of-day schedule**, the OLT's mean rate in each hour of the day, calibrated on the true load; an oracle that knows the current expected rate | The last five share piggyback's routing rule and differ only in which OLT rate they see |
| Load drift | The OLT's load is BurstGPT's average day times a multiplier: exp of an Ornstein–Uhlenbeck process, mean 1, correlation time 12 h, log-sd σ = 0 (the average day exactly), 0.25 or 0.5, over 14 days. It stands for busier and quieter days and surges that a schedule set in advance cannot follow | Without drift the schedule *is* the oracle at hourly resolution, so a live signal cannot be told apart from a well-kept schedule; σ is swept, not measured |
| Accounting | **Average** (primary): a query pays its share of the OLT batch's energy, the measured J per token at that batch. **Marginal**: it pays what it adds to the network (§8.3). User and ONU are priced the same under both | Average follows Google's per-prompt attribution [18]; marginal answers "how much does the network's total energy change" |
| Frontier | J per query to reach *at least* the target accuracy: interpolated between Pareto points, and a policy whose least accurate point is above the target is charged that point | A policy that is cheaper and more accurate dominates; it is not "out of range" |
| Noise | Common random numbers (every policy meets the same batches); the 306 J, 61 J and 15 J ONU runs, and every drift and marginal run, repeated over three seeds | |

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

**On the average day, a time-of-day schedule does as well.** When the OLT's load
follows BurstGPT's average day exactly, the schedule, a table of the OLT's
mean rate for each hour, matches piggyback within ±1% at every load and both ONU
costs (three seeds, 0.80 accuracy; at the 61 J ONU, load 16: 128.6 J schedule,
128.7 J piggyback, 127.7 J oracle). This follows from the setup: with no drift,
the schedule is the oracle at hourly resolution. The results above therefore
show that the cheapest topology changes with the hour, not yet that it takes a
live signal to follow it. That needs a load that departs from the average day.

**When the load drifts off the average day, the live signal edges ahead, by a
few percent.** Over 14 days with the load multiplied by a drifting factor
(log-sd σ, 12 h correlation), J per query at 0.80 accuracy, mean of three seeds:

| ONU | Drift | Load 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 306 J | σ 0.25 | ≈ 0 | ≈ 0 | +0.3% | +1.4% |
| 306 J | σ 0.5 | ≈ 0 | +0.2% | +1.1% | +3.3% |
| 61 J | σ 0.25 | +0.9% | +0.2% | +0.5% | +1.0% |
| 61 J | σ 0.5 | +2.0% | +2.3% | +2.5% | +3.7% |

Piggyback's saving over the time-of-day schedule. At the 61 J ONU with σ 0.5 it
is positive in every seed at loads 4–32 (range +0.1% to +4.6%).

- **The ceiling is low.** Perfect live information, the oracle, beats the
  schedule by at most 5% here (61 J ONU, σ 0.5, load 32: 86.3 J against
  90.4 J), and piggyback recovers most of it (87.0 J). Under average
  accounting the OLT's cost per query changes smoothly with load, so a schedule
  that knows the typical day is already close.
- **Against the fixed chains the drift changes little.** Piggyback still
  saves 6–9% over the better fixed chain at the 61 J ONU and loads 8–32, as on
  the average day.
- **The largest figure, 16% at a 15 J ONU and load 32, holds across three seeds**
  (16.3%, 17.3%, 16.7%) against the fixed chains, but is ≈ 0 against the
  schedule (+0.7%, −0.2%, −0.3%).

What these results mean for the case study is drawn together in §8.4, after
the second way of counting the OLT's energy.

### 8.3 Marginal accounting

§8.2 charges an OLT query its **share** of the batch's energy, as Google's
per-prompt accounting does [18]. That answers "what does this query cost", not
"how much does the network's total energy change if it goes there". The OLT is
on and serving other PONs whether or not the query arrives, and its card sits
at its power limit from batch 1 (§3.2). So a query sent to a busy OLT adds only
the extra step time it causes.

**The marginal cost.** A query that makes the batch size b adds:

- if the OLT was idle (b = 1): the energy above idle, i.e. the net-of-idle
  rates at batch 1, 0.020 J per prompt token and 1.46 J per generated token at
  the card;
- otherwise: the slope of the batch's energy per step, b × rate(b), against b,
  fitted over batches 1–64: 0.0138 J per prompt token and 0.0096 J per
  generated token at the card.

That is 4.1 J per query at the card on a busy OLT, matching the 3.6–4.5 J
measured between adjacent batches (§3.3), or 10 J at the whole-system
boundary. The first query on an idle OLT costs about 900 J. The same × 2.47
boundary factor is applied, although the idle-capacity share in it is
arguably sunk, which would make the marginal figure lower still. User and ONU
are priced as before; the ONU's figure includes its idle draw, so for an
always-on ONU it is an upper bound on its marginal cost, a case the ONU-cost
sweep already covers.

**Choosing the topology is worth far more.** A busy OLT now answers more cheaply
than the phone itself, so at busy hours the cost-aware policies skip both lower
tiers. At 0.80 accuracy, three seeds, piggyback saves 40–54% over the better
fixed chain at loads 8–32, for both ONU costs, and up to 95% over three-tier
RecServe.

**But a single answer's report is the wrong signal.** Under this accounting a
report is heavy-tailed: a query that found the OLT idle reports ~900 J, the
next one ~10 J. One such report moves the lower tier's running average (weight
0.05) far off for dozens of answers. With per-query reports piggyback loses to
the time-of-day schedule by up to 26% on the average day (61 J ONU, load 8:
21.4 J against 16.9 J) and up to 18% with drift.

**The improved report.** The OLT serves every PON, so it can report its own
mean rate over the last 5 minutes of traffic instead of one query's rate: in
the simulation, the mean over Poisson(arrival rate × 5 min) arrivals, each
meeting its own batch. The lower tiers then weight the reported rate 0.3
(`--report window --rate-alpha 0.3`); answer lengths and escalation rates keep
0.05. Piggyback's saving over the time-of-day schedule, 0.80 accuracy, mean of
three seeds:

| ONU | Load | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| 306 J | average day | +0.1% | −2.9% | −2.2% | +0.9% | −0.6% |
| 306 J | drift σ 0.5 | +1.4% | +9.1% | +8.8% | +3.1% | +2.0% |
| 61 J | average day | +0.4% | −3.0% | −6.9% | −4.3% | −0.7% |
| 61 J | drift σ 0.5 | +3.7% | +12.0% | +11.9% | −0.9% | +1.6% |

- **With drift it beats the schedule by up to 12%**, positive in every seed at
  loads 4–8 for both ONUs (61 J: +11.1% to +13.1% at load 4).
- **On the average day the schedule still wins, by up to 7%** (61 J, load 8,
  seeds −11% to −1%). The lower tiers hear the OLT only through their own
  cascade's answers, about 40 an hour, so their estimate trails the morning
  drop in cost. A schedule has no lag.
- **Under share accounting the improved report helps a little too**: +2–5% over
  the schedule with drift at the 61 J ONU, loads 4–32 (per-query reports:
  +2–4%), and ≈ 0 on the average day.

### 8.4 What this means for the case study

The mechanism works. Piggyback keeps RecServe's three tiers and RecServe's
escalation rule, and uses the answers already flowing back to decide, hour by
hour, whether the ONU is worth using or should be skipped. It never needs to be
told which topology is right: it matches the better fixed chain wherever one
topology wins all day, and beats both where the answer changes with the hour.
The ONU-dropped chain is a control, not a proposal. It works only if someone
already knows the ONU is not worth using.

What a live signal is worth depends on the baseline:

- **Against fixed chains**, 5–17% with an ONU below about 60 J per query, under
  share accounting, and 40–54% under marginal accounting, where a busy OLT
  undercuts even the phone. With the Orin Nano Super as published and share
  accounting, nothing: the ONU is not worth using at any hour.
- **Against a time-of-day schedule**, which knows the typical day, only what the
  day's departures from it are worth: ≈ 0 on the average day, up to 5% with a
  ±50% drift under share accounting, up to 12% under marginal accounting. And
  only with the improved report; per-query reports lose to the schedule under
  marginal accounting.

So the claim the thesis can make is conditional. Energy piggyback replaces a
per-deployment configuration, and beats a well-kept schedule when the load is
not predictable, most clearly when energy is counted as what each query adds.
The packet should carry the OLT's recent mean, not one query's rate.

§8.5 reruns the comparison on real load, with realistic households and
consistent marginal accounting, and narrows this further.

### 8.5 The baseline on real load

§8.2–8.4 ran on an average day, with a drift whose size was assumed, one ONU
carrying all the cascade's traffic, and marginal accounting that charged the
ONU for its idle draw but not the OLT. All three were checked against data
and replaced:

| What | Before | Now |
|---|---|---|
| OLT load | BurstGPT's average day, all traffic (89% API), optionally × a synthetic drift, σ 0.25 / 0.5, 12 h | BurstGPT's own hourly counts replayed [12]: the static configuration and the schedule are calibrated on its first 30 days, and every policy runs on the 31 after. The schedule knows the hour of day and weekday/weekend |
| Households | one ONU with 2,000 queries a day, hearing a report on almost every answer | the same traffic split over 1, 10, 40 or 100 households (2,000 down to 20 queries a day each), each learning the OLT's cost from its own answers only |
| Marginal accounting | OLT net of idle × 2.47; ONU gross, idle included | no tier pays for being on: OLT × 2.20 (the idle-machines share left out), ONU 1.11 − 4.7 W ÷ 9.37 tok/s = 0.61 J per token (168 J a query) [23], phone unchanged |

RecServe's confidence thresholds stay one per tier, shared by all households.
Everything else is §8.1.

**What real load looks like** (`implementation/data/load_traces/drift_fit.json`):

| Trace | Requests an hour | Peak / trough | Drift around the schedule: σ | τ |
|---|---|---|---|---|
| BurstGPT, conversation log | 105 | 46× | 0.61 | ~1–2 h |
| BurstGPT, all traffic | 977 | 12× | 1.91 | ~3 h |
| Azure 2024, conversation service [24] | 171,753 | 1.8× | 0.24 | 14 h |

σ is the log-sd of load over schedule, net of counting noise; BurstGPT's is
out of sample (schedule from the first 30 days), Azure's in sample over its six
whole days with an hour-of-day schedule, too short for more.

- **People chatting follow office hours and drift fast.** BurstGPT's
  conversation traffic swings 46× over the day, with a lunch dip, and wanders
  around a weekday-aware schedule by σ ≈ 0.6 with a correlation time of 1–2
  hours, not the 12 h assumed before. Its API traffic is batch jobs, bursting
  to 8.6× the calibrated peak.
- **A large, global service is smooth.** Azure's conversation service, 1,600
  times BurstGPT's volume across time zones, is nearly flat over the day and
  drifts slowly. An OLT, one city's households, sits between the two. The
  synthetic drift of §8.2 (σ 0.25, 12 h) is close to Azure's.

**Results.** Piggyback's saving over the schedule, and in brackets the oracle's
(the most any live signal could save), marginal accounting, 0.80 accuracy, mean
of three seeds (`implementation/results/energy_tests/trace/SUMMARY.md`):

| Traffic | Households × queries a day | Load 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| conversation | 1 × 2,000 | −0.3% (+0.1%) | −5.5% (−3.2%) | −3.9% (+3.7%) | +1.5% (+10.7%) | −2.8% (+3.1%) |
| conversation | 10 × 200 | −0.6% (−0.2%) | −6.2% (−2.8%) | −21.0% (+3.8%) | −7.7% (+7.1%) | −5.2% (+4.0%) |
| conversation | 40 × 50 | −1.0% (−0.2%) | −9.5% (−2.1%) | −15.9% (+4.9%) | −11.7% (+3.7%) | −1.9% (+1.0%) |
| conversation | 100 × 20 | −1.1% (−0.2%) | −9.5% (−1.2%) | −10.5% (+6.0%) | −3.6% (+2.4%) | −0.5% (+0.3%) |
| all | 40 × 50 | +10.0% (+19.4%) | +18.4% (+37.7%) | +12.7% (+29.4%) | +4.0% (+15.8%) | −0.1% (+3.4%) |

ONU at 168 J. At a 34 J ONU the pattern is the same and stronger: on
conversation traffic piggyback trails the schedule by up to 38% (40 × 50,
load 16) against an oracle ceiling of at most +11%; on all traffic it beats it
by up to 27% (load 4), against a ceiling of 53%. Under average accounting the
conversation traffic leaves nothing to gain (oracle within ±1% of the
schedule; piggyback −7% to +1%), and all traffic up to 9% at the 61 J ONU
(ceiling 17.5%).

- **Routing by energy still beats RecServe's fixed chain.** With the schedule
  as its cost source, it saves 18–59% over the better fixed chain at loads 4–32
  under marginal accounting (conversation, 1 × 2,000), and 5–11% at loads 8–32
  under average accounting at the 61 J ONU. The cascade should decide where a query goes,
  not only whether.
- **On human traffic, a weekday-aware schedule is close to perfect
  information.** The oracle beats it by at most 11% (marginal) and about 0
  (average). That is the ceiling for any live signal on this traffic.
- **The present piggyback does not reach that ceiling.** It trails the schedule
  in almost every conversation scenario, more as households get sparser and the
  ONU cheaper. Its only input is its own household's answers: at 50 queries a
  day, a few OLT reports an hour, each weighted 0.3 however old it is.
- **Where load is unpredictable, the live signal pays.** On all traffic, whose
  API bursts no schedule anticipates, piggyback beats the schedule by 10–27% at
  loads 2–8, where the replayed load mostly stays within the measured batch
  range (§9).
- **Sparse households cost every cost-aware policy, the oracle included.** From
  one household of 2,000 queries a day to 100 of 20, the oracle's cost at 0.80
  rises from 92.7 J to 110.5 J (average accounting, 306 J ONU, load 32), 13% above
  simply dropping the ONU (98.0 J). The OLT signal is not the cause. Each
  household learns escalation rates and answer lengths from its own few answers,
  and runs as plain RecServe until 20 OLT reports have arrived. Those statistics
  describe the questions, not the household, so they could be shared; that is
  untested.
- **"Oracle" means perfect OLT cost, not perfect routing.** It sometimes trails
  the schedule by up to 3% (conversation, load 4), because the rest of its
  decision is as noisy as the others'.

**What this means for the case study.** Two claims now separate. *Energy-aware
routing* beats RecServe's one-tier-up rule, on real load and under both
accountings. *Learning the cost live from piggybacked reports* beats a
weekday-aware schedule only when the load is bursty. On human chat traffic the
schedule is within 11% of perfect information, and the present piggyback trails
it. The policy still to be designed has to do three things: share the question
statistics instead of learning them per household; weight a report by its age,
not its count; and be judged against the schedule on both kinds of traffic.

### 8.6 The case study: three policies, two kinds of traffic

§8.1–8.5 consolidated into the one comparison the thesis makes. Every setting
is in `implementation/config/study.yaml`, with the reason for its value; the
runs come from `src/scripts/run_study.sh` and the tables from
`src/scripts/summarize_study.py` (`implementation/results/energy_tests/study/SUMMARY.md`).

**Policies.** Three, a reference and two controls:

| Policy | Where the OLT's cost comes from | Role |
|---|---|---|
| RecServe | nowhere: one tier up, always | the baseline [1] |
| Timetable | the OLT's mean cost for each hour, weekday and weekend, from the first 30 days | energy-aware routing on the best information available in advance |
| Dynamic | the OLT's own 5-minute mean, broadcast to every ONU on the PON every 10 s | energy-aware routing on live information |
| Oracle | the true expected cost at that moment | the most any live signal could achieve |
| RecServe without the ONU | nowhere | control: is the saving only from dropping the ONU? |
| Piggyback | the OLT's 5-minute mean, on the household's own answers only | ablation: why broadcast |

The energy-aware policies share one routing rule (expected cost to completion,
§8.1) and the question statistics it needs, answer lengths and escalation rates,
pooled across households: they describe the questions, not the household, and an
operator can aggregate them. A household knows its own devices' costs. So the
energy-aware policies differ only in where the OLT's cost comes from.

**Why broadcast.** A PON's downstream is a broadcast: every frame reaches every
ONU, which keeps only its own GEM ports [25]. An answer is encrypted for its
household, so a report riding on it serves that household alone, and a household
of 20–200 queries a day misjudged the OLT's cost by 80–290% (§8.5). The PON's
downstream multicast channel, the one IPTV uses, reaches every ONU, asked or not.
A few dozen bytes every 10 s is negligible on a 1.25–10 Gb/s link and adds no
message per ONU. It needs OLT and ONU software that does not exist today, as does
an ONU with an LLM accelerator.

**Traffic.** *Predictable*: BurstGPT's conversation log as recorded (§8.5).
*Unpredictable*: the same, plus events no timetable or calendar knows about, such
as a news surge, a big game running long, a neighbouring OLT's traffic moved here,
or an outage elsewhere. They arrive about once a day (Poisson), last 3 hours, and
multiply the OLT's load by a factor or divide it by that factor with equal
probability, affecting about 12% of the time; the factor is swept from 1.5 to 5.
Events fall on the training days too, so the timetable learns an average that
includes them. The cascade's own queries are unchanged: the events are other
traffic at the OLT. Holidays are not used: they are on the calendar, and a fair
timetable would include them.

**Settings.** Marginal accounting, ONU 168 J, 40 households of 50 queries a day,
0.80 accuracy, three seeds (§8.5). Each sensitivity run changes one setting, at
factors 1 and 3. The OLT's report is modelled over its window, each arrival at the
load of its own moment, so it lags by up to 5 minutes (§8.5's runs took the load at
the moment of the report).

**Saving over RecServe**, timetable / dynamic, at equal accuracy:

| Traffic | Load 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| predictable | 31% / 31% | 60% / 58% | 85% / 85% | 91% / 92% | 93% / 93% |
| surprises × 1.5 | 31% / 31% | 59% / 58% | 85% / 85% | 91% / 92% | 93% / 93% |
| surprises × 2 | 31% / 32% | 58% / 58% | 84% / 84% | 91% / 92% | 92% / 93% † |
| surprises × 3 | 31% / 32% | 57% / 57% | 82% / 84% | 90% / 92% | 92% / 93% † |
| surprises × 5 | 31% / 32% | 54% / 57% | 77% / 84% | 88% / 92% † | 91% / 93% † |

**Saving over the timetable**: dynamic (oracle, the ceiling) · piggyback:

| Traffic | Load 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| predictable | −0.4% (0.0%) · −1% | −6.0% (−3.3%) · −15% | −0.4% (+3.2%) · −47% | +8.7% (+9.6%) · −45% | +3.6% (+3.9%) · −34% |
| surprises × 1.5 | −0.1% (+0.4%) · −1% | −3.9% (−1.1%) · −13% | −0.7% (+3.2%) · −49% | +8.1% (+9.7%) · −46% | +2.7% (+3.3%) · −36% |
| surprises × 2 | +0.7% (+1.0%) · 0% | −1.6% (+0.9%) · −11% | +3.2% (+6.5%) · −43% | +8.2% (+9.9%) · −49% | +4.5% (+4.7%) · −33% † |
| surprises × 3 | +1.6% (+2.1%) · 0% | +1.7% (+4.1%) · −8% | +13.0% (+16.1%) · −31% | +13.1% (+15.0%) · −44% | +6.6% (+7.0%) · −34% † |
| surprises × 5 | +2.1% (+2.8%) · 0% | +5.8% (+8.4%) · −5% | +28.8% (+32.0%) · −12% | +26.7% (+27.4%) · −32% † | +13.2% (+14.3%) · −32% † |

Mean of three seeds; † the OLT's load exceeded the measured batch range (64 in
service) for over 1% of arrivals, indicative only.

- **Energy-aware routing is the main saving.** The timetable and the dynamic
  policy use 31–93% less energy than RecServe at the same accuracy, more the
  busier the OLT. It is not only dropping the ONU: at loads 8–32 they use about
  half what RecServe without the ONU does (predictable traffic, load 8: 25.5 J
  against 51.2 J), because a busy OLT answers more cheaply than the phone itself.
- **On predictable traffic a timetable is enough.** The dynamic policy is within
  −6% to +9% of it, and even perfect information would add at most 10%.
- **On unpredictable traffic the live signal pays, and more the bigger the
  surprise.** Over the timetable, dynamic saves up to 13% with surprises of × 3
  and up to 29% with × 5, largest at loads 8–16. On BurstGPT's real API-inclusive
  traffic it saves 22–56% at loads 2–8 (`alltraffic`).
- **Broadcast captures nearly all of it.** The dynamic policy stays within about
  three points of the oracle everywhere. Its OLT cost is off by a median 26% at
  load 8, piggyback's by 210–340%.
- **Piggyback on answers alone does not.** It trails the timetable by up to 49%.
  With one household of 2,000 queries a day (misjudging the OLT's cost by only 18%)
  it recovers much of the gain (+8% against dynamic's +13%, surprises of × 3,
  load 8). The difference is how often a household hears the OLT, and that is what
  broadcast fixes.
- **Household size no longer matters.** With surprises of × 3, dynamic's saving
  over the timetable is +1.1–13.7% for 1 × 2,000, 10 × 200, 40 × 50 and 100 × 20
  households alike.
- **Pooling the question statistics is part of the design.** Learned per
  household instead, every energy-aware policy loses ground: at load 8 the saving
  over RecServe falls from 85% to 78%, and dynamic's lead over the timetable with
  surprises of × 3 from up to 13% to up to 9%.
- **A cheaper ONU makes live information worth more.** At 34 J a query and
  surprises of × 3, dynamic saves up to 16% over the timetable (ceiling 20%).
- **Under average accounting there is nothing to follow.** A query's share of the
  batch changes smoothly with load, so the timetable already knows it: dynamic is
  within −0.4% to +1.9% of it even with surprises of × 3. Energy-aware routing still
  saves 16–70% over RecServe.
- **The rule is not optimal.** On predictable traffic at load 4 even the oracle
  trails the timetable by 3%. Its cost to completion assumes the tiers above step
  up one at a time, and it uses pooled averages, so perfect OLT cost is not a
  perfect route.
- **RecServe's thresholds keep their meaning.** Skipping changes how many queries
  a tier sees, not which kind: the skip depends on the hour and the OLT's cost, not
  on the question. The phone still escalates β of what it answers (20%, 49–50%
  and 79–81% at β 0.2, 0.5 and 0.8), and the ONU sees the same confidences as under
  RecServe (mean 0.884, 0.892 and 0.897 either way). But the ONU now answers
  0.1–2% of queries instead of 20–80%, so its window fills slowly and its
  escalation rate strays from β, by up to 14 points at β 0.8 (66–71%). The totals
  include this (`src/scripts/check_beta_windows.py`; surprises of × 3, loads 8
  and 32).

**What the thesis can claim.**

1. *Energy-aware routing*, deciding where a query goes by energy and not only
   whether it escalates, cuts RecServe's energy by 31–93% at the same accuracy on
   real household traffic when each query pays what it adds, and by 16–70% when it
   pays its share of the batch.
2. *On predictable traffic* a timetable carries that saving; live information adds
   nothing.
3. *On unpredictable traffic* the OLT's cost, broadcast on the PON, adds up to 13%
   (surprises of × 3) to 29% (× 5) over the timetable, within about three points
   of perfect information, for any household size. Piggyback on answers alone
   cannot, because a household hears the OLT too rarely.

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
  out; the grid in §8.2 is one seed per cell (three for the 306 J, 61 J and
  15 J rows). The load drift is a swept assumption (σ, 12 h correlation), not
  fitted to a trace.
- **No herding, no capacity limit.** Every ONU reads the same report, so in a
  real network they would all skip to the OLT together. For energy that is
  benign, since a fuller batch is cheaper per query, but it raises the OLT's
  latency and nothing in the rule caps its load. A deployment needs a latency
  or occupancy limit in the routing rule.
- **The replayed API bursts exceed the measured batch range.** On BurstGPT's
  all traffic, load reaches 8.6× the calibrated peak: at peak loads 16–32 that
  is 138–276 queries in service, where one L4 could not hold them and the curve
  is measured only to 64 (the simulation charges batch-64 rates). Those rows
  are indicative only; conversation traffic stays within 1.4× (46 in service).
- **The replay keeps the trace's counting noise.** BurstGPT's conversation log
  has ~105 requests an hour, so part of its hour-to-hour variation is Poisson
  noise, scaled up with the load (σ 0.66 raw against 0.61 net). It slightly
  handicaps the schedule.
- **The surprises are assumed.** How often unforeseen events come (one a day),
  how long they last (3 h) and how large they are (× 1.5 to × 5, swept) are not
  fitted to data; the results are read as a function of their size. At × 2 and
  above they push the OLT beyond its measured batch range at the highest loads
  (the † rows of §8.6).
- **The broadcast is not standardized.** A PON's downstream multicast channel
  exists [25], but an OLT that reports its energy on it, and an ONU agent that
  reads it, would be new software.
- **One confidence window per tier, shared by all households.** A real ONU keeps
  its own. Skipped most of the day, it would see a few queries a day and take
  months to fill a 1,000-answer window, so its threshold would be noisy; the
  operator would have to calibrate it from the population, which the shared
  window stands for, or pool it like the question statistics. The question mix
  does not vary by hour here; if it did, a tier used only at some hours would
  calibrate on those hours' questions. With more than three tiers, a middle tier
  that receives queries forwarded past the tiers below would see easier questions
  than the ones escalated to it, which shifts its β-quantile; windows would then
  have to be kept per arrival path.

## 10. Reproducing

```bash
cd implementation
.venv/bin/modal run src/modal_apps/measure_gpu_energy.py     # OLT sweep, ~15 min on one L4
.venv/bin/modal run src/modal_apps/collect_answers.py        # all tiers' answers, 1,319 queries
.venv/bin/python src/scripts/check_confidence.py results/energy_tests/<answers>.raw.jsonl
# every simulation default, with what it means and where its value comes from:
# config/sim_piggyback.yaml (flags override it for one run)
.venv/bin/python src/scripts/sim_piggyback.py                # piggyback simulation (§8), ~30 s
.venv/bin/python src/scripts/sim_piggyback.py --boundary gpu
for s in 0.05 0.1 0.2 0.5; do .venv/bin/python src/scripts/sim_piggyback.py --onu-scale $s; done
# load drifting off the average day, 14 days (§8.2), and marginal accounting (§8.3); add --seed 8 / 9
.venv/bin/python src/scripts/sim_piggyback.py --load-sigma 0.5 --days 14 [--onu-scale 0.2]
.venv/bin/python src/scripts/sim_piggyback.py --accounting marginal [--load-sigma 0.5 --days 14]
# the trace-driven baseline (§8.5): hourly load from BurstGPT (and Azure 2024 for the drift check),
# then 60 runs (~30 min at 10 in parallel) and their tables
.venv/bin/python src/scripts/prepare_load_traces.py --azure  # writes data/load_traces/
bash src/scripts/run_trace_baseline.sh 10
.venv/bin/python src/scripts/summarize_trace_runs.py         # results/energy_tests/trace/SUMMARY.md
# the case study (§8.6): config/study.yaml, 54 runs (~1 h at 10 in parallel; keep the lid open)
bash src/scripts/run_study.sh 10
.venv/bin/python src/scripts/summarize_study.py              # results/energy_tests/study/SUMMARY.md
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
22. Suresh, Canini, Schmid and Feldmann. *C3: Cutting Tail Latency in Cloud Data Stores via Adaptive Replica Selection.* NSDI 2015.
23. NVIDIA Developer Forums. *Reducing idle power on Orin Nano Super Dev Kit* (2026-01-23): 4.7 W idle, tegrastats VDD_IN, 7 W mode. https://forums.developer.nvidia.com/t/reducing-idle-power-on-orin-nano-super-dev-kit/358482
24. Stojkovic et al. *DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency.* HPCA 2025. Azure LLM Inference Dataset 2024, https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md
25. Cisco. *Understand GPON Technology* (downstream broadcast, GEM port filtering, AES per ONU, downstream multicast GEM ports); ITU-T G.984.3 (GPON) and G.9807.1 (XGS-PON). https://www.cisco.com/c/en/us/support/docs/switches/catalyst-pon-series/216230-understand-gpon-technology.html
