# Energy-aware offloading policy — design and results

**Status (2026-08-30): investigation complete, result is negative.** The
proposed mechanism — weighting RecServe's confidence threshold by an energy
cost — turned out to be *redundant with RecServe's existing β parameter*.
See **§14** (in Portuguese) for that result and what to do about it.

This document is now a record of the investigation, not a proposal. §1–§11
define the mechanism and the reasoning behind it (needed to understand §14);
§12–§14 are the measured results.

## Summary — the idea, in plain terms

RecServe (and Pakpahan & Hwang's PON architecture, which this repo follows)
has a simple rule: at each tier — user, ONU, fog, cloud — the local model
answers and reports how confident it is. Confident enough, it stops there.
Not confident enough, the query moves up a tier. The bar for "confident
enough" isn't fixed: each tier calibrates it from the β-quantile of its own
recent confidence history. But that bar has never had anything to do with
energy — a query escalates purely because the model was unsure, no matter
what that escalation costs.

**The idea was: add one number, λ, answering "how many joules is one extra
percentage point of confidence worth?"** With λ and the measured per-tier
energy costs, each tier's bar gets nudged: cheap hops barely move it,
expensive hops lower it (don't demand near-certainty when escalating is
costly). λ=0 recovers plain RecServe exactly, so it only ever added an
option.

**Why it failed:** with a *static* cost per tier, `exp(−λ·cost)` is a
constant multiplier on the threshold, and scaling a quantile by a constant
just yields a different quantile — the same knob β already provides. Three
variants (static, per-hop, per-query) all measured within noise of tuning β
alone. Details and the algebra in §14.

**What the work produced anyway**, and what stands independent of the
negative result:

- a cross-tier energy characterization (verified literature + first-party
  RAPL measurements) — `config/layer_energy.yaml`
- RecServe validated on a generative task at scales its authors never tested
  (they topped out at 355M encoder models; this runs 1B–10.7B decoders) — §13
- an empirical test of RecServe's own Assumption 1: confidence tracks model
  capacity but **not** task difficulty — §13.3
- `exp(min logprob)` separates correct from wrong better than their specified
  metric, replicated across three model scales — §13.2
- non-monotonic capability ladders silently corrupt stepwise recursion — §13.1

## 1. What the base architecture gives us

- **The escalation rule is local and stepwise.** Each tier compares its own
  confidence against a threshold derived from a sliding window of its own
  history, and escalates exactly one tier up. It never reads another tier's
  state.
- **The base paper's SDN controller knows everything and deliberately doesn't
  use it here.** Pakpahan & Hwang's ONOS controller keeps an "LLM-capability
  registry" per ONU — accelerator type, model cache size, *energy
  constraints* — yet the paper states: *"Although the ONOS controller is
  aware of all registered devices and their capabilities, the system always
  follows stepwise escalation to preserve locality and minimize network
  usage."* Any energy-aware extension has to reckon with that stated goal.
- **`config/layer_energy.yaml` has real per-tier energy data**, including
  batch-sensitivity curves for fog (12 points, RTX 4090) and cloud (13
  points across Samsi, Oviedo, MLPerf v5.1, Caravaca).

## 2. Rejected: live cross-tier telemetry

Considered and rejected: each tier reports its current J/token to the SDN
controller, which broadcasts it to all tiers.

- **Works against the base paper's stated design goal** (locality, minimal
  network usage). Pakpahan's Table 1 scores RecServe high on bandwidth
  efficiency *because* it's decentralized, and PerLLM lower because it needs
  a central controller. A periodic broadcast moves this design toward PerLLM.
- **No timescale data exists to size the broadcast period.** Every energy
  number gathered is a static snapshot at a declared batch; there is no
  time-series showing how fast a tier's load actually changes.
- **A bare J/token number is ambiguous without its regime** — it varies ~3×
  (fog) to ~100× (cloud) with batch alone, and a received number doesn't say
  whether it reflects stable hardware cost or a momentary spike.

## 3. Local-only energy signal: valid, but for a narrower question

A tier tracking its *own* recent energy needs no coordination, but local
inference always runs first — by the time any decision is made, this query's
local cost is already sunk. So a local signal can only inform "should I
*also* pay to escalate", not "should I run locally at all" (the latter would
need a pre-inference admission gate, which the base paper doesn't have).

**Why local-only can't minimize total system energy:** the between-tier gap
dwarfs and can invert what a local window sees.

| Tier | Typical J/token |
|---|---|
| User (best backend) | 0.074 – 0.21 |
| ONU (best precision) | 0.22 – 1.89 |
| Fog (production) | 0.38 |
| Cloud (MLPerf, batched, FP4) | 0.094 – 0.097 |
| Cloud (production, BF16) | 0.40 |

Cloud's best regime beats fog's production point. Fog's own local swing is
~3× while cloud's full range spans ~2700× depending on regime — a local tier
cannot see which side of that range its target currently sits on.

Local-only remains valid for a *different* objective: tier self-protection
(battery, thermal, load-shedding), where the point was never to reason about
the target tier's cost. Note that objective can call for the *opposite*
threshold adjustment, so which one is being served must be stated.

## 4. The mechanism

Give the β-quantile threshold an explicit joules-per-quality-point exchange
rate, using **static** per-tier-pair cost data from `layer_energy.yaml`:

```
Escalate iff  λ × (expected quality gain)  >  (static joule cost of this hop)
```

Composed with RecServe's Eq. 2 as a single threshold check, not a second
gate:

```
Serve locally  iff  C(x)  ≥  T(β) − cost/λ          [additive form]
```

`T(β)` is computed exactly as RecServe does; the energy term only offsets it.
Sign check: an expensive hop *lowers* the effective threshold (easier to
clear → more likely to stay local); a cheap or negative-cost hop raises it.
`cost=0` or `λ→∞` recovers Eq. 2 exactly.

> The additive form above is superseded as the default (Prof. Nazar,
> 2026-08-24): it is unbounded and its λ convention is inverted. The
> **exponential** form `T_eff = T(β)·exp(−λ·cost)` is bounded in `(0, T(β)]`
> and recovers the baseline at λ=0. All three candidate forms are in §11.1.

**What λ is, in units:** joules per percentage-point of confidence. Worked
example — bar `T(β)=85%`, query at 83% confidence, hop costs 0.38 J:

| λ (J per point) | bar shift | effective bar | outcome |
|---|---|---|---|
| 0.01 (generous) | 38 pts | 47% | stays local |
| 0.1 (moderate) | 3.8 pts | 81.2% | stays local |
| 1.0 (stingy) | 0.38 pts | 84.6% | **escalates** |

Same query, three outcomes, purely from what a joule is declared to be worth.
On a cheap hop (0.094 J) the bar barely moves at any λ — cheap hops stay near
plain-RecServe behaviour; expensive hops are where λ does work.

**Quality-gain proxy (never resolved):** either `1 − confidence` (cheap, but
an optimistic ceiling) or a per-tier-pair gain calibrated offline from trace
data. §14 shows why this mattered more than expected — the cost term carries
no information about escalation value, and this proxy was the only place that
information could have entered.

**Why per-tier-pair, not one global constant:** the sign of the escalation
cost isn't consistent across hops. `user→onu` is a real cost; `fog→cloud` can
be free or a net saving depending on cloud's regime.

## 5. How a tier learns its neighbour's cost — without telemetry

No runtime discovery needed: measure once offline (done —
`layer_energy.yaml`), ship the relevant slice to each tier as static
deployment config, reusing the base paper's existing ONU registration flow
(which already carries an "energy constraints" field), and refresh only when
hardware or model actually changes — never per query, never on a polling
loop.

## 6. Batch variation without going live

A single static point poorly summarizes a tier whose cost swings 3×–100× with
load. The fix is a small **static profile**, not a live signal:

- **Indexed by time-of-day bucket** — free to know locally, just a clock.
- **Cloud:** effectively one bucket near its production point (0.3–0.4
  J/token); continuous batching keeps hyperscale near-saturated in practice.
- **Fog:** 2–3 buckets from the measured RTX 4090 curve — conservative
  (~7–8.5 J/token, batch≈1) for likely-idle periods, cheaper (~2.7–3 J/token,
  batch 10–20) when likely loaded. Bias pessimistic: underestimating fog's
  cost is the costlier error.
- **Bucket selection stays local:** use the tier's own recent *escalation
  frequency*. A target tier's batch level is driven by aggregate escalation
  traffic from its children, and household traffic is time-correlated — so
  "how often am I escalating" is a causally grounded, zero-coordination proxy
  for "how loaded is my target", in a way local J/token history is not.
- **Unverified assumption:** presumes a day/night household-broadband traffic
  pattern — standard in telecom capacity planning, unmeasured here.

**Per §14 this is now the most important untested item**, not a refinement:
cost variability is the only thing that could break the β-degeneracy, and
this is the one source of it not yet tried.

## 7. Positioning against existing literature

**The general "trade quality against energy via a tunable threshold" pattern
is not novel** — confirmed from four directly-read sources. An earlier draft
of this section claimed no comparator was energy-aware; that was wrong and is
corrected below.

**Energy-aware comparators (all read directly, not from abstracts):**

- **PerLLM** (Yang et al., arXiv:2405.14636 — ref. [29] in Pakpahan's own
  bibliography) minimizes `Σ(ω_tran·E_tran + ω_infer·E_infer + ω_idle·E_idle)`
  under a latency deadline. *More* granular than this design's single
  exchange rate.
- **GreenServ** (Ziller et al.) routes across 16 LLMs trading accuracy against
  energy via an online multi-armed bandit (+22% accuracy, −31% energy vs
  random). Its reward `α·Accuracy − β·Cost` with `α=1−λ, β=λ` is the *same
  single-dial parameterization* arrived at independently here.
- **CR²** (Xue et al., arXiv:2605.12001) is if anything more rigorous:
  `c̄ₘ = ωₜ·tₘ/T₀ + ωₑ·eₘ/E₀` with real Watts and Joules.
- **EcoThink** (Li & Lu, arXiv:2603.25498) — constrained optimization,
  minimize energy subject to a quality floor. Its energy model uses hardware
  *TDP* (a max-power spec), not measured draw.

**Not energy-aware** (also read directly): **Tabi** (EuroSys '23) — pure
calibrated-confidence dispatcher; **EdgeShard** (IEEE IoT J. 2025) — latency
and throughput only. The fifth mechanism in Pakpahan's Table 1
("Draft-Verify") is **Hao et al., EdgeFM '24** (DOI 10.1145/3662006.3662067),
identified from Pakpahan's actual bibliography after an earlier guess (SLED)
proved wrong; its full text is paywalled, and "cost" there appears to mean
compute/token usage, so *likely* not energy-aware at lower confidence than
the others.

**The setting difference was verified, not assumed** (read into each paper's
system model):

- **PerLLM**: strict **two-tier** edge/cloud. Network is abstract bandwidth
  constants ("300 Mbps and 100 Mbps", ±20% noise) — no PON, TDMA, or
  wavelength allocation.
- **GreenServ**: **not multi-tier at all** — all 16 models on one A100 in one
  rack, routed by local async queueing. Its own §6.4 admits concurrency,
  batching, and queuing delays are unaddressed.
- **CR²**: strictly **two-tier** (UE + edge server), generic wireless channel
  (Shannon capacity, path loss, Rayleigh fading), not optical access.

None models a hierarchy deeper than two tiers, and none touches PON/SDN
access-network constraints. That is the checked basis for "different setting".

**Design idea worth borrowing:** PerLLM (CS-UCB) and GreenServ (LinUCB) both
*learn* their energy/quality tradeoff online rather than fixing it. Directly
relevant to this design's unresolved λ-calibration — and, given §14, possibly
to escaping the degeneracy, since a learned per-context weight varies in a way
a static constant cannot.

Cite PerLLM, GreenServ, CR² and EcoThink on this axis when the related-work
chapter is written; Pakpahan's Table 1 is a reasonable template to extend
with an energy column.

## 8. Scope decision: stay faithful to RecServe's recursion

**Decision (2026-08-24), and it still holds:** §4–§6 never change *which*
tier is next or let a query skip a hop — only *when* the current tier hands
off. λ=0 recovers the original rule exactly, so the design inherits
everything Pakpahan's Table 1 validated for RecServe (decentralized, high
bandwidth efficiency) without re-arguing it.

**Multi-hop / entry-point selection remains explicit future work.** Letting a
query start at a smarter entry tier (chosen once, before RecServe's
unmodified recursion runs) would capture sunk-cost savings for queries known
in advance to need a high tier — but it needs a second, harder-to-calibrate
prior (per-tier success likelihood by query type), and is honestly *wrapping*
RecServe rather than extending it. §13.1 gives it new motivation: with an
inverted tier, skipping would avoid real damage.

## 9. Open questions

- **λ calibration** — a value judgment, or learnable online via a bandit
  (§7). Largely moot given §14 unless the degeneracy is broken first.
- **Profile-bucket sizing** (§6) needs traffic-dynamics data this repo
  doesn't have — every energy number is a static snapshot.
- **Day/night traffic assumption** behind §6's fog buckets is unverified for
  this deployment context.

## 10. Where this plugs into the code

- `src/scripts/sweep_energy_policy.py` — the mechanism and the sweep
- `src/scripts/run_policy_matrix.py`, `run_generative_matrix.py` — collection
- `src/layers/ollama_layer.py`, `src/tasks/gsm8k.py` — generative cascade
- `src/scripts/measure_tier_energy.py` — first-party RAPL measurement
- `config/layer_energy.yaml` — per-tier costs and batch profiles
- `src/recserve/traced_recursive_serve.py` — the β-quantile mechanism being
  extended

## 11. Weighting function (Prof. Nazar, 2026-08-24)

Guidance: keep both signals, apply energy as a multiplicative or exponential
weighting factor rather than a raw offset, and sweep configurations.

Three candidate forms, all applied to the same `T(β)`:

| Form | Effective threshold | λ units | Baseline at | Bounded? |
|---|---|---|---|---|
| **A — additive** | `T(β) − λ⁻¹·cost` | J per confidence-point | λ→∞ | No — needs clamping |
| **B — multiplicative** | `T(β)·(1 − λ·cost)` | 1/J | λ=0 | No — clamps when λ·cost>1 |
| **C — exponential** *(default)* | `T(β)·exp(−λ·cost)` | 1/J | λ=0 | **Yes** — `(0, T(β)]`, no clamp |

C is the default: bounded by construction (no arbitrary clamp, which would
be an unjustifiable free parameter) and its effect saturates smoothly. Under
B and C, **λ=0 means energy is ignored and the mechanism is exactly plain
RecServe** — more intuitive than A's λ→∞ baseline, and matching GreenServ's
`α=1−λ, β=λ` convention. All λ results are reported under this convention.

§12.2 measured all three: they trace the same frontier, so C wins on
boundedness alone.

## 12. Results (run 2026-08-29)

Everything below was produced by `src/scripts/sweep_energy_policy.py` over
the full SST-2 test split (n=872), replaying the matrix from
`run_policy_matrix.py`. Commands and caveats are in those scripts'
docstrings; this section records what came out and what it supports.

### 12.1 The mechanism works, and the baseline check passes

At λ=0 the bounded forms reproduce plain RecServe *exactly* — same
accuracy, same final-tier distribution, same energy — which is the
correctness check §11.4 asked for. Raising λ shifts traffic down-tier
monotonically. So the mechanism does what §4 says, and the strict-superset
property holds in practice, not just on paper.

### 12.2 The weighting form does not matter; use the bounded one

Matched operating points across all three forms land on the *same*
frontier, differing only in how λ is parameterised:

| Form | λ | Accuracy | J/query |
|---|---|---|---|
| additive | 100 | 0.9346 | 2.662 |
| multiplicative | 0.01 | 0.9346 | 2.662 |
| exponential | 0.01 | 0.9346 | 2.680 |

This is the outcome §11.1 flagged as possible: since nothing is lost in
the tradeoff itself, **exponential should be adopted purely on its
engineering merit** — bounded in `(0, T(β)]`, no clamp, exact baseline at
λ=0. The additive form's unboundedness is a defect with no compensating
benefit. Consider the form question settled.

### 12.3 Per-tier standalone accuracy — and why n=40 was misleading

| Tier | n=40 | n=872 |
|---|---|---|
| user | 1.0000 | 0.9117 |
| onu | 0.9500 | 0.9438 |
| fog | 0.9750 | 0.9576 |
| cloud | 0.9750 | 0.9495 |

At n=40 the smallest model looked *most* accurate, making escalation
strictly harmful and the tradeoff degenerate. That inverted ladder was a
small-sample artifact and disappears at n=872. **Do not run this
experiment at small n** — it produces a qualitatively wrong picture, not
just a noisy one. One real inversion survives: cloud (deberta-large)
sits slightly *below* fog (roberta-large) on SST-2, so the top hop
currently buys nothing on this task.

### 12.4 The frontier, priced with locally measured energy

Using the RAPL measurements recorded in `layer_energy.yaml`'s
`local_measurement` block (exponential form):

| λ | Accuracy | J/query | Δ energy | Δ accuracy |
|---|---|---|---|---|
| 0 (baseline) | 0.9530 | 0.638 | — | — |
| 0.01 | 0.9518 | 0.440 | −31% | −0.1 pt |
| 0.025 | 0.9450 | 0.375 | −41% | −0.8 pt |
| 0.05 | 0.9404 | 0.339 | −47% | −1.3 pt |
| 0.1 | 0.9381 | 0.296 | −54% | −1.5 pt |
| 1.0 | 0.9163 | 0.231 | −64% | −3.7 pt |

The knee is sharp: **λ=0.01 gives 31% less energy for 0.1 accuracy points**
(one query in 872). Most of the saving lands before the accuracy cost
becomes visible.

### 12.5 What these numbers do and do not support

**Supported:**

- The decision rule is correct and behaves as designed (12.1).
- The choice of weighting form is settled empirically (12.2).
- A real, monotonic accuracy/energy frontier exists for this cascade, on
  measured joules, with a usefully sharp knee (12.4).
- The §4 "expensive tier degenerates toward pass-through" diagnostic is
  observable: with an oversized ONU (14B, 1.89 J/token) baseline energy
  jumps to 20.1 J/query and the mechanism routes around that tier far
  more aggressively (λ=0.02 drives everything local, vs λ=0.2 in the
  monotonic config).

**Not supported — do not carry these forward as thesis results:**

- **No specific λ value transfers.** λ=0.01 is right for *this* cost
  ladder on *this* hardware. The generative cascade's costs differ by
  orders of magnitude; λ must be re-derived there.
- **The tiers are 66M–400M encoder classifiers on one CPU**, not a
  1B→70B generative cascade on heterogeneous hardware. This validates
  the mechanism, not the target system.
- **SST-2 is nearly saturated** — a ~4.6-point spread between the
  best and worst tier is a narrow window to trade within. A harder task
  (imdb, yelp_polarity — both already supported by the harness) would
  give the frontier more room.
- **Ladder rung 2 (batched cloud) remains untested.** Only 34 of 872
  queries reach cloud at baseline, so cheapening cloud changes almost no
  decision. It needs a workload that actually escalates that far.
- **§6's batch-aware profile is still unbuilt**; every run so far uses one
  flat static cost per tier.

## 13. Generative cascade: first results (GSM8K, 2026-08-30)

First run of the generative track -- GSM8K chain-of-thought, 200 queries,
three tiers collected locally via Ollama (~10 h of CPU inference). This
replaces the classification harness's zero-output-token workload with one in
the decode-dominated regime the energy tables actually measure (median
~170-230 generated tokens per query).

Raw data: `results/traces/gsm8k_generative.{raw,matrix}.jsonl` -- generated,
not version-controlled, but reproducible (temperature 0, fixed models). The
raw file also stores every per-token logprob and token string, so the
confidence analyses below are recomputable without re-running any model.

### 13.1 The capability ladder is NOT monotonic

| Tier | Model | Accuracy |
|---|---|---|
| user | llama3.2:1b | 0.425 |
| onu | llama3.1:8b | **0.860** |
| fog | solar:10.7b | **0.675** |

Paired, per-query (n=200):

| Hop | Fixes | Breaks | Net |
|---|---|---|---|
| user -> onu | 91 (79% of user's errors) | 4 | **+87** |
| onu -> fog | 7 (25% of onu's errors) | 44 | **-37** |
| user -> fog | 65 (57%) | 15 | +50 |

**`solar:10.7b` is worse at GSM8K than the smaller `llama3.1:8b`**, and not
marginally: escalating onu -> fog destroys 44 correct answers to rescue 7.
Parameter count does not track math-reasoning capability -- SOLAR-10.7B is an
older base model than Llama 3.1.

This matters more than a bad benchmark number because the cascade is
deliberately stepwise (§8): a query escalating past onu *must* traverse fog.
An inverted tier does not merely waste its own energy, it converts that
energy into wrong answers, and every higher tier inherits a degraded input.
Before the generative experiment can demonstrate the lambda mechanism, fog
needs a model actually better than 8B -- a model-choice problem, not a
mechanism problem.

The user -> onu hop, by contrast, is exactly the regime the mechanism needs:
escalation repairs 79% of the small tier's errors and breaks almost nothing.
The classification cascade never had this (all four tiers within 4.6 points).

### 13.2 exp(min logprob) beats RecServe's specified confidence, on every tier

Welch t, separating correct from incorrect answers (higher |t| = stronger):

| Confidence definition | user (1B) | onu (8B) | fog (10.7B) |
|---|---|---|---|
| **exp(min logprob)** -- weakest link | **+6.36** | **+7.16** | **+5.09** |
| stdev of logprobs -- dispersion | -5.28 | -5.58 | -5.33 |
| exp(mean logprob) -- *RecServe spec* | +4.62 | +4.75 | +4.86 |
| fraction of tokens with logprob < -1 | -4.60 | -4.27 | -4.73 |

RecServe's normalized-perplexity definition **works** (all |t| > 4.6, clearly
significant) -- an earlier n=20 pilot suggested otherwise and was wrong,
small-sample noise. But `exp(min logprob)` separates ~10-50% better and does
so on all three tiers, across an 8x model-size range. The mechanism is the
same in each case: a wrong answer betrays itself by its *worst* token, not by
its average fluency, which averaging over ~200 tokens washes out.

Tested and rejected: confidence over the answer region only (tokens after the
`####` marker) has **no signal at all** (t = -0.10 / +0.81). By the time the
model emits the final number it is already determined by the preceding
reasoning, so it is high-probability whether or not that reasoning was sound.

Adopting `exp(min logprob)` is a small, measured, replicated improvement to
RecServe's mechanism -- worth reporting as such rather than silently
substituting.

### 13.3 Confidence is blind to problem difficulty -- on every tier

Accuracy falls sharply with GSM8K's own gold difficulty label, while
confidence stays flat:

| difficulty | user acc / conf | onu acc / conf | fog acc / conf |
|---|---|---|---|
| 2 steps (n=71) | 0.56 / 0.8745 | 0.90 / 0.8989 | 0.79 / 0.9010 |
| 4 steps (n=39) | 0.31 / 0.8785 | 0.77 / 0.8926 | 0.54 / 0.8985 |
| 5 steps (n=19) | 0.16 / 0.8751 | 0.84 / 0.8929 | 0.42 / 0.9010 |

Confidence varies by under 0.01 across difficulty levels where accuracy
varies by 0.3-0.4, and on the user tier it drifts *upward* as problems get
harder. So the signal separates **correct from incorrect** well, but carries
no information about **how hard a query is**.

Consequence for the policy: escalation will be partially effective (it does
catch wrong answers) but cannot preferentially route the genuinely hard
queries upward. This is a property of perplexity-based confidence itself, not
of model scale -- it holds identically at 1B, 8B and 10.7B. It is a real
limitation of the mechanism this thesis extends, and it is visible only
because GSM8K supplies an objective per-query difficulty label.

### 13.4 What this changes

- **Blocked until fixed:** the fog tier needs a model stronger than
  llama3.1:8b before the four-tier generative sweep means anything.
  Candidate: `qwen2.5:14b` (Qwen2.5 is strong at math; energy tabulated in
  layer_energy.yaml, though measured on Jetson Orin rather than fog-class
  A30 -- a documented caveat, and a better trade than a broken ladder).
- **Adopt** `exp(min logprob)` as the confidence definition, citing §13.2.
- **Report** the difficulty-blindness (§13.3) as a limitation of the
  confidence mechanism, not of this implementation.
- Cloud tier (`qwq:32b`) still uncollected.

## 14. O termo de energia é redundante com o β do RecServe (2026-08-30)

Esta seção está em português porque é o resultado mais importante do
trabalho até agora e precisa ser discutido com o orientador.

### 14.1 Em termos simples: o que aconteceu

A proposta era: **na hora de decidir se escalona uma consulta para a camada
de cima, leve em conta quanto de energia isso custa.** Se está caro e
provavelmente não vale a pena, não escalona.

Testamos. **Não funcionou** — no sentido de que não adiciona nada. Os
resultados são iguais aos de simplesmente ajustar o β que o RecServe já tem.

O β do RecServe controla o quão "exigente" cada camada é antes de escalonar.
β baixo = escalona pouco. β alto = escalona muito. Descobrimos que o nosso
termo de energia faz **exatamente a mesma coisa**, só que por outro caminho.

### 14.2 Por que isso acontece (a explicação matemática)

A regra que propusemos é:

```
limiar_efetivo = T(β) × exp(−λ · custo)
```

Se o `custo` é uma constante (um número fixo por camada), então
`exp(−λ · custo)` também é uma constante. Chamemos de `k`:

```
limiar_efetivo = k × T(β)
```

Só que `T(β)` é um **quantil** da distribuição de confiança. Multiplicar um
quantil por uma constante dá... outro valor que também é um quantil, só que
diferente. Ou seja: `k × T(β) = T(β')` para algum β'.

**Multiplicar o limiar por uma constante é a mesma coisa que escolher outro
β.** É por isso que não adiciona nada — é o mesmo botão, com outro nome.

### 14.3 As três tentativas

Testamos três formas de fazer o custo variar, para tentar fugir dessa
redundância. Todas medidas na cascata generativa (GSM8K, 200 consultas,
Llama-3.2-1B → Llama-3.1-8B), comparando a fronteira de Pareto
(acurácia × energia) contra ajustar só o β:

| Tentativa | Como o custo varia | Ganho médio sobre β sozinho |
|---|---|---|
| 1. Custo estático | Constante por par de camadas | **+0.0039** |
| 2. Custo por salto | Diferente entre user→onu e onu→fog | **+0.0012** |
| 3. Custo por consulta | Proporcional ao tamanho da resposta | **+0.0026** |

Todos os ganhos são de ~0.3 ponto percentual de acurácia, com vários pontos
negativos. Isso é **ruído** com n=200, não efeito real.

**Por que a tentativa 2 falhou:** custo diferente por camada é replicável por
um β diferente por camada. Cada camada já tem seu próprio histórico de
confiança e seu próprio limiar, então nada impede ajustar β_i por camada.

**Por que a tentativa 3 falhou** (essa era a mais promissora): custo por
consulta *é* algo que nenhum β consegue expressar — duas consultas
simultâneas, mesma confiança, recebem limiares diferentes. Mas o custo varia
com o **tamanho da resposta**, e o tamanho da resposta não tem relação com
**se escalonar resolveria o problema**. Ou seja: o mecanismo passa a tomar
decisões *diferentes*, mas não decisões *melhores*.

### 14.4 A causa raiz: o sinal de energia não sabe o que importa

Isso conecta com o achado da §13.3 (confiança é cega à dificuldade).

Para superar o β, o termo de energia teria que **concentrar os escalonamentos
nas consultas onde escalonar realmente ajuda**. Nem o custo estático, nem o
custo por salto, nem o custo por tamanho de resposta sabem nada sobre isso.

A energia diz quanto custa. Não diz se vale a pena.

### 14.5 O que isso significa para o TCC

Isso **não** invalida o trabalho. É um resultado negativo bem fundamentado,
com explicação algébrica *e* três verificações empíricas independentes. A
afirmação defensável passa a ser:

> Ponderar o limiar de confiança por um custo de energia é redundante com o
> próprio limiar de confiança, porque o sinal de energia não carrega
> informação sobre o valor de escalonar.

Isso é útil: evita que o próximo pesquisador siga o mesmo caminho.

E o restante do trabalho continua de pé:
- caracterização de energia entre camadas (literatura verificada + medições
  próprias via RAPL) — §1, `layer_energy.yaml`
- validação do RecServe em tarefa generativa, em escalas que o artigo
  original nunca testou (60M-355M lá, 1B-10.7B aqui) — §13
- teste empírico da Suposição 1 do RecServe: confiança acompanha a
  capacidade do modelo, mas **não** a dificuldade da tarefa — §13.3
- `exp(min logprob)` separa melhor que a métrica especificada, replicado em
  três escalas — §13.2
- escadas não-monotônicas quebram a recursão em cascata (SOLAR-10.7B é pior
  que Llama-3.1-8B) — §13.1

### 14.6 Caminhos possíveis (a decidir com o orientador)

1. **Assumir o resultado negativo** como contribuição central e escrever o
   TCC em torno dele + das caracterizações já feitas. É o caminho seguro:
   todo o material já existe.
2. **Buscar um sinal de custo que saiba o que importa** — por exemplo, custo
   ponderado pelo ganho de qualidade esperado daquele salto (calibrado
   offline por par de camadas, §4). Aí o termo deixaria de ser só "energia" e
   passaria a ser "energia por ponto de qualidade esperado", que é o que
   originalmente se pretendia.
3. **Mudar o alvo**: em vez de melhorar a decisão de escalonamento, usar a
   caracterização de energia para responder "qual modelo colocar em cada
   camada" — questão para a qual já temos dados e que o achado da escada
   não-monotônica torna concreta.

A opção 2 é a que mais preserva a proposta original. A opção 1 é a de menor
risco dado o prazo.

## 15. Camada de nuvem a partir do leaderboard público (2026-08-31)

Em português, como a §14, porque decide o desenho experimental e precisa ser
discutido com o orientador.

### 15.1 A pergunta

O Prof. Nazar propôs: em vez de rodar o topo da cascata — inviável nesta
máquina —, pegar os resultados já publicados de um modelo grande e preencher
o resto da matriz rodando os modelos menores nas **mesmas instâncias**.

Funciona, mas por um motivo diferente do esperado.

### 15.2 O que o arquivo v1 do Open LLM Leaderboard tem

`open-llm-leaderboard-old/details_*` **não é gated** e tem **GSM8K 5-shot**
(o leaderboard v2 substituiu GSM8K por MATH-Lvl-5; só o arquivo antigo
serve). Schema por instância, 1319 linhas = split de teste completo:

| Campo | Conteúdo |
|---|---|
| `example` | texto da pergunta — chave de pareamento |
| `full_prompt` | o prompt 5-shot exato entregue ao modelo |
| `predictions` | texto gerado |
| `metrics` | `{'acc': bool}` — correção por instância |
| `input_tokens`, `cont_tokens` | IDs de token, dão as contagens para energia |
| `pred_logits` | **dtype `null` para GSM8K** — vazio |

Ou seja: **não há logprobs**, portanto não há confiança. O campo só é
preenchido em tarefas de log-likelihood (múltipla escolha).

### 15.3 Por que isso não bloqueia

**A camada de nuvem nunca usa a própria confiança.** Na regra do RecServe a
confiança decide escalar-ou-parar; a nuvem é terminal, não tem para onde
escalar. Ela só precisa produzir uma resposta e consumir energia. Para o topo,
`acc` mais contagem de tokens é exata e completamente suficiente.

Efeito colateral: cai também a exigência de que um provedor de API exponha
logprobs, caso a rota da API seja escolhida no lugar desta.

### 15.4 Correção de um erro cometido durante a análise

Numa primeira passada comparei "Llama-3-8B-Instruct (0.6869) contra
Llama-3-70B-Instruct (0.5406)" e concluí que a escada instruct invertia e que
seria preciso migrar para modelos base. **Estava errado:** o arquivo rotulado
como 70B-Instruct havia sido baixado de `details_meta-llama__Llama-2-70b-hf`,
isto é, Llama-2-70B base. A comparação estava contaminada por diferença de
geração.

Números corretos, todos nas mesmas 1319 instâncias:

| Modelo | GSM8K (harness 5-shot) |
|---|---|
| Llama-3-8B base | 0.4579 |
| Llama-3-70B base | 0.7688 |
| Llama-3-8B instruct | 0.6869 |
| **Llama-3-70B instruct** | **0.8544** |
| Llama-2-70B base | 0.5406 |

| Salto | conserta | quebra | líquido |
|---|---|---|---|
| 8B base -> 70B base | 454 | 44 | **+410** |
| 8B instruct -> 70B instruct | 268 | 47 | **+221** |
| 8B instruct -> Llama-2-70B | 132 | 325 | **-193** |

**A escada instruct não inverte.** A inversão de -193 é efeito de geração
(Llama-2 contra Llama-3), que é a mesma causa já documentada na §13.1 para o
SOLAR-10.7B — logo, uma replicação independente daquele achado, não um
mecanismo novo. Não há motivo para migrar para modelos base.

### 15.5 O cruzamento: qual 70B tem os dois lados

| Modelo | Confiança/resultados | Energia | Veredito |
|---|---|---|---|
| **Meta-Llama-3-70B** | leaderboard v1, **0.8544** | **Caravaca et al., arXiv:2511.05597, Tab. IV, 4xH100: 1.002 J/token** | **escolhido** |
| Llama-2-70B | leaderboard v1, 0.5406 | MLPerf v5.1, 0.0938 J/token (PDU, SPEC PTD) | energia impecável, escada invertida |
| Llama-3.1-70B | ausente do arquivo v1 | Oviedo, 0.3989 J/token | sem dado de confiança |
| Qwen2.5-72B | não verificado | Caravaca, 1.044 J/token | alternativa |

Meta-Llama-3-70B é **o mesmo modelo dos dois lados** — correspondência mais
apertada que a de qualquer outra camada desta escada, incluindo user e ONU,
onde a quantização diverge (§7.4 da proposta). O `layer_energy.yaml` já
registrava essa linha com a nota "closest exact-size match".

### 15.6 O que isso obriga a refazer

**Sim, as três camadas de baixo precisam ser recoletadas.** E o prompt não
pode ser reconstruído: os prefixos few-shot são **diferentes em cada
instância** (1319 prefixos distintos — o harness sorteia por documento). O
procedimento é ler `full_prompt[i]` do parquet e enviá-lo **verbatim** a cada
tier local. Isso elimina qualquer risco de divergência de formato e dá
pareamento exato, sem reimplementar a lógica de few-shot do harness.

Também é preciso alinhar `is_correct` em `src/tasks/gsm8k.py` com a regra do
harness (exact-match sobre o marcador `#### N`).

**Validação grátis do pipeline:** `Meta-Llama-3-8B` base está no leaderboard
com 0.4579. Rodá-lo localmente com o mesmo `full_prompt` e reproduzir esse
número valida de uma vez o replay do prompt, a extração da resposta e o
critério de correção — numa camada que seria rodada de qualquer forma.

### 15.7 O custo real não é a recoleta, é o regime de tokens

O leaderboard gera **mediana 83 tokens de saída para 858 de prompt**. Todas as
fontes de energia disponíveis medem no regime oposto:

| Fonte | Prompt | Saída |
|---|---|---|
| Leaderboard (o dado a usar) | ~858 | ~83 |
| Caravaca (1.002 J/token) | 300 | 300 |
| Oviedo (0.3989 J/token) | — | 361 |

Prefill passa a ser ~10x o decode, invertendo o regime em que este trabalho
vinha operando (~50 de prompt para ~200 gerados). Consequências:

- Multiplicar 1.002 J/token por 83 tokens seria substancialmente errado: esse
  valor foi amortizado sobre 300 tokens de saída.
- O argumento da §5.1 da proposta — a escalada repaga prefill em cada camada —
  deixa de ser segunda ordem e passa a ser termo dominante.
- Os achados §6.6 (transporte de segunda ordem) e §7.1 (cota do prefill em
  ~1-3.4%) foram estabelecidos no regime decode-dominado e **precisam ser
  recalculados** neste.

O caminho honesto é reportar a nuvem como **faixa** entre os pontos
publicados, declarando que nenhum foi medido no regime do harness.

### 15.8 Fonte

- Dataset: https://huggingface.co/datasets/open-llm-leaderboard-old/details_meta-llama__Meta-Llama-3-70B-Instruct
- Arquivo: `2024-04-21T11-59-48.701689/details_harness|gsm8k|5_2024-04-21T11-59-48.701689.parquet`
- Companheiro para validação: `open-llm-leaderboard-old/details_meta-llama__Meta-Llama-3-8B`
