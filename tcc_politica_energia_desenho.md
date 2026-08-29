# Energy-aware offloading policy — design notes

**Status (updated 2026-08-29): implemented and evaluated on the
classification harness.** The mechanism of §4/§11 is built
(`src/scripts/sweep_energy_policy.py`), and has been run over the full
872-query SST-2 split, priced both with the borrowed literature values
and with energy measured locally via RAPL. Results and what they do and
do not support are in **§12**. Still *not* built: the generative
cascade this design ultimately targets, and §6's batch-aware profile.
Referenced from `README.md`'s "Next steps" and from
`implementation/src/layers/generative_layer.py`'s open decisions.

**Extends, does not replace:** the confidence-based stepwise escalation
rule already used by RecServe and by Pakpahan and Hwang's reference
architecture (Eq. 1-2 of their paper) — the sliding-window β-quantile
threshold this repo's `traced_recursive_serve.py` already implements for
the classification cascade. Everything below is about giving that same
mechanism an energy term, not inventing a new decision engine.

## Summary — the idea in plain terms

RecServe (and Pakpahan & Hwang's PON architecture, which this repo
follows) already has a simple rule: at each tier — user, ONU, fog,
cloud — the local model answers a query and reports how confident it
is. Confident enough, it stops there. Not confident enough, the query
moves up to the next tier, which tries again. This repeats, tier by
tier, until something is confident enough or the query reaches the
cloud. The bar for "confident enough" isn't fixed — each tier
calibrates it from its own recent history (the top-β-fraction of its
recent confidence scores) — but that bar has never had anything to do
with energy. A query escalates purely because the model wasn't sure,
no matter how much that escalation costs.

**The idea: add one number, λ ("lambda"), that answers a single
question — how many joules is one extra percentage point of confidence
worth to you?** With that number, and the real energy costs measured
for every tier in `layer_energy.yaml`, each tier's confidence bar gets
nudged automatically. Cheap hops (fog escalating to a well-batched
cloud) barely move the bar — energy isn't the deciding factor there.
Expensive hops (ONU running an oversized model, paying more locally
than it would cost to escalate) get their bar nudged down, because it's
not worth demanding near-perfect confidence when a cheap hop is right
there. Set λ to "infinity" (energy doesn't matter at all) and RecServe's
original behavior comes back exactly — this only adds an option, it
doesn't take one away.

**Concretely:** a tier normally requires, say, 85% confidence to answer
locally. If escalating costs 0.38 J and λ says "1 J buys 10 points of
confidence," the bar drops to about 81%. A borderline query at 83%
confidence — which would have escalated under plain RecServe — now
stays local, because the small remaining doubt wasn't worth the joules.
Same query, same model, different outcome, purely because of what a
joule is worth to you. (Full worked examples with real per-tier numbers
are in §4.)

**What this deliberately does *not* do, and why:**

- **No live network chatter between tiers.** Every cost a tier needs is
  measured once, offline, and shipped as static configuration — reusing
  the same registration handshake the base architecture already has for
  other device info. Nothing gets polled or broadcast while the system
  runs (§2, §5).
- **No jumping tiers.** A query still climbs one hop at a time, exactly
  as RecServe defines it — the mechanism only ever asks "is it worth
  going one hop further from here," never "should I skip straight to
  the cloud." Keeping this untouched means the whole thing inherits
  everything already proven about RecServe — decentralized, low
  overhead — for free (§8).
- **No claim that this is a brand-new idea.** Other systems (PerLLM,
  GreenServ, CR²) already trade energy against quality, in their own
  settings. What's actually being contributed is fitting that same kind
  of tradeoff into *this specific* mechanism, for *this specific*
  four-tier PON setting none of those systems touch (§7).

**What's still an open decision, not something the data can settle by
itself:** the actual value of λ (a value judgment — how much you care
about energy vs. quality), and how to estimate "how much would
escalating actually help" before you've tried it (necessarily a proxy,
since you can't know a tier's answer without running it) (§4, §9).

**How this gets tested (§11):** rather than picking one λ and one
weighting formula and hoping, the plan is to sweep both — three
candidate ways of applying the energy weight (additive, multiplicative,
exponential) across a range of λ values — and measure the resulting
accuracy-vs-energy tradeoff curve for each. λ=0 is plain RecServe, so
the baseline is a point on the curve rather than a separate experiment.
The first run fixes each tier's energy cost as a simple increasing
ladder (cheapest at the user device, priciest at the cloud) using real
measured numbers, and complexity — the cases where a higher tier is
actually *cheaper*, batch effects, time-of-day variation — gets layered
in one step at a time afterwards, each step swapping in a different real
row from the same measured table.

Everything below spells out the reasoning, the rejected alternatives,
and the exact mechanics behind all of this.

## 1. What the base architecture already gives us

- **The escalation rule is local and stepwise by design.** Each tier
  computes its own confidence score, compares it to a threshold derived
  from a sliding window of its own recent confidence history, and serves
  locally or escalates exactly one tier up. It never reads another tier's
  state.
- **The base paper's own SDN controller already knows everything, and
  deliberately doesn't use it for this decision.** Pakpahan and Hwang's
  ONOS controller maintains an "LLM-capability registry" per ONU —
  populated at registration with accelerator type, model cache size, and
  *energy constraints* — yet their paper states explicitly: "Although the
  ONOS controller is aware of all registered devices and their
  capabilities, the system always follows stepwise escalation to preserve
  locality and minimize network usage." Any energy-aware extension has to
  reckon with that stated design goal, not casually override it.
- **`implementation/config/layer_energy.yaml` now has real per-tier energy
  data**, including batch-sensitivity curves for fog (12 points, RTX 4090,
  Fadel Argerich & Patiño-Martínez 2024) and cloud (13+ points across
  Samsi, Oviedo, MLPerf v5.1, and Caravaca et al.), gathered specifically
  to inform this design. See the Hardware Ledger artifact for the
  consolidated view.

## 2. Rejected: live cross-tier telemetry broadcast

The first version of this idea considered: each tier reports its current
J/token to the SDN controller, which periodically broadcasts it to all
tiers so they can use it for offloading decisions.

Rejected, for three concrete reasons:

- **It works against the base paper's own stated design goal** ("preserve
  locality and minimize network usage"). Pakpahan's own Table 1 scores
  competing mechanisms on exactly this axis — RecServe scores high on
  bandwidth efficiency specifically *because* it's decentralized; PerLLM
  scores lower because it requires a central controller. A periodic
  all-tiers broadcast pulls this design toward the PerLLM end of that
  spectrum.
- **No timescale data exists to size the broadcast period.** Every energy
  number gathered so far is a static snapshot at a declared batch — there
  is no time-series / arrival-process data anywhere in this repo's sources
  showing how fast a tier's real load actually changes. Picking a
  polling/broadcast interval would be a guess.
- **A bare J/token number is ambiguous without its regime.** We showed
  (see §6) that J/token varies by ~3x (fog) to ~100x+ (cloud) depending
  purely on current batch/precision — a received number doesn't say
  whether it reflects stable hardware cost or a momentary load spike.

## 3. Local-only energy signal: valid, but only for a narrower question

A tier tracking its *own* recent energy consumption in a sliding window
(directly analogous in shape to the existing confidence window) requires
zero coordination and is architecturally clean. But it has a hard
timing/causality constraint worth being precise about:

**Local inference always runs first.** Every tier computes a full local
forward pass to get the confidence score the existing decision already
needs. By the time any decision is made, this query's local energy cost
is a sunk cost — spent regardless of the outcome. This splits the
possible mechanisms into two, which should not be conflated:

| | Timing | Can use | Decides |
|---|---|---|---|
| **Escalate gate** (extends the existing accept/escalate step) | Post-inference | This query's own now-known energy + own recent history | Accept the local answer, or *also* pay to escalate |
| **Admission/skip gate** (not in the base paper) | Pre-inference | Only recent history (a regime proxy) — never this query's own cost, which is unknowable in advance | Run locally at all, or forward immediately without local compute |

The escalate gate has no timing problem — it's structurally identical to
how the confidence threshold already works (built from history, applied
to a value only known post-hoc for *this* query). The admission gate is a
genuinely new mechanism, useful specifically for resource-constrained
tiers (battery on the `user` tier) where skipping local compute entirely
under a bad self-observed regime is worth the coarser decision.

**Why local-only is insufficient for minimizing total system energy, even
at the escalate gate:** the between-tier gap dwarfs and can invert
relative to what a local window can see. Typical production points:

| Tier | Typical J/token |
|---|---|
| User (best backend) | 0.074 – 0.21 |
| ONU (best precision) | 0.22 – 1.89 |
| Fog (production) | 0.38 |
| Cloud (MLPerf, batched, FP4) | 0.094 – 0.097 |
| Cloud (production, BF16) | 0.40 |

Cloud's best regime *beats* fog's production point. Fog's own local swing
is only ~3x (RTX 4090 batch curve, 2.7–8.5 J/token) while cloud's full
range spans ~2700x (0.094 to 260 J/token) depending on regime. A local
tier tracking only its own history cannot see which side of that range an
escalation target currently sits on — a 3x local signal is not enough
evidence to steer a decision where the real stakes are orders of magnitude
larger, on a variable it cannot observe.

**Where local-only remains fully valid:** a different objective — tier
self-protection (battery, thermal, load-shedding) — where the point was
never to reason about the target tier's cost at all. Note this objective
can call for the *opposite* threshold adjustment from the
energy-minimizing one for the same observed signal (shed load when
locally expensive, vs. accept more locally when escalating is relatively
worse) — which objective is being served needs to be a stated design
choice, not left implicit.

## 4. The proposed mechanism: static, per-tier-pair cost folded into the existing threshold

Give the existing β-quantile threshold an explicit joules-per-quality-point
exchange rate, using **static** (not live) per-tier-pair cost data already
in `layer_energy.yaml`.

Informal decision rule:

```
Escalate iff  λ × (expected quality gain from escalating)
              >  (known static joule cost of this specific hop)
```

**Exact composition with Eq. 2 (derived, not asserted — corrected
2026-08-24, an earlier draft of this section had the sign backwards):**
this is *not* a second gate run after the β-quantile check. Rearranging
the rule above as a threshold on confidence gives

```
Serve locally  iff  C_{M,τ}(x)  ≥  T_{M,τ}(β)  −  cost / λ
```

— i.e., `T_{M,τ}(β)` is computed exactly as RecServe already does
(untouched), and gets a single static offset, `cost/λ`, subtracted per
hop. One unified threshold check, same shape as Eq. 2, not two
sequential decisions. Sign check: expensive escalation (`cost` large
relative to λ) *lowers* the effective threshold — easier to clear, more
likely to stay local, which is the right direction. Cheap or
net-negative-cost escalation (cloud in its batched regime, cheaper than
fog) raises the effective bar, pushing toward escalating more readily —
also correct. `cost=0` or `λ→∞` collapses the offset to zero and
recovers Eq. 2 exactly, consistent with the strict-superset claim below.

> **Superseded as the primary form (2026-08-24, Prof. Nazar).** The
> additive offset below is kept as one of three candidate weighting
> functions to sweep, but it is **not** the default any more: it is
> unbounded (a large `cost` with a small λ drives the effective
> threshold below zero, needing an arbitrary clamp) and its λ
> convention is inverted relative to how a weight normally reads
> (λ→∞ = baseline). The **exponential** form
> `T_eff = T(β)·exp(−λ·cost)` is bounded by construction and recovers
> the baseline at λ=0. See §11 for all three forms, the λ-convention
> change, and the staged experimental plan.

**What λ actually is, in units:** joules per percentage-point of
confidence — "how many joules am I willing to spend to buy myself one
more point of confidence." Worked example: a tier's calibrated bar
`T(β) = 85%`, a query arrives at 83% confidence (would escalate under
plain RecServe), escalating this hop costs 0.38 J (ONU → fog):

| λ (J per confidence-point) | Bar shift = 0.38 ÷ λ | Effective bar | 83%-confidence query |
|---|---|---|---|
| 0.01 (generous) | 38 points | 85 − 38 = 47% | clears easily → **stays local** |
| 0.1 (moderate) | 3.8 points | 85 − 3.8 = 81.2% | clears (83 ≥ 81.2) → **stays local** |
| 1.0 (stingy) | 0.38 points | 85 − 0.38 = 84.6% | fails (83 < 84.6) → **escalates**, same as plain RecServe |

Same query, same model, three different outcomes, purely from what a
joule is declared to be worth. On a cheap hop (fog → cloud in its
batched regime, 0.094 J) the same three λ values barely move the bar at
all (0.9, 0.09 points) — cheap hops stay close to plain-RecServe
behavior regardless of λ; expensive hops are where λ actually does
work. A reasonable starting point, not derived from data but a sane
place to begin tuning: pick λ so that a *typical* hop cost (~0.5 J)
moves the bar by something noticeable-but-not-drastic (~4 points), i.e.
`λ ≈ 0.1 J/point`.

- **Quality gain proxy (open decision, not yet resolved):** two options,
  with a real accuracy/effort tradeoff.
  1. **Cheap default:** `gain ≈ 1 − confidence`. Costs nothing extra —
     already available. Caveat: an *optimistic ceiling* (assumes
     escalating always fully resolves uncertainty), not a true
     expectation — treats "genuinely hard query" and "query this tier is
     just weak on" identically, which biases the system toward
     escalating somewhat more than it should.
  2. **Calibrated per-tier-pair typical gain:** measured offline from
     trace data this repo's harness already produces —
     `run_classification_cascade.py`'s output already records, per
     escalated query, whether the current tier was wrong and whether the
     next tier resolved it. A calibration pass over that gives a real,
     tier-pair-specific empirical gain constant, calibrated the same way
     the joule costs already are (offline, once, not live) rather than a
     per-query guess. More accurate, needs a calibration pass to exist
     first. Start with (1) for an evaluable v1, swap in (2) once
     calibration passes are running.
- **λ (joules willing to spend per quality point):** a deliberately chosen
  policy parameter, not derivable from data alone — a value judgment.
  λ → ∞ recovers the original pure-confidence rule exactly, so this is a
  strict superset of the existing mechanism, not a replacement.

**Why per-tier-pair, not one global constant:** the sign of the escalation
cost isn't consistent across hops. `user → onu` is a reliable real cost.
`fog → cloud` can be free or a net savings depending on cloud's regime. A
single global β cannot represent that asymmetry; per-pair calibration,
straight from the Hardware Ledger, can.

**A counterintuitive-looking consequence, worth documenting rather than
"fixing":** this mechanism can make a tier escalate *almost everything*
when its own model is expensive relative to what's next — and the data
shows this isn't hypothetical. ONU's pricier configurations already cost
more than fog's typical production point:

| | J/token |
|---|---|
| ONU, 8B, W4 | 0.98 |
| ONU, 14B, W4 | 1.89 |
| Fog, production | 0.38 |

If ONU is running its 14B config, escalating costs only 0.38 J more,
against 1.89 J already spent (sunk) on ONU's own answer — cheap insurance
for any real quality gain. That's not the mechanism misbehaving; it's
correctly pricing that ONU is, for that model size, worse than pass-
through. The mechanism doesn't need to be "fixed" for this — it's a
**diagnostic**, not a bug: if a deployed tier ends up escalating nearly
everything, that's a signal the model/precision chosen for that tier is
mismatched for its position in the hierarchy, which feeds directly back
into the still-open per-layer model/device choice
(`generative_layer.py`). Note the effect flips for ONU's cheaper configs
(1.5B, W4, 0.22 J/token — cheaper than fog) — staying local is correctly
favored there. The behavior tracks whatever hardware/model choice gets
made per tier; it doesn't have an opinion of its own to correct.

## 5. Automating "how does a tier know the next tier's cost" — without live telemetry

This does **not** require runtime discovery:

1. **Measure once, offline.** Already done — `layer_energy.yaml`.
2. **Ship the relevant slice to each tier as static deployment config**,
   the same way a tier already loads its model path or its β parameter.
   No network round-trip, no polling.
3. **Reuse the base paper's existing registration flow** rather than
   inventing a new protocol: the ONOS `device.activate` RPC / `AttachProfile`
   intent that already carries the "energy constraints" field in the
   LLM-capability registry is the natural place to also hand a tier its
   neighbor's typical cost, at registration time.
4. **Refresh only when topology/hardware/model actually changes** (rare —
   nobody swaps a GPU generation mid-afternoon), never per-query, never on
   a live polling loop.

## 6. Encompassing batch variation without going live

A single static point estimate is a poor summary of a tier whose real
cost swings 3x–100x with load. The fix is a **small static profile**, not
a live signal:

- **Index by something free to know locally — time-of-day bucket.** No
  network round-trip, just a clock.
- **Cloud:** effectively one bucket, near its production point
  (0.3–0.4 J/token) — continuous batching keeps hyperscale serving close
  to permanently saturated in practice, so a flat value is a defensible
  simplification here specifically.
- **Fog:** 2–3 buckets calibrated from the measured RTX 4090 curve — a
  conservative value near the expensive end (~7–8.5 J/token, batch≈1) for
  likely-idle periods, a cheaper value near the flattened end
  (~2.7–3 J/token, batch 10–20) for likely-loaded periods. Bias toward the
  pessimistic end when uncertain: underestimating fog's cost is the
  costlier mistake (it leads to escalating expecting a saving that isn't
  there).
- **Selecting which bucket applies — still fully local, no coordination:**
  use the tier's own recent **escalation frequency** as the signal. This
  is the same proxy already named (but not mechanized) in this repo's
  README ("regime inferred locally from recent escalation frequency"),
  and it turns out to be a better-targeted signal for *this specific
  problem* than raw local energy history: a target tier's aggregate batch
  level is mechanically driven by the sum of escalation traffic arriving
  from all its child tiers on the same access segment, and household
  traffic tends to be time-correlated (shared peak hours) — so "how often
  am I escalating right now" is a causally grounded, zero-coordination
  correlate for "how loaded is my escalation target probably right now,"
  in a way that this tier's own J/token history isn't.
- **Open, unverified assumption:** this presumes a day/night traffic
  pattern typical of household broadband demand — standard in telecom
  capacity planning generally, but not measured for this specific
  deployment. Flagged, not resolved, here.

## 7. Positioning against existing literature

**Revised 2026-08-24 after actually reading Pakpahan's four other
compared mechanisms — the original version of this section overclaimed.**
It stated "none of Pakpahan's five compared mechanisms are energy-aware."
That's false for one of them, found by directly reading the paper rather
than trusting its one-line summary in Pakpahan's Table 1:

- **PerLLM** (Yang et al., arXiv:2405.14636, ref. [29] in Pakpahan's own
  bibliography) is explicitly energy-aware — its scheduling objective is
  `min (1/T)Σ(ω_tran·E_tran + ω_infer·E_infer + ω_idle·E_idle)` subject to
  a latency deadline: transmission, inference, and idle energy, explicitly
  summed and weighted. That is a *more* granular energy formulation than
  this design's single joules-per-quality exchange rate, not a lesser one.
- **Tabi** (Wang et al., EuroSys '23, DOI 10.1145/3552326.3587438, ref.
  [28]) and **EdgeShard** (Zhang et al., IEEE IoT J. 2025, DOI
  10.1109/JIOT.2024.3524255, ref. [31]) were both read directly and
  confirmed *not* energy-aware — Tabi's dispatcher is a pure calibrated-
  confidence threshold; EdgeShard optimizes latency and throughput only,
  via model-partitioning dynamic programming.
- **The fifth mechanism ("Draft-Verify" in Pakpahan's Table 1) was
  initially misidentified.** An earlier pass guessed SLED (arXiv:2506.09397)
  without access to Pakpahan's actual bibliography. Having now read that
  bibliography directly (`thesis/papers/TieredPONLLM.pdf`, references
  section), the real match — sitting in the exact right citation cluster,
  immediately after Tabi/PerLLM/EdgeShard — is **Hao, Jiang, Jiang, Ren
  and Cao, "Hybrid SLM and LLM for Edge-Cloud Collaborative Inference,"
  EdgeFM '24 (DOI 10.1145/3662006.3662067, ref. [32])**: an edge SLM
  drafts tokens, a cloud LLM verifies/corrects them, reported as
  "25.8–31.2% of the LLM's cost" for comparable quality. Full text is
  paywalled (ACM/ResearchGate both blocked direct access); consistent
  wording across multiple independent search sources frames "cost" as
  compute/token-usage cost, with no mention of energy, power, or joules
  anywhere found — treated as *likely* not energy-aware, at moderate
  rather than PerLLM-level confidence, since the primary text couldn't be
  read directly. Worth a direct read before citing this claim in the
  thesis text itself.
- A broader search (not just Pakpahan's own citations) found **closer**
  prior art still, both read directly: **GreenServ** (Ziller et al.,
  "Energy-Efficient Context-Aware Dynamic Routing for Multi-Model LLM
  Inference") routes queries across a pool of 16 LLMs trading accuracy
  against energy explicitly, via an online multi-armed bandit — reporting
  +22% accuracy and −31% energy vs. random routing. **CR²** (Xue et al.,
  arXiv:2605.12001, "Cost-Aware Risk-Controlled Routing for Wireless
  Device-Edge LLM Inference") is, if anything, *more* rigorously
  energy-aware than PerLLM: its cost function
  `c̄ₘ(x,ξ) = ωₜ·tₘ(x,ξ)/T₀ + ωₑ·eₘ(x,ξ)/E₀` combines latency and energy
  with real physical units (transmission/reception/idle power in Watts,
  energy in Joules), not just a motivating mention. Neither GreenServ nor
  CR² is cited by Pakpahan, because his paper isn't primarily about
  energy and didn't survey that angle.

The general "trade quality against energy via a tunable threshold"
pattern is **not novel** — confirmed now from four independent, directly-
read directions (EcoThink, PerLLM, GreenServ, CR²), not just one. Don't
claim to have invented cost-aware routing; that claim doesn't survive a
reviewer who's read any of these four.

**What's still genuinely defensible, narrowed to what it actually is:**
applying an explicit, *static*, per-tier-pair joules-per-quality exchange
rate *inside RecServe's specific β-quantile confidence mechanism*
(§4) — a minimal, structure-preserving extension of one particular
cascade, not a claim that energy-aware LLM routing is new. Grounded in
real measured cross-tier energy data (this repo's own profiling, not an
analytical TDP estimate like EcoThink's) is still a genuine, checkable
strength relative to all four comparators.

**The setting difference was verified, not assumed (2026-08-24, read past
the abstracts into each paper's actual system model):**

- **PerLLM** is a strict **two-tier** edge/cloud split. Network is modeled
  as abstract bandwidth constants ("network bandwidth of the cloud and
  edge is set to 300 Mbps and 100 Mbps," ±20% noise for "dynamically
  changing environment") — no PON, TDMA, wavelength allocation, or any
  access-network protocol. Evaluated on a generic testbed (5 CPUs +
  1×A100), not anything PON-adjacent.
- **GreenServ** is **not even multi-tier** — all 16 models sit on one
  A100 server in one rack ("a server running Ubuntu 22.04.5... an NVIDIA
  A100 GPU with 80 GB VRAM"), routed via local async queueing (FastAPI +
  Redis). Zero network modeling of any kind. Their own §6.4 names this as
  a limitation: "operational conditions should account for... request
  concurrency, batch processing, queuing delays... runtime model
  switching" — an explicit admission the network/deployment dimension
  isn't addressed at all.
- **CR²** is also strictly **two-tier** (UE + edge server — "a two-tier
  collaborative inference system consisting of a UE and an edge server"),
  and its channel model is generic wireless (Shannon-capacity uplink/
  downlink with path loss and Rayleigh fading), not any optical access
  technology.

None of the four models a hierarchy deeper than two tiers (this design
has four, with real measured per-tier hardware differences — phone,
Jetson, A30, hyperscale), and none touches PON/SDN access-network
constraints in any form. That is the real, checked basis for "different
setting," not an assumption to revisit later.

**Design idea worth borrowing regardless of the novelty question:** both
PerLLM (CS-UCB) and GreenServ use an online multi-armed bandit to learn
their energy/quality tradeoff rather than fixing it as a constant.
GreenServ's reward function in particular —
`r_t(m,q) = α·Accuracy − β·Cost, with α=1−λ, β=λ` — is structurally the
same single-dial parameterization landed on independently in §4
(λ=0 pure quality, λ=1 pure cost), which is a useful piece of convergent
validation for that specific parameterization choice, separate from the
novelty question. This is directly relevant to this design's open
λ-calibration problem (§9) — a bandit could learn λ (or the per-tier-pair
threshold bias) from observed outcomes instead of it being a one-time
hand-set value judgment. Worth evaluating as an alternative to static λ
before committing to "value judgment, not derivable from data" as the
final word on §9's open item.

Cite PerLLM, GreenServ, CR², and EcoThink explicitly, on this exact axis,
when the related-work chapter (currently a skeleton, `thesis/latex/tcc.tex`
§"Revisão Bibliográfica") gets rewritten — the comparison table Pakpahan's
own paper uses (Table 1) is a reasonable template to extend with an
energy column and these additional rows.

## 8. Scope: core contribution vs. future work

**Decision (2026-08-24): stay 100% structurally faithful to RecServe's
recursive, one-hop-at-a-time escalation.** §4-§6 (the λ-weighted
threshold, static per-pair costs, batch-aware profile) never change
*which tier is next* or let a query skip a hop — only *when* the current
tier hands off, using the same Eq. 1-2 shape. This is deliberate, not a
missed opportunity:

- It's a strict superset of the original rule (λ → ∞ recovers it
  exactly), so it inherits every property Pakpahan's Table 1 already
  validated for RecServe (decentralized, high bandwidth efficiency, high
  latency awareness) without having to re-argue any of them.
- It's evaluable **today**, on infrastructure that already exists —
  `run_classification_cascade.py` already produces per-hop correctness
  and hop counts, enough to plot accuracy vs. energy as λ varies against
  the λ→∞ baseline, without waiting on the generative cascade.
- The alternative (letting a query jump non-adjacent tiers mid-cascade)
  is not an extension of RecServe's actual mechanism, it's a different
  escalation topology that merely *uses* RecServe's confidence signal —
  see below.

**§4-§6 is the core contribution to build and evaluate.** The remaining
items are secondary or explicitly out of scope for now:

- **§6 (batch-aware profile) is a refinement, not a prerequisite.** A v1
  can ship §4-§5 with one flat static cost per tier-pair and still be a
  complete, evaluable contribution; §6 sharpens accuracy once that's
  working.
- **Multi-hop / entry-point selection is explicit future work, not part
  of this contribution.** Letting a query start its journey at a smarter
  entry tier (chosen once, before RecServe's unmodified recursion takes
  over) would capture the sunk-cost savings for queries known in advance
  to need a high tier — but it requires a second, harder-to-calibrate
  prior (per-tier success likelihood by query type) and is honestly
  described as *wrapping* RecServe, not extending it, so it shouldn't be
  conflated with §4's claim. Its payoff is also unconfirmed for the real
  target workload: in this repo's own SST2 classification reproduction,
  only 10% of queries escalated past ONU and 0% reached cloud, so the
  wasted-sunk-cost problem it would fix was small for *that* benchmark —
  worth re-measuring once the generative cascade (harder, more evenly
  distributed difficulty) exists, not assumed from this one.

## 9. Open questions — still need resolving for the core contribution

- **λ** (joules per quality point) needs to be set deliberately, possibly
  per deployment scenario (battery-constrained vs. quality-critical); it
  is a value judgment the data cannot supply — *or* it could be learned
  online via a multi-armed bandit instead of hand-set, per §7's PerLLM/
  GreenServ note. Worth comparing both before treating "value judgment"
  as final.
- **Sliding-window / profile-bucket size** (§6, if pursued) needs
  traffic-dynamics data this repo doesn't have — every energy number
  gathered so far is a static snapshot, never a time series.
- **Day/night traffic-pattern assumption** for fog's static profile
  buckets (§6) is unverified for this specific deployment context.

## 10. Where this plugs into the code

- `implementation/src/layers/generative_layer.py`: this design determines
  how that module's escalation decision should eventually be implemented
  — the interface (`GenerationResult`, `GenerativeLayer.generate`) already
  fixed there is unaffected; only the not-yet-written orchestration logic
  around it would use §4-§6.
- `implementation/config/layer_energy.yaml`: source of the per-tier-pair
  static costs (§4) and batch profiles (§6).
- `implementation/src/recserve/traced_recursive_serve.py`: the existing
  β-quantile sliding-window mechanism this design extends rather than
  replaces.

## 11. Weighting function and staged experimental plan

**Origin: Prof. Nazar, 2026-08-24** — "manter os dois e colocar um fator
multiplicativo ou exponencial de ponderação... a gente testa o sistema
com várias configurações diferentes desses parâmetros e medimos o
impacto deles," plus: start by fixing per-layer energy cost as a
*monotonically increasing* progression, then add complexity (batches
etc.) from there.

### 11.1 Keep both signals; sweep the weighting function

Confidence and energy both stay in the decision — energy weights the
confidence threshold rather than replacing or gating it. Three candidate
forms, all applied to the *same* `T_{M,τ}(β)` RecServe already computes:

| Form | Effective threshold | λ units | Baseline at | Bounded? |
|---|---|---|---|---|
| **A — additive** (the original §4 draft) | `T(β) − λ⁻¹·cost` | J per confidence-point | λ→∞ | No — needs clamping to [0,1] |
| **B — multiplicative** | `T(β) · (1 − λ·cost)` | 1/J | λ=0 | No — needs clamping when λ·cost>1 |
| **C — exponential** *(proposed default)* | `T(β) · exp(−λ·cost)` | 1/J | λ=0 | **Yes** — `exp(−λ·cost) ∈ (0,1]` for cost,λ ≥ 0, so `T_eff ∈ (0, T(β)]` with no clamp |

**Why C as the default:** bounded by construction (no arbitrary clamp,
which would otherwise be a free parameter nobody can justify), and its
effect saturates smoothly — doubling a hop's cost squares the
weighting factor rather than doubling a subtraction, so no single
expensive hop can collapse the threshold to nonsense. A and B stay in
the sweep because measuring the impact of the *form itself* (not just
its parameter) is exactly what was asked for.

**λ convention changes with this (worth stating explicitly, it flips
relative to earlier sections):** under B and C, **λ=0 means energy is
ignored and the mechanism is exactly plain RecServe**; larger λ means
energy weighs more heavily. This is both more intuitive than form A's
λ→∞ baseline and consistent with GreenServ's own `α=1−λ, β=λ`
parameterization (§7). All λ-sweep results should be reported under this
convention. The strict-superset property is unchanged and in fact
cleaner: λ=0 is now an exact, trivially-checkable baseline rather than a
limit.

### 11.2 Step 1 — monotonically increasing cost, real numbers only

Prof. Nazar's "progressão crescente" starting point is achievable
**without fabricating anything** — a monotonic ladder exists inside the
already-measured data, by selecting the cheapest ONU configuration and
the BF16 (not FP4) cloud point:

| Tier | J/token | Real source (`layer_energy.yaml`) |
|---|---|---|
| user | 0.074 | Llama 3.2 1B, W4, CPU backend (Cai et al.) |
| onu | 0.22 | Qwen2.5 1.5B, W4, Jetson AGX Orin (Kubwimana & Huang) |
| fog | 0.38 | A30, SOLAR-10.7B, production (Watt Counts) |
| cloud | 0.40 | 4×H100, BF16, production regime (Oviedo et al., Joule) |

This makes step 1 a defensible real-data baseline rather than a toy
config, and it makes every later complexity step a matter of *swapping
in other real rows from the same table* rather than changing the kind of
input.

**Hop cost accounting for step 1:** cost of escalating into tier *t* =
(tokens generated at *t*) × (J/token for *t*) + one link hop (0.1 J,
`link.per_hop_energy_J`). For realistic generation lengths the link term
is negligible (300 tokens × 0.38 J/token = 114 J vs 0.1 J), but note it
is *not* negligible in the current classification harness, where each
tier does a single forward pass and generates no tokens — another reason
the classification-harness energy numbers stay labelled as smoke-test
only.

### 11.3 Complexity ladder — what to add, in what order, and what each tests

Each step swaps a real row from `layer_energy.yaml`; nothing synthetic
is introduced at any stage:

1. **Monotonic baseline** (§11.2) — establishes the mechanism behaves
   sanely when the intuition "higher tier costs more" actually holds.
2. **Break monotonicity with the batched-cloud point** (cloud FP4
   MLPerf, 0.094 J/token — *cheaper than fog's 0.38*). Directly tests
   the inversion case discussed in §3, and whether the per-tier-pair
   formulation (§4) handles a negative-cost hop correctly rather than
   just tolerating it.
3. **Oversized ONU** (14B W4, 1.89 J/token — pricier than fog). Tests
   the "tier degenerates to pass-through" diagnostic predicted in §4;
   the expected, *correct* result is that ONU escalates nearly
   everything.
4. **Fog batch curve** (2.74–8.53 J/token across batch 1→20). Introduces
   §6's batch sensitivity as a real swept variable.
5. **Time-of-day profile buckets** (§6) — the full batch-aware profile,
   selected by local escalation frequency.

### 11.4 What to measure

Primary: an **accuracy-vs-total-energy Pareto curve**, one point per
(form, λ) configuration, with **λ=0 as the explicit baseline** (plain
RecServe — the curve must pass exactly through it, which doubles as a
correctness check on the implementation).

Secondary, and cheap to record from the trace the harness already
writes:

- **Hop distribution shift** per configuration (does traffic actually
  move between tiers, or does λ do nothing until some knee?).
- **Per-tier escalation rate** — flags any tier degenerating into
  pass-through (the §4 diagnostic), which is the expected outcome of
  ladder step 3 and would be a *finding*, not a failure.
- **Sensitivity of the ranking to the weighting form** — if A, B and C
  produce the same Pareto ordering, the form doesn't matter and C can be
  adopted purely for its boundedness; if they diverge, that divergence
  is itself a result worth reporting.

All of this is evaluable on the existing classification harness
(§8) — `run_classification_cascade.py` already records per-hop
confidence, correctness and tier, which is everything the sweep needs
except the λ loop itself.

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
