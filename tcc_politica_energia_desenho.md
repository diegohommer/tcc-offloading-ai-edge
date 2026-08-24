# Energy-aware offloading policy — design notes

**Status:** design only, not implemented. Referenced from `README.md`'s
"Next steps" and from `implementation/src/layers/generative_layer.py`'s
open decisions. Written up 2026-08-24 from a design discussion; no code
changes accompany this document.

**Extends, does not replace:** the confidence-based stepwise escalation
rule already used by RecServe and by Pakpahan and Hwang's reference
architecture (Eq. 1-2 of their paper) — the sliding-window β-quantile
threshold this repo's `traced_recursive_serve.py` already implements for
the classification cascade. Everything below is about giving that same
mechanism an energy term, not inventing a new decision engine.

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

- **Quality gain proxy:** `1 − confidence` (or a per-tier calibrated
  accuracy figure, if measured — this repo's own reproduction run already
  produces the raw material for that: layer hop counts and per-hop
  correctness from `run_classification_cascade.py`'s trace).
- **λ (joules willing to spend per quality point):** a deliberately chosen
  policy parameter, not derivable from data alone — a value judgment.
  λ → ∞ recovers the original pure-confidence rule exactly, so this is a
  strict superset of the existing mechanism, not a replacement.
- **Lighter-weight alternative**, smaller diff against the existing code:
  keep Eq. 1's sliding-window quantile mechanics unchanged; just buffer an
  energy-adjusted confidence value (`confidence − λ × per-tier-pair joule
  penalty`) instead of raw confidence. Same equation, different input.

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

- **PerLLM** (Yang et al., arXiv:2405.14636) is explicitly energy-aware —
  its scheduling objective is `min (1/T)Σ(ω_tran·E_tran + ω_infer·E_infer
  + ω_idle·E_idle)` subject to a latency deadline: transmission,
  inference, and idle energy, explicitly summed and weighted. That is a
  *more* granular energy formulation than this design's single joules-
  per-quality exchange rate, not a lesser one. (Tabi, EdgeShard, and the
  Draft-Verify-style approach — likely SLED, arXiv:2506.09397 — were also
  checked and confirmed *not* energy-aware; only PerLLM was missed by
  trusting the table's one-line gloss instead of the paper itself.)
- A broader search (not just Pakpahan's own citations) found **closer**
  prior art still: **GreenServ** (Ziller et al., "Energy-Efficient
  Context-Aware Dynamic Routing for Multi-Model LLM Inference") routes
  queries across a pool of 16 LLMs trading accuracy against energy
  explicitly, via an online multi-armed bandit — reporting +22% accuracy
  and −31% energy vs. random routing. **CR²** ("Cost-Aware Risk-
  Controlled Routing for Wireless Device-Edge LLM Inference") does
  something structurally similar for device-edge routing. Neither is
  cited by Pakpahan, because his paper isn't primarily about energy and
  didn't survey that angle.

The general "trade quality against energy via a tunable threshold"
pattern is **not novel** — confirmed now from three independent
directions (EcoThink, PerLLM, GreenServ), not just one. Don't claim to
have invented cost-aware routing; that claim doesn't survive a reviewer
who's read any of these three.

**What's still genuinely defensible, narrowed to what it actually is:**
applying an explicit, *static*, per-tier-pair joules-per-quality exchange
rate *inside RecServe's specific β-quantile confidence mechanism*
(§4) — a minimal, structure-preserving extension of one particular
cascade, not a claim that energy-aware LLM routing is new. Grounded in
real measured cross-tier energy data (this repo's own profiling, not an
analytical TDP estimate like EcoThink's) is still a genuine, checkable
strength relative to all three comparators. Applied to a physically
distributed PON/SDN hierarchy (vs. EcoThink's single system, PerLLM's and
GreenServ's model-pool routing rather than a tiered hardware hierarchy)
is still a real difference in setting, not just framing.

**Design idea worth borrowing regardless of the novelty question:** both
PerLLM (CS-UCB) and GreenServ use an online multi-armed bandit to learn
their energy/quality tradeoff rather than fixing it as a constant. That's
directly relevant to this design's open λ-calibration problem (§9) — a
bandit could learn λ (or the per-tier-pair threshold bias) from observed
outcomes instead of it being a one-time hand-set value judgment. Worth
evaluating as an alternative to static λ before committing to "value
judgment, not derivable from data" as the final word on §9's open item.

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
