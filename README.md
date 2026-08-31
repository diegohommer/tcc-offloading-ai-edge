# TCC: Energy Cost of Hierarchical LLM Inference over PON

**Topic:** energy cost of confidence-based offloading in hierarchical LLM
inference cascades (user → edge/ONU → fog/OLT → cloud) over passive optical
networks.

## Scope, results and limitations live in the proposal

> **[Joules per Query](https://claude.ai/code/artifact/80752dd5-f88f-47a0-ab4b-9cb2d632a92f)**
> — the revised proposal, and the **source of truth** for what this work
> claims: research questions, methodology, established results, and the
> bounded methodological limitations.
>
> Source: `thesis/proposal/joules-per-query.html`

This README covers the repository only — what the code is and how to run it.
Anything about the thesis argument belongs in the proposal, so it does not
drift out of sync here.

The one result worth stating twice, because it sets the scope of everything
else: **weighting the confidence threshold by an energy cost is redundant with
RecServe's existing β parameter.** A quantile is a rank statistic, so
tier-constant terms cannot re-rank queries; three variants (static per-pair,
per-hop, per-query by output length) all landed within noise of tuning β alone
(+0.0039 / +0.0012 / +0.0026 accuracy at matched energy). The full analysis is
`tcc_politica_energia_desenho.md` §14, in Portuguese, for discussion with the
advisor.

## What is here

Two halves: `implementation/` (code, config, generated output) and `thesis/`
(proposal, LaTeX draft, reference papers). Nothing code-related at the root.

```
implementation/
  config/
    layer_energy.yaml           # per-tier energy tables; every value carries its source and caveats
  src/
    recserve/
      vendor/                     # unmodified vendored RecServe (see NOTICE.md)
      traced_recursive_serve.py   # RecServe's escalation loop, reimplemented with per-query tracing
      run_classification_cascade.py
    layers/
      generative_layer.py         # the interface a tier implements
      ollama_layer.py             # local (Ollama) and OpenAI-compatible backends
    tasks/
      gsm8k.py                    # prompt construction and answer checking
    energy/
      layer_energy.py             # loads config/layer_energy.yaml
      cost.py                     # J/query, J/token, gCO2/query, latency, cascade cost
    scripts/
      run_policy_matrix.py        # classification: every tier on every query
      run_generative_matrix.py    # generative: same, plus full per-token logprobs
      sweep_energy_policy.py      # replays a matrix offline: weighting form x lambda x beta
      measure_tier_energy.py      # first-party per-tier energy on this machine, via Intel RAPL
      compute_energy_report.py    # trace -> energy report (CSV)
  results/traces/                 # generated, not version-controlled
thesis/
  proposal/                       # the revised proposal (source of truth)
  latex/                          # LaTeX draft, infufrgs/abntex2 template
  papers/                         # local copies of cited papers, not version-controlled
```

### The collection design worth knowing about

Both matrix runners **run every tier on every query** and store the raw output —
for the generative runner, the full per-token log-probability vector and token
strings, at temperature 0. Escalation is path-dependent: β changes which tiers a
query visits, which changes each tier's confidence history, so a single cascade
trace cannot answer what a skipped tier would have said.

Collecting the full matrix once makes every downstream question — any β, any λ,
any confidence definition, any energy model — answerable offline, with no model
re-executed. The generative runner is also resumable per `(tier, query)`, which
matters because the cloud tier is billed.

## Running it

```bash
cd implementation
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

### Generative cascade (GSM8K)

Needs a local Ollama with the ladder's models pulled. Collection is the only
expensive step; the sweep replays it offline.

```bash
# smoke test -- confirm logprobs come back and confidence looks sane
python src/scripts/run_generative_matrix.py --limit 20 --tiers user

# local tiers (free)
python src/scripts/run_generative_matrix.py --limit 200 --tiers user,onu,fog

# cloud tier (billed; needs a provider that exposes logprobs)
export LLAMA_API_KEY=...
python src/scripts/run_generative_matrix.py --limit 200 --tiers cloud

# override the ladder without touching code
python src/scripts/run_generative_matrix.py --config config/ladder.json --tiers onu
```

### Classification cascade (SST-2)

The encoder cascade that validated the escalation mechanism and the
trace-to-energy pipeline before the generative track existed.

```bash
python src/recserve/run_classification_cascade.py --dataset sst2 --limit 40
python src/scripts/compute_energy_report.py results/traces/sst2_test.jsonl --smoke-test-energy

python src/scripts/run_policy_matrix.py --dataset sst2 --limit 0
python src/scripts/sweep_energy_policy.py results/traces/sst2_test.matrix.jsonl
```

`--smoke-test-energy` prices each forward pass as if it were decode on that
tier's representative model. It exercises the cost formulas end to end and is
**never** a citable result: no published energy measurement exists for
distilroberta / roberta-base / roberta-large / deberta-large.

### First-party energy measurement

Measures these models' real energy on the local CPU via Intel RAPL, recorded
under `local_measurement` in `layer_energy.yaml`. Genuinely measured — but of
66M–400M encoders on one CPU, in a different unit (J per *prompt* token, single
forward pass) from the decode-phase literature tables. Do not mix the two.

```bash
# RAPL counters are root-only by default (CVE-2020-8694); resets on reboot
sudo find -L /sys/class/powercap -name energy_uj -exec chmod a+r {} +

python src/scripts/measure_tier_energy.py --limit 40 --repeats 3
python src/scripts/sweep_energy_policy.py results/traces/sst2_test.matrix.jsonl \
    --measured-energy results/traces/measured_tier_energy.json
```

## Open items

- **Cloud tier of the generative cascade.** 32B+ is not viable locally
  (QwQ-32B measured at 1.2 tok/s), so it needs the GPPD cluster or a hosted
  endpoint exposing log-probabilities.
- **A fog model actually stronger than the ONU tier.** SOLAR-10.7B measured
  *worse* than Llama-3.1-8B, which breaks the cascade's monotonicity
  assumption and, because escalation is stepwise, converts energy into wrong
  answers at every tier above it.
- **The batch-aware cost profile.** The one untested source of cost
  variability; the cloud tier's batching swing is ~30×.

---
**Author:** Diego Amorim
