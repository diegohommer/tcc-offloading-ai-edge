# TCC: Energy Cost of Hierarchical LLM Inference over PON

**Topic:** energy cost of confidence-based offloading policies in
hierarchical LLM inference cascades (user → edge/ONU → fog/OLT → cloud)
over passive optical networks (PON).

This repository hosts the thesis's experimental harness: a real offloading
cascade (RecServe), instrumented to produce a per-query trace, plus an
energy-costing module that converts that trace into J/query using layer
energy tables measured from published sources.

## Current state: a classification harness, not the thesis's generative cascade

The published open-source RecServe implementation
(`implementation/src/recserve/vendor/`, vendored unmodified) only does
**sentiment classification** with **three** tiers (end/edge/cloud), no
token generation, and no decode phase. This repo's instrumented wrapper
(`implementation/src/recserve/traced_recursive_serve.py`) extends that to
the thesis's **four**-tier architecture — user / onu / fog / cloud,
matching Pakpahan and Hwang (IEEE Access vol. 14, 2026) Fig. 1 — using
four encoder-only classifiers of increasing capability (distilroberta /
roberta-base / roberta-large / deberta-large, 66M-400M parameters)
escalating via the same beta-quantile confidence threshold RecServe (and
Pakpahan's own architecture) uses, one tier at a time. The thesis's actual
target needs a **generative** cascade of four decoder LLMs (0.5B-70B+)
with prompt/output token counts per layer (design not yet written up;
section 12 of the master document is the energy-aware policy design
instead — see below).

This repository sits at that midpoint, by explicit decision:

1. **Real and working now:** run the 4-tier classification cascade end to
   end, recording a per-query trace — layer visited, prompt tokens (via
   each model's own tokenizer), confidence, latency. This validates the
   beta-quantile escalation mechanism and the whole trace-to-energy
   pipeline. Tier names (`user`/`onu`/`fog`/`cloud`) are the same keys used
   by `implementation/config/layer_energy.yaml`, so a hop's tier doubles
   as its energy lookup key directly, with no tier-to-layer proxy
   indirection.
2. **Deliberately not fabricated:** the layer energy tables
   (`implementation/config/layer_energy.yaml`) were measured on decoder
   LLMs doing multi-token decode, not on these tiny classifiers — no
   *published* energy measurement exists for distilroberta/roberta-base/
   roberta-large/deberta-large. So no "real" energy number is invented for
   the classification cascade. An optional, clearly-labeled proxy
   (`--smoke-test-energy`) prices each tier's forward pass as if it were
   decode on that tier's representative model, solely to exercise the cost
   formulas end to end — never to cite as a thesis result.
   Since 2026-08-29 there is also a **first-party** alternative to that
   proxy: `src/scripts/measure_tier_energy.py` measures these four models'
   actual energy on the local CPU via Intel RAPL, recorded under
   `local_measurement` in `layer_energy.yaml`. Those numbers are really
   measured, but of 66M-400M encoders on one CPU — a different unit
   (J per *prompt* token, single forward pass) from the decode-phase
   literature tables, and never to be mixed with them.
3. **Not built yet:** the real generative cascade (four decoder models,
   per-layer precision choice, local quantized and/or hosted-API execution).
   See `implementation/src/layers/generative_layer.py` for the interface
   already fixed and what's left to decide before implementing it.

## Structure

Two top-level halves: `implementation/` (all code, config, and generated
output) and `thesis/` (the LaTeX draft and reference papers) — nothing
code-related sits at the repo root.

```
implementation/
  requirements.txt
  config/
    layer_energy.yaml         # layer energy tables, each value with its source and caveats
  src/
    recserve/                   # everything RecServe-specific lives here
      vendor/                     # unmodified vendored copy of RecServe (see NOTICE.md)
      traced_recursive_serve.py   # instrumented reimplementation of RecServe's escalation loop
      run_classification_cascade.py   # runs RecServe over a dataset, writes a trace (JSONL)
    energy/                    # reusable costing library, not tied to RecServe
      layer_energy.py          # loads config/layer_energy.yaml
      cost.py                  # derivation formulas: J/query, J/token, gCO2/query, latency, cascade cost
    layers/
      generative_layer.py       # interface stub for the generative cascade (not implemented)
    scripts/
      compute_energy_report.py  # converts a trace into an energy report (CSV) via the layer energy tables
      run_policy_matrix.py      # runs every tier on every query -> answer matrix (phase 1 of the lambda sweep)
      sweep_energy_policy.py    # replays the energy-aware policy over that matrix (phase 2), form x lambda
      measure_tier_energy.py    # measures real per-tier energy on this machine via Intel RAPL
  results/
    traces/                    # output of the scripts above (generated, not version-controlled)
thesis/
  latex/                     # TCC LaTeX draft (infufrgs/abntex2 template), see thesis/latex/README.md
  papers/                    # local copies of cited papers, not version-controlled, see thesis/papers/README.md
```

## Running it

```bash
cd implementation
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

# Part 1 (functional): run the real cascade, write the trace
python src/recserve/run_classification_cascade.py --dataset sst2 --limit 40

# Part 2 (cost): convert the trace into energy
python src/scripts/compute_energy_report.py results/traces/sst2_test.jsonl --smoke-test-energy

# Part 3 (energy-aware policy): sweep the escalation rule's weighting form x lambda
#   3a. run every tier on every query once (the sweep replays this offline)
python src/scripts/run_policy_matrix.py --dataset sst2 --limit 0
#   3b. sweep, priced with the literature tables (labelled smoke-test energy)
python src/scripts/sweep_energy_policy.py results/traces/sst2_test.matrix.jsonl

# Optional: price the sweep with energy measured on THIS machine instead of
# the borrowed decode-phase numbers. Needs read access to the RAPL counters:
#   sudo find -L /sys/class/powercap -name energy_uj -exec chmod a+r {} +
# (root-only by default since CVE-2020-8694; resets on reboot, revert with chmod 400)
python src/scripts/measure_tier_energy.py --limit 40 --repeats 3
python src/scripts/sweep_energy_policy.py results/traces/sst2_test.matrix.jsonl \
    --measured-energy results/traces/measured_tier_energy.json
```

Models (`distilroberta-base-sst2-distilled`, `roberta-base-SST-2`,
`roberta-large-sst2`, `deberta-large-finetuned-sst2`) download automatically
from the Hugging Face Hub on first use. `--device -1` (default) runs on CPU;
this environment has no GPU.

## Next steps (not implemented here)

See section 11 of the TCC master document for the full pending list
(advisor confirmation, the L40S figure, MLPerf Power, load constants from a
real trace). From this repository's side specifically:

- Decide and implement the generative cascade
  (`implementation/src/layers/generative_layer.py`): per-layer
  model/precision choice, local quantized vs. hosted-API execution.
- Reimplement the confidence rule for Seq2Seq (normalized perplexity, per
  the RecServe paper's abstract), since the released code only covers
  Seq2Class.
- Fix `|T_prompt|` and `|T_gen|` from a real generative-cascade trace, not
  a separately estimated distribution.
- Extend the energy-aware policy beyond what is now built. The core
  mechanism — RecServe's β-quantile threshold weighted by a static
  per-tier-pair energy cost — **is implemented and evaluated** on the
  classification harness (`src/scripts/sweep_energy_policy.py`; design
  and results in `tcc_politica_energia_desenho.md`, §4/§11/§12). On the
  full 872-query SST-2 split it traces a real accuracy/energy frontier,
  and priced with locally measured RAPL energy it cuts 31% of energy for
  0.1 accuracy points at λ=0.01. What remains:
  - **§6's batch-aware cost profile** (time-of-day buckets selected by
    local escalation frequency) — every run so far uses one flat static
    cost per tier.
  - **A harder dataset.** SST-2 is nearly saturated; the tier spread is
    only ~4.6 points. `imdb` and `yelp_polarity` are already supported.
  - **A workload that actually reaches cloud**, so the batched-cloud
    inversion (design doc §11.3 rung 2) becomes testable — only 34/872
    queries escalate that far today.
  - **Re-deriving λ for the generative cascade.** No λ value from the
    classification runs transfers; the costs differ by orders of magnitude.

---
**Author:** Diego Amorim
