# TCC: Energy Cost of Hierarchical LLM Inference over PON

**Topic:** energy cost of confidence-based offloading policies in
hierarchical LLM inference cascades (user → edge/ONU → fog/OLT → cloud)
over passive optical networks (PON).

This repository hosts the thesis's experimental harness: a real offloading
cascade (RecServe), instrumented to produce a per-query trace, plus an
energy-costing module that converts that trace into J/query using layer
energy tables measured from published sources.

## Current state: a classification harness, not the thesis's generative cascade

The published open-source RecServe implementation (`third_party/recserve/`,
vendored unmodified) only does **sentiment classification**: three
encoder-only classifiers (distilroberta / roberta-base / roberta-large,
66M-355M parameters) escalating via a beta-quantile confidence threshold,
with no token generation and no decode phase. The thesis's actual target
(see the master document, section 12) needs a **generative** cascade of
four decoder LLMs (user/ONU/fog/cloud, 0.5B-70B+) with prompt/output token
counts per layer.

This repository sits at that midpoint, by explicit decision:

1. **Real and working now:** run RecServe's classification cascade end to
   end, recording a per-query trace — layer visited, prompt tokens (via
   each model's own tokenizer), confidence, latency. This validates the
   beta-quantile escalation mechanism and the whole trace-to-energy
   pipeline.
2. **Deliberately not fabricated:** the layer energy tables
   (`config/layer_energy.yaml`) were measured on decoder LLMs doing
   multi-token decode, not on RecServe's tiny classifiers — no published
   energy measurement exists for distilroberta/roberta-base/roberta-large.
   So no "real" energy number is invented for the classification cascade.
   An optional, clearly-labeled proxy (`--smoke-test-energy`) maps
   RecServe's end/edge/cloud tiers onto three of the four layers and prices
   the forward pass as if it were decode, solely to exercise the cost
   formulas end to end — never to cite as a thesis result.
3. **Not built yet:** the real generative cascade (four decoder models,
   per-layer precision choice, local quantized and/or hosted-API execution).
   See `src/layers/generative_layer.py` for the interface already fixed and
   what's left to decide before implementing it.

## Structure

```
config/
  layer_energy.yaml         # layer energy tables, each value with its source and caveats
third_party/
  recserve/                 # unmodified vendored copy of RecServe (see NOTICE.md)
src/
  recserve_trace/
    traced_recursive_serve.py   # instrumented reimplementation of RecServe's escalation loop
  energy/
    layer_energy.py          # loads config/layer_energy.yaml
    cost.py                  # derivation formulas: J/query, J/token, gCO2/query, latency, cascade cost
  layers/
    generative_layer.py       # interface stub for the generative cascade (not implemented)
scripts/
  run_classification_cascade.py   # runs RecServe over a dataset, writes a trace (JSONL)
  compute_energy_report.py        # converts a trace into an energy report (CSV) via the layer energy tables
results/
  traces/                    # output of the scripts above (generated, not version-controlled)
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

# Part 1 (functional): run the real cascade, write the trace
python scripts/run_classification_cascade.py --dataset sst2 --limit 40

# Part 2 (cost): convert the trace into energy
python scripts/compute_energy_report.py results/traces/sst2_test.jsonl --smoke-test-energy
```

Models (`distilroberta-base-sst2-distilled`, `roberta-base-SST-2`,
`roberta-large-sst2`) download automatically from the Hugging Face Hub on
first use. `--device -1` (default) runs on CPU; this environment has no GPU.

## Next steps (not implemented here)

See section 11 of the TCC master document for the full pending list
(advisor confirmation, the L40S figure, MLPerf Power, load constants from a
real trace). From this repository's side specifically:

- Decide and implement the generative cascade
  (`src/layers/generative_layer.py`): per-layer model/precision choice,
  local quantized vs. hosted-API execution.
- Reimplement the confidence rule for Seq2Seq (normalized perplexity, per
  the RecServe paper's abstract), since the released code only covers
  Seq2Class.
- Fix `|T_prompt|` and `|T_gen|` from a real generative-cascade trace, not
  a separately estimated distribution.

---
**Author:** Diego Amorim
