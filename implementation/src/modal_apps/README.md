# Modal apps: the cloud tier, and first-party GPU energy

Two Modal apps. Neither is imported by the harness — they are run from the CLI
and produce data the harness consumes.

Directory is `modal_apps/`, not `modal/`, on purpose: the scripts in `src/`
do `sys.path.insert(0, SRC)`, so a package named `modal` here would shadow the
real `modal` client and break every import.

| file | what it does |
|---|---|
| `cloud_tier_server.py` | serves Qwen2.5-72B via vLLM as an OpenAI-compatible endpoint, so `run_generative_matrix.py --tiers cloud` can finally be collected |
| `measure_gpu_energy.py` | measures J/token vs. batch size on the rented GPU via NVML, for the fog and cloud tiers |

## Why this is worth doing

**The cloud tier is the one missing measurement.** Three of the results the thesis
rests on cannot be computed without it: the cascade-vs-direct-to-cloud comparison,
the destination-selection mechanism, and any four-tier sweep.

**The batch curve is the second.** `config/layer_energy.yaml`'s fog and cloud
`batch_curve` blocks are transcribed from other people's charts, and their own
caveat lists record the damage: an outlier flagged by its authors, an SLA ceiling
used as a point estimate, an RTX 4090 standing in for fog-class hardware at the
wrong measurement boundary, and a hardware contradiction inside one source. Two
points per model is also too few to model `E_cloud(β)`, which is the quantity the
cascade's own filtering ratio controls.

Renting the GPU replaces all of that with one consistent measurement: same
hardware, same model, same boundary (`chip_or_module`, matching the file's
declared base), as many batch points as we pay for.

Modal has T4, L4 and L40S — three of the six GPUs in the file's declared fog
`hardware_class` — so the fog tier can be measured on hardware that is actually
in class, which the current RTX 4090 curve is not.

## Running

```bash
# --- cloud tier: collect the missing answers ---
modal deploy src/modal_apps/cloud_tier_server.py     # prints the URL
# paste that URL into config/tier_cloud_modal.json
export MODAL_VLLM_KEY=unused
python src/scripts/run_generative_matrix.py --limit 200 --tiers cloud \
    --config config/tier_cloud_modal.json
modal app stop tcc-cloud-tier                        # stop billing

# --- energy: measure the batch curves ---
MODEL=Qwen/Qwen2.5-72B-Instruct GPU=H100:2 \
    modal run src/modal_apps/measure_gpu_energy.py

MODEL=meta-llama/Llama-3.1-8B-Instruct GPU=L4:1 BATCHES=1,2,4,8,16,32 \
    modal run src/modal_apps/measure_gpu_energy.py
```

## Cost

Modal's published rates: H100 $3.95/h, L40S $1.95/h, A10 $1.10/h, L4 $0.80/h,
T4 $0.59/h. The starter plan includes $30/month of free credits.

| step | GPU | rough time | rough cost |
|---|---|---|---|
| cloud tier, 200 queries | 2×H100 | 50–60 min | **$7–8** |
| cloud batch curve | 2×H100 | ~25 min | **$3–4** |
| fog batch curve | 1×L4 | ~30 min | **<$1** |

**Total ≈ $11–13**, most likely inside the free credits. First runs dominate:
they download ~145 GB of weights, which the `huggingface-cache` volume then
keeps for every later run.

### The cost lever worth knowing about

`run_generative_matrix.py` sends requests **one at a time** — deliberately, since
it is resumable and was written against a billed-per-token API where concurrency
bought nothing. Against a rented GPU billed per *second*, that inverts: vLLM
would happily batch 200 concurrent requests and finish in a few minutes instead
of ~40, cutting the cloud-tier bill several-fold. Making the collector issue
concurrent requests is a small change and is the single biggest saving available
here.

## Measurement caveats to carry into the thesis

- **Boundary is the GPU chip/module** (NVML per-device counter). No host CPU, no
  DRAM, no PUE. This *matches* `layer_energy.yaml`'s declared base — and unlike
  the MLPerf cloud point (node-level, 8 GPUs plus host) and the RTX 4090 fog
  curve (node-level), it needs no boundary adjustment before comparison.
- **Values are gross**; idle GPU power is measured and reported separately rather
  than subtracted, so the boundary definition does not shift mid-table.
- **`ignore_eos=True`** forces every sequence to exactly `OUT_TOKENS`, so J/token
  is comparable across batch sizes instead of tracking answer length.
- **Shared infrastructure.** A rented container is noisier than a dedicated
  bench; hence repeats and a median. Report the spread, not just the median.
- **vLLM at temperature 0 is not bit-deterministic across batch sizes** — batching
  changes reduction order. Answers collected at different concurrency levels can
  differ slightly. Collect the *answer matrix* in one configuration, and treat the
  energy sweep as a separate run.
