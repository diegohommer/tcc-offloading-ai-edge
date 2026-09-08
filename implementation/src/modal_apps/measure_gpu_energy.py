"""First-party J/token-vs-batch curves for the fog and cloud tiers, measured on rented GPUs.

WHAT THIS FIXES
---------------
config/layer_energy.yaml's fog and cloud blocks are the weakest evidence in the
whole table, and their caveat lists say so at length:

  * fog's primary_point is an A30/SOLAR-10.7B number its own authors flag as an
    outlier (21 GB model on a 24 GB card), at batch=1000, with no phase split;
  * fog's cross_check is an SLA *ceiling* derived from a filter threshold, not a
    measurement;
  * fog's batch_curve comes from an RTX 4090 -- not in the declared fog hardware
    class -- at NODE-level boundary (CPU+GPU+DRAM stacked), which the file
    explicitly warns must not be blended with the chip-boundary points;
  * cloud's batch_curve has TWO points per model, and its 405B row carries an
    unresolved A100-vs-H100 hardware contradiction inside the source paper.

Every one of those is a consequence of transcribing other people's charts. Renting
the GPU directly removes the whole class of problem: same hardware, same model,
same boundary, same workload, as many batch points as we care to pay for.

WHY THE BATCH CURVE IS THE MEASUREMENT THAT MATTERS
----------------------------------------------------
The cascade delivers only beta^k of its traffic to tier k. Cloud J/token varies
by ~30x over the measured batch range (QwQ-32B: 12.904 at batch=1 vs 0.4332
optimized). So E_cloud is not a constant, it is E_cloud(beta): the cascade's own
filtering decides how batched the cloud gets, and therefore how efficient it is.
Arguing that from two literature points is weak; measuring the curve makes it a
result. This is also the item section 6 of the design doc flagged as the most
important untested one.

METHOD
------
Energy comes from NVML's cumulative counter, nvmlDeviceGetTotalEnergyConsumption,
which reports millijoules consumed since the last driver reload on Volta and
later. Taking a delta across a generation is the GPU analogue of what
measure_tier_energy.py already does with Intel RAPL for the local tiers -- a
counter difference, not an integrated power sample, so it cannot miss a spike
between polls.

Deliberate methodological choices, each with a reason:

  * boundary = GPU chip/module. This MATCHES config/layer_energy.yaml's declared
    base (`boundary: chip_or_module`), unlike the MLPerf cloud point (node-level,
    8 GPUs plus host) and unlike the RTX 4090 fog curve (node-level). Host CPU and
    DRAM are not counted, and neither is datacenter PUE.
  * ignore_eos=True with a fixed max_tokens. Every batch size then generates
    exactly the same number of tokens, so J/token comparisons across the sweep are
    not contaminated by different answer lengths. Without this, a batch whose
    prompts happen to end early would look artificially cheap.
  * one warm-up generation per batch size, discarded. The first call pays CUDA
    graph capture and allocator warm-up, which is startup cost, not decode cost.
  * an idle baseline is measured and reported alongside. Reported values are GROSS
    (what the GPU actually drew). The idle figure lets a reader compute net if
    their comparison demands it; subtracting it by default would silently change
    the boundary definition mid-table.
  * repeats with a median, because a shared multi-tenant host is noisier than the
    dedicated bench a paper would use.

USAGE
-----
    # cloud tier: 72B on 2xH100 (~$7.90/h)
    MODEL=Qwen/Qwen2.5-72B-Instruct GPU=H100:2 \
        modal run src/modal_apps/measure_gpu_energy.py

    # fog tier: 8B on one L4 (~$0.80/h) -- L4 IS in the declared fog hardware class,
    # unlike the RTX 4090 the current curve is transcribed from
    MODEL=meta-llama/Llama-3.1-8B-Instruct GPU=L4:1 BATCHES=1,2,4,8,16,32 \
        modal run src/modal_apps/measure_gpu_energy.py

Writes results/traces/gpu_energy_<model>_<gpu>.json, shaped to drop into
layer_energy.yaml's batch_curve lists.
"""

import json
import os
import sys
from pathlib import Path

import modal

# Read at import time on the CLIENT, so `MODEL=... GPU=... modal run ...` works
# without editing the file. Modal imports this module locally to build the app.
MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen2.5-72B-Instruct")
GPU_SPEC = os.environ.get("GPU", "H100:2")
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,4,8,16,32,64").split(",")]
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", "300"))   # matches Caravaca et al.'s P(300 in, 300 out)
REPEATS = int(os.environ.get("REPEATS", "3"))
VLLM_VERSION = "0.21.0"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "nvidia-ml-py")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("tcc-gpu-energy")


@app.function(
    image=vllm_image,
    gpu=GPU_SPEC,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=2 * 60 * 60,
)
def sweep(prompts: list[str], batches: list[int], out_tokens: int, repeats: int) -> dict:
    import statistics
    import time

    import pynvml
    from vllm import LLM, SamplingParams

    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    gpu_names = [pynvml.nvmlDeviceGetName(h) for h in handles]
    gpu_names = [n.decode() if isinstance(n, bytes) else n for n in gpu_names]

    def energy_J() -> float:
        """Cumulative GPU energy across every visible device, in joules.

        nvmlDeviceGetTotalEnergyConsumption is a millijoule counter since the last
        driver reload (Volta+). A delta of it is exact over the interval -- it
        cannot miss a power spike the way periodic sampling can.
        """
        return sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles) / 1000.0

    def idle_power_W(seconds: float = 10.0) -> float:
        e0 = energy_J()
        t0 = time.perf_counter()
        time.sleep(seconds)
        return (energy_J() - e0) / (time.perf_counter() - t0)

    n_gpu = len(handles)
    print(f"Loading {MODEL_NAME} on {n_gpu}x {gpu_names[0]} ...", flush=True)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=n_gpu,
        max_model_len=4096,
        # The measurement is of steady-state decode, so let vLLM use the memory
        # it normally would in production rather than an artificially small pool.
        gpu_memory_utilization=0.90,
    )

    # ignore_eos forces exactly out_tokens per sequence, so token counts are
    # identical across every batch size and every repeat.
    sampling = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)

    baseline_W = idle_power_W()
    print(f"idle GPU power (all {n_gpu} devices): {baseline_W:.1f} W", flush=True)

    results = []
    for batch in batches:
        # Cycle the prompt pool if the batch is larger than the sample we shipped.
        chosen = [prompts[i % len(prompts)] for i in range(batch)]

        llm.generate(chosen, sampling)  # warm-up, discarded

        trials = []
        for r in range(repeats):
            e0 = energy_J()
            t0 = time.perf_counter()
            outputs = llm.generate(chosen, sampling)
            elapsed = time.perf_counter() - t0
            joules = energy_J() - e0

            gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            trials.append({
                "joules": joules,
                "seconds": elapsed,
                "gen_tokens": gen_tokens,
                "J_per_token": joules / gen_tokens,
                "tokens_per_s": gen_tokens / elapsed,
                "mean_power_W": joules / elapsed,
            })
            print(f"  batch={batch:>4} trial {r+1}/{repeats}: "
                  f"{trials[-1]['J_per_token']:.4f} J/tok, "
                  f"{trials[-1]['tokens_per_s']:.1f} tok/s, "
                  f"{trials[-1]['mean_power_W']:.0f} W", flush=True)

        med = lambda key: statistics.median(t[key] for t in trials)
        results.append({
            "batch": batch,
            "decode_J_per_token": round(med("J_per_token"), 6),
            "tokens_per_s": round(med("tokens_per_s"), 2),
            "mean_power_W": round(med("mean_power_W"), 1),
            "gen_tokens_per_trial": trials[0]["gen_tokens"],
            "trials_J_per_token": [round(t["J_per_token"], 6) for t in trials],
        })

    pynvml.nvmlShutdown()
    return {
        "model": MODEL_NAME,
        "gpu": gpu_names[0],
        "gpu_count": n_gpu,
        "gpu_spec": GPU_SPEC,
        "vllm_version": VLLM_VERSION,
        "precision": "bf16 (vLLM default for this checkpoint)",
        "boundary": "chip_or_module (NVML per-GPU counter; excludes host CPU/DRAM and datacenter PUE)",
        "method": "nvmlDeviceGetTotalEnergyConsumption delta; ignore_eos fixed-length decode; warm-up discarded; median of repeats",
        "output_tokens_per_sequence": out_tokens,
        "repeats": repeats,
        "idle_power_W": round(baseline_W, 1),
        "values_are": "GROSS (idle not subtracted)",
        "batch_curve": results,
    }


@app.local_entrypoint()
def main():
    """Build the prompt pool from GSM8K using the cascade's own template, then sweep.

    Using build_prompt() rather than synthetic text matters: the curve is meant to
    price the workload the cascade actually runs, and prompt length feeds prefill
    cost, which is inside the measured interval.
    """
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1]))          # implementation/src
    from tasks.gsm8k import build_prompt, load_gsm8k  # noqa: E402

    pool = max(BATCHES)
    items = load_gsm8k("test", limit=pool)
    prompts = [build_prompt(i.question) for i in items]
    print(f"{len(prompts)} GSM8K prompts, batches {BATCHES}, "
          f"{OUT_TOKENS} output tokens each, {REPEATS} repeats")

    report = sweep.remote(prompts, BATCHES, OUT_TOKENS, REPEATS)

    slug = MODEL_NAME.split("/")[-1].lower()
    gpu_slug = GPU_SPEC.replace(":", "x").lower()
    out = here.parents[2] / "results" / "traces" / f"gpu_energy_{slug}_{gpu_slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nwrote {out}")
    print(f"idle {report['idle_power_W']} W on {report['gpu_count']}x {report['gpu']}")
    for row in report["batch_curve"]:
        print(f"  batch {row['batch']:>4}: {row['decode_J_per_token']:>9.4f} J/token   "
              f"{row['tokens_per_s']:>9.1f} tok/s   {row['mean_power_W']:>7.1f} W")
    first, last = report["batch_curve"][0], report["batch_curve"][-1]
    print(f"\nswing over the swept range: "
          f"{first['decode_J_per_token'] / last['decode_J_per_token']:.1f}x")
