"""Measure the OLT's energy per token against batch size, on a rented GPU through NVML.

The OLT is the only tier measured first-hand. Per batch size it reports prefill energy per
prompt token, decode energy per generated token, and the average and marginal energy per
query, each gross and net of idle power. Writes
results/measurements/gpu_energy_<model>_<gpu>_<UTC>.json.

Usage:
    modal run src/measure/measure_gpu_energy.py   # Qwen2.5-7B, fp8, one L4 (~15 min)
    MODEL=google/gemma-2-9b-it GPU=L4:1 QUANTIZATION=fp8 modal run src/measure/measure_gpu_energy.py
"""

import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import modal

# ==========================================
# Settings
# ==========================================
# Read from the environment on the machine that launches the run. They must reach sweep()
# as arguments: inside the container these variables do not exist, so a module-level
# constant read there silently reverts to its default (two runs once loaded a 72B model at
# bf16 this way and ran out of memory). The decorator's gpu= is the one exception.
MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU_SPEC = os.environ.get("GPU", "L4:1")
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,4,8,16,32,64").split(",")]
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", "300"))
REPEATS = int(os.environ.get("REPEATS", "3"))
# GSM8K prompts run ~150 tokens and answers ~300, so 2048 is ample. Keeping it tight matters
# on a 24 GB card: the KV cache is what is left after the weights (a 7.6B model is ~15 GB in
# bf16, ~7.6 GB in fp8), and max_model_len x batch has to fit in it.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
QUANTIZATION = os.environ.get("QUANTIZATION", "fp8") or None
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.90"))

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


# ==========================================
# Arithmetic on measured numbers
# ==========================================
# Pure functions, so they run the same in the container and can be checked without a GPU.
def summarize_trial(prefill: dict, full: dict, idle_watts: float, batch: int) -> dict:
    """Split one trial into its two phases, gross and net of idle.

    Args:
        prefill: The max_tokens=1 pass: {"joules", "seconds", "in_tokens"}.
        full: The full pass: {"joules", "seconds", "in_tokens", "out_tokens"}.
        idle_watts: The GPU's idle power with the model loaded.
        batch: The batch size.

    Returns:
        Per-token and per-batch energy, throughput and power for this trial.
    """
    prefill_net = prefill["joules"] - idle_watts * prefill["seconds"]
    full_net = full["joules"] - idle_watts * full["seconds"]
    # The first pass already produced one token per sequence, so the difference
    # between the passes is the decode of the remaining out_tokens - batch.
    decode_tokens = full["out_tokens"] - batch
    return {
        "prefill_J_per_input_token": prefill["joules"] / prefill["in_tokens"],
        "prefill_J_per_input_token_net": prefill_net / prefill["in_tokens"],
        "decode_J_per_output_token": (full["joules"] - prefill["joules"]) / decode_tokens,
        "decode_J_per_output_token_net": (full_net - prefill_net) / decode_tokens,
        "total_J_per_output_token": full["joules"] / full["out_tokens"],
        "batch_J": full["joules"],
        "batch_J_net": full_net,
        "batch_seconds": full["seconds"],
        "tokens_per_s": full["out_tokens"] / full["seconds"],
        "mean_power_W": full["joules"] / full["seconds"],
        "input_tokens": full["in_tokens"],
        "output_tokens": full["out_tokens"],
    }


def median_row(batch: int, trials: list[dict]) -> dict:
    """Return one row per batch size: the median of every field across repeats.

    Args:
        batch: The batch size.
        trials: The repeats' summarize_trial results.
    """
    row = {"batch": batch}
    for key in trials[0]:
        row[key] = statistics.median(t[key] for t in trials)
    row["trials_decode_J_per_output_token"] = [t["decode_J_per_output_token"] for t in trials]
    return row


def add_per_query(rows: list[dict]) -> list[dict]:
    """Add the average and marginal energy per query, gross and net, to every row.

    Marginal is the slope of batch energy between this batch size and the previous
    measured one: the extra joules one more query added in that range. For the smallest
    batch there is nothing to share with, so marginal = average.

    Args:
        rows: One row per batch size.

    Returns:
        The rows, sorted by batch size.
    """
    rows = sorted(rows, key=lambda r: r["batch"])
    prev = None
    for row in rows:
        for suffix in ("", "_net"):
            row[f"average_J_per_query{suffix}"] = row[f"batch_J{suffix}"] / row["batch"]
            if prev is None:
                row[f"marginal_J_per_query{suffix}"] = row[f"average_J_per_query{suffix}"]
            else:
                row[f"marginal_J_per_query{suffix}"] = (
                    row[f"batch_J{suffix}"] - prev[f"batch_J{suffix}"]
                ) / (row["batch"] - prev["batch"])
        prev = row
    return rows


# ==========================================
# The measurement, on the rented GPU
# ==========================================
# Method, as the thesis reports it:
#   * energy is a delta of NVML's cumulative counter (nvmlDeviceGetTotalEnergyConsumption,
#     the API ML.ENERGY recommends); unlike power sampling it cannot miss a spike, and
#     llm.generate() blocks until the GPU has finished, so the window is exact
#   * boundary = the GPU card; energy/three_tier.py converts it to the whole system
#   * prefix caching off: the discarded warm-up would otherwise cache every prompt and the
#     measured trials would skip prefill almost entirely
#   * phases split by difference: each trial runs the prompts with max_tokens=1 (prefill)
#     and in full; decode = full minus prefill, over the remaining tokens (a close
#     approximation, since the phases can overlap slightly under continuous batching)
#   * short passes repeated until they span >= 3 s: the counter advances in ~0.1 s steps,
#     so a single batch-1 prefill reads either 0 J or one whole step
#   * ignore_eos with a fixed max_tokens, so every batch size generates the same tokens
#   * one discarded warm-up per batch size; the median of the repeats
#   * static batches (all prompts at once): a live OLT sees random arrivals instead


@app.function(
    image=vllm_image,
    gpu=GPU_SPEC,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=2 * 60 * 60,
)
def sweep(
    prompts: list[str],
    batches: list[int],
    out_tokens: int,
    repeats: int,
    model_name: str,
    quantization: str | None,
    max_model_len: int,
    gpu_mem_util: float,
) -> dict:
    """Measure the energy of every batch size, on a Modal GPU.

    Everything this body needs is an argument, never a module-level setting (see Settings).

    Args:
        prompts: The GSM8K prompt pool.
        batches: The batch sizes to measure.
        out_tokens: Tokens generated per sequence.
        repeats: Trials per batch size.
        model_name: The Hugging Face model.
        quantization: vLLM's quantization, or None for bf16.
        max_model_len: vLLM's context length.
        gpu_mem_util: vLLM's GPU memory fraction.

    Returns:
        The report: model, GPU, method, idle power and the batch curve.
    """
    # These packages exist only inside the Modal image, not on the machine that launches it.
    # pylint: disable=import-outside-toplevel,import-error
    import pynvml
    from vllm import LLM, SamplingParams

    # --- GPUs and energy source ---
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    gpu_names = [pynvml.nvmlDeviceGetName(h) for h in handles]
    gpu_names = [n.decode() if isinstance(n, bytes) else n for n in gpu_names]

    # Prefer the cumulative counter; fall back to integrating instantaneous power
    # only if the driver or a virtualized device refuses it. The fallback CAN miss
    # a spike between polls, so which one was used is recorded in the report.
    try:
        for h in handles:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        energy_method = "nvmlDeviceGetTotalEnergyConsumption (cumulative mJ counter)"
    except pynvml.NVMLError as exc:
        energy_method = f"power-sampling fallback ({type(exc).__name__}: counter unsupported)"
        print(f"WARNING: energy counter unavailable ({exc}); integrating power instead", flush=True)
    counter_ok = energy_method.startswith("nvml")

    def energy_joules() -> float:
        """Return the cumulative GPU energy across every visible device, in joules."""
        return sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles) / 1000.0

    def power_watts() -> float:
        """Return the instantaneous power of every visible GPU, in watts (fallback path only)."""
        return sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in handles) / 1000.0

    def measure(fn):
        """Run fn, returning (result, joules, seconds) over exactly its execution."""
        if counter_ok:
            e0, t0 = energy_joules(), time.perf_counter()
            out = fn()
            return out, energy_joules() - e0, time.perf_counter() - t0

        samples: list[tuple[float, float]] = []
        stop = threading.Event()

        def poll():
            """Sample power every 50 ms until told to stop (fallback path only)."""
            while not stop.is_set():
                samples.append((time.perf_counter(), power_watts()))
                time.sleep(0.05)

        thread = threading.Thread(target=poll, daemon=True)
        t0 = time.perf_counter()
        thread.start()
        out = fn()
        stop.set()
        thread.join()
        elapsed = time.perf_counter() - t0
        joules = sum(
            (samples[i + 1][1] + samples[i][1]) / 2 * (samples[i + 1][0] - samples[i][0])
            for i in range(len(samples) - 1)
        )
        return out, joules, elapsed

    # --- Model ---
    n_gpu = len(handles)
    print(
        f"Loading {model_name} on {n_gpu}x {gpu_names[0]} "
        f"(quantization={quantization}, max_model_len={max_model_len}) ...",
        flush=True,
    )
    llm = LLM(
        model=model_name,
        tensor_parallel_size=n_gpu,
        max_model_len=max_model_len,
        quantization=quantization,
        gpu_memory_utilization=gpu_mem_util,
        # ON by default in vLLM. Left on, the warm-up would cache every prompt and
        # the measured trials would skip prefill -- the term this run exists to price.
        enable_prefix_caching=False,
    )
    full_pass = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)
    prefill_pass = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    # --- Idle power ---
    # With the model resident: what the GPU draws whether or not a query arrives, measured
    # once and used for every _net figure.
    _, idle_joules, idle_seconds = measure(lambda: time.sleep(10.0))
    idle_watts = idle_joules / idle_seconds
    print(f"idle GPU power (all {n_gpu} devices): {idle_watts:.1f} W", flush=True)

    def run(chosen, params, min_seconds: float = 0.0):
        """Run one measured pass over `chosen` and return its per-pass figures.

        NVML's counter advances in ~0.1 s steps on the L4, so a pass shorter than that is
        repeated back to back until the window lasts at least min_seconds, then averaged.

        Args:
            chosen: The prompts of this batch.
            params: The sampling parameters (prefill or full pass).
            min_seconds: The shortest window the counter can time reliably (0: one pass).
        """
        reps = 1
        if min_seconds > 0:
            t0 = time.perf_counter()
            llm.generate(chosen, params)  # timing only, not measured
            single = max(time.perf_counter() - t0, 1e-3)
            reps = max(1, int(-(-min_seconds // single)))  # ceil
        outputs, joules, seconds = measure(
            lambda: [llm.generate(chosen, params) for _ in range(reps)][-1]
        )
        return {
            "joules": joules / reps,
            "seconds": seconds / reps,
            "reps": reps,
            "in_tokens": sum(len(o.prompt_token_ids) for o in outputs),
            "out_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
        }

    # --- Sweep the batch sizes ---
    rows = []
    for batch in batches:
        chosen = [prompts[i % len(prompts)] for i in range(batch)]
        llm.generate(chosen, full_pass)  # warm-up, discarded
        trials = []
        for r in range(repeats):
            prefill = run(chosen, prefill_pass, min_seconds=3.0)
            trial = summarize_trial(prefill, run(chosen, full_pass), idle_watts, batch)
            trial["prefill_reps"] = prefill["reps"]
            trials.append(trial)
            print(
                f"  batch={batch:>3} trial {r + 1}/{repeats}: "
                f"prefill {trial['prefill_J_per_input_token']:.5f} J/in-tok, "
                f"decode {trial['decode_J_per_output_token']:.4f} J/out-tok, "
                f"{trial['tokens_per_s']:.0f} tok/s, {trial['mean_power_W']:.0f} W",
                flush=True,
            )
        rows.append(median_row(batch, trials))

    pynvml.nvmlShutdown()
    return {
        "model": model_name,
        "gpu": gpu_names[0],
        # From NVML inside the container, so it is ground truth; GPU_SPEC would
        # revert to its default here, like MODEL_NAME once did.
        "gpu_count": n_gpu,
        "vllm_version": VLLM_VERSION,
        "precision": quantization or "bf16 (vLLM default for this checkpoint)",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_mem_util,
        "prefix_caching": "disabled",
        "boundary": "chip_or_module (NVML per-GPU counter; excludes host CPU/DRAM and datacenter PUE)",
        "energy_source": energy_method,
        "method": (
            "static batches; ignore_eos fixed-length generation; warm-up discarded; "
            "median of repeats; phases split by difference between a max_tokens=1 "
            "pass and a full pass over the same prompts"
        ),
        "output_tokens_per_sequence": out_tokens,
        "repeats": repeats,
        "idle_power_W": round(idle_watts, 2),
        "values_are": "gross (what the GPU drew); fields ending _net subtract idle power x duration",
        "batch_curve": add_per_query(rows),
    }


@app.local_entrypoint()
def main():
    """Build the prompt pool from GSM8K with the cascade's own template, then sweep and write.

    Real GSM8K prompts rather than synthetic text, because prompt length feeds the prefill
    cost this run measures.
    """
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1]))  # implementation/src
    # imported here: this file also runs inside the Modal container, which has no src/
    from measure.gsm8k import build_prompt, load_gsm8k  # pylint: disable=import-outside-toplevel

    items = load_gsm8k("test", limit=max(BATCHES))
    prompts = [build_prompt(i.question) for i in items]
    print(
        f"{MODEL_NAME} on {GPU_SPEC}, {QUANTIZATION or 'bf16'}: {len(prompts)} GSM8K prompts, "
        f"batches {BATCHES}, {OUT_TOKENS} output tokens each, {REPEATS} repeats"
    )

    report = sweep.remote(
        prompts,
        BATCHES,
        OUT_TOKENS,
        REPEATS,
        MODEL_NAME,
        QUANTIZATION,
        MAX_MODEL_LEN,
        GPU_MEM_UTIL,
    )

    slug = MODEL_NAME.split("/")[-1].lower()
    gpu_slug = GPU_SPEC.replace(":", "x").lower()
    # Timestamped, so a rerun never overwrites an earlier result.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = (
        here.parents[2] / "results" / "measurements" / f"gpu_energy_{slug}_{gpu_slug}_{stamp}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {out}")
    print(f"idle {report['idle_power_W']} W on {report['gpu_count']}x {report['gpu']}\n")
    print(
        f"{'batch':>5} {'prefill J/in':>13} {'decode J/out':>13} {'avg J/query':>12} "
        f"{'marg J/query':>13} {'tok/s':>7} {'W':>5}"
    )
    for row in report["batch_curve"]:
        print(
            f"{row['batch']:>5} {row['prefill_J_per_input_token']:>13.5f} "
            f"{row['decode_J_per_output_token']:>13.4f} {row['average_J_per_query']:>12.2f} "
            f"{row['marginal_J_per_query']:>13.2f} {row['tokens_per_s']:>7.0f} "
            f"{row['mean_power_W']:>5.0f}"
        )
    first, last = report["batch_curve"][0], report["batch_curve"][-1]
    print(
        f"\ndecode swing over the sweep: "
        f"{first['decode_J_per_output_token'] / last['decode_J_per_output_token']:.1f}x"
    )
