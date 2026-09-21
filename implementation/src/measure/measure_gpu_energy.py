"""First-party energy per token vs batch size, measured on a rented GPU through NVML.

ROLE IN THE PIPELINE
    measure/measure_gpu_energy.py (on a Modal L4) -> results/measurements/gpu_energy_*.json
        -> energy/three_tier.py (OltCurve) -> simulate/, analyze/tier_energy.py
    This is the OLT tier's energy: the only tier measured first-party (energy_tests.md §3).

Built for the OLT tier of the three-tier study case (user -> ONU -> OLT): a
shared tier whose cost per query should fall as its batch grows, possibly far
enough to undercut the ONU below it. Whether it does is the question this
measures. Nothing in the literature answers it for an OLT-class GPU past batch 4.

WHAT IT REPORTS, PER BATCH SIZE
-------------------------------
  prefill_J_per_input_token    energy of reading the prompt, per prompt token
  decode_J_per_output_token    energy of generating, per generated token
  total_J_per_output_token     the old blended figure (both phases / output tokens),
                               kept so earlier runs stay comparable
  average_J_per_query          batch energy / batch size -- what a query COSTS
  marginal_J_per_query         extra energy of one more query in the batch --
                               what SENDING it up actually adds

Every energy figure appears twice: gross (what the GPU drew) and `_net` (with
idle power x duration subtracted). Papers split on which they report, so both
are recorded and the comparison picks.

Why phases are split: the phase-split papers (TokenPowerBench, Solovyeva and
Castor) and this project's own convention price prefill per INPUT token and
decode per OUTPUT token, because the two scale differently. Prefill is
compute-bound and does not get cheaper with batching; decode is memory-bound and
does. A blended figure hides exactly the difference that decides whether the OLT
can undercut the ONU.

Why marginal as well as average: while decode is memory-bound, the GPU draws the
same power whether it carries one sequence or several, so adding a query to a
batch costs far less than the batch's average cost per query. A lower tier
deciding whether to send a query up should weigh the marginal figure; accounting
for who spent what should use the average.

METHOD
------
Energy is a delta of NVML's cumulative counter, nvmlDeviceGetTotalEnergyConsumption
(millijoules since driver load, Volta and later). This is the API ML.ENERGY
recommends, and unlike power sampling it cannot miss a spike between polls.
llm.generate() blocks until the GPU has finished, so the window is exact.

  * boundary = GPU chip/module: host CPU, DRAM and datacenter PUE are not counted.
    energy/three_tier.py converts it to the whole system (factors in
    config/energy_sources.yaml).
  * prefix caching OFF. vLLM enables it by default, and the discarded warm-up
    would otherwise cache every prompt, so measured trials would skip prefill
    almost entirely -- zeroing the very term being measured. (Runs made before
    this fix therefore measured close to pure decode, by accident.)
  * phase split by difference. Each trial runs the same prompts twice: once with
    max_tokens=1, which is essentially the prefill forward pass (the first token
    comes out of it), then in full. Prefill = first pass; decode = full minus
    first, spread over the remaining output tokens. Under continuous batching the
    two phases can overlap slightly, so this is a close approximation, not an
    exact partition.
  * the prefill pass is repeated back to back until it spans >= 3 s, then
    averaged. NVML's counter advances in steps of ~0.1 s of energy, so a single
    batch-1 prefill (tens of milliseconds) reads either 0 J or one whole step --
    observed on the first L4 run, where one trial read 6.75 J (= 72 W x 0.094 s)
    and the next two read exactly 0. Same practice the SC'24 study of NVIDIA's
    power sensor recommends for short kernels.
  * ignore_eos=True with a fixed max_tokens, so every batch size generates the
    same number of tokens per sequence and J/token is comparable across the sweep.
  * one warm-up per batch size, discarded -- CUDA graph capture and allocator
    warm-up are start-up cost, not inference cost.
  * repeats with a median, since a rented container is noisier than a bench.
  * static batches: all prompts submitted at once. Standard for batch curves, but
    a live OLT sees random arrivals and a fluctuating batch; state that when
    using the numbers.

USAGE
-----
    # the OLT study case -- the defaults below: Qwen2.5-7B, fp8, one L4 (~$0.80/h)
    modal run src/measure/measure_gpu_energy.py        # ~15 min on one L4 (~$0.80/h)

    # another model inside the 7-13B OLT band
    MODEL=google/gemma-2-9b-it GPU=L4:1 QUANTIZATION=fp8 \
        modal run src/measure/measure_gpu_energy.py

Writes results/measurements/gpu_energy_<model>_<gpu>_<UTC>.json. The thesis uses the two
runs saved as gpu_energy_qwen2.5-7b-instruct_l4x1_run{1,2}.json (renamed by hand).

CONFIG MUST TRAVEL AS ARGUMENTS, NOT AS MODULE GLOBALS
-------------------------------------------------------
The env vars are read on the CLIENT at import. The remote function body runs in
a container where they do not exist, so a module-level constant read inside
`sweep` silently reverts to its default. That is not hypothetical: two runs
launched with MODEL=Qwen/Qwen2.5-7B-Instruct QUANTIZATION=fp8 actually loaded
the then-default 72B at bf16 and died on OOM. Everything `sweep` needs is
therefore a parameter. The decorator's `gpu=` is the one exception --
decorators are evaluated on the client.

CARD SIZES
----------
An L4 exposes 22.03 GiB. At ~2 bytes/param for bf16 and ~1 for fp8, a 7.6B model
is 15.2 GB in bf16 (fits, but leaves little KV cache for batching) and 7.6 GB in
fp8 (comfortable; Ada supports fp8 natively). A 13B in fp8 fits with much less
room to batch; in bf16 it does not fit at all.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import modal

MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU_SPEC = os.environ.get("GPU", "L4:1")
BATCHES = [int(b) for b in os.environ.get("BATCHES", "1,2,4,8,16,32,64").split(",")]   # the thesis sweep: 1..64
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", "300"))
REPEATS = int(os.environ.get("REPEATS", "3"))
# GSM8K prompts run ~150 tokens and answers ~300, so 2048 is ample. Keeping it
# tight matters on a 24 GB card: KV cache is what is left after the weights, and
# max_model_len x batch is what has to fit in it.
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


# ---------------------------------------------------------------------------
# Arithmetic on measured numbers. Pure functions, so they run the same in the
# container and can be checked locally without a GPU.
# ---------------------------------------------------------------------------

def summarize_trial(prefill: dict, full: dict, idle_W: float, batch: int) -> dict:
    """Split one trial into its two phases, gross and net of idle.

    prefill -- the max_tokens=1 pass: {"joules", "seconds", "in_tokens"}
    full    -- the full pass: {"joules", "seconds", "in_tokens", "out_tokens"}
    """
    prefill_net = prefill["joules"] - idle_W * prefill["seconds"]
    full_net = full["joules"] - idle_W * full["seconds"]
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
    """One row per batch size: the median of every field across repeats."""
    row = {"batch": batch}
    for key in trials[0]:
        row[key] = statistics.median(t[key] for t in trials)
    row["trials_decode_J_per_output_token"] = [t["decode_J_per_output_token"] for t in trials]
    return row


def add_per_query(rows: list[dict]) -> list[dict]:
    """Average and marginal energy per query, gross and net.

    Marginal is the slope of batch energy between this batch size and the
    previous measured one: the extra joules one more query added in that range.
    For the smallest batch there is nothing to share with, so marginal = average.
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
                    (row[f"batch_J{suffix}"] - prev[f"batch_J{suffix}"])
                    / (row["batch"] - prev["batch"]))
        prev = row
    return rows


# ---------------------------------------------------------------------------
# The measurement, on the rented GPU.
# ---------------------------------------------------------------------------

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
    """Everything this body needs is an ARGUMENT, never a module-level global --
    see CONFIG MUST TRAVEL AS ARGUMENTS in the module docstring."""
    import time

    import pynvml
    from vllm import LLM, SamplingParams

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

    def energy_J() -> float:
        """Cumulative GPU energy across every visible device, in joules."""
        return sum(pynvml.nvmlDeviceGetTotalEnergyConsumption(h) for h in handles) / 1000.0

    def power_W() -> float:
        """Instantaneous power of every visible GPU, in watts (fallback path only)."""
        return sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in handles) / 1000.0

    def measure(fn):
        """Run fn, returning (result, joules, seconds) over exactly its execution."""
        if counter_ok:
            e0, t0 = energy_J(), time.perf_counter()
            out = fn()
            return out, energy_J() - e0, time.perf_counter() - t0

        import threading
        samples: list[tuple[float, float]] = []
        stop = threading.Event()

        def poll():
            """Sample power every 50 ms until told to stop (fallback path only)."""
            while not stop.is_set():
                samples.append((time.perf_counter(), power_W()))
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

    n_gpu = len(handles)
    print(f"Loading {model_name} on {n_gpu}x {gpu_names[0]} "
          f"(quantization={quantization}, max_model_len={max_model_len}) ...", flush=True)
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

    # Idle with the model resident: the power the GPU draws whether or not a
    # query arrives. Measured once, used for every _net figure.
    _, idle_joules, idle_seconds = measure(lambda: time.sleep(10.0))
    idle_W = idle_joules / idle_seconds
    print(f"idle GPU power (all {n_gpu} devices): {idle_W:.1f} W", flush=True)

    def run(chosen, params, min_seconds: float = 0.0):
        """One pass over `chosen`, measured, per-pass figures returned.

        NVML's counter advances in steps -- about 0.1 s of energy on the L4 (a
        batch-1 prefill read either 0 J or exactly 72 W x 0.094 s). A pass shorter
        than that cannot be timed by the counter, so short passes are repeated back
        to back until the measured window lasts at least min_seconds, then averaged.
        Decode passes run for seconds and need no repetition.
        """
        reps = 1
        if min_seconds > 0:
            t0 = time.perf_counter()
            llm.generate(chosen, params)                    # timing only, not measured
            single = max(time.perf_counter() - t0, 1e-3)
            reps = max(1, int(-(-min_seconds // single)))   # ceil
        outputs, joules, seconds = measure(
            lambda: [llm.generate(chosen, params) for _ in range(reps)][-1])
        return {
            "joules": joules / reps,
            "seconds": seconds / reps,
            "reps": reps,
            "in_tokens": sum(len(o.prompt_token_ids) for o in outputs),
            "out_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
        }

    rows = []
    for batch in batches:
        chosen = [prompts[i % len(prompts)] for i in range(batch)]
        llm.generate(chosen, full_pass)  # warm-up, discarded
        trials = []
        for r in range(repeats):
            prefill = run(chosen, prefill_pass, min_seconds=3.0)
            trial = summarize_trial(prefill, run(chosen, full_pass), idle_W, batch)
            trial["prefill_reps"] = prefill["reps"]
            trials.append(trial)
            print(f"  batch={batch:>3} trial {r + 1}/{repeats}: "
                  f"prefill {trial['prefill_J_per_input_token']:.5f} J/in-tok, "
                  f"decode {trial['decode_J_per_output_token']:.4f} J/out-tok, "
                  f"{trial['tokens_per_s']:.0f} tok/s, {trial['mean_power_W']:.0f} W", flush=True)
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
        "method": ("static batches; ignore_eos fixed-length generation; warm-up discarded; "
                   "median of repeats; phases split by difference between a max_tokens=1 "
                   "pass and a full pass over the same prompts"),
        "output_tokens_per_sequence": out_tokens,
        "repeats": repeats,
        "idle_power_W": round(idle_W, 2),
        "values_are": "gross (what the GPU drew); fields ending _net subtract idle power x duration",
        "batch_curve": add_per_query(rows),
    }


@app.local_entrypoint()
def main():
    """Build the prompt pool from GSM8K with the cascade's own template, then sweep.

    build_prompt() rather than synthetic text, because prompt length feeds the
    prefill cost this run measures, and it should be the workload the cascade runs.
    """
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1]))          # implementation/src
    from measure.gsm8k import build_prompt, load_gsm8k  # noqa: E402

    items = load_gsm8k("test", limit=max(BATCHES))
    prompts = [build_prompt(i.question) for i in items]
    print(f"{MODEL_NAME} on {GPU_SPEC}, {QUANTIZATION or 'bf16'}: {len(prompts)} GSM8K prompts, "
          f"batches {BATCHES}, {OUT_TOKENS} output tokens each, {REPEATS} repeats")

    report = sweep.remote(
        prompts, BATCHES, OUT_TOKENS, REPEATS,
        MODEL_NAME, QUANTIZATION, MAX_MODEL_LEN, GPU_MEM_UTIL,
    )

    slug = MODEL_NAME.split("/")[-1].lower()
    gpu_slug = GPU_SPEC.replace(":", "x").lower()
    # Timestamped, so a rerun never overwrites an earlier result.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = here.parents[2] / "results" / "measurements" / f"gpu_energy_{slug}_{gpu_slug}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nwrote {out}")
    print(f"idle {report['idle_power_W']} W on {report['gpu_count']}x {report['gpu']}\n")
    print(f"{'batch':>5} {'prefill J/in':>13} {'decode J/out':>13} {'avg J/query':>12} "
          f"{'marg J/query':>13} {'tok/s':>7} {'W':>5}")
    for row in report["batch_curve"]:
        print(f"{row['batch']:>5} {row['prefill_J_per_input_token']:>13.5f} "
              f"{row['decode_J_per_output_token']:>13.4f} {row['average_J_per_query']:>12.2f} "
              f"{row['marginal_J_per_query']:>13.2f} {row['tokens_per_s']:>7.0f} "
              f"{row['mean_power_W']:>5.0f}")
    first, last = report["batch_curve"][0], report["batch_curve"][-1]
    print(f"\ndecode swing over the sweep: "
          f"{first['decode_J_per_output_token'] / last['decode_J_per_output_token']:.1f}x")
