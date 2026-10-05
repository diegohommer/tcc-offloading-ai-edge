"""Measure the OLT's energy under continuous batching, on a rented GPU through NVML.

The static sweep (measure_gpu_energy_static.py) starts every sequence of a batch together;
a live OLT admits queries as they arrive and lets each leave when its answer is done. This
runs the OLT's real GSM8K answer lengths through vLLM's async engine at fixed concurrency
and under Poisson arrivals. Writes
results/measurements/gpu_energy_continuous_<model>_<gpu>_<UTC>.json.

Usage:
    modal run src/measure/measure_gpu_energy_continuous.py          # Qwen2.5-7B, fp8, one L4 (~1 h)
    SMOKE=1 modal run src/measure/measure_gpu_energy_continuous.py  # a short check (~5 min)
"""

import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import modal

# ==========================================
# Settings
# ==========================================
# Read from the environment on the machine that launches the run, and passed to measure()
# as arguments: inside the container these variables do not exist (measure_gpu_energy_static.py).
SMOKE = os.environ.get("SMOKE") == "1"
MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU_SPEC = os.environ.get("GPU", "L4:1")
QUANTIZATION = os.environ.get("QUANTIZATION", "fp8") or None
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
MAX_SLOTS = int(os.environ.get("MAX_SLOTS", "64"))
CONCURRENCY = [
    int(level)
    for level in os.environ.get(
        "CONCURRENCY", "1,8" if SMOKE else "1,2,3,4,6,8,12,16,24,32,48,64"
    ).split(",")
]
WARMUP_S = float(os.environ.get("WARMUP_S", "10" if SMOKE else "30"))
WINDOW_S = float(os.environ.get("WINDOW_S", "20" if SMOKE else "90"))
LOADS = [float(load) for load in os.environ.get("LOADS", "2" if SMOKE else "1,2,4,8,16").split(",")]
POISSON_S = float(os.environ.get("POISSON_S", "40" if SMOKE else "300"))
REPEATS = int(os.environ.get("REPEATS", "1" if SMOKE else "3"))
"""Back-to-back windows per concurrency level, and arrival seeds per Poisson load."""
SEED = int(os.environ.get("SEED", "7"))

SERVICE_S = 8.7
"""Seconds an average OLT answer takes at small batch (251 tokens at ~29 tokens/s): turns a
target number of requests in flight into an arrival rate, by Little's law."""

VLLM_VERSION = "0.21.0"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "nvidia-ml-py")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("tcc-continuous-batching")


# ==========================================
# Arithmetic on measured numbers
# ==========================================
# Pure functions, so they run the same in the container and can be checked without a GPU.
def poisson_arrivals(load: float, duration_s: float, pool_size: int, seed: int) -> list:
    """Return the requests of one Poisson load, in arrival order.

    Args:
        load: The mean number of requests in flight the arrivals should produce.
        duration_s: How long requests keep arriving.
        pool_size: How many requests there are to draw from.
        seed: Random seed.

    Returns:
        [(seconds after the start, request index)].
    """
    rng = random.Random(seed)
    rate = load / SERVICE_S
    arrivals, clock = [], rng.expovariate(rate)
    while clock < duration_s:
        arrivals.append((clock, rng.randrange(pool_size)))
        clock += rng.expovariate(rate)
    return arrivals


def mean_in_flight(spans: list, start: float, end: float) -> float:
    """Return the time-weighted number of requests in flight between start and end.

    Args:
        spans: [(submitted, finished)] per request, in seconds; inf for one still running.
        start: Window start.
        end: Window end.
    """
    overlap = sum(max(0.0, min(done, end) - max(sent, start)) for sent, done in spans)
    return overlap / (end - start)


def window_tokens(events: list, start: float, end: float) -> tuple[int, int]:
    """Return the prompt and generated tokens processed in [start, end).

    Args:
        events: [(seconds, prompt tokens, generated tokens)] as the engine streamed them.
        start: Window start.
        end: Window end.

    Returns:
        (prompt tokens, generated tokens).
    """
    inside = [event for event in events if start <= event[0] < end]
    return sum(event[1] for event in inside), sum(event[2] for event in inside)


def summarize_window(
    level: int,
    events: list,
    spans: list,
    window: tuple[float, float],
    joules: float,
    idle_watts: float,
) -> dict:
    """Return one fixed-concurrency row: tokens, energy and power over the measured window.

    Args:
        level: The requests kept in flight.
        events: [(seconds, prompt tokens, generated tokens)] as the engine streamed them.
        spans: [(submitted, finished)] per request, in seconds.
        window: (start, end) of the measured window, in seconds.
        joules: What the energy counter advanced over the window.
        idle_watts: The GPU's idle power with the model loaded.
    """
    start, end = window
    seconds = end - start
    prompt_tokens, generated_tokens = window_tokens(events, start, end)
    net = joules - idle_watts * seconds
    return {
        "concurrency": level,
        "window_s": round(seconds, 2),
        "mean_in_flight": round(mean_in_flight(spans, start, end), 3),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "joules": joules,
        "joules_net": net,
        "tokens_per_s": generated_tokens / seconds,
        "mean_power_W": joules / seconds,
        "J_per_generated_token": joules / generated_tokens,
        "J_per_generated_token_net": net / generated_tokens,
    }


def median_of_windows(level: int, trials: list[dict]) -> dict:
    """Return one row per concurrency level: the median of every field across its windows.

    Args:
        level: The requests kept in flight.
        trials: The level's summarize_window results.
    """
    row = {"concurrency": level}
    for key in trials[0]:
        if key != "concurrency":
            row[key] = statistics.median(trial[key] for trial in trials)
    row["trials_J_per_generated_token"] = [trial["J_per_generated_token"] for trial in trials]
    return row


# ==========================================
# The measurement, on the rented GPU
# ==========================================
# Method, as the thesis reports it:
#   * vLLM's async engine, as a server runs it: requests join the running batch as they
#     arrive and leave when done, with chunked prefill mixing joiners into decode steps
#   * the OLT's own GSM8K questions in its chat template, each generating exactly the
#     tokens the OLT answered it with (ignore_eos), so lengths are real and repeatable
#   * fixed concurrency: a new request starts the moment one finishes; energy and tokens
#     are read over steady-state windows after a warm-up, as ML.ENERGY measures, and the
#     median of the windows reported with every window kept
#   * Poisson loads, each at several arrival seeds: arrival, finish and tokens of every
#     request, and the energy from first arrival to last finish, so the simulator can
#     replay the same arrivals
#   * the GPU's temperature and SM clock read at every window's edges, to show it did not
#     throttle; its UUID, driver and power limit recorded
#   * the same counter, boundary, precision and prefix-caching setting as the static sweep


@app.function(
    image=vllm_image,
    gpu=GPU_SPEC,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=3 * 60 * 60,
)
def measure(
    requests: list[dict],
    concurrency: list[int],
    warmup_s: float,
    window_s: float,
    repeats: int,
    poisson_loads: list[tuple[float, int, list]],
    model_name: str,
    quantization: str | None,
    max_model_len: int,
    gpu_mem_util: float,
    max_slots: int,
) -> dict:
    """Measure the fixed-concurrency sweep and the Poisson loads, on a Modal GPU.

    Everything this body needs is an argument, never a module-level setting (see Settings).

    Args:
        requests: [{"prompt", "generated_tokens"}]: the OLT's questions and answer lengths.
        concurrency: The numbers of requests to keep in flight.
        warmup_s: Seconds each level runs before its window opens.
        window_s: Seconds each measured window lasts.
        repeats: Back-to-back windows measured per level.
        poisson_loads: [(target load, seed, arrivals)], arrivals as poisson_arrivals() returns.
        model_name: The Hugging Face model.
        quantization: vLLM's quantization, or None for bf16.
        max_model_len: vLLM's context length.
        gpu_mem_util: vLLM's GPU memory fraction.
        max_slots: vLLM's max_num_seqs: the most requests it batches at once.

    Returns:
        The report: model, GPU, method, idle power, the fixed rows and the Poisson runs.
    """
    # These packages exist only inside the Modal image, not on the machine that launches it.
    # pylint: disable=import-outside-toplevel,import-error
    import asyncio
    import itertools

    import pynvml
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    # --- GPU and energy counter ---
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)

    def energy_joules() -> float:
        """Return the GPU's cumulative energy counter, in joules."""
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0

    def gpu_state() -> tuple[int, int]:
        """Return the GPU's (temperature in C, SM clock in MHz) right now."""
        return (
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        )

    def text(value) -> str:
        """Return an NVML string as text, whichever type the binding gives."""
        return value.decode() if isinstance(value, bytes) else value

    # --- Prompts, in the chat template the OLT answered them in ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": request["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for request in requests
    ]
    request_ids = itertools.count()

    async def stream(engine, index: int, events: list) -> tuple[float, float, int, int]:
        """Send one request and record its tokens as the engine streams them.

        Args:
            engine: The running engine.
            index: The request to send.
            events: Where (seconds, prompt tokens, generated tokens) are appended.

        Returns:
            (submitted, finished, prompt tokens, generated tokens), clock in seconds.
        """
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max(int(requests[index]["generated_tokens"]), 1),
            ignore_eos=True,
        )
        submitted, generated, prompt_tokens = time.perf_counter(), 0, 0
        async for output in engine.generate(prompts[index], params, str(next(request_ids))):
            now = time.perf_counter()
            total = len(output.outputs[0].token_ids)
            if not prompt_tokens:
                prompt_tokens = len(output.prompt_token_ids)
                events.append((now, prompt_tokens, 0))
            events.append((now, 0, total - generated))
            generated = total
        return submitted, time.perf_counter(), prompt_tokens, generated

    async def fixed_concurrency(engine, level: int, idle_watts: float) -> dict:
        """Keep `level` requests in flight, and measure back-to-back windows after the warm-up.

        Args:
            engine: The running engine.
            level: The requests to keep in flight.
            idle_watts: The GPU's idle power with the model loaded.

        Returns:
            The level's row (median_of_windows), every window kept under "trials".
        """
        order = list(range(len(requests)))
        random.Random(level).shuffle(order)
        queue = itertools.cycle(order)
        events, spans = [], []

        async def worker():
            """Send requests one after another, for as long as the level runs.

            A request's span is opened when it is sent, so one still running when a window
            closes counts as in flight up to that moment.
            """
            while True:
                span = [time.perf_counter(), math.inf]
                spans.append(span)
                span[1] = (await stream(engine, next(queue), events))[1]

        workers = [asyncio.create_task(worker()) for _ in range(level)]
        await asyncio.sleep(warmup_s)
        trials = []
        for _ in range(repeats):
            state_start = gpu_state()
            start, energy_start = time.perf_counter(), energy_joules()
            await asyncio.sleep(window_s)
            end, energy_end = time.perf_counter(), energy_joules()
            state_end = gpu_state()
            trial = summarize_window(
                level, events, spans, (start, end), energy_end - energy_start, idle_watts
            )
            trial["max_temperature_C"] = max(state_start[0], state_end[0])
            trial["min_sm_clock_MHz"] = min(state_start[1], state_end[1])
            trials.append(trial)
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await asyncio.sleep(2.0)  # let the cancelled requests leave the engine
        row = median_of_windows(level, trials)
        row["trials"] = trials
        return row

    async def poisson(engine, load: float, seed: int, arrivals: list, idle_watts: float) -> dict:
        """Send every request at its arrival time; record each one and the run's energy.

        Args:
            engine: The running engine.
            load: The target number of requests in flight.
            seed: The seed the arrivals were drawn with.
            arrivals: [(seconds after the start, request index)].
            idle_watts: The GPU's idle power with the model loaded.

        Returns:
            The run: its energy, duration, mean in flight, and one record per request.
        """
        events: list = []
        launched = time.perf_counter()

        async def arrive(offset: float, index: int) -> dict:
            """Wait for the request's arrival time, send it and return its record."""
            await asyncio.sleep(max(0.0, launched + offset - time.perf_counter()))
            sent, done, prompt_tokens, generated = await stream(engine, index, events)
            return {
                "index": index,
                "arrival_s": sent - launched,
                "finish_s": done - launched,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated,
            }

        state_start, energy_start = gpu_state(), energy_joules()
        records = await asyncio.gather(*(arrive(offset, index) for offset, index in arrivals))
        seconds, joules = time.perf_counter() - launched, energy_joules() - energy_start
        state_end = gpu_state()
        spans = [(record["arrival_s"], record["finish_s"]) for record in records]
        return {
            "target_load": load,
            "seed": seed,
            "seconds": seconds,
            "joules": joules,
            "joules_net": joules - idle_watts * seconds,
            "mean_in_flight": round(mean_in_flight(spans, 0.0, seconds), 3),
            "max_temperature_C": max(state_start[0], state_end[0]),
            "min_sm_clock_MHz": min(state_start[1], state_end[1]),
            "requests": records,
        }

    async def run() -> dict:
        """Load the engine, then measure idle power, every level and every load.

        Returns:
            {"idle_power_W", "fixed", "poisson"}.
        """
        engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model_name,
                quantization=quantization,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_mem_util,
                max_num_seqs=max_slots,
                # ON by default in vLLM; off, as in the static sweep, so every prompt pays
                # its prefill.
                enable_prefix_caching=False,
                seed=0,
            )
        )
        await stream(engine, 0, [])  # warm-up, discarded

        # --- Idle power, with the model resident ---
        start, energy_start = time.perf_counter(), energy_joules()
        await asyncio.sleep(10.0)
        idle_watts = (energy_joules() - energy_start) / (time.perf_counter() - start)
        print(f"idle GPU power: {idle_watts:.1f} W", flush=True)

        # --- Fixed concurrency ---
        rows = []
        for level in concurrency:
            row = await fixed_concurrency(engine, level, idle_watts)
            rows.append(row)
            print(
                f"  N={level:>3}: in flight {row['mean_in_flight']:.2f}, "
                f"{row['tokens_per_s']:.0f} tok/s, {row['J_per_generated_token']:.4f} J/out-tok "
                f"[{min(row['trials_J_per_generated_token']):.4f}-"
                f"{max(row['trials_J_per_generated_token']):.4f}], {row['mean_power_W']:.0f} W, "
                f"{row['max_temperature_C']:.0f} C",
                flush=True,
            )

        # --- Poisson loads ---
        runs = []
        for load, seed, arrivals in poisson_loads:
            result = await poisson(engine, load, seed, arrivals, idle_watts)
            runs.append(result)
            print(
                f"  load {load:g} seed {seed}: {len(arrivals)} requests, in flight "
                f"{result['mean_in_flight']:.2f}, {result['joules']:.0f} J over "
                f"{result['seconds']:.0f} s",
                flush=True,
            )

        if hasattr(engine, "shutdown"):
            engine.shutdown()
        return {"idle_power_W": round(idle_watts, 2), "fixed": rows, "poisson": runs}

    report = asyncio.run(run())
    gpu_uuid = text(pynvml.nvmlDeviceGetUUID(handle))
    driver = text(pynvml.nvmlSystemGetDriverVersion())
    power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
    pynvml.nvmlShutdown()
    return {
        "model": model_name,
        "gpu": text(gpu_name),
        "gpu_uuid": gpu_uuid,
        "driver_version": driver,
        "power_limit_W": power_limit,
        "vllm_version": VLLM_VERSION,
        "precision": quantization or "bf16 (vLLM default for this checkpoint)",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_mem_util,
        "max_num_seqs": max_slots,
        "prefix_caching": "disabled",
        "boundary": "chip_or_module (NVML per-GPU counter; excludes host CPU/DRAM and datacenter PUE)",
        "energy_source": "nvmlDeviceGetTotalEnergyConsumption (cumulative mJ counter)",
        "method": (
            "vLLM async engine, continuous batching; the OLT's GSM8K questions at their "
            "recorded answer lengths (ignore_eos); fixed concurrency measured over a "
            "steady-state window after a warm-up; Poisson loads from first arrival to last "
            "finish"
        ),
        "warmup_s": warmup_s,
        "window_s": window_s,
        "repeats": repeats,
        **report,
    }


@app.local_entrypoint()
def main():
    """Build the requests from the OLT's recorded answers, measure, and write the report.

    The OLT's own answers rather than synthetic text, because answer length decides how
    long a request stays in the batch.
    """
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1]))  # implementation/src
    # imported here: this file also runs inside the Modal container, which has no src/
    # pylint: disable=import-outside-toplevel
    from energy.three_tier import answer_sources
    from measure.gsm8k import build_prompt

    # --- The OLT's questions and how long it answered them ---
    requests = []
    for path, tiers in answer_sources():
        if "olt" not in tiers:
            continue
        with open(path, encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                if record["tier"] == "olt":
                    requests.append(
                        {
                            "prompt": build_prompt(record["question"]),
                            "generated_tokens": record["tokens_gen"],
                        }
                    )
    poisson_loads = [
        (load, seed, poisson_arrivals(load, POISSON_S, len(requests), seed))
        for index, load in enumerate(LOADS)
        for seed in range(SEED + 100 * index, SEED + 100 * index + REPEATS)
    ]
    print(
        f"{MODEL_NAME} on {GPU_SPEC}, {QUANTIZATION or 'bf16'}: {len(requests)} requests, "
        f"concurrency {CONCURRENCY} x {REPEATS} windows, Poisson loads {LOADS} x {REPEATS} "
        f"seeds over {POISSON_S:g} s" + (" (smoke)" if SMOKE else "")
    )

    report = measure.remote(
        requests,
        CONCURRENCY,
        WARMUP_S,
        WINDOW_S,
        REPEATS,
        poisson_loads,
        MODEL_NAME,
        QUANTIZATION,
        MAX_MODEL_LEN,
        GPU_MEM_UTIL,
        MAX_SLOTS,
    )
    report["seed"] = SEED

    slug = MODEL_NAME.split("/")[-1].lower()
    gpu_slug = GPU_SPEC.replace(":", "x").lower()
    # Timestamped, so a rerun never overwrites an earlier result.
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    kind = "continuous_smoke" if SMOKE else "continuous"
    out = (
        here.parents[2]
        / "results"
        / "measurements"
        / f"gpu_energy_{kind}_{slug}_{gpu_slug}_{stamp}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"\nwrote {out}")
    print(f"idle {report['idle_power_W']} W on {report['gpu']}\n")
    print(
        f"{'N':>4} {'in flight':>10} {'tok/s':>7} {'J/out-tok':>10} {'spread':>15} "
        f"{'net J/out-tok':>14} {'W':>5} {'max C':>6}"
    )
    for row in report["fixed"]:
        spread = row["trials_J_per_generated_token"]
        print(
            f"{row['concurrency']:>4} {row['mean_in_flight']:>10.2f} {row['tokens_per_s']:>7.0f} "
            f"{row['J_per_generated_token']:>10.4f} {min(spread):>7.4f}-{max(spread):<7.4f} "
            f"{row['J_per_generated_token_net']:>14.4f} {row['mean_power_W']:>5.0f} "
            f"{row['max_temperature_C']:>6.0f}"
        )
    print(
        f"\n{'load':>5} {'seed':>5} {'requests':>9} {'in flight':>10} {'seconds':>8} {'joules':>9}"
    )
    for run in report["poisson"]:
        print(
            f"{run['target_load']:>5g} {run['seed']:>5} {len(run['requests']):>9} "
            f"{run['mean_in_flight']:>10.2f} {run['seconds']:>8.0f} {run['joules']:>9.0f}"
        )
