"""Measure the OLT's static sweep and continuous batching back to back, on one GPU.

measure_gpu_energy_static.py and measure_gpu_energy_continuous.py each rent their own
container, so they can land on different cards (their idle powers once differed by 8 W).
This runs both in a single container, one after the other, each in its own process so the
second starts with the GPU's memory free, and records the card's UUID before and after.
The reports go to the Modal volume tcc-measurements as each phase ends, so the launching
machine can sleep or close while it runs.

Usage:
    modal deploy src/measure/measure_gpu_energy_one_card.py
    python src/measure/measure_gpu_energy_one_card.py launch            # ~3 h on one L4
    SMOKE=1 python src/measure/measure_gpu_energy_one_card.py launch    # a short check (~15 min)
    python src/measure/measure_gpu_energy_one_card.py collect <stamp>   # once it has finished
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

# ==========================================
# Settings
# ==========================================
# Read on the launching machine and passed to one_card() as arguments, as in the two scripts
# it runs (see measure_gpu_energy_static.py, Settings).
SMOKE = os.environ.get("SMOKE") == "1"
MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
QUANTIZATION = os.environ.get("QUANTIZATION", "fp8") or None
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
STATIC_BATCHES = [1, 8] if SMOKE else [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
"""The batch sizes of the reference sweep (run3)."""
STATIC_REPEATS = 1 if SMOKE else 5
STATIC_OUT_TOKENS = 300
STATIC_PROMPTS = 64
"""GSM8K test problems the static batches are drawn from: the first `batch` of them."""

APP_NAME = "tcc-olt-one-card"
GPU_SPEC = "L4:1"
VLLM_VERSION = "0.21.0"
HERE = Path(__file__).resolve()
SCRIPTS = ("measure_gpu_energy_static.py", "measure_gpu_energy_continuous.py")
REMOTE_DIR = "/root/olt"
"""Where the two measurement scripts are placed inside the container."""
RESULTS_DIR = "/results"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "nvidia-ml-py")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
for script in SCRIPTS:
    vllm_image = vllm_image.add_local_file(HERE.parent / script, f"{REMOTE_DIR}/{script}")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("tcc-measurements", create_if_missing=True)

app = modal.App(APP_NAME)

PHASES = (
    ("static", "measure_gpu_energy_static", "run_sweep"),
    ("continuous", "measure_gpu_energy_continuous", "run_measure"),
)
"""(name, module, function) of each phase, in the order they run on the card."""

RUN_PHASE = (
    "import json, sys\n"
    f"sys.path.insert(0, {REMOTE_DIR!r})\n"
    "module, function, arguments, out = sys.argv[1:5]\n"
    "measure = getattr(__import__(module), function)\n"
    "report = measure(**json.load(open(arguments)))\n"
    "open(out, 'w').write(json.dumps(report, indent=1) + '\\n')\n"
)
"""What each phase's process runs: one measurement function, its report written to a file."""


# ==========================================
# The measurement, on one rented GPU
# ==========================================
@app.function(
    image=vllm_image,
    gpu=GPU_SPEC,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        RESULTS_DIR: results_vol,
    },
    timeout=6 * 60 * 60,
    # Modal has no non-preemptible GPUs: a preempted container restarts the run from the top
    # on another card, so collect() checks that every report names the same GPU.
)
def one_card(arguments: dict, stamp: str) -> dict:
    """Run every phase on this container's GPU and save each report to the volume.

    Args:
        arguments: {phase name: the keyword arguments of its measurement function}.
        stamp: The run's folder on the volume.

    Returns:
        The card's identity before and after the phases.
    """
    # pylint: disable=import-outside-toplevel,import-error
    import pynvml

    def card() -> dict:
        """Return the identity of the container's GPU, read from NVML."""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        text = lambda value: value.decode() if isinstance(value, bytes) else value  # noqa: E731
        identity = {
            "gpu": text(pynvml.nvmlDeviceGetName(handle)),
            "gpu_uuid": text(pynvml.nvmlDeviceGetUUID(handle)),
            "pci_bus_id": text(pynvml.nvmlDeviceGetPciInfo(handle).busId),
            "driver_version": text(pynvml.nvmlSystemGetDriverVersion()),
            "power_limit_W": pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0,
            "read_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        pynvml.nvmlShutdown()
        return identity

    folder = Path(RESULTS_DIR) / stamp
    folder.mkdir(parents=True, exist_ok=True)
    before = card()
    print(f"card before: {before}", flush=True)

    for phase, module, function in PHASES:
        argument_file = Path("/tmp") / f"{phase}_arguments.json"
        argument_file.write_text(json.dumps(arguments[phase]), encoding="utf-8")
        print(f"--- {phase}: {module}.{function} ---", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-c",
                RUN_PHASE,
                module,
                function,
                str(argument_file),
                str(folder / f"{phase}.json"),
            ],
            check=True,
        )
        results_vol.commit()

    after = card()
    identity = {"stamp": stamp, "before": before, "after": after}
    identity["same_card"] = before["gpu_uuid"] == after["gpu_uuid"]
    (folder / "card.json").write_text(json.dumps(identity, indent=1) + "\n", encoding="utf-8")
    results_vol.commit()
    print(f"card after: {after}; same card: {identity['same_card']}", flush=True)
    return identity


# ==========================================
# Launch and collect, on the local machine
# ==========================================
def phase_arguments() -> dict:
    """Return each phase's keyword arguments, built from GSM8K and the OLT's recorded answers."""
    sys.path.insert(0, str(HERE.parent))  # the two measurement scripts
    sys.path.insert(0, str(HERE.parents[1]))  # implementation/src
    # pylint: disable=import-outside-toplevel
    import measure_gpu_energy_continuous as continuous
    from measure.gsm8k import build_prompt, load_gsm8k

    prompts = [build_prompt(item.question) for item in load_gsm8k("test", limit=STATIC_PROMPTS)]
    requests = continuous.olt_requests()
    return {
        "static": {
            "prompts": prompts,
            "batches": STATIC_BATCHES,
            "out_tokens": STATIC_OUT_TOKENS,
            "repeats": STATIC_REPEATS,
            "model_name": MODEL_NAME,
            "quantization": QUANTIZATION,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_mem_util": GPU_MEM_UTIL,
        },
        "continuous": {
            "requests": requests,
            "concurrency": continuous.CONCURRENCY,
            "warmup_s": continuous.WARMUP_S,
            "window_s": continuous.WINDOW_S,
            "repeats": continuous.REPEATS,
            "poisson_loads": continuous.poisson_plan(len(requests)),
            "model_name": MODEL_NAME,
            "quantization": QUANTIZATION,
            "max_model_len": MAX_MODEL_LEN,
            "gpu_mem_util": GPU_MEM_UTIL,
            "max_slots": continuous.MAX_SLOTS,
        },
    }


def launch() -> None:
    """Start one_card on the deployed app and return at once; the run goes on without us."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ("_smoke" if SMOKE else "")
    arguments = phase_arguments()
    call = modal.Function.from_name(APP_NAME, "one_card").spawn(arguments, stamp)
    print(f"launched {APP_NAME}.one_card, stamp {stamp}, call {call.object_id}")
    print(f"collect with: python {HERE.relative_to(Path.cwd())} collect {stamp}")


def collect(stamp: str) -> None:
    """Download a finished run from the volume into results/measurements/.

    The static report keeps the static script's naming and the continuous one the continuous
    script's, each tagged onecard_<stamp>, and both carry the card's identity.

    Args:
        stamp: The run's folder on the volume, as launch() printed it.
    """
    volume = modal.Volume.from_name("tcc-measurements")
    read = lambda name: json.loads(b"".join(volume.read_file(f"{stamp}/{name}")))  # noqa: E731
    identity = read("card.json")
    reports = {phase: read(f"{phase}.json") for phase, _, _ in PHASES}
    # A preempted, restarted run could leave one phase from an earlier card behind.
    cards = {identity["before"]["gpu_uuid"], identity["after"]["gpu_uuid"]}
    cards |= {report["gpu_uuid"] for report in reports.values()}
    if len(cards) != 1:
        sys.exit(f"not one card: {sorted(cards)}; launch the run again")
    slug = MODEL_NAME.split("/")[-1].lower()
    gpu_slug = GPU_SPEC.replace(":", "x").lower()
    out = HERE.parents[2] / "results" / "measurements"
    for phase, name in (
        ("static", f"gpu_energy_{slug}_{gpu_slug}_onecard_{stamp}.json"),
        ("continuous", f"gpu_energy_continuous_{slug}_{gpu_slug}_onecard_{stamp}.json"),
    ):
        reports[phase]["one_card"] = identity
        (out / name).write_text(json.dumps(reports[phase], indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out / name}")
    print(f"one card for every phase: {cards.pop()}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "launch":
        launch()
    elif len(sys.argv) == 3 and sys.argv[1] == "collect":
        collect(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
