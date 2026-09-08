"""Cloud tier of the generative cascade, served from Modal as an OpenAI-compatible endpoint.

WHY THIS EXISTS
---------------
run_generative_matrix.py can collect user/onu/fog locally under Ollama, but the
cloud tier needs a 70B-class model that will not run on this machine (QwQ-32B
measured at 1.2 tok/s locally -- see the README's "next steps"). That single
missing tier blocks the two results the thesis actually turns on:

  1. the cascade-vs-direct-to-cloud comparison (is hierarchical inference even
     cheaper than sending everything to a batched cloud?), and
  2. the destination-selection mechanism, which needs to know what the cloud
     would have answered for queries that stopped lower down.

WHY MODAL RATHER THAN A HOSTED API
-----------------------------------
A hosted provider (Together, Fireworks) returns tokens and logprobs but tells us
nothing about energy -- it is someone else's GPU, at an unknown batch size, on
unknown hardware. Renting the GPU directly means the SAME hardware can be metered
with NVML (see measure_gpu_energy.py), so the cloud tier stops being a borrowed
literature number with a node-vs-chip boundary caveat and becomes a first-party
measurement on declared hardware, exactly as Intel RAPL did for the local tiers.

MODEL CHOICE: Qwen2.5-72B-Instruct
-----------------------------------
Three reasons, in order of importance:

  1. config/layer_energy.yaml ALREADY tabulates this exact model on the cloud
     tier (1.044 J/token, 4xH100, optimized batch, Caravaca et al.). The ladder
     in run_generative_matrix.py is built on the principle that confidence and
     energy must refer to the same model rather than to unrelated ones; picking
     Qwen2.5-72B keeps that principle intact for the cloud tier too.
  2. It is not gated on the Hugging Face Hub. Llama-3.3-70B is, which means a
     license click and a token before anything runs.
  3. Qwen2.5 is strong at math (the reason section 13.4 nominates qwen2.5:14b
     for the broken fog tier), so the cloud tier will not repeat the SOLAR-10.7B
     inversion that made the fog hop destroy 44 correct answers.

BF16 at 72B is ~145 GB of weights, so it needs 2xH100 (160 GB) with
tensor-parallel-size 2. That also matches the precision the tabulated energy
point was (presumably) measured at -- the source does not state quantization,
so BF16 is the least-assumption choice.

USAGE
-----
    # 1. deploy (stays warm for scaledown_window after the last request)
    modal deploy src/modal_apps/cloud_tier_server.py

    # 2. point the collector at the printed URL -- no harness change needed,
    #    run_generative_matrix.py already takes a --config override
    python src/scripts/run_generative_matrix.py --limit 200 --tiers cloud \
        --config config/tier_cloud_modal.json

    # 3. stop paying
    modal app stop tcc-cloud-tier

COST NOTE
---------
2xH100 bills at ~$7.90/hour while a container is up. scaledown_window is set
deliberately short (3 min) so an idle server stops billing quickly; the tradeoff
is that a gap longer than that in the collection run pays the model-load cost
again. The HF cache volume makes that reload a load, not a re-download.
"""

import modal

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
GPU_TYPE = "H100"
N_GPU = 2                      # 72B BF16 ~= 145 GB of weights; 2x80 GB fits with room for KV cache
VLLM_VERSION = "0.21.0"

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    # pip_install rather than uv_pip_install: both exist in the pinned client, but
    # pip_install is stable across every Modal version this may be re-run under.
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Weights are cached across runs. Without this, every cold start re-downloads
# ~145 GB, which costs more in GPU-minutes than the inference itself.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("tcc-cloud-tier")


@app.server(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    # First boot downloads ~145 GB and builds CUDA graphs; 30 min is generous on
    # purpose, since a startup timeout mid-download wastes the whole download.
    startup_timeout=30 * 60,
    # Idle containers bill. 3 minutes is short enough to bound an accidental
    # overnight charge and long enough to survive gaps in the collection loop.
    scaledown_window=3 * 60,
    port=8000,
    # The endpoint is unauthenticated but the URL is unguessable, and it serves
    # an open-weights model over a public dataset -- nothing sensitive transits
    # it. Set unauthenticated=False and pass a Modal proxy token if that changes.
    unauthenticated=True,
)
class CloudTier:
    @modal.enter()
    def start(self):
        import subprocess

        self.process = subprocess.Popen([
            "vllm", "serve", MODEL_NAME,
            "--tensor-parallel-size", str(N_GPU),
            "--host", "0.0.0.0",
            "--port", "8000",
            # OllamaGenerativeLayer sends {"logprobs": true} and reads
            # choices[].logprobs.content[].logprob -- the chosen token's logprob
            # only, no top-k. vLLM serves that shape natively; the confidence
            # definition (exp of mean logprob) therefore needs no client change.
            "--max-logprobs", "1",
            # Long CoT answers plus the prompt; max_tokens is capped client-side
            # at 512, so this only needs to leave room for both.
            "--max-model-len", "4096",
        ])

    @modal.exit()
    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=60)
