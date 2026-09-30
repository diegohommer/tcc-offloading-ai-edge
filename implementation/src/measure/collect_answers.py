"""Collect every tier's answer to every GSM8K test question, on Modal.

Every tier answers every question once, with every token's logprob stored, so the simulator
can replay any routing, and recompute any confidence, without running a model again.
Writes results/measurements/gsm8k_zeroshot_<tiers>_n<N>_<UTC>.raw.jsonl.

Usage:
    modal run src/measure/collect_answers.py                    # all 1,319 questions, all tiers
    LIMIT=20 TIERS=olt modal run src/measure/collect_answers.py # smoke test
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import modal

# ==========================================
# Tiers
# ==========================================
# A tier's answers and its energy figure must describe the same model at the same precision:
# quantization changes the weights, so the answers and the confidence the cascade routes on.
#   user  Llama-3.2-1B-Instruct, GGUF Q4_K_M: the same engine family and bit width as its
#         energy source (Cai et al., llama.cpp 4-bit on a phone CPU).
#   onu   Qwen2.5-1.5B-Instruct, GGUF Q4_K_M: the precision Cloud to Edge states for the
#         Hailo-10H board. That board runs its own compiled 4-bit build, so these answers
#         approximate the ONU's.
#   olt   Qwen2.5-7B-Instruct at fp8: exactly the configuration of the L4 energy sweep.
# All three are ungated on Hugging Face (the Llama GGUF is a re-upload, with the tokenizer
# from an ungated mirror; both remain under Meta's Llama 3.2 Community License). Where the
# answers are generated does not matter, only the model and its precision, so all three run
# on Modal L4s in parallel.
TIER_SPECS = {
    "user": {
        "model": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "gguf_file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "tokenizer": "unsloth/Llama-3.2-1B-Instruct",
        "quantization": None,
        "precision": "GGUF Q4_K_M (llama.cpp 4-bit)",
    },
    "onu": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "gguf_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
        "quantization": None,
        "precision": "GGUF Q4_K_M (llama.cpp 4-bit)",
    },
    "olt": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "quantization": "fp8",
        "precision": "fp8",
    },
}

TIERS = [t.strip() for t in os.environ.get("TIERS", "user,onu,olt").split(",") if t.strip()]
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = the whole test split
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
GPU_SPEC = os.environ.get("GPU", "L4:1")

# Identical to measure_gpu_energy.py's image, so Modal reuses the cached build.
VLLM_VERSION = "0.21.0"
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "nvidia-ml-py")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("tcc-collect-answers")


@app.function(
    image=vllm_image,
    gpu=GPU_SPEC,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=2 * 60 * 60,
)
def collect(tier: str, spec: dict, prompts: list[str], max_tokens: int, max_model_len: int) -> dict:
    """Answer every prompt with one tier's model, on a Modal GPU.

    Configuration travels as arguments only (see measure_gpu_energy.py): module-level
    settings are re-read inside the container and would not carry the caller's values.

    Args:
        tier: The tier's name.
        spec: The tier's model, GGUF file, tokenizer and quantization (TIER_SPECS).
        prompts: The GSM8K prompts, in question order.
        max_tokens: The generation limit per answer.
        max_model_len: vLLM's context length.

    Returns:
        {"tier", "rows": one dict per answer, "elapsed_s", "gpu", "vllm_version"}.
    """
    # These packages exist only inside the Modal image, not on the machine that launches it.
    # pylint: disable=import-outside-toplevel,import-error
    import pynvml
    from vllm import LLM, SamplingParams

    model = spec["model"]
    if spec.get("gguf_file"):
        from huggingface_hub import hf_hub_download

        model = hf_hub_download(spec["model"], spec["gguf_file"])

    llm = LLM(
        model=model,
        tokenizer=spec.get("tokenizer"),
        quantization=spec.get("quantization"),
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        seed=0,
    )
    # logprobs=1 returns the sampled token's logprob at every step.
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=1)
    conversations = [[{"role": "user", "content": p}] for p in prompts]

    # --- Generate ---
    t0 = time.perf_counter()
    outputs = llm.chat(conversations, params)
    elapsed = time.perf_counter() - t0

    # --- Keep every answer with its per-token logprobs ---
    tokenizer = llm.get_tokenizer()
    rows = []
    for o in outputs:
        c = o.outputs[0]
        logprobs = [
            step[tid].logprob if tid in step else None
            for step, tid in zip(c.logprobs or [], c.token_ids)
        ]
        rows.append(
            {
                "generated_text": c.text,
                "tokens": [tokenizer.decode([t]) for t in c.token_ids],
                "logprobs": logprobs,
                "tokens_prompt": len(o.prompt_token_ids),
                "tokens_gen": len(c.token_ids),
                "finish_reason": c.finish_reason,
            }
        )

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
    gpu = gpu.decode() if isinstance(gpu, bytes) else gpu
    pynvml.nvmlShutdown()
    return {
        "tier": tier,
        "rows": rows,
        "elapsed_s": elapsed,
        "gpu": gpu,
        "vllm_version": VLLM_VERSION,
    }


def confidence(logprobs: list) -> float:
    """Return RecServe's generative confidence: exp(mean token logprob) = 1 / perplexity.

    Args:
        logprobs: The generated tokens' logprobs (None where missing).
    """
    lps = [x for x in logprobs if x is not None]
    return math.exp(sum(lps) / len(lps)) if lps else 0.0


@app.local_entrypoint()
def main():
    """Build the GSM8K prompts locally, collect every tier's answers on Modal, score and write them."""
    # --- Prompts ---
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parents[1]))  # implementation/src
    # imported here: this file also runs inside the Modal container, which has no src/
    from measure.gsm8k import (  # pylint: disable=import-outside-toplevel
        build_prompt,
        is_correct,
        load_gsm8k,
    )

    unknown = [t for t in TIERS if t not in TIER_SPECS]
    if unknown:
        raise SystemExit(f"unknown tier(s) {unknown}; choose from {list(TIER_SPECS)}")

    items = load_gsm8k("test", limit=LIMIT or None)
    prompts = [build_prompt(i.question) for i in items]
    print(f"{len(items)} GSM8K test items, tiers {TIERS}, max_tokens {MAX_TOKENS}, on {GPU_SPEC}")

    # --- Collect, one container per tier, in parallel ---
    calls = {t: collect.spawn(t, TIER_SPECS[t], prompts, MAX_TOKENS, MAX_MODEL_LEN) for t in TIERS}

    # --- Score and write (timestamped, so a rerun never overwrites an earlier one) ---
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = (
        here.parents[2]
        / "results"
        / "measurements"
        / f"gsm8k_zeroshot_{'-'.join(TIERS)}_n{len(items)}_{stamp}.raw.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with open(out, "w", encoding="utf-8") as f:
        for tier, call in calls.items():
            res = call.get()
            spec = TIER_SPECS[tier]
            n_ok = 0
            for idx, (item, r) in enumerate(zip(items, res["rows"])):
                ok = is_correct(r["generated_text"], item.reference_answer)
                n_ok += ok
                f.write(
                    json.dumps(
                        {
                            "dataset": "gsm8k",
                            "split": "test",
                            "index": idx,
                            "tier": tier,
                            "model": spec["model"]
                            + (f"/{spec['gguf_file']}" if spec.get("gguf_file") else ""),
                            "precision": spec["precision"],
                            "hardware": f"modal_{res['gpu'].replace(' ', '_')}",
                            "engine": f"vllm {res['vllm_version']}",
                            "prompt_source": "zero_shot_build_prompt_chat_template",
                            "question": item.question,
                            "reference_answer": item.reference_answer,
                            "difficulty_steps": item.difficulty_steps,
                            "generated_text": r["generated_text"],
                            "logprobs": r["logprobs"],
                            "tokens": r["tokens"],
                            "confidence": confidence(r["logprobs"]),
                            "correct": ok,
                            "tokens_prompt": r["tokens_prompt"],
                            "tokens_gen": r["tokens_gen"],
                            "finish_reason": r["finish_reason"],
                            "latency_s": None,  # batched collection: per-query latency is not meaningful
                        }
                    )
                    + "\n"
                )
            n = len(res["rows"])
            gen = [r["tokens_gen"] for r in res["rows"]]
            summary[tier] = {
                "accuracy": n_ok / n,
                "mean_tokens_prompt": sum(r["tokens_prompt"] for r in res["rows"]) / n,
                "mean_tokens_gen": sum(gen) / n,
                "truncated": sum(r["finish_reason"] == "length" for r in res["rows"]),
                "mean_confidence": sum(confidence(r["logprobs"]) for r in res["rows"]) / n,
                "seconds": res["elapsed_s"],
            }

    # --- Summary ---
    print(f"\nwrote {out}\n")
    print(
        f"{'tier':>5} {'accuracy':>9} {'conf':>7} {'prompt tok':>11} {'gen tok':>8} "
        f"{'truncated':>10} {'secs':>6}"
    )
    for tier, s in summary.items():
        print(
            f"{tier:>5} {s['accuracy']:>9.3f} {s['mean_confidence']:>7.4f} {s['mean_tokens_prompt']:>11.1f} "
            f"{s['mean_tokens_gen']:>8.1f} {s['truncated']:>10} {s['seconds']:>6.0f}"
        )
    order = [t for t in ("user", "onu", "olt") if t in summary]
    accs = [summary[t]["accuracy"] for t in order]
    if len(accs) > 1:
        print(
            "\naccuracy rises tier by tier"
            if accs == sorted(accs)
            else f"\nWARNING: accuracy does NOT rise tier by tier ({dict(zip(order, accs))}) -- "
            "skipping a tier is no longer safe by dominance"
        )
