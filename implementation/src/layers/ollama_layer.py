"""Generative cascade tier backed by Ollama or any OpenAI-compatible endpoint.

Implements the GenerativeLayer protocol in generative_layer.py, which was left
as an interface stub. One class covers both backends because the cascade needs
the same signal from tiers that run in very different places:

  - small tiers (1B / 8B / 10.7B) run locally under Ollama
  - the 70B tier runs on a hosted OpenAI-compatible endpoint, or on a cluster
    GPU large enough to hold it

That split is sound because **confidence is hardware-independent**: the logprobs
a model emits depend only on its weights and the prompt, not on where it runs.
(Energy is the opposite -- it is only meaningful on representative hardware, and
is never sourced from here.)

CONFIDENCE DEFINITION
---------------------
RecServe uses peak softmax probability for Seq2Class and *normalized perplexity*
for Seq2Seq. This module implements the Seq2Seq case as

    confidence = exp(mean over generated tokens of logprob)
               = 1 / perplexity

which is the geometric mean per-token probability. Two reasons for this form
rather than raw perplexity:

  1. Range and direction match Seq2Class. It lands in (0, 1] with higher =
     more confident, exactly like peak softmax -- so the escalation rule,
     the sliding-window history, and the beta-quantile threshold in
     traced_recursive_serve.py all work unchanged. Raw perplexity is in
     [1, inf) with *lower* = more confident, which would invert every
     comparison in the mechanism.
  2. It is length-normalized by construction, so a long chain of thought is
     not automatically judged less confident than a short one.

Requires a backend that returns per-token logprobs. Ollama supports this from
v0.12.11 (`logprobs: true`); note Ollama Cloud returns null for the field, so
the local server or an OpenAI-compatible provider is required.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import requests

from layers.generative_layer import GenerationResult

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def confidence_from_logprobs(logprobs: list[float]) -> float:
    """Geometric mean token probability = 1/perplexity, in (0, 1].

    Empty generations return 0.0 -- a model that produced nothing is maximally
    unconfident, and this keeps the value inside the same range as every other
    confidence rather than raising or returning NaN into the history window.
    """
    if not logprobs:
        return 0.0
    return math.exp(sum(logprobs) / len(logprobs))


@dataclass
class OllamaGenerativeLayer:
    """One tier of the generative cascade.

    layer_name must be one of user/onu/fog/cloud -- the same keys used by
    config/layer_energy.yaml, so a hop's tier doubles as its energy lookup key
    (the same convention the classification harness already follows).
    """

    layer_name: str
    model: str
    base_url: str = DEFAULT_OLLAMA_URL
    api_key: str | None = None          # set for hosted OpenAI-compatible endpoints
    openai_compatible: bool = False     # True -> /v1/chat/completions, else Ollama native
    max_tokens: int = 512
    temperature: float = 0.0            # deterministic: confidence must be reproducible
    timeout_s: float = 300.0
    num_thread: int | None = None       # None -> Ollama decide; menor = menos calor

    def generate(self, prompt: str, stop: list[str] | None = None) -> GenerationResult:
        """Generate for `prompt`, optionally halting at any string in `stop`.

        `stop` exists for replaying a benchmark harness's own prompts, where the
        harness truncates the continuation at a fixed marker. It must be applied
        at generation time rather than by trimming the text afterwards: the
        confidence is computed over the returned logprob vector, so tokens
        emitted past the stop would otherwise contaminate it -- and under
        exp(min logprob) a single bad trailing token dominates the score.
        """
        start = time.perf_counter()
        if self.openai_compatible:
            text, logprobs, tokens, n_prompt, n_gen = self._call_openai_compatible(prompt, stop)
        else:
            text, logprobs, tokens, n_prompt, n_gen = self._call_ollama_native(prompt, stop)
        elapsed = time.perf_counter() - start

        return GenerationResult(
            text=text,
            tokens_prompt=n_prompt,
            tokens_gen=n_gen,
            confidence=confidence_from_logprobs(logprobs),
            logprobs=logprobs,
            tokens=tokens,
            # Neither backend reports a prefill/decode split, so the whole wall
            # time is attributed to decode. Decode dominates >=99% of time in
            # this regime (per the Jetson Orin measurements in layer_energy.yaml),
            # so the distortion is small -- but it is an attribution choice, not
            # a measurement, and must not be cited as a phase-separated latency.
            prefill_latency_s=0.0,
            decode_latency_s=elapsed,
        )

    def _call_ollama_native(self, prompt: str, stop: list[str] | None = None) -> tuple[str, list[float], list[str], int, int]:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "logprobs": True,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens,
                            **({"stop": stop} if stop else {}),
                            **({"num_thread": self.num_thread} if self.num_thread else {})},
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()

        raw = payload.get("logprobs")
        if not raw:
            raise RuntimeError(
                f"{self.model}: no logprobs returned. Ollama needs >= v0.12.11, and "
                "Ollama Cloud returns null for this field -- use a local server or an "
                "OpenAI-compatible provider."
            )
        kept = [entry for entry in raw if entry.get("logprob") is not None]
        logprobs = [entry["logprob"] for entry in kept]
        tokens = [entry.get("token", "") for entry in kept]
        return (
            payload.get("response", ""),
            logprobs,
            tokens,
            int(payload.get("prompt_eval_count", 0)),
            int(payload.get("eval_count", len(logprobs))),
        )

    def _call_openai_compatible(self, prompt: str, stop: list[str] | None = None) -> tuple[str, list[float], list[str], int, int]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "logprobs": True,
                **({"stop": stop} if stop else {}),
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]

        content = choice["message"]["content"] or ""
        entries = ((choice.get("logprobs") or {}).get("content")) or []
        kept = [e for e in entries if e.get("logprob") is not None]
        logprobs = [e["logprob"] for e in kept]
        tokens = [e.get("token", "") for e in kept]
        if not logprobs:
            raise RuntimeError(
                f"{self.model}: endpoint returned no logprobs. Verify the provider "
                "exposes the `logprobs` field -- support varies."
            )

        usage = payload.get("usage") or {}
        return (
            content,
            logprobs,
            tokens,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", len(logprobs))),
        )
