"""Scaffold for the deferred generative track (section 12 of the master doc).

RecServe's shipped code (src/recserve/vendor) only implements a Seq2Class
cascade: single forward pass, no output-token generation. The thesis's real
experiment needs a Seq2Seq cascade -- four decoder LLMs (user/onu/fog/cloud
scale), a perplexity- or confidence-based escalation rule reimplementing
RecServe's beta-quantile logic for generation instead of classification, and
per-layer prefill/decode token traces feeding src/energy/cost.py.

That build-out is intentionally NOT started yet: model choice (local
quantized via llama.cpp/GGUF vs. hosted API for fog/cloud, per section
12.2), precision per layer, and prompt/output token sourcing are still open
decisions. This module only fixes the interface future layers should
implement, so TracedRecursiveServe-equivalent code and src/energy/cost.py
don't need to change shape when that work starts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationResult:
    text: str
    tokens_prompt: int
    tokens_gen: int
    confidence: float  # e.g. normalized perplexity, per RecServe's Seq2Seq confidence definition
    prefill_latency_s: float
    decode_latency_s: float


class GenerativeLayer(Protocol):
    """One tier of a generative cascade (user, onu, fog, or cloud)."""

    layer_name: str

    def generate(self, prompt: str) -> GenerationResult:
        """Run prefill+decode for `prompt` and return counts needed to price
        the hop via src/energy/cost.py (tokens_prompt, tokens_gen) plus the
        confidence signal the escalation rule will threshold on."""
        raise NotImplementedError(
            f"{type(self).__name__} is a scaffold for the deferred generative track "
            "(see module docstring) -- no model backend is wired up yet."
        )
