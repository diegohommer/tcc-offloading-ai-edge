"""Per-query energy formulas from the TCC master document, section 5.

Primitives in (E_pf, E_dec, tokens_prompt, tokens_gen), out J/query and
J/token. This module is generic over any cascade trace shaped like a list of
per-layer visits -- it does not know or care whether those visits came from a
generative LLM cascade (the thesis's actual target) or from RecServe's shipped
classification cascade (today's smoke test). The caller is responsible for
supplying energy primitives that are actually valid for what ran.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerVisit:
    layer: str
    tokens_prompt: int
    tokens_gen: int
    E_dec_J_per_token: float
    E_pf_J_per_token: float | None = None  # None means "not tabulated" -- see layer_energy.yaml notes
    latency_s: float | None = None  # measured wall-clock latency for this hop, if available


def query_energy_J(visit: LayerVisit) -> float:
    """J/query for a single layer visit: E_pf * |T_prompt| + E_dec * |T_gen|.

    If E_pf is not tabulated for this layer (true for user/onu today -- only
    a qualitative "1-2 orders of magnitude cheaper than decode" note exists,
    no hard number), the prefill term is dropped and the result under-counts
    energy. Callers must surface that, not silently treat the result as complete.
    """
    decode_term = visit.E_dec_J_per_token * visit.tokens_gen
    prefill_term = (visit.E_pf_J_per_token or 0.0) * visit.tokens_prompt
    return prefill_term + decode_term


def aggregate_J_per_token(query_energy_j: float, tokens_prompt: int, tokens_gen: int) -> float:
    """J/token (green-paper style), section 5."""
    total_tokens = tokens_prompt + tokens_gen
    if total_tokens == 0:
        return 0.0
    return query_energy_j / total_tokens
