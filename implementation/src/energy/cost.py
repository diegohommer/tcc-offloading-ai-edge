"""Derivation formulas from the TCC master document, section 5.

Primitives in (E_pf, E_dec, tokens_prompt, tokens_gen, PUE, C_grid) out
(J/query, J/token, gCO2/query, latency/query, cascade cost). This module is
generic over any cascade trace shaped like a list of per-layer visits -- it
does not know or care whether those visits came from a generative LLM cascade
(the thesis's actual target) or from RecServe's shipped classification
cascade (today's smoke test). The caller is responsible for supplying
energy primitives that are actually valid for what ran.
"""
from __future__ import annotations

from dataclasses import dataclass

J_PER_KWH = 3.6e6


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


def gCO2_per_query(query_energy_j: float, pue: float, c_grid_g_per_kwh: float) -> float:
    """gCO2/query (EcoThink style): J/query * PUE * C_grid / 3.6e6."""
    return query_energy_j * pue * c_grid_g_per_kwh / J_PER_KWH


def latency_s(tokens_prompt: int, tokens_gen: int, nu_pf_tok_s: float | None, nu_dec_tok_s: float | None) -> float | None:
    """Latency/query = |T_prompt| / nu_pf + |T_gen| / nu_dec.

    Returns None if either throughput needed for a nonzero token count is
    missing, rather than silently skipping a term.
    """
    total = 0.0
    if tokens_prompt > 0:
        if not nu_pf_tok_s:
            return None
        total += tokens_prompt / nu_pf_tok_s
    if tokens_gen > 0:
        if not nu_dec_tok_s:
            return None
        total += tokens_gen / nu_dec_tok_s
    return total


def cascade_cost_J(visits: list[LayerVisit], num_link_hops: int, link_energy_per_hop_J: float) -> dict:
    """Sum of per-layer J/query across every layer the cascade actually
    visited, plus link energy for every hop between them (section 5,
    "custo da cascata"). Returns a breakdown, not just a total, so a report
    can show which layer dominated.
    """
    per_layer = [(v.layer, query_energy_J(v)) for v in visits]
    compute_total = sum(e for _, e in per_layer)
    link_total = num_link_hops * link_energy_per_hop_J
    return {
        "per_layer_J": per_layer,
        "compute_J": compute_total,
        "link_J": link_total,
        "total_J": compute_total + link_total,
    }
