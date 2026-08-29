"""Thin instrumentation layer around the vendored RecServe cascade.

RecServe (src/recserve/vendor) makes its offloading decision by escalating
through tiers based on a beta-quantile confidence threshold, but it does
not expose *how* a given query was routed: it only returns the final
(label, confidence) and mutates internal history state, and its shipped
code only implements 3 tiers (end/edge/cloud).

This module re-implements RecursiveServe's routing loop with a 4th tier
inserted, so the tier names and count match the reference architecture
(Pakpahan and Hwang, IEEE Access vol. 14, 2026): user -> onu -> fog ->
cloud, corresponding to Customer/Edge(ONU)/Fog(OLT)/Cloud in that paper's
Fig. 1. These are also exactly the layer keys used by
config/layer_energy.yaml, so a hop's tier name doubles as its energy-table
lookup key -- no separate tier-to-layer mapping is needed (contrast with
the 3-tier version, which had to proxy RecServe's "cloud" tier onto the
"fog" energy layer for lack of a 4th tier).

For the energy-costing pipeline we need, per query, the full path taken
(which layers ran, how many prompt tokens each one saw, how long each hop
took). This module re-implements RecursiveServe's routing loop against
four pipelines while recording that path, instead of monkeypatching or
forking the vendored code.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, pipeline

VENDORED_RECSERVE = Path(__file__).resolve().parent / "vendor"
if str(VENDORED_RECSERVE) not in sys.path:
    sys.path.insert(0, str(VENDORED_RECSERVE))

from utils import clean_text  # noqa: E402  (vendored RecServe module)

# Tier order mirrors the 4-tier reference architecture: user -> onu -> fog -> cloud.
NEXT_TIER = {"user": "onu", "onu": "fog", "fog": "cloud"}
MAX_INPUT_TOKENS = 512  # matches the pipeline's truncation=True, max_length=512


@dataclass
class Hop:
    layer: str
    model_name: str
    tokens_prompt: int
    confidence: float
    predicted_label: str
    latency_s: float
    escalated: bool
    escalation_reason: str | None = None


@dataclass
class QueryTrace:
    input_text: str
    hops: list[Hop] = field(default_factory=list)
    final_label: str | None = None
    final_confidence: float | None = None

    @property
    def final_layer(self) -> str | None:
        return self.hops[-1].layer if self.hops else None


class TracedRecursiveServe:
    """Re-implements RecServe's beta-quantile escalation with per-query tracing.

    device=-1 forces CPU inference (this environment has no GPU); pass
    device=0 if run somewhere with CUDA available.
    """

    def __init__(
        self,
        user_model_name: str,
        onu_model_name: str,
        fog_model_name: str,
        cloud_model_name: str,
        beta: float = 0.3,
        max_history_size: int = 10000,
        device: int = -1,
    ):
        self.beta = beta
        self.max_history_size = max_history_size
        self.model_names = {
            "user": user_model_name,
            "onu": onu_model_name,
            "fog": fog_model_name,
            "cloud": cloud_model_name,
        }
        self.pipelines = {
            tier: pipeline("sentiment-analysis", model=name, device=device, top_k=1)
            for tier, name in self.model_names.items()
        }
        self.tokenizers = {tier: AutoTokenizer.from_pretrained(name) for tier, name in self.model_names.items()}
        self.confidence_history: dict[str, list[float]] = {tier: [] for tier in self.model_names}

    def _count_tokens(self, tier: str, text: str) -> int:
        encoded = self.tokenizers[tier](text, truncation=True, max_length=MAX_INPUT_TOKENS)
        return len(encoded["input_ids"])

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        upper = raw_label.upper()
        if upper in ("LABEL_1", "POSITIVE"):
            return "POSITIVE"
        if upper in ("LABEL_0", "NEGATIVE"):
            return "NEGATIVE"
        return raw_label

    def _classify_at(self, tier: str, text: str) -> tuple[str, float, float]:
        pipe = self.pipelines[tier]
        start = time.perf_counter()
        prediction = pipe(text, truncation=True, max_length=MAX_INPUT_TOKENS, top_k=1)
        latency_s = time.perf_counter() - start
        predicted_label = self._normalize_label(prediction[0]["label"])
        confidence = prediction[0]["score"]
        return predicted_label, confidence, latency_s

    def classify_one(self, tier: str, text: str) -> tuple[str, float, float]:
        """Run a single tier on already-cleaned text. Public entry point for
        benchmarks that drive one tier at a time (src/scripts/measure_tier_energy.py)."""
        return self._classify_at(tier, text)

    def count_tokens(self, tier: str, text: str) -> int:
        """Prompt-token count under `tier`'s own tokenizer."""
        return self._count_tokens(tier, text)

    def classify_all_tiers(self, input_text: str) -> dict[str, dict]:
        """Run *every* tier on `input_text`, ignoring escalation entirely.

        Used to build the offline policy matrix the lambda sweep replays
        against (src/scripts/run_policy_matrix.py). The escalation decision
        is path-dependent -- changing lambda changes which tiers a query
        visits, which changes each tier's confidence history, which changes
        its future thresholds -- so a sweep cannot be post-processed from a
        single cascade trace: it needs every tier's answer for every query.

        Each tier's output depends only on the input text (the pipelines are
        stateless and deterministic), so running them all up front and
        replaying policies offline is equivalent to re-running the cascade
        once per configuration, minus the repeated inference cost.
        """
        text = clean_text(input_text)
        out: dict[str, dict] = {}
        for tier in self.model_names:
            predicted_label, confidence, latency_s = self._classify_at(tier, text)
            out[tier] = {
                "model_name": self.model_names[tier],
                "predicted_label": predicted_label,
                "confidence": confidence,
                "latency_s": latency_s,
                "tokens_prompt": self._count_tokens(tier, text),
            }
        return out

    def predict(self, input_text: str) -> QueryTrace:
        text = clean_text(input_text)
        trace = QueryTrace(input_text=text)
        tier = "user"

        while True:
            predicted_label, confidence, latency_s = self._classify_at(tier, text)
            tokens_prompt = self._count_tokens(tier, text)

            history = self.confidence_history[tier]
            escalate = False
            reason = None
            next_tier = NEXT_TIER.get(tier)
            if next_tier is not None and len(history) > 1:
                beta_threshold = float(np.percentile(history, self.beta * 100))
                if confidence < beta_threshold:
                    escalate = True
                    reason = f"confidence {confidence:.4f} < beta_threshold {beta_threshold:.4f}"

            history.append(confidence)
            if len(history) > self.max_history_size:
                history.pop(0)

            trace.hops.append(
                Hop(
                    layer=tier,
                    model_name=self.model_names[tier],
                    tokens_prompt=tokens_prompt,
                    confidence=confidence,
                    predicted_label=predicted_label,
                    latency_s=latency_s,
                    escalated=escalate,
                    escalation_reason=reason,
                )
            )

            if escalate and next_tier is not None:
                tier = next_tier
                continue

            trace.final_label = predicted_label
            trace.final_confidence = confidence
            return trace
