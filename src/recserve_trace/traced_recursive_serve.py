"""Thin instrumentation layer around the vendored RecServe cascade.

RecServe (third_party/recserve) makes its offloading decision by escalating
through end -> edge -> cloud tiers based on a beta-quantile confidence
threshold, but it does not expose *how* a given query was routed: it only
returns the final (label, confidence) and mutates internal history state.

For the energy-costing pipeline we need, per query, the full path taken
(which layers ran, how many prompt tokens each one saw, how long each hop
took). This module re-implements RecursiveServe's routing loop against the
same three pipelines while recording that path, instead of monkeypatching
or forking the vendored code.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, pipeline

THIRD_PARTY_RECSERVE = Path(__file__).resolve().parents[2] / "third_party" / "recserve"
if str(THIRD_PARTY_RECSERVE) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_RECSERVE))

from utils import clean_text  # noqa: E402  (vendored RecServe module)

# Tier order mirrors RecServe's escalation chain: end -> edge -> cloud.
NEXT_TIER = {"end": "edge", "edge": "cloud"}
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
        end_model_name: str,
        edge_model_name: str,
        cloud_model_name: str,
        beta: float = 0.3,
        max_history_size: int = 10000,
        device: int = -1,
    ):
        self.beta = beta
        self.max_history_size = max_history_size
        self.model_names = {"end": end_model_name, "edge": edge_model_name, "cloud": cloud_model_name}
        self.pipelines = {
            tier: pipeline("sentiment-analysis", model=name, device=device, top_k=1)
            for tier, name in self.model_names.items()
        }
        self.tokenizers = {tier: AutoTokenizer.from_pretrained(name) for tier, name in self.model_names.items()}
        self.confidence_history: dict[str, list[float]] = {"end": [], "edge": [], "cloud": []}

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

    def predict(self, input_text: str) -> QueryTrace:
        text = clean_text(input_text)
        trace = QueryTrace(input_text=text)
        tier = "end"

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
