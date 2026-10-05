"""The energy model of the three-tier case study: what each tier costs per query.

Every script that prices energy reads it through this module, so the simulator and the
tables always use the same numbers. Where each number comes from: energy_tests.md §3-§7.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
"""The implementation/ folder."""

RESULTS = ROOT / "results" / "measurements"
"""First-party measurements and recorded answers (inputs)."""

STUDY = ROOT / "results" / "study"
"""The case study's simulation runs (outputs)."""

TIERS = ("user", "onu", "olt")
"""The tiers, from the bottom up."""


# ==========================================
# Phone and ONU (published figures)
# ==========================================
def _sources() -> dict:
    """Return the literature figures and boundary factors (config/energy_sources.yaml)."""
    with open(ROOT / "config" / "energy_sources.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def published_rates(accounting: str = "average") -> dict:
    """Return J per prompt token (pf) and per generated token (dec) for the phone and the ONU.

    The phone (Cai et al.) prices prefill and decode separately. The ONU (Cloud to Edge)
    has one all-in figure per generated token, its short prompt's prefill included, so its
    prefill rate is 0. Under marginal accounting the ONU board's idle draw comes off, since
    the board is on whether or not a query arrives; the phone's figure is kept under both.

    Args:
        accounting: "average" (the figures as published) or "marginal".

    Returns:
        {"user": {"pf": ..., "dec": ...}, "onu": {"pf": ..., "dec": ...}}.
    """
    src = _sources()
    user, onu = src["user"], src["onu"]
    dec = float(onu["J_per_generated_token_all_in"])
    if accounting == "marginal":
        dec -= float(onu["idle_W"]) / float(onu["tokens_per_s"])
    return {
        "user": {
            "pf": float(user["prefill_J_per_prompt_token"]),
            "dec": float(user["decode_J_per_generated_token"]),
        },
        "onu": {"pf": 0.0, "dec": dec},
    }


def published_speeds() -> dict:
    """Return the phone's and the ONU's tokens per second, for latency only.

    The phone has prefill and decode speeds (Cai et al., Table 5); the ONU has one speed,
    generated tokens over the whole inference time (Cloud to Edge, Table 3).

    Returns:
        {"user": {"pf": tok/s, "dec": tok/s}, "onu": {"pf": None, "dec": tok/s}}.
    """
    src = _sources()
    return {
        "user": {
            "pf": float(src["user"]["prefill_tokens_per_s"]),
            "dec": float(src["user"]["decode_tokens_per_s"]),
        },
        "onu": {"pf": None, "dec": float(src["onu"]["tokens_per_s"])},
    }


# ==========================================
# OLT boundary
# ==========================================
def boundary() -> dict:
    """Return the factors that convert the OLT's GPU-card figure to the whole system.

    Google's per-prompt shares give (accelerators + host + idle) / accelerators for the
    server; the site's PUE then adds the building. Google's own overhead share is left
    out, and the chosen site's PUE applied instead.

    Returns:
        The server factor with and without the idle-machines share, and the two PUE
        values: pue_isp (1.54, the industry average) and pue_low (1.09, Google's fleet).
    """
    y = _sources()["olt_boundary"]
    sh = y["shares"]
    it_over_accel = (
        sh["active_accelerators"] + sh["host_cpu_and_dram"] + sh["idle_machines"]
    ) / sh["active_accelerators"]
    return {
        "it_over_accel": it_over_accel,
        "it_over_accel_marginal": (sh["active_accelerators"] + sh["host_cpu_and_dram"])
        / sh["active_accelerators"],
        "pue_isp": float(y["pue"]["isp_site"]),
        "pue_low": float(y["pue"]["lower_bound"]),
    }


def olt_factor(which: str = "system", accounting: str = "average") -> float:
    """Return the multiplier on the OLT's GPU-card figure.

    Under marginal accounting the idle machines held for load spikes are not something a
    query adds, so their share is left out (x 2.20 instead of x 2.47).

    Args:
        which: "system" (whole server at PUE 1.54: x 2.47), "system-low" (PUE 1.09:
            x 1.75) or "gpu" (the card alone: x 1).
        accounting: "average" or "marginal".
    """
    b = boundary()
    it = b["it_over_accel_marginal"] if accounting == "marginal" else b["it_over_accel"]
    return {"system": it * b["pue_isp"], "system-low": it * b["pue_low"], "gpu": 1.0}[which]


# ==========================================
# OLT measurements
# ==========================================
# The OLT sweeps (measure/measure_gpu_energy_static.py), in the order they were run. Run 1's
# prefill figures are unreliable (the energy counter's resolution, energy_tests.md §3.4);
# its decode figures replicate the others.
OLT_RUN_FILES = {
    "run1": "gpu_energy_qwen2.5-7b-instruct_l4x1_run1.json",
    "run2": "gpu_energy_qwen2.5-7b-instruct_l4x1_run2.json",
    "run3": "gpu_energy_qwen2.5-7b-instruct_l4x1_run3.json",
}
OLT_REFERENCE = "run3"
"""The sweep every energy figure is priced on: 5 repeats, 12 batch sizes."""


def olt_runs() -> dict:
    """Return every OLT sweep, by name: {"run1": report, "run2": report, ...}."""
    out = {}
    for name, file in OLT_RUN_FILES.items():
        with open(RESULTS / file, encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def olt_reference() -> dict:
    """Return the OLT sweep every energy figure is priced on (OLT_REFERENCE)."""
    return olt_runs()[OLT_REFERENCE]


def _slope(xs: list[float], ys: list[float]) -> float:
    """Return the least-squares slope of ys on xs."""
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


class OltCurve:
    """The OLT's measured energy rates as a function of batch size.

    Between measured batches rates are interpolated log-log; above 64 they hold at 64's,
    which errs against the OLT.
    """

    # ==========================================
    # Initialization
    # ==========================================
    def __init__(self, rows: list[dict], factor: float = 1.0):
        """Build the curve from one sweep.

        Args:
            rows: The sweep's batch_curve (batch 1, 2, 4, ..., 64).
            factor: Boundary multiplier on every energy figure (olt_factor).
        """
        self.b = [r["batch"] for r in rows]
        self.pf = [r["prefill_J_per_input_token"] * factor for r in rows]
        self.dec = [r["decode_J_per_output_token"] * factor for r in rows]
        self.tps = [r["tokens_per_s"] for r in rows]
        self.pf_net = [r["prefill_J_per_input_token_net"] * factor for r in rows]
        self.dec_net = [r["decode_J_per_output_token_net"] * factor for r in rows]
        self.pf1_net, self.dec1_net = self.pf_net[0], self.dec_net[0]
        self.pf_slope = _slope(self.b, [b * y for b, y in zip(self.b, self.pf)])
        self.dec_slope = _slope(self.b, [b * y for b, y in zip(self.b, self.dec)])

    # ==========================================
    # Rates and times
    # ==========================================
    def marginal_rates(self, batch: float) -> tuple[float, float]:
        """Return each sequence's share, per token, of what a batch this size draws above idle.

        Read off the measured net-of-idle curve, so a batch of n draws n times this per
        step, as the GPU did (energy_tests.md §3.5, §8.4).

        Args:
            batch: Sequences in the batch.
        """
        return self._at(self.pf_net, batch), self._at(self.dec_net, batch)

    def _at(self, ys: list[float], batch: float) -> float:
        """Return ys interpolated log-log at this batch size, clamped to the measured range."""
        b = min(max(batch, self.b[0]), self.b[-1])
        for x0, x1, y0, y1 in zip(self.b, self.b[1:], ys, ys[1:]):
            if b <= x1:
                f = (math.log(b) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
        return ys[-1]

    def rates(self, batch: float) -> tuple[float, float]:
        """Return average accounting's (J per prompt token, J per generated token) at this batch size."""
        return self._at(self.pf, batch), self._at(self.dec, batch)

    def service_s(self, batch: float, gen_tokens: float) -> float:
        """Return the seconds one sequence takes to generate gen_tokens inside a batch of this size."""
        return gen_tokens / (self._at(self.tps, batch) / min(max(batch, 1), self.b[-1]))


# ==========================================
# Recorded answers
# ==========================================
# The answers the thesis uses, pinned by name so a later collection cannot silently replace
# them. The ONU's answers were re-collected at Q4_K_M, the precision its energy source
# states, so they come from their own file; phone and OLT from the three-tier one.
ANSWER_FILES = {
    "gsm8k_zeroshot_user-onu-olt_n1319_20260911T023919Z.raw.jsonl": {"user", "olt"},
    "gsm8k_zeroshot_onu_n1319_20260911T031326Z.raw.jsonl": {"onu"},
}


def answer_sources() -> list[tuple[str, set]]:
    """Return which file each tier's answers come from: [(path, {tiers to take from it}), ...]."""
    return [(str(RESULTS / name), tiers) for name, tiers in ANSWER_FILES.items()]


def load_answers() -> tuple[dict, dict, list[str]]:
    """Load every tier's answers, one compact record per question, in question order.

    Each record holds: index, correct, cmean (exp mean token logprob: RecServe's
    confidence), cmin (exp min token logprob), tp / tg (prompt and generated tokens),
    qb / ab (question and answer bytes) and trunc (hit the 512-token cap).

    Returns:
        (records by tier, model name by tier, source file names).
    """
    recs: dict = {t: [] for t in TIERS}
    models: dict = {}
    sources = answer_sources()
    for path, keep in sources:
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["tier"] not in keep:
                    continue
                lps = [x for x in r["logprobs"] if x is not None]
                models[r["tier"]] = r["model"]
                recs[r["tier"]].append(
                    {
                        "index": r["index"],
                        "correct": bool(r["correct"]),
                        "cmean": r["confidence"],
                        "cmin": math.exp(min(lps)) if lps else 0.0,
                        "tp": r["tokens_prompt"],
                        "tg": r["tokens_gen"],
                        "qb": len(r["question"].encode()),  # bytes on the wire: RecServe's |x|, |y|
                        "ab": len(r["generated_text"].encode()),
                        "trunc": r.get("finish_reason") == "length",
                    }
                )
    for t in TIERS:
        recs[t].sort(key=lambda r: r["index"])
    return recs, models, [Path(p).name for p, _ in sources]
