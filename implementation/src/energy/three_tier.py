"""The energy model of the three-tier case study: what each tier costs per query.

ROLE IN THE PIPELINE
    measure/  ->  energy/three_tier.py  ->  simulate/, analyze/

    Every script that prices energy reads it through this module, so the
    simulator and the tables always use the same numbers.

WHAT IT PROVIDES
    published_rates()   J per prompt token and per generated token for the user
                        (phone) and ONU (Hailo-10H NPU) tiers, from the literature
                        (config/energy_sources.yaml).
    published_speeds()  their tokens per second, for latency only.
    olt_runs(), olt_reference()
                        the first-party OLT sweeps (Qwen2.5-7B fp8 on an L4):
                        energy per token at batch sizes 1..64; the reference one
                        prices every OLT figure, the others replicate it.
    OltCurve            the OLT's measured rates as a function of batch size:
                        average (a query's share of the batch) and marginal
                        (what one more query adds).
    boundary(), olt_factor()
                        the multiplier that turns the OLT's GPU-card figure into
                        a whole-system one (host, idle machines, building PUE).
    load_answers()      every tier's recorded answer to every GSM8K test question
                        (correctness, confidence, token counts).

INPUTS
    config/energy_sources.yaml
    results/measurements/gpu_energy_qwen2.5-7b-instruct_l4x1_run{1,2}.json
    results/measurements/gsm8k_zeroshot_*_n1319_*.raw.jsonl

See energy_tests.md §3-§7 for where every number comes from.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]          # implementation/
RESULTS = ROOT / "results" / "measurements"         # first-party measurements and answers (inputs)
STUDY = ROOT / "results" / "study"                  # the case study's simulation runs (outputs)
TIERS = ("user", "onu", "olt")


def _sources() -> dict:
    """The literature figures and boundary factors, config/energy_sources.yaml."""
    with open(ROOT / "config" / "energy_sources.yaml") as f:
        return yaml.safe_load(f)


def published_rates(accounting: str = "average") -> dict:
    """J per prompt token (pf) and per generated token (dec) for the tiers not measured here.

    User: Cai et al., phone CPU, prefill and decode priced separately.
    ONU: Cloud to Edge, Raspberry Pi 5 + Hailo-10H NPU -- one ALL-IN figure per generated
    token (whole board at its plug, idle and its short prompt's prefill
    included), so its prefill rate is 0.

    accounting: 'average' uses the figures as published. 'marginal' charges only
    what a query adds: the ONU board is on whether or not a query arrives, so
    its idle draw comes off (idle_W x seconds per token). The phone's SoC figure
    is kept under both.

    Returns {"user": {"pf": ..., "dec": ...}, "onu": {"pf": ..., "dec": ...}}.
    """
    src = _sources()
    user, onu = src["user"], src["onu"]
    dec = float(onu["J_per_generated_token_all_in"])
    if accounting == "marginal":
        dec -= float(onu["idle_W"]) / float(onu["tokens_per_s"])
    return {
        "user": {"pf": float(user["prefill_J_per_prompt_token"]), "dec": float(user["decode_J_per_generated_token"])},
        "onu": {"pf": 0.0, "dec": dec},
    }


def published_speeds() -> dict:
    """Tokens per second of the phone and the ONU, from the same sources as their energy.

    Used only to estimate how long a query takes (latency), never for energy.
    Phone: prefill and decode speeds (Cai et al., Table 5). ONU: one speed, generated
    tokens over the whole inference time, prompt included (Cloud to Edge, Table 3),
    so it has no separate prefill speed.

    Returns {"user": {"pf": tok/s or None, "dec": tok/s}, "onu": {...}}.
    """
    src = _sources()
    return {"user": {"pf": float(src["user"]["prefill_tokens_per_s"]), "dec": float(src["user"]["decode_tokens_per_s"])},
            "onu": {"pf": None, "dec": float(src["onu"]["tokens_per_s"])}}


def boundary() -> dict:
    """Factors that convert the OLT's GPU-card figure to the whole-system boundary (§7).

    Google's per-prompt shares give (accelerators + host + idle) / accelerators
    for the server; the site's PUE then adds the building. The shares already
    include Google's own overhead at PUE 1.09, so that share is left out and the
    chosen site PUE applied instead.

    Returns the server factor (with and without the idle-machines share) and the
    two PUE values: isp_site (1.54, the industry average, primary) and pue_low
    (1.09, Google's fleet, a best case).
    """
    y = _sources()["olt_boundary"]
    sh = y["shares"]
    it_over_accel = (sh["active_accelerators"] + sh["host_cpu_and_dram"]
                     + sh["idle_machines"]) / sh["active_accelerators"]
    return {"it_over_accel": it_over_accel,
            "it_over_accel_marginal": (sh["active_accelerators"] + sh["host_cpu_and_dram"])
                                      / sh["active_accelerators"],
            "pue_isp": float(y["pue"]["isp_site"]), "pue_low": float(y["pue"]["lower_bound"])}


def olt_factor(which: str = "system", accounting: str = "average") -> float:
    """Multiplier on the OLT's GPU-card figure.

    which: 'system' (whole server at PUE 1.54, the primary boundary: x 2.47),
    'system-low' (PUE 1.09: x 1.75) or 'gpu' (the card alone: x 1).
    Under 'marginal' accounting the idle machines held for load spikes are not
    something a query adds, so their share is left out (x 2.20 instead of x 2.47).
    """
    b = boundary()
    it = b["it_over_accel_marginal"] if accounting == "marginal" else b["it_over_accel"]
    return {"system": it * b["pue_isp"], "system-low": it * b["pue_low"], "gpu": 1.0}[which]


# The OLT sweeps (measure/measure_gpu_energy.py), in the order they were run. Run 1's
# prefill figures are unreliable (the energy counter's resolution, energy_tests.md
# §3.4); its decode figures replicate the others.
OLT_RUN_FILES = {
    "run1": "gpu_energy_qwen2.5-7b-instruct_l4x1_run1.json",
    "run2": "gpu_energy_qwen2.5-7b-instruct_l4x1_run2.json",
    "run3": "gpu_energy_qwen2.5-7b-instruct_l4x1_run3.json",
}
OLT_REFERENCE = "run3"          # the run every energy figure is priced on: 5 repeats, 12 batch sizes


def olt_runs() -> dict:
    """Every OLT sweep, by name: {"run1": report, "run2": report, ...}."""
    out = {}
    for name, file in OLT_RUN_FILES.items():
        with open(RESULTS / file) as f:
            out[name] = json.load(f)
    return out


def olt_reference() -> dict:
    """The OLT sweep every energy figure is priced on (OLT_REFERENCE)."""
    return olt_runs()[OLT_REFERENCE]


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys on xs."""
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


class OltCurve:
    """The OLT's measured energy rates as a function of batch size.

    Built from one sweep's batch_curve rows (batch 1, 2, 4, ..., 64), each rate
    multiplied by `factor` (olt_factor: the boundary conversion). Between
    measured batches rates are interpolated log-log; above 64 they hold at 64's,
    which errs against the OLT.
    """

    def __init__(self, rows: list[dict], factor: float = 1.0):
        """rows: the sweep's batch_curve; factor: boundary multiplier on every energy figure."""
        self.b = [r["batch"] for r in rows]
        self.pf = [r["prefill_J_per_input_token"] * factor for r in rows]
        self.dec = [r["decode_J_per_output_token"] * factor for r in rows]
        self.tps = [r["tokens_per_s"] for r in rows]
        self.pf1_net = rows[0]["prefill_J_per_input_token_net"] * factor
        self.dec1_net = rows[0]["decode_J_per_output_token_net"] * factor
        self.pf_slope = _slope(self.b, [b * y for b, y in zip(self.b, self.pf)])
        self.dec_slope = _slope(self.b, [b * y for b, y in zip(self.b, self.dec)])

    def marginal_rates(self, batch: float) -> tuple[float, float]:
        """Energy one more query adds, per token, when it makes the batch this size.

        An OLT with nothing in service idles anyway, so the first query adds only
        the energy above idle: the net rates at batch 1. A later query joins steps
        that run anyway; the card sits at its power limit, so it adds only the
        step time it causes. That is the slope of the batch's total energy per
        step, b x rate(b), against b, fitted by least squares over the measured
        batches (energy_tests.md §8.3).
        """
        if batch <= 1:
            return self.pf1_net, self.dec1_net
        return self.pf_slope, self.dec_slope

    def _at(self, ys: list[float], batch: float) -> float:
        """ys interpolated log-log at this batch size, clamped to the measured range."""
        b = min(max(batch, self.b[0]), self.b[-1])
        for x0, x1, y0, y1 in zip(self.b, self.b[1:], ys, ys[1:]):
            if b <= x1:
                f = (math.log(b) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
        return ys[-1]

    def rates(self, batch: float) -> tuple[float, float]:
        """Average accounting: (J per prompt token, J per generated token) at this batch size."""
        return self._at(self.pf, batch), self._at(self.dec, batch)

    def service_s(self, batch: float, gen_tokens: float) -> float:
        """Seconds one sequence takes to generate gen_tokens inside a batch of this size."""
        return gen_tokens / (self._at(self.tps, batch) / min(max(batch, 1), self.b[-1]))


# The answers the thesis uses, pinned by name so a later collection cannot silently
# replace them. The ONU's answers were re-collected at Q4_K_M, the precision its energy
# source states, so they come from their own file; phone and OLT from the 3-tier one.
ANSWER_FILES = {
    "gsm8k_zeroshot_user-onu-olt_n1319_20260911T023919Z.raw.jsonl": {"user", "olt"},
    "gsm8k_zeroshot_onu_n1319_20260911T031326Z.raw.jsonl": {"onu"},
}


def answer_sources() -> list[tuple[str, set]]:
    """Which answers file each tier's answers come from: [(path, {tiers to take from it}), ...]."""
    return [(str(RESULTS / name), tiers) for name, tiers in ANSWER_FILES.items()]


def load_answers() -> tuple[dict, dict, list[str]]:
    """Every tier's answers, one compact record per question, in question order.

    Each record: index, correct, cmean (exp mean token logprob: RecServe's
    confidence), cmin (exp min token logprob), tp / tg (prompt and generated
    tokens), qb / ab (question and answer bytes), trunc (hit the 512-token cap).

    Returns (records by tier, model name by tier, source file names).
    """
    recs: dict = {t: [] for t in TIERS}
    models: dict = {}
    sources = answer_sources()
    for path, keep in sources:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["tier"] not in keep:
                    continue
                lps = [x for x in r["logprobs"] if x is not None]
                models[r["tier"]] = r["model"]
                recs[r["tier"]].append({
                    "index": r["index"],
                    "correct": bool(r["correct"]),
                    "cmean": r["confidence"],
                    "cmin": math.exp(min(lps)) if lps else 0.0,
                    "tp": r["tokens_prompt"],
                    "tg": r["tokens_gen"],
                    "qb": len(r["question"].encode()),      # bytes on the wire: RecServe's |x| and |y|
                    "ab": len(r["generated_text"].encode()),
                    "trunc": r.get("finish_reason") == "length",
                })
    for t in TIERS:
        recs[t].sort(key=lambda r: r["index"])
    return recs, models, [Path(p).name for p, _ in sources]
