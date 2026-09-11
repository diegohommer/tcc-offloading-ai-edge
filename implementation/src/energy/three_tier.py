"""The three-tier case study's data, read once for every script that uses it.

User and ONU energy come from the energy table (published sources), the OLT's
from the first-party L4 sweep, and every tier's answers from the GSM8K
collection (energy_tests.md §3-§7). The results page and the piggyback
simulator both read through here, so they always price the same numbers.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]          # implementation/
RESULTS = ROOT / "results" / "energy_tests"
TIERS = ("user", "onu", "olt")


def _table() -> dict:
    return yaml.safe_load(open(ROOT / "config" / "layer_energy.yaml"))


def published_rates() -> dict:
    """J per prompt token (pf) and per generated token (dec) for the tiers not measured here.

    User: Cai et al., Snapdragon CPU, prefill and decode priced separately.
    ONU: Cloud to Edge, Jetson Orin Nano Super -- one ALL-IN figure per generated
    token (whole board at its plug, idle and its short prompt's prefill
    included), so its prefill rate is 0.
    """
    nano = _table()["layers"]["onu"]["orin_nano_super"]["models"]["qwen2.5_1.5b"]
    return {
        "user": {"pf": 0.016, "dec": 0.074},
        "onu": {"pf": 0.0, "dec": float(nano["J_per_generated_token_all_in"]["q4_k_m"])},
    }


def boundary() -> dict:
    """Conversion of the OLT's GPU-card figure to the whole-system boundary (§7).

    Google's per-prompt shares give (accelerators + host + idle) / accelerators;
    a site PUE then adds the building. The shares already include Google's own
    overhead at PUE 1.09, so the overhead share is left out and re-applied at
    the chosen site PUE instead.
    """
    y = _table()["boundary_consolidation"]
    sh = y["comprehensive_shares"]
    it_over_accel = (sh["active_accelerators"] + sh["host_cpu_and_dram"]
                     + sh["idle_machines"]) / sh["active_accelerators"]
    return {"it_over_accel": it_over_accel,
            "pue_isp": float(y["pue"]["isp_site"]), "pue_low": float(y["pue"]["lower_bound"])}


def olt_factor(which: str = "system") -> float:
    """Multiplier on the GPU-card figure: 'system' (PUE 1.54), 'system-low' (1.09) or 'gpu' (1)."""
    b = boundary()
    return {"system": b["it_over_accel"] * b["pue_isp"],
            "system-low": b["it_over_accel"] * b["pue_low"], "gpu": 1.0}[which]


def olt_runs() -> tuple[dict, dict]:
    """The two L4 sweeps; run 2 is the reference (prefill fixed), run 1 its replication."""
    return (json.load(open(RESULTS / "gpu_energy_qwen2.5-7b-instruct_l4x1_run1.json")),
            json.load(open(RESULTS / "gpu_energy_qwen2.5-7b-instruct_l4x1_run2.json")))


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys on xs."""
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


class OltCurve:
    """The OLT's measured rates as a function of batch size, log-log interpolated.

    Batches above the largest measured (64) hold at 64's rates, which errs
    against the OLT.
    """

    def __init__(self, rows: list[dict], factor: float = 1.0):
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
        b = min(max(batch, self.b[0]), self.b[-1])
        for x0, x1, y0, y1 in zip(self.b, self.b[1:], ys, ys[1:]):
            if b <= x1:
                f = (math.log(b) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
        return ys[-1]

    def rates(self, batch: float) -> tuple[float, float]:
        return self._at(self.pf, batch), self._at(self.dec, batch)

    def service_s(self, batch: float, gen_tokens: float) -> float:
        """Seconds one sequence takes to generate gen_tokens inside a batch of this size."""
        return gen_tokens / (self._at(self.tps, batch) / min(max(batch, 1), self.b[-1]))


def answer_sources() -> list[tuple[str, set]]:
    """Which file each tier's answers come from.

    The ONU was re-collected at Q4_K_M when its hardware became the Orin Nano
    Super; take it from that file when present, user and OLT from the 3-tier one.
    """
    three = sorted(glob.glob(str(RESULTS / "gsm8k_zeroshot_user-onu-olt_n1319_*.raw.jsonl")))[-1]
    onu_only = sorted(glob.glob(str(RESULTS / "gsm8k_zeroshot_onu_n1319_*.raw.jsonl")))
    sources = [(three, {"user", "olt"} if onu_only else set(TIERS))]
    if onu_only:
        sources.append((onu_only[-1], {"onu"}))
    return sources


def load_answers() -> tuple[dict, dict, list[str]]:
    """Every tier's answers, one compact record per query, in question order.

    Returns (records by tier, model by tier, source file names).
    """
    recs: dict = {t: [] for t in TIERS}
    models: dict = {}
    sources = answer_sources()
    for path, keep in sources:
        for line in open(path):
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
                "trunc": r.get("finish_reason") == "length",
            })
    for t in TIERS:
        recs[t].sort(key=lambda r: r["index"])
    return recs, models, [Path(p).name for p, _ in sources]
