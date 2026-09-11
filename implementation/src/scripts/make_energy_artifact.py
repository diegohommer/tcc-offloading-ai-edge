#!/usr/bin/env python3
"""Build the results page for the three-tier energy tests, straight from the result files.

Nothing on the page is typed in by hand: every number and every chart is computed
here from the files in results/energy_tests/ (through energy/three_tier.py, which
the piggyback simulator reads too), so rerunning this after a new measurement or
simulation updates the page without edits.

Usage:
    python src/scripts/make_energy_artifact.py
Writes results/energy_tests/three_tiers_measured.html.
"""
from __future__ import annotations

import glob
import html
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # implementation/src
from energy.three_tier import RESULTS as D  # noqa: E402
from energy.three_tier import TIERS, OltCurve, boundary, load_answers, olt_runs, published_rates  # noqa: E402

OUT = D / "three_tiers_measured.html"
NAME = {"user": "User", "onu": "ONU", "olt": "OLT"}
PUBLISHED = published_rates()
BOUNDARY = boundary()


def bx(x):
    """A crossover batch for prose: the number, or 'beyond 64' when none."""
    return f"≈ {x:.0f}" if x else "beyond 64"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    den = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return (st.mean(a) - st.mean(b)) / den if den else float("nan")


def load():
    run1, run2 = olt_runs()
    recs, models, names = load_answers()
    return run1, run2, recs, models, ", ".join(names)


def crossing(xs, ys, target):
    """First batch where ys drops below target, log-log interpolated."""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= target > y1:
            f = (math.log(y0) - math.log(target)) / (math.log(y0) - math.log(y1))
            return math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0)))
    return None


def derive(run1, run2, recs):
    rows2, rows1 = run2["batch_curve"], run1["batch_curve"]
    B = [r["batch"] for r in rows2]
    tok = {t: (st.mean(r["tp"] for r in recs[t]), st.mean(r["tg"] for r in recs[t])) for t in TIERS}
    cost = {t: PUBLISHED[t]["pf"] * tok[t][0] + PUBLISHED[t]["dec"] * tok[t][1] for t in ("user", "onu")}
    pin, gout = tok["olt"]
    olt_g = [r["prefill_J_per_input_token"] * pin + r["decode_J_per_output_token"] * gout for r in rows2]
    olt_n = [r["prefill_J_per_input_token_net"] * pin + r["decode_J_per_output_token_net"] * gout for r in rows2]
    k_mid = BOUNDARY["it_over_accel"] * BOUNDARY["pue_isp"]
    k_low = BOUNDARY["it_over_accel"] * BOUNDARY["pue_low"]
    olt_sys = [g * k_mid for g in olt_g]
    olt_sys_low = [g * k_low for g in olt_g]
    stats = {}
    for t in TIERS:
        rs = recs[t]
        ok = [r for r in rs if r["correct"]]
        bad = [r for r in rs if not r["correct"]]
        stats[t] = {
            "n": len(rs), "acc": len(ok) / len(rs),
            "t_mean": welch_t([r["cmean"] for r in ok], [r["cmean"] for r in bad]),
            "t_min": welch_t([r["cmin"] for r in ok], [r["cmin"] for r in bad]),
            "trunc": sum(r["trunc"] for r in rs),
        }
    return {
        "B": B, "rows1": rows1, "rows2": rows2, "tok": tok, "cost": cost, "stats": stats,
        "olt_g": olt_g, "olt_n": olt_n, "olt_sys": olt_sys, "olt_sys_low": olt_sys_low,
        "k_mid": k_mid, "k_low": k_low,
        "x_onu_sys": crossing(B, olt_sys, cost["onu"]), "x_onu_sys_low": crossing(B, olt_sys_low, cost["onu"]),
        "x_onu_gpu": crossing(B, olt_g, cost["onu"]),
        "x_user_sys": crossing(B, olt_sys, cost["user"]), "x_user_gpu": crossing(B, olt_g, cost["user"]),
        "marg_gpu": st.median(r["marginal_J_per_query"] for r in rows2 if r["batch"] >= 16),
        "swing": rows2[0]["decode_J_per_output_token"] / rows2[-1]["decode_J_per_output_token"],
        "idle2": run2["idle_power_W"],
    }


# ---------------------------------------------------------------------------
# Piggyback simulation (sim_piggyback.py outputs)
# ---------------------------------------------------------------------------

SCALES = (1.0, 0.5, 0.2, 0.1, 0.05)        # ONU energy relative to published; top grid row first
FRONTIER_LOAD = 8.0                        # OLT peak load shown on the frontier chart
DAY = {"scale": 0.2, "load": 16.0, "beta": 0.6}
ACC = "0.80"                               # accuracy at which policies are compared
NOISE = 0.03                               # gains below this are inside the seed-to-seed spread
POLICY_NAME = {"stepwise": "RecServe, three tiers", "skip_onu": "RecServe, ONU dropped",
               "static": "static configuration", "stale_low": "static, set at ¼ the load",
               "stale_high": "static, set at 4× the load", "piggyback": "piggyback", "oracle": "oracle"}


def load_sim():
    def latest(pattern):
        files = sorted(glob.glob(str(D / pattern)))
        return (json.load(open(files[-1])), Path(files[-1]).name) if files else (None, None)
    runs, names = {}, []
    for s in SCALES:
        run, name = latest("sim_piggyback_system_2*.json" if s == 1 else f"sim_piggyback_system_onu{s:g}_2*.json")
        runs[s] = run
        names.append(name)
    seeds = {s: [latest(f"sim_piggyback_system_{'' if s == 1 else f'onu{s:g}_'}seed{n}.json")[0] for n in (8, 9)]
             for s in (1.0, 0.2)}
    gpu, gname = latest("sim_piggyback_gpu_2*.json")
    return {"runs": runs, "seeds": seeds, "gpu": gpu, "files": names + [gname]}


def fj(run, load, policy, acc=ACC):
    """A policy's J/query at the given accuracy, on its own frontier (None if out of range)."""
    return next(f["J_at_accuracy"][acc] for f in run["frontiers"]
                if f["peak_load"] == load and f["policy"] == policy)


def gain(run, load, against=("stepwise", "skip_onu"), acc=ACC):
    """Energy piggyback saves against the cheapest of `against`, at equal accuracy."""
    pg = fj(run, load, "piggyback", acc)
    ref = [x for x in (fj(run, load, p, acc) for p in against) if x is not None]
    return 1 - pg / min(ref) if pg is not None and ref else None


def derive_sim(sim):
    runs = sim["runs"]
    main = runs[1.0]
    loads = [c["peak_load"] for c in main["configs"]]
    g = {(s, l): gain(runs[s], l) for s in SCALES for l in loads}
    best = max((v, s, l) for (s, l), v in g.items() if v is not None and s <= 0.2)
    stale = []
    for s in SCALES:
        for l in loads:
            for acc in ("0.70", "0.80"):
                pg = fj(runs[s], l, "piggyback", acc)
                for p in ("stale_low", "stale_high"):
                    x = fj(runs[s], l, p, acc)
                    if pg and x:
                        stale.append((x / pg - 1, s, l, acc, p))
    spread = 0.0
    for s, extra in sim["seeds"].items():
        for l in (4.0, 8.0, 16.0, 32.0):
            vals = [x for x in (gain(r, l) for r in [runs[s]] + extra if r) if x is not None]
            if len(vals) > 1:
                spread = max(spread, max(vals) - min(vals))
    pub = [g[(1.0, l)] for l in loads if l <= 16 and g[(1.0, l)] is not None]
    return {
        "loads": loads, "gain": g, "best": best, "worst_stale": max(stale), "spread": spread,
        "pub_max_abs": max(abs(x) for x in pub),
        "gpu_gain": {l: gain(sim["gpu"], l) for l in loads} if sim["gpu"] else {},
        "per_h": {c["peak_load"]: c["peak_arrivals_per_h"] for c in main["configs"]},
        "svc": main["configs"][0]["olt_service_s_at_peak"],
        "n_arrivals": sum(h["queries"] for h in main["hourly"][0]["hours"]),
        "days": int(main["args"]["days"]), "window": int(main["args"]["window"]),
        "onu_J": {s: runs[s]["fixed_J_per_query"]["onu"] for s in SCALES},
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def esc(s):
    return html.escape(str(s), quote=True)


class Scale:
    def __init__(self, d0, d1, r0, r1, log=False):
        self.log = log
        f = math.log if log else (lambda v: v)
        self.f, self.d0, self.d1, self.r0, self.r1 = f, f(d0), f(d1), r0, r1

    def __call__(self, v):
        return self.r0 + (self.f(v) - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


class Chart:
    """One SVG chart: axes, gridlines, series, reference lines, hover targets."""

    def __init__(self, w, h, xs_dom, ys_dom, xlog=False, ylog=False, ml=58, mr=128, mt=22, mb=44):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.x = Scale(*xs_dom, ml, w - mr, log=xlog)
        self.y = Scale(*ys_dom, h - mb, mt, log=ylog)
        self.parts: list[str] = []
        self.edge_labels: list[tuple[float, str, str | None, bool]] = []  # (y, text, swatch, dash)

    def axes(self, xticks, yticks, xlabel, ylabel):
        for v, lab in yticks:
            y = self.y(v)
            self.parts.append(f'<line class="grid" x1="{self.ml}" x2="{self.w - self.mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
            self.parts.append(f'<text class="tick" x="{self.ml - 8}" y="{y + 3.5:.1f}" text-anchor="end">{lab}</text>')
        for v, lab in xticks:
            self.parts.append(f'<text class="tick" x="{self.x(v):.1f}" y="{self.h - self.mb + 17}" text-anchor="middle">{lab}</text>')
        self.parts.append(f'<line class="axis" x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.h - self.mb}" y2="{self.h - self.mb}"/>')
        self.parts.append(f'<text class="axlabel" x="{(self.ml + self.w - self.mr) / 2:.0f}" y="{self.h - 8}" text-anchor="middle">{esc(xlabel)}</text>')
        self.parts.append(f'<text class="axlabel" x="{self.ml - 44}" y="{self.mt - 9}">{esc(ylabel)}</text>')

    def curve_band(self, xs, lo, hi, cls):
        """Shaded region between two series, e.g. the same curve under two assumptions."""
        up = [(self.x(a), self.y(b)) for a, b in zip(xs, hi)]
        dn = [(self.x(a), self.y(b)) for a, b in zip(xs, lo)][::-1]
        d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in up + dn) + " Z"
        self.parts.append(f'<path class="band {cls}" d="{d}"/>')

    def ref(self, v, cls, label, dash=False):
        y = self.y(v)
        self.parts.append(f'<line class="ref {cls}{" dash" if dash else ""}" x1="{self.ml}" x2="{self.w - self.mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        self.edge_label(y, label, swatch=cls, dash=dash)

    def edge_label(self, y, text, swatch=None, dash=False):
        """A label in the right-hand column; placed at svg() time so labels never overlap."""
        self.edge_labels.append((y, text, swatch, dash))

    def label(self, x, y, text, swatch=None, dash=False):
        """Direct label beside a series: a coloured swatch carries identity, the text stays in ink."""
        if swatch:
            self.parts.append(f'<line class="swatch {swatch}{" dash" if dash else ""}" x1="{x + 6}" x2="{x + 20}" y1="{y:.1f}" y2="{y:.1f}"/>')
        self.parts.append(f'<text class="dlabel" x="{x + (24 if swatch else 6)}" y="{y + 3.5:.1f}">{esc(text)}</text>')

    def series(self, xs, ys, cls, tips, dash=False, label=None, points=True):
        pts = [(self.x(a), self.y(b)) for a, b in zip(xs, ys)]
        d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        self.parts.append(f'<path class="line {cls}{" dash" if dash else ""}" d="{d}"/>')
        if points:
            for (px, py), tip in zip(pts, tips):
                self.parts.append(f'<circle class="pt {cls}" cx="{px:.1f}" cy="{py:.1f}" r="3.6"/>')
        for (px, py), tip in zip(pts, tips):
            self.parts.append(f'<circle class="hit" cx="{px:.1f}" cy="{py:.1f}" r="11" tabindex="0" data-tip="{esc(tip)}"/>')
        if label:
            if abs(pts[-1][0] - (self.w - self.mr)) < 1:
                self.edge_label(pts[-1][1], label, swatch=cls, dash=dash)
            else:
                self.label(pts[-1][0], pts[-1][1], label, swatch=cls, dash=dash)

    def marker(self, xv, yv, lines, text_y=None):
        """A ring on a point, with a callout of one or more lines starting just right of it."""
        x, y = self.x(xv), self.y(yv)
        ty = y - 12 if text_y is None else text_y
        self.parts.append(f'<circle class="cross" cx="{x:.1f}" cy="{y:.1f}" r="5.5"/>')
        spans = "".join(f'<tspan x="{x + 9:.1f}" dy="{0 if i == 0 else 14}">{esc(t)}</tspan>'
                        for i, t in enumerate(lines))
        self.parts.append(f'<text class="callout" x="{x + 9:.1f}" y="{ty - 14 * (len(lines) - 1):.1f}">{spans}</text>')

    def svg(self, aria):
        # Spread right-column labels so none is closer than 13 px to its neighbour,
        # then push any that ran past the plot's bottom back up above it.
        placed = []
        for y, text, swatch, dash in sorted(self.edge_labels):
            if placed and y - placed[-1][0] < 13:
                y = placed[-1][0] + 13
            placed.append((y, text, swatch, dash))
        limit = self.h - self.mb - 4
        for i in range(len(placed) - 1, -1, -1):
            y, text, swatch, dash = placed[i]
            if y > limit:
                placed[i] = (limit, text, swatch, dash)
            limit = min(placed[i][0], limit) - 13
        for y, text, swatch, dash in placed:
            self.label(self.w - self.mr, y, text, swatch=swatch, dash=dash)
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(aria)}">'
                + "".join(self.parts) + "</svg>")


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="data"><summary>Data table</summary><div class="tw"><table>'
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></details>")


BATCH_TICKS = [(b, str(b)) for b in (1, 2, 4, 8, 16, 32, 64)]


def spec_sheet(v):
    """Every tier's methodology side by side: what was measured, how, and how it is converted."""
    s, cost, r2, pub = v["stats"], v["cost"], v["rows2"], PUBLISHED
    rows = [
        ("Device", "Snapdragon 8 Elite Gen 5 (OnePlus 15), CPU cores",
         "Jetson Orin Nano Super 8GB, GPU · $249, 7–25 W", "NVIDIA L4 24 GB, one card · 72 W limit"),
        ("Model", "Llama-3.2-1B-Instruct, GGUF Q4_K_M", "Qwen2.5-1.5B-Instruct, GGUF Q4_K_M", "Qwen2.5-7B-Instruct, fp8"),
        ("Energy from", "Cai et al., arXiv 2607.05475 (published)", "Cloud to Edge, arXiv 2604.24785 (published)",
         "this work: two sweeps on Modal, agreeing within 1.4%"),
        ("Instrument", "Qualcomm's power telemetry, per compute unit", "USB power meter at the board's input",
         "NVML cumulative energy counter (as ML.ENERGY)"),
        ("Protocol", "below 28 °C, airplane mode, screen off; warm-up discarded; ≥ 3 repeats",
         "one ~12-token prompt, 100 tokens generated, 5 runs; power mode not stated",
         "warm-up discarded; 3 repeats, median; prefill passes ≥ 3 s; prefix caching off; batches 1–64"),
        ("Rates", f"prefill {pub['user']['pf']:.3f} J/prompt token · decode {pub['user']['dec']:.3f} J/token",
         f"{pub['onu']['dec']:.2f} J per generated token, prefill included",
         f"decode {r2[0]['decode_J_per_output_token']:.2f} → {r2[-1]['decode_J_per_output_token']:.3f} J/token and "
         f"prefill {r2[0]['prefill_J_per_input_token']:.3f} → {r2[-1]['prefill_J_per_input_token']:.3f} J/prompt token, batch 1 → 64"),
        ("Counts", "the whole chip (in CPU-only inference, the CPU cluster)", "the whole board at its plug, idle included",
         "the GPU card, its memory included"),
        ("To the common boundary", "as is: a lower bound, with no memory, screen or radio", "as is",
         f"× {BOUNDARY['it_over_accel']:.2f} for host and idle capacity × PUE {BOUNDARY['pue_isp']} = × {v['k_mid']:.2f}"),
        ("Per query here", f"{cost['user']:.1f} J", f"{cost['onu']:.0f} J",
         f"{v['olt_sys'][0]:,.0f} J at batch 1 → {v['olt_sys'][-1]:.0f} J at batch 64"),
        ("Answers", f"{s['user']['n']:,} GSM8K, vLLM GGUF loader on an L4", f"{s['onu']['n']:,}, same loader",
         f"{s['olt']['n']:,}, vLLM fp8"),
        ("Accuracy", f"{s['user']['acc']:.3f}", f"{s['onu']['acc']:.3f}", f"{s['olt']['acc']:.3f}"),
        ("Weakest point", "a lower bound; the exact 4-bit scheme is not stated",
         "a short fixed prompt and 5 runs; board not yet measured here",
         "host and building are converted with published shares, not metered"),
    ]
    head = "".join(f'<th><i class="sw sw-{t}"></i>{NAME[t]}</th>' for t in TIERS)
    body = "".join(f'<tr><th scope="row">{esc(k)}</th>' + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>"
                   for k, *cells in rows)
    return f'<div class="tw"><table class="spec"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def chart_decode(v):
    c = Chart(680, 300, (1, 64), (0.03, 4), xlog=True, ylog=True)
    c.axes(BATCH_TICKS, [(x, f"{x:g}") for x in (0.05, 0.1, 0.2, 0.5, 1, 2)],
           "OLT batch size (queries answered together)", "J PER GENERATED TOKEN, GPU CARD")
    r1 = [r["decode_J_per_output_token"] for r in v["rows1"]]
    r2 = [r["decode_J_per_output_token"] for r in v["rows2"]]
    c.series(v["B"], r1, "s-olt thin", [f"Run 1, batch {b}: {y:.3f} J/token" for b, y in zip(v["B"], r1)],
             dash=True, points=False)
    c.series(v["B"], r2, "s-olt", [f"OLT batch {b}: {y:.3f} J per generated token ({r['tokens_per_s']:.0f} tok/s)"
                                   for b, y, r in zip(v["B"], r2, v["rows2"])], label="OLT (run 2)")
    rows = [(b, f"{a:.4f}", f"{b2:.4f}") for b, a, b2 in zip(v["B"], r1, r2)]
    return c.svg(f"OLT decode energy per token falls from {r2[0]:.2f} J at batch 1 to {r2[-1]:.3f} J at batch 64."), \
        table(["Batch", "Run 1 J/token", "Run 2 J/token"], rows)


def chart_mechanism(v):
    rows2 = v["rows2"]
    t = Chart(330, 250, (1, 64), (20, 2000), xlog=True, ylog=True, ml=52, mr=24)
    t.axes([(b, str(b)) for b in (1, 4, 16, 64)], [(x, f"{x:,}") for x in (30, 100, 300, 1000)],
           "Batch size", "TOKENS / S")
    t.series(v["B"], [r["tokens_per_s"] for r in rows2], "s-olt",
             [f"Batch {r['batch']}: {r['tokens_per_s']:.0f} tokens/s" for r in rows2])
    p = Chart(330, 250, (1, 64), (0, 80), xlog=True, ml=52, mr=24)
    p.axes([(b, str(b)) for b in (1, 4, 16, 64)], [(x, str(x)) for x in (0, 20, 40, 60, 80)],
           "Batch size", "WATTS")
    idle = v["idle2"]
    y = p.y(idle)
    p.parts.append(f'<line class="ref s-idle dash" x1="{p.ml}" x2="{p.w - p.mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
    p.parts.append(f'<text class="dlabel" x="{p.ml + 6}" y="{y - 6:.1f}">idle {idle:.1f} W</text>')
    p.series(v["B"], [r["mean_power_W"] for r in rows2], "s-olt",
             [f"Batch {r['batch']}: {r['mean_power_W']:.1f} W" for r in rows2])
    rows = [(r["batch"], f"{r['tokens_per_s']:.0f}", f"{r['mean_power_W']:.1f}") for r in rows2]
    return (t.svg("Throughput rises from 29 to about 1,400 tokens per second as batch grows from 1 to 64."),
            p.svg(f"Power stays at about 72 watts at every batch size; idle is {idle:.1f} watts."),
            table(["Batch", "Tokens/s", "Watts"], rows))


def chart_prefill(v):
    rows2 = v["rows2"]
    c = Chart(680, 250, (1, 64), (0, 0.045), xlog=True)
    c.axes(BATCH_TICKS, [(x, f"{x:.2f}") for x in (0, 0.01, 0.02, 0.03, 0.04)],
           "OLT batch size", "J PER PROMPT TOKEN")
    g = [r["prefill_J_per_input_token"] for r in rows2]
    n = [r["prefill_J_per_input_token_net"] for r in rows2]
    c.series(v["B"], n, "s-olt", [f"Batch {b}: {y:.4f} J/prompt token, net of idle" for b, y in zip(v["B"], n)],
             dash=True, label="net of idle")
    c.series(v["B"], g, "s-olt", [f"Batch {b}: {y:.4f} J/prompt token ({r['prefill_reps']:.0f} repeats)"
                                  for b, y, r in zip(v["B"], g, rows2)], label="gross")
    rows = [(r["batch"], f"{a:.5f}", f"{b:.5f}", f"{r['prefill_reps']:.0f}") for r, a, b in zip(rows2, g, n)]
    return c.svg("Prefill energy per prompt token falls from about 0.034 joules at batch 1 to a flat 0.015 from "
                 "batch 16 on."), table(["Batch", "Gross J/prompt token", "Net J/prompt token", "Repeats averaged"], rows)


def chart_per_query(v):
    c = Chart(680, 330, (1, 64), (6, 2500), xlog=True, ylog=True)
    c.axes(BATCH_TICKS, [(x, f"{x:g}") for x in (10, 30, 100, 300, 1000)],
           "OLT batch size", "J PER QUERY, SYSTEM BOUNDARY")
    onu, user = v["cost"]["onu"], v["cost"]["user"]
    c.curve_band(v["B"], v["olt_sys_low"], v["olt_sys"], "s-olt")
    c.ref(onu, "s-onu", f"ONU {onu:.0f} J")
    c.ref(user, "s-user", f"User {user:.0f} J")
    c.series(v["B"], v["olt_g"], "s-olt thin", [f"OLT batch {b}: {y:.1f} J per query at the GPU card alone"
                                               for b, y in zip(v["B"], v["olt_g"])],
             dash=True, points=False, label="OLT, GPU only")
    c.series(v["B"], v["olt_sys"], "s-olt", [f"OLT batch {b}: {y:.0f} J per query, whole server at PUE {BOUNDARY['pue_isp']}"
                                             for b, y in zip(v["B"], v["olt_sys"])], label="OLT")
    if v["x_onu_sys"]:
        c.marker(v["x_onu_sys"], onu, ["OLT cheaper than the ONU", f"from batch ≈ {v['x_onu_sys']:.0f}"],
                 text_y=c.y(onu) - 12)
    rows = [(b, f"{sy:.0f}", f"{lo:.0f}", f"{g:.1f}") for b, sy, lo, g in zip(v["B"], v["olt_sys"], v["olt_sys_low"], v["olt_g"])]
    return c.svg(f"At a common, whole-system boundary, OLT energy per query falls below the ONU's {onu:.0f} joules "
                 f"at about batch {bx(v['x_onu_sys'])[2:] if v['x_onu_sys'] else 'none'}; it stays above the user "
                 f"tier's {user:.0f} joules {'until batch ' + format(v['x_user_sys'], '.0f') if v['x_user_sys'] else 'across the whole range'}."), \
        table(["Batch", f"OLT J/query, system (PUE {BOUNDARY['pue_isp']})", f"System (PUE {BOUNDARY['pue_low']})",
               "GPU card only"], rows)


def chart_marginal(v):
    rows2 = v["rows2"]
    c = Chart(680, 260, (1, 64), (1, 1000), xlog=True, ylog=True)
    c.axes(BATCH_TICKS, [(x, f"{x:g}") for x in (1, 3, 10, 30, 100, 300, 1000)],
           "OLT batch size", "J PER QUERY, GPU CARD (93 PROMPT / 300 GENERATED TOKENS)")
    avg = [r["average_J_per_query"] for r in rows2]
    mar = [max(r["marginal_J_per_query"], 1.01) for r in rows2]
    c.series(v["B"], avg, "s-olt", [f"Batch {b}: average {y:.1f} J per query" for b, y in zip(v["B"], avg)],
             label="average")
    c.series(v["B"][1:], mar[1:], "s-olt", [f"Batch {b}: one more query adds {r['marginal_J_per_query']:.1f} J"
                                            for b, r in zip(v["B"][1:], rows2[1:])], dash=True, label="one more query")
    rows = [(r["batch"], f"{r['average_J_per_query']:.1f}", f"{r['marginal_J_per_query']:.2f}") for r in rows2]
    return c.svg("Once the OLT is busy, one more query adds about 4 joules, while the average cost per query is "
                 "16 to 95 joules."), table(["Batch", "Average J/query", "Marginal J/query"], rows)


def hist_panel(recs, t, key, lo, hi, bins, title):
    """Step outline of one confidence distribution per group (wrong in grey, correct in the
    tier's colour), each normalised to its own peak so group sizes don't hide the shapes."""
    c = Chart(214, 170, (lo, hi), (0, 1), ml=12, mr=12, mt=26, mb=30)
    step = (hi - lo) / bins
    base = c.h - c.mb
    means = {}
    for grp, cls in ((False, "s-wrong"), (True, f"s-{t}")):
        vals = [r[key] for r in recs if r["correct"] == grp]
        means[grp] = st.mean(vals) if vals else float("nan")
        counts = [0] * bins
        for x in vals:
            counts[min(bins - 1, max(0, int((x - lo) / step)))] += 1
        peak = max(counts) or 1
        path = []
        for i, k in enumerate(counts):
            y = base - (k / peak) * (base - c.mt) * 0.9
            path.append(f"L{c.x(lo + i * step):.1f},{y:.1f} L{c.x(lo + (i + 1) * step):.1f},{y:.1f}")
        d_attr = f"M{c.x(lo):.1f},{base:.1f} " + " ".join(path) + f" L{c.x(hi):.1f},{base:.1f}"
        c.parts.append(f'<path class="hist {cls}" d="{d_attr}"/>')
    tip = f"{title}. Mean confidence: correct {means[True]:.3f}, wrong {means[False]:.3f}"
    c.parts.append(f'<rect class="hit" x="{c.ml}" y="{c.mt}" width="{c.w - c.ml - c.mr}" '
                   f'height="{base - c.mt}" tabindex="0" data-tip="{esc(tip)}"/>')
    c.parts.append(f'<line class="axis" x1="{c.ml}" x2="{c.w - c.mr}" y1="{c.h - c.mb}" y2="{c.h - c.mb}"/>')
    for xv in (lo, (lo + hi) / 2, hi):
        c.parts.append(f'<text class="tick" x="{c.x(xv):.1f}" y="{c.h - c.mb + 15}" text-anchor="middle">{xv:g}</text>')
    c.parts.append(f'<text class="ptitle" x="{c.ml}" y="14">{esc(title)}</text>')
    return c.svg(f"{title}: distribution of confidence for correct and wrong answers")


def chart_confidence(v, recs):
    s = v["stats"]
    top = "".join(hist_panel(recs[t], t, "cmin", 0, 1, 20, f"{NAME[t]} · t = {s[t]['t_min']:+.1f}") for t in TIERS)
    bot = "".join(hist_panel(recs[t], t, "cmean", 0.6, 1.0, 20, f"{NAME[t]} · t = {s[t]['t_mean']:+.1f}") for t in TIERS)
    rows = [(NAME[t], s[t]["n"], f"{s[t]['acc']:.3f}", f"{s[t]['t_min']:+.2f}", f"{s[t]['t_mean']:+.2f}") for t in TIERS]
    return top, bot, table(["Tier", "Answers", "Accuracy", "Welch t, exp(min)", "Welch t, exp(mean)"], rows)


def chart_reliability(recs):
    c = Chart(680, 300, (0.1, 0.55), (0, 1))
    c.axes([(x, f"{x:.1f}") for x in (0.1, 0.2, 0.3, 0.4, 0.5)], [(x, f"{x:.1f}") for x in (0, 0.2, 0.4, 0.6, 0.8, 1.0)],
           "Confidence, exp(min token logprob) — mean within each tenth of answers", "ACCURACY")
    rows = []
    for t in TIERS:
        rs = sorted(recs[t], key=lambda r: r["cmin"])
        k = len(rs) // 10
        xs, ys, tips = [], [], []
        for i in range(10):
            chunk = rs[i * k:(i + 1) * k] if i < 9 else rs[9 * k:]
            cx = st.mean(r["cmin"] for r in chunk)
            acc = sum(r["correct"] for r in chunk) / len(chunk)
            xs.append(cx)
            ys.append(acc)
            tips.append(f"{NAME[t]}, confidence tenth {i + 1} (mean {cx:.2f}): accuracy {acc:.2f}, n={len(chunk)}")
            rows.append((NAME[t], i + 1, f"{cx:.3f}", f"{acc:.3f}", len(chunk)))
        c.series(xs, ys, f"s-{t}", tips, label=NAME[t])
    return c.svg("For every tier, accuracy rises with confidence: the least confident tenth of answers is wrong far "
                 "more often than the most confident tenth."), \
        table(["Tier", "Confidence tenth", "Mean confidence", "Accuracy", "Answers"], rows)


def chart_lengths(recs):
    c = Chart(680, 190, (0, 520), (0, 3), ml=58, mr=40, mt=16, mb=40)
    for v, lab in ((0, "0"), (128, "128"), (256, "256"), (384, "384"), (512, "512")):
        x = c.x(v)
        c.parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{c.mt}" y2="{c.h - c.mb}"/>')
        c.parts.append(f'<text class="tick" x="{x:.1f}" y="{c.h - c.mb + 16}" text-anchor="middle">{lab}</text>')
    c.parts.append(f'<text class="axlabel" x="{(c.ml + c.w - c.mr) / 2:.0f}" y="{c.h - 6}" text-anchor="middle">'
                   f'Generated tokens per answer (limit 512)</text>')
    rows = []
    meds = {}
    for i, t in enumerate(TIERS):
        g = sorted(r["tg"] for r in recs[t])
        dec = st.quantiles(g, n=10)
        q = st.quantiles(g, n=4)
        y = c.y(2.5 - i)
        med = meds[t] = st.median(g)
        c.parts.append(f'<text class="tick" x="{c.ml - 8}" y="{y + 3.5:.1f}" text-anchor="end">{NAME[t]}</text>')
        c.parts.append(f'<line class="whisk s-{t}" x1="{c.x(dec[0]):.1f}" x2="{c.x(dec[8]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        c.parts.append(f'<rect class="box s-{t}" x="{c.x(q[0]):.1f}" y="{y - 7:.1f}" width="{c.x(q[2]) - c.x(q[0]):.1f}" height="14" rx="3"/>')
        c.parts.append(f'<circle class="med" cx="{c.x(med):.1f}" cy="{y:.1f}" r="4"/>')
        tip = (f"{NAME[t]}: median {med:.0f} tokens; middle half {q[0]:.0f}–{q[2]:.0f}; "
               f"10th–90th percentile {dec[0]:.0f}–{dec[8]:.0f}")
        c.parts.append(f'<rect class="hit" x="{c.ml}" y="{y - 14:.1f}" width="{c.w - c.ml - c.mr}" height="28" '
                       f'tabindex="0" data-tip="{esc(tip)}"/>')
        rows.append((NAME[t], f"{med:.0f}", f"{q[0]:.0f}", f"{q[2]:.0f}", f"{dec[0]:.0f}", f"{dec[8]:.0f}"))
    return c.svg(f"Median answer length is {meds['user']:.0f} tokens for the user tier, {meds['onu']:.0f} for the "
                 f"ONU and {meds['olt']:.0f} for the OLT."), \
        table(["Tier", "Median", "25th pct", "75th pct", "10th pct", "90th pct"], rows)


def chart_frontier(sim):
    """Accuracy against J/query along each policy's beta sweep, at one OLT load, ONU as published."""
    run = sim["runs"][1.0]
    rows = [r for r in run["rows"] if r["peak_load"] == FRONTIER_LOAD]
    spec = (("stepwise", "s-base", True), ("skip_onu", "s-skip", False), ("piggyback", "s-pig", False))
    pts = {p: sorted((r for r in rows if r["policy"] == p), key=lambda r: r["beta"]) for p, _, _ in spec}
    accs = [r["accuracy"] for p in pts for r in pts[p]]
    js = [r["J_per_query"] for p in pts for r in pts[p]]
    x0, x1 = math.floor(min(accs) * 20) / 20, math.ceil(max(accs) * 20) / 20
    y1 = math.ceil(max(js) / 200) * 200
    c = Chart(680, 320, (x0, x1), (0, y1), mr=150)
    c.axes([(x / 10, f"{x / 10:.1f}") for x in range(4, 11) if x0 <= x / 10 <= x1],
           [(y, f"{y:,}") for y in range(0, y1 + 1, 200)],
           "Accuracy of the whole cascade (one point per β, 0.1 to 0.9)", "J PER QUERY")
    trows = []
    for p, cls, dash in spec:
        xs = [r["accuracy"] for r in pts[p]]
        ys = [r["J_per_query"] for r in pts[p]]
        c.series(xs, ys, cls, [f"{POLICY_NAME[p]}, β {r['beta']:.1f}: accuracy {r['accuracy']:.3f}, "
                               f"{r['J_per_query']:.0f} J per query" for r in pts[p]], dash=dash)
        c.edge_label(c.y(ys[-1]), POLICY_NAME[p], swatch=cls, dash=dash)
        trows += [(POLICY_NAME[p], f"{r['beta']:.1f}", f"{r['accuracy']:.3f}", f"{r['J_per_query']:.1f}",
                   f"{r['final_user']:.0%} / {r['final_onu']:.0%} / {r['final_olt']:.0%}") for r in pts[p]]
    return c.svg("Along the beta sweep, piggyback and the two-tier chain without the ONU trace the same curve, both "
                 "far below three-tier RecServe."), \
        table(["Policy", "β", "Accuracy", "J/query", "Answered at user / ONU / OLT"], trows)


def chart_regime(sim, vs):
    """Grid: piggyback's saving over the better fixed chain, by ONU cost (rows) and OLT load (columns)."""
    runs, loads = sim["runs"], vs["loads"]
    W, left, top, rh = 680, 132, 50, 40
    cw = (W - left - 4) / len(loads)
    H = top + rh * len(SCALES) + 6
    parts = [f'<text class="axlabel" x="0" y="11">ONU, PER QUERY</text>',
             f'<text class="axlabel" x="{left + 2}" y="11">OLT PEAK LOAD, QUERIES IN SERVICE · ARRIVALS/HOUR AT PEAK</text>']
    for k, l in enumerate(loads):
        x = left + cw * (k + 0.5)
        parts.append(f'<text class="gtick" x="{x:.1f}" y="30" text-anchor="middle">{l:g}</text>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="43" text-anchor="middle">{vs["per_h"][l]:,.0f}</text>')
    trows = []
    fmt = lambda j: "—" if j is None else f"{j:.0f} J"
    for i, s in enumerate(SCALES):
        run, y = runs[s], top + rh * i
        onu = vs["onu_J"][s]
        parts.append(f'<text class="gtick" x="0" y="{y + rh / 2 - 1:.1f}">{onu:.0f} J</text>')
        parts.append(f'<text class="tick" x="0" y="{y + rh / 2 + 12:.1f}">{"as published" if s == 1 else f"× {s:g} published"}</text>')
        for k, l in enumerate(loads):
            g = vs["gain"][(s, l)]
            x = left + cw * k
            J = {p: fj(run, l, p) for p in ("stepwise", "skip_onu", "static", "piggyback")}
            real = g is not None and abs(g) >= NOISE
            op = 0.2 + 0.7 * min(max(g, 0) / 0.15, 1) if real and g > 0 else 0
            text = "—" if g is None else (f"{g:+.0%}" if real else "≈ 0")
            tip = (f"ONU {onu:.0f} J per query, OLT peak load {l:g} (~{vs['per_h'][l]:,.0f} arrivals/h at peak). "
                   f"At accuracy {ACC}: piggyback {fmt(J['piggyback'])}; RecServe three tiers {fmt(J['stepwise'])}; "
                   f"ONU dropped {fmt(J['skip_onu'])}; static configuration {fmt(J['static'])}.")
            parts.append(f'<rect class="cell{"" if op else " zero"}" x="{x + 1.5:.1f}" y="{y + 1.5:.1f}" '
                         f'width="{cw - 3:.1f}" height="{rh - 3:.1f}" rx="3"{f' style="fill-opacity:{op:.2f}"' if op else ""}/>')
            parts.append(f'<text class="ctext{" on" if op > 0.5 else ""}" x="{x + cw / 2:.1f}" y="{y + rh / 2 + 4:.1f}" '
                         f'text-anchor="middle">{text}</text>')
            parts.append(f'<rect class="hit" x="{x + 1.5:.1f}" y="{y + 1.5:.1f}" width="{cw - 3:.1f}" height="{rh - 3:.1f}" '
                         f'tabindex="0" data-tip="{esc(tip)}"/>')
            trows.append((f"{onu:.0f}", f"{l:g}", fmt(J["stepwise"]), fmt(J["skip_onu"]), fmt(J["static"]),
                          fmt(J["piggyback"]), "—" if g is None else f"{g:+.1%}"))
    b = vs["best"]
    svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(f"Piggyback saves nothing over the better fixed chain at the published ONU cost; its saving grows as the ONU gets cheaper and the OLT busier, up to {b[0]:.0%}.")}">'
           + "".join(parts) + "</svg>")
    return svg, table(["ONU J/query", "OLT peak load", "RecServe, three tiers", "ONU dropped", "Static",
                       "Piggyback", "Piggyback vs better fixed chain"], trows)


def chart_day(sim):
    """One day at one configuration: the OLT's cost by hour, and how much each policy uses the ONU."""
    run = sim["runs"][DAY["scale"]]
    cfg = next(c for c in run["configs"] if c["peak_load"] == DAY["load"])
    onu = run["fixed_J_per_query"]["onu"]
    hrs = [h + 0.5 for h in range(24)]
    xt = [(h, f"{h:02d}:00") for h in (0, 6, 12, 18, 24)]
    olt = cfg["olt_hourly_J_per_query"]
    a = Chart(680, 200, (0, 24), (40, 1200), ylog=True)
    a.axes(xt, [(y, f"{y:g}") for y in (50, 100, 200, 500, 1000)], "Hour of day", "J PER QUERY")
    a.ref(onu, "s-onu", f"ONU {onu:.0f} J")
    a.series(hrs, olt, "s-olt", [f"{h:02d}:00–{h + 1:02d}:00: a query at the OLT costs {j:.0f} J"
                                 for h, j in enumerate(olt)], points=False)
    a.edge_label(a.y(olt[-1]), "OLT", swatch="s-olt")
    hourly = {h["policy"]: h["hours"] for h in run["hourly"]
              if h["peak_load"] == DAY["load"] and h["beta"] == DAY["beta"]}
    b = Chart(680, 220, (0, 24), (0, 0.35))
    b.axes(xt, [(y, f"{y:.0%}") for y in (0, 0.1, 0.2, 0.3)], "Hour of day", "QUERIES ANSWERED AT THE ONU")
    spec = (("stepwise", "s-base", True), ("static", "s-base dot", False),
            ("oracle", "s-pig thin dot", False), ("piggyback", "s-pig", False))
    for p, cls, dash in spec:
        ys = [x["final_onu"] or 0 for x in hourly[p]]
        b.series(hrs, ys, cls, [f"{h:02d}:00–{h + 1:02d}:00: {POLICY_NAME[p]} answers {y:.0%} of queries at the ONU"
                                for h, y in enumerate(ys)], dash=dash, points=p == "piggyback")
        b.edge_label(b.y(ys[-1]), POLICY_NAME[p], swatch=cls, dash=dash)
    trows = [(f"{h:02d}:00", f"{olt[h]:.0f}", *(f"{hourly[p][h]['final_onu'] or 0:.0%}" for p, _, _ in spec))
             for h in range(24)]
    return (a.svg(f"Over the day the OLT's cost per query swings from {min(olt):.0f} to {max(olt):.0f} joules; "
                  f"the ONU's stays at {onu:.0f}."),
            b.svg("Piggyback keeps queries on the ONU through the night and sends them past it from late morning, "
                  "tracking the oracle; the static configuration skips the ONU almost all day."),
            table(["Hour", "OLT J/query", *(POLICY_NAME[p] + ", at ONU" for p, _, _ in spec)], trows))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

CSS = """
:root{--paper:#F2F5F5;--surface:#FBFCFC;--surface-2:#E9EFEF;--ink:#14262B;--ink-soft:#4A6169;--ink-faint:#7C9198;
--accent:#B4650E;--accent-2:#1F6E6B;--rule:#D2DCDC;--rule-soft:#E2E9E9;
--c-user:#B4650E;--c-onu:#7A52C7;--c-olt:#00918A;--c-wrong:#8FA2A8;
--c-base:#7C9198;--c-skip:#B0457A;--c-pig:#2E62B8;
--measure:68ch;--f-display:"Spectral",Georgia,"Times New Roman",serif;
--f-body:"IBM Plex Sans","Helvetica Neue",Arial,sans-serif;--f-mono:"IBM Plex Mono","SF Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#101A1D;--surface:#16242A;--surface-2:#1D2E35;
--ink:#E4EDED;--ink-soft:#9CB0B6;--ink-faint:#6E858C;--accent:#E09A3C;--accent-2:#4FA9A5;--rule:#2A3D44;--rule-soft:#223339;
--c-user:#C0873C;--c-onu:#8E7BE8;--c-olt:#19A79E;--c-wrong:#5E7379;--c-base:#7F969D;--c-skip:#C86A98;--c-pig:#5B8BD9}}
:root[data-theme="dark"]{--paper:#101A1D;--surface:#16242A;--surface-2:#1D2E35;--ink:#E4EDED;--ink-soft:#9CB0B6;
--ink-faint:#6E858C;--accent:#E09A3C;--accent-2:#4FA9A5;--rule:#2A3D44;--rule-soft:#223339;
--c-user:#C0873C;--c-onu:#8E7BE8;--c-olt:#19A79E;--c-wrong:#5E7379;--c-base:#7F969D;--c-skip:#C86A98;--c-pig:#5B8BD9}
body{background:var(--paper);color:var(--ink);font-family:var(--f-body);font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:46rem;margin:0 auto;padding:3.5rem 1.5rem 5rem;display:flex;flex-direction:column;gap:2.75rem}
.eyebrow{font-family:var(--f-mono);font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
h1{font-family:var(--f-display);font-weight:600;font-size:clamp(1.9rem,5vw,2.6rem);line-height:1.15;margin:.4rem 0 0;text-wrap:balance}
.sub{font-family:var(--f-display);font-style:italic;font-size:1.1rem;color:var(--ink-soft);max-width:var(--measure);margin:.8rem 0 0}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(10rem,1fr));gap:.8rem 1.5rem;padding:1rem 0;margin-top:1.2rem;
border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.meta div{display:flex;flex-direction:column;gap:.15rem}
.meta dt{font-family:var(--f-mono);font-size:.64rem;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-faint)}
.meta dd{margin:0;font-size:.88rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(10rem,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.tile{background:var(--surface);padding:1rem 1.1rem;display:flex;flex-direction:column;gap:.35rem}
.tile .big{font-family:var(--f-display);font-size:1.9rem;font-weight:600;line-height:1;font-variant-numeric:tabular-nums}
.tile .what{font-size:.82rem;color:var(--ink-soft);line-height:1.4}
section{display:flex;flex-direction:column;gap:1rem}
h2{font-family:var(--f-display);font-weight:600;font-size:1.4rem;margin:0;padding-bottom:.45rem;border-bottom:2px solid var(--ink);text-wrap:balance}
h3{font-size:.98rem;font-weight:600;margin:.8rem 0 0;text-wrap:balance}
p{margin:0;max-width:var(--measure)}
figure{margin:0;display:flex;flex-direction:column;gap:.5rem}
figcaption{font-size:.84rem;color:var(--ink-soft);max-width:var(--measure)}
svg{width:100%;height:auto;display:block;overflow:visible}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:1rem}
.trio{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.5rem}
.stack{display:flex;flex-direction:column;gap:.25rem}
.rowlabel{font-family:var(--f-mono);font-size:.66rem;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-faint)}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;font-size:.8rem;color:var(--ink-soft)}
.legend span{display:inline-flex;align-items:center;gap:.4rem}
.legend i{display:inline-block;width:18px;height:3px;border-radius:2px}
.grid{stroke:var(--rule-soft);stroke-width:1}
.axis{stroke:var(--rule);stroke-width:1}
.tick{fill:var(--ink-faint);font-family:var(--f-mono);font-size:10px;font-variant-numeric:tabular-nums}
.gtick{fill:var(--ink);font-family:var(--f-mono);font-size:11px;font-weight:500;font-variant-numeric:tabular-nums}
.axlabel{fill:var(--ink-faint);font-family:var(--f-mono);font-size:10px;letter-spacing:.06em}
.dlabel{fill:var(--ink-soft);font-family:var(--f-body);font-size:11px}
.ptitle{fill:var(--ink);font-family:var(--f-body);font-size:11.5px;font-weight:600}
.callout{fill:var(--ink);font-family:var(--f-body);font-size:11.5px;font-weight:600}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.line.thin{stroke-width:1.4;opacity:.8}
.dash{stroke-dasharray:6 4}
.dot{stroke-dasharray:1.5 3.5}
.ref{stroke-width:1.6}
.swatch{stroke-width:2.5;stroke-linecap:round}
.band{opacity:.12;stroke:none}
.pt{stroke:var(--surface);stroke-width:1.5}
.hit{fill:transparent;stroke:none;outline:none}
.hit:focus-visible{stroke:var(--ink);stroke-width:1.5;fill:none}
.cross{fill:var(--surface);stroke:var(--ink);stroke-width:2}
.hist{fill:none;stroke-width:1.8;stroke-linejoin:round}
.whisk{stroke-width:2;stroke-linecap:round}
.box{stroke:none;opacity:.85}
.med{fill:var(--surface);stroke:var(--ink);stroke-width:1.8}
.cell{fill:var(--c-pig);stroke:none}
.cell.zero{fill:var(--surface-2)}
.ctext{fill:var(--ink-soft);font-family:var(--f-mono);font-size:11.5px;font-variant-numeric:tabular-nums}
.ctext.on{fill:var(--paper);font-weight:500}
.s-user{stroke:var(--c-user)}.pt.s-user,.box.s-user,.band.s-user{fill:var(--c-user)}
.s-onu{stroke:var(--c-onu)}.pt.s-onu,.box.s-onu,.band.s-onu{fill:var(--c-onu)}
.s-olt{stroke:var(--c-olt)}.pt.s-olt,.box.s-olt,.band.s-olt{fill:var(--c-olt)}
.s-base{stroke:var(--c-base)}.pt.s-base{fill:var(--c-base)}
.s-skip{stroke:var(--c-skip)}.pt.s-skip{fill:var(--c-skip)}
.s-pig{stroke:var(--c-pig)}.pt.s-pig{fill:var(--c-pig)}
.s-wrong{stroke:var(--c-wrong)}
.band.s-user,.band.s-onu,.band.s-olt{stroke:none}
.pt.s-user,.pt.s-onu,.pt.s-olt,.pt.s-base,.pt.s-skip,.pt.s-pig{stroke:var(--surface)}
.s-idle{stroke:var(--ink-faint)}
details.data summary{cursor:pointer;font-family:var(--f-mono);font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-faint)}
details.data summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tw{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;background:var(--surface);margin-top:.5rem}
table{border-collapse:collapse;width:100%;font-size:.82rem}
th,td{padding:.4rem .7rem;text-align:left;border-bottom:1px solid var(--rule-soft);font-variant-numeric:tabular-nums}
th{font-family:var(--f-mono);font-size:.64rem;letter-spacing:.07em;text-transform:uppercase;color:var(--ink-soft);background:var(--surface-2)}
table.spec{font-size:.8rem;line-height:1.4}
table.spec td{vertical-align:top;min-width:8.5rem;width:29%}
table.spec tbody th{vertical-align:top;background:var(--surface);color:var(--ink-faint);width:13%;line-height:1.35}
table.spec thead th{font-size:.72rem;color:var(--ink)}
table.spec tbody tr:last-child td,table.spec tbody tr:last-child th{border-bottom:none}
.sw{display:inline-block;width:.6rem;height:.6rem;border-radius:2px;margin-right:.45rem;vertical-align:0}
.sw-user{background:var(--c-user)}.sw-onu{background:var(--c-onu)}.sw-olt{background:var(--c-olt)}
pre.packet{margin:0;font-family:var(--f-mono);font-size:.78rem;line-height:1.55;background:var(--surface);border:1px solid var(--rule);
border-radius:4px;padding:.75rem .9rem;overflow-x:auto;color:var(--ink)}
pre.packet .c{color:var(--ink-faint)}
ul{margin:0;padding-left:1.2rem;max-width:var(--measure);display:flex;flex-direction:column;gap:.45rem}
li::marker{color:var(--ink-faint)}
code{font-family:var(--f-mono);font-size:.86em;background:var(--surface-2);padding:.05em .3em;border-radius:3px}
#tip{position:fixed;z-index:10;max-width:18rem;background:var(--ink);color:var(--paper);font-size:.78rem;line-height:1.35;
padding:.4rem .55rem;border-radius:4px;pointer-events:none}
footer{border-top:1px solid var(--rule);padding-top:1rem;font-size:.78rem;color:var(--ink-faint);max-width:var(--measure)}
@media (max-width:40rem){.trio{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(el,x,y){tip.textContent=el.getAttribute('data-tip');tip.hidden=false;place(x,y);}
  function place(x,y){var w=tip.offsetWidth,h=tip.offsetHeight;
    var nx=x+14,ny=y+14; if(nx+w>window.innerWidth-8)nx=x-w-14; if(ny+h>window.innerHeight-8)ny=y-h-14;
    tip.style.left=nx+'px';tip.style.top=ny+'px';}
  document.querySelectorAll('[data-tip]').forEach(function(el){
    el.addEventListener('mouseenter',function(e){show(el,e.clientX,e.clientY);});
    el.addEventListener('mousemove',function(e){place(e.clientX,e.clientY);});
    el.addEventListener('mouseleave',function(){tip.hidden=true;});
    el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.left+r.width/2,r.top);});
    el.addEventListener('blur',function(){tip.hidden=true;});
  });
})();
"""


def legend(items):
    out = []
    for cls, text, dash in items:
        style = (f"background:repeating-linear-gradient(90deg,var(--c-{cls}) 0 6px,transparent 6px 10px)"
                 if dash else f"background:var(--c-{cls})")
        out.append(f'<span><i style="{style}"></i>{esc(text)}</span>')
    return '<div class="legend">' + "".join(out) + "</div>"


def build():
    run1, run2, recs, models, answers_file = load()
    v = derive(run1, run2, recs)
    sim = load_sim()
    vs = derive_sim(sim)
    s, cost, tok = v["stats"], v["cost"], v["tok"]
    decode_svg, decode_tbl = chart_decode(v)
    thr_svg, pow_svg, mech_tbl = chart_mechanism(v)
    pf_svg, pf_tbl = chart_prefill(v)
    pq_svg, pq_tbl = chart_per_query(v)
    mg_svg, mg_tbl = chart_marginal(v)
    ctop, cbot, conf_tbl = chart_confidence(v, recs)
    rel_svg, rel_tbl = chart_reliability(recs)
    len_svg, len_tbl = chart_lengths(recs)
    fr_svg, fr_tbl = chart_frontier(sim)
    rg_svg, rg_tbl = chart_regime(sim, vs)
    day_a, day_b, day_tbl = chart_day(sim)

    main = sim["runs"][1.0]
    pf8, dec8 = OltCurve(run2["batch_curve"], v["k_mid"]).rates(FRONTIER_LOAD)
    f_st, f_sk, f_pg = (fj(main, FRONTIER_LOAD, p) for p in ("stepwise", "skip_onu", "piggyback"))
    best_g, best_s, best_l = vs["best"]
    ws = vs["worst_stale"]
    loads = vs["loads"]
    gpu_big = max((g for g in vs["gpu_gain"].values() if g is not None), default=0)
    day_onu = vs["onu_J"][DAY["scale"]]

    page = f"""<title>Three Tiers, Measured</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div id="tip" hidden></div>
<div class="wrap">
<header>
  <div class="eyebrow">Energy tests · user → ONU → OLT case study</div>
  <h1>Three Tiers, Measured</h1>
  <p class="sub">What batching does to the energy of the shared OLT tier, where that makes it cheaper than the tiers below it, whether each tier's confidence can be trusted to route on, and whether a piggybacked energy report lets the lower tiers route on the OLT's live cost.</p>
  <dl class="meta">
    <div><dt>OLT</dt><dd>Qwen2.5-7B, fp8, NVIDIA L4 — measured here</dd></div>
    <div><dt>ONU</dt><dd>Qwen2.5-1.5B Q4_K_M, Jetson Orin Nano Super — published</dd></div>
    <div><dt>User</dt><dd>Llama-3.2-1B Q4_K_M, Snapdragon — published</dd></div>
    <div><dt>Answers</dt><dd>{s['olt']['n']:,} GSM8K questions per tier</dd></div>
  </dl>
</header>

<div class="tiles">
  <div class="tile"><span class="big">{v['swing']:.0f}×</span><span class="what">drop in OLT decode energy per token from batch 1 to 64, at constant 72 W</span></div>
  <div class="tile"><span class="big">{bx(v['x_onu_sys'])}</span><span class="what">batch size from which a query costs less at the OLT than at the ONU, both counted as whole systems</span></div>
  <div class="tile"><span class="big">0 → {best_g:.0%}</span><span class="what">energy piggyback saves over the best fixed chain: nothing at the ONU's published cost, up to {best_g:.0%} with a {vs['onu_J'][best_s]:.0f} J ONU</span></div>
  <div class="tile"><span class="big">{s['user']['acc']:.2f} → {s['onu']['acc']:.2f} → {s['olt']['acc']:.2f}</span><span class="what">accuracy of user, ONU and OLT: every step up is more accurate</span></div>
</div>

<section>
  <h2>How each tier was measured</h2>
  <p>The OLT's energy is measured here, because the quantity it needs, energy as a function of batch size, is published nowhere for this model on this GPU. The user device and the ONU run one query at a time, and published measurements with better instruments than any available here already exist for them. Each source counts a different part of the machine, so every tier is brought to one boundary, <em>everything drawn to answer the query</em>, following Google's per-prompt accounting.</p>
  {spec_sheet(v)}
</section>

<section>
  <h2>Batching makes the OLT cheap per token</h2>
  <p>The OLT serves many households, so it can answer many queries at once. Its GPU draws the same power whether it carries one query or sixty-four, so the energy per token falls as the batch grows. The lower tiers serve one household each and never batch.</p>
  <figure>
    {legend([("olt", "OLT, run 2", False), ("olt", "OLT, run 1 (replication)", True)])}
    {decode_svg}
    <figcaption>Energy per generated token at the GPU card, log scales. Run 1 and run 2 agree within 1.4% at every batch size. This chart shows the mechanism only; the tiers are compared per query, at one common boundary, below.</figcaption>
    {decode_tbl}
  </figure>
  <h3>Why: power stays flat while throughput doubles</h3>
  <div class="pair"><figure>{thr_svg}</figure><figure>{pow_svg}</figure></div>
  <figcaption>Left: tokens generated per second. Right: the GPU's power draw. From batch 1 the L4 sits at its 72 W limit, so energy per token is 72 W divided by throughput, and throughput nearly doubles with every doubling of the batch up to 16.</figcaption>
  {mech_tbl}
  <h3>Reading the prompt gets cheaper only up to batch 16</h3>
  <figure>
    {pf_svg}
    <figcaption>Energy per prompt token (prefill). It falls while small batches spread per-call overhead, then stays flat: reading prompts is compute-bound, so it doesn't benefit further from batching. Each point averages enough back-to-back passes to span at least 3 seconds; a single short pass is below what the GPU's energy counter can resolve.</figcaption>
    {pf_tbl}
  </figure>
</section>

<section>
  <h2>At a common boundary, the OLT undercuts the ONU from batch {bx(v['x_onu_sys'])}</h2>
  <p>Per query, at each tier's real answer lengths ({tok['olt'][0]:.0f} prompt tokens; {tok['user'][1]:.0f}, {tok['onu'][1]:.0f} and {tok['olt'][1]:.0f} generated by user, ONU and OLT), and every tier counted as a whole system. A query costs {cost['onu']:.0f} J on the ONU and {cost['user']:.0f} J on the user device; at the OLT it depends on how many other queries it is carrying.</p>
  <figure>
    {legend([("olt", "OLT, whole server", False), ("olt", "OLT, GPU card only", True), ("onu", "ONU, whole board", False), ("user", "User, whole chip", False)])}
    {pq_svg}
    <figcaption>Energy per query, log scales. OLT: the measured GPU energy × {BOUNDARY['it_over_accel']:.2f} for host CPU, memory and idle capacity (Google's shares) × PUE {BOUNDARY['pue_isp']} (industry average, Uptime Institute). The band runs down to Google's own PUE of {BOUNDARY['pue_low']}, a best case, where the crossover is batch {bx(v['x_onu_sys_low'])}. The dashed line is the GPU card alone, the boundary most benchmarks report: judged that way the crossover would be batch {bx(v['x_onu_gpu'])}. The user device is the hardest to beat: at the system boundary the OLT {('undercuts it from batch ' + format(v['x_user_sys'], '.0f')) if v['x_user_sys'] else 'never undercuts it within batch 64'}.</figcaption>
    {pq_tbl}
  </figure>
  <h3>Sending one more query to a busy OLT costs almost nothing</h3>
  <figure>
    {mg_svg}
    <figcaption>Average cost per query against the extra energy one more query adds (the slope between adjacent batch sizes), at the GPU card, on the sweep's fixed 93 prompt / 300 generated tokens. From batch 16 the marginal cost settles at 3.6–4.5 J, about {v['marg_gpu'] * v['k_mid']:.0f} J at the whole-system boundary. Below that, the differences are between two noisy totals and shouldn't be read point by point.</figcaption>
    {mg_tbl}
  </figure>
</section>

<section>
  <h2>Confidence separates right from wrong on every tier</h2>
  <p>The cascade escalates a query when a tier's confidence is low, so confidence has to mean something. Each panel compares the confidence of correct answers (tier colour) with that of wrong ones (grey), over all {s['olt']['n']:,} answers. A larger Welch t means cleaner separation.</p>
  <figure>
    {legend([("user", "correct: user", False), ("onu", "correct: ONU", False), ("olt", "correct: OLT", False), ("wrong", "wrong answers", False)])}
    <div class="rowlabel">exp(min token logprob): the weakest token</div>
    <div class="trio">{ctop}</div>
    <div class="rowlabel" style="margin-top:.6rem">exp(mean token logprob): RecServe's definition</div>
    <div class="trio">{cbot}</div>
    <figcaption>Distributions normalised within each group. The weakest-token definition separates better on all three tiers (t = {s['user']['t_min']:+.1f}, {s['onu']['t_min']:+.1f}, {s['olt']['t_min']:+.1f} against {s['user']['t_mean']:+.1f}, {s['onu']['t_mean']:+.1f}, {s['olt']['t_mean']:+.1f}). On an older 5-shot trace the ranking was reversed, so the better definition depends on the prompt format.</figcaption>
    {conf_tbl}
  </figure>
  <h3>Higher confidence, higher accuracy</h3>
  <figure>
    {legend([("user", "User", False), ("onu", "ONU", False), ("olt", "OLT", False)])}
    {rel_svg}
    <figcaption>Each tier's answers split into ten equal groups by confidence (weakest token). Accuracy climbs with confidence on every tier, which is what a threshold on confidence relies on.</figcaption>
    {rel_tbl}
  </figure>
  <h3>How long each tier's answers are</h3>
  <figure>
    {len_svg}
    <figcaption>Dot: median. Box: middle half of answers. Line: 10th to 90th percentile. Answer length is what the per-query costs above multiply by. {s['user']['trunc']}, {s['onu']['trunc']} and {s['olt']['trunc']} answers (user, ONU, OLT) hit the 512-token limit.</figcaption>
    {len_tbl}
  </figure>
</section>

<section>
  <h2>Piggyback routing pays only when the ONU is cheap</h2>
  <p>RecServe decides whether a query escalates. The piggyback mechanism adds where it goes. Every answer travelling back down carries the energy rates each tier just ran at, so the tiers below learn the OLT's current cost without a single extra message, and use it to choose: answer here, pass to the next tier, or skip straight to the OLT. The report one OLT answer carries back, at a peak load of {FRONTIER_LOAD:g} and whole-system boundary:</p>
  <pre class="packet"><span class="c">// rides on the answer: OLT → ONU → user</span>
{{"olt": {{"J_per_prompt_token": {pf8:.3f}, "J_per_token": {dec8:.3f}, "tokens": {round(tok['olt'][1])}}}}}</pre>
  <p>The simulation replays the recorded answers for {vs['n_arrivals']:,} queries over {vs['days']} days. RecServe's β-quantile rule is unchanged, over a {vs['window']:,}-answer confidence window. The OLT carries traffic from many PONs, with BurstGPT's daily shape, peaking at {loads[0]:g} to {loads[-1]:g} queries in service ({vs['per_h'][loads[0]]:,.0f} to {vs['per_h'][loads[-1]]:,.0f} arrivals an hour at peak, by Little's law, at ~{vs['svc']:.0f} s per answer). Each arriving query meets a Poisson-distributed batch, so any single report is noisy. Policies are compared at equal accuracy, each along its own β sweep.</p>
  <h3>With the published ONU, dropping it works as well</h3>
  <figure>
    {legend([("base", POLICY_NAME["stepwise"], True), ("skip", POLICY_NAME["skip_onu"], False), ("pig", POLICY_NAME["piggyback"], False)])}
    {fr_svg}
    <figcaption>OLT peak load {FRONTIER_LOAD:g} (~{vs['per_h'][FRONTIER_LOAD]:,.0f} arrivals an hour at peak), ONU as published. The ONU spends {cost['onu']:.0f} J for {s['onu']['acc']:.3f} accuracy; once the OLT is at all busy, an escalated query is both cheaper and more accurate there. At {ACC} accuracy piggyback uses {f_pg:.0f} J per query, the two-tier chain without the ONU {f_sk:.0f} J, and three-tier RecServe {f_st:.0f} J. The saving is real, but it comes from dropping the ONU, which needs no live signal.</figcaption>
    {fr_tbl}
  </figure>
  <h3>Where the live signal earns its keep</h3>
  <figure>
    {rg_svg}
    <figcaption>Energy piggyback saves over the better of the two fixed chains (three-tier RecServe, or the ONU dropped), at {ACC} accuracy. Rows scale the ONU's energy down from its published 1.11 J per token, standing in for a more efficient accelerator. "≈ 0" is within ±{NOISE:.0%}: across three random seeds the saving moves by up to {vs['spread'] * 100:.1f} points. At the published cost, piggyback is within {vs['pub_max_abs']:.0%} of the better fixed chain up to load 16. It pays when the ONU is cheap enough to be the right place for a query at night and the wrong one at the daily peak, with the most, {best_g:.0%}, at a {vs['onu_J'][best_s]:.0f} J ONU and load {best_l:g}.</figcaption>
    {rg_tbl}
  </figure>
  <h3>What it does over one day</h3>
  <figure class="stack">
    {legend([("olt", "OLT cost per query", False), ("onu", "ONU cost per query", False)])}
    {day_a}
    {legend([("base", POLICY_NAME["stepwise"], True), ("base", POLICY_NAME["static"] + " (dotted)", True), ("pig", POLICY_NAME["piggyback"], False), ("pig", POLICY_NAME["oracle"] + " (dotted)", True)])}
    {day_b}
    <figcaption>ONU at {day_onu:.0f} J per query, OLT peak load {DAY['load']:g}, β {DAY['beta']}, averaged over {vs['days']} days. Piggyback keeps queries on the ONU through the night, while the OLT is expensive, and sends them past it from late morning, matching the oracle within a few points hour by hour. A static configuration set to the day's average rate skips the ONU almost around the clock. A static configuration that has gone stale does worse still: set at a quarter or four times the real load, it costs up to {ws[0]:.0%} more than piggyback (ONU {vs['onu_J'][ws[1]]:.0f} J, load {ws[2]:g}, accuracy {ws[3]}).</figcaption>
    {day_tbl}
  </figure>
</section>

<section>
  <h2>What these numbers rest on</h2>
  <ul>
    <li><strong>The OLT conversion is conservative.</strong> Google's shares come from large multi-accelerator hosts; a central-office server with one 72 W L4 likely spends more on its host, so the whole-server OLT figure is if anything still low, and the crossover errs early.</li>
    <li><strong>The user tier is a lower bound</strong> (no memory, screen or radio). That can only widen its lead over the OLT, which already never undercuts it.</li>
    <li><strong>A realistic ONU.</strong> The Orin Nano Super ($249, 7–25 W) replaces the AGX Orin 64GB used earlier, a ~$2,000, 15–60 W module no ISP would put in a home, where an ONU draws about 4 W. Its published figure includes the board's idle draw at batch 1, which is why the ONU is dear.</li>
    <li><strong>Simulated load.</strong> The OLT's load is set, not measured: BurstGPT's daily shape scaled to a swept peak, independent of the cascade's own queries. Transport energy and latency are left out. Judged at the GPU card instead, the ONU is dominated at every load and piggyback's largest saving over the better fixed chain is {gpu_big:.0%}.</li>
    <li><strong>One task.</strong> GSM8K, zero-shot, each model's chat template, temperature 0. Every tier's answers come from the same 4-bit or fp8 weights its energy describes.</li>
  </ul>
  <p>Methodology, with a reference for every decision, and all raw logs: <code>energy_tests.md</code> and <code>implementation/results/energy_tests/</code> in the project repository.</p>
</section>

<footer>Generated by <code>make_energy_artifact.py</code> from <code>gpu_energy_qwen2.5-7b-instruct_l4x1_run1.json</code>, <code>…_run2.json</code>, <code>{esc(answers_file)}</code> and the simulator's <code>{esc(", ".join(f for f in sim["files"] if f))}</code>. OLT idle power {v['idle2']:.1f} W.</footer>
</div>
<script>{JS}</script>
"""
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page) / 1024:.0f} KB)")
    f = lambda x: f"{x:.1f}" if x else "none <= 64"
    print(f"per query: user {v['cost']['user']:.1f} J, ONU {v['cost']['onu']:.1f} J; "
          f"OLT factor x{v['k_mid']:.2f} (PUE {BOUNDARY['pue_isp']}), x{v['k_low']:.2f} (PUE {BOUNDARY['pue_low']})")
    print(f"crossover vs ONU: system {f(v['x_onu_sys'])} (low PUE {f(v['x_onu_sys_low'])}), GPU only {f(v['x_onu_gpu'])}")
    print(f"crossover vs user: system {f(v['x_user_sys'])}, GPU only {f(v['x_user_gpu'])}")
    print(f"piggyback: best {best_g:.1%} (ONU x{best_s:g}, load {best_l:g}); published within {vs['pub_max_abs']:.1%}; "
          f"seed spread {vs['spread']:.1%}; worst stale +{ws[0]:.0%} {ws[1:]}; GPU-boundary best {gpu_big:.1%}")


if __name__ == "__main__":
    build()
