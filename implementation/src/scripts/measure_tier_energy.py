#!/usr/bin/env python3
"""Measure the REAL energy each cascade tier draws on this machine, via Intel RAPL.

Why this exists
---------------
Every J/token value in config/layer_energy.yaml was measured by someone else,
on other hardware, for decoder LLMs of 1B-405B parameters. The models this
repo actually runs are 66M-400M encoder classifiers. compute_energy_report.py
and sweep_energy_policy.py therefore price a classifier's forward pass as if
its prompt tokens were decode tokens on an unrelated large model, and label
that a smoke test -- explicitly not a thesis result.

This script removes that borrowing for the tiers that actually run here: it
measures the energy of the real models, on the real CPU, with a real counter.
The result is a genuinely measured 4-tier energy profile -- of small models,
but really measured -- which the sweep can use instead of the proxy.

Method
------
RAPL exposes a monotonically increasing microjoule counter per power domain
(package, core, uncore, psys). Energy for an interval = counter delta, with
wraparound handled via max_energy_range_uj.

Two numbers are reported per tier, because both are defensible and they
answer different questions:

  total  -- all package energy during inference. What the socket actually
            drew, including idle/background draw.
  net    -- total minus (measured idle power x elapsed time). Attempts to
            isolate the marginal cost of the inference itself.

Neither is "the" answer: `total` overstates the model's own cost on a mostly
idle machine, `net` understates it if the model's work also raises the
components being subtracted. Both are recorded.

Caveats that must travel with any number this produces
------------------------------------------------------
- Package-scope, not per-process: any other load on this socket is counted.
  Run on an otherwise quiet machine.
- CPU inference only. Says nothing about the GPU/NPU/accelerator deployments
  layer_energy.yaml describes.
- Prompt tokens, single forward pass, no generation -- so the unit here is
  J per inference and J per PROMPT token, NOT the J per OUTPUT token that
  layer_energy.yaml's decode-phase numbers use. Do not mix the two.

Example:
    python src/scripts/measure_tier_energy.py --limit 40 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "recserve" / "vendor"))

from recserve.run_classification_cascade import (  # noqa: E402
    CLOUD_MODEL,
    FOG_MODEL,
    ONU_MODEL,
    USER_MODEL,
)
from recserve.traced_recursive_serve import TracedRecursiveServe  # noqa: E402
from utils import load_sentiment_dataset  # noqa: E402  (vendored RecServe module)

TIERS = ("user", "onu", "fog", "cloud")
POWERCAP = Path("/sys/class/powercap")


class RaplDomain:
    """One RAPL power domain, read through sysfs."""

    def __init__(self, path: Path):
        self.path = path
        self.name = (path / "name").read_text().strip()
        self.max_uj = int((path / "max_energy_range_uj").read_text().strip())

    def read_uj(self) -> int:
        return int((self.path / "energy_uj").read_text().strip())

    def delta_j(self, before: int, after: int) -> float:
        """Counter delta in joules, correcting for wraparound."""
        raw = after - before
        if raw < 0:
            raw += self.max_uj
        return raw / 1e6


def discover_domains() -> list[RaplDomain]:
    """Top-level readable RAPL domains (package-N, psys). Subdomains such as
    core/uncore are nested inside package and would double-count."""
    found: list[RaplDomain] = []
    for entry in sorted(POWERCAP.glob("intel-rapl:*")):
        if entry.name.count(":") != 1 or not (entry / "energy_uj").exists():
            continue
        try:
            domain = RaplDomain(entry)
            domain.read_uj()
        except (PermissionError, OSError):
            continue
        found.append(domain)
    return found


def measure_idle_w(domains: list[RaplDomain], seconds: float) -> dict[str, float]:
    """Baseline power with this process doing nothing."""
    before = {d.name: d.read_uj() for d in domains}
    start = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - start
    return {d.name: d.delta_j(before[d.name], d.read_uj()) / elapsed for d in domains}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="sst2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40, help="queries per measured pass")
    parser.add_argument("--repeats", type=int, default=3, help="measured passes per tier (reports mean/stdev)")
    parser.add_argument("--warmup", type=int, default=5, help="unmeasured inferences before each pass")
    parser.add_argument("--idle-seconds", type=float, default=5.0, help="idle baseline sampling window")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--out", default=None,
                        help="output JSON (default: results/traces/measured_tier_energy.json)")
    args = parser.parse_args()

    domains = discover_domains()
    if not domains:
        print("ERROR: no readable RAPL domain found under /sys/class/powercap.", file=sys.stderr)
        print("       The counters exist but are root-only by default. Grant read access with:", file=sys.stderr)
        print("       sudo find /sys/class/powercap -name energy_uj -exec chmod a+r {} +", file=sys.stderr)
        raise SystemExit(2)
    print("RAPL domains: " + ", ".join(d.name for d in domains))

    out_path = Path(args.out) if args.out else ROOT / "results" / "traces" / "measured_tier_energy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_sentiment_dataset(args.dataset, args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    texts = [ex["text"] for ex in dataset]

    print(f"Sampling idle baseline for {args.idle_seconds:.0f}s (keep the machine quiet)...")
    idle_w = measure_idle_w(domains, args.idle_seconds)
    for name, watts in idle_w.items():
        print(f"  idle {name}: {watts:.2f} W")

    print("\nLoading pipelines (user/onu/fog/cloud) ...")
    service = TracedRecursiveServe(USER_MODEL, ONU_MODEL, FOG_MODEL, CLOUD_MODEL, device=args.device)
    primary = domains[0].name

    results: dict[str, dict] = {}
    for tier in TIERS:
        passes = []
        for r in range(args.repeats):
            for text in texts[: args.warmup]:
                service.classify_one(tier, text)

            before = {d.name: d.read_uj() for d in domains}
            start = time.perf_counter()
            tokens = 0
            for text in texts:
                service.classify_one(tier, text)
                tokens += service.count_tokens(tier, text)
            elapsed = time.perf_counter() - start
            after = {d.name: d.read_uj() for d in domains}

            per_domain = {}
            for d in domains:
                total_j = d.delta_j(before[d.name], after[d.name])
                net_j = max(0.0, total_j - idle_w[d.name] * elapsed)
                per_domain[d.name] = {
                    "total_J": total_j,
                    "net_J": net_j,
                    "mean_W": total_j / elapsed,
                }
            passes.append({
                "elapsed_s": elapsed,
                "n_queries": len(texts),
                "tokens_prompt_total": tokens,
                "domains": per_domain,
            })
            print(f"  {tier:>5} pass {r + 1}/{args.repeats}: "
                  f"{per_domain[primary]['total_J']:.2f} J total, "
                  f"{per_domain[primary]['net_J']:.2f} J net, "
                  f"{elapsed:.2f}s, {tokens} tokens")

        net = [p["domains"][primary]["net_J"] for p in passes]
        total = [p["domains"][primary]["total_J"] for p in passes]
        toks = passes[0]["tokens_prompt_total"]
        n = passes[0]["n_queries"]
        results[tier] = {
            "model": service.model_names[tier],
            "passes": passes,
            "primary_domain": primary,
            "net_J_per_inference": statistics.mean(net) / n,
            "net_J_per_prompt_token": statistics.mean(net) / toks,
            "total_J_per_inference": statistics.mean(total) / n,
            "total_J_per_prompt_token": statistics.mean(total) / toks,
            "net_J_stdev_across_passes": statistics.stdev(net) if len(net) > 1 else 0.0,
        }

    payload = {
        "method": "intel-rapl sysfs counter delta, top-level domain scope, CPU inference",
        "idle_W": idle_w,
        "device": args.device,
        "dataset": f"{args.dataset}/{args.split}",
        "queries_per_pass": len(texts),
        "repeats": args.repeats,
        "unit_note": ("J per PROMPT token (single forward pass, no generation). NOT comparable "
                      "to layer_energy.yaml's J per OUTPUT token from decode-phase measurements."),
        "tiers": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"\nWrote: {out_path}\n")
    header = f"{'tier':<8}{'model':<46}{'J/infer(net)':>14}{'J/token(net)':>14}{'stdev':>9}"
    print(header)
    print("-" * len(header))
    for tier in TIERS:
        r = results[tier]
        print(f"{tier:<8}{r['model'][:44]:<46}{r['net_J_per_inference']:>14.4f}"
              f"{r['net_J_per_prompt_token']:>14.5f}{r['net_J_stdev_across_passes']:>9.2f}")
    print("\nCaveats: top-level RAPL scope (counts any other load on this socket), CPU-only,")
    print("         per-PROMPT-token on a single forward pass. See module docstring.")


if __name__ == "__main__":
    main()
