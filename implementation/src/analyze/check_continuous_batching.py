#!/usr/bin/env python3
"""Check the simulator's batching against the GPU measured under continuous batching.

Applies the criterion of energy_tests.md §3.5, fixed before the data: the simulator's OLT,
replaying each measured Poisson run, must predict its energy and mean latency within ±10%,
and every fixed-concurrency median must be steady. Writes results/continuous_batching.md.

Usage: python src/analyze/check_continuous_batching.py [results/measurements/gpu_energy_continuous_*.json]
"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # implementation/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulate"))
from energy.three_tier import olt_reference, OltCurve, RESULTS, ROOT
from olt_energy import Energy
from olt_server import OltServer

OUT = ROOT / "results" / "continuous_batching.md"
"""Where the tables are written."""

TOLERANCE = 0.10
"""The criterion's band: simulated within ±10% of measured."""


# ==========================================
# Replaying a measured run
# ==========================================
def curve_at_idle(idle_watts: float | None) -> OltCurve:
    """Return the static curve at the card boundary, its net rates taken against this idle.

    The gross curve does not depend on the machine: the L4 runs at its power limit whatever
    its idle draw. The net-of-idle rates do, so a run on another card is compared with net
    rates rebuilt from the same gross curve minus that card's idle power.

    Args:
        idle_watts: The idle power to subtract, or None for the sweep's own.
    """
    reference = olt_reference()
    rows = [dict(row) for row in reference["batch_curve"]]
    if idle_watts is not None:
        # seconds per token, as the sweep's own gross and net figures imply them
        sweep_idle = reference["idle_power_W"]
        for row in rows:
            for gross, net in (
                ("prefill_J_per_input_token", "prefill_J_per_input_token_net"),
                ("decode_J_per_output_token", "decode_J_per_output_token_net"),
            ):
                seconds_per_token = (row[gross] - row[net]) / sweep_idle
                row[net] = row[gross] - idle_watts * seconds_per_token
    return OltCurve(rows)


def replay(
    requests: list[dict], accounting: str, idle_watts: float | None = None
) -> tuple[float, list[float], float]:
    """Run a measured Poisson run's requests through the simulator's OLT at the card boundary.

    Args:
        requests: The run's records: arrival, prompt and generated tokens.
        accounting: "average" or "marginal".
        idle_watts: The idle power the net rates are taken against (None: the sweep's).

    Returns:
        (joules charged to the requests, each request's latency in seconds, seconds the
        OLT spent generating).
    """
    curve = curve_at_idle(idle_watts)
    server = OltServer(curve, Energy(curve, {}, accounting))
    arrivals = sorted(enumerate(requests), key=lambda item: item[1]["arrival_s"])
    finished = []
    for query, request in arrivals:
        finished += server.advance(request["arrival_s"])
        server.admit(query, request["prompt_tokens"], request["generated_tokens"])
    finished += server.drain()
    joules = sum(prefill + decode for _, _, prefill, decode in finished)
    done_at = {query: clock for query, clock, _, _ in finished}
    latencies = [done_at[query] - request["arrival_s"] for query, request in arrivals]
    return joules, latencies, server._busy_s  # pylint: disable=protected-access


def compare_run(run: dict, idle_watts: float) -> dict:
    """Return one Poisson run's measured and simulated energy and latency.

    Under marginal accounting both sides are net of the measured card's idle power (and,
    as first written, the simulator net of the sweep card's). Under average accounting the
    simulated side adds the idle power over the time it predicts the OLT sat idle, so both
    sides are gross.

    Args:
        run: One Poisson run of the report.
        idle_watts: The idle power measured in the same container.
    """
    requests = run["requests"]
    measured_latency = statistics.mean(r["finish_s"] - r["arrival_s"] for r in requests)
    net_joules, latencies, _ = replay(requests, "marginal", idle_watts)
    as_written, _, _ = replay(requests, "marginal")
    busy_joules, _, busy_s = replay(requests, "average")
    gross_joules = busy_joules + idle_watts * max(run["seconds"] - busy_s, 0.0)
    simulated_latency = statistics.mean(latencies)
    return {
        "load": run["target_load"],
        "seed": run.get("seed"),
        "requests": len(requests),
        "in_flight": run["mean_in_flight"],
        "net": (run["joules_net"], net_joules),
        "net_as_written": (run["joules_net"], as_written),
        "gross": (run["joules"], gross_joules),
        "latency": (measured_latency, simulated_latency),
    }


def off_by(pair: tuple[float, float]) -> float:
    """Return how far the simulated value is from the measured one (0.05 = 5% above)."""
    measured, simulated = pair
    return simulated / measured - 1


# ==========================================
# Tables
# ==========================================
def table(headers: list[str], rows: list[list]) -> list[str]:
    """Return a markdown table, as a list of lines."""
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(str(cell) for cell in row) + " |" for row in rows
    ]


def main() -> int:
    """Apply the criterion to a continuous-batching report and write the tables.

    Returns:
        The process exit code: 0 if the criterion holds, 1 if it fails, 2 without a report.
    """
    reports = sorted(RESULTS.glob("gpu_energy_continuous_*.json"))
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (reports[-1] if reports else None)
    if path is None:
        print(__doc__)
        return 2
    with open(path, encoding="utf-8") as file:
        report = json.load(file)
    static = {row["batch"]: row for row in olt_reference()["batch_curve"]}
    lines = [
        "# Continuous batching against the simulator",
        "",
        f"Generated by `src/analyze/check_continuous_batching.py` from `{path.name}`: "
        f"{report['model']} ({report['precision']}) on one {report['gpu']}, idle "
        f"{report['idle_power_W']} W. Criterion: energy_tests.md §3.5.",
        "",
    ]

    # --- Fixed concurrency: steady, and against the static sweep ---
    steady = True
    rows = []
    for row in report["fixed"]:
        trials = row.get("trials_J_per_generated_token", [row["J_per_generated_token"]])
        clocks = [trial.get("min_sm_clock_MHz") for trial in row.get("trials", [])]
        within = min(trials) <= row["J_per_generated_token"] <= max(trials)
        no_drop = len({clock for clock in clocks if clock is not None}) <= 1 or min(
            clocks
        ) >= 0.95 * max(clocks)
        steady &= within and no_drop
        reference = static.get(row["concurrency"])
        rows.append(
            [
                row["concurrency"],
                f"{row['mean_in_flight']:.2f}",
                f"{row['tokens_per_s']:.0f}",
                f"{row['J_per_generated_token']:.4f} [{min(trials):.4f}–{max(trials):.4f}]",
                f"{reference['decode_J_per_output_token']:.4f}" if reference else "–",
                (
                    f"{row['J_per_generated_token'] / reference['decode_J_per_output_token'] - 1:+.1%}"
                    if reference
                    else "–"
                ),
                f"{row['mean_power_W']:.0f}",
                f"{row.get('max_temperature_C', float('nan')):.0f}",
                "yes" if within and no_drop else "**no**",
            ]
        )
    lines += ["## 1. Fixed concurrency against the static sweep", ""]
    lines += table(
        [
            "Requests in flight",
            "Measured in flight",
            "Tokens/s",
            "J/generated token [windows]",
            "Static sweep",
            "Difference",
            "Power W",
            "Max °C",
            "Steady",
        ],
        rows,
    )
    lines += [""]

    # --- Poisson loads: the simulator replaying the same arrivals ---
    passed = True
    rows = []
    for run in report["poisson"]:
        result = compare_run(run, report["idle_power_W"])
        misses = [abs(off_by(result[key])) > TOLERANCE for key in ("net", "gross", "latency")]
        passed &= not any(misses)
        rows.append(
            [
                f"{result['load']:g}",
                result["seed"] if result["seed"] is not None else "–",
                result["requests"],
                f"{result['in_flight']:.2f}",
                f"{result['net'][0]:.0f} / {result['net'][1]:.0f} ({off_by(result['net']):+.1%})",
                f"{off_by(result['net_as_written']):+.1%}",
                f"{result['gross'][0]:.0f} / {result['gross'][1]:.0f} ({off_by(result['gross']):+.1%})",
                f"{result['latency'][0]:.1f} / {result['latency'][1]:.1f} "
                f"({off_by(result['latency']):+.1%})",
                "**no**" if any(misses) else "yes",
            ]
        )
    lines += [
        "## 2. Poisson arrivals: measured / simulated",
        "",
        "Energy net of the measured card's idle under marginal accounting, gross under "
        "average accounting; latency is the mean of finish minus arrival, in seconds. "
        '"As first written" nets the simulator against the sweep card\'s idle instead '
        "(the amendment of energy_tests.md §3.5).",
        "",
    ]
    lines += table(
        [
            "Load",
            "Seed",
            "Requests",
            "In flight",
            "Net J (marginal)",
            "As first written",
            "Gross J (average)",
            "Latency s",
            f"Within ±{TOLERANCE:.0%}",
        ],
        rows,
    )
    lines += [
        "",
        f"**Criterion 1 (the simulator reproduces the GPU): {'met' if passed else 'NOT met'}.** "
        f"**Criterion 2 (the measurement is steady): {'met' if steady else 'NOT met'}.**",
        "",
    ]

    # --- Write ---
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {OUT}", file=sys.stderr)
    return 0 if passed and steady else 1


if __name__ == "__main__":
    raise SystemExit(main())
