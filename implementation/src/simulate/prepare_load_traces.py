#!/usr/bin/env python3
"""Hourly LLM request counts from public traces: the OLT's load for simulate.py --load-trace.

BurstGPT [12] (61 days of Azure OpenAI traffic; HPMLL/BurstGPT v1.1, BurstGPT_1.csv) is counted
separately for its two log types and in total:

    conversation   'Conversation log': people chatting with the model
    api            'API log': programs calling the API, much of it batch jobs
    all            both; the daily shape the simulation used until now (89% API)

A household's queries are people chatting, so `conversation` is the case study's load;
`all` is kept as a burstier stress case.

The Azure LLM inference trace 2024 (conversation service, one week, UTC) is a second,
independent source. It is too short to calibrate a schedule on, so it only cross-checks the
drift fitted on BurstGPT.

Drift is the log of the hourly count over what a schedule predicts for that hour, for two
schedules: hour of day, and hour of day x weekday/weekend (the two quietest days of the week).
It is reported as a log-sd net of Poisson counting noise (sigma) and the lag at which its
autocorrelation falls below 1/e (tau), in-sample over all days and out-of-sample the way the
simulation meets it: schedule fitted on the first --train-days, drift measured on the rest.

Usage:
    python src/simulate/prepare_load_traces.py            # BurstGPT only
    python src/simulate/prepare_load_traces.py --azure    # + Azure 2024 conversation (streams 1.1 GB)
Writes data/load_traces/burstgpt_hourly.csv, azure2024_conv_hourly.csv and drift_fit.json.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # implementation/
CACHE = ROOT / ".cache" / "load_traces"
OUT = ROOT / "data" / "load_traces"
BURSTGPT_URL = "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv"
AZURE_URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/"
             "AzureLLMInferenceTrace_conv_1week.csv")
BURSTGPT_DAYS = 61


def burstgpt_hourly() -> pd.DataFrame:
    """BurstGPT's requests per hour, 61 days, for conversation, api and all traffic.

    Downloads BurstGPT_1.csv once into implementation/.cache/load_traces/.
    """
    raw = CACHE / "BurstGPT_1.csv"
    if not raw.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"downloading {BURSTGPT_URL}", file=sys.stderr)
        urllib.request.urlretrieve(BURSTGPT_URL, raw)
    df = pd.read_csv(raw, usecols=["Timestamp", "Log Type"])
    n = BURSTGPT_DAYS * 24
    cols = {}
    for name, sub in (("conversation", df[df["Log Type"] == "Conversation log"]),
                      ("api", df[df["Log Type"] == "API log"]), ("all", df)):
        h = (sub["Timestamp"].to_numpy() // 3600).astype(int)
        cols[name] = np.bincount(h[h < n], minlength=n)
    idx = np.arange(n)
    return pd.DataFrame({"day": idx // 24, "hour": idx % 24, **cols})


def azure_hourly() -> pd.DataFrame:
    """Stream the 1.1 GB file once, counting requests per UTC hour; keep whole days only."""
    counts: collections.Counter = collections.Counter()
    print(f"streaming {AZURE_URL}", file=sys.stderr)
    with urllib.request.urlopen(AZURE_URL) as f:
        next(f)                                           # header
        for line in f:
            counts[line[:13]] += 1                        # b"YYYY-MM-DD HH"
    hours = {dt.datetime.strptime(k.decode(), "%Y-%m-%d %H"): v for k, v in counts.items()}
    first = min(hours).replace(hour=0) + dt.timedelta(days=1) if min(hours).hour else min(hours)
    last = max(hours).replace(hour=0)                      # exclusive: the last partial day is dropped
    rows, t = [], first
    while t < last:
        rows.append({"day": (t - first).days, "hour": t.hour, "date": t.date().isoformat(),
                     "conversation": hours.get(t, 0)})
        t += dt.timedelta(hours=1)
    return pd.DataFrame(rows)


def schedule(c: np.ndarray, days: np.ndarray, weekend: set | None) -> np.ndarray:
    """Mean count per hour of day (per day type if weekend is given), from the rows `days`."""
    if weekend is None:
        return np.broadcast_to(c[days].mean(axis=0), c.shape)
    we = np.array([d % 7 in weekend for d in range(len(c))])
    sel_we, sel_wd = [d for d in days if we[d]], [d for d in days if not we[d]]
    return np.where(we[:, None], c[sel_we].mean(axis=0), c[sel_wd].mean(axis=0))


def drift(c: np.ndarray, pred: np.ndarray, rows: np.ndarray) -> dict:
    """sigma (log-sd net of Poisson noise) and tau (ACF < 1/e) of log(count / schedule)."""
    cc, pp = np.maximum(c[rows], 0.5), np.maximum(pred[rows], 0.5)
    r = np.log(cc / pp).ravel()
    poisson = np.mean(1.0 / np.maximum(c[rows].ravel(), 1))
    x = r - r.mean()
    acf = [float(np.sum(x[:-k] * x[k:]) / np.sum(x * x)) for k in range(1, 97)]
    tau = next((k + 1 for k, a in enumerate(acf) if a < np.exp(-1)), None)
    q = np.exp(r)
    return {"sigma": float(np.sqrt(max(r.var() - poisson, 0.0))), "sigma_raw": float(r.std()),
            "tau_h": tau, "acf": {str(k): round(acf[k - 1], 3) for k in (1, 3, 6, 12, 24, 48)},
            "load_over_schedule_p10_p50_p90": [round(float(v), 3) for v in np.percentile(q, [10, 50, 90])]}


def fit(c: np.ndarray, train_days: int, weekly: bool) -> dict:
    """Describe one trace: size, daily swing, and the drift around a schedule (in and out of sample)."""
    days = len(c)
    daily = c.sum(axis=1)
    weekend = None
    if weekly:
        dow = np.array([daily[d::7].mean() for d in range(7)])
        weekend = set(int(d) for d in np.argsort(dow)[:2])
    shape = c.mean(axis=0)
    out = {"days": days, "mean_per_hour": float(c.mean()), "daily_cv": float(daily.std() / daily.mean()),
           "peak_over_trough": float(shape.max() / max(shape.min(), 1e-9)),
           "weekend_days_mod7": sorted(weekend) if weekend else None}
    allrows = np.arange(days)
    for name, we in (("hour_of_day", None), ("hour_x_daytype", weekend)):
        if name == "hour_x_daytype" and not weekly:
            continue
        out[name] = {"in_sample": drift(c, schedule(c, allrows, we), allrows)}
        if train_days and train_days < days:
            tr, te = allrows[:train_days], allrows[train_days:]
            out[name]["out_of_sample"] = drift(c, schedule(c, tr, we), te)
    return out


def main() -> int:
    """Write data/load_traces/ (hourly counts and drift_fit.json) and print a summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--azure", action="store_true", help="also stream and count the Azure 2024 conversation trace")
    ap.add_argument("--train-days", type=int, default=30, help="BurstGPT days a schedule is fitted on")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    b = burstgpt_hourly()
    b.to_csv(OUT / "burstgpt_hourly.csv", index=False)
    result = {"burstgpt": {"source": BURSTGPT_URL, "train_days": args.train_days}}
    for col in ("conversation", "api", "all"):
        c = b[col].to_numpy().reshape(-1, 24).astype(float)
        result["burstgpt"][col] = fit(c, args.train_days, weekly=True)

    if args.azure:
        az = OUT / "azure2024_conv_hourly.csv"
        if az.exists():                                   # the 1.1 GB stream is only needed once
            a = pd.read_csv(az)
        else:
            a = azure_hourly()
            a.to_csv(az, index=False)
        c = a["conversation"].to_numpy().reshape(-1, 24).astype(float)
        result["azure2024_conv"] = {"source": AZURE_URL, "first_day_utc": a["date"].iloc[0],
                                    **fit(c, 0, weekly=False)}

    with open(OUT / "drift_fit.json", "w") as f:
        json.dump(result, f, indent=1)
    for src, r in result.items():
        for col, v in r.items():
            if not isinstance(v, dict) or "days" not in v:
                continue
            print(f"{src} {col}: {v['days']} days, {v['mean_per_hour']:.0f}/h, peak/trough {v['peak_over_trough']:.1f}, "
                  f"daily cv {v['daily_cv']:.2f}")
            for sched in ("hour_of_day", "hour_x_daytype"):
                for k, d in v.get(sched, {}).items():
                    print(f"   {sched:15s} {k:13s} sigma {d['sigma']:.2f}  tau {d['tau_h']} h  "
                          f"ACF 1/6/24 h {d['acf']['1']:.2f} {d['acf']['6']:.2f} {d['acf']['24']:.2f}  "
                          f"load/schedule p10/p50/p90 {d['load_over_schedule_p10_p50_p90']}")
        if isinstance(r.get("days"), int):
            v = r
            print(f"{src}: {v['days']} days, {v['mean_per_hour']:.0f}/h, peak/trough {v['peak_over_trough']:.1f}, "
                  f"daily cv {v['daily_cv']:.2f}")
            d = v["hour_of_day"]["in_sample"]
            print(f"   hour_of_day     in_sample     sigma {d['sigma']:.2f}  tau {d['tau_h']} h  "
                  f"ACF 1/6/24 h {d['acf']['1']:.2f} {d['acf']['6']:.2f} {d['acf']['24']:.2f}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
