#!/usr/bin/env python3
"""Collect the generative cascade's answer matrix: every tier, every query.

Phase 1 of the generative experiment -- the analogue of run_policy_matrix.py for
the classification harness, and the only step that costs money or GPU time.
Everything downstream (sweep_energy_policy.py) replays this file offline.

DESIGN PRINCIPLE: COLLECT MAXIMALLY, DERIVE LATER
-------------------------------------------------
This run is expensive and, for the cloud tier, billed. It is written so it never
has to be repeated: the raw generation text and the full per-token logprob vector
are stored for every (tier, query), not just the derived confidence. Any future
change -- a different confidence definition, a new metric, a reviewer's question
about token distributions -- is then answered offline from this file. Temperature
is 0 so the run is reproducible rather than a one-off sample.

RESUMABILITY
------------
Records are appended one (tier, query) pair at a time and completed pairs are
skipped on restart. A crash at query 900 of 1319 costs nothing already spent.
This matters more than it looks: the cloud tier is billed per call, and a
non-resumable script turns any transient 500 into a repeat charge.

TIER-OUTER ORDERING
-------------------
The loop is tier-outer, query-inner -- the opposite of the classification
version. Ollama keeps one model resident and evicts it when another is
requested, so a query-outer loop would reload a multi-GB model on every single
query. Tier-outer loads each model once.

WHY EVERY TIER RUNS EVERY QUERY
-------------------------------
The escalation decision is path-dependent: lambda changes whether a query
escalates, which changes which tiers it visits, which changes each tier's
confidence history, which changes its future thresholds. A trace of one cascade
run cannot answer "what would the cloud tier have said?" for a query that
stopped at user. Running every tier once removes that gap, and is exact rather
than approximate, since a tier's output depends only on the prompt.

Example:
    # 1. smoke test -- confirm logprobs come back and perplexity looks sane
    python src/scripts/run_generative_matrix.py --limit 20 --tiers user

    # 2. full local tiers (free), then the billed cloud tier
    python src/scripts/run_generative_matrix.py --limit 500 --tiers user,onu,fog
    export LLAMA_API_KEY=...
    python src/scripts/run_generative_matrix.py --limit 500 --tiers cloud
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from layers.ollama_layer import OllamaGenerativeLayer  # noqa: E402
from tasks.gsm8k import build_prompt, is_correct, load_gsm8k  # noqa: E402

TIERS = ("user", "onu", "fog", "cloud")

# Default ladder. Every model here has published energy for that exact model in
# config/layer_energy.yaml, on hardware of the right class for its tier -- which
# is the whole point: confidence and energy then refer to the same models
# instead of being borrowed from unrelated ones.
DEFAULT_TIER_CONFIG = {
    "user":  {"model": "llama3.2:1b",  "backend": "ollama"},   # 0.074 J/tok, Snapdragon W4
    "onu":   {"model": "llama3.1:8b",  "backend": "ollama"},   # 0.98  J/tok, Jetson Orin W4
    "fog":   {"model": "solar:10.7b",  "backend": "ollama"},   # 0.3785 J/tok, A30
    "cloud": {                                                  # 0.3989 J/tok, 4xH100 BF16
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "backend": "openai",
        "base_url": "https://api.together.xyz",
        "api_key_env": "LLAMA_API_KEY",
    },
}


def build_layer(tier: str, cfg: dict, max_tokens: int, temperature: float) -> OllamaGenerativeLayer:
    if cfg["backend"] == "openai":
        api_key = os.environ.get(cfg.get("api_key_env", "LLAMA_API_KEY"))
        if not api_key:
            raise SystemExit(
                f"tier {tier!r} needs an API key: export {cfg.get('api_key_env', 'LLAMA_API_KEY')}=...\n"
                "Use a provider serving an OPEN Llama-70B that exposes logprobs "
                "(Together / Fireworks / DeepInfra) -- verify the logprobs field before a full run."
            )
        return OllamaGenerativeLayer(
            layer_name=tier, model=cfg["model"], base_url=cfg["base_url"],
            api_key=api_key, openai_compatible=True,
            max_tokens=max_tokens, temperature=temperature,
        )
    return OllamaGenerativeLayer(
        layer_name=tier, model=cfg["model"],
        base_url=cfg.get("base_url", "http://localhost:11434"),
        max_tokens=max_tokens, temperature=temperature,
    )


def load_done(path: Path) -> set[tuple[str, int]]:
    """(tier, index) pairs already collected, so a restart skips them."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            done.add((rec["tier"], rec["index"]))
        except (json.JSONDecodeError, KeyError):
            continue  # tolerate a torn final line from a hard kill
    return done


def consolidate(raw_path: Path, out_path: Path, tiers: list[str]) -> int:
    """Pivot flat (tier, query) records into the per-query matrix the sweep reads."""
    by_index: dict[int, dict] = {}
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line from a hard kill -- same tolerance as load_done
        entry = by_index.setdefault(rec["index"], {
            "index": rec["index"], "dataset": rec["dataset"], "split": rec["split"],
            "true_label": rec["reference_answer"], "difficulty_steps": rec["difficulty_steps"],
            "tiers": {},
        })
        entry["tiers"][rec["tier"]] = {
            "model_name": rec["model"], "confidence": rec["confidence"],
            "correct": rec["correct"], "tokens_prompt": rec["tokens_prompt"],
            "tokens_gen": rec["tokens_gen"], "latency_s": rec["latency_s"],
        }
    complete = [e for e in by_index.values() if all(t in e["tiers"] for t in tiers)]
    complete.sort(key=lambda e: e["index"])
    with open(out_path, "w") as f:
        for entry in complete:
            f.write(json.dumps(entry) + "\n")
    return len(complete)


def generate_all(layer, todo: list, concurrency: int):
    """Yield (index, item, result_or_None) for every query, in completion order.

    concurrency=1 is the original sequential path and stays the default: the local
    Ollama tiers keep ONE model resident, so parallel requests queue behind each
    other and buy nothing.

    Concurrency only pays against a rented GPU, where billing is per SECOND rather
    than per token: vLLM batches the in-flight requests, so 200 concurrent calls
    finish in roughly the time 200 sequential ones spend on their first few. That
    inverts the original design assumption (written against a per-token API, where
    concurrency saved nothing and only risked rate limits).

    Ordering is by completion, not by index -- which is fine because every record
    carries its own index, and consolidate() pivots by index rather than by
    position. Writes stay in the caller's single thread, so resumability is
    unaffected: still one flushed record at a time.
    """
    def run_one(pair):
        index, item = pair
        try:
            return index, item, layer.generate(build_prompt(item.question))
        except Exception as exc:  # noqa: BLE001 - one bad query must not end the run
            print(f"  [{index}] FAILED: {type(exc).__name__}: {exc}")
            return index, item, None

    if concurrency <= 1:
        for pair in todo:
            yield run_one(pair)
        return

    # ThreadPoolExecutor rather than asyncio: OllamaGenerativeLayer calls
    # requests.post, which releases the GIL while blocking on the socket, and is
    # stateless per call -- so one layer instance is safely shared across threads.
    #
    # as_completed rather than pool.map: map yields in SUBMISSION order, so one
    # slow query would block every finished result behind it and a hard kill would
    # lose them. On a per-second-billed GPU that is real money, so results are
    # persisted the moment they land.
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, pair) for pair in todo]
        for future in as_completed(futures):
            yield future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200, help="queries (0 = full split)")
    parser.add_argument("--tiers", default="user,onu,fog,cloud",
                        help="comma-separated subset to collect (local tiers are free; cloud is billed)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 keeps the run reproducible; EcoThink uses 0 for math tasks too")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="concurrent requests for remote (openai) tiers; local tiers stay at 1. "
                             "Against a per-second-billed GPU this is the main cost lever -- vLLM "
                             "batches the in-flight requests, so 32 cuts wall time (and the bill) "
                             "several-fold. Against a per-token API it saves nothing.")
    parser.add_argument("--config", type=Path, default=None, help="JSON overriding the tier ladder")
    parser.add_argument("--raw-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="show the plan and a rough token estimate, call nothing")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in TIERS]
    if unknown:
        raise SystemExit(f"unknown tier(s) {unknown}; valid: {list(TIERS)}")

    tier_config = dict(DEFAULT_TIER_CONFIG)
    if args.config:
        tier_config.update(json.loads(args.config.read_text()))

    raw_path = args.raw_out or ROOT / "results" / "traces" / "gsm8k_generative.raw.jsonl"
    out_path = args.out or ROOT / "results" / "traces" / "gsm8k_generative.matrix.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading GSM8K/{args.split} ...")
    items = load_gsm8k(args.split, limit=args.limit or None)
    print(f"  {len(items)} items")

    if args.dry_run:
        avg_prompt_words = sum(len(build_prompt(i.question).split()) for i in items) / len(items)
        est_in = int(avg_prompt_words * 1.3)
        print(f"\nPlan: {len(items)} queries x {len(tiers)} tiers = {len(items) * len(tiers)} generations")
        print(f"  ~{est_in} input tokens/query, up to {args.max_tokens} output tokens/query")
        for tier in tiers:
            cfg = tier_config[tier]
            billed = cfg["backend"] == "openai"
            print(f"  {tier:>5}: {cfg['model']:<45} {'BILLED (API)' if billed else 'local'}")
        billed_tiers = [t for t in tiers if tier_config[t]["backend"] == "openai"]
        if billed_tiers:
            out_est = len(items) * 300  # CoT answers land well under max_tokens in practice
            print(f"\n  Rough cost for {billed_tiers}: ~{len(items) * est_in / 1e6:.3f}M in + "
                  f"~{out_est / 1e6:.3f}M out tokens")
            print("  At ~$0.60-0.90 per M output tokens that is well under $1 for a run this size.")
        print("\n(dry run -- nothing called)")
        return

    done = load_done(raw_path)
    if done:
        print(f"Resuming: {len(done)} (tier, query) pairs already collected")

    started = time.perf_counter()
    with open(raw_path, "a") as f:
        for tier in tiers:  # tier-outer: keeps one model resident per pass
            cfg = tier_config[tier]
            todo = [(i, it) for i, it in enumerate(items) if (tier, i) not in done]
            if not todo:
                print(f"\n{tier}: already complete, skipping")
                continue
            print(f"\n{tier}: {cfg['model']} ({cfg['backend']}) -- {len(todo)} to go")
            layer = build_layer(tier, cfg, args.max_tokens, args.temperature)

            # Concurrency is per-tier: only the remote (openai) tiers benefit, and
            # forcing 1 for local Ollama avoids thrashing its single resident model.
            tier_concurrency = args.concurrency if cfg["backend"] == "openai" else 1
            if tier_concurrency > 1:
                print(f"  sending up to {tier_concurrency} concurrent requests")

            n_correct = n_ok = 0
            for n, (index, item, result) in enumerate(
                generate_all(layer, todo, tier_concurrency), start=1
            ):
                if result is None:
                    continue  # already reported by run_one
                n_ok += 1

                correct = is_correct(result.text, item.reference_answer)
                n_correct += int(correct)
                f.write(json.dumps({
                    "dataset": "gsm8k", "split": args.split, "index": index, "tier": tier,
                    "model": cfg["model"],
                    "question": item.question,
                    "reference_answer": item.reference_answer,
                    "difficulty_steps": item.difficulty_steps,
                    # collect-maximally: raw text and the full logprob vector, so any
                    # future confidence definition is recomputable without re-running
                    "generated_text": result.text,
                    "logprobs": result.logprobs,
                    "tokens": result.tokens,
                    "confidence": result.confidence,
                    "correct": correct,
                    "tokens_prompt": result.tokens_prompt,
                    "tokens_gen": result.tokens_gen,
                    "latency_s": result.decode_latency_s,
                }) + "\n")
                f.flush()  # survive a hard kill without losing the last calls

                if n % 10 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}  running accuracy {n_correct / n_ok:.3f}")

    n_rows = consolidate(raw_path, out_path, tiers)
    print(f"\nElapsed: {time.perf_counter() - started:.1f}s")
    print(f"Raw records:      {raw_path}")
    print(f"Matrix ({n_rows} complete queries): {out_path}")
    if n_rows:
        print(f"\nNext: python src/scripts/sweep_energy_policy.py {out_path}")
    else:
        print("\nNo query has all requested tiers yet -- collect the remaining tiers first.")


if __name__ == "__main__":
    main()
