#!/usr/bin/env python3
"""Replay the Open LLM Leaderboard's GSM8K prompts through the local tiers.

An alternative phase 1 to run_generative_matrix.py, for the ladder whose cloud
tier comes from published data instead of a model this machine cannot run.
Everything downstream (sweep_energy_policy.py) reads the same matrix format.

WHY REPLAY INSTEAD OF REBUILDING THE PROMPT
-------------------------------------------
The harness samples its 5 few-shot examples per document, not once per run:
all 1319 rows of the GSM8K file carry a *different* prefix. There is no "the
5-shot GSM8K prompt" to reimplement. So this script does not construct
anything -- it reads `full_prompt` from the parquet and sends it verbatim to
every local tier. Pairing is exact by construction and no format drift is
possible.

WHY THE CLOUD TIER NEEDS NO CONFIDENCE
--------------------------------------
The leaderboard's details files carry the prompt, the generation and per-
instance correctness, but `pred_logits` is empty for generative tasks -- there
are no logprobs, so no confidence. That does not block the cloud tier: under
RecServe's rule confidence decides escalate-or-stop, and the top tier is
terminal. It has nowhere to escalate, so its confidence is never thresholded.
Accuracy plus token counts is all the costing and the policy ever ask of it.

Cloud records are therefore written with confidence=None, and any downstream
code that thresholds a tier's confidence must never be pointed at the top
tier. That was already true; here it is also enforced by the data.

THE VALIDATION GATE COMES FIRST
-------------------------------
Meta-Llama-3-8B (base) is in the same leaderboard archive at a published
accuracy. Replaying its prompts locally and reproducing that number validates,
in one shot, the prompt replay, the stop condition, the answer extraction and
the correctness rule. If it does not reproduce, the bug is here -- and finding
it costs one short run instead of the full re-collection of every tier.

Run --validate first. It is not optional in spirit, only in argv.

    python src/scripts/run_leaderboard_matrix.py --validate --limit 200
    python src/scripts/run_leaderboard_matrix.py --tiers user,onu,fog --limit 200
    python src/scripts/run_leaderboard_matrix.py --import-cloud
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from layers.ollama_layer import OllamaGenerativeLayer  # noqa: E402
from tasks.gsm8k import is_correct, load_gsm8k  # noqa: E402

# Leaderboard archive files. Not gated; download with `huggingface-cli download`
# or plain curl. Kept as constants so the provenance of every number that enters
# the matrix is greppable from the code that consumes it.
HF = "https://huggingface.co/datasets/open-llm-leaderboard-old"
SOURCES = {
    "cloud": {
        "repo": "details_meta-llama__Meta-Llama-3-70B-Instruct",
        "file": "2024-04-21T11-59-48.701689/"
                "details_harness|gsm8k|5_2024-04-21T11-59-48.701689.parquet",
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "published_acc": 0.8544,
    },
    "validation": {
        "repo": "details_meta-llama__Meta-Llama-3-8B",
        "file": "2024-04-19T01-26-07.544774/"
                "details_harness|gsm8k|5_2024-04-19T01-26-07.544774.parquet",
        "model": "meta-llama/Meta-Llama-3-8B",
        "ollama_model": "llama3:8b-text-q4_K_M",  # base, not instruct -- must match
        "published_acc": 0.4579,
    },
}

# The harness truncates the continuation at the first of these. Replaying the
# prompt without replaying the stop condition inflates generation length and
# changes accuracy, which is the most likely way the validation gate fails.
STOP = ["Question:", "\n\n"]

# Correctness reuses tasks.gsm8k.is_correct rather than reimplementing the rule,
# so the leaderboard replay and the native runs are scored identically. Whether
# that rule agrees with the harness's own verdict is not assumed -- --check-rule
# measures it against the stored predictions, with no model call.


def load_parquet(path: Path) -> list[dict]:
    """Rows as dicts, in file order -- row position is the pairing index."""
    import pyarrow.parquet as pq

    d = pq.read_table(path).to_pydict()
    rows = []
    for i in range(len(d["example"])):
        pred = d["predictions"][i]
        rows.append({
            "index": i,
            "question": d["example"][i],
            "full_prompt": d["full_prompt"][i],
            "prediction": pred[0] if pred else "",
            "acc": bool(d["metrics"][i]["acc"]),
            "tokens_prompt": len(d["input_tokens"][i][0]) if d["input_tokens"][i] else 0,
            "tokens_gen": len(d["cont_tokens"][i][0]) if d["cont_tokens"][i] else 0,
        })
    return rows


def gold_by_question() -> dict[str, tuple[str, int | None]]:
    """GSM8K's own reference answers, keyed by question text.

    The leaderboard file stores the question and the generation but leaves
    `gold` empty, so references come from the dataset itself. Matching on the
    question string is exact: the harness copies it verbatim.
    """
    out = {}
    for item in load_gsm8k(split="test", limit=None):
        out[item.question.strip()] = (item.reference_answer, item.difficulty_steps)
    return out


def check_rule(rows, golds) -> float:
    """Score the stored generations with our rule and compare to the harness's.

    Runs no model. It isolates one of the two things the validation gate tests:
    if agreement here is low, the correctness rule diverges and no amount of
    correct prompt replay will reproduce the published accuracy. Diagnosing that
    before spending GPU-hours is the entire point of separating the two.
    """
    matched = [r for r in rows if r["question"].strip() in golds]
    print(f"pareamento pergunta->gabarito: {len(matched)}/{len(rows)}")
    agree = sum(1 for r in matched
                if is_correct(r["prediction"], golds[r["question"].strip()][0]) == r["acc"])
    ours = sum(1 for r in matched if is_correct(r["prediction"], golds[r["question"].strip()][0]))
    theirs = sum(1 for r in matched if r["acc"])
    print(f"concordancia com o 'acc' do harness: {agree}/{len(matched)} "
          f"({100 * agree / len(matched):.2f}%)")
    print(f"acuracia pela nossa regra: {ours / len(matched):.4f}")
    print(f"acuracia do harness:       {theirs / len(matched):.4f}")
    return agree / len(matched)


def load_done(path: Path) -> set[tuple[str, int]]:
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
            continue
    return done


def write_record(f, tier, model, row, gold, steps, result, correct):
    f.write(json.dumps({
        "dataset": "gsm8k", "split": "test", "index": row["index"], "tier": tier,
        "model": model,
        "question": row["question"],
        "reference_answer": gold,
        "difficulty_steps": steps,
        "generated_text": result.text,
        "logprobs": result.logprobs,
        "tokens": result.tokens,
        "confidence": result.confidence,
        "correct": correct,
        "tokens_prompt": result.tokens_prompt,
        "tokens_gen": result.tokens_gen,
        "latency_s": result.decode_latency_s,
        "prompt_source": "leaderboard_v1_harness_gsm8k_5shot",
    }) + "\n")
    f.flush()


def replay(rows, layer, tier, model, golds, out_path, done, limit):
    """Send each stored full_prompt to one local tier, verbatim."""
    todo = [r for r in rows[:limit or len(rows)] if (tier, r["index"]) not in done]
    print(f"[{tier}] {model}: {len(todo)} de {min(limit or len(rows), len(rows))} pendentes")
    n = n_correct = 0
    with open(out_path, "a") as f:
        for row in todo:
            gold, steps = golds.get(row["question"].strip(), (None, None))
            if gold is None:
                print(f"  ! índice {row['index']}: pergunta não encontrada no GSM8K, pulando")
                continue
            t0 = time.perf_counter()
            result = None
            for tentativa in range(1, 4):
                try:
                    result = layer.generate(row["full_prompt"], stop=STOP)
                    break
                except Exception as e:                      # noqa: BLE001
                    # Um timeout isolado no meio de 200 consultas nao deve matar a
                    # coleta inteira -- foi assim que a primeira tentativa do fog
                    # morreu na consulta 1, com o modelo ainda carregando do disco.
                    espera = 30 * tentativa
                    print("  ! indice %d tentativa %d/3 falhou (%s); nova em %ds"
                          % (row["index"], tentativa, type(e).__name__, espera))
                    if tentativa == 3:
                        print("  ! indice %d desistindo apos 3 tentativas" % row["index"])
                        break
                    time.sleep(espera)
            if result is None:
                continue
            correct = is_correct(result.text, gold)
            n += 1
            n_correct += int(correct)
            write_record(f, tier, model, row, gold, steps, result, correct)
            if n % 10 == 0 or row is todo[-1]:
                print(f"  {n}/{len(todo)}  acurácia corrente {n_correct / n:.4f}"
                      f"  ({time.perf_counter() - t0:.1f}s/consulta)")
    return n_correct / n if n else 0.0


def import_cloud(rows, golds, out_path, done):
    """Write the cloud tier straight from the leaderboard file.

    No model runs. confidence and logprobs are None because the file has none,
    and the top tier never needs them (see the module docstring).
    """
    src = SOURCES["cloud"]
    n = n_correct = 0
    with open(out_path, "a") as f:
        for row in rows:
            if ("cloud", row["index"]) in done:
                continue
            gold, steps = golds.get(row["question"].strip(), (None, None))
            if gold is None:
                continue
            n += 1
            n_correct += int(row["acc"])
            f.write(json.dumps({
                "dataset": "gsm8k", "split": "test", "index": row["index"],
                "tier": "cloud", "model": src["model"],
                "question": row["question"],
                "reference_answer": gold,
                "difficulty_steps": steps,
                "generated_text": row["prediction"],
                "logprobs": None,   # absent from the source; terminal tier needs none
                "tokens": None,
                "confidence": None,
                "correct": row["acc"],           # the harness's own verdict
                "tokens_prompt": row["tokens_prompt"],
                "tokens_gen": row["tokens_gen"],
                "latency_s": None,               # not measured on this machine
                "prompt_source": "leaderboard_v1_harness_gsm8k_5shot",
                "energy_source": "caravaca_2511.05597_tableIV_4xH100_1.002_J_per_token",
            }) + "\n")
    print(f"[cloud] importadas {n} instâncias, acurácia {n_correct / n:.4f}" if n
          else "[cloud] nada a importar")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-rule", action="store_true",
                   help="score the stored generations with our rule vs the harness's; no model runs")
    p.add_argument("--validate", action="store_true",
                   help="replay Meta-Llama-3-8B and check it reproduces the published accuracy")
    p.add_argument("--tiers", default="", help="comma-separated local tiers to replay")
    p.add_argument("--import-cloud", action="store_true",
                   help="write the cloud tier from the leaderboard file (no model runs)")
    p.add_argument("--limit", type=int, default=200, help="instances (0 = all 1319)")
    p.add_argument("--parquet-dir", type=Path, default=ROOT / "results" / "leaderboard")
    p.add_argument("--config", type=Path, default=None, help="JSON overriding the tier ladder")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--timeout", type=float, default=1800.0,
                   help="timeout HTTP em segundos; um 9B FP16 leva minutos so para carregar")
    p.add_argument("--threads", type=int, default=None,
                   help="threads do Ollama; menor reduz calor (decode e memory-bound, "
                        "entao o custo em tempo e sublinear)")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="max |measured - published| accuracy gap the gate accepts")
    args = p.parse_args()

    out = args.out or ROOT / "results" / "traces" / "gsm8k_leaderboard.raw.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    golds = gold_by_question()

    if args.check_rule:
        path = args.parquet_dir / "validation.parquet"
        if not path.exists():
            raise SystemExit(f"faltando {path} -- veja --validate para o comando de download")
        agreement = check_rule(load_parquet(path), golds)
        if agreement < 0.98:
            raise SystemExit(
                f"\nREGRA DIVERGENTE ({agreement:.2%} de concordancia).\n"
                "Corrija tasks.gsm8k.is_correct antes de rodar --validate: um portao\n"
                "reprovado seria ambiguo entre regra errada e replay errado.")
        print("\nREGRA OK. Siga com --validate.")
        return

    if args.validate:
        src = SOURCES["validation"]
        path = args.parquet_dir / "validation.parquet"
        if not path.exists():
            raise SystemExit(
                f"faltando {path}\nbaixe com:\n"
                f"  mkdir -p {args.parquet_dir} && curl -L \\\n"
                f"    '{HF}/{src['repo']}/resolve/main/{src['file'].replace('|', '%7C')}' \\\n"
                f"    -o {path}")
        rows = load_parquet(path)
        layer = OllamaGenerativeLayer(layer_name="validation", model=src["ollama_model"],
                                      max_tokens=256, temperature=0.0,
                                      timeout_s=args.timeout, num_thread=args.threads)
        acc = replay(rows, layer, "validation", src["ollama_model"], golds,
                     out, done, args.limit)
        gap = abs(acc - src["published_acc"])
        print(f"\nmedida {acc:.4f} · publicada {src['published_acc']:.4f} · Δ {gap:.4f}")
        if gap > args.tolerance:
            raise SystemExit(
                f"\nPORTÃO REPROVADO (Δ {gap:.4f} > {args.tolerance}).\n"
                "O replay do prompt, a condição de parada, a extração da resposta ou\n"
                "o critério de correção divergem do harness. Corrija antes de coletar\n"
                "as outras camadas -- é o que este passo existe para evitar.")
        print("\nPORTÃO APROVADO. O replay reproduz o leaderboard; siga com --tiers.")
        return

    path = args.parquet_dir / "cloud.parquet"
    if not path.exists():
        src = SOURCES["cloud"]
        raise SystemExit(
            f"faltando {path}\nbaixe com:\n"
            f"  mkdir -p {args.parquet_dir} && curl -L \\\n"
            f"    '{HF}/{src['repo']}/resolve/main/{src['file'].replace('|', '%7C')}' \\\n"
            f"    -o {path}")
    rows = load_parquet(path)

    if args.import_cloud:
        import_cloud(rows[:args.limit or len(rows)], golds, out, done)
        return

    if not args.tiers:
        raise SystemExit("nada a fazer: passe --validate, --tiers ou --import-cloud")

    ladder = json.loads(args.config.read_text()) if args.config else {}
    if not ladder:
        raise SystemExit("--tiers exige --config com a escada (tier -> {model, backend})")

    for tier in args.tiers.split(","):
        tier = tier.strip()
        cfg = ladder[tier]
        layer = OllamaGenerativeLayer(layer_name=tier, model=cfg["model"],
                                      max_tokens=256, temperature=0.0,
                                      timeout_s=args.timeout, num_thread=args.threads)
        acc = replay(rows, layer, tier, cfg["model"], golds, out, done, args.limit)
        print(f"[{tier}] acurácia final {acc:.4f}\n")


if __name__ == "__main__":
    main()
