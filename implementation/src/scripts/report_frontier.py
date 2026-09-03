#!/usr/bin/env python3
"""Fronteira acuracia-energia da cascata, precificada com os DOIS termos.

    E_query = E_pf_per_token * n_prompt  +  E_dec_per_token * n_gen

Substitui o corte so-decode do report_ladder.py. No regime do replay do
leaderboard (~1000 tokens de prompt reprocessados a cada degrau visitado contra
34-90 gerados) o prefill deixa de ser segunda ordem, e a camada de usuario
passa a ser dominada por ele.

Os J/token de prefill vem do bloco `prefill:` do layer_energy.yaml, adicionado
em 2026-09-03: user e onu medidos, fog e cloud derivados de razoes publicadas.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHAIN = ["user", "onu", "fog", "cloud"]
LINK = 0.1


def carregar_energia():
    d = yaml.safe_load(open(ROOT / "config/layer_energy.yaml"))
    pf = {t: float(d["prefill"][t]["J_per_input_token"]) for t in CHAIN}
    # decode: a escada travada da secao 17.1 do documento de desenho
    dec = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
    return pf, dec


def carregar_matriz(path: Path):
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [r for r in recs if all(t in r["tiers"] for t in CHAIN)]


def quantil(vals, b):
    s = sorted(vals)
    if b <= 0:
        return float("-inf")
    if b >= 1:
        return float("inf")
    return s[min(len(s) - 1, int(b * len(s)))]


def simular(recs, beta, pf, dec, so_decode=False):
    """Escalonamento passo a passo; energia acumulada em cada camada visitada.

    O prompt e reprocessado em cada degrau -- nao ha transferencia de KV cache
    entre camadas -- entao o termo de prefill entra uma vez por camada visitada,
    e nao uma vez por consulta.
    """
    thr = {t: quantil([r["tiers"][t]["confidence"] for r in recs], beta)
           for t in CHAIN[:-1]}
    ok = 0
    energia = 0.0
    fin = collections.Counter()
    for r in recs:
        for k, t in enumerate(CHAIN):
            d = r["tiers"][t]
            energia += dec[t] * d["tokens_gen"]
            if not so_decode:
                energia += pf[t] * d["tokens_prompt"]
            if k:
                energia += LINK
            if k == len(CHAIN) - 1 or d["confidence"] >= thr[t]:
                ok += d["correct"]
                fin[t] += 1
                break
    n = len(recs)
    return ok / n, energia / n, {t: fin[t] for t in CHAIN if fin[t]}


def main():
    matriz = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "results/traces/lb_full.matrix.jsonl"
    recs = carregar_matriz(matriz)
    pf, dec = carregar_energia()

    print("n = %d consultas com as quatro camadas\n" % len(recs))

    print("CUSTO POR CAMADA, decomposto (medianas)")
    print("%-7s %-9s %-9s %-11s %-11s %s" %
          ("camada", "tok_ent", "tok_ger", "prefill J", "decode J", "prefill %"))
    print("-" * 68)
    for t in CHAIN:
        ent = sorted(r["tiers"][t]["tokens_prompt"] for r in recs)[len(recs) // 2]
        ger = sorted(r["tiers"][t]["tokens_gen"] for r in recs)[len(recs) // 2]
        ep, ed = pf[t] * ent, dec[t] * ger
        print("%-7s %-9d %-9d %-11.2f %-11.2f %.0f%%"
              % (t, ent, ger, ep, ed, 100 * ep / (ep + ed)))

    print("\nFRONTEIRA")
    print("%-6s %-9s %-13s %-13s %-9s %s" %
          ("beta", "acuracia", "J (2 termos)", "J (so decode)", "erro", "distribuicao"))
    print("-" * 78)
    for beta in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        acc, e2, fin = simular(recs, beta, pf, dec)
        _, e1, _ = simular(recs, beta, pf, dec, so_decode=True)
        print("%-6.2f %-9.3f %-13.1f %-13.1f %-9s %s"
              % (beta, acc, e2, e1, "+%.0f%%" % (100 * (e2 - e1) / e1), fin))


if __name__ == "__main__":
    main()
