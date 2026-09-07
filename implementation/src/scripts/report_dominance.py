#!/usr/bin/env python3
"""Teste da condicao de non-dominated pool sobre a escada medida.

Bouchard (arXiv:2605.06350, secao 3.1) enuncia a pre-condicao formal das
cascatas de k modelos:

    "We assume the model pool is non-dominated: the models are ordered such
     that c1 < c2 < ... < ck and E[U1] < E[U2] < ... < E[Uk]."

E a nota de rodape 3 deixa a porta aberta:

    "Average dominance does not preclude a model from being useful on
     particular conditional subpopulations; ruling this out would require a
     stronger conditional-dominance condition."

Este script testa as duas coisas na escada PON medida: dominancia MARGINAL
(sobre todas as consultas) e CONDICIONAL (por bucket de dificuldade do GSM8K).

Um modelo i e dominado por j quando custa MAIS e acerta MENOS -- pior nos dois
eixos. Se custa mais e acerta mais, nao ha dominancia: e trade-off, e o modelo
permanece na fronteira de Pareto daquela subpopulacao.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAIN = ["user", "onu", "fog", "cloud"]
DEC = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}


def custo(r, t):
    d = r["tiers"][t]
    return PRE[t] * d["tokens_prompt"] + DEC[t] * d["tokens_gen"]


def perfil(recs):
    """(custo medio, acuracia) por camada sobre um conjunto de consultas."""
    n = len(recs)
    return {t: (sum(custo(r, t) for r in recs) / n,
                sum(r["tiers"][t]["correct"] for r in recs) / n) for t in CHAIN}


def dominados(p):
    """Quais camadas sao estritamente dominadas por alguma outra."""
    out = []
    for i in CHAIN:
        ci, ui = p[i]
        for j in CHAIN:
            if i == j:
                continue
            cj, uj = p[j]
            if cj <= ci and uj >= ui and (cj < ci or uj > ui):
                out.append((i, j))
                break
    return out


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "results/traces/lb_full.matrix.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if all(t in r["tiers"] for t in CHAIN)]

    print("=" * 70)
    print("DOMINANCIA MARGINAL  (todas as %d consultas)" % len(recs))
    print("=" * 70)
    p = perfil(recs)
    print("%-7s %-12s %-10s %s" % ("camada", "custo (J)", "acuracia", "situacao"))
    dom = dict(dominados(p))
    for t in CHAIN:
        c, u = p[t]
        s = "DOMINADA por %s" % dom[t] if t in dom else "na fronteira"
        print("%-7s %-12.2f %-10.3f %s" % (t, c, u, s))

    livres = [t for t in CHAIN if t not in dom]
    print("\n  pool non-dominated = %s" % livres)
    print("  A pre-condicao do Bouchard exige c1<c2<...<ck E U1<U2<...<Uk.")
    print("  Aqui ela falha: %d das 4 camadas sao estritamente dominadas."
          % len(dom))

    print("\n" + "=" * 70)
    print("DOMINANCIA CONDICIONAL  (por dificuldade do GSM8K)")
    print("=" * 70)
    print("Testa a ressalva da nota 3: uma camada dominada na media pode")
    print("permanecer na fronteira em alguma subpopulacao.\n")

    buckets = collections.defaultdict(list)
    for r in recs:
        buckets[r["difficulty_steps"]].append(r)

    print("%-8s %-5s %s" % ("passos", "n", "  ".join("%-16s" % t for t in CHAIN)))
    print("-" * 78)
    resgatadas = collections.defaultdict(list)
    for k in sorted(x for x in buckets if x is not None):
        sub = buckets[k]
        if len(sub) < 15:
            continue
        pk = perfil(sub)
        domk = dict(dominados(pk))
        cels = []
        for t in CHAIN:
            c, u = pk[t]
            mark = "" if t in domk else " *"
            cels.append("%6.1fJ %.3f%s" % (c, u, mark))
            if t in dom and t not in domk:
                resgatadas[t].append(k)
        print("%-8d %-5d %s" % (k, len(sub), "  ".join("%-16s" % x for x in cels)))
    print("\n  * = na fronteira de Pareto daquele bucket")

    print("\nCAMADAS DOMINADAS NA MEDIA QUE SOBREVIVEM CONDICIONALMENTE")
    if resgatadas:
        for t, ks in resgatadas.items():
            print("  %-6s permanece na fronteira em: %s passos" % (t, ks))
        print("\n  Isto e exatamente a ressalva da nota 3 do Bouchard, verificada:")
        print("  dominancia media nao implica inutilidade condicional. Podar essas")
        print("  camadas pelo criterio marginal descartaria as subpopulacoes onde")
        print("  elas ainda sao Pareto-otimas.")
    else:
        print("  nenhuma -- a dominancia e uniforme sobre todos os buckets, o que")
        print("  e o caso FORTE: a poda seria justificada mesmo condicionalmente.")


if __name__ == "__main__":
    main()
