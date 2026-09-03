"""A conclusao da secao 17 sobrevive ao modelo de dois termos?

O teste original usou energia so de decode. Com prefill, os custos de salto
mudam -- especialmente o user->onu, ja que o user e 85% prefill.
"""
import itertools
import json
import math

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.matrix.jsonl')
CHAIN = ["user", "onu", "fog", "cloud"]
DEC = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}
LINK = 0.1

recs = [json.loads(l) for l in open(P) if l.strip()]
med = lambda t, k: sorted(r["tiers"][t][k] for r in recs)[len(recs) // 2]
CONF = {t: sorted(r["tiers"][t]["confidence"] for r in recs) for t in CHAIN[:-1]}


def q(t, b):
    s = CONF[t]
    return float('-inf') if b <= 0 else (float('inf') if b >= 1
                                         else s[min(len(s) - 1, int(b * len(s)))])


def avalia(thr, dois_termos):
    ok = 0
    e = 0.0
    for r in recs:
        for k, t in enumerate(CHAIN):
            d = r["tiers"][t]
            e += DEC[t] * d["tokens_gen"]
            if dois_termos:
                e += PRE[t] * d["tokens_prompt"]
            if k:
                e += LINK
            if k == len(CHAIN) - 1 or d["confidence"] >= thr[t]:
                ok += d["correct"]
                break
    return ok / len(recs), e / len(recs)


for rotulo, dois in (("SO DECODE (o teste original da secao 17)", False),
                     ("DOIS TERMOS (prefill + decode)", True)):
    # custo do salto = custo total da camada de destino
    HOP = {a: (DEC[b] * med(b, "tokens_gen")
               + (PRE[b] * med(b, "tokens_prompt") if dois else 0) + LINK)
           for a, b in zip(CHAIN, CHAIN[1:])}

    grade = [i / 10 for i in range(11)]
    front = [avalia({"user": q("user", bu), "onu": q("onu", bo), "fog": q("fog", bf)}, dois)
             for bu, bo, bf in itertools.product(grade, repeat=3)]
    front.sort(key=lambda x: x[1])
    pareto, m = [], -1
    for acc, en in front:
        if acc > m:
            pareto.append((acc, en))
            m = acc

    ganhos = []
    for b in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for lam in [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03]:
            thr = {t: q(t, b) * math.exp(-lam * HOP[t]) for t in CHAIN[:-1]}
            acc, en = avalia(thr, dois)
            c = [a for a, e in pareto if e <= en + 1e-9]
            if c:
                ganhos.append(acc - max(c))

    print("\n%s" % rotulo)
    print("  saltos (J): " + "  ".join("%s->%s %.1f" % (a, b, HOP[a])
                                       for a, b in zip(CHAIN, CHAIN[1:])))
    print("  ganho medio do lambda sobre beta-por-camada: %+.4f"
          % (sum(ganhos) / len(ganhos)))
    print("  ganho maximo: %+.4f  |  lambda supera em %d de %d"
          % (max(ganhos), sum(1 for g in ganhos if g > 0.001), len(ganhos)))
