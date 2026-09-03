"""lambda contra beta-por-camada, sob cenarios de energia diferentes.

A acuracia e a confianca vem do trace e nao mudam -- o que muda e a atribuicao
de J/token por camada. Isso e analise de sensibilidade: a conclusao da secao 14
depende do formato da escada de energia, ou nao?

Todos os valores sao linhas reais do layer_energy.yaml.
"""
import itertools
import json
import math

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.matrix.jsonl')
CHAIN = ["user", "onu", "fog", "cloud"]
LINK = 0.1

CENARIOS = {
    # o que rodamos: fog em stream unico, nuvem em lote -> INVERTE no topo
    "medido (fog L4 stream unico / cloud 4xH100 lote)":
        {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002},
    # config 'monotonic' do sweep: fog SOLAR-10.7B A30, cloud producao BF16
    "monotonica (fog A30 producao / cloud producao BF16)":
        {"user": 0.074, "onu": 0.22, "fog": 0.3785, "cloud": 0.3989},
    # nuvem ociosa: QwQ-32B lote=1 em 1xH100 -> monotonica FORTE
    "monotonica forte (cloud ocioso, lote=1)":
        {"user": 0.074, "onu": 0.22, "fog": 0.3785, "cloud": 12.904},
}

recs = [json.loads(l) for l in open(P) if l.strip()]
TOK = {t: sorted(r["tiers"][t]["tokens_gen"] for r in recs)[len(recs) // 2] for t in CHAIN}
CONF = {t: sorted(r["tiers"][t]["confidence"] for r in recs) for t in CHAIN[:-1]}


def q(t, b):
    s = CONF[t]
    return float('-inf') if b <= 0 else (float('inf') if b >= 1
                                         else s[min(len(s) - 1, int(b * len(s)))])


def avalia(thr, J):
    ok = 0
    e = 0.0
    for r in recs:
        for k, t in enumerate(CHAIN):
            d = r["tiers"][t]
            e += J[t] * d["tokens_gen"] + (LINK if k else 0)
            if k == len(CHAIN) - 1 or d["confidence"] >= thr[t]:
                ok += d["correct"]
                break
    return ok / len(recs), e / len(recs)


grade = [i / 10 for i in range(11)]
for nome, J in CENARIOS.items():
    HOP = {a: J[b] * TOK[b] + LINK for a, b in zip(CHAIN, CHAIN[1:])}
    front = [avalia({"user": q("user", bu), "onu": q("onu", bo), "fog": q("fog", bf)}, J)
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
            acc, en = avalia(thr, J)
            c = [a for a, e in pareto if e <= en + 1e-9]
            if c:
                ganhos.append(acc - max(c))

    print("\n%s" % nome)
    print("  saltos (J): " + "  ".join("%s->%s %.1f" % (a, b, HOP[a])
                                       for a, b in zip(CHAIN, CHAIN[1:])))
    print("  monotonica em energia? %s"
          % ("SIM" if HOP["user"] <= HOP["onu"] <= HOP["fog"] else "NAO"))
    print("  ganho medio do lambda sobre beta-por-camada: %+.4f" % (sum(ganhos) / len(ganhos)))
    print("  ganho maximo: %+.4f   |   lambda supera em %d de %d"
          % (max(ganhos), sum(1 for g in ganhos if g > 0.001), len(ganhos)))
