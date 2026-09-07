"""Quanto da energia da cascata e gasta em camadas cujas respostas sao DESCARTADAS?

Bouchard (arXiv:2605.06350) chama isso de "structural cost": a cascata paga o
modelo barato antes de qualquer decisao de escalonamento. Aqui medimos em
joules, no hardware real.
"""
import collections
import json

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.matrix.jsonl')
CHAIN = ["user", "onu", "fog", "cloud"]
DEC = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}

recs = [json.loads(l) for l in open(P) if l.strip()]
recs = [r for r in recs if all(t in r["tiers"] for t in CHAIN)]
CONF = {t: sorted(r["tiers"][t]["confidence"] for r in recs) for t in CHAIN[:-1]}


def q(t, b):
    s = CONF[t]
    return float('-inf') if b <= 0 else (float('inf') if b >= 1
                                         else s[min(len(s) - 1, int(b * len(s)))])


def custo(r, t):
    d = r["tiers"][t]
    return PRE[t] * d["tokens_prompt"] + DEC[t] * d["tokens_gen"]


print("%-6s %-9s %-11s %-13s %-13s %s"
      % ("beta", "acuracia", "J total", "J descartado", "% desperdicio", "onde para"))
print("-" * 84)
for beta in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    thr = {t: q(t, beta) for t in CHAIN[:-1]}
    ok = 0
    tot = 0.0
    desp = 0.0
    fin = collections.Counter()
    for r in recs:
        for k, t in enumerate(CHAIN):
            c = custo(r, t)
            tot += c
            if k == len(CHAIN) - 1 or r["tiers"][t]["confidence"] >= thr[t]:
                ok += r["tiers"][t]["correct"]
                fin[t] += 1
                break
            desp += c    # esta camada respondeu e a resposta foi jogada fora
    n = len(recs)
    print("%-6.2f %-9.3f %-11.1f %-13.1f %-13.1f %s"
          % (beta, ok / n, tot / n, desp / n, 100 * desp / tot,
             {t: fin[t] for t in CHAIN if fin[t]}))

print("\nSe a consulta pudesse ir DIRETO para a camada onde ela para:")
for beta in (0.5, 0.75, 0.9, 1.0):
    thr = {t: q(t, beta) for t in CHAIN[:-1]}
    tot = 0.0
    direto = 0.0
    for r in recs:
        for k, t in enumerate(CHAIN):
            tot += custo(r, t)
            if k == len(CHAIN) - 1 or r["tiers"][t]["confidence"] >= thr[t]:
                direto += custo(r, t)
                break
    print("  beta %.2f: %.1f J -> %.1f J   economia %.0f%%"
          % (beta, tot / len(recs), direto / len(recs), 100 * (1 - direto / tot)))
