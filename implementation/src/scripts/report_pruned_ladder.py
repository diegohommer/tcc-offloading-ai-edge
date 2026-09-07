"""Quanto vale podar a escada para o pool non-dominated?

A proposta nao e um mecanismo novo: e aplicar o criterio que a propria
literatura de cascatas declara (non-dominated pool) usando ENERGIA MEDIDA
em vez de escala de modelo como proxy de custo.

Compara a cascata de 4 camadas do Pakpahan contra a de 2 camadas que sobra
depois da poda: {onu, cloud}.
"""
import json

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.matrix.jsonl')
DEC = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}

recs = [json.loads(l) for l in open(P) if l.strip()]
recs = [r for r in recs if all(t in r["tiers"] for t in DEC)]


def custo(r, t):
    d = r["tiers"][t]
    return PRE[t] * d["tokens_prompt"] + DEC[t] * d["tokens_gen"]


def q(vals, b):
    s = sorted(vals)
    return float('-inf') if b <= 0 else (float('inf') if b >= 1
                                         else s[min(len(s) - 1, int(b * len(s)))])


def frente(chain):
    conf = {t: [r["tiers"][t]["confidence"] for r in recs] for t in chain[:-1]}
    pts = []
    for i in range(0, 101, 5):
        b = i / 100
        thr = {t: q(conf[t], b) for t in chain[:-1]}
        ok = 0
        e = 0.0
        for r in recs:
            for k, t in enumerate(chain):
                e += custo(r, t)
                if k == len(chain) - 1 or r["tiers"][t]["confidence"] >= thr[t]:
                    ok += r["tiers"][t]["correct"]
                    break
        pts.append((ok / len(recs), e / len(recs)))
    # fronteira de Pareto
    pts.sort(key=lambda x: x[1])
    out, m = [], -1
    for a, c in pts:
        if a > m:
            out.append((a, c))
            m = a
    return out


f4 = frente(["user", "onu", "fog", "cloud"])
f2 = frente(["onu", "cloud"])

print("Para cada acuracia alvo, quanta energia cada escada gasta:\n")
print("%-10s %-16s %-16s %s" % ("acuracia", "4 camadas", "2 camadas (podada)", "economia"))
print("-" * 64)
for alvo in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85):
    c4 = min((c for a, c in f4 if a >= alvo), default=None)
    c2 = min((c for a, c in f2 if a >= alvo), default=None)
    if c4 and c2:
        print("%-10.2f %-16.1f %-16.1f %+.0f%%"
              % (alvo, c4, c2, 100 * (c2 - c4) / c4))

print("\nExtremos das duas fronteiras:")
for nome, f in (("4 camadas", f4), ("2 camadas", f2)):
    print("  %-10s de %.3f @ %6.1f J  ate  %.3f @ %6.1f J"
          % (nome, f[0][0], f[0][1], f[-1][0], f[-1][1]))
