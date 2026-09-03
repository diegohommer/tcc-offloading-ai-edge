"""O teste da secao 14, refeito na escada de quatro camadas.

Pergunta: variar lambda alcanca algum ponto (acuracia, energia) que variar beta
sozinho nao alcanca? Se nao, o termo de energia e redundante com o beta.

Este e o caso mais favoravel ao lambda ja testado, porque aqui fog->cloud e
energeticamente NEGATIVO (fog 1.75 J/tok, cloud 1.002) -- a assimetria de sinal
entre saltos que a secao 4 diz que um beta global nao consegue expressar.
"""
import collections
import json
import math

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.matrix.jsonl')

CHAIN = ["user", "onu", "fog", "cloud"]
J = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
LINK = 0.1

recs = [json.loads(l) for l in open(P) if l.strip()]
# custo estatico do salto: J/token do proximo tier x tokens tipicos daquele tier
TOK = {t: sorted(r["tiers"][t]["tokens_gen"] for r in recs)[len(recs) // 2] for t in CHAIN}
HOP = {a: J[b] * TOK[b] + LINK for a, b in zip(CHAIN, CHAIN[1:])}


def quantile(v, b):
    s = sorted(v)
    if b <= 0:
        return float('-inf')
    if b >= 1:
        return float('inf')
    return s[min(len(s) - 1, int(b * len(s)))]


def run(beta, lam, forma="exponential"):
    conf = {t: [r["tiers"][t]["confidence"] for r in recs] for t in CHAIN[:-1]}
    thr = {}
    for t in CHAIN[:-1]:
        base = quantile(conf[t], beta)
        if lam == 0:
            thr[t] = base
        elif forma == "exponential":
            thr[t] = base * math.exp(-lam * HOP[t])
        else:                                   # multiplicativa
            thr[t] = base * max(0.0, 1 - lam * HOP[t])
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


print("custo estatico por salto (J):",
      {k: round(v, 1) for k, v in HOP.items()})
print("  NOTE: onu->fog custa %.1f J, fog->cloud custa %.1f J -- o salto para a"
      % (HOP["onu"], HOP["fog"]))
print("  nuvem e MAIS BARATO que o salto para o fog.\n")

# Fronteira so com beta
betas = [i / 100 for i in range(0, 101, 2)]
front_beta = sorted({run(b, 0) for b in betas})

# Todos os pontos alcancaveis variando lambda, sobre varios beta
pts_lambda = []
for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for lam in [0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]:
        acc, en = run(b, lam)
        pts_lambda.append((acc, en, b, lam))


def melhor_beta_em(energia):
    """Melhor acuracia que beta sozinho alcanca gastando <= essa energia."""
    cands = [a for a, e in front_beta if e <= energia + 1e-9]
    return max(cands) if cands else None


print("%-8s %-7s %-9s %-11s %-11s %s" %
      ("beta", "lambda", "acuracia", "J/consulta", "beta-so", "ganho"))
print("-" * 62)
ganhos = []
for acc, en, b, lam in sorted(pts_lambda, key=lambda x: x[1]):
    ref = melhor_beta_em(en)
    if ref is None or lam == 0:
        continue
    g = acc - ref
    ganhos.append(g)
    if abs(g) > 0.001:
        print("%-8.2f %-7.4g %-9.3f %-11.1f %-11.3f %+.4f" % (b, lam, acc, en, ref, g))

if ganhos:
    print("\nganho medio do lambda sobre beta sozinho: %+.4f" % (sum(ganhos) / len(ganhos)))
    print("ganho maximo:                              %+.4f" % max(ganhos))
    print("pontos onde lambda supera beta:            %d de %d"
          % (sum(1 for g in ganhos if g > 0.001), len(ganhos)))
