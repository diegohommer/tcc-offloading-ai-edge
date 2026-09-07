"""Quanto custa NAO adaptar o beta ao regime de energia?

O beta calibrado para um regime, aplicado em outro, perde quanto? Essa perda e
o premio maximo que um beta dinamico poderia capturar -- e da para medi-la sem
nenhum dado temporal, so reprecificando a mesma matriz.

Regimes: a nuvem varia 38x com o lote (Caravaca, secao 17.3), entao o mesmo
trace precificado sob nuvem-em-lote e nuvem-ociosa da duas fronteiras
diferentes, com betas otimos diferentes.
"""
import itertools
import json

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.matrix.jsonl')
CHAIN = ["user", "onu", "fog", "cloud"]
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}

REGIMES = {
    "nuvem em lote otimizado": {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002},
    "nuvem ociosa (lote=1)":   {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 38.6},
}

recs = [json.loads(l) for l in open(P) if l.strip()]
recs = [r for r in recs if all(t in r["tiers"] for t in CHAIN)]
CONF = {t: sorted(r["tiers"][t]["confidence"] for r in recs) for t in CHAIN[:-1]}


def q(t, b):
    s = CONF[t]
    return float('-inf') if b <= 0 else (float('inf') if b >= 1
                                         else s[min(len(s) - 1, int(b * len(s)))])


def avalia(betas, DEC):
    thr = {t: q(t, betas[i]) for i, t in enumerate(CHAIN[:-1])}
    ok = 0
    e = 0.0
    for r in recs:
        for k, t in enumerate(CHAIN):
            d = r["tiers"][t]
            e += PRE[t] * d["tokens_prompt"] + DEC[t] * d["tokens_gen"]
            if k == len(CHAIN) - 1 or d["confidence"] >= thr[t]:
                ok += d["correct"]
                break
    return ok / len(recs), e / len(recs)


grade = [i / 10 for i in range(11)]
VETORES = list(itertools.product(grade, repeat=3))

# Para cada regime, o melhor vetor de beta em cada nivel de acuracia alvo
otimo = {}
for nome, DEC in REGIMES.items():
    tab = {}
    for v in VETORES:
        a, e = avalia(v, DEC)
        alvo = round(a, 1)
        if alvo not in tab or e < tab[alvo][1]:
            tab[alvo] = (v, e)
    otimo[nome] = tab

print("PERDA POR USAR O BETA DO REGIME ERRADO\n")
print("%-10s %-14s %-14s %-14s %s"
      % ("acuracia", "beta certo", "beta errado", "energia certa", "penalidade"))
print("-" * 74)

nomes = list(REGIMES)
for alvo in sorted(set(otimo[nomes[0]]) & set(otimo[nomes[1]])):
    if alvo < 0.2:
        continue
    for i, j in ((0, 1), (1, 0)):
        rn, rw = nomes[i], nomes[j]
        v_certo, e_certo = otimo[rn][alvo]
        v_errado = otimo[rw][alvo][0]
        _, e_errado = avalia(v_errado, REGIMES[rn])
        pen = 100 * (e_errado - e_certo) / e_certo
        if i == 0:
            print("%-10.1f %-14s %-14s %-14.1f %+.1f%%"
                  % (alvo, str(v_certo), str(v_errado), e_certo, pen))
        else:
            print("%-10s %-14s %-14s %-14.1f %+.1f%%"
                  % ("", str(v_certo), str(v_errado), e_certo, pen))
    print()

print("O beta certo por regime esta na coluna 1; a penalidade e o que se paga")
print("por aplicar a calibracao do outro regime. Esse e o premio maximo que um")
print("beta dinamico poderia capturar.")
