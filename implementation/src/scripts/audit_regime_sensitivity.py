"""A assimetria de regime sobrevive a incerteza do numero interpolado?

A nuvem ociosa a 38.6 J/token vem de interpolar em log(params) a razao
lote1/otimizado medida em quatro modelos do Caravaca. As razoes observadas vao
de 25.5x a 61.3x. Aqui variamos a razao nessa faixa inteira e vemos se a
conclusao muda.
"""
import itertools
import json

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.matrix.jsonl')
CHAIN = ["user", "onu", "fog", "cloud"]
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}
BASE = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}

recs = [json.loads(l) for l in open(P) if l.strip()]
recs = [r for r in recs if all(t in r["tiers"] for t in CHAIN)]
CONF = {t: sorted(r["tiers"][t]["confidence"] for r in recs) for t in CHAIN[:-1]}


def q(t, b):
    s = CONF[t]
    return float('-inf') if b <= 0 else (float('inf') if b >= 1
                                         else s[min(len(s) - 1, int(b * len(s)))])


def avalia(v, DEC):
    thr = {t: q(t, v[i]) for i, t in enumerate(CHAIN[:-1])}
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
VET = list(itertools.product(grade, repeat=3))

print("razao lote1/otimizado observada no Caravaca: 25.5x a 61.3x")
print("interpolada para 70.55B: 38.5x\n")
print("%-10s %-14s %-22s %s" % ("razao", "cloud ocioso", "penalidade otimista",
                                "penalidade conservadora"))
print("-" * 72)

for razao in (25.5, 38.5, 61.3):
    ocioso = dict(BASE, cloud=1.002 * razao)
    tab = {}
    for nome, DEC in (("lote", BASE), ("ocioso", ocioso)):
        t = {}
        for v in VET:
            a, e = avalia(v, DEC)
            k = round(a, 1)
            if k not in t or e < t[k][1]:
                t[k] = (v, e)
        tab[nome] = t

    # penalidade: usar o beta do outro regime
    pens_ot, pens_cons = [], []
    for alvo in sorted(set(tab["lote"]) & set(tab["ocioso"])):
        if alvo < 0.4:
            continue
        # otimista: beta calibrado para lote, rodando ocioso
        v_lote = tab["lote"][alvo][0]
        _, e_err = avalia(v_lote, ocioso)
        e_certo = tab["ocioso"][alvo][1]
        pens_ot.append(100 * (e_err - e_certo) / e_certo)
        # conservador: beta calibrado para ocioso, rodando em lote
        v_oc = tab["ocioso"][alvo][0]
        _, e_err2 = avalia(v_oc, BASE)
        e_certo2 = tab["lote"][alvo][1]
        pens_cons.append(100 * (e_err2 - e_certo2) / e_certo2)

    print("%-10s %-14.1f %-22s %s"
          % ("%.1fx" % razao, 1.002 * razao,
             "mediana %+.0f%%  max %+.0f%%" % (sorted(pens_ot)[len(pens_ot)//2], max(pens_ot)),
             "mediana %+.0f%%  max %+.0f%%" % (sorted(pens_cons)[len(pens_cons)//2], max(pens_cons))))

print("\notimista   = calibrar para nuvem barata, rodar com nuvem cara")
print("conservador = calibrar para nuvem cara, rodar com nuvem barata")
