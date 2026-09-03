"""Escada completa das quatro camadas + simulacao da cascata RecServe."""
import collections
import json
import math
import statistics as st

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.raw.jsonl')

CHAIN = ["user", "onu", "fog", "cloud"]
NOME = {"user": "llama3.2:1b Q4", "onu": "qwen2.5:1.5b Q4",
        "fog": "gemma-2-9b FP16", "cloud": "llama-3-70B BF16"}
# J/token de decode, cada um da fonte declarada para aquela camada
J = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
LINK = 0.1

rec = collections.defaultdict(dict)
for line in open(P):
    line = line.strip()
    if line:
        r = json.loads(line)
        rec[r["index"]][r["tier"]] = r
idx = sorted(i for i, d in rec.items() if all(t in d for t in CHAIN))

print("=" * 68)
print("ESCADA COMPLETA  (n=%d, mesmas instancias em todas as camadas)" % len(idx))
print("=" * 68)
print("%-7s %-18s %-9s %-8s %s" % ("camada", "modelo", "acuracia", "tok_gen", "J/consulta"))
for t in CHAIN:
    acc = sum(rec[i][t]["correct"] for i in idx) / len(idx)
    tg = st.median([rec[i][t]["tokens_gen"] for i in idx])
    print("%-7s %-18s %-9.4f %-8d %.1f" % (t, NOME[t], acc, tg, J[t] * tg))

print("\n" + "=" * 68)
print("SALTOS PAREADOS  (conserta / quebra, por instancia)")
print("=" * 68)
for a, b in zip(CHAIN, CHAIN[1:]):
    fix = sum(1 for i in idx if not rec[i][a]["correct"] and rec[i][b]["correct"])
    brk = sum(1 for i in idx if rec[i][a]["correct"] and not rec[i][b]["correct"])
    print("%-6s -> %-6s  conserta %3d   quebra %3d   liquido %+4d"
          % (a, b, fix, brk, fix - brk))


def welch(x, y):
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    se = math.sqrt(st.variance(x) / len(x) + st.variance(y) / len(y))
    return (st.mean(x) - st.mean(y)) / se if se else float('nan')


print("\n" + "=" * 68)
print("SEPARACAO DA CONFIANCA  (so camadas nao-terminais decidem)")
print("=" * 68)
for t in CHAIN[:-1]:
    rs = [rec[i][t] for i in idx if rec[i][t]["confidence"] is not None]
    rs.sort(key=lambda r: r["confidence"], reverse=True)
    base = sum(r["correct"] for r in rs) / len(rs)
    k = len(rs) // 4
    top = sum(r["correct"] for r in rs[:k]) / k
    ok = [r["confidence"] for r in rs if r["correct"]]
    no = [r["confidence"] for r in rs if not r["correct"]]
    print("%-6s base %.3f | top 25%% conf %.3f (%.1fx) | Welch t %+.2f"
          % (t, base, top, top / base if base else 0, welch(ok, no)))

print("\n" + "=" * 68)
print("CASCATA RecServe  (escalonamento passo a passo, limiar = quantil beta)")
print("=" * 68)
print("%-6s %-9s %-11s %s" % ("beta", "acuracia", "J/consulta", "distribuicao final"))


def quantile(v, b):
    s = sorted(v)
    return float('-inf') if b <= 0 else (float('inf') if b >= 1
                                         else s[min(len(s) - 1, int(b * len(s)))])


for beta in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    thr = {t: quantile([rec[i][t]["confidence"] for i in idx], beta) for t in CHAIN[:-1]}
    ok = 0
    e = 0.0
    fin = collections.Counter()
    for i in idx:
        for k, t in enumerate(CHAIN):
            d = rec[i][t]
            e += J[t] * d["tokens_gen"] + (LINK if k else 0)
            if k == len(CHAIN) - 1 or d["confidence"] >= thr[t]:
                ok += d["correct"]
                fin[t] += 1
                break
    print("%-6.2f %-9.3f %-11.1f %s"
          % (beta, ok / len(idx), e / len(idx),
             {t: fin[t] for t in CHAIN if fin[t]}))
