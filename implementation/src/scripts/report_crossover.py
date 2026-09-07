"""Em J/token a escada inverte? E em J/consulta? Onde esta o cruzamento?"""
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}
DEC = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
CHAIN = ["user", "onu", "fog", "cloud"]

print("=== TABELA A: J/token (os parametros) ===")
print("%-7s %-12s %-12s" % ("camada", "prefill", "decode"))
for t in CHAIN:
    print("%-7s %-12.4f %-12.3f" % (t, PRE[t], DEC[t]))
print("\n  decode:  0.074 < 0.22 < 1.75 > 1.002   -> inverte SO no topo")
print("  prefill: 0.016 > 0.0009 < 0.033 > 0.0027 -> inverte no user->onu tambem")

print("\n=== TABELA B: J/consulta (parametros x tokens do regime) ===")
REGIMES = {
    "leaderboard 5-shot": {"user": (900, 45), "onu": (1024, 62),
                           "fog": (1010, 96), "cloud": (875, 88)},
    "CoT original":       {"user": (50, 200), "onu": (50, 200),
                           "fog": (50, 200), "cloud": (50, 200)},
}
for nome, toks in REGIMES.items():
    print("\n  %s:" % nome)
    vals = []
    for t in CHAIN:
        p, g = toks[t]
        c = PRE[t] * p + DEC[t] * g
        vals.append(c)
        print("    %-7s %4d ent + %3d ger = %7.1f J" % (t, p, g, c))
    inv = [CHAIN[i] for i in range(3) if vals[i] > vals[i + 1]]
    print("    inverte em: %s" % (inv if inv else "nenhum ponto"))

print("\n=== ONDE ESTA O CRUZAMENTO user/onu ===")
print("  user = 0.016*P + 0.074*G     onu = 0.0009*P + 0.22*G")
print("  iguais quando  0.0151*P = 0.146*G  ->  P = 9.67*G\n")
for g in (34, 45, 62, 100, 200):
    print("    com %3d tokens gerados, cruza em P = %4.0f tokens de prompt" % (g, 9.67 * g))
print("\n  Abaixo do cruzamento: user mais barato (ordem normal).")
print("  Acima: onu mais barato (invertido). O regime do leaderboard tem ~1000.")
