import json
import statistics as st

P = ('/home/diego-amorim/dev/personal/tcc-offloading-ai-edge/.claude/worktrees/'
     'tcc-proposal-doc/implementation/results/traces/lb_full.matrix.jsonl')
CHAIN = ["user", "onu", "fog", "cloud"]
DEC = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}

recs = [json.loads(l) for l in open(P) if l.strip()]
recs = [r for r in recs if all(t in r["tiers"] for t in CHAIN)]

print("%-7s %-24s %-24s %s" % ("camada", "mediana dos tokens (18.4)",
                               "media dos custos (21.2)", "diferenca"))
print("-" * 76)
for t in CHAIN:
    mp = st.median([r["tiers"][t]["tokens_prompt"] for r in recs])
    mg = st.median([r["tiers"][t]["tokens_gen"] for r in recs])
    por_mediana = PRE[t] * mp + DEC[t] * mg
    por_media = st.mean([PRE[t] * r["tiers"][t]["tokens_prompt"]
                         + DEC[t] * r["tiers"][t]["tokens_gen"] for r in recs])
    print("%-7s %-24.2f %-24.2f %+.1f%%"
          % (t, por_mediana, por_media, 100 * (por_media - por_mediana) / por_mediana))

print("\nA media e maior porque a distribuicao de tokens gerados tem cauda longa:")
for t in CHAIN:
    g = [r["tiers"][t]["tokens_gen"] for r in recs]
    print("  %-6s mediana %3d  media %5.1f  max %3d" % (t, st.median(g), st.mean(g), max(g)))
