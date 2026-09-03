#!/usr/bin/env python3
"""RQ4: que fracao da energia da cascata e transporte da rede de acesso?

O layer_energy.yaml carrega uma constante de 0.1 J por salto, sem derivacao a
partir dos bytes movidos. Este script substitui a constante por um calculo, e a
distincao que ele expoe e especifica de PON.

POR QUE PON MUDA A CONTA
------------------------
Uma PON e passiva: nao ha elemento alimentado na rede de distribuicao, so um
splitter optico. Toda a energia esta nos dois extremos -- a ONU na casa do
assinante e a porta do OLT no escritorio central -- e AMBOS SAO SEMPRE-LIGADOS.
Uma ONU consome ~4 W esteja transmitindo ou nao.

Disso saem duas grandezas muito diferentes, e qual usar e decisao de modelagem:

  MARGINAL   o custo de mandar UMA consulta a mais: tempo de transmissao x
             potencia. E o custo relevante para a DECISAO de escalonar, porque
             a ONU esta ligada de qualquer forma.

  AMORTIZADO a parcela da potencia sempre-ligada atribuida a cada consulta:
             potencia / (consultas por segundo). E o custo relevante para a
             contabilidade TOTAL do sistema.

Elas diferem por ordens de magnitude, e conflati-las e o erro que a constante
de 0.1 J escondia.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAIN = ["user", "onu", "fog", "cloud"]

# --- parametros da PON ------------------------------------------------------
# Potencia da ONU: Sarigiannidis et al., IET Networks 2016 (ja no layer_energy.yaml)
ONU_ATIVA_W = 3.98
ONU_SLEEP_W = 0.40

# Taxas. Pakpahan e Hwang especificam TWDM-PON (ITU-T G.989 NG-PON2) com a ONU
# em "symmetric 50 Gb/s" sobre dois pares de comprimento de onda de 25 Gb/s.
# GPON legado incluido como limite inferior conservador.
TAXAS = {
    "GPON upstream (1.244 Gb/s)": 1.244e9,
    "XGS-PON (10 Gb/s)": 10e9,
    "TWDM-PON, Pakpahan (25 Gb/s por lambda)": 25e9,
}

# Bytes por token. O prompt e ASCII ingles; ~4 bytes/token e a regra usual e
# bate com os textos deste trace.
BYTES_POR_TOKEN = 4

# Energia de computacao por camada (escada travada, secao 17.1 do design doc)
DEC = {"user": 0.074, "onu": 0.22, "fog": 1.75, "cloud": 1.002}
PRE = {"user": 0.016, "onu": 0.0009, "fog": 0.033, "cloud": 0.0027}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "results/traces/lb_full.matrix.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if all(t in r["tiers"] for t in CHAIN)]

    med = lambda t, k: st.median([r["tiers"][t][k] for r in recs])

    # Um salto carrega o prompt para cima e a resposta para baixo.
    prompt_tok = med("onu", "tokens_prompt")
    resp_tok = med("onu", "tokens_gen")
    bytes_salto = (prompt_tok + resp_tok) * BYTES_POR_TOKEN

    print("n = %d consultas\n" % len(recs))
    print("BYTES POR SALTO DE ESCALONAMENTO")
    print("  prompt   %5.0f tokens" % prompt_tok)
    print("  resposta %5.0f tokens" % resp_tok)
    print("  total    %5.0f tokens x %d B = %.1f KB\n"
          % (prompt_tok + resp_tok, BYTES_POR_TOKEN, bytes_salto / 1024))

    print("ENERGIA MARGINAL POR SALTO (tempo de transmissao x potencia da ONU)")
    print("  %-42s %-12s %s" % ("taxa", "tempo", "energia"))
    print("  " + "-" * 66)
    marginais = {}
    for nome, bps in TAXAS.items():
        t = bytes_salto * 8 / bps
        e = ONU_ATIVA_W * t
        marginais[nome] = e
        print("  %-42s %-12s %.3e J" % (nome, "%.1f us" % (t * 1e6), e))

    # --- comparacao com computacao ---
    comp = {t: PRE[t] * med(t, "tokens_prompt") + DEC[t] * med(t, "tokens_gen")
            for t in CHAIN}
    print("\nCOMPUTACAO POR CAMADA (prefill + decode, J/consulta)")
    for t in CHAIN:
        print("  %-6s %8.2f J" % (t, comp[t]))

    e_marg = marginais["TWDM-PON, Pakpahan (25 Gb/s por lambda)"]
    print("\nTRANSPORTE MARGINAL COMO FRACAO DA COMPUTACAO")
    print("  %-6s %-14s %-14s %s" % ("camada", "computacao", "transporte", "fracao"))
    print("  " + "-" * 54)
    for t in CHAIN:
        print("  %-6s %-14.2f %-14.3e %.2e" % (t, comp[t], e_marg, e_marg / comp[t]))

    # --- amortizado ---
    print("\nENERGIA AMORTIZADA SEMPRE-LIGADA (potencia da ONU / consultas por s)")
    print("  A ONU consome %.2f W ativa e %.2f W dormindo, transmitindo ou nao." % (ONU_ATIVA_W, ONU_SLEEP_W))
    print("  %-24s %-16s %s" % ("carga da residencia", "J/consulta", "vs computacao do user"))
    print("  " + "-" * 62)
    for qps, rotulo in ((1 / 3600, "1 consulta/hora"), (1 / 60, "1 consulta/minuto"),
                        (1.0, "1 consulta/segundo"), (40.0, "40 consultas/segundo")):
        e = ONU_ATIVA_W / qps
        print("  %-24s %-16.3f %.1fx" % (rotulo, e, e / comp["user"]))

    print("\n  A constante de 0.1 J/salto que estava no config corresponde a")
    print("  %.1f consultas por segundo por ONU -- isto e, ela ja era um numero" % (ONU_ATIVA_W / 0.1))
    print("  amortizado, nao marginal, e nunca esteve documentado como tal.")


if __name__ == "__main__":
    main()
