import collections
import csv
import statistics as st

P = '/home/diego-amorim/.claude/jobs/4305b469/tmp/burst.csv'
rows = [r for r in csv.DictReader(open(P)) if r["Model"] == "ChatGPT"]
ts = [int(r["Timestamp"]) for r in rows]        # segundos desde o inicio
span_h = (max(ts) - min(ts)) / 3600
print("requisicoes ChatGPT: %d" % len(rows))
print("span: %.1f horas = %.1f dias" % (span_h, span_h / 24))

req = [int(r["Request tokens"]) for r in rows]
res = [int(r["Response tokens"]) for r in rows]
print("\nrequest tokens  mediana %4d  media %6.1f" % (st.median(req), st.mean(req)))
print("response tokens mediana %4d  media %6.1f" % (st.median(res), st.mean(res)))
print("razao P/G mediana %.1f   (cruzamento user/onu esta em 9.7)"
      % (st.median(req) / st.median(res)))

# taxa de chegada por hora do dia
por_h = collections.Counter((t // 3600) % 24 for t in ts)
dias = span_h / 24
print("\nchegadas por hora do dia (media sobre %.0f dias):" % dias)
mx = max(por_h.values())
for h in range(24):
    n = por_h.get(h, 0)
    print("  %02dh %7.1f req/h %s" % (h, n / dias, "#" * int(46 * n / mx)))

pico, vale = max(por_h.values()), min(por_h.values())
print("\npico/vale = %.1fx" % (pico / vale))
