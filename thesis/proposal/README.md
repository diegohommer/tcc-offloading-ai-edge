# TCC proposal (revised)

`joules-per-query.html` — the revised proposal, written in English, superseding
the original mechanism-oriented scope.

**Published artifact:**
https://claude.ai/code/artifact/80752dd5-f88f-47a0-ab4b-9cb2d632a92f

The artifact is private until shared from the page's own share menu. It lives
in the Claude artifact gallery independently of any conversation — reachable at
`claude.ai/code/artifacts`, or via `/artifacts` in the Claude Code terminal.
This file is the source it was published from; editing here and republishing to
the same URL updates the page in place.

## What changed against the original scope

The proposal is now an **evaluation**, not a proposal for a new mechanism. The
driver is the result recorded in section 14 of
`tcc_politica_energia_desenho.md`: weighting RecServe's confidence threshold by
an energy cost is redundant with its existing β parameter, because a quantile is
a rank statistic and tier-constant terms cannot re-rank queries.

Structure: ten sections anchored on four research questions (RQ1 characterisation,
RQ2 frontier, RQ3 the rule, RQ4 transport). Every result in §6 answers one of
them; every item in §9 closes one that is still open.

New material not present in the design document:

- **§5.1, energy accounting.** J/token is a ratio whose denominator is
  task-dependent (`e_decode + E_prefill/n_gen`). The field convention is
  phase-separated reporting — input tokens as the prefill denominator, output
  tokens as the decode denominator — which is the form `src/energy/cost.py`
  already implements. Includes the observation that stepwise escalation re-runs
  prefill at every tier.
- **§7, methodological limitations**, quantified rather than declared: phase
  reporting is inconsistent across sources (bounded by three independent
  measurements agreeing prefill is ~1–3.4% of total), and batch regimes differ
  between the lower and upper tiers.

Two claims were **narrowed** after a literature check, and this is deliberate:

- §6.4 (`exp(min logprob)` beating the specified confidence measure) is reported
  as a replication of Gupta et al., ICLR 2024 (arXiv:2404.10136), who formalise
  it as Chow-Quantile with α = 0.
- §6.5 (confidence blind to difficulty) cites Michael et al. (arXiv:2605.23909)
  for the hard–easy effect; what is claimed here is its consequence for stepwise
  cascade routing.

## Cut from the earlier document

The λ mechanism design (rejected telemetry, time-of-day profiles, admission
gate, skip-ahead), the classification cascade as a standalone result, and the
weighting-form comparison — all superseded by the negative result or reduced to
a single line.

## Rebuilding

Self-contained HTML: fonts from Google Fonts, no other external resources, no
build step. Open it directly in a browser, or republish it as an artifact from
the path above.
