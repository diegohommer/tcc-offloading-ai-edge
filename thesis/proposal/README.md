# TCC proposal

`joules-per-query.html` is the source of the published proposal, which is the
**source of truth** for this work's scope, research questions, results and
limitations.

**Published:** https://claude.ai/code/artifact/80752dd5-f88f-47a0-ab4b-9cb2d632a92f

Private until shared from the page's own share menu. It lives in the artifact
gallery independently of any conversation — `claude.ai/code/artifacts`, or
`/artifacts` in the Claude Code terminal.

## Updating

Edit this file and republish it **at this same path** to update the page in
place. Publishing from a conversation that did not create it needs the URL
above passed explicitly, or it creates a separate artifact.

Self-contained HTML — fonts from Google Fonts, nothing else external, no build
step. Opens directly in a browser.

## Provenance

Written 2026-08-30, replacing a mechanism-oriented scope with an evaluation of
RecServe's mechanism. The driver is the result in
`../../tcc_politica_energia_desenho.md` §14: the energy term is redundant with
RecServe's β. Two claims are deliberately reported as replications rather than
discoveries — `exp(min logprob)` is Chow-Quantile(α=0) from Gupta et al.
(ICLR 2024), and the confidence/difficulty blindness is the documented
hard–easy effect (Michael et al.) — with the contribution being the transfer to
a multi-tier cascade with measured energy.
