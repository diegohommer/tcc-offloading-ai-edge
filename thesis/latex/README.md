# TCC, LaTeX

Built on the INF/UFRGS class `infufrgs` (https://github.com/schnorr/infufrgs)
with options `[cic,dipl,english]`: Computer Science, undergraduate thesis
(Trabalho de Conclusão), written in English.

## Files

- `tcc.tex` — the thesis. At present a skeleton: metadata, the English
  Abstract and the Portuguese Resumo (both empty), contents and lists, the
  chapter and section headings, and the references.
- `tcc.bib` — the references cited so far, their metadata checked against the
  original pages (title, authors, venue, DOI/arXiv).
- `infufrgs.cls` — the INF/UFRGS class (UFRGS TeX Users Group), unmodified.
- `abntex2cite.sty`, `abntex2-alf.bst`, `abntex2-options.bib` — the abnTeX2
  ABNT citation package, bundled because the system this was set up on lacked
  a complete abnTeX2. An installation that already has abnTeX2 (most TeX Live
  and MiKTeX installs, and Overleaf) can use its own.
- `abntex2-alf-en.bst` — a copy of `abntex2-alf.bst` whose Portuguese labels
  are translated ("Available at", "Accessed", "and", "chap.", "Tech. Rep.",
  English month abbreviations …). The ABNT author-date layout is unchanged.
  `tcc.tex` uses this one.
- `watch.sh` — recompiles `tcc.pdf` whenever a source file changes (polling;
  `latexmk -pvc` hangs in this environment).

## What stays in Portuguese, by design

- The title page's institutional lines (Universidade Federal do Rio Grande do
  Sul, Instituto de Informática, Curso de Ciência da Computação), the
  "Trabalho de Conclusão" label and the page listing university officials:
  the class prints them as the university's names.
- The Resumo (`translatedabstract`) with its title (`\translatedtitle`) and
  keywords (`\translatedkeyword`), which UFRGS requires beside the Abstract.
- The class's `agradecimentos` environment has a fixed Portuguese heading:
  for acknowledgments, use `\chapter*{Acknowledgments}` instead.

## Building

```sh
latexmk -pdf tcc.tex
```

or by hand: `pdflatex tcc`, `bibtex tcc`, `pdflatex tcc`, `pdflatex tcc`.
Builds without errors with TeX Live 2023 (pdfTeX).

## Citing

The ABNT author-date style: `\cite{key}` gives (WU et al., 2025),
`\citeonline{key}` gives Wu et al. (2025).

## Open items

- The title and the chapter plan still describe the project's earlier
  framing; the case study (`../../energy_tests.md` §8.6) is the one to write up.
- `tcc.bib` has 11 of the sources `energy_tests.md` cites.
