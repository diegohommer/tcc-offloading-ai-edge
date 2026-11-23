# Thesis and slides

| Folder | What | Entry point |
|---|---|---|
| `document/` | The thesis, UFRGS `infufrgs` class, ABNT author-date citations | `tcc.tex` |
| `slides/` | The defense slides, Beamer with the INF theme | `main.tex` |

Both folders are mirrors of Overleaf projects, so edits made there are pulled before any
local change.

## Building

```bash
(cd document && latexmk -pdf tcc.tex)
(cd slides && latexmk -pdf main.tex)
```

## Figures

- `document/figures/*.pdf` are copies of `implementation/results/study/figures/`, written by
  `implementation/src/analyze/plot_study.py`. Regenerate them there and copy them over.
- `*/figures/*.tex` are TikZ diagrams drawn by hand.
- `pakpahan2026_fig1_tiered_continuum.jpg` is Figure 1 of Pakpahan and Hwang (IEEE Access,
  2026), reproduced with attribution.
