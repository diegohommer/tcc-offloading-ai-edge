# Reference papers

Local copies of papers cited in the TCC master document and LaTeX draft,
kept here for convenience while writing. **Not committed** (see
`.gitignore`) — licensing hasn't been checked per source, and this repo
may be public, so re-copy these from your own sources if you clone this
repo elsewhere; they won't come with it.

- `RecServe.pdf` — Wu et al., *Recursive Offloading for LLM Serving in
  Multi-tier Networks*, arXiv:2505.16502. Primary baseline; the vendored
  code in `implementation/src/recserve/vendor/` implements this paper.
- `TieredPONLLM.pdf` — Pakpahan and Hwang, *Enabling Software-Defined
  Tiered LLM Inference Continuum on Passive Optical Network*, IEEE
  Access vol. 14, 2026, doi:10.1109/ACCESS.2026.3651558. Reference
  architecture (user/onu/fog/cloud tiers, Fig. 1); CC BY 4.0.
- `EcoThink.pdf` — Li and Lu, *EcoThink: A Green Adaptive Inference
  Framework for Sustainable and Accessible Agents*, WWW 2026. Closest
  related-work metric (gCO2/query).
- `TowardsGreenLLM.pdf` — Solovyeva and Castor, *Towards Green AI:
  Decoding the Energy of LLM Inference in Software Development*.
  Method reference for J/token-by-phase reporting.
- `ModelingLLMEnergyConsumption.pdf` — Raskind, Babakol, Mahmoud, and
  Liu, *VESTA: Power Modeling with Language Runtime Events*, PLDI 2024,
  doi:10.1145/3656402. Background reading on power-modeling methodology
  (not yet cited in the master document).
