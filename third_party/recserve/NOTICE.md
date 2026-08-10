# Provenance

This directory is an unmodified vendored copy of the official RecServe
implementation, fetched for reproduction purposes as part of a Bachelor's
Thesis (TCC) on energy-aware offloading in hierarchical LLM inference.

- Upstream: https://github.com/wuzhiyuan2000/RecServe
- Commit: `3c953366998ec174162d101a3e1972a61c0f043e`
- Fetched: 2026-08-10
- No LICENSE file is published upstream at this commit. Code is reproduced
  here strictly for academic, non-distributed thesis work, citing the
  original publication below. Do not redistribute this directory outside
  that context without checking upstream licensing status first.

```bibtex
@article{wu2025recursive,
  author={Wu, Zhiyuan and Sun, Sheng and Wang, Yuwei and Liu, Min and Gao, Bo and Lu, Jinda and Wu, Tingting and Yang, Zheming and Wen, Tian},
  journal={IEEE Transactions on Mobile Computing},
  title={Recursive Offloading for LLM Serving in Multi-tier Networks},
  year={2025},
  pages={1-16},
  doi={10.1109/TMC.2025.3642580}
}
```

## Known scope limitation (important for this thesis)

The released code implements only the **Seq2Class** path described in the
paper: three RoBERTa-family text classifiers (end/edge/cloud), each doing a
single forward pass, escalating via a beta-quantile confidence threshold
over a sliding window of prior confidences. The paper's abstract also
describes a **Seq2Seq** path (generative LLMs, perplexity-based confidence)
for cascades of decoder models with prefill/decode phases, but that path is
**not present in this release**. There is no token generation, no decode
loop, and no output-token count in this codebase.

This matters for the thesis's energy-costing plan (see the project's root
`README.md`): the layer energy tables (`config/layer_energy.yaml`) are
calibrated from measurements of decoder LLM *decode* throughput (0.5B-70B+
class models), not tiny encoder classifiers (66M-355M class models here).
Do not treat energy numbers produced by running this vendored code as real
thesis results — see `README.md` at the repo root for how this is handled.
