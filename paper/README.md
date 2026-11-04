# Paper

`paper.tex` — *What a Protein–Ligand Affinity Model Actually Learns: A Preprocessing Bug,
Three Null Ablations, and a Proximity Ceiling on CASF-2016.*

A self-contained, dependency-light LaTeX source (standard `article` class plus `booktabs`,
`amsmath`, `hyperref`, `microtype`, `textcomp`). It compiles on a stock TeX Live, Overleaf,
or arXiv with no external style files or data inputs — every number is inlined.

```bash
pdflatex paper.tex && pdflatex paper.tex   # twice, to resolve refs
```

All quantitative claims trace to `../RESULTS.md` and the committed run records. It compiles
cleanly (no overfull boxes or warnings) with Tectonic 0.17 (`tectonic paper.tex`), which
produced the committed `paper.pdf` (9 pages); arXiv's TeX Live builds the same source.
