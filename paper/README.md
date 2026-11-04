# Paper

`paper.tex` — *What a Protein–Ligand Affinity Model Actually Learns: A Preprocessing Bug,
Three Null Ablations, and a Proximity Ceiling on CASF-2016.*

A self-contained, dependency-light LaTeX source (standard `article` class plus `booktabs`,
`amsmath`, `hyperref`, `microtype`, `textcomp`). It compiles on a stock TeX Live, Overleaf,
or arXiv with no external style files or data inputs — every number is inlined.

```bash
pdflatex paper.tex && pdflatex paper.tex   # twice, to resolve refs
```

All quantitative claims trace to `../RESULTS.md` and the committed run records. The paper
was not compiled in the environment where it was written (no local TeX toolchain); it
passed a structural lint (brace/math-delimiter balance, environment and table-column
matching, special-character escaping) but has not been run through `pdflatex`, so give it
one compile pass before submission.
