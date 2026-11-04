# arXiv submission sheet

Copy-paste into the arXiv submission form. Upload `arxiv/bindformer-arxiv.tar.gz`
(source only — arXiv builds the PDF itself; do not upload `paper.pdf`).

---

## Title
What a Protein-Ligand Affinity Model Actually Learns: A Preprocessing Bug, Three Null Ablations, and a Proximity Ceiling on CASF-2016

## Authors
Pankaj Kumar

## Abstract (plain text — paste as-is)

We report a controlled empirical study of BindFormer, a residue-atom co-attention model
that predicts protein-ligand binding affinity (pK) from a complex with a known crystal
pose, evaluated for scoring power on CASF-2016. The study is organized around an
evaluation protocol built *before* any architectural change: every comparison uses three
random seeds and a paired bootstrap over the 285 test complexes, so that an effect the
size of a typical architectural tweak (0.01-0.03 Pearson R) can be told apart from seed
and test-set noise. The protocol repays its cost immediately. Our single-seed baseline
reaches R = 0.748 +/- 0.011, but trivial predictors that use no binding information reach
R = 0.50 (ligand heavy-atom count) and R = 0.51 (pocket residue count), so roughly two
thirds of the raw correlation is explained by molecular size and pocket identity. A
diagnostic baseline then exposed a train/test preprocessing bug -- hydrogen atoms were
silently retained in the CASF ligands but not the training ligands -- whose correction
raised Pearson R by about 0.08 everywhere and reversed two conclusions that had otherwise
looked significant, including an apparent +0.14 benefit of ESM-2 protein embeddings that
corrected to +0.05 at p = 0.095. Of four architectural or objective interventions, three
produced no significant change on CASF and one significantly hurt, each with an
accompanying diagnostic confirming the intervention was mechanistically active rather
than inert. A pose-perturbation analysis explains the pattern: rotating the ligand in
place costs only 0.02-0.03 Pearson R, while translating it out of the pocket drives the
score to zero. BindFormer scores *composition and proximity*, not interaction geometry,
and refining how attention weights distances cannot help a model that does not read fine
geometry. We offer the protocol, the diagnostics, and the preprocessing lesson as
transferable, and release all code.

## Comments (optional field)
9 pages. Code and full evidence tree (per-seed run records, per-complex predictions,
diagnostics) at https://github.com/pankajlp/bindformer-hierarchical-residue-atom-co-attention

## Categories
- **Primary: `q-bio.BM`** (Quantitative Biology > Biomolecules) — best fit: protein-ligand
  binding affinity, CASF-2016 scoring power.
- Cross-list: **`cs.LG`** (Machine Learning) — the measurement protocol and ablation
  methodology.

## License
Recommended: **CC BY 4.0** — maximum reach and citation, you keep copyright.
Alternative if you want to limit reuse: **arXiv non-exclusive license to distribute**
(the most restrictive option arXiv offers; no redistribution rights granted).

## AI-assistance disclosure
Already in the author footnote of the paper: the study was carried out with substantial
assistance from an AI coding agent, which wrote the implementation, ran the experiments
and drafted the manuscript under the author's direction. Keep this — several venues now
require it.

## Before you submit — checks
- [ ] Your arXiv account has **endorsement for `q-bio.BM`**. New submitters usually need
      an endorsement from an established author in that category; without it the
      submission will not go through. Check this first — it is the most common blocker.
- [ ] Upload the **source tarball**, not the PDF.
- [ ] After upload, arXiv shows its own compiled PDF — **read it** before clicking submit.
- [ ] There is a ~24h hold before announcement; you can still withdraw during it.
