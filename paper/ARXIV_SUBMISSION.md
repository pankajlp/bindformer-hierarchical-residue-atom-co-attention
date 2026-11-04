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

**Why not cs.LG as primary.** Endorsement is required per *primary* category. Checking the
"Which authors of this paper are endorsers?" link on the closest related papers shows that
the researchers whose work actually matches this paper are qualified for `q-bio.BM`, not
`cs.LG`: Seokhyun Moon (PIGNet2, arXiv:2307.01066) and Haoyu Lin (DeepRLI,
arXiv:2401.10806) both endorse for q-bio.BM only. With cs.LG as primary you would have to
ask an ML researcher with no connection to protein-ligand binding, which runs against
arXiv's own advice to "choose an endorser whose work is related to the subject of your
article". (PLANET v2.0, arXiv:2601.07415, is filed under cs.LG, but its endorser page
would not load, so its authors' eligibility is unverified.)

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
- [ ] **Endorsement for the primary category.** You need exactly ONE positive endorsement
      per category — not several. arXiv states it is "inappropriate to email large numbers
      of potential endorsers at once", and reserves the right to revoke endorsement for
      policy violations. Send one personalised request at a time.
- [ ] Upload the **source tarball**, not the PDF.
- [ ] After upload, arXiv shows its own compiled PDF — **read it** before clicking submit.
- [ ] There is a ~24h hold before announcement; you can still withdraw during it.
