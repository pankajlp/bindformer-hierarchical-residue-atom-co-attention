from __future__ import annotations

import numpy as np
import pytest

from bindformer.data.ligand_featurizer import ATOM_FEATURE_DIM, MAX_SPD, LigandFeaturizer
from bindformer.data.pdbbind_index import (
    SplitConfig,
    find_complex_dirs,
    load_core_set,
    load_pdbbind_index,
    make_splits,
    parse_affinity_string,
    parse_index_file,
)
from bindformer.data.protein_featurizer import Esm2Embedder, OneHotEmbedder, ProteinFeaturizer
from bindformer.utils.pdb_utils import parse_protein_pdb


def test_parse_protein_drops_water_ions_hydrogens_and_altlocs(pdbbind_tree):
    code = pdbbind_tree["codes"][0]
    structure = parse_protein_pdb(next(pdbbind_tree["root"].rglob(f"{code}_protein.pdb")))
    assert len(structure.residues) == 40
    assert {r.resname for r in structure.residues}.isdisjoint({"HOH", "ZN"})
    assert all("H" not in r.atom_names for r in structure.residues)
    first = structure.residues[0]
    assert first.atom_names.count("CA") == 1
    assert np.abs(first.representative_coord).max() < 100  # altloc B CA at +100 A was ignored
    seqs = structure.chain_sequences()
    assert set(seqs) == {"A", "B"}
    assert "M" in seqs["A"][0]  # MSE -> MET


def test_ligand_featurizer_shapes(pdbbind_tree):
    code = pdbbind_tree["codes"][0]
    feats = LigandFeaturizer().featurize_sdf(next(pdbbind_tree["root"].rglob(f"{code}_ligand.sdf")))
    n = feats.x.shape[0]
    assert feats.x.shape == (n, ATOM_FEATURE_DIM) == (13, 74)  # aspirin heavy atoms
    assert "H" not in feats.atom_symbols
    assert feats.coords.shape == (n, 3)
    assert feats.spd.shape == (n, n)
    assert (feats.spd == feats.spd.T).all() and (feats.spd.diagonal() == 0).all()
    assert int(feats.spd.max()) <= MAX_SPD + 1
    assert feats.edge_index.shape == (2, 26) and feats.edge_attr.shape == (26, 6)
    assert (feats.x[:, :43].sum(dim=1) == 1).all()  # every atom has a known element


def test_index_and_affinity_parsing(tmp_path):
    path = tmp_path / "INDEX_general_PL_data.2020"
    path.write_text(
        "# header\n"
        "3zzf  2.20  2012   0.40  Ki=400mM      // 3zzf.pdf (NLG)\n"
        "1ABC  NMR   1999   6.00  IC50~1uM      // 1abc.pdf (XYZ)\n"
        "bad line\n",
        encoding="utf-8",
    )
    entries = parse_index_file(path)
    assert set(entries) == {"3zzf", "1abc"}
    assert entries["3zzf"].pk == pytest.approx(0.40) and entries["3zzf"].is_exact
    assert entries["1abc"].resolution is None and entries["1abc"].measure == "IC50" and not entries["1abc"].is_exact
    assert parse_affinity_string("Kd<=5.2nM") == ("Kd", "<=")


def test_splits_exclude_casf_and_are_disjoint(pdbbind_tree):
    root = pdbbind_tree["root"]
    entries = load_pdbbind_index(root)
    assert entries[pdbbind_tree["refined_codes"][0]].source == "refined"
    core = load_core_set(root)
    assert set(core) == set(pdbbind_tree["core_codes"])
    dirs = find_complex_dirs(root)
    assert set(dirs) == set(pdbbind_tree["codes"])

    splits = make_splits(entries, set(core), set(dirs), SplitConfig(val_fraction=0.25))
    train, val, test = map(set, (splits["train"], splits["val"], splits["casf2016"]))
    assert test == set(core)
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) + len(val) == len(pdbbind_tree["codes"]) - len(core)

    time_splits = make_splits(entries, set(core), set(dirs), SplitConfig(strategy="time", val_min_year=2018))
    assert all(entries[c].year >= 2018 for c in time_splits["val"])
    assert all(entries[c].year < 2018 for c in time_splits["train"])


def test_pocket_cropping_and_contacts(pdbbind_tree):
    code = pdbbind_tree["codes"][0]
    cdir = next(pdbbind_tree["root"].rglob(f"{code}_protein.pdb")).parent
    ligand = LigandFeaturizer().featurize_sdf(cdir / f"{code}_ligand.sdf")
    feats = ProteinFeaturizer(OneHotEmbedder(), pocket_cutoff=10.0).featurize_pocket(cdir / f"{code}_protein.pdb", ligand.coords)
    k = feats.x.shape[0]
    assert 0 < k < feats.num_residues_total
    assert feats.x.shape == (k, 21) and (feats.x.sum(dim=1) == 1).all()
    assert feats.contacts.shape == (k, ligand.x.shape[0])
    assert set(feats.contacts.unique().tolist()) <= {0.0, 1.0}
    # Pocket residues have some heavy atom within the cutoff; CA may sit slightly further out.
    dist = np.linalg.norm(feats.coords.numpy()[:, None] - ligand.coords.numpy()[None], axis=-1).min(axis=1)
    assert dist.max() < 10.0 + 3.0


def test_esm_windowing_averages_overlaps():
    embedder = object.__new__(Esm2Embedder)
    embedder.model_name, embedder.dim, embedder.max_len, embedder.stride, embedder.cache_dir = "fake", 2, 4, 2, None
    calls = []

    def fake_window(window):
        calls.append(window)
        return np.ones((len(window), 2), dtype=np.float32) * len(calls)

    embedder._embed_window = fake_window
    emb = embedder.embed("ABCDEFG")
    assert calls == ["ABCD", "CDEF", "EFG"]
    assert emb.shape == (7, 2)
    np.testing.assert_allclose(emb[:, 0], [1, 1, 1.5, 1.5, 2.5, 2.5, 3])
