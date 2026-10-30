"""Parsers for PDBbind INDEX files, CASF core sets, and train/val/test splits."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

AFFINITY_PATTERN = re.compile(r"^(Kd|Ki|IC50)\s*(=|<=|>=|<|>|~)\s*([0-9.]+)\s*([a-zA-Z]+)$")


@dataclass
class IndexEntry:
    code: str
    resolution: float | None
    year: int | None
    pk: float
    measure: str | None
    relation: str | None
    source: str

    @property
    def is_exact(self) -> bool:
        return self.relation == "="


def parse_affinity_string(text: str) -> tuple[str | None, str | None]:
    """Parse e.g. `Kd=10mM` or `IC50<1nM` into (measure, relation)."""

    match = AFFINITY_PATTERN.match(text.strip())
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def parse_index_file(path: str | Path, source: str | None = None) -> dict[str, IndexEntry]:
    """Parse a PDBbind INDEX file or a CASF `CoreSet.dat`.

    Both share the leading columns `code resolution year pK affinity`.
    """

    path = Path(path)
    source = source or path.name
    entries: dict[str, IndexEntry] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4 or len(parts[0]) != 4:
                continue
            try:
                pk = float(parts[3])
            except ValueError:
                continue
            try:
                resolution = float(parts[1])
            except ValueError:
                resolution = None
            try:
                year = int(parts[2])
            except ValueError:
                year = None
            measure, relation = parse_affinity_string(parts[4]) if len(parts) > 4 else (None, None)
            code = parts[0].lower()
            entries[code] = IndexEntry(code, resolution, year, pk, measure, relation, source)
    return entries


def find_file(root: str | Path, names: list[str]) -> Path | None:
    root = Path(root)
    for name in names:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def load_pdbbind_index(root: str | Path) -> dict[str, IndexEntry]:
    """Load general + refined INDEX files found under `root`.

    The refined set is a subset of the general set; refined entries take precedence.
    """

    entries: dict[str, IndexEntry] = {}
    general = find_file(root, ["INDEX_general_PL_data.*", "INDEX_general_PL.*"])
    refined = find_file(root, ["INDEX_refined_data.*", "INDEX_refined_set.*"])
    for path, source in ((general, "general"), (refined, "refined")):
        if path is not None:
            entries.update(parse_index_file(path, source))
    if not entries:
        raise FileNotFoundError(f"No PDBbind INDEX files found under {root}")
    return entries


def load_core_set(path_or_root: str | Path | None) -> dict[str, IndexEntry]:
    """Load CASF-2016 core set entries from `CoreSet.dat` (or a directory containing it)."""

    if path_or_root is None:
        return {}
    path = Path(path_or_root)
    if path.is_dir():
        # CASF-2016 ships a CoreSet.dat per power test; the scoring one carries the affinities.
        found = find_file(path, ["power_scoring/CoreSet.dat", "CoreSet.dat"])
        if found is None:
            return {}
        path = found
    return parse_index_file(path, "casf2016")


def find_complex_dirs(root: str | Path) -> dict[str, Path]:
    """Find PDBbind-style complex directories `<code>/<code>_protein.pdb` under `root`."""

    found: dict[str, Path] = {}
    for protein in sorted(Path(root).rglob("*_protein.pdb")):
        code = protein.parent.name.lower()
        if len(code) == 4 and protein.name.lower() == f"{code}_protein.pdb":
            found.setdefault(code, protein.parent)
    return found


@dataclass
class SplitConfig:
    strategy: str = "random"  # "random" or "time"
    val_fraction: float = 0.1
    val_min_year: int = 2019  # used by the "time" strategy
    seed: int = 42
    exact_only: bool = False  # drop entries with <, >, ~ affinity relations
    refined_only: bool = False


def make_splits(
    entries: dict[str, IndexEntry],
    core_codes: set[str],
    available: set[str] | None = None,
    config: SplitConfig | None = None,
) -> dict[str, list[str]]:
    """Build train/val/casf2016 code lists. CASF-2016 complexes never enter train or val."""

    config = config or SplitConfig()
    pool = []
    for code, entry in sorted(entries.items()):
        if code in core_codes:
            continue
        if available is not None and code not in available:
            continue
        if config.exact_only and not entry.is_exact:
            continue
        if config.refined_only and entry.source != "refined":
            continue
        pool.append(code)

    if config.strategy == "time":
        val = [c for c in pool if (entries[c].year or 0) >= config.val_min_year]
        train = [c for c in pool if (entries[c].year or 0) < config.val_min_year]
    elif config.strategy == "random":
        shuffled = pool[:]
        random.Random(config.seed).shuffle(shuffled)
        n_val = int(round(len(shuffled) * config.val_fraction))
        val, train = sorted(shuffled[:n_val]), sorted(shuffled[n_val:])
    else:
        raise ValueError(f"Unknown split strategy: {config.strategy}")

    test = sorted(c for c in core_codes if available is None or c in available)
    return {"train": train, "val": val, "casf2016": test}


def save_splits(splits: dict[str, list[str]], path: str | Path, config: SplitConfig | None = None) -> None:
    payload = {"splits": splits, "config": asdict(config) if config else None}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["splits"]
