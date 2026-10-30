from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_attention_heatmap(attention, out_path: str | Path, title: str = "Residue-Atom Attention") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(attention, cmap="viridis")
    plt.title(title)
    plt.xlabel("Ligand atom")
    plt.ylabel("Protein residue")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

