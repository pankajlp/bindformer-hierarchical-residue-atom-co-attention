from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local PDBbind raw-data directory.")
    parser.add_argument("--data_dir", default="data/raw")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    readme = data_dir / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Download PDBbind v2020 general/refined sets and CASF-2016 archives from "
            "http://www.pdbbind.org.cn/ after registration and extract them here (index/, v2020-other-PL/, refined-set/, CASF-2016/).\n",
            encoding="utf-8",
        )
    print(f"Ready for PDBbind archives: {data_dir.resolve()}")


if __name__ == "__main__":
    main()

