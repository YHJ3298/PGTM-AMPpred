"""Render all five BDMA-ready figures from the four modules' result folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE / "common"))
from journal_figures import figure5, figure6, figure7, figure8, figure9  # noqa: E402

MODULES = [
    "01_representation_space_reorganization",
    "02_attribution_fidelity",
    "03_local_evidence_robustness",
    "04_residue_level_case_studies",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PACKAGE,
                        help="Root containing the four module folders with results/ subfolders")
    parser.add_argument("--output-root", type=Path, default=PACKAGE,
                        help="Root to receive the four modules' figures/ subfolders")
    args = parser.parse_args()
    inp, out = args.input_root.resolve(), args.output_root.resolve()
    roots = {name: (inp / name / "results", out / name / "figures") for name in MODULES}
    figure5(*roots[MODULES[0]])
    figure6(*roots[MODULES[0]])
    figure7(*roots[MODULES[1]])
    figure8(*roots[MODULES[2]])
    figure9(*roots[MODULES[3]])
    print("Rendered five 600-dpi RGB TIFF files and five PNG previews")


if __name__ == "__main__":
    main()
