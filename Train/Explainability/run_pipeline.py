"""Execute all four complete explainability modules from features/checkpoint/FASTA.

Dependency order: representation -> perturbation -> AOPC -> residue cases.
This is a computationally heavy run and writes only to a new --output-root.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Train directory containing features, checkpoints, and datasets/")
    parser.add_argument("--model-root", type=Path, required=True,
                        help="Local ProTrek_650M model directory")
    parser.add_argument("--protrek-code-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    model = args.model_root.resolve()
    protrek = (args.protrek_code_root or source / "ProTrek-main").resolve()
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dirs = {name: output / name for name in (
        "01_representation_space_reorganization",
        "02_attribution_fidelity",
        "03_local_evidence_robustness",
        "04_residue_level_case_studies",
    )}
    commands = [
        [sys.executable, str(PACKAGE / "01_representation_space_reorganization" / "code" / "run_full.py"),
         "--source-root", str(source), "--output-dir", str(dirs["01_representation_space_reorganization"])],
        [sys.executable, str(PACKAGE / "03_local_evidence_robustness" / "code" / "run_full.py"),
         "--source-root", str(source), "--model-root", str(model),
         "--protrek-code-root", str(protrek), "--output-dir", str(dirs["03_local_evidence_robustness"])],
        [sys.executable, str(PACKAGE / "02_attribution_fidelity" / "code" / "run_full.py"),
         "--source-root", str(source), "--model-root", str(model),
         "--protrek-code-root", str(protrek),
         "--perturbation-root", str(dirs["03_local_evidence_robustness"]),
         "--output-dir", str(dirs["02_attribution_fidelity"])],
        [sys.executable, str(PACKAGE / "04_residue_level_case_studies" / "code" / "run_full.py"),
         "--perturbation-root", str(dirs["03_local_evidence_robustness"]),
         "--output-dir", str(dirs["04_residue_level_case_studies"])],
        [sys.executable, str(PACKAGE / "render_all.py"),
         "--input-root", str(output), "--output-root", str(output)],
    ]
    for index, command in enumerate(commands, 1):
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    print(f"Complete four-module run: {output}")


if __name__ == "__main__":
    main()
