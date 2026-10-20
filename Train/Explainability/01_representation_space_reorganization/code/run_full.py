"""Run all computations for manuscript section 3.5.1 from the original features.

The UMAP experiment and high-dimensional metrics use the same frozen checkpoint.
Outputs go only to a new --output-dir. Journal figures are generated separately
from the resulting CSV files by the package-level render_all.py script.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Train directory containing feature CSV files and explain_outputs/")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stage1 = output / "stage1"
    highdim = output / "highdim"
    commands = [
        [sys.executable, str(CODE / "run_umap_full.py"),
         "--train_merged", str(source / "train_protrek650m_gated_tabm.csv"),
         "--pos_test", str(source / "positive_protrek650m_test.csv"),
         "--neg_test", str(source / "negative_protrek650m_test.csv"),
         "--checkpoint", str(source / "explain_outputs" / "gated_tabm_model.pt"),
         "--out_dir", str(stage1)],
        [sys.executable, str(CODE / "run_highdim_full.py"),
         "--source-root", str(source), "--output-dir", str(highdim)],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    results = output / "results"
    results.mkdir(exist_ok=True)
    for source_file in [stage1 / "umap_raw_embedding.csv", stage1 / "umap_gated_embedding.csv",
                        stage1 / "umap_seed_stability.csv", stage1 / "stage1_summary.json",
                        highdim / "representation_metric_estimates.csv",
                        highdim / "representation_paired_differences.csv",
                        highdim / "representation_per_sample_metrics.csv"]:
        shutil.copy2(source_file, results / source_file.name)
    print(f"Module 1 complete: {results}")


if __name__ == "__main__":
    main()
