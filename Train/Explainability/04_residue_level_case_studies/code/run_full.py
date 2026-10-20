"""Run manuscript section 3.5.4 from all independent-test X-masking windows.

Select the within-group median of each peptide's maximum sensitivity for TP,
FN, and FP; average overlapping length-7 window effects at each residue.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perturbation-root", type=Path, required=True,
                        help="Module 3 full-run output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.perturbation_root.resolve() / "data"
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir()
    windows = pd.read_csv(source / "perturbation_all_windows_X.csv")
    cohort = pd.read_csv(source / "perturbation_selected_cohort.csv")
    assert cohort.group.value_counts().to_dict() == {"TP": 76, "FN": 76, "FP": 76, "TN": 76}
    maxima = windows.groupby(["protein_name", "group"], as_index=False).delta_logit.max()
    selected_ids = []
    for group in ("TP", "FN", "FP"):
        part = maxima[maxima.group == group].sort_values("delta_logit")
        selected_ids.append(str(part.iloc[len(part) // 2].protein_name))
    selected = cohort[cohort.protein_name.isin(selected_ids)].copy()
    selected_windows = windows[windows.protein_name.isin(selected_ids)].copy()
    case_map = selected.set_index("protein_name")
    rows = []
    summary = []
    for case_id in selected_ids:
        case = case_map.loc[case_id]
        sequence = case.sequence
        scores = np.zeros(len(sequence), dtype=float)
        counts = np.zeros(len(sequence), dtype=int)
        for row in selected_windows[selected_windows.protein_name == case_id].itertuples(index=False):
            scores[int(row.start): int(row.end)] += float(row.delta_logit)
            counts[int(row.start): int(row.end)] += 1
        assert (counts > 0).all()
        scores /= counts
        rows.extend({"protein_name": case_id, "group": case.group, "position": int(i + 1),
                     "mean_delta_logit": float(score), "covering_windows": int(counts[i])}
                    for i, score in enumerate(scores))
        summary.append({"protein_name": case_id, "group": case.group, "length": len(sequence),
                        "amp_probability": float(case.prob), "maximum": float(scores.max()),
                        "maximum_position": int(scores.argmax() + 1), "minimum": float(scores.min()),
                        "minimum_position": int(scores.argmin() + 1)})
    selected.to_csv(results / "representative_cases.csv", index=False, encoding="utf-8-sig")
    selected_windows.to_csv(results / "example_X_windows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(results / "residue_sensitivity_profiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(results / "representative_case_summary.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
