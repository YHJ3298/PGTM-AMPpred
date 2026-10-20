"""Audit manuscript values and technical specifications of final BDMA figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODULES = [
    "01_representation_space_reorganization",
    "02_attribution_fidelity",
    "03_local_evidence_robustness",
    "04_residue_level_case_studies",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def close(actual: float, expected: float, decimals: int = 3) -> None:
    if round(float(actual), decimals) != expected:
        raise AssertionError(f"Expected {expected} at {decimals} decimals; got {actual}")


def main() -> None:
    r1 = ROOT / MODULES[0] / "results"
    r2 = ROOT / MODULES[1] / "results"
    r3 = ROOT / MODULES[2] / "results"
    r4 = ROOT / MODULES[3] / "results"

    raw = pd.read_csv(r1 / "umap_raw_embedding.csv")
    gated = pd.read_csv(r1 / "umap_gated_embedding.csv")
    assert len(raw) == len(gated) == 3072
    assert raw.label.value_counts().to_dict() == {0: 1536, 1: 1536}
    assert np.array_equal(raw.protein_name.astype(str).to_numpy(), gated.protein_name.astype(str).to_numpy())
    metrics = pd.read_csv(r1 / "representation_metric_estimates.csv")
    differences = pd.read_csv(r1 / "representation_paired_differences.csv")
    expected = {"silhouette": (0.0362, 0.3054, 0.2692),
                "knn_accuracy": (0.7454, 0.7731, 0.0277),
                "centroid_margin": (0.2157, 0.6469, 0.4312)}
    for key, target in expected.items():
        sub = metrics[metrics.metric == key].set_index("representation")
        diff = differences[differences.metric == key].iloc[0]
        for actual, expected_value in zip((sub.loc["Raw ProTrek", "estimate"],
                                           sub.loc["Gated Adapter", "estimate"],
                                           diff.adapted_minus_raw), target):
            close(actual, expected_value, 4)
        assert diff.ci_low > 0
    stability = pd.read_csv(r1 / "umap_seed_stability.csv")
    assert set(stability.seed) == {7, 21, 42, 101, 2026}
    assert stability.groupby("representation").size().to_dict() == {"Raw ProTrek": 5, "Gated Adapter": 5}

    cohort = pd.read_csv(r2 / "perturbation_selected_cohort.csv")
    assert cohort.group.value_counts().to_dict() == {"TP": 76, "FN": 76, "FP": 76, "TN": 76}
    aopc = pd.read_csv(r2 / "aopc_cumulative_alanine_statistics.csv")
    aopc_targets = {1: (6.432, -0.107, 6.539), 2: (7.017, 1.092, 5.926),
                    3: (7.103, 2.820, 4.283)}
    for step, target in aopc_targets.items():
        row = aopc[(aopc.step == step) & (aopc.group == "All")].iloc[0]
        assert int(row.n_peptides) == 289
        for actual, expected_value in zip((row.selected_mean_evidence_removed,
                                           row.random_mean_evidence_removed,
                                           row.selected_minus_random), target):
            close(actual, expected_value)
    assert aopc.fdr_q.max() < 0.0001
    randomized = pd.read_csv(r2 / "randomized_model_sanity_summary.csv")
    random_rows = randomized[randomized.model == "Randomized model"]
    assert set(random_rows.seed.astype(int)) == {17, 73, 211, 907, 2027}
    assert random_rows["mean"].abs().max() < 0.0002

    robustness = pd.read_csv(r3 / "perturbation_robustness_summary.csv")
    table7 = {
        "TP": (3.159, 1.824, 1.525),
        "FN": (2.773, 0.154, 0.615),
        "FP": (2.651, 1.712, 1.068),
        "TN": (1.366, -0.094, 0.420),
    }
    for group, target in table7.items():
        for method, expected_value in zip(("X masking", "Alanine scan", "Window shuffle"), target):
            row = robustness[(robustness.group == group) & (robustness.role == "important")
                             & (robustness.method == method)].iloc[0]
            assert int(row.n_peptides) == 76
            close(row.mean_delta_logit, expected_value)
    peptide = pd.read_csv(r3 / "peptide_level_perturbation_results.csv")
    shifts = peptide.groupby(["method", "role"]).embedding_cosine_shift.mean()
    for method, imp, bg in (("X masking", 0.049, 0.030),
                            ("Alanine scan", 0.117, 0.092),
                            ("Window shuffle", 0.042, 0.029)):
        close(shifts[method, "important"], imp)
        close(shifts[method, "background"], bg)
    high_fp = pd.read_csv(r3 / "high_confidence_false_positive_statistics.csv")
    assert high_fp[high_fp.fp_subset.str.startswith("High-confidence FP")].n_peptides.eq(32).all()

    profiles = pd.read_csv(r4 / "residue_sensitivity_profiles.csv")
    cases = pd.read_csv(r4 / "representative_cases.csv")
    case_expectations = {
        "ALL000830": (57, 2.168, 9, -0.786, 1),
        "ALL003142": (91, 1.608, 74, -2.450, 90),
        "sp|Q91G85|009R_IIV6": (85, 2.731, 75, -1.683, 17),
    }
    for case_id, (length, maximum, max_pos, minimum, min_pos) in case_expectations.items():
        case = cases[cases.protein_name == case_id].iloc[0]
        part = profiles[profiles.protein_name == case_id]
        assert len(case.sequence) == len(part) == length
        top = part.loc[part.mean_delta_logit.idxmax()]
        bottom = part.loc[part.mean_delta_logit.idxmin()]
        close(top.mean_delta_logit, maximum)
        close(bottom.mean_delta_logit, minimum)
        assert int(top.position) == max_pos and int(bottom.position) == min_pos

    images = sorted(ROOT.rglob("*.tiff"))
    assert len(images) == 5
    figure_specs = []
    for path in images:
        with Image.open(path) as im:
            assert im.mode == "RGB"
            assert tuple(round(v) for v in im.info.get("dpi", (0, 0))) == (600, 600)
            width_cm = im.width / 600 * 2.54
            height_cm = im.height / 600 * 2.54
            assert abs(width_cm - 17) < 0.02
            assert height_cm < 19
            figure_specs.append({"file": str(path.relative_to(ROOT)), "pixels": [im.width, im.height],
                                 "dpi": 600, "mode": im.mode,
                                 "width_cm": round(width_cm, 3), "height_cm": round(height_cm, 3),
                                 "sha256": digest(path)})
    report = {
        "status": "PASS",
        "scope": "Archived full-inference outputs; modules 1–4 numeric summaries and journal-image specifications",
        "figures": figure_specs,
        "independent_test_n": 3072,
        "cohort_n": 304,
        "aopc_complete_n": 289,
        "important_note": "5-NN was trained on training representations and evaluated on independent-test representations, not leave-one-out.",
    }
    (ROOT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS: manuscript Tables 6–8, Figures 5–9 source values, and TIFF format/size/dpi")


if __name__ == "__main__":
    main()
