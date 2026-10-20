# PGTM-AMPpred Explainability Analysis

This package contains the code and archived results for the four explainability analyses in Sections 3.5.1–3.5.4 of the PGTM-AMPpred manuscript. It includes complete module entry points, the saved analysis tables, five manuscript figures with PNG previews, and a script for checking archived values and figure specifications.

The package is prepared to be placed at **PGTM-AMPpred/Train/explainability/** in the [PGTM-AMPpred repository](https://github.com/YHJ3298/PGTM-AMPpred). Large feature tables, FASTA datasets, ProTrek_650M weights, and the trained classifier checkpoint are not duplicated here; the scripts read them from the existing Train directory.

| Manuscript section | Module directory | Analysis | Figures |
| --- | --- | --- | --- |
| 3.5.1 | 01_representation_space_reorganization | UMAP, five-seed stability, and high-dimensional representation metrics | Figures 5–6 |
| 3.5.2 | 02_attribution_fidelity | Cumulative alanine perturbation, within-peptide random controls, and randomized model parameters | Figure 7 |
| 3.5.3 | 03_local_evidence_robustness | X masking, alanine scanning, window shuffling, prediction changes, and representation shifts | Figure 8 |
| 3.5.4 | 04_residue_level_case_studies | Representative cases selected from X-masking windows and residue-level sensitivity profiles | Figure 9 |

The manuscript presents the modules in the order 1 → 2 → 3 → 4. The full computation runs in the order 1 → 3 → 2 → 4: Module 2 uses the interpretation cohort and alanine-window results produced by Module 3, and Module 4 uses Module 3's X-masking results. The top-level run_pipeline.py handles these dependencies and renders the five figures after the calculations finish.

## Figure files

The five archived figures are formatted at approximately 17 cm wide and use Arial, RGB color, 600 dpi, and lossless LZW-compressed TIFF. Each TIFF has a PNG preview. Use the TIFF files for manuscript submission and the PNG files for quick review. Figure captions should remain in the manuscript and follow the Figure 5–9 numbering.

common/journal_figures.py contains the shared figure styling and export functions. render_all.py redraws the five figures from the saved results folders. audit_results.py checks the archived numerical summaries and TIFF color mode, resolution, and dimensions; its report is validation_report.json.

## Archived results and manuscript notes

The included results correspond to an independent test set of 3,072 peptides, with 1,536 AMPs and 1,536 non-AMPs.

- In Module 1, the silhouette score changes from 0.0362 to 0.3054, 5-nearest-neighbor accuracy from 0.7454 to 0.7731, and the cosine interclass separation margin from 0.2157 to 0.6469. The saved high-dimensional metrics and paired-difference table match the archived values.
- The Module 2 interpretation cohort contains 304 peptides, with 76 each in the TP, FN, FP, and TN groups. The first three cumulative windows have complete paired results for 289 peptides. The selected-minus-random AOPC differences for steps 1–3 are 6.539, 5.926, and 4.283. The absolute mean differences for the five randomized-parameter models are each below 0.0002.
- Module 3's archived results include the 12 perturbation means across the four prediction groups, paired statistical tests, and the high-confidence false-positive subgroup. Of 33 high-confidence false positives, 32 have valid backgrounds for the paired test.
- Module 4 selects ALL000830, ALL003142, and sp|Q91G85|009R_IIV6 as representative cases. The saved residue-level sensitivity extrema and positions match Table 8 in the manuscript.

**Manuscript wording to correct before submission:** Section 3.5.1 currently describes the 5-nearest-neighbor result as leave-one-out accuracy. The code fits the 5-NN classifier on training-set representations and evaluates it on independent-test representations. Please update that sentence to describe an independent-test evaluation; the archived values and figures do not need to change.

UMAP is a stochastic nonlinear projection. A complete rerun previously reproduced the independent-test predictions and high-dimensional statistics, while the gated UMAP coordinates and five-seed projection values showed environment-sensitive variation. The included Figure 5 and five-seed results therefore retain the confirmed archived coordinates and statistics; the submission figure reflects layout and export formatting changes only.

The archived validation report lists PASS for the saved numerical summaries and TIFF specifications. The supplied version reports that Module 1 was rerun end to end, Modules 2 and 3 were recalculated from saved inference records rather than rerun over all perturbed sequences, and Module 4 was recalculated from all saved X-masking windows. It does not claim that the full ProTrek mutation inference for Modules 2 and 3 was rerun. Model inference was not run during this translation and packaging step. See validation_report.json for the archived report scope.

## Inputs expected under Train

The full pipeline expects the following existing files and directories:

    Train/
    ├── train_protrek650m_gated_tabm.csv
    ├── positive_protrek650m_test.csv
    ├── negative_protrek650m_test.csv
    ├── explain_outputs/
    │   ├── gated_tabm_model.pt
    │   └── gated_tabm_independent_predictions.csv
    ├── datasets/XUAMP/XU_test/
    │   ├── positive/XU_AMP.fasta
    │   └── negative/XU_nonAMP.fasta
    ├── Models/ProTrek_650M/
    └── ProTrek-main/

The model directory should contain ProTrek_650M.pt and the protein, text, and structure encoder files expected by ProTrek. Downloaded model weights and the ProTrek source code are external dependencies and are not included in this package.

## Requirements

Use Python 3.9 or later. Install the direct dependencies listed in requirements.txt. Install a PyTorch build that matches the local CPU or CUDA environment, and install any additional packages required by the external ProTrek source code.

## Run the complete analysis

From this package directory, run:

    python run_pipeline.py --source-root .. --model-root ../Models/ProTrek_650M --protrek-code-root ../ProTrek-main --output-root ./rerun_outputs

The output root must be new or empty. The scripts do not overwrite a nonempty output directory. Modules 2 and 3 encode and score many perturbed sequences, so a full run can be computationally expensive. The run produces one output directory per module and redraws the figures from those outputs.

To redraw figures from the included result tables, run:

    python render_all.py

To check the included archived results and figure specifications, run:

    python audit_results.py
