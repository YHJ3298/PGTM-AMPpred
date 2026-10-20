# PGTM-AMPpred

**PGTM-AMPpred: Antimicrobial Peptide Identification and Interpretability Analysis Based on a Trimodal Protein Language Model and Gated Adapter**

PGTM-AMPpred is a computational framework for antimicrobial peptide (AMP) identification. It uses sequence-level representations from ProTrek_650M, adapts them with a lightweight Gated Adapter, and classifies AMP/non-AMP sequences with TabM. The repository provides scripts for feature extraction, model training, independent testing, four-module interpretability analysis, and local prediction.

> **Modality clarification.** ProTrek_650M was pretrained through sequence, structure, and function-text alignment. For the downstream experiments in this study, only its sequence-encoding interface was used; structural sequences and functional text were not supplied to the AMP classifier, and the three modalities were not fused again during downstream inference.

## Highlights

- Sequence-level embeddings from the frozen ProTrek_650M encoder (1,024 dimensions).
- A Gated Adapter for task-specific feature mapping and dimension-wise recalibration.
- TabM for parameter-efficient ensemble classification.
- Evaluation on the XUAMP training and independent test sets.
- Four complementary interpretability analyses, from representation geometry to residue-neighborhood sensitivity.
- A local prediction application for peptide sequences and FASTA files.

## Framework

![PGTM-AMPpred framework](./PGTM-AMPpred_framework.png)

The workflow is:

1. Parse and normalize peptide sequences.
2. Extract sequence-level representations with the frozen ProTrek_650M encoder.
3. Apply the Gated Adapter to adapt the representations to AMP classification.
4. Use TabM to predict AMP/non-AMP labels.
5. Evaluate performance with five-fold cross-validation and an independent test set, and characterize model behavior using the explainability analyses.

## Repository layout

```text
PGTM-AMPpred/
├── Predictor/                              # Local AMP prediction application
│   ├── examples/                            # Example prediction inputs
│   ├── models/                              # Local model weights/checkpoint
│   ├── outputs/                             # Local prediction results
│   ├── ProTrek-main/                        # ProTrek source code, if required locally
│   ├── app.py
│   ├── config.py
│   ├── gated_tabm_model.py
│   ├── predictor_engine.py
│   └── protrek_feature_extractor.py
├── Train/
│   ├── datasets/XUAMP/                      # XUAMP FASTA inputs
│   ├── Models/ProTrek_650M/                 # Local pretrained model files
│   ├── ProTrek-main/                        # ProTrek source code, if required locally
│   ├── explain_outputs/                     # Checkpoint and independent-test outputs
│   ├── explainability/                      # Four-module interpretability package
│   │   ├── 01_representation_space_reorganization/
│   │   ├── 02_attribution_fidelity/
│   │   ├── 03_local_evidence_robustness/
│   │   ├── 04_residue_level_case_studies/
│   │   ├── common/
│   │   ├── run_pipeline.py
│   │   ├── render_all.py
│   │   └── audit_results.py
│   ├── 5-fold-CV.py
│   ├── independent-test.py
│   └── ProTrek650M-AMP-Embedding.py
├── PGTM-AMPpred_framework.png
└── README.md
```

The model weights, generated checkpoints, feature tables, and analysis outputs can be large. Keep them out of source control unless they are intentionally being distributed; see `.gitignore` and the repository's release plan.

## Requirements

The explainability dependencies were pinned and checked with Python 3.9. Install a PyTorch build that matches your CPU or CUDA environment. ProTrek may require additional packages from its source repository.

From the repository root:

```bash
python -m pip install -r Train/explainability/requirements.txt
```

Install PyTorch separately using the command appropriate for your platform and CUDA version. For feature extraction and full perturbation analysis, a CUDA-enabled GPU is recommended, but availability and runtime depend on the local ProTrek setup.

## Pretrained model and ProTrek source

Download ProTrek_650M from [Hugging Face](https://huggingface.co/westlake-repl/ProTrek_650M). Place the downloaded model files in both locations if you use both training and prediction workflows:

```text
Train/Models/ProTrek_650M/
Predictor/models/ProTrek_650M/
```

The model directory should contain `ProTrek_650M.pt` and the sequence, structure, and text encoder files required by the released ProTrek implementation. Place the ProTrek source code under `Train/ProTrek-main/` and `Predictor/ProTrek-main/`, or provide its path to scripts that accept a path argument.

## Dataset and feature preparation

The training and independent-test workflows use the XUAMP splits and labels described in the manuscript. The expected FASTA layout is:

```text
Train/datasets/XUAMP/
├── XU_train/
│   ├── positive/XU_AMP_train_positive.fasta
│   └── negative/XU_AMP_train_negative.fasta
└── XU_test/
    ├── positive/XU_AMP.fasta
    └── negative/XU_nonAMP.fasta
```

Obtain the XUAMP data from its original source and follow its terms of use. The feature-extraction script writes these CSV files in `Train/`:

```text
positive_protrek650m_train.csv
negative_protrek650m_train.csv
positive_protrek650m_test.csv
negative_protrek650m_test.csv
```

Each file contains a `protein_name` metadata column and numeric embedding columns (`feature_0`, `feature_1`, …). Training and testing scripts assign the binary labels: AMP = `1`, non-AMP = `0`.

## Reproduction workflow

Run the training and feature-extraction commands from `Train/`, unless a command below specifies another working directory. Check the input, model, and ProTrek paths in the relevant script before running.

### 1. Extract ProTrek_650M embeddings

```bash
cd Train
python ProTrek650M-AMP-Embedding.py
```

The script parses the four FASTA files, normalizes sequence characters, applies the encoder's maximum sequence length, and saves sequence embeddings. In the XUAMP data used in the manuscript, all sequences were at most 100 residues and contained canonical amino acids.

### 2. Run five-fold cross-validation

```bash
python 5-fold-CV.py --pos positive_protrek650m_train.csv --neg negative_protrek650m_train.csv
```

The script creates the merged labeled training table and writes metrics, out-of-fold predictions, summary statistics, and ROC/PR curves under `gated_tabm_cv_outputs/`. The reported decision threshold is 0.5. The selected configuration was:

```python
BEST_CONFIG = {
    "k": 32,
    "arch_type": "tabm",
    "n_blocks": 2,
    "d_block": 256,
    "dropout": 0.1,
    "epochs": 200,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 20,
    "adapter_dropout": 0.1,
    "adapter_activation": "gelu",
    "early_stop_metric": "ROC_AUC",
}
```

### 3. Evaluate on the independent test set

```bash
python independent-test.py \
  --train_merged train_protrek650m_gated_tabm.csv \
  --pos_test positive_protrek650m_test.csv \
  --neg_test negative_protrek650m_test.csv \
  --checkpoint_out explain_outputs/gated_tabm_model.pt \
  --pred_out explain_outputs/gated_tabm_independent_predictions.csv \
  --metrics_json explain_outputs/gated_tabm_independent_metrics.json
```

The checkpoint is used by the explainability package and can also be configured for the local predictor.

### 4. Run the four-module explainability analysis

The package is located at `Train/explainability/`. Its expected inputs are existing files under `Train/`, including the merged training features, independent-test features, trained checkpoint, prediction records, XUAMP test FASTA files, ProTrek_650M weights, and ProTrek source code. The exact input tree and module details are documented in [`Train/explainability/README.md`](Train/explainability/README.md).

From the repository root:

```bash
cd Train/explainability
python run_pipeline.py \
  --source-root .. \
  --model-root ../Models/ProTrek_650M \
  --protrek-code-root ../ProTrek-main \
  --output-root ./rerun_outputs
```

The output directory must be new or empty. Modules run in dependency order **1 → 3 → 2 → 4**: the attribution-fidelity and residue-level modules use perturbation results generated by Module 3. Modules 2 and 3 encode and score perturbed sequences and may require substantial compute time.

The four modules are:

| Module | Analysis | Manuscript figures |
| --- | --- | --- |
| `01_representation_space_reorganization` | UMAP visualization, seed stability, and high-dimensional representation metrics | Figures 5–6 |
| `02_attribution_fidelity` | Cumulative alanine perturbation, within-peptide random controls, and randomized-model sanity checks | Figure 7 |
| `03_local_evidence_robustness` | X masking, alanine substitution, window shuffling, prediction changes, and representation shifts | Figure 8 |
| `04_residue_level_case_studies` | Residue-neighborhood sensitivity profiles and representative peptide cases | Figure 9 |

The analysis uses an independent test set of 3,072 sequences (1,536 AMPs and 1,536 non-AMPs). Local perturbation analyses use a balanced 304-peptide interpretation cohort (76 each in TP, FN, FP, and TN groups); 289 peptides have complete paired results for the top-three attribution windows and random-window controls. X masking can create out-of-distribution sequences, so its effects are interpreted alongside alanine substitution, window shuffling, and background controls. Sensitivity profiles describe model responses to residue neighborhoods and should not be interpreted as causal biological effects.

To redraw figures from saved results, run from `Train/explainability/`:

```bash
python render_all.py
```

To check archived summary values and figure specifications:

```bash
python audit_results.py
```

The archived validation report covers the saved numerical summaries and publication-figure specifications. It does not claim that every perturbation inference was recomputed from raw sequences during packaging; see `Train/explainability/README.md` and `Train/explainability/validation_report.json` for the precise validation scope. The included TIFF figures are RGB, approximately 17 cm wide, 600 dpi, and LZW-compressed; PNG files are provided as previews.

### 5. Run the local predictor

1. Place ProTrek_650M under `Predictor/models/ProTrek_650M/`.
2. Place the trained checkpoint at `Predictor/models/gated_tabm_model.pt`.
3. Review the model and output paths in `Predictor/config.py`.
4. Run:

```bash
cd Predictor
python app.py
```

Prediction results are saved under `Predictor/outputs/`. Typical output fields are `protein_name`, `sequence`, `AMP_probability`, and `predicted_label`; `predicted_label = 1` denotes AMP and `0` denotes non-AMP.

## Reported performance

The following values are reported in the manuscript on XUAMP. They are provided for reference; rerunning the workflow may produce small differences if software, hardware, or input files differ.

| Evaluation | SN | SP | ACC | MCC | AUC | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Five-fold cross-validation | 0.9566 | 0.9574 | 0.9570 | 0.9142 | 0.9913 | 0.9570 |
| Independent test | 0.6133 | 0.9505 | 0.7819 | 0.5989 | 0.8662 | 0.7377 |

SN, sensitivity; SP, specificity; ACC, accuracy; MCC, Matthews correlation coefficient; AUC, area under the ROC curve; F1, F1 score.

## Reproducibility and interpretation notes

- Run commands from the directories specified above; relative paths resolve from the current working directory.
- Keep the pretrained model files and ProTrek source tree in the expected locations, or pass their paths explicitly where supported.
- Use the fixed decision threshold of 0.5 when reproducing the reported classification metrics.
- Record Python, PyTorch, CUDA, and dependency versions for new runs. The explainability package includes a pinned `requirements.txt` for its direct dependencies.
- UMAP is a stochastic nonlinear projection. Its coordinates and pairwise geometry should not be compared directly across independently fitted projections; the manuscript also reports high-dimensional metrics.
- **5-NN manuscript alignment:** the saved analysis fits the 5-nearest-neighbor classifier on training-set representations and evaluates it on independent-test representations. The manuscript currently labels this as leave-one-out accuracy in one sentence; that wording should be corrected to independent-test evaluation. The archived value is unchanged.
- Model-sensitive regions indicate statistical patterns associated with the dataset labels and model outputs. They do not establish causal mechanisms or antimicrobial activity; experimental validation is required for biological claims.

## Citation

If you use this code or analysis, please cite the associated manuscript. The manuscript's publication details should be updated here when available.

```bibtex
@unpublished{yan2026pgtm_amppred,
  title  = {PGTM-AMPpred: Antimicrobial Peptide Identification and Interpretability Analysis Based on a Trimodal Protein Language Model and Gated Adapter},
  author = {Yan, Hongjin and Cao, Yujie and Zuo, Yun},
  year   = {2026},
  note   = {Manuscript; publication details to be updated}
}
```

## Funding

This work was supported by the National Natural Science Foundation of China (Grant No. 62302198) and the Basic Research Program of Jiangsu Province (Grant No. BK20231035).

## Authors and contact

- **Hongjin Yan** and **Yujie Cao** — School of Artificial Intelligence and Computer Science (School of Software), Jiangnan University, Wuxi 214122, China. Emails: `yanhongjin2003@163.com`; `caoyujie_0101@163.com`.
- **Yun Zuo** — School of Artificial Intelligence and Computer Science, Jiangnan University, Wuxi 214000, China. Corresponding author: `zuoyun@jiangnan.edu.cn`.

For code questions or bug reports, please open a GitHub issue. For matters requiring author input, contact the corresponding author.

## License

See the repository's `LICENSE` file for reuse terms. If no license file is present, contact the authors before redistributing or reusing the code.
