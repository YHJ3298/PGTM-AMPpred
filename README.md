# PGTM-AMPpred

**PGTM-AMPpred** is an interpretable antimicrobial peptide (AMP) identification framework based on **ProTrek_650M**, **Gated Adapter**, and **TabM**. The framework first extracts sequence-level representations from a tri-modal pretrained protein language model, then performs task-specific feature reweighting with a lightweight Gated Adapter, and finally predicts AMP/non-AMP labels using a parameter-efficient TabM ensemble classifier.

This repository provides the complete workflow for **feature extraction, model training, independent testing, explainability analysis, and local AMP prediction**.

---

## Highlights

- **Tri-modal protein representation**: uses ProTrek_650M sequence embeddings pretrained through sequence, structure, and function-text alignment.
- **Lightweight task adaptation**: introduces a Gated Adapter for task-specific feature transformation and feature-wise reweighting.
- **Parameter-efficient ensemble learning**: adopts TabM to produce multiple prediction branches within a single model.
- **Interpretable prediction pipeline**: supports UMAP-based representation visualization and fragment-level perturbation analysis.
- **No explicit structure preprocessing required**: the downstream prediction workflow only requires peptide sequences and local ProTrek_650M weights.
- **End-to-end local workflow**: includes training scripts, independent-test scripts, explainability modules, and a deployable local predictor.

---

## Framework

![PGTM-AMPpred framework](./PGTM-AMPpred_framework.png)

PGTM-AMPpred contains five main steps:

1. Standardize peptide sequences and replace non-standard amino acids with `X`.
2. Extract sequence-level embeddings using the local ProTrek_650M model.
3. Refine pretrained features with a Gated Adapter for AMP-specific representation learning.
4. Classify AMP/non-AMP samples using TabM.
5. Evaluate the model using 5-fold cross-validation, independent testing, and explainability analysis.

---

## Repository Structure

```text
PGTM-AMPpred/
├── Predictor/                              # Local AMP predictor
│   ├── examples/                           # Example input files for local prediction
│   ├── models/
│   │   ├── ProTrek_650M/                   # Download from Hugging Face and place here
│   │   └── gated_tabm_model.pt             # Trained Gated-TabM checkpoint for prediction
│   ├── outputs/                            # Local prediction outputs
│   ├── ProTrek-main/                       # Local ProTrek source code for the predictor
│   ├── app.py                              # Predictor entry script
│   ├── config.py                           # Predictor configuration
│   ├── gated_tabm_model.py                 # Gated-TabM model definition
│   ├── predictor_engine.py                 # Prediction pipeline
│   └── protrek_feature_extractor.py        # ProTrek_650M feature extractor for prediction
│
├── Train/                                  # Training, testing, and explainability scripts
│   ├── datasets/                           # Dataset directory
│   ├── explain_outputs/                    # Independent-test checkpoint and outputs
│   ├── explainability_outputs/             # Explainability results
│   ├── gated_tabm_cv_outputs/              # 5-fold CV outputs
│   ├── Models/
│   │   └── ProTrek_650M/                   # Download from Hugging Face and place here
│   ├── ProTrek-main/                       # Local ProTrek source code for training scripts
│   ├── 5-fold-CV.py                        # 5-fold cross-validation
│   ├── independent-test.py                 # Independent test and checkpoint export
│   ├── ProTrek650M-AMP-Embedding.py        # Batch ProTrek_650M embedding extraction
│   ├── explainability_pipeline_utils.py
│   ├── explainability_run_all.py
│   ├── explainability_stage1_umap.py       # UMAP-based explainability analysis
│   ├── explainability_stage2_fragments.py  # Fragment-level perturbation analysis
│   ├── positive_protrek650m_train.csv
│   ├── negative_protrek650m_train.csv
│   ├── positive_protrek650m_test.csv
│   └── negative_protrek650m_test.csv
│
├── PGTM-AMPpred_framework.png              # Framework figure used in README
└── README.md
```

> Large pretrained weights, generated checkpoints, intermediate feature tables, cache files, and temporary analysis outputs are not required for source-code tracking. They can be regenerated or distributed through external links/releases when necessary.

---

## Requirements

Recommended environment:

- Python >= 3.9
- CUDA-enabled GPU recommended for ProTrek_650M feature extraction
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- tqdm
- tabm
- torchmetrics
- transformers and other dependencies required by ProTrek

Install common dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib tqdm tabm torch torchmetrics transformers
```

If GPU acceleration is used, please install the PyTorch version matching your CUDA environment.

---

## External Model Preparation

This project depends on a local copy of **ProTrek_650M**. The pretrained model can be downloaded from Hugging Face:

```text
https://huggingface.co/westlake-repl/ProTrek_650M
```

After downloading, please place the ProTrek_650M model files in **both** of the following directories according to the current repository structure:

```text
PGTM-AMPpred/Predictor/models/ProTrek_650M/
PGTM-AMPpred/Train/Models/ProTrek_650M/
```

On Windows, the corresponding paths are:

```text
.\Predictor\models\ProTrek_650M
.\Train\Models\ProTrek_650M
```

The expected directory layout is:

```text
ProTrek_650M/
├── ProTrek_650M.pt
├── esm2_t33_650M_UR50D/
├── BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/
└── foldseek_t30_150M/
```

Therefore, the training-side model directory should look like:

```text
PGTM-AMPpred/Train/Models/ProTrek_650M/
├── ProTrek_650M.pt
├── esm2_t33_650M_UR50D/
├── BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/
└── foldseek_t30_150M/
```

The predictor-side model directory should look like:

```text
PGTM-AMPpred/Predictor/models/ProTrek_650M/
├── ProTrek_650M.pt
├── esm2_t33_650M_UR50D/
├── BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/
└── foldseek_t30_150M/
```

When running commands inside `Train/`, the relative path should be:

```python
model_root = r"./Models/ProTrek_650M"
```

When running the local predictor inside `Predictor/`, the relative path should be configured as:

```python
model_root = r"./models/ProTrek_650M"
```

The ProTrek source code should be placed under:

```text
PGTM-AMPpred/Train/ProTrek-main/
PGTM-AMPpred/Predictor/ProTrek-main/
```

or specified explicitly through script arguments/configuration where supported.

---

## Dataset Preparation

The training pipeline expects positive and negative AMP feature CSV files generated from FASTA sequences.

A typical raw dataset layout is:

```text
Train/datasets/XUAMP/
├── XU_train/
│   ├── positive/XU_AMP_train_positive.fasta
│   └── negative/XU_AMP_train_negative.fasta
└── XU_test/
    ├── positive/XU_AMP.fasta
    └── negative/XU_nonAMP.fasta
```

The embedding extraction script generates the following CSV files in `Train/`:

```text
positive_protrek650m_train.csv
negative_protrek650m_train.csv
positive_protrek650m_test.csv
negative_protrek650m_test.csv
```

Each generated CSV contains:

```text
protein_name, feature_0, feature_1, ..., feature_n
```

The training and testing scripts automatically add the binary label column:

- positive AMP samples: `label = 1`
- negative non-AMP samples: `label = 0`

---

## Step 1: Extract ProTrek_650M Embeddings

Enter the training directory:

```bash
cd Train
```

Before running, check the paths in `ProTrek650M-AMP-Embedding.py`, especially:

```python
tasks = [
    ("./datasets/XUAMP/XU_train/positive/XU_AMP_train_positive.fasta", "positive_protrek650m_train.csv"),
    ("./datasets/XUAMP/XU_train/negative/XU_AMP_train_negative.fasta", "negative_protrek650m_train.csv"),
    ("./datasets/XUAMP/XU_test/negative/XU_nonAMP.fasta", "negative_protrek650m_test.csv"),
    ("./datasets/XUAMP/XU_test/positive/XU_AMP.fasta", "positive_protrek650m_test.csv"),
]

model_root = r"./Models/ProTrek_650M"
```

Because the script is executed inside `Train/`, `model_root = r"./Models/ProTrek_650M"` corresponds to:

```text
PGTM-AMPpred/Train/Models/ProTrek_650M/
```

Run feature extraction:

```bash
python ProTrek650M-AMP-Embedding.py
```

The script will:

- parse FASTA files;
- normalize amino acid sequences;
- replace ambiguous or non-standard residues such as `B`, `J`, `O`, `U`, `Z` with `X`;
- truncate long sequences according to `max_length`;
- extract ProTrek_650M sequence embeddings;
- save feature CSV files for downstream training and testing.

---

## Step 2: 5-Fold Cross-Validation

五折的指令：

```bash
python 5-fold-CV.py --pos positive_protrek650m_train.csv --neg negative_protrek650m_train.csv
```

Default outputs:

```text
train_protrek650m_gated_tabm.csv

gated_tabm_cv_outputs/
├── gated_tabm_5fold_metrics.csv
├── gated_tabm_oof_predictions.csv
├── gated_tabm_5fold_summary.json
├── gated_tabm_roc_curve.png
└── gated_tabm_pr_curve.png
```

The model uses a fixed decision threshold of `0.5`.

Default selected Gated-TabM configuration:

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

---

## Step 3: Independent Test

独立测试的指令：

```bash
python independent-test.py --train_merged train_protrek650m_gated_tabm.csv --pos_test positive_protrek650m_test.csv --neg_test negative_protrek650m_test.csv --checkpoint_out explain_outputs/gated_tabm_model.pt --pred_out explain_outputs/gated_tabm_independent_predictions.csv --metrics_json explain_outputs/gated_tabm_independent_metrics.json
```

Default outputs:

```text
explain_outputs/
├── gated_tabm_model.pt
├── gated_tabm_independent_predictions.csv
└── gated_tabm_independent_metrics.json
```

The exported checkpoint can be used by the explainability scripts and the local predictor.

---

## Step 4: Explainability Analysis

### Stage 1: UMAP Visualization

可解释性分析stage1：

```bash
python explainability_stage1_umap.py --train_merged train_protrek650m_gated_tabm.csv --pos_test positive_protrek650m_test.csv --neg_test negative_protrek650m_test.csv --checkpoint explain_outputs/gated_tabm_model.pt --out_dir explainability_outputs/stage1_umap
```

This stage visualizes the representation space before and after Gated Adapter mapping.

### Stage 2: Fragment-Level Perturbation Analysis

可解释性分析stage2:

```bash
python explainability_stage2_fragments.py --train_merged train_protrek650m_gated_tabm.csv --pos_test positive_protrek650m_test.csv --neg_test negative_protrek650m_test.csv --pos_test_fasta ./datasets/XUAMP/XU_test/positive/XU_AMP.fasta --neg_test_fasta ./datasets/XUAMP/XU_test/negative/XU_nonAMP.fasta --protrek_model_root ./Models/ProTrek_650M --protrek_code_root ./ProTrek-main --checkpoint explain_outputs/gated_tabm_model.pt --out_dir explainability_outputs/stage2_fragments --window_size 7 --stride 1 --focus_label 1 --n_samples 20 --require_correct --top_n_per_seq 3
```

This command assumes it is executed inside `Train/`. Therefore:

- `--protrek_model_root ./Models/ProTrek_650M` points to `PGTM-AMPpred/Train/Models/ProTrek_650M/`;
- `--protrek_code_root ./ProTrek-main` points to `PGTM-AMPpred/Train/ProTrek-main/`;
- FASTA paths under `./datasets/` point to `PGTM-AMPpred/Train/datasets/`.

This stage uses sliding-window perturbation to identify local peptide fragments that strongly affect AMP prediction probability.

---

## Step 5: Local Predictor

The local predictor is located in:

```text
Predictor/
```

Recommended preparation:

1. Download ProTrek_650M from Hugging Face and place it under:

```text
PGTM-AMPpred/Predictor/models/ProTrek_650M/
```

2. Put the trained Gated-TabM checkpoint into:

```text
PGTM-AMPpred/Predictor/models/gated_tabm_model.pt
```

3. Check and modify paths in:

```text
Predictor/config.py
```

Important paths usually include:

```text
ProTrek_650M model root
ProTrek-main source code path
Gated-TabM checkpoint path
Output directory
```

Start the local predictor:

```bash
cd Predictor
python app.py
```

The predictor supports local AMP screening from user-provided peptide sequences or FASTA files and saves prediction results under:

```text
Predictor/outputs/
```

Typical output fields include:

```text
protein_name, sequence, AMP_probability, predicted_label
```

where `predicted_label = 1` denotes AMP and `predicted_label = 0` denotes non-AMP.

---

## Performance

### 5-Fold Cross-Validation on XUAMP Training Set

| Model        |     SN |     SP |    ACC |    MCC |    AUC |     F1 |
| ------------ | -----: | -----: | -----: | -----: | -----: | -----: |
| PGTM-AMPpred | 0.9566 | 0.9574 | 0.9570 | 0.9142 | 0.9913 | 0.9570 |

### Independent Test on XUAMP Test Set

| Model        |     SN |     SP |    ACC |    MCC |    AUC |     F1 |
| ------------ | -----: | -----: | -----: | -----: | -----: | -----: |
| PGTM-AMPpred | 0.6133 | 0.9505 | 0.7819 | 0.5989 | 0.8662 | 0.7377 |

Compared with representative AMP prediction methods, PGTM-AMPpred shows improved sensitivity and a better sensitivity-specificity trade-off on the XUAMP independent test set.

---

## Output Files

| File / Directory                                             | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `Train/positive_protrek650m_train.csv`                       | ProTrek_650M features for positive training samples          |
| `Train/negative_protrek650m_train.csv`                       | ProTrek_650M features for negative training samples          |
| `Train/train_protrek650m_gated_tabm.csv`                     | Merged labeled training feature table generated by 5-fold CV |
| `Train/gated_tabm_cv_outputs/`                               | 5-fold CV metrics, out-of-fold predictions, and ROC/PR curves |
| `Train/explain_outputs/gated_tabm_model.pt`                  | Final trained Gated-TabM checkpoint                          |
| `Train/explain_outputs/gated_tabm_independent_predictions.csv` | Independent test prediction results                          |
| `Train/explain_outputs/gated_tabm_independent_metrics.json`  | Independent test metrics                                     |
| `Train/explainability_outputs/stage1_umap/`                  | UMAP visualization results                                   |
| `Train/explainability_outputs/stage2_fragments/`             | Fragment-level explainability results                        |
| `Predictor/outputs/`                                         | Local predictor outputs                                      |

---

## Notes and Troubleshooting

### 1. ProTrek_650M directory not found

Download ProTrek_650M from Hugging Face:

```text
https://huggingface.co/westlake-repl/ProTrek_650M
```

Then place the downloaded files under the required path depending on the module you run:

```text
Training scripts:
PGTM-AMPpred/Train/Models/ProTrek_650M/

Local predictor:
PGTM-AMPpred/Predictor/models/ProTrek_650M/
```

Windows-style paths:

```text
.\Train\Models\ProTrek_650M
.\Predictor\models\ProTrek_650M
```

If you execute commands inside `Train/`, keep the path as:

```python
model_root = r"./Models/ProTrek_650M"
```

For command-line arguments in `Train/`, use:

```bash
--protrek_model_root ./Models/ProTrek_650M
```

If you execute commands from the project root `PGTM-AMPpred/`, use:

```python
model_root = r"./Train/Models/ProTrek_650M"
```

or:

```bash
--protrek_model_root ./Train/Models/ProTrek_650M
```

The key point is that all relative paths are resolved from the current working directory where the command is executed.

### 2. `faiss` is missing

For feature extraction, `faiss` is generally not required. The embedding script includes a compatibility patch that activates a stub module when `faiss` is unavailable. If retrieval/indexing functions from ProTrek are required, install it manually:

```bash
conda install -c conda-forge faiss-cpu
```

### 3. `torchmetrics` API mismatch

Newer `torchmetrics` versions require a `task` argument for some metrics. The embedding script contains an inference-only compatibility patch for this issue.

### 4. No numeric feature columns found

Training scripts only use numeric feature columns. Make sure your feature CSV contains columns like:

```text
feature_0, feature_1, ..., feature_n
```

The column `protein_name` is treated as metadata and is removed before model training.

---

## Reproducibility Notes

For best reproducibility:

- run all training and explainability commands from the `Train/` directory;
- keep the ProTrek_650M directory structure unchanged after downloading;
- use the fixed decision threshold `0.5` unless explicitly evaluating threshold sensitivity;
- record CUDA, PyTorch, and Python versions when reporting new results;
- keep generated feature files and checkpoints associated with the same model configuration.

---

## Citation

If you use this repository in your research, please cite the corresponding paper after publication.

```bibtex
@article{PGTM_AMPpred,
  title   = {PGTM-AMPpred: An Interpretable Antimicrobial Peptide Identification Framework Based on a Tri-modal Protein Language Model},
  author  = {Hongjin Yan and Yun Zuo},
  journal = {To be updated},
  year    = {2026}
}
```

---

## License

Please add a license according to your release plan. For academic open-source projects, common choices include MIT, Apache-2.0, and GPL-3.0.

---

## Contact

For questions, bug reports, or collaboration, please open an issue on GitHub.
