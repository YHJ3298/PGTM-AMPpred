# PGTM-AMPpred

**PGTM-AMPpred** is an interpretable antimicrobial peptide (AMP) identification framework based on **ProTrek_650M**, **Gated Adapter**, and **TabM**. It uses tri-modal pretrained protein representations to extract sequence-level embeddings, applies a lightweight gated adaptation module for task-specific feature reweighting, and performs AMP/non-AMP classification with a parameter-efficient TabM ensemble classifier.

This project focuses on the task of identifying antimicrobial peptides (AMPs) and provides a complete code workflow ranging from **feature extraction using ProTrek_650M, training and testing with Gated-TabM, and interpretability analysis, through to the deployment of a local predictor**.

---

## Highlights

- **Tri-modal pretrained representation**: uses ProTrek_650M sequence embeddings pretrained with sequence, structure, and function-text alignment.
- **Lightweight task adaptation**: introduces a Gated Adapter to perform task-specific mapping and feature-wise reweighting before classification.
- **Parameter-efficient ensemble classifier**: uses TabM to generate multiple prediction branches inside one model, improving robustness without heavy explicit ensembling.
- **Interpretable analysis**: supports UMAP-based representation visualization and fragment-level perturbation analysis.
- **No explicit structure preprocessing required**: prediction only requires peptide sequences and the local ProTrek_650M model.
- **Complete local workflow**: includes training scripts, independent testing, explainability scripts, and a local AMP predictor.

---

## Framework

![PGTM-AMPpred_framework.png](.\PGTM-AMPpred\assets\PGTM-AMPpred_framework.png.png)

The overall workflow contains five major steps:

1. Standardize peptide sequences and replace non-standard amino acids with `X`.
2. Extract sequence-level embeddings using the local ProTrek_650M model.
3. Apply a Gated Adapter to map and reweight pretrained features for AMP recognition.
4. Use TabM for AMP/non-AMP binary classification.
5. Evaluate the model through cross-validation, independent testing, and explainability analysis.

---

## Repository Structure

```text
PGTM-AMPpred/
├── Predictor/                         # Local AMP predictor
│   ├── .idea/
│   ├── examples/                      # Example input files for prediction
│   ├── models/                        # Trained Gated-TabM checkpoint for local prediction
│   ├── outputs/                       # Local prediction outputs
│   ├── ProTrek-main/                  # Local ProTrek source code used by the predictor
│   ├── app.py                         # Predictor entry script
│   ├── config.py                      # Predictor configuration
│   ├── gated_tabm_model.py            # Gated-TabM model definition
│   ├── predictor_engine.py            # Prediction pipeline
│   └── protrek_feature_extractor.py   # ProTrek_650M feature extractor for prediction
│
├── Train/                             # Training, testing, and explainability scripts
│   ├── .idea/
│   ├── datasets/                      # Dataset directory
│   ├── explain_outputs/               # Independent-test checkpoint and outputs
│   ├── explainability_outputs/        # Explainability results
│   ├── gated_tabm_cv_outputs/         # 5-fold CV outputs
│   ├── Models/                        # Local pretrained model directory
│   │   └── ProTrek_650M/              # Downloaded ProTrek_650M model files
│   ├── ProTrek-main/                  # Local ProTrek source code used by training scripts
│   ├── 5-fold-CV.py                   # 5-fold cross-validation
│   ├── explainability_pipeline_utils.py
│   ├── explainability_run_all.py
│   ├── explainability_stage1_umap.py
│   ├── explainability_stage2_fragments.py
│   ├── independent-test.py            # Independent test and checkpoint export
│   ├── negative_protrek650m_test.csv
│   ├── negative_protrek650m_train.csv
│   ├── positive_protrek650m_test.csv
│   ├── positive_protrek650m_train.csv
│   ├── ProTrek650M-AMP-Embedding.py   # Batch ProTrek_650M embedding extraction
│   ├── test_protrek650m_gated_tabm.csv
│   └── train_protrek650m_gated_tabm.csv
│
└── README.md
```

> Note: `.idea/`, large pretrained weights, intermediate caches, generated outputs, and temporary result files are not recommended for Git tracking.

---

## Requirements

Recommended environment:

- Python >= 3.9
- CUDA-enabled GPU is recommended for ProTrek_650M feature extraction
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

If you use GPU acceleration, install the PyTorch version matching your CUDA environment from the official PyTorch installation guide.

---

## External Model Preparation

This project uses a local copy of **ProTrek_650M**. The pretrained model can be downloaded from Hugging Face:

```text
https://huggingface.co/westlake-repl/ProTrek_650M
```

According to the current project structure, place the downloaded model files under:

```text
PGTM-AMPpred/Train/Models/ProTrek_650M/
```

The expected model directory should contain `ProTrek_650M.pt` and the required encoder subdirectories:

```text
PGTM-AMPpred/Train/Models/ProTrek_650M/
├── ProTrek_650M.pt
├── esm2_t33_650M_UR50D/
├── BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/
└── foldseek_t30_150M/
```

When running scripts inside the `Train/` directory, the following path points to the correct local model directory:

```python
model_root = r"./Models/ProTrek_650M"
```

For example:

```bash
cd Train
python ProTrek650M-AMP-Embedding.py
```

In this case, `./Models/ProTrek_650M` is resolved as:

```text
PGTM-AMPpred/Train/Models/ProTrek_650M/
```

If you run scripts from another directory, use an absolute path or modify the relative path accordingly. The same rule applies to `--protrek_model_root` in the explainability commands.

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

The embedding extraction script will generate CSV files in `Train/`:

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

Because the script is executed in `Train/`, `model_root = r"./Models/ProTrek_650M"` corresponds to:

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
- replace ambiguous/non-standard residues such as `B`, `J`, `O`, `U`, `Z` with `X`;
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

1. Put the trained checkpoint into:

```text
Predictor/models/gated_tabm_model.pt
```

2. Check and modify paths in:

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

3. Start the predictor:

How to start the local predictor:

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

| Model | SN | SP | ACC | MCC | AUC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| PGTM-AMPpred | 0.9566 | 0.9574 | 0.9570 | 0.9142 | 0.9913 | 0.9570 |

### Independent Test on XUAMP Test Set

| Model | SN | SP | ACC | MCC | AUC | F1 |
|---|---:|---:|---:|---:|---:|---:|
| PGTM-AMPpred | 0.6133 | 0.9505 | 0.7819 | 0.5989 | 0.8662 | 0.7377 |

Compared with representative AMP prediction methods, PGTM-AMPpred shows improved sensitivity and a better sensitivity-specificity trade-off on the XUAMP independent test set.

---

## Output Files

| File / Directory | Description |
|---|---|
| `Train/positive_protrek650m_train.csv` | ProTrek_650M features for positive training samples |
| `Train/negative_protrek650m_train.csv` | ProTrek_650M features for negative training samples |
| `Train/train_protrek650m_gated_tabm.csv` | Merged labeled training feature table |
| `Train/gated_tabm_cv_outputs/` | 5-fold CV metrics, OOF predictions, and curves |
| `Train/explain_outputs/gated_tabm_model.pt` | Final trained Gated-TabM checkpoint |
| `Train/explain_outputs/gated_tabm_independent_predictions.csv` | Independent test prediction results |
| `Train/explain_outputs/gated_tabm_independent_metrics.json` | Independent test metrics |
| `Train/explainability_outputs/stage1_umap/` | UMAP visualization results |
| `Train/explainability_outputs/stage2_fragments/` | Fragment-level explainability results |
| `Predictor/outputs/` | Local predictor outputs |

---

## Notes and Troubleshooting

### 1. ProTrek_650M directory not found

The ProTrek_650M pretrained model can be downloaded from Hugging Face:

```text
https://huggingface.co/westlake-repl/ProTrek_650M
```

After downloading, place the model files under:

```text
PGTM-AMPpred/Train/Models/ProTrek_650M/
```

The model directory should contain `ProTrek_650M.pt` and the required encoder subdirectories:

```text
Train/Models/ProTrek_650M/
├── ProTrek_650M.pt
├── esm2_t33_650M_UR50D/
├── BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/
└── foldseek_t30_150M/
```

If you execute commands inside `Train/`, keep the path as:

```python
model_root = r"./Models/ProTrek_650M"
```

For command-line arguments, use:

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

For feature extraction, `faiss` is generally not required. The embedding script includes a compatibility patch that activates a stub module when `faiss` is unavailable. If you need retrieval/indexing functions from ProTrek, install it manually:

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

## Citation

If you use this repository in your research, please cite the corresponding paper after publication.

```bibtex
@article{PGTM_AMPpred,
  title   = {PGTM-AMPpred: An Interpretable Antimicrobial Peptide Identification Framework Based on a Tri-modal Protein Language Model},
  author  = {Hongjin Yan, Yun Zuo},
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
