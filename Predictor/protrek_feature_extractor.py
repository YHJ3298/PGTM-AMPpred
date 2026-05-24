# -*- coding: utf-8 -*-
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import DataLoader, Dataset


# =========================================================
# FASTA Parsing
# =========================================================
def parse_fasta(file_path: str) -> List[Tuple[str, str]]:
    file_path = str(file_path).strip()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file does not exist or path is incorrect: {file_path}")

    data: List[Tuple[str, str]] = []
    protein_name = None
    seq_chunks: List[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if protein_name is not None:
                    data.append((protein_name, "".join(seq_chunks)))

                protein_name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if protein_name is not None and seq_chunks:
        data.append((protein_name, "".join(seq_chunks)))

    if len(data) == 0:
        raise ValueError("No sequences parsed from FASTA file, please check the FASTA format.")

    return data


class FastaDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


# =========================================================
# Sequence Cleaning
# =========================================================
_VALID_AA = set(list("ACDEFGHIKLMNPQRSTVWY") + ["X"])
_REPLACE_TO_X = set(list("BJOUZ"))


def normalize_aa(seq: str, max_len: int = 1024) -> str:
    seq = seq.strip().replace(" ", "").upper()

    if not seq:
        return ""

    if len(seq) > max_len:
        seq = seq[:max_len]

    seq = "".join(("X" if ch in _REPLACE_TO_X else ch) for ch in seq)
    seq = "".join((ch if ch in _VALID_AA else "X") for ch in seq)

    return seq


# =========================================================
# Check ProTrek_650M Local Model Directory
# =========================================================
def _ensure_protrek650m_layout(model_root: str) -> dict:
    model_root = Path(model_root)

    if not model_root.exists():
        raise FileNotFoundError(f"ProTrek model directory does not exist: {model_root}")

    ckpt = model_root / "ProTrek_650M.pt"

    if not ckpt.exists():
        pts = list(model_root.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt checkpoint found under {model_root}")
        ckpt = pts[0]

    protein_dir = model_root / "esm2_t33_650M_UR50D"
    text_dir = model_root / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    structure_dir = model_root / "foldseek_t30_150M"

    for p in [protein_dir, text_dir, structure_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required ProTrek subdirectory: {p}")

    return {
        "ckpt": ckpt,
        "protein_dir": protein_dir,
        "text_dir": text_dir,
        "structure_dir": structure_dir,
    }


# =========================================================
# faiss patch
# =========================================================
def _patch_faiss_if_missing():
    try:
        import faiss  # noqa: F401
        return
    except ModuleNotFoundError:
        import types
        import importlib.machinery

        class _FaissStub(types.ModuleType):
            def __getattr__(self, item):
                raise ModuleNotFoundError(
                    "faiss is not installed in the current environment.\n"
                    "This software is only used for ProTrek feature extraction and AMP prediction, and generally does not require faiss.\n"
                    "If you use ProTrek retrieval functionality later, please install:\n"
                    "conda install -c conda-forge faiss-cpu"
                )

        stub = _FaissStub("faiss")
        stub.__spec__ = importlib.machinery.ModuleSpec(
            name="faiss",
            loader=None,
        )
        sys.modules["faiss"] = stub

        print(f"[{time.ctime()}] [WARN] faiss not detected, placeholder module injected.")


# =========================================================
# torchmetrics patch
# =========================================================
def _patch_torchmetrics_for_protrek_inference():
    """
    ProTrek inference only requires get_protein_repr to extract protein features,
    and does not need the actual torchmetrics.

    Note:
    The Dummy module must explicitly set __file__, __package__ and other attributes here;
    Meanwhile, __getattr__ cannot handle double underscore attributes, otherwise inspect will misjudge __file__
    as DummyMetric, resulting in:
    AttributeError: type object 'DummyMetric' has no attribute 'endswith'
    """
    import sys
    import types
    import importlib.machinery
    import torch
    import torch.nn as nn

    class DummyMetric(nn.Module):
        full_state_update = False
        higher_is_better = None
        is_differentiable = False

        def __init__(self, *args, **kwargs):
            super().__init__()

        def update(self, *args, **kwargs):
            pass

        def compute(self):
            return torch.tensor(0.0)

        def reset(self):
            pass

        def clone(self, *args, **kwargs):
            return self

        def to(self, *args, **kwargs):
            return self

        def add_state(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)

    class DummyMetricCollection(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def update(self, *args, **kwargs):
            pass

        def compute(self):
            return {}

        def reset(self):
            pass

        def clone(self, *args, **kwargs):
            return self

        def to(self, *args, **kwargs):
            return self

    def _dummy_function(*args, **kwargs):
        return torch.tensor(0.0)

    def _make_dummy_package(module_name: str):
        mod = types.ModuleType(module_name)
        mod.__spec__ = importlib.machinery.ModuleSpec(
            name=module_name,
            loader=None,
            is_package=True,
        )
        mod.__path__ = []
        mod.__file__ = f"<dummy-{module_name}>"
        mod.__package__ = module_name.rpartition(".")[0]
        return mod

    # torchmetrics main module
    tm = _make_dummy_package("torchmetrics")
    tm.Metric = DummyMetric
    tm.MetricCollection = DummyMetricCollection

    metric_class_names = [
        "Accuracy",
        "BinaryAccuracy",
        "MulticlassAccuracy",
        "MultilabelAccuracy",
        "F1Score",
        "BinaryF1Score",
        "MulticlassF1Score",
        "MultilabelF1Score",
        "AUROC",
        "BinaryAUROC",
        "MulticlassAUROC",
        "MultilabelAUROC",
        "AveragePrecision",
        "BinaryAveragePrecision",
        "MulticlassAveragePrecision",
        "MultilabelAveragePrecision",
        "Precision",
        "BinaryPrecision",
        "MulticlassPrecision",
        "MultilabelPrecision",
        "Recall",
        "BinaryRecall",
        "MulticlassRecall",
        "MultilabelRecall",
        "MatthewsCorrCoef",
        "BinaryMatthewsCorrCoef",
        "MulticlassMatthewsCorrCoef",
        "MeanSquaredError",
        "MeanAbsoluteError",
        "SpearmanCorrCoef",
        "PearsonCorrCoef",
        "R2Score",
    ]

    for name in metric_class_names:
        setattr(tm, name, DummyMetric)

    def _tm_getattr(name):
        # Critical: Double underscore attributes must raise AttributeError, not return DummyMetric
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return DummyMetric

    tm.__getattr__ = _tm_getattr

    # torchmetrics.functional
    functional = _make_dummy_package("torchmetrics.functional")

    functional_names = [
        "accuracy",
        "binary_accuracy",
        "multiclass_accuracy",
        "multilabel_accuracy",
        "f1_score",
        "binary_f1_score",
        "multiclass_f1_score",
        "multilabel_f1_score",
        "auroc",
        "binary_auroc",
        "multiclass_auroc",
        "multilabel_auroc",
        "average_precision",
        "binary_average_precision",
        "multiclass_average_precision",
        "multilabel_average_precision",
        "precision",
        "binary_precision",
        "multiclass_precision",
        "multilabel_precision",
        "recall",
        "binary_recall",
        "multiclass_recall",
        "multilabel_recall",
        "matthews_corrcoef",
        "mean_squared_error",
        "mean_absolute_error",
        "spearman_corrcoef",
        "pearson_corrcoef",
        "r2_score",
    ]

    for name in functional_names:
        setattr(functional, name, _dummy_function)

    def _functional_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _dummy_function

    functional.__getattr__ = _functional_getattr

    # torchmetrics.classification
    classification = _make_dummy_package("torchmetrics.classification")

    for name in metric_class_names:
        setattr(classification, name, DummyMetric)

    def _classification_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return DummyMetric

    classification.__getattr__ = _classification_getattr

    # torchmetrics.regression
    regression = _make_dummy_package("torchmetrics.regression")

    for name in metric_class_names:
        setattr(regression, name, DummyMetric)

    def _regression_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return DummyMetric

    regression.__getattr__ = _regression_getattr

    # torchmetrics.utilities
    utilities = _make_dummy_package("torchmetrics.utilities")

    def _utilities_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _dummy_function

    utilities.__getattr__ = _utilities_getattr

    # Force override to prevent real torchmetrics from being imported
    sys.modules["torchmetrics"] = tm
    sys.modules["torchmetrics.functional"] = functional
    sys.modules["torchmetrics.classification"] = classification
    sys.modules["torchmetrics.regression"] = regression
    sys.modules["torchmetrics.utilities"] = utilities

    print(f"[{time.ctime()}] [WARN] Dummy torchmetrics injected, only for ProTrek inference.")

# =========================================================
# Load ProTrek_650M
# =========================================================
def load_protrek_650m_local(
    device: torch.device,
    model_root: str,
    protrek_code_root: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
):
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    paths = _ensure_protrek650m_layout(model_root)

    if not protrek_code_root:
        here = Path(__file__).resolve().parent
        guess = here / "ProTrek-main"
        protrek_code_root = str(guess.resolve()) if guess.exists() else ""

    protrek_code_root = str(Path(protrek_code_root).resolve())

    if not Path(protrek_code_root).exists():
        raise RuntimeError(
            f"ProTrek-main source code directory not found: {protrek_code_root}\n"
            "Please ensure ProTrek-main is in the same directory level as this project."
        )

    if protrek_code_root not in sys.path:
        sys.path.insert(0, protrek_code_root)

    # Critical: Must patch first, then import ProTrekTrimodalModel
    _patch_faiss_if_missing()
    _patch_torchmetrics_for_protrek_inference()

    from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

    config = {
        "protein_config": str(paths["protein_dir"]),
        "text_config": str(paths["text_dir"]),
        "structure_config": str(paths["structure_dir"]),
        "from_checkpoint": str(paths["ckpt"]),
    }

    log(f"[{time.ctime()}] Starting to load ProTrek_650M")
    log(f"  model_root       = {model_root}")
    log(f"  protrek_code_root= {protrek_code_root}")
    log(f"  protein_config   = {config['protein_config']}")
    log(f"  text_config      = {config['text_config']}")
    log(f"  structure_config = {config['structure_config']}")
    log(f"  from_checkpoint  = {config['from_checkpoint']}")

    model = ProTrekTrimodalModel(**config).to(device)
    model.eval()

    log(f"[{time.ctime()}] ProTrek_650M loaded successfully")
    return model


# =========================================================
# Extract Embeddings from FASTA
# =========================================================
def extract_features_from_fasta(
    fasta_file: str,
    client,
    batch_size: int = 32,
    max_length: int = 1024,
    log_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log(f"[{time.ctime()}] Starting to parse FASTA file: {fasta_file}")

    items = parse_fasta(fasta_file)
    dataset = FastaDataset(items)

    def collate_fn(batch):
        names = [x[0] for x in batch]
        raw_seqs = [x[1] for x in batch]

        norm_seqs = [
            normalize_aa(seq, max_len=max_length)
            for seq in raw_seqs
        ]

        keep = [
            (name, raw_seq, norm_seq)
            for name, raw_seq, norm_seq in zip(names, raw_seqs, norm_seqs)
            if norm_seq
        ]

        if not keep:
            return [], [], []

        names, raw_seqs, norm_seqs = zip(*keep)

        return list(names), list(raw_seqs), list(norm_seqs)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_names: List[str] = []
    all_sequences: List[str] = []
    all_lengths: List[int] = []
    all_embs: List[torch.Tensor] = []

    total_batches = len(data_loader)

    log(f"[{time.ctime()}] Starting to extract ProTrek_650M features")
    log(f"  Number of sequences: {len(items)}")
    log(f"  batch_size : {batch_size}")
    log(f"  max_length : {max_length}")

    with torch.inference_mode():
        for batch_idx, (names, raw_seqs, norm_seqs) in enumerate(data_loader, start=1):
            if not names:
                continue

            # Note: The model uses cleaned norm_seqs for feature extraction
            embs = client.get_protein_repr(norm_seqs)
            embs = embs.float().detach().cpu()

            # Keep the original sequences from FASTA (raw_seqs) in the result display
            all_names.extend(names)
            all_sequences.extend(raw_seqs)
            all_lengths.extend([len(seq) for seq in raw_seqs])
            all_embs.append(embs)

            if progress_callback:
                progress_callback(batch_idx, total_batches)

    if not all_embs:
        raise RuntimeError("Failed to extract any features, please check the FASTA input.")

    embeddings = torch.cat(all_embs, dim=0)

    log(f"[{time.ctime()}] Feature extraction completed: {tuple(embeddings.shape)}")

    return all_names, all_sequences, all_lengths, embeddings