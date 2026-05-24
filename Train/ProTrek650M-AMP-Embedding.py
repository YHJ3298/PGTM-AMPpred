import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


# --------------------------
# FASTA Parsing (Original Logic Retained)
# --------------------------
def parse_fasta(file_path: str) -> List[Tuple[str, str]]:
    file_path = file_path.strip()
    print(f"[{time.ctime()}] Start parsing FASTA file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file does not exist or invalid path: {file_path}")

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

    print(f"[{time.ctime()}] FASTA parsing completed, total {len(data)} sequences")
    return data


class FastaDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]  # (name, raw_seq)


# --------------------------
# Sequence Cleansing (Robust: non-standard characters replaced with X)
# --------------------------
_VALID_AA = set(list("ACDEFGHIKLMNPQRSTVWY") + ["X"])
_REPLACE_TO_X = set(list("BJOUZ"))  # Common ambiguous / non-standard characters


def normalize_aa(seq: str, max_len: int) -> str:
    seq = seq.strip().replace(" ", "").upper()
    if not seq:
        return ""
    if len(seq) > max_len:
        seq = seq[:max_len]

    seq = "".join(("X" if ch in _REPLACE_TO_X else ch) for ch in seq)
    seq = "".join((ch if ch in _VALID_AA else "X") for ch in seq)
    return seq


# --------------------------
# ProTrek_650M Local Weight Structure Verification
# --------------------------
def _ensure_protrek650m_layout(model_root: str) -> dict:
    model_root = Path(model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"Model directory not found: {model_root}")

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
            raise FileNotFoundError(f"Missing required subdirectory: {p}")

    return {
        "ckpt": ckpt,
        "protein_dir": protein_dir,
        "text_dir": text_dir,
        "structure_dir": structure_dir,
    }


# --------------------------
# Patch 1: Inject stub module with __spec__ when faiss is missing
# Avoid ValueError caused by null __spec__ in transformers import check
# --------------------------
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
                    "faiss is not installed in current environment.\n"
                    "Faiss is generally unnecessary for feature extraction. Stub module enabled.\n"
                    "Install faiss if retrieval/index functions are required:\n"
                    "  conda install -c conda-forge faiss-cpu\n"
                )

        stub = _FaissStub("faiss")
        stub.__spec__ = importlib.machinery.ModuleSpec(name="faiss", loader=None)
        sys.modules["faiss"] = stub
        print(f"[{time.ctime()}] [WARN] faiss not detected, stub module activated (feature extraction available)")


# --------------------------
# Patch 2: Resolve API incompatibility between new torchmetrics and ProTrek
# Accuracy requires task parameter in new version
# Dummy metrics injected as metrics are unused during embedding inference
# --------------------------
def _patch_torchmetrics_for_protrek_inference():
    import torch
    import torchmetrics
    from torchmetrics import Metric

    # Skip patch if old compatible version exists
    try:
        _ = torchmetrics.Accuracy()
        return
    except TypeError as e:
        if "task" not in str(e):
            return

    class DummyMetric(Metric):
        full_state_update = False

        def __init__(self, *args, **kwargs):
            super().__init__()

        def update(self, *args, **kwargs):
            pass

        def compute(self):
            return torch.tensor(0.0)

    for name in [
        "Accuracy", "F1Score", "AUROC", "AveragePrecision",
        "Precision", "Recall", "MatthewsCorrCoef",
        "MeanSquaredError", "MeanAbsoluteError"
    ]:
        if hasattr(torchmetrics, name):
            setattr(torchmetrics, name, DummyMetric)

    print(f"[{time.ctime()}] [WARN] torchmetrics API mismatch fixed with dummy metrics (inference only)")


# --------------------------
# Local ProTrek_650M Loading
# --------------------------
def load_protrek_650m_local(
    device: torch.device,
    model_root: str,
    protrek_code_root: Optional[str] = None,
):
    paths = _ensure_protrek650m_layout(model_root)

    # Locate ProTrek source directory
    if not protrek_code_root:
        here = Path(__file__).resolve().parent
        guess = here / "ProTrek-main"
        protrek_code_root = str(guess) if guess.exists() else ""

    protrek_code_root = str(Path(protrek_code_root).resolve())
    if not Path(protrek_code_root).exists():
        raise RuntimeError(
            f"ProTrek source repository not found: {protrek_code_root}\n"
            "Ensure ProTrek-main is placed in the same directory, or specify path manually."
        )

    # Add source path to system import
    if protrek_code_root not in sys.path:
        sys.path.insert(0, protrek_code_root)

    # Apply patches prior to model import (critical execution order)
    _patch_faiss_if_missing()
    _patch_torchmetrics_for_protrek_inference()

    from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

    config = {
        "protein_config": str(paths["protein_dir"]),
        "text_config": str(paths["text_dir"]),
        "structure_config": str(paths["structure_dir"]),
        "from_checkpoint": str(paths["ckpt"]),
    }

    print(f"[{time.ctime()}] Start loading local ProTrek_650M: {Path(model_root)}")
    print(f"  protrek_code_root = {protrek_code_root}")
    print(f"  protein_config    = {config['protein_config']}")
    print(f"  text_config       = {config['text_config']}")
    print(f"  structure_config  = {config['structure_config']}")
    print(f"  from_checkpoint   = {config['from_checkpoint']}")

    model = ProTrekTrimodalModel(**config).to(device)
    model.eval()
    print(f"[{time.ctime()}] ProTrek_650M loaded successfully")
    return model


# --------------------------
# Embedding Extraction (Original function name, parameters and output format reserved)
# --------------------------
def extract_features_esmc(
    fasta_file: str,
    client,
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 1024,
) -> Tuple[List[str], torch.Tensor]:
    print(f"[{time.ctime()}] Start feature extraction via ProTrek_650M")

    items = parse_fasta(fasta_file)
    dataset = FastaDataset(items)

    def collate_fn(batch):
        names = [x[0] for x in batch]
        seqs = [normalize_aa(x[1], max_length) for x in batch]
        keep = [(n, s) for n, s in zip(names, seqs) if s]
        if not keep:
            return [], []
        names, seqs = zip(*keep)
        return list(names), list(seqs)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_names: List[str] = []
    all_embs: List[torch.Tensor] = []

    with torch.inference_mode():
        for names, seqs in tqdm(data_loader, desc="Processing batches", unit="batch"):
            if not names:
                continue

            embs = client.get_protein_repr(seqs)  # (B, H)
            embs = embs.float().detach().cpu()

            all_names.extend(names)
            all_embs.append(embs)

    embeddings = torch.cat(all_embs, dim=0) if all_embs else torch.empty((0, 0))
    print(f"[{time.ctime()}] Feature extraction finished: {embeddings.shape}")
    return all_names, embeddings


def save_features_as_csv(names: List[str], embeddings: torch.Tensor, output_file: str):
    print(f"[{time.ctime()}] Saving features to CSV file: {output_file}")

    emb_np = embeddings.numpy()
    column_names = [f"feature_{i}" for i in range(emb_np.shape[1])]
    df = pd.DataFrame(emb_np, columns=column_names)
    df.insert(0, "protein_name", names)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[{time.ctime()}] Features saved to {output_file}")


def main():
    print(f"[{time.ctime()}] Program started")

    # --------------------------
    # Batch FASTA files and corresponding CSV outputs
    # --------------------------
    tasks = [
        (
            r"./datasets/XUAMP/XU_train/positive/XU_AMP_train_positive.fasta",
            "positive_protrek650m_train.csv"
        ),
        (
            r"./datasets/XUAMP/XU_train/negative/XU_AMP_train_negative.fasta",
            "negative_protrek650m_train.csv"
        ),
        (
            r"./datasets/XUAMP/XU_test/negative/XU_nonAMP.fasta",
            "negative_protrek650m_test.csv"
        ),
        (
            r"./datasets/XUAMP/XU_test/positive/XU_AMP.fasta",
            "positive_protrek650m_test.csv"
        ),
    ]

    # --------------------------
    # Model directory path
    # --------------------------
    model_root = r".\Models\Protrek_650M"

    # ProTrek-main located at the same directory as current script
    protrek_code_root = str(
        (Path(__file__).resolve().parent / "ProTrek-main").resolve()
    )

    # --------------------------
    # Computing device configuration
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------
    # Load model only once
    # --------------------------
    client = load_protrek_650m_local(
        device=device,
        model_root=model_root,
        protrek_code_root=protrek_code_root,
    )

    # --------------------------
    # Batch dataset processing
    # --------------------------
    for fasta_file, output_file in tasks:

        print("\n" + "=" * 80)
        print(f"[{time.ctime()}] Current processing file:")
        print(f"FASTA : {fasta_file}")
        print(f"Output CSV: {output_file}")
        print("=" * 80)

        names, embeddings = extract_features_esmc(
            fasta_file=fasta_file,
            client=client,
            device=device,
            batch_size=256,
            max_length=1024,
        )

        save_features_as_csv(
            names=names,
            embeddings=embeddings,
            output_file=output_file,
        )

    print(f"\n[{time.ctime()}] All datasets processed completely")


if __name__ == "__main__":
    main()