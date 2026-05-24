# -*- coding: utf-8 -*-
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

# Input and output paths
EXAMPLE_FASTA = PROJECT_ROOT / "examples" / "example.fasta"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Model paths
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "gated_tabm_final_checkpoint.pt"
PROTREK_MODEL_ROOT = PROJECT_ROOT / "models" / "Protrek_650M"
PROTREK_CODE_ROOT = PROJECT_ROOT / "ProTrek-main"

# Inference parameters
FIXED_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 1024


def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")