# -*- coding: utf-8 -*-
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tabm import TabM


FIXED_THRESHOLD = 0.5


def build_activation(name: str) -> nn.Module:
    name = name.lower()

    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()

    raise ValueError(f"Unsupported activation function: {name}")


def resolve_gated_dims(
    in_dim: int,
    adapter_dim: int,
    adapter_hidden_dim: int,
) -> Tuple[int, int]:
    out_dim = int(adapter_dim) if int(adapter_dim) > 0 else int(in_dim)
    hidden_dim = int(adapter_hidden_dim) if int(adapter_hidden_dim) > 0 else max(32, int(in_dim) // 2)
    return out_dim, hidden_dim


class GatedAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()

        self.value = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) * self.gate(x)


class GatedTabM(nn.Module):
    def __init__(self, in_dim: int, config: Dict, seed: int = 42):
        super().__init__()

        torch.manual_seed(seed)

        out_dim, hidden_dim = resolve_gated_dims(
            in_dim=in_dim,
            adapter_dim=config.get("adapter_dim", 0),
            adapter_hidden_dim=config.get("adapter_hidden_dim", 0),
        )

        self.adapter = GatedAdapter(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            dropout=config.get("adapter_dropout", 0.1),
            activation=config.get("adapter_activation", "gelu"),
        )

        num_embeddings = None

        if config.get("use_num_embeddings", False):
            from rtdl_num_embeddings import LinearReLUEmbeddings
            num_embeddings = LinearReLUEmbeddings(out_dim)

        self.backbone = TabM.make(
            n_num_features=out_dim,
            cat_cardinalities=None,
            d_out=1,
            arch_type=config.get("arch_type", "tabm"),
            k=config.get("k", 32),
            n_blocks=config.get("n_blocks", 2),
            d_block=config.get("d_block", 256),
            dropout=config.get("dropout", 0.1),
            num_embeddings=num_embeddings,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return self.backbone(x)


def load_checkpoint_payload(checkpoint_path: str):
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    required_keys = [
        "config",
        "feature_cols",
        "model_state",
        "scaler_mean",
        "scaler_scale",
    ]

    for key in required_keys:
        if key not in payload:
            raise KeyError(f"Missing required field in checkpoint: {key}")

    return payload


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    seed: int = 42,
):
    payload = load_checkpoint_payload(checkpoint_path)

    config = payload["config"]
    feature_cols = payload["feature_cols"]
    model_state = payload["model_state"]

    in_dim = len(feature_cols)

    model = GatedTabM(
        in_dim=in_dim,
        config=config,
        seed=seed,
    )

    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()

    scaler_mean = np.array(payload["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(payload["scaler_scale"], dtype=np.float32)
    scaler_scale[scaler_scale == 0] = 1.0

    return {
        "model": model,
        "config": config,
        "feature_cols": feature_cols,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
    }


def embeddings_to_feature_df(
    embeddings: torch.Tensor,
    feature_cols: List[str],
) -> pd.DataFrame:
    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)

    generated_cols = [
        f"feature_{i}"
        for i in range(emb_np.shape[1])
    ]

    df = pd.DataFrame(
        emb_np,
        columns=generated_cols,
    )

    df = df.reindex(
        columns=feature_cols,
        fill_value=0.0,
    )

    df = df.fillna(0.0)

    return df


def standardize_features(
    feature_df: pd.DataFrame,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> np.ndarray:
    X = feature_df.values.astype(np.float32)

    if X.shape[1] != len(scaler_mean):
        raise ValueError(
            f"Feature dimension mismatch: current input {X.shape[1]}, "
            f"scaler dimension {len(scaler_mean)}"
        )

    X = (X - scaler_mean) / scaler_scale
    return X.astype(np.float32)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()

    ds = TensorDataset(
        torch.from_numpy(X).float()
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    probs_all = []

    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)

        logits = model(xb)
        probs = torch.sigmoid(logits).mean(dim=1).squeeze(-1)

        probs_all.append(
            probs.detach().cpu().numpy().astype(np.float64)
        )

    return np.concatenate(probs_all, axis=0)


def predict_label_from_prob(
    prob: np.ndarray,
    threshold: float = FIXED_THRESHOLD,
) -> np.ndarray:
    return (prob >= threshold).astype(int)


def label_to_text(pred: int) -> str:
    return "AMP" if int(pred) == 1 else "non-AMP"