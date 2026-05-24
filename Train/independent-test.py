# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tabm import TabM  # pip install tabm


FIXED_THRESHOLD = 0.5

# Best gated-TabM configuration currently selected
BEST_CONFIG = {
    "k": 32,
    "arch_type": "tabm",
    "n_blocks": 2,
    "d_block": 256,
    "dropout": 0.1,
    "use_num_embeddings": False,
    "epochs": 200,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 20,
    "val_ratio": 0.1,
    "adapter_dim": 0,            # 0 -> same dimension as input
    "adapter_hidden_dim": 0,     # 0 -> automatically set to in_dim // 2
    "adapter_dropout": 0.1,
    "adapter_activation": "gelu",
    "early_stop_metric": "ROC_AUC",
}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation function: {name}")


def resolve_gated_dims(in_dim: int, adapter_dim: int, adapter_hidden_dim: int) -> Tuple[int, int]:
    out_dim = int(adapter_dim) if int(adapter_dim) > 0 else int(in_dim)
    hidden_dim = int(adapter_hidden_dim) if int(adapter_hidden_dim) > 0 else max(32, int(in_dim) // 2)
    return out_dim, hidden_dim


class GatedAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float, activation: str):
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
    def __init__(self, in_dim: int, config: Dict, seed: int):
        super().__init__()
        torch.manual_seed(seed)

        out_dim, hidden_dim = resolve_gated_dims(
            in_dim=in_dim,
            adapter_dim=config["adapter_dim"],
            adapter_hidden_dim=config["adapter_hidden_dim"],
        )

        self.adapter = GatedAdapter(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            dropout=config["adapter_dropout"],
            activation=config["adapter_activation"],
        )

        num_embeddings = None
        if config["use_num_embeddings"]:
            from rtdl_num_embeddings import LinearReLUEmbeddings
            num_embeddings = LinearReLUEmbeddings(out_dim)

        self.backbone = TabM.make(
            n_num_features=out_dim,
            cat_cardinalities=None,
            d_out=1,
            arch_type=config["arch_type"],
            k=config["k"],
            n_blocks=config["n_blocks"],
            d_block=config["d_block"],
            dropout=config["dropout"],
            num_embeddings=num_embeddings,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return self.backbone(x)  # (B, k, 1)


def load_and_label(csv_path: str, label_value: int, label_col: str = "label") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df[label_col] = int(label_value)
    return df


def get_numeric_feature_df(
    df: pd.DataFrame,
    label_col: str = "label",
    drop_non_feature_cols=("protein_name",),
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Label column not found in dataset: {label_col}")

    drop_cols = [c for c in drop_non_feature_cols if c in df.columns] + [label_col]
    feature_df = df.drop(columns=drop_cols, errors="ignore")
    X_df = feature_df.select_dtypes(include=[np.number]).copy()

    if X_df.shape[1] == 0:
        raise ValueError("No numerical feature columns found. Please check the CSV format.")
    return X_df


def build_X_y_with_alignment(
    df: pd.DataFrame,
    label_col: str = "label",
    drop_non_feature_cols=("protein_name",),
    feature_cols: Optional[List[str]] = None,
):
    y = df[label_col].astype(int).values
    X_df = get_numeric_feature_df(df, label_col=label_col, drop_non_feature_cols=drop_non_feature_cols)

    if feature_cols is None:
        feature_cols = list(X_df.columns)
    else:
        X_df = X_df.reindex(columns=feature_cols, fill_value=0.0)

    X_df = X_df.fillna(0.0)
    X = X_df.values.astype(np.float32)
    return X, y, feature_cols


def safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom != 0 else 0.0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sn = safe_div(tp, tp + fn)
    sp = safe_div(tn, tn + fp)
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        mcc = 0.0
    else:
        mcc = (tp * tn - fp * fn) / math.sqrt(denom)

    return {
        "SN": sn,
        "ACC": acc,
        "SP": sp,
        "F1": f1,
        "MCC": mcc,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def score_by_metric(metric_name: str, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> float:
    if metric_name == "ROC_AUC":
        return safe_roc_auc(y_true, y_prob)
    if metric_name == "PR_AUC":
        return safe_pr_auc(y_true, y_prob)
    if metric_name in {"SN", "ACC", "SP", "F1", "MCC"}:
        return float(compute_metrics(y_true, y_pred)[metric_name])
    raise ValueError(f"Unsupported metric: {metric_name}")


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    probs_all = []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)  # (B, k, 1)
        probs = torch.sigmoid(logits).mean(dim=1).squeeze(-1)
        probs_all.append(probs.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(probs_all, axis=0)


def train_full_with_inner_val(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    config: Dict,
    seed: int,
) -> nn.Module:
    """
    Split an internal validation set from the training set for early stopping, without touching the independent test set.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config["val_ratio"],
        random_state=seed,
    )
    tr_i, va_i = next(sss.split(X_train, y_train))
    X_tr, y_tr = X_train[tr_i], y_train[tr_i]
    X_va, y_va = X_train[va_i], y_train[va_i]

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    best_score = -1.0
    best_state = None
    bad_epochs = 0

    for ep in range(1, config["epochs"] + 1):
        model.train()
        losses = []

        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).view(-1, 1, 1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)  # (B, k, 1)
            y_expand = yb.expand(-1, logits.shape[1], -1)
            loss = bce(logits, y_expand).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_prob = predict_proba(model, X_va, device=device, batch_size=4096)
        val_pred = (val_prob >= FIXED_THRESHOLD).astype(int)
        val_score = score_by_metric(config["early_stop_metric"], y_va, val_prob, val_pred)

        print(
            f"  Epoch {ep:03d}/{config['epochs']} | "
            f"train_loss={np.mean(losses):.6f} | "
            f"val_{config['early_stop_metric']}={val_score:.6f}"
        )

        improved = (not np.isnan(val_score)) and (val_score > best_score + 1e-6)
        if improved:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                print(f"  Early stop. Best val_{config['early_stop_metric']}={best_score:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def save_checkpoint(model, scaler, feat_cols, checkpoint_out: str):
    os.makedirs(os.path.dirname(checkpoint_out) or ".", exist_ok=True)
    payload = {
        "config": BEST_CONFIG,
        "feature_cols": feat_cols,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "n_samples_seen": int(getattr(scaler, "n_samples_seen_", 0)),
    }
    torch.save(payload, checkpoint_out)
    print(f"Model checkpoint saved to: {checkpoint_out}")


def run_independent_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    label_col: str,
    seed: int,
    pred_out: str,
    metrics_json: str,
    checkpoint_out: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Fixed classification threshold = {FIXED_THRESHOLD}")
    print("Current fixed configuration:")
    print(json.dumps(BEST_CONFIG, ensure_ascii=False, indent=2))

    X_train_raw, y_train, feat_cols = build_X_y_with_alignment(
        df_train,
        label_col=label_col,
        drop_non_feature_cols=("protein_name",),
        feature_cols=None,
    )
    X_test_raw, y_test, _ = build_X_y_with_alignment(
        df_test,
        label_col=label_col,
        drop_non_feature_cols=("protein_name",),
        feature_cols=feat_cols,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    model = GatedTabM(in_dim=X_train.shape[1], config=BEST_CONFIG, seed=seed)
    model = train_full_with_inner_val(
        model=model,
        X_train=X_train,
        y_train=y_train,
        device=device,
        config=BEST_CONFIG,
        seed=seed,
    )

    save_checkpoint(model=model, scaler=scaler, feat_cols=feat_cols, checkpoint_out=checkpoint_out)

    prob = predict_proba(model, X_test, device=device, batch_size=4096)
    y_pred = (prob >= FIXED_THRESHOLD).astype(int)

    metrics = compute_metrics(y_test, y_pred)
    metrics["ROC_AUC"] = safe_roc_auc(y_test, prob)
    metrics["PR_AUC"] = safe_pr_auc(y_test, prob)

    out_df = df_test.copy()
    out_df["prob"] = prob
    out_df["pred"] = y_pred
    out_df.to_csv(pred_out, index=False, encoding="utf-8-sig")
    print(f"Prediction results saved to: {pred_out}")

    payload = {
        "config": BEST_CONFIG,
        "metrics": metrics,
        "checkpoint_out": checkpoint_out,
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Test metrics saved to: {metrics_json}")

    print("\n===== Independent Test Metrics (Gated-TabM) =====")
    print(
        f"SN={metrics['SN']:.4f}, ACC={metrics['ACC']:.4f}, "
        f"SP={metrics['SP']:.4f}, F1={metrics['F1']:.4f}, MCC={metrics['MCC']:.4f}, "
        f"ROC_AUC={metrics['ROC_AUC']:.4f}, PR_AUC={metrics['PR_AUC']:.4f} | "
        f"TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}"
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Independent test for fixed best Gated-TabM on ProTrek650M features")

    parser.add_argument("--train_merged", type=str, default="train_protrek650m_gated_tabm.csv")
    parser.add_argument("--pos_test", type=str, default="positive_protrek650m_test.csv")
    parser.add_argument("--neg_test", type=str, default="negative_protrek650m_test.csv")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=FIXED_THRESHOLD)

    parser.add_argument("--merged_test_out", type=str, default="test_protrek650m_gated_tabm.csv")
    parser.add_argument("--pred_out", type=str, default="gated_tabm_independent_predictions.csv")
    parser.add_argument("--metrics_json", type=str, default="gated_tabm_independent_metrics.json")
    parser.add_argument("--checkpoint_out", type=str, default="gated_tabm_final_checkpoint.pt")

    args = parser.parse_args()
    set_seed(args.seed)

    if abs(float(args.threshold) - FIXED_THRESHOLD) > 1e-12:
        raise ValueError(f"Threshold must be fixed to {FIXED_THRESHOLD}，but received {args.threshold}")

    if not os.path.exists(args.train_merged):
        raise FileNotFoundError(f"Training dataset file not found: {args.train_merged}")
    df_train = pd.read_csv(args.train_merged)
    if args.label_col not in df_train.columns:
        raise ValueError(f"Label column not found in training dataset {args.label_col}: {args.train_merged}")
    print(f"Train loaded: {args.train_merged} | rows={len(df_train)}")

    df_pos_te = load_and_label(args.pos_test, 1, label_col=args.label_col)
    df_neg_te = load_and_label(args.neg_test, 0, label_col=args.label_col)
    df_test = pd.concat([df_pos_te, df_neg_te], axis=0, ignore_index=True)
    df_test.to_csv(args.merged_test_out, index=False, encoding="utf-8-sig")
    print(
        f"Test merged saved: {args.merged_test_out} | "
        f"pos={len(df_pos_te)}, neg={len(df_neg_te)}, total={len(df_test)}"
    )

    run_independent_test(
        df_train=df_train,
        df_test=df_test,
        label_col=args.label_col,
        seed=args.seed,
        pred_out=args.pred_out,
        metrics_json=args.metrics_json,
        checkpoint_out=args.checkpoint_out,
    )


if __name__ == "__main__":
    main()
