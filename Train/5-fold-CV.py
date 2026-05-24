# -*- coding: utf-8 -*-
import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
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
    """
    gated adapter:
        value(x) = MLP(x)
        gate(x)  = sigmoid(Wx)
        output   = value(x) * gate(x)
    """
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


def build_X_y(
    df: pd.DataFrame,
    label_col: str = "label",
    drop_non_feature_cols=("protein_name",),
):
    if label_col not in df.columns:
        raise ValueError(f"Label column not found in dataset: {label_col}")

    y = df[label_col].astype(int).values
    drop_cols = [c for c in drop_non_feature_cols if c in df.columns] + [label_col]
    feature_df = df.drop(columns=drop_cols, errors="ignore")

    X_df = feature_df.select_dtypes(include=[np.number]).copy()
    if X_df.shape[1] == 0:
        raise ValueError("No numerical feature columns found. Please check the CSV format.")

    X_df = X_df.fillna(0.0)
    X = X_df.values.astype(np.float32)
    return X, y, list(X_df.columns)


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
        probs = torch.sigmoid(logits).mean(dim=1).squeeze(-1)  # Average ensemble probability
        probs_all.append(probs.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(probs_all, axis=0)


def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    config: Dict,
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
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

        val_prob = predict_proba(model, X_val, device=device, batch_size=4096)
        val_pred = (val_prob >= FIXED_THRESHOLD).astype(int)
        val_score = score_by_metric(config["early_stop_metric"], y_val, val_prob, val_pred)

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
                print(f"  Early stop triggered. Best val_{config['early_stop_metric']}={best_score:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def summarize_fold_results(fold_results: List[Dict[str, float]]) -> Dict[str, float]:
    summary = {}
    for key in ["SN", "ACC", "SP", "F1", "MCC", "ROC_AUC", "PR_AUC"]:
        arr = np.array([r.get(key, float("nan")) for r in fold_results], dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
        elif arr.size == 1:
            summary[f"{key}_mean"] = float(arr.mean())
            summary[f"{key}_std"] = 0.0
        else:
            summary[f"{key}_mean"] = float(arr.mean())
            summary[f"{key}_std"] = float(arr.std(ddof=1))
    return summary


def plot_cv_curves(
    roc_items: List[Dict],
    pr_items: List[Dict],
    y_all: np.ndarray,
    out_dir: str = ".",
    prefix: str = "gated_tabm",
    show: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""

    if len(roc_items) > 0:
        mean_fpr = np.linspace(0.0, 1.0, 200)
        tprs = []
        aucs = []

        plt.figure()
        for it in roc_items:
            fpr, tpr, aucv = it["fpr"], it["tpr"], it["auc"]
            tpr_i = np.interp(mean_fpr, fpr, tpr)
            tpr_i[0] = 0.0
            tprs.append(tpr_i)
            aucs.append(aucv)
            plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {it['fold']} (AUC={aucv:.3f})")

        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0, ddof=1) if tprs.shape[0] > 1 else np.zeros_like(mean_tpr)
        mean_tpr[-1] = 1.0

        plt.plot(
            mean_fpr,
            mean_tpr,
            linewidth=2,
            label=f"Mean ROC (AUC={np.mean(aucs):.3f} ± {np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0:.3f})",
        )
        plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), alpha=0.2)
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("5-Fold ROC Curve (Gated-TabM)")
        plt.legend(loc="lower right", fontsize=8)
        roc_path = os.path.join(out_dir, f"{prefix}roc_curve_5fold.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        if show:
            plt.show()
        plt.close()
        print(f"ROC curve saved to: {roc_path}")

    if len(pr_items) > 0:
        mean_recall = np.linspace(0.0, 1.0, 200)
        precisions = []
        aps = []

        plt.figure()
        for it in pr_items:
            recall, precision, apv = it["recall"], it["precision"], it["auc"]
            uniq_recall, uniq_idx = np.unique(recall, return_index=True)
            uniq_precision = precision[uniq_idx]
            p_i = np.interp(mean_recall, uniq_recall, uniq_precision)
            precisions.append(p_i)
            aps.append(apv)
            plt.plot(recall, precision, alpha=0.3, label=f"Fold {it['fold']} (AP={apv:.3f})")

        precisions = np.array(precisions)
        mean_p = precisions.mean(axis=0)
        std_p = precisions.std(axis=0, ddof=1) if precisions.shape[0] > 1 else np.zeros_like(mean_p)

        pos_rate = float(np.mean(y_all))
        plt.hlines(pos_rate, 0, 1, linestyles="--", linewidth=1, label=f"Baseline (pos_rate={pos_rate:.3f})")
        plt.plot(
            mean_recall,
            mean_p,
            linewidth=2,
            label=f"Mean PR (AP={np.mean(aps):.3f} ± {np.std(aps, ddof=1) if len(aps) > 1 else 0.0:.3f})",
        )
        plt.fill_between(mean_recall, np.maximum(mean_p - std_p, 0), np.minimum(mean_p + std_p, 1), alpha=0.2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("5-Fold Precision-Recall Curve (Gated-TabM)")
        plt.legend(loc="lower left", fontsize=8)
        pr_path = os.path.join(out_dir, f"{prefix}pr_curve_5fold.png")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=300)
        if show:
            plt.show()
        plt.close()
        print(f"PR curve saved to: {pr_path}")


def run_5fold_cv(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    config: Dict,
    make_plots: bool,
    plot_dir: str,
    show_plots: bool,
    oof_pred_path: str,
    fold_result_path: str,
    summary_json_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Fixed classification threshold = {FIXED_THRESHOLD}")
    print("Current fixed configuration:")
    print(json.dumps(config, ensure_ascii=False, indent=2))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    roc_items = []
    pr_items = []

    oof_prob = np.zeros(len(y), dtype=np.float64)
    oof_pred = np.zeros(len(y), dtype=np.int64)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n========== Fold {fold} ==========")
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train_raw, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test = scaler.transform(X_test_raw).astype(np.float32)

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config["val_ratio"],
            random_state=seed + fold,
        )
        tr_i, va_i = next(sss.split(X_train_scaled, y_train_raw))
        X_tr, y_tr = X_train_scaled[tr_i], y_train_raw[tr_i]
        X_va, y_va = X_train_scaled[va_i], y_train_raw[va_i]

        model = GatedTabM(in_dim=X.shape[1], config=config, seed=seed + fold)

        model = train_one_fold(
            model=model,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            device=device,
            config=config,
        )

        prob = predict_proba(model, X_test, device=device, batch_size=4096)
        y_pred = (prob >= FIXED_THRESHOLD).astype(int)

        oof_prob[test_idx] = prob
        oof_pred[test_idx] = y_pred

        metrics = compute_metrics(y_test, y_pred)
        metrics["ROC_AUC"] = safe_roc_auc(y_test, prob)
        metrics["PR_AUC"] = safe_pr_auc(y_test, prob)
        metrics["fold"] = fold
        fold_results.append(metrics)

        if len(np.unique(y_test)) >= 2:
            fpr, tpr, _ = roc_curve(y_test, prob)
            roc_items.append({"fold": fold, "fpr": fpr, "tpr": tpr, "auc": metrics["ROC_AUC"]})
            precision, recall, _ = precision_recall_curve(y_test, prob)
            pr_items.append({"fold": fold, "precision": precision, "recall": recall, "auc": metrics["PR_AUC"]})

        print(
            f"[Fold {fold}] SN={metrics['SN']:.4f}, ACC={metrics['ACC']:.4f}, "
            f"SP={metrics['SP']:.4f}, F1={metrics['F1']:.4f}, MCC={metrics['MCC']:.4f}, "
            f"ROC_AUC={metrics['ROC_AUC']:.4f}, PR_AUC={metrics['PR_AUC']:.4f} | "
            f"TP={metrics['TP']}, TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}"
        )

    summary = summarize_fold_results(fold_results)

    print("\n===== 5-Fold CV Summary (mean ± std) =====")
    for key in ["SN", "ACC", "SP", "F1", "MCC", "ROC_AUC", "PR_AUC"]:
        print(f"{key}: {summary[f'{key}_mean']:.4f} ± {summary[f'{key}_std']:.4f}")

    os.makedirs(os.path.dirname(fold_result_path) or ".", exist_ok=True)
    pd.DataFrame(fold_results).to_csv(fold_result_path, index=False, encoding="utf-8-sig")
    print(f"Fold metrics saved to: {fold_result_path}")

    oof_df = pd.DataFrame({
        "label": y,
        "prob": oof_prob,
        "pred": oof_pred,
    })
    oof_df.to_csv(oof_pred_path, index=False, encoding="utf-8-sig")
    print(f"OOF predictions saved to: {oof_pred_path}")

    payload = {
        "config": config,
        "summary": summary,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Summary results saved to: {summary_json_path}")

    if make_plots:
        plot_cv_curves(
            roc_items=roc_items,
            pr_items=pr_items,
            y_all=y,
            out_dir=plot_dir,
            prefix="gated_tabm",
            show=show_plots,
        )

    return summary, fold_results


def main():
    parser = argparse.ArgumentParser(description="5-fold CV for fixed best Gated-TabM on ProTrek650M features")

    parser.add_argument("--pos", type=str, default="positive_protrek650m_train.csv")
    parser.add_argument("--neg", type=str, default="negative_protrek650m_train.csv")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--merged_out", type=str, default="train_protrek650m_gated_tabm.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--threshold", type=float, default=FIXED_THRESHOLD)
    parser.add_argument("--plot_dir", type=str, default="gated_tabm_cv_outputs")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--no_plot", action="store_true")

    parser.add_argument("--fold_result_csv", type=str, default="gated_tabm_cv_outputs/gated_tabm_5fold_metrics.csv")
    parser.add_argument("--oof_pred_csv", type=str, default="gated_tabm_cv_outputs/gated_tabm_oof_predictions.csv")
    parser.add_argument("--summary_json", type=str, default="gated_tabm_cv_outputs/gated_tabm_5fold_summary.json")

    args = parser.parse_args()
    set_seed(args.seed)

    if abs(float(args.threshold) - FIXED_THRESHOLD) > 1e-12:
        raise ValueError(f"This script fixes threshold={FIXED_THRESHOLD}，modification is not allowed {args.threshold}")

    df_pos = load_and_label(args.pos, 1, label_col=args.label_col)
    df_neg = load_and_label(args.neg, 0, label_col=args.label_col)
    df_all = pd.concat([df_pos, df_neg], axis=0, ignore_index=True)

    df_all.to_csv(args.merged_out, index=False, encoding="utf-8-sig")
    print(f"Merged training dataset saved to: {args.merged_out}")
    print(f"Positive samples: {len(df_pos)}")
    print(f"Negative samples: {len(df_neg)}")
    print(f"Total samples:    {len(df_all)}")

    X, y, feat_cols = build_X_y(df_all, label_col=args.label_col, drop_non_feature_cols=("protein_name",))
    print(f"Feature dim: {X.shape[1]} (Example columns: {feat_cols[:5]}{'...' if len(feat_cols) > 5 else ''})")

    run_5fold_cv(
        X=X,
        y=y,
        seed=args.seed,
        config=BEST_CONFIG,
        make_plots=(not args.no_plot),
        plot_dir=args.plot_dir,
        show_plots=args.show_plots,
        oof_pred_path=args.oof_pred_csv,
        fold_result_path=args.fold_result_csv,
        summary_json_path=args.summary_json,
    )


if __name__ == "__main__":
    main()
