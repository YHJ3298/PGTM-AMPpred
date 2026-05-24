# -*- coding: utf-8 -*-
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from tabm import TabM


FIXED_THRESHOLD = 0.5

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
    "adapter_dim": 0,
    "adapter_hidden_dim": 0,
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
    raise ValueError(f"Unsupported activation function: {name}")  # 不支持的激活函数


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

    def forward_with_gate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(x)
        value = self.value(x)
        out = value * gate
        return out, gate


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
        adapted = self.adapter(x)
        return self.backbone(adapted)

    def forward_with_intermediate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adapted, gate = self.adapter.forward_with_gate(x)
        logits = self.backbone(adapted)
        return logits, adapted, gate


def load_and_label(csv_path: str, label_value: int, label_col: str = "label") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")  # 找不到文件
    df = pd.read_csv(csv_path)
    df[label_col] = int(label_value)
    return df


def get_numeric_feature_df(
    df: pd.DataFrame,
    label_col: str = "label",
    drop_non_feature_cols=("protein_name",),
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Label column does not exist in data: {label_col}")  # 数据中不存在标签列
    drop_cols = [c for c in drop_non_feature_cols if c in df.columns] + [label_col]
    feature_df = df.drop(columns=drop_cols, errors="ignore")
    X_df = feature_df.select_dtypes(include=[np.number]).copy()
    if X_df.shape[1] == 0:
        raise ValueError("No numeric feature columns found, please check CSV column format.")  # 未找到任何数值特征列，请检查 CSV 列格式
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
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = 0.0 if denom == 0 else (tp * tn - fp * fn) / math.sqrt(denom)
    return {"SN": sn, "ACC": acc, "SP": sp, "MCC": mcc, "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}


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
    if metric_name in {"SN", "ACC", "SP", "MCC"}:
        return float(compute_metrics(y_true, y_pred)[metric_name])
    raise ValueError(f"Unsupported metric: {metric_name}")  # 不支持的指标


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    probs_all = []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits).mean(dim=1).squeeze(-1)
        probs_all.append(probs.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(probs_all, axis=0)


@torch.no_grad()
def extract_intermediate_outputs(
    model: GatedTabM,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    probs_all, adapted_all, gate_all = [], [], []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        logits, adapted, gate = model.forward_with_intermediate(xb)
        probs = torch.sigmoid(logits).mean(dim=1).squeeze(-1)
        probs_all.append(probs.detach().cpu().numpy().astype(np.float64))
        adapted_all.append(adapted.detach().cpu().numpy().astype(np.float32))
        gate_all.append(gate.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(probs_all, axis=0), np.concatenate(adapted_all, axis=0), np.concatenate(gate_all, axis=0)


def train_full_with_inner_val(model: GatedTabM, X_train: np.ndarray, y_train: np.ndarray, device: torch.device, config: Dict, seed: int) -> GatedTabM:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config["val_ratio"], random_state=seed)
    tr_i, va_i = next(sss.split(X_train, y_train))
    X_tr, y_tr = X_train[tr_i], y_train[tr_i]
    X_va, y_va = X_train[va_i], y_train[va_i]
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=False)
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
            logits = model(xb)
            y_expand = yb.expand(-1, logits.shape[1], -1)
            loss = bce(logits, y_expand).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_prob = predict_proba(model, X_va, device=device, batch_size=4096)
        val_pred = (val_prob >= FIXED_THRESHOLD).astype(int)
        val_score = score_by_metric(config["early_stop_metric"], y_va, val_prob, val_pred)
        print(f"  Epoch {ep:03d}/{config['epochs']} | train_loss={np.mean(losses):.6f} | val_{config['early_stop_metric']}={val_score:.6f}")
        improved = (not np.isnan(val_score)) and (val_score > best_score + 1e-6)
        if improved:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config["patience"]:
                print(f"  Early stop. Best val_{config['early_stop_metric']}={best_score:.6f}")  # 提前停止
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_or_load_gated_tabm(train_csv: str, label_col: str, checkpoint_path: Optional[str], seed: int = 42, config: Optional[Dict] = None, force_retrain: bool = False):
    config = BEST_CONFIG.copy() if config is None else dict(config)
    set_seed(seed)
    if checkpoint_path and os.path.exists(checkpoint_path) and not force_retrain:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        feat_cols = payload["feature_cols"]
        scaler = StandardScaler()
        scaler.mean_ = np.array(payload["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.array(payload["scaler_scale"], dtype=np.float64)
        scaler.var_ = np.square(scaler.scale_)
        scaler.n_features_in_ = len(feat_cols)
        scaler.n_samples_seen_ = int(payload.get("n_samples_seen", 0))
        model = GatedTabM(in_dim=len(feat_cols), config=payload["config"], seed=seed)
        model.load_state_dict(payload["model_state"])
        return model, scaler, feat_cols, payload["config"]
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training set file not found: {train_csv}")  # 找不到训练集文件
    df_train = pd.read_csv(train_csv)
    if label_col not in df_train.columns:
        raise ValueError(f"Training set file missing label column {label_col}: {train_csv}")  # 训练集文件缺少标签列
    X_train_raw, y_train, feat_cols = build_X_y_with_alignment(df_train, label_col=label_col, drop_non_feature_cols=("protein_name",), feature_cols=None)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedTabM(in_dim=X_train.shape[1], config=config, seed=seed)
    model = train_full_with_inner_val(model=model, X_train=X_train, y_train=y_train, device=device, config=config, seed=seed)
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        payload = {
            "config": config,
            "feature_cols": feat_cols,
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "n_samples_seen": int(getattr(scaler, "n_samples_seen_", 0)),
        }
        torch.save(payload, checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")  # 模型 checkpoint 已保存
    return model, scaler, feat_cols, config


def prepare_test_data(pos_test_csv: str, neg_test_csv: str, feat_cols: Sequence[str], scaler: StandardScaler, label_col: str = "label") -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df_pos = load_and_label(pos_test_csv, 1, label_col=label_col)
    df_neg = load_and_label(neg_test_csv, 0, label_col=label_col)
    df_test = pd.concat([df_pos, df_neg], axis=0, ignore_index=True)
    X_raw, y, _ = build_X_y_with_alignment(df_test, label_col=label_col, drop_non_feature_cols=("protein_name",), feature_cols=list(feat_cols))
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    return df_test, X_scaled, y


def evaluate_predictions(y_true: np.ndarray, prob: np.ndarray, threshold: float = FIXED_THRESHOLD) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    metrics = compute_metrics(y_true, pred)
    metrics["ROC_AUC"] = safe_roc_auc(y_true, prob)
    metrics["PR_AUC"] = safe_pr_auc(y_true, prob)
    return metrics


def parse_fasta(file_path: str) -> List[Tuple[str, str]]:
    file_path = file_path.strip()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file does not exist or path is incorrect: {file_path}")  # FASTA 文件不存在或路径错误
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
    return data


class SequenceDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str]]):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx: int):
        return self.items[idx]


_VALID_AA = set(list("ACDEFGHIKLMNPQRSTVWY") + ["X"])
_REPLACE_TO_X = set(list("BJOUZ"))


def normalize_aa(seq: str, max_len: int) -> str:
    seq = seq.strip().replace(" ", "").upper()
    if not seq:
        return ""
    if len(seq) > max_len:
        seq = seq[:max_len]
    seq = "".join(("X" if ch in _REPLACE_TO_X else ch) for ch in seq)
    seq = "".join((ch if ch in _VALID_AA else "X") for ch in seq)
    return seq


def build_fasta_df(pos_fasta: str, neg_fasta: str) -> pd.DataFrame:
    pos_items = parse_fasta(pos_fasta)
    neg_items = parse_fasta(neg_fasta)
    df_pos = pd.DataFrame(pos_items, columns=["protein_name", "sequence"])
    df_pos["label"] = 1
    df_neg = pd.DataFrame(neg_items, columns=["protein_name", "sequence"])
    df_neg["label"] = 0
    return pd.concat([df_pos, df_neg], axis=0, ignore_index=True)


def _ensure_protrek650m_layout(model_root: str) -> dict:
    model_root = Path(model_root)
    if not model_root.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_root}")  # 模型目录不存在
    ckpt = model_root / "ProTrek_650M.pt"
    if not ckpt.exists():
        pts = list(model_root.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt checkpoint found under {model_root}")  # 在指定目录下没有找到任何 .pt checkpoint
        ckpt = pts[0]
    protein_dir = model_root / "esm2_t33_650M_UR50D"
    text_dir = model_root / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    structure_dir = model_root / "foldseek_t30_150M"
    for p in [protein_dir, text_dir, structure_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Required subdirectory missing: {p}")  # 缺少必要子目录
    return {"ckpt": ckpt, "protein_dir": protein_dir, "text_dir": text_dir, "structure_dir": structure_dir}


def _patch_faiss_if_missing():
    try:
        import faiss  # noqa: F401
        return
    except ModuleNotFoundError:
        import importlib.machinery
        import types
        class _FaissStub(types.ModuleType):
            def __getattr__(self, item):
                if item.startswith("__"):
                    raise AttributeError(item)
                raise ModuleNotFoundError(
                    "faiss is not installed in the current environment.\nYou can ignore this if you only perform ProTrek feature extraction/inference;\nPlease install faiss-cpu if you need to use index/retrieval related functions later."
                )  # 当前环境未安装 faiss。如果只是做 ProTrek 特征提取/推理，可以忽略；如果后续要做索引/检索相关功能，请安装 faiss-cpu
        stub = _FaissStub("faiss")
        stub.__spec__ = importlib.machinery.ModuleSpec(name="faiss", loader=None)
        stub.__file__ = "<faiss_stub>"
        stub.__path__ = []
        sys.modules["faiss"] = stub


def _install_torchmetrics_stub():
    import importlib.machinery
    import types
    tm = types.ModuleType("torchmetrics")
    tm.__spec__ = importlib.machinery.ModuleSpec(name="torchmetrics", loader=None)
    tm.__file__ = "<torchmetrics_stub>"
    metric_mod = types.ModuleType("torchmetrics.metric")
    metric_mod.__spec__ = importlib.machinery.ModuleSpec(name="torchmetrics.metric", loader=None)
    metric_mod.__file__ = "<torchmetrics_metric_stub>"
    functional_mod = types.ModuleType("torchmetrics.functional")
    functional_mod.__spec__ = importlib.machinery.ModuleSpec(name="torchmetrics.functional", loader=None)
    functional_mod.__file__ = "<torchmetrics_functional_stub>"
    class Metric(torch.nn.Module):
        full_state_update = False
        def __init__(self, *args, **kwargs):
            super().__init__()
        def update(self, *args, **kwargs):
            pass
        def compute(self):
            return torch.tensor(0.0)
        def reset(self):
            pass
    class DummyMetric(Metric):
        pass
    tm.Metric = Metric
    metric_mod.Metric = Metric
    for name in ["Accuracy", "F1Score", "AUROC", "AveragePrecision", "Precision", "Recall", "MatthewsCorrCoef", "MeanSquaredError", "MeanAbsoluteError"]:
        setattr(tm, name, DummyMetric)
    tm.functional = functional_mod
    sys.modules["torchmetrics"] = tm
    sys.modules["torchmetrics.metric"] = metric_mod
    sys.modules["torchmetrics.functional"] = functional_mod


def _patch_torchmetrics_for_protrek_inference():
    try:
        import torchmetrics
        from torchmetrics import Metric
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
        for name in ["Accuracy", "F1Score", "AUROC", "AveragePrecision", "Precision", "Recall", "MatthewsCorrCoef", "MeanSquaredError", "MeanAbsoluteError"]:
            if hasattr(torchmetrics, name):
                setattr(torchmetrics, name, DummyMetric)
        return
    except Exception as e:
        _install_torchmetrics_stub()
        print(f"[WARN] Failed to import torchmetrics, inference-only stub enabled: {e}")  # torchmetrics 导入失败，已启用推理专用 stub


def load_protrek_650m_local(device: torch.device, model_root: str, protrek_code_root: str):
    paths = _ensure_protrek650m_layout(model_root)
    protrek_code_root = str(Path(protrek_code_root).resolve())
    if not Path(protrek_code_root).exists():
        raise RuntimeError(f"ProTrek code repository path not found: {protrek_code_root}")  # 未找到 ProTrek 代码仓库路径
    if protrek_code_root not in sys.path:
        sys.path.insert(0, protrek_code_root)
    _patch_faiss_if_missing()
    _patch_torchmetrics_for_protrek_inference()
    from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
    config = {
        "protein_config": str(paths["protein_dir"]),
        "text_config": str(paths["text_dir"]),
        "structure_config": str(paths["structure_dir"]),
        "from_checkpoint": str(paths["ckpt"]),
    }
    print(f"[{time.ctime()}] Starting to load ProTrek_650M")  # 开始加载 ProTrek_650M
    model = ProTrekTrimodalModel(**config).to(device)
    model.eval()
    print(f"[{time.ctime()}] ProTrek_650M loaded successfully")  # ProTrek_650M 加载完成
    return model


@torch.no_grad()
def extract_protrek_embeddings_from_sequences(items: List[Tuple[str, str]], client, batch_size: int = 256, max_length: int = 1024) -> Tuple[List[str], np.ndarray]:
    dataset = SequenceDataset(items)
    def collate_fn(batch):
        names = [x[0] for x in batch]
        seqs = [normalize_aa(x[1], max_length) for x in batch]
        keep = [(n, s) for n, s in zip(names, seqs) if s]
        if not keep:
            return [], []
        names, seqs = zip(*keep)
        return list(names), list(seqs)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    all_names: List[str] = []
    all_embs: List[np.ndarray] = []
    for names, seqs in dl:
        if not names:
            continue
        embs = client.get_protein_repr(seqs)
        embs = embs.float().detach().cpu().numpy().astype(np.float32)
        all_names.extend(names)
        all_embs.append(embs)
    if all_embs:
        mat = np.concatenate(all_embs, axis=0)
    else:
        mat = np.empty((0, 0), dtype=np.float32)
    return all_names, mat


KD_SCALE = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3, "X": 0.0}
POS_AA = set("KRH")
HYDRO_AA = set("AILMFWVYC")


def net_charge(seq: str) -> float:
    seq = seq.upper()
    charge = 0.0
    for aa in seq:
        if aa in {"K", "R"}:
            charge += 1.0
        elif aa == "H":
            charge += 0.1
        elif aa in {"D", "E"}:
            charge -= 1.0
    return charge


def mean_hydrophobicity(seq: str) -> float:
    vals = [KD_SCALE.get(aa.upper(), 0.0) for aa in seq if aa.strip()]
    return float(np.mean(vals)) if vals else 0.0


def hydrophobic_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(aa in HYDRO_AA for aa in seq.upper()) / len(seq)


def positive_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(aa in POS_AA for aa in seq.upper()) / len(seq)


def hydrophobic_moment_alpha(seq: str, angle_deg: float = 100.0) -> float:
    if not seq:
        return 0.0
    angle = math.radians(angle_deg)
    x, y = 0.0, 0.0
    vals = [KD_SCALE.get(aa.upper(), 0.0) for aa in seq]
    for i, v in enumerate(vals):
        theta = i * angle
        x += v * math.cos(theta)
        y += v * math.sin(theta)
    return math.sqrt(x * x + y * y) / len(seq)


def summarize_fragment_properties(seq: str) -> Dict[str, float]:
    return {
        "length": len(seq),
        "net_charge": net_charge(seq),
        "mean_hydrophobicity": mean_hydrophobicity(seq),
        "hydrophobic_fraction": hydrophobic_fraction(seq),
        "positive_fraction": positive_fraction(seq),
        "hydrophobic_moment": hydrophobic_moment_alpha(seq),
    }


def sequence_logo_matrix(seqs: List[str]) -> pd.DataFrame:
    if not seqs:
        return pd.DataFrame()
    length = len(seqs[0])
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    mat = np.zeros((length, len(aa_order)), dtype=np.float64)
    for i in range(length):
        chars = [s[i] for s in seqs if len(s) == length]
        total = len(chars)
        if total == 0:
            continue
        for j, aa in enumerate(aa_order):
            mat[i, j] = sum(ch == aa for ch in chars) / total
    return pd.DataFrame(mat, columns=aa_order)


def mask_sequence_with_X(seq: str, start: int, end: int) -> str:
    return seq[:start] + ("X" * (end - start)) + seq[end:]


def sliding_windows(seq_len: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    if seq_len <= 0:
        return []
    if seq_len <= window_size:
        return [(0, seq_len)]
    windows = []
    for start in range(0, seq_len - window_size + 1, stride):
        windows.append((start, start + window_size))
    if windows and windows[-1][1] < seq_len:
        windows.append((seq_len - window_size, seq_len))
    elif not windows:
        windows.append((0, seq_len))
    dedup = []
    seen = set()
    for s, e in windows:
        if (s, e) not in seen:
            dedup.append((s, e))
            seen.add((s, e))
    return dedup