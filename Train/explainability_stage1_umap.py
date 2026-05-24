# -*- coding: utf-8 -*-
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from explainability_pipeline_utils import (
    BEST_CONFIG,
    FIXED_THRESHOLD,
    evaluate_predictions,
    extract_intermediate_outputs,
    prepare_test_data,
    set_seed,
    train_or_load_gated_tabm,
)

COLOR_NON_AMP = "#6EAFDD"
COLOR_AMP = "#2C2C64"


def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_umap(X: np.ndarray, random_state: int, n_neighbors: int = 15, min_dist: float = 0.1):
    try:
        import umap.umap_ as umap
    except Exception as e:
        raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn") from e
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=random_state)
    return reducer.fit_transform(X)


def draw_umap_subplot(ax, csv_path: str, title: str, sizes: dict):
    df = pd.read_csv(csv_path)
    required_cols = {"umap1", "umap2", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    labels = df["label"].astype(int).values
    idx0 = labels == 0
    idx1 = labels == 1
    ax.scatter(df.loc[idx0, "umap1"], df.loc[idx0, "umap2"], s=18, alpha=0.78, c=COLOR_NON_AMP, label="non-AMP", edgecolors="none")
    ax.scatter(df.loc[idx1, "umap1"], df.loc[idx1, "umap2"], s=18, alpha=0.78, c=COLOR_AMP, label="AMP", edgecolors="none")
    ax.set_xlabel("UMAP-1", fontsize=sizes["label"])
    ax.set_ylabel("UMAP-2", fontsize=sizes["label"])
    ax.set_title(title, fontsize=sizes["title"], fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=sizes["tick"])
    ax.legend(frameon=False, fontsize=sizes["tick"])


def run_stage1(args):
    set_seed(args.seed)
    make_output_dir(args.out_dir)
    model, scaler, feat_cols, config = train_or_load_gated_tabm(
        train_csv=args.train_merged,
        label_col=args.label_col,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        config=BEST_CONFIG,
        force_retrain=args.force_retrain,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    df_test, X_scaled, y = prepare_test_data(args.pos_test, args.neg_test, feat_cols, scaler, label_col=args.label_col)
    prob, gated_emb, _ = extract_intermediate_outputs(model=model, X=X_scaled, device=device, batch_size=4096)
    pred = (prob >= FIXED_THRESHOLD).astype(int)
    metrics = evaluate_predictions(y, prob)
    pred_df = df_test.copy()
    pred_df["prob"] = prob
    pred_df["pred"] = pred
    pred_df.to_csv(os.path.join(args.out_dir, "stage1_test_predictions.csv"), index=False, encoding="utf-8-sig")
    raw_coords = compute_umap(X_scaled, random_state=args.seed, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist)
    gated_coords = compute_umap(gated_emb, random_state=args.seed, n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist)
    protein_names = df_test["protein_name"] if "protein_name" in df_test.columns else pd.Series([f"sample_{i}" for i in range(len(df_test))])
    pd.DataFrame({"protein_name": protein_names, "label": y, "prob": prob, "pred": pred, "umap1": raw_coords[:, 0], "umap2": raw_coords[:, 1]}).to_csv(os.path.join(args.out_dir, "umap_raw_embedding.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"protein_name": protein_names, "label": y, "prob": prob, "pred": pred, "umap1": gated_coords[:, 0], "umap2": gated_coords[:, 1]}).to_csv(os.path.join(args.out_dir, "umap_gated_embedding.csv"), index=False, encoding="utf-8-sig")
    font_sizes = {"title": args.title_size, "label": args.label_size, "tick": args.tick_size}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    draw_umap_subplot(ax1, os.path.join(args.out_dir, "umap_raw_embedding.csv"), "UMAP of raw input embeddings", font_sizes)
    draw_umap_subplot(ax2, os.path.join(args.out_dir, "umap_gated_embedding.csv"), "UMAP of gated embeddings", font_sizes)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "combined_umap_plots.png"), dpi=300, bbox_inches="tight")
    plt.close()
    summary = {"config": config, "test_metrics": metrics, "n_test_samples": int(len(df_test)), "outputs": ["stage1_test_predictions.csv", "umap_raw_embedding.csv", "umap_gated_embedding.csv", "combined_umap_plots.png"]}
    with open(os.path.join(args.out_dir, "stage1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Stage-1 outputs saved to:", args.out_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Clean stage-1 explainability: UMAP only")
    parser.add_argument("--train_merged", type=str, default="train_protrek650m_gated_tabm.csv")
    parser.add_argument("--pos_test", type=str, default="positive_protrek650m_test.csv")
    parser.add_argument("--neg_test", type=str, default="negative_protrek650m_test.csv")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="explainability_outputs/gated_tabm_model.pt")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--out_dir", type=str, default="explainability_outputs/stage1_umap")
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--title_size", type=int, default=16)
    parser.add_argument("--label_size", type=int, default=14)
    parser.add_argument("--tick_size", type=int, default=12)
    return parser


if __name__ == "__main__":
    run_stage1(build_parser().parse_args())