# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors

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
        raise ImportError("Install umap-learn to run UMAP: pip install umap-learn") from e
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean", random_state=random_state)
    return reducer.fit_transform(X)


def neighbor_preservation(x_high: np.ndarray, x_low: np.ndarray, k: int = 15) -> float:
    high = (NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
            .fit(x_high).kneighbors(x_high, return_distance=False)[:, 1:])
    low = (NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
           .fit(x_low).kneighbors(x_low, return_distance=False)[:, 1:])
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(high, low)]))


def run_seed_stability(raw: np.ndarray, gated: np.ndarray, labels: np.ndarray, out_dir: str) -> None:
    import umap.umap_ as umap
    rows = []
    for name, values in (("Raw ProTrek", raw), ("Gated Adapter", gated)):
        for seed in (7, 21, 42, 101, 2026):
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                metric="euclidean", random_state=seed, transform_seed=seed,
                                n_jobs=1, low_memory=True)
            coords = reducer.fit_transform(values)
            rows.append({"representation": name, "seed": seed,
                         "umap_2d_silhouette": float(silhouette_samples(coords, labels, metric="euclidean").mean()),
                         "local_15NN_preservation": neighbor_preservation(values, coords, k=15)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "umap_seed_stability.csv"),
                              index=False, encoding="utf-8-sig")


def draw_umap_subplot(ax, csv_path: str, title: str, sizes: dict, legend_loc: str = "upper right"):
    df = pd.read_csv(csv_path)
    required_cols = {"umap1", "umap2", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    labels = df["label"].astype(int).values
    idx0 = labels == 0
    idx1 = labels == 1
    ax.scatter(df.loc[idx0, "umap1"], df.loc[idx0, "umap2"], s=18, alpha=0.78, c=COLOR_NON_AMP, label="non-AMP", edgecolors="none")
    ax.scatter(df.loc[idx1, "umap1"], df.loc[idx1, "umap2"], s=18, alpha=0.78, c=COLOR_AMP, label="AMP", edgecolors="none")
    ax.set_xlabel("UMAP-1", fontsize=sizes["label"])
    ax.set_ylabel("UMAP-2", fontsize=sizes["label"])
    ax.set_title(title, fontsize=sizes["title"], fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=sizes["tick"])
    ax.legend(loc=legend_loc, frameon=False, fontsize=sizes["tick"])


def render_saved_umap(raw_csv: str, gated_csv: str, figure_path: str, sizes: dict):
    """Redraw the original UMAP layout without recomputing the coordinates."""
    output = Path(figure_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    draw_umap_subplot(ax1, raw_csv, "UMAP of raw input embeddings", sizes,
                      legend_loc="upper left")
    draw_umap_subplot(ax2, gated_csv, "UMAP of gated embeddings", sizes)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    run_seed_stability(X_scaled, gated_emb, y, args.out_dir)
    font_sizes = {"title": args.title_size, "label": args.label_size, "tick": args.tick_size}
    render_saved_umap(
        os.path.join(args.out_dir, "umap_raw_embedding.csv"),
        os.path.join(args.out_dir, "umap_gated_embedding.csv"),
        os.path.join(args.out_dir, "combined_umap_plots.png"),
        font_sizes,
    )
    summary = {"config": config, "test_metrics": metrics, "n_test_samples": int(len(df_test)), "outputs": ["stage1_test_predictions.csv", "umap_raw_embedding.csv", "umap_gated_embedding.csv", "umap_seed_stability.csv", "combined_umap_plots.png"]}
    with open(os.path.join(args.out_dir, "stage1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Stage 1 outputs saved to:", args.out_dir)
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
    parser.add_argument("--plot_only", action="store_true", help="Render figures from existing UMAP coordinates only")
    parser.add_argument("--raw_csv", type=str)
    parser.add_argument("--gated_csv", type=str)
    parser.add_argument("--figure_path", type=str)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.plot_only:
        if not all((args.raw_csv, args.gated_csv, args.figure_path)):
            parser.error("--plot_only requires --raw_csv, --gated_csv and --figure_path")
        render_saved_umap(args.raw_csv, args.gated_csv, args.figure_path,
                          {"title": args.title_size, "label": args.label_size, "tick": args.tick_size})
    else:
        run_stage1(args)
