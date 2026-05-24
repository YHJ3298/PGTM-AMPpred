# -*- coding: utf-8 -*-
import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu

from explainability_pipeline_utils import (
    BEST_CONFIG,
    FIXED_THRESHOLD,
    build_fasta_df,
    evaluate_predictions,
    extract_protrek_embeddings_from_sequences,
    load_protrek_650m_local,
    mask_sequence_with_X,
    prepare_test_data,
    predict_proba,
    sequence_logo_matrix,
    set_seed,
    sliding_windows,
    summarize_fragment_properties,
    train_or_load_gated_tabm,
)

COLOR_BG = "#6EAFDD"
COLOR_IMP = "#2C2C64"


def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def merge_test_csv_with_fasta(df_test_csv: pd.DataFrame, pos_fasta: str, neg_fasta: str):
    df_fasta = build_fasta_df(pos_fasta, neg_fasta)
    merged = df_test_csv.merge(df_fasta[["protein_name", "sequence"]], on="protein_name", how="left")
    if merged["sequence"].isna().any():
        missing = merged.loc[merged["sequence"].isna(), "protein_name"].tolist()[:10]
        raise ValueError(f"The following protein_names were not found in FASTA, cannot perform occlusion analysis: {missing}")
    return merged


def pick_focus_samples(df_pred: pd.DataFrame, focus_label: int, n_samples: int, require_correct: bool = True):
    df = df_pred.copy()
    if require_correct:
        df = df[df["pred"] == df["label"]]
    df = df[df["label"] == focus_label].copy()
    df = df.sort_values("prob", ascending=(focus_label == 0)).reset_index(drop=True)
    return df.head(n_samples)


@torch.no_grad()
def predict_from_raw_embeddings(model, scaler, feat_cols, raw_embeddings: np.ndarray, device: torch.device):
    X_df = pd.DataFrame(raw_embeddings, columns=feat_cols)
    X = scaler.transform(X_df.values).astype(np.float32)
    return predict_proba(model, X, device=device, batch_size=4096)


def run_occlusion_for_sample(sample_row, client, model, scaler, feat_cols, device, window_size: int, stride: int, mask_batch_size: int, max_length: int):
    protein_name = sample_row["protein_name"]
    seq = sample_row["sequence"]
    base_prob = float(sample_row["prob"])
    windows = sliding_windows(len(seq), window_size=window_size, stride=stride)
    masked_items = []
    meta = []
    for idx, (start, end) in enumerate(windows):
        masked_seq = mask_sequence_with_X(seq, start, end)
        masked_name = f"{protein_name}__mask_{idx}__{start}_{end}"
        masked_items.append((masked_name, masked_seq))
        meta.append((start, end, seq[start:end]))
    _, masked_emb = extract_protrek_embeddings_from_sequences(masked_items, client=client, batch_size=mask_batch_size, max_length=max_length)
    masked_prob = predict_from_raw_embeddings(model=model, scaler=scaler, feat_cols=feat_cols, raw_embeddings=masked_emb, device=device)
    rows = []
    residue_score = np.zeros(len(seq), dtype=np.float64)
    residue_count = np.zeros(len(seq), dtype=np.float64)
    for (start, end, fragment), p_mask in zip(meta, masked_prob):
        delta = base_prob - float(p_mask)
        rows.append({
            "protein_name": protein_name,
            "label": int(sample_row["label"]),
            "pred": int(sample_row["pred"]),
            "base_prob": base_prob,
            "masked_prob": float(p_mask),
            "delta_prob": delta,
            "start": start,
            "end": end,
            "window_size": end - start,
            "fragment": fragment,
            "sequence": seq
        })
        residue_score[start:end] += delta
        residue_count[start:end] += 1.0
    residue_importance = np.divide(residue_score, np.maximum(residue_count, 1.0), out=np.zeros_like(residue_score), where=residue_count > 0)
    residue_rows = []
    for i, aa in enumerate(seq):
        residue_rows.append({
            "protein_name": protein_name,
            "label": int(sample_row["label"]),
            "pred": int(sample_row["pred"]),
            "position": i,
            "aa": aa,
            "residue_importance": float(residue_importance[i])
        })
    return rows, residue_rows


def summarize_key_fragments(window_df: pd.DataFrame, top_n_per_seq: int):
    picked = window_df.sort_values(["protein_name", "delta_prob"], ascending=[True, False]) \
                       .groupby("protein_name", as_index=False).head(top_n_per_seq).reset_index(drop=True)
    prop_rows = []
    for _, row in picked.iterrows():
        props = summarize_fragment_properties(row["fragment"])
        prop_rows.append({**row.to_dict(), **props})
    return pd.DataFrame(prop_rows)


def build_background_fragments(window_df: pd.DataFrame, top_n_per_seq: int):
    bg = window_df.sort_values(["protein_name", "delta_prob"], ascending=[True, True]) \
                   .groupby("protein_name", as_index=False).head(top_n_per_seq).reset_index(drop=True)
    prop_rows = []
    for _, row in bg.iterrows():
        props = summarize_fragment_properties(row["fragment"])
        prop_rows.append({**row.to_dict(), **props})
    return pd.DataFrame(prop_rows)


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


def cohen_d(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled))


def run_fragment_property_stats(important_df: pd.DataFrame, background_df: pd.DataFrame):
    metrics = ["net_charge", "mean_hydrophobicity", "hydrophobic_fraction", "positive_fraction", "hydrophobic_moment"]
    rows = []
    for metric in metrics:
        x = important_df[metric].dropna().values.astype(np.float64)
        y = background_df[metric].dropna().values.astype(np.float64)
        stat = mannwhitneyu(y, x, alternative="two-sided")
        d = cohen_d(x, y)
        rows.append({
            "metric": metric,
            "important_mean": float(np.mean(x)) if len(x) else float("nan"),
            "background_mean": float(np.mean(y)) if len(y) else float("nan"),
            "important_median": float(np.median(x)) if len(x) else float("nan"),
            "background_median": float(np.median(y)) if len(y) else float("nan"),
            "mannwhitney_u": float(stat.statistic),
            "p_value": float(stat.pvalue),
            "cliffs_delta": cliffs_delta(x, y),
            "cohen_d": d,
            "direction": "important > background" if np.mean(x) > np.mean(y) else "important < background"
        })
    return pd.DataFrame(rows)


def save_motif_matrices(important_frags, background_frags, out_dir: str):
    imp_mat = sequence_logo_matrix(important_frags)
    bg_mat = sequence_logo_matrix(background_frags)
    imp_mat.to_csv(os.path.join(out_dir, "important_fragments_motif_freq_matrix.csv"), index=True, encoding="utf-8-sig")
    bg_mat.to_csv(os.path.join(out_dir, "background_fragments_motif_freq_matrix.csv"), index=True, encoding="utf-8-sig")
    return imp_mat, bg_mat


def plot_motif_comparison(df_imp: pd.DataFrame, df_bg: pd.DataFrame, out_path: str):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    sns.set_context("paper", font_scale=1.2)
    vmax = max(df_imp.max().max(), df_bg.max().max())
    vmin = min(df_imp.min().min(), df_bg.min().min())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    cmap_color = "YlGnBu"
    sns.heatmap(df_imp, ax=ax1, cmap=cmap_color, vmin=vmin, vmax=vmax, annot=False,
                cbar_kws={"label": "Frequency"}, linewidths=0.5, linecolor="white")
    ax1.set_title("AA frequency heatmap of important fragments", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Amino Acids", fontsize=12)
    ax1.set_ylabel("Position", fontsize=12)
    highlight_aas = {"K", "R", "V", "L"}
    for label in ax1.get_xticklabels():
        aa_text = label.get_text()
        if aa_text in highlight_aas:
            label.set_color("red")
            label.set_fontweight("bold")
            label.set_fontsize(14)
    sns.heatmap(df_bg, ax=ax2, cmap=cmap_color, vmin=vmin, vmax=vmax, annot=False,
                cbar_kws={"label": "Frequency"}, linewidths=0.5, linecolor="white")
    ax2.set_title("AA frequency heatmap of background fragments", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Amino Acids", fontsize=12)
    ax2.set_ylabel("")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_biophysical_boxplots(df_long: pd.DataFrame, out_path: str):
    import warnings
    warnings.filterwarnings("ignore")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    features = ["net_charge", "mean_hydrophobicity", "hydrophobic_fraction", "positive_fraction", "hydrophobic_moment"]
    title_map = {
        "net_charge": "Net Charge",
        "mean_hydrophobicity": "Hydrophobicity",
        "hydrophobic_fraction": "Hydro. Fraction",
        "positive_fraction": "Pos. Fraction",
        "hydrophobic_moment": "Hydro. Moment"
    }
    palette = {"background": COLOR_BG, "important": COLOR_IMP}
    group_order = ["background", "important"]
    fig, axes = plt.subplots(1, 5, figsize=(10, 3), constrained_layout=True)
    for i, feature in enumerate(features):
        ax = axes[i]
        g1 = df_long[df_long["group"] == "background"][feature]
        g2 = df_long[df_long["group"] == "important"][feature]
        _, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
        d_val = cohen_d(g2.values, g1.values)
        sns.boxplot(data=df_long, x="group", y=feature, ax=ax, order=group_order,
                    hue="group", palette=palette, width=0.6, showfliers=False,
                    linewidth=1.5, legend=False)
        y_max = df_long[feature].max()
        y_min = df_long[feature].min()
        y_range = y_max - y_min if y_max > y_min else 1.0
        line_y = y_max + y_range * 0.1
        tip_h = y_range * 0.04
        ax.plot([0, 0, 1, 1], [line_y - tip_h, line_y, line_y, line_y - tip_h], lw=1, c="black")
        p_str = f"p={p_val:.3f}" if p_val >= 0.001 else f"p={p_val:.2e}"
        d_str = f"d={abs(d_val):.2f}"
        ax.text(0.5, line_y + y_range * 0.02, f"{p_str}\n{d_str}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title_map[feature], fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["BG", "IMP"], fontsize=10)
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.4)
        sns.despine(ax=ax)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def run_stage2(args):
    set_seed(args.seed)
    make_output_dir(args.out_dir)
    model, scaler, feat_cols, config = train_or_load_gated_tabm(
        args.train_merged, args.label_col, args.checkpoint,
        seed=args.seed, config=BEST_CONFIG, force_retrain=args.force_retrain
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    df_test_csv, X_scaled, y = prepare_test_data(args.pos_test, args.neg_test, feat_cols, scaler, label_col=args.label_col)
    prob = predict_proba(model, X_scaled, device=device, batch_size=4096)
    pred = (prob >= FIXED_THRESHOLD).astype(int)
    df_pred = df_test_csv.copy()
    df_pred["prob"] = prob
    df_pred["pred"] = pred
    df_pred = merge_test_csv_with_fasta(df_pred, args.pos_test_fasta, args.neg_test_fasta)
    df_pred.to_csv(os.path.join(args.out_dir, "stage2_test_predictions_with_sequence.csv"), index=False, encoding="utf-8-sig")
    metrics = evaluate_predictions(y, prob)
    focus_df = pick_focus_samples(
        df_pred=df_pred, focus_label=args.focus_label,
        n_samples=args.n_samples, require_correct=args.require_correct
    )
    focus_df.to_csv(os.path.join(args.out_dir, "selected_focus_samples.csv"), index=False, encoding="utf-8-sig")
    client = load_protrek_650m_local(device=device, model_root=args.protrek_model_root, protrek_code_root=args.protrek_code_root)
    all_window_rows = []
    all_residue_rows = []
    for _, row in focus_df.iterrows():
        print(f"[Occlusion] processing {row['protein_name']}")
        window_rows, residue_rows = run_occlusion_for_sample(
            row, client, model, scaler, feat_cols, device,
            args.window_size, args.stride, args.mask_batch_size, args.max_length
        )
        all_window_rows.extend(window_rows)
        all_residue_rows.extend(residue_rows)
    window_df = pd.DataFrame(all_window_rows)
    residue_df = pd.DataFrame(all_residue_rows)
    window_df.to_csv(os.path.join(args.out_dir, "occlusion_window_importance.csv"), index=False, encoding="utf-8-sig")
    residue_df.to_csv(os.path.join(args.out_dir, "occlusion_residue_importance.csv"), index=False, encoding="utf-8-sig")
    important_df = summarize_key_fragments(window_df, top_n_per_seq=args.top_n_per_seq)
    background_df = build_background_fragments(window_df, top_n_per_seq=args.top_n_per_seq)
    important_df.to_csv(os.path.join(args.out_dir, "key_fragments_top_windows.csv"), index=False, encoding="utf-8-sig")
    background_df.to_csv(os.path.join(args.out_dir, "background_fragments_bottom_windows.csv"), index=False, encoding="utf-8-sig")
    frag_summary = important_df.groupby("fragment").agg(
        n_occurrence=("fragment", "count"),
        mean_delta_prob=("delta_prob", "mean"),
        mean_charge=("net_charge", "mean"),
        mean_hydrophobicity=("mean_hydrophobicity", "mean"),
        mean_positive_fraction=("positive_fraction", "mean")
    ).sort_values(["n_occurrence", "mean_delta_prob"], ascending=[False, False]).reset_index()
    frag_summary.to_csv(os.path.join(args.out_dir, "key_fragment_summary.csv"), index=False, encoding="utf-8-sig")
    important_frags = important_df["fragment"].tolist()
    background_frags = background_df["fragment"].tolist()
    imp_mat, bg_mat = save_motif_matrices(important_frags, background_frags, args.out_dir)
    plot_motif_comparison(imp_mat, bg_mat, os.path.join(args.out_dir, "motif_heatmap_comparison_highlighted.png"))
    fragment_properties = pd.concat([
        background_df.assign(group="background"),
        important_df.assign(group="important")
    ], axis=0, ignore_index=True)
    fragment_properties.to_csv(os.path.join(args.out_dir, "fragment_properties_for_origin.csv"), index=False, encoding="utf-8-sig")
    stats_df = run_fragment_property_stats(important_df, background_df)
    stats_df.to_csv(os.path.join(args.out_dir, "fragment_property_stat_tests.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, "fragment_property_stat_tests.json"), "w", encoding="utf-8") as f:
        json.dump(stats_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    plot_biophysical_boxplots(fragment_properties, os.path.join(args.out_dir, "biophysical_p_and_d_values.png"))
    summary = {
        "config": config,
        "test_metrics": metrics,
        "n_focus_samples": int(len(focus_df)),
        "window_size": args.window_size,
        "stride": args.stride,
        "top_n_per_seq": args.top_n_per_seq,
        "focus_label": args.focus_label,
        "require_correct": bool(args.require_correct),
        "outputs": [
            "selected_focus_samples.csv",
            "occlusion_window_importance.csv",
            "occlusion_residue_importance.csv",
            "key_fragments_top_windows.csv",
            "background_fragments_bottom_windows.csv",
            "key_fragment_summary.csv",
            "important_fragments_motif_freq_matrix.csv",
            "background_fragments_motif_freq_matrix.csv",
            "fragment_properties_for_origin.csv",
            "fragment_property_stat_tests.csv",
            "fragment_property_stat_tests.json",
            "motif_heatmap_comparison_highlighted.png",
            "biophysical_p_and_d_values.png"
        ]
    }
    with open(os.path.join(args.out_dir, "stage2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Stage-2 outputs saved to:", args.out_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Clean stage-2 explainability: motif heatmap + biophysical properties")
    parser.add_argument("--train_merged", type=str, default="train_protrek650m_gated_tabm.csv")
    parser.add_argument("--pos_test", type=str, default="positive_protrek650m_test.csv")
    parser.add_argument("--neg_test", type=str, default="negative_protrek650m_test.csv")
    parser.add_argument("--pos_test_fasta", type=str, required=True)
    parser.add_argument("--neg_test_fasta", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default="explainability_outputs/gated_tabm_model.pt")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--out_dir", type=str, default="explainability_outputs/stage2_fragments")
    parser.add_argument("--protrek_model_root", type=str, required=True)
    parser.add_argument("--protrek_code_root", type=str, required=True)
    parser.add_argument("--mask_batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--focus_label", type=int, default=1, choices=[0, 1])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--require_correct", action="store_true")
    parser.add_argument("--top_n_per_seq", type=int, default=3)
    return parser


if __name__ == "__main__":
    run_stage2(build_parser().parse_args())