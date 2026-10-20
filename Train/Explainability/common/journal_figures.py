"""BDMA-ready rendering of manuscript Figures 5–9 from retained result tables.

Canonicals: 17 cm wide, RGB TIFF (lossless LZW) at 600 dpi. PNGs are previews.
The journal specifies figure file format and preferred resolution, but no
mandatory in-art font. Arial at ~9–10 pt follows IEEE graphics guidance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


WIDTH_IN = 17 / 2.54
DPI = 600
BLUE = "#6EAFDD"
NAVY = "#2C2C64"
LIGHT_BLUE = "#9ECAE1"
MUTED_NAVY = "#4A4A79"
METHODS = ["X masking", "Alanine scan", "Window shuffle"]
GROUPS = ["TP", "FN", "FP", "TN"]
METHOD_COLORS = {"X masking": MUTED_NAVY, "Alanine scan": "#D08770",
                 "Window shuffle": "#70A288"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9.5,
        "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def panel(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.05) -> None:
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", ha="left", va="bottom")


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tiff = output / f"{stem}.tiff"
    fig.savefig(tiff, dpi=DPI, format="tiff",
                pil_kwargs={"compression": "tiff_lzw"})
    with Image.open(tiff) as im:
        rgb = im.convert("RGB")
        rgb.save(tiff, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    fig.savefig(output / f"{stem}.png", dpi=DPI, format="png")
    plt.close(fig)


def figure5(result_dir: Path, output_dir: Path) -> None:
    style()
    raw = pd.read_csv(result_dir / "umap_raw_embedding.csv")
    gated = pd.read_csv(result_dir / "umap_gated_embedding.csv")
    assert len(raw) == len(gated) == 3072
    assert raw.label.value_counts().to_dict() == {0: 1536, 1: 1536}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.35), constrained_layout=True)
    for ax, frame, letter, title in zip(axes, (raw, gated), "ab",
                                         ("Raw ProTrek embedding", "Gated embedding")):
        for value, name, color in ((0, "non-AMP", BLUE), (1, "AMP", NAVY)):
            part = frame[frame.label == value]
            ax.scatter(part.umap1, part.umap2, s=4, alpha=0.75,
                       c=color, edgecolors="none", label=name, rasterized=True)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(title, pad=8)
        ax.legend(loc="upper left" if letter == "a" else "upper right",
                  frameon=False, markerscale=2.5,
                  handletextpad=0.3, borderaxespad=0.2)
        clean(ax)
        panel(ax, letter, x=-0.14)
    save(fig, output_dir, "Figure_5_UMAP")


def figure6(result_dir: Path, output_dir: Path) -> None:
    style()
    frame = pd.read_csv(result_dir / "representation_metric_estimates.csv")
    metrics = [("silhouette", "Silhouette score"),
               ("knn_accuracy", "5-NN accuracy"),
               ("centroid_margin", "Cosine separation margin")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 2.50), constrained_layout=True)
    for ax, (metric, title), letter in zip(axes, metrics, "abc"):
        part = frame[frame.metric == metric].set_index("representation").loc[["Raw ProTrek", "Gated Adapter"]]
        values = part.estimate.to_numpy(float)
        low, high = part.ci_low.to_numpy(float), part.ci_high.to_numpy(float)
        x = np.arange(2)
        ax.bar(x, values, width=0.62, color=[LIGHT_BLUE, MUTED_NAVY],
               edgecolor="black", linewidth=0.45)
        ax.errorbar(x, values, yerr=np.vstack([values - low, high - values]),
                    fmt="none", ecolor="black", capsize=2.5, lw=0.8)
        ax.set_xticks(x, ["Raw", "Gated"])
        ax.set_title(title, pad=7)
        ax.set_ylim(0, max(high) * 1.22)
        for xi, val, ci_high in zip(x, values, high):
            ax.text(xi, ci_high + max(high) * 0.035, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8.5)
        clean(ax)
        panel(ax, letter, x=-0.15)
    save(fig, output_dir, "Figure_6_High_dimensional_separability")


def figure7(result_dir: Path, output_dir: Path) -> None:
    style()
    stats = pd.read_csv(result_dir / "aopc_cumulative_alanine_statistics.csv")
    randomization = pd.read_csv(result_dir / "randomized_model_sanity_peptide_level.csv")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN, 3.05), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.05, 0.95]})
    overall = stats[stats.group == "All"].sort_values("step")
    ax = axes[0]
    ax.plot(overall.step, overall.selected_mean_evidence_removed, marker="o", ms=4,
            color=NAVY, lw=1.5, label="Ranked")
    ax.plot(overall.step, overall.random_mean_evidence_removed, marker="o", ms=4,
            color=BLUE, lw=1.5, label="Random")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Windows perturbed")
    ax.set_ylabel("Evidence removed")
    ax.set_title("Cumulative alanine perturbation", pad=8)
    ax.legend(frameon=False, loc="center right")
    clean(ax)
    panel(ax, "a", x=-0.19)

    ax = axes[1]
    grouped = stats[(stats.step == 3) & stats.group.isin(GROUPS)].set_index("group").loc[GROUPS]
    values = grouped.selected_minus_random.to_numpy(float)
    ax.errorbar(values, np.arange(4),
                xerr=np.vstack([values - grouped.ci_low, grouped.ci_high - values]),
                fmt="o", ms=4, color=NAVY, capsize=2.5, lw=1)
    ax.axvline(0, color="#777777", ls="--", lw=0.7)
    ax.set_yticks(np.arange(4), GROUPS)
    ax.invert_yaxis()
    ax.set_xlabel("Ranked − random (95% CI)")
    ax.set_title("By outcome, three windows", pad=8)
    clean(ax)
    panel(ax, "b", x=-0.10)

    ax = axes[2]
    trained = randomization[randomization.model == "Trained model"].top_minus_random.to_numpy(float)
    randomized = (randomization[randomization.model == "Randomized model"]
                  .groupby("seed").top_minus_random.mean().to_numpy(float))
    bp = ax.boxplot([trained], positions=[0], widths=0.48, patch_artist=True,
                    showfliers=False, medianprops={"color": "#D08770", "linewidth": 1.5})
    bp["boxes"][0].set_facecolor(MUTED_NAVY)
    bp["boxes"][0].set_alpha(0.8)
    ax.scatter(1 + np.linspace(-0.09, 0.09, len(randomized)), randomized,
               color=BLUE, s=19, zorder=3)
    ax.axhline(0, color="#777777", ls="--", lw=0.7)
    ax.set_xticks([0, 1], ["Trained\n(peptides)", "Randomized\n(5 seeds)"])
    ax.set_ylabel("Ranked − random")
    ax.set_title("Parameter randomization", pad=8)
    clean(ax)
    panel(ax, "c", x=-0.10)
    save(fig, output_dir, "Figure_7_AOPC_randomization")


def figure8(result_dir: Path, output_dir: Path) -> None:
    style()
    summary = pd.read_csv(result_dir / "perturbation_robustness_summary.csv")
    peptide = pd.read_csv(result_dir / "peptide_level_perturbation_results.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 3.15), constrained_layout=True)
    important = summary[summary.role == "important"]
    x = np.arange(4)
    width = 0.23
    for offset, method in enumerate(METHODS):
        part = important[important.method == method].set_index("group").loc[GROUPS]
        values = part.mean_delta_logit.to_numpy(float)
        low, high = part.ci_low.to_numpy(float), part.ci_high.to_numpy(float)
        xpos = x + (offset - 1) * width
        ax1.bar(xpos, values, width=width, color=METHOD_COLORS[method],
                edgecolor="black", linewidth=0.4, label=method)
        ax1.errorbar(xpos, values, yerr=np.vstack([values - low, high - values]),
                     fmt="none", ecolor="black", capsize=2, lw=0.7)
    ax1.axhline(0, color="black", lw=0.7)
    ax1.set_xticks(x, GROUPS)
    ax1.set_ylabel("Mean change in AMP logit")
    ax1.set_title("Sensitivity by prediction outcome", pad=8)
    ax1.legend(frameon=False, loc="upper right", fontsize=8.2)
    clean(ax1)
    panel(ax1, "a", x=-0.13)

    shift = (peptide.groupby(["method", "role"], as_index=False).embedding_cosine_shift
             .agg(["mean", "sem"]).reset_index())
    positions = np.arange(3)
    for role, delta, color in (("background", -0.17, LIGHT_BLUE),
                               ("important", 0.17, MUTED_NAVY)):
        part = shift[shift.role == role].set_index("method").loc[METHODS]
        ax2.bar(positions + delta, part["mean"], width=0.33,
                yerr=1.96 * part["sem"], color=color, edgecolor="black",
                linewidth=0.4, capsize=2, label=role.capitalize())
    ax2.set_xticks(positions, ["X mask", "Alanine", "Shuffle"])
    ax2.set_ylabel("Embedding cosine shift")
    ax2.set_title("Task-representation response", pad=8)
    ax2.legend(frameon=False, loc="upper right")
    clean(ax2)
    panel(ax2, "b", x=-0.15)
    save(fig, output_dir, "Figure_8_Perturbation_robustness")


def figure9(result_dir: Path, output_dir: Path) -> None:
    style()
    profiles = pd.read_csv(result_dir / "residue_sensitivity_profiles.csv")
    cases = pd.read_csv(result_dir / "representative_cases.csv").set_index("protein_name")
    ids = ["ALL000830", "ALL003142", "sp|Q91G85|009R_IIV6"]
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH_IN, 6.15), constrained_layout=True)
    for ax, case_id, letter in zip(axes, ids, "abc"):
        part = profiles[profiles.protein_name == case_id].sort_values("position")
        case = cases.loc[case_id]
        positions = part.position.to_numpy(int)
        scores = part.mean_delta_logit.to_numpy(float)
        ax.fill_between(positions, 0, scores, where=scores >= 0,
                        color="#C44E52", alpha=0.55, interpolate=True)
        ax.fill_between(positions, 0, scores, where=scores < 0,
                        color="#5B9BD5", alpha=0.55, interpolate=True)
        ax.plot(positions, scores, color="#333333", lw=0.8)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel("Mean Δlogit")
        ax.set_title(f"{case.group}: {case_id} (length {len(positions)})", loc="left", pad=8)
        clean(ax)
        panel(ax, letter, x=-0.08)
    axes[-1].set_xlabel("Residue position")
    save(fig, output_dir, "Figure_9_Residue_sensitivity")
