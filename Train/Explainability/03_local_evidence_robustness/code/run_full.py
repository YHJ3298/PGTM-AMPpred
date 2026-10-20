from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC = Path()
OUT = PACKAGE_ROOT / "03_local_evidence_robustness" / "scratch_full_run"
MODEL_ROOT = Path()
PROTREK_CODE = Path()

sys.path.insert(0, str(PACKAGE_ROOT / "common"))

from explainability_pipeline_utils import (  # noqa: E402
    build_fasta_df,
    load_protrek_650m_local,
    predict_proba,
    set_seed,
    sliding_windows,
    train_or_load_gated_tabm,
)


SEED = 42
WINDOW_SIZE = 7
STRIDE = 1
TOP_N = 3
N_PER_GROUP = 76
SHUFFLE_REPEATS = 10
N_BOOT = 3000
N_PERM = 20000
GROUP_ORDER = ["TP", "FN", "FP", "TN"]


def dirs() -> tuple[Path, Path, Path]:
    figures = OUT / "figures"
    tables = OUT / "tables"
    data = OUT / "data"
    for path in (figures, tables, data):
        path.mkdir(parents=True, exist_ok=True)
    return figures, tables, data


def logit(p: np.ndarray | float) -> np.ndarray:
    value = np.asarray(p, dtype=np.float64)
    value = np.clip(value, 1e-7, 1.0 - 1e-7)
    return np.log(value / (1.0 - value))


def add_group(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["group"] = np.select(
        [
            (out["label"] == 1) & (out["pred"] == 1),
            (out["label"] == 1) & (out["pred"] == 0),
            (out["label"] == 0) & (out["pred"] == 1),
        ],
        ["TP", "FN", "FP"],
        default="TN",
    )
    return out


def proportional_stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    work = frame.copy()
    work["length_bin"] = pd.qcut(work["length"], q=4, duplicates="drop").astype(str)
    work["prob_bin"] = pd.qcut(work["prob"], q=4, duplicates="drop").astype(str)
    work["stratum"] = work["length_bin"] + "|" + work["prob_bin"]
    counts = work["stratum"].value_counts().sort_index()
    exact = counts / counts.sum() * n
    allocation = np.floor(exact).astype(int)
    allocation = np.minimum(allocation, counts)
    remainder = n - int(allocation.sum())
    fractions = (exact - np.floor(exact)).sort_values(ascending=False)
    while remainder > 0:
        progressed = False
        for stratum in fractions.index:
            if allocation[stratum] < counts[stratum]:
                allocation[stratum] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            break
    pieces = []
    for offset, stratum in enumerate(counts.index):
        size = int(allocation[stratum])
        if size:
            pieces.append(work[work["stratum"] == stratum].sample(n=size, random_state=seed + offset))
    result = pd.concat(pieces, ignore_index=True)
    return result.drop(columns=["length_bin", "prob_bin", "stratum"])


def select_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for offset, group in enumerate(GROUP_ORDER):
        part = frame[frame["group"] == group].copy()
        selected = part if group == "FP" else proportional_stratified_sample(part, N_PER_GROUP, SEED + 100 * offset)
        selected["selection_rule"] = "all" if group == "FP" else "length/probability-quantile stratified random sample"
        pieces.append(selected)
    cohort = pd.concat(pieces, ignore_index=True)
    cohort["high_confidence_fp"] = (cohort["group"] == "FP") & (cohort["prob"] >= 0.90)
    return cohort.sort_values(["group", "protein_name"]).reset_index(drop=True)


def build_all_window_records(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in cohort.itertuples(index=False):
        for window_index, (start, end) in enumerate(sliding_windows(len(row.sequence), WINDOW_SIZE, STRIDE)):
            rows.append(
                {
                    "protein_name": row.protein_name,
                    "group": row.group,
                    "label": int(row.label),
                    "pred": int(row.pred),
                    "base_prob": float(row.prob),
                    "sequence": row.sequence,
                    "length": int(row.length),
                    "window_index": window_index,
                    "start": int(start),
                    "end": int(end),
                    "fragment": row.sequence[start:end],
                }
            )
    return pd.DataFrame(rows)


def perturb_sequence(sequence: str, start: int, end: int, method: str, rng: np.random.Generator | None = None) -> str:
    if method == "X masking":
        replacement = "X" * (end - start)
    elif method == "Alanine scan":
        replacement = "A" * (end - start)
    elif method == "Window shuffle":
        chars = list(sequence[start:end])
        if rng is None:
            raise ValueError("Window shuffle requires an RNG")
        replacement = "".join(rng.permutation(chars).tolist())
    else:
        raise ValueError(method)
    return sequence[:start] + replacement + sequence[end:]


def predict_embeddings(model, scaler, feat_cols: list[str], embeddings: np.ndarray, device: torch.device) -> np.ndarray:
    scaled = scaler.transform(embeddings).astype(np.float32)
    return predict_proba(model, scaled, device=device, batch_size=4096)


def output_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def run_embedding_records(
    records: pd.DataFrame,
    method: str,
    output_path: Path,
    client,
    classifier,
    scaler,
    feat_cols: list[str],
    original_embedding_map: dict[str, np.ndarray],
    device: torch.device,
    outer_batch: int,
    inner_batch: int,
) -> pd.DataFrame:
    if output_path.exists() and output_row_count(output_path) == len(records):
        print(f"[reuse] {output_path.name}: {len(records):,} rows")
        return pd.read_csv(output_path)
    partial = output_path.with_suffix(".partial.csv")
    completed = output_row_count(partial)
    if completed > len(records):
        raise RuntimeError(f"Partial output has too many rows: {partial}")
    print(f"[{method}] total={len(records):,}, resume_at={completed:,}")
    rng = np.random.default_rng(SEED + {"X masking": 1, "Alanine scan": 2, "Window shuffle": 3}[method])
    if method == "Window shuffle" and completed:
        # Advance the deterministic generator to preserve the exact perturbation sequence.
        for row in records.iloc[:completed].itertuples(index=False):
            _ = perturb_sequence(row.sequence, int(row.start), int(row.end), method, rng)
    for batch_start in range(completed, len(records), outer_batch):
        batch = records.iloc[batch_start : batch_start + outer_batch].copy()
        sequences = [
            perturb_sequence(r.sequence, int(r.start), int(r.end), method, rng)
            for r in batch.itertuples(index=False)
        ]
        with torch.inference_mode():
            embeddings_tensor = client.get_protein_repr(sequences, batch_size=inner_batch, verbose=False)
        embeddings = embeddings_tensor.float().detach().cpu().numpy().astype(np.float32)
        probs = predict_embeddings(classifier, scaler, feat_cols, embeddings, device)
        original = np.stack([original_embedding_map[str(name)] for name in batch["protein_name"]], axis=0).astype(np.float32)
        cosine = np.sum(original * embeddings, axis=1) / np.clip(
            np.linalg.norm(original, axis=1) * np.linalg.norm(embeddings, axis=1), 1e-12, None
        )
        output = batch.drop(columns=["sequence"]).copy()
        output["method"] = method
        output["perturbed_prob"] = probs
        output["delta_prob"] = output["base_prob"].to_numpy() - probs
        output["delta_logit"] = logit(output["base_prob"].to_numpy()) - logit(probs)
        output["embedding_cosine_shift"] = 1.0 - cosine
        output.to_csv(
            partial,
            mode="a",
            header=not partial.exists() or output_row_count(partial) == 0,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[{method}] {min(batch_start + len(batch), len(records)):,}/{len(records):,}")
        del embeddings_tensor, embeddings, original, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    partial.replace(output_path)
    return pd.read_csv(output_path)


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int, threshold: float = 0.5) -> bool:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    return inter / max(1, min(a_end - a_start, b_end - b_start)) > threshold


def choose_windows(x_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    important_rows = []
    background_rows = []
    rng = np.random.default_rng(SEED + 991)
    for protein_name, part in x_results.groupby("protein_name", sort=True):
        selected = []
        for row in part.sort_values("delta_logit", ascending=False).itertuples(index=False):
            if all(not overlaps(row.start, row.end, other.start, other.end) for other in selected):
                selected.append(row)
            if len(selected) == TOP_N:
                break
        important_rows.extend([r._asdict() for r in selected])

        candidates = part.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        background = []
        for row in candidates.itertuples(index=False):
            if any(overlaps(row.start, row.end, imp.start, imp.end, threshold=0.0) for imp in selected):
                continue
            if all(not overlaps(row.start, row.end, other.start, other.end) for other in background):
                background.append(row)
            if len(background) == len(selected):
                break
        background_rows.extend([r._asdict() for r in background])
    imp = pd.DataFrame(important_rows).assign(role="important")
    bg = pd.DataFrame(background_rows).assign(role="background")
    return imp, bg


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def bootstrap_mean_diff(diff: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        values[i] = diff[rng.integers(0, len(diff), size=len(diff))].mean()
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_permutation_p(diff: np.ndarray, n_perm: int = N_PERM, seed: int = SEED) -> float:
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = abs(float(diff.mean()))
    extreme = 0
    done = 0
    batch_size = 1000
    while done < n_perm:
        current = min(batch_size, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, len(diff)))
        perm = np.abs((signs * diff).mean(axis=1))
        extreme += int(np.sum(perm >= observed - 1e-15))
        done += current
    return float((extreme + 1) / (n_perm + 1))


def build_shuffle_records(important: pd.DataFrame, background: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    sequence_map = cohort.set_index("protein_name")["sequence"].to_dict()
    rows = []
    for role_frame in (important, background):
        for row in role_frame.itertuples(index=False):
            for repeat in range(SHUFFLE_REPEATS):
                item = row._asdict()
                item["sequence"] = sequence_map[row.protein_name]
                item["shuffle_repeat"] = repeat
                rows.append(item)
    return pd.DataFrame(rows)


def selected_method_results(
    important: pd.DataFrame,
    background: pd.DataFrame,
    x_results: pd.DataFrame,
    alanine_results: pd.DataFrame,
    shuffle_results: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["protein_name", "start", "end"]
    roles = pd.concat([important[keys + ["role"]], background[keys + ["role"]]], ignore_index=True)
    x = roles.merge(x_results, on=keys, how="left")
    a = roles.merge(alanine_results, on=keys, how="left")
    # Shuffle records are generated from the selected-window manifest and
    # therefore already carry their important/background role.  Merging the
    # role manifest again would create role_x/role_y and break the grouping.
    s = shuffle_results.copy() if "role" in shuffle_results.columns else roles.merge(shuffle_results, on=keys, how="left")
    s = (
        s.groupby(keys + ["role", "group", "label", "pred", "fragment", "method"], as_index=False)
        .agg(
            base_prob=("base_prob", "first"),
            perturbed_prob=("perturbed_prob", "mean"),
            delta_prob=("delta_prob", "mean"),
            delta_logit=("delta_logit", "mean"),
            embedding_cosine_shift=("embedding_cosine_shift", "mean"),
        )
    )
    common = [
        "protein_name",
        "start",
        "end",
        "role",
        "group",
        "label",
        "pred",
        "fragment",
        "method",
        "base_prob",
        "perturbed_prob",
        "delta_prob",
        "delta_logit",
        "embedding_cosine_shift",
    ]
    return pd.concat([x[common], a[common], s[common]], ignore_index=True)


def summarize_perturbation(selected_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    peptide = (
        selected_results.groupby(["protein_name", "group", "role", "method"], as_index=False)
        .agg(delta_logit=("delta_logit", "mean"), delta_prob=("delta_prob", "mean"), embedding_cosine_shift=("embedding_cosine_shift", "mean"))
    )
    rows = []
    for (group, role, method), part in peptide.groupby(["group", "role", "method"]):
        values = part["delta_logit"].to_numpy(dtype=float)
        low, high = bootstrap_mean_diff(values, seed=SEED + len(rows))
        rows.append(
            {
                "group": group,
                "role": role,
                "method": method,
                "n_peptides": int(len(values)),
                "mean_delta_logit": float(values.mean()),
                "median_delta_logit": float(np.median(values)),
                "ci_low": low,
                "ci_high": high,
                "positive_fraction": float((values > 0).mean()),
                "mean_embedding_cosine_shift": float(part["embedding_cosine_shift"].mean()),
            }
        )
    return peptide, pd.DataFrame(rows)


def perturbation_role_statistics(peptide: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    """Paired important-minus-background tests at the peptide level."""
    rows = []
    for (stratum, method), part in peptide.groupby([stratum_col, "method"]):
        wide = part.pivot(index="protein_name", columns="role", values="delta_logit").dropna()
        if wide.empty:
            continue
        imp = wide["important"].to_numpy(dtype=float)
        bg = wide["background"].to_numpy(dtype=float)
        diff = imp - bg
        low, high = bootstrap_mean_diff(diff, seed=SEED + len(rows) + 2100)
        sd = np.std(diff, ddof=1) if len(diff) > 1 else float("nan")
        rows.append(
            {
                stratum_col: stratum,
                "method": method,
                "n_peptides": int(len(wide)),
                "important_mean_delta_logit": float(imp.mean()),
                "background_mean_delta_logit": float(bg.mean()),
                "paired_mean_difference": float(diff.mean()),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "paired_cohen_dz": float(diff.mean() / sd) if np.isfinite(sd) and sd > 0 else float("nan"),
                "permutation_p": paired_permutation_p(diff, seed=SEED + len(rows) + 2100),
            }
        )
    stats = pd.DataFrame(rows)
    stats["fdr_q"] = np.nan
    for stratum in stats[stratum_col].unique():
        mask = stats[stratum_col] == stratum
        stats.loc[mask, "fdr_q"] = bh_adjust(stats.loc[mask, "permutation_p"])
    return stats


def main() -> None:
    global SRC, OUT, MODEL_ROOT, PROTREK_CODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer_batch", type=int, default=32)
    parser.add_argument("--inner_batch", type=int, default=16)
    parser.add_argument("--analysis_only", action="store_true")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Train directory containing features, explain_outputs/, and datasets/")
    parser.add_argument("--model-root", type=Path,
                        help="Local ProTrek_650M model directory; required for full inference")
    parser.add_argument("--protrek-code-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    SRC = args.source_root.resolve()
    OUT = args.output_dir.resolve()
    MODEL_ROOT = args.model_root.resolve() if args.model_root else Path()
    PROTREK_CODE = (args.protrek_code_root or SRC / "ProTrek-main").resolve()
    fasta_root = SRC / "datasets" / "XUAMP" / "XU_test"
    required = [SRC / "explain_outputs" / "gated_tabm_independent_predictions.csv",
                SRC / "explain_outputs" / "gated_tabm_model.pt",
                SRC / "train_protrek650m_gated_tabm.csv",
                fasta_root / "positive" / "XU_AMP.fasta",
                fasta_root / "negative" / "XU_nonAMP.fasta"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs missing: " + ", ".join(missing))
    if not args.analysis_only and args.model_root is None:
        parser.error("--model-root is required for a full inference run")

    set_seed(SEED)
    figures, tables, data = dirs()
    prediction_cols = ["protein_name", *[f"feature_{i}" for i in range(1024)], "label", "prob", "pred"]
    predictions = pd.read_csv(SRC / "explain_outputs" / "gated_tabm_independent_predictions.csv", usecols=prediction_cols)
    fasta = build_fasta_df(
        str(fasta_root / "positive" / "XU_AMP.fasta"),
        str(fasta_root / "negative" / "XU_nonAMP.fasta"),
    )
    predictions = predictions.merge(fasta[["protein_name", "sequence"]], on="protein_name", how="left", validate="one_to_one")
    if predictions["sequence"].isna().any():
        raise RuntimeError("Some prediction records could not be matched to FASTA sequences")
    predictions["length"] = predictions["sequence"].str.len()
    predictions = add_group(predictions)
    cohort = select_cohort(predictions)
    cohort.drop(columns=[f"feature_{i}" for i in range(1024)]).to_csv(data / "perturbation_selected_cohort.csv", index=False, encoding="utf-8-sig")
    cohort_summary = (
        cohort.groupby("group", as_index=False)
        .agg(n=("protein_name", "size"), mean_length=("length", "mean"), median_length=("length", "median"), mean_probability=("prob", "mean"))
    )
    cohort_summary.to_csv(tables / "perturbation_cohort_summary.csv", index=False, encoding="utf-8-sig")

    all_windows = build_all_window_records(cohort)
    all_windows.drop(columns=["sequence"]).to_csv(data / "perturbation_window_manifest.csv", index=False, encoding="utf-8-sig")
    original_embedding_map = {
        str(row.protein_name): row[[f"feature_{i}" for i in range(1024)]].to_numpy(dtype=np.float32)
        for _, row in cohort.iterrows()
    }

    checkpoint = SRC / "explain_outputs" / "gated_tabm_model.pt"
    classifier, scaler, feat_cols, config = train_or_load_gated_tabm(
        str(SRC / "train_protrek650m_gated_tabm.csv"), "label", str(checkpoint), seed=SEED, force_retrain=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device).eval()

    x_path = data / "perturbation_all_windows_X.csv"
    a_path = data / "perturbation_all_windows_alanine.csv"
    shuffle_path = data / "perturbation_selected_windows_shuffle.csv"

    if args.analysis_only:
        x_results = pd.read_csv(x_path)
        alanine_results = pd.read_csv(a_path)
        important, background = choose_windows(x_results)
        shuffle_results = pd.read_csv(shuffle_path)
    else:
        print("Loading ProTrek_650M on CPU, then moving only the sequence encoder to GPU...")
        client = load_protrek_650m_local(torch.device("cpu"), str(MODEL_ROOT), str(PROTREK_CODE))
        if hasattr(client, "text_encoder"):
            client.text_encoder = None
        if hasattr(client, "structure_encoder"):
            client.structure_encoder = None
        gc.collect()
        client.protein_encoder = client.protein_encoder.to(device).eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_results = run_embedding_records(
            all_windows,
            "X masking",
            x_path,
            client,
            classifier,
            scaler,
            feat_cols,
            original_embedding_map,
            device,
            args.outer_batch,
            args.inner_batch,
        )
        alanine_results = run_embedding_records(
            all_windows,
            "Alanine scan",
            a_path,
            client,
            classifier,
            scaler,
            feat_cols,
            original_embedding_map,
            device,
            args.outer_batch,
            args.inner_batch,
        )
        important, background = choose_windows(x_results)
        shuffle_records = build_shuffle_records(important, background, cohort)
        shuffle_results = run_embedding_records(
            shuffle_records,
            "Window shuffle",
            shuffle_path,
            client,
            classifier,
            scaler,
            feat_cols,
            original_embedding_map,
            device,
            args.outer_batch,
            args.inner_batch,
        )
        del client
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    important.to_csv(data / "important_nonoverlapping_windows.csv", index=False, encoding="utf-8-sig")
    background.to_csv(data / "background_matched_nonoverlapping_windows.csv", index=False, encoding="utf-8-sig")
    selected_results = selected_method_results(important, background, x_results, alanine_results, shuffle_results)
    selected_results.to_csv(data / "selected_windows_all_perturbations.csv", index=False, encoding="utf-8-sig")
    perturb_peptide, perturb_summary = summarize_perturbation(selected_results)
    perturb_peptide.to_csv(data / "peptide_level_perturbation_results.csv", index=False, encoding="utf-8-sig")
    perturb_summary.to_csv(tables / "perturbation_robustness_summary.csv", index=False, encoding="utf-8-sig")
    role_stats = perturbation_role_statistics(perturb_peptide, "group")
    role_stats.to_csv(tables / "perturbation_role_paired_statistics.csv", index=False, encoding="utf-8-sig")

    fp_flags = cohort[["protein_name", "high_confidence_fp"]].copy()
    fp_peptide = perturb_peptide.merge(fp_flags, on="protein_name", how="left")
    fp_peptide = fp_peptide[fp_peptide["group"] == "FP"].copy()
    fp_peptide["fp_subset"] = np.where(fp_peptide["high_confidence_fp"], "High-confidence FP (p>=0.90)", "Other FP")
    high_confidence_fp_stats = perturbation_role_statistics(fp_peptide, "fp_subset")
    high_confidence_fp_stats.to_csv(tables / "high_confidence_false_positive_statistics.csv", index=False, encoding="utf-8-sig")

    imp_selected = selected_results[selected_results["role"] == "important"]
    pivot = imp_selected.groupby(["protein_name", "start", "end", "method"])["delta_logit"].mean().unstack("method")
    correlations = []
    for method in ("Alanine scan", "Window shuffle"):
        common = pivot[["X masking", method]].dropna()
        rho, p = spearmanr(common["X masking"], common[method])
        correlations.append({"comparison": f"X masking vs {method}", "n_windows": len(common), "spearman_rho": rho, "p_value": p})
    pd.DataFrame(correlations).to_csv(tables / "perturbation_method_correlations.csv", index=False, encoding="utf-8-sig")

    curated = OUT / "results"
    curated.mkdir(parents=True, exist_ok=True)
    for file in [tables / "perturbation_robustness_summary.csv",
                 tables / "perturbation_cohort_summary.csv",
                 tables / "perturbation_role_paired_statistics.csv",
                 tables / "high_confidence_false_positive_statistics.csv",
                 tables / "perturbation_method_correlations.csv",
                 data / "peptide_level_perturbation_results.csv"]:
        shutil.copy2(file, curated / file.name)


    metadata = {
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "top_n_per_peptide": TOP_N,
        "top_window_overlap_rule": "Greedy selection by delta_logit; candidates with >50% overlap to an already selected window were excluded.",
        "background_rule": "Within-peptide random non-overlapping windows, excluding any overlap with selected high-contribution windows; fixed seed 42.",
        "sampling": {
            "target_per_group": N_PER_GROUP,
            "FP": "All false positives were included.",
            "TP_FN_TN": "Length- and probability-quantile stratified random samples were selected.",
        },
        "perturbations": {
            "X masking": "Replace all residues in each window with X.",
            "Alanine scan": "Replace all residues in each window with A.",
            "Window shuffle": f"Shuffle residues within selected important/background windows; {SHUFFLE_REPEATS} repeats per window.",
        },
        "importance": "Primary importance = original AMP logit - perturbed AMP logit; probability difference also retained.",
        "statistics": {
            "unit": "peptide",
            "paired_permutations": N_PERM,
            "cluster_bootstrap_replicates": N_BOOT,
            "multiple_testing": "Benjamini-Hochberg on retained paired perturbation comparisons.",
            "effect_direction": "important minus background",
        },
        "classifier_config": config,
        "n_selected": int(len(cohort)),
        "n_all_windows": int(len(all_windows)),
        "high_confidence_fp_threshold": 0.90,
        "n_high_confidence_fp": int(cohort["high_confidence_fp"].sum()),
    }
    with open(data / "perturbation_analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"cohort": cohort_summary.to_dict("records"), "metadata": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
