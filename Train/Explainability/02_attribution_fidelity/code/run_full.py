from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SRC = Path()
MODEL_ROOT = Path()
PROTREK_CODE = Path()
CACHE = Path()
OUT = PACKAGE_ROOT / "02_attribution_fidelity" / "scratch_full_run"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data"
sys.path.insert(0, str(PACKAGE_ROOT / "common"))

from explainability_pipeline_utils import (  # noqa: E402
    GatedTabM,
    build_fasta_df,
    extract_intermediate_outputs,
    load_protrek_650m_local,
    normalize_aa,
    predict_proba,
    prepare_test_data,
    train_or_load_gated_tabm,
)


SEED = 42
RANDOM_MODEL_SEEDS = [17, 73, 211, 907, 2027]
RANDOM_REPEATS = 20
GROUP_ORDER = ["TP", "FN", "FP", "TN"]


def setup() -> None:
    for path in (FIG, TAB, DATA):
        path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42}
    )


def publish_results() -> None:
    curated = OUT / "results"
    curated.mkdir(parents=True, exist_ok=True)
    for file in [TAB / "aopc_cumulative_alanine_statistics.csv",
                 TAB / "randomized_model_sanity_summary.csv",
                 DATA / "aopc_peptide_level.csv",
                 DATA / "randomized_model_sanity_peptide_level.csv",
                 CACHE / "data" / "perturbation_selected_cohort.csv"]:
        if file.is_file():
            shutil.copy2(file, curated / file.name)


def stable_logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int, threshold: float = 0.5) -> bool:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    return inter / max(1, min(a_end - a_start, b_end - b_start)) > threshold


def choose_nonoverlap(part: pd.DataFrame, score_col: str, n: int = 3) -> list:
    chosen = []
    for row in part.sort_values(score_col, ascending=False).itertuples(index=False):
        if all(not overlaps(row.start, row.end, other.start, other.end) for other in chosen):
            chosen.append(row)
        if len(chosen) == n:
            break
    return chosen


def choose_random_nonoverlap(part: pd.DataFrame, selected: list, n: int, rng: np.random.Generator) -> list:
    chosen = []
    order = rng.permutation(len(part))
    records = list(part.itertuples(index=False))
    for index in order:
        row = records[index]
        if any(overlaps(row.start, row.end, other.start, other.end) for other in selected):
            continue
        if any(overlaps(row.start, row.end, other.start, other.end) for other in chosen):
            continue
        chosen.append(row)
        if len(chosen) == n:
            break
    return chosen


def alanine_mutant(sequence: str, windows: list) -> str:
    chars = list(sequence)
    for window in windows:
        for index in range(int(window.start), int(window.end)):
            chars[index] = "A"
    return "".join(chars)


def prepare_aopc_records() -> pd.DataFrame:
    cohort = pd.read_csv(CACHE / "data" / "perturbation_selected_cohort.csv")
    alanine = pd.read_csv(CACHE / "data" / "perturbation_all_windows_alanine.csv")
    alanine["class_evidence"] = np.where(alanine["pred"].to_numpy(int) == 1, alanine["delta_logit"], -alanine["delta_logit"])
    seq_map = cohort.set_index("protein_name")["sequence"].astype(str).to_dict()
    rng = np.random.default_rng(SEED + 912)
    rows = []
    for protein, part in alanine.groupby("protein_name", sort=True):
        selected = choose_nonoverlap(part, "class_evidence", n=3)
        sequence = seq_map[protein]
        if not selected:
            continue
        meta = part.iloc[0]
        for step in range(1, len(selected) + 1):
            windows = selected[:step]
            rows.append(
                {
                    "record_id": f"{protein}|selected|0|{step}",
                    "protein_name": protein,
                    "group": meta.group,
                    "label": int(meta.label),
                    "pred": int(meta.pred),
                    "base_prob": float(meta.base_prob),
                    "selection": "Attribution-ranked",
                    "repeat": 0,
                    "step": step,
                    "n_ranked_windows": len(selected),
                    "windows": ";".join(f"{w.start + 1}-{w.end}" for w in windows),
                    "sequence": alanine_mutant(sequence, windows),
                }
            )
        for repeat in range(RANDOM_REPEATS):
            random_windows = choose_random_nonoverlap(part, selected, len(selected), rng)
            if len(random_windows) < len(selected):
                continue
            for step in range(1, len(random_windows) + 1):
                windows = random_windows[:step]
                rows.append(
                    {
                        "record_id": f"{protein}|random|{repeat}|{step}",
                        "protein_name": protein,
                        "group": meta.group,
                        "label": int(meta.label),
                        "pred": int(meta.pred),
                        "base_prob": float(meta.base_prob),
                        "selection": "Random non-overlapping",
                        "repeat": repeat,
                        "step": step,
                        "n_ranked_windows": len(selected),
                        "windows": ";".join(f"{w.start + 1}-{w.end}" for w in windows),
                        "sequence": alanine_mutant(sequence, windows),
                    }
                )
    records = pd.DataFrame(rows)
    records.to_csv(DATA / "aopc_cumulative_mutant_manifest.csv", index=False, encoding="utf-8-sig")
    return records


def make_random_models(in_dim: int, config: dict) -> list[tuple[int, GatedTabM]]:
    models = []
    for seed in RANDOM_MODEL_SEEDS:
        torch.manual_seed(seed)
        model = GatedTabM(in_dim=in_dim, config=config, seed=seed).cpu().eval()
        models.append((seed, model))
    return models


@torch.no_grad()
def infer_embeddings(
    records: pd.DataFrame,
    output_csv: Path,
    client,
    classifier,
    random_models,
    scaler,
    outer_batch: int = 64,
    inner_batch: int = 8,
) -> pd.DataFrame:
    completed = 0
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        completed = len(existing)
        if completed and existing["record_id"].tolist() != records.iloc[:completed]["record_id"].tolist():
            raise RuntimeError(f"Resume file does not match manifest: {output_csv}")
        print(f"[resume] {output_csv.name}: {completed}/{len(records)}", flush=True)
    else:
        existing = pd.DataFrame()

    classifier = classifier.cpu().eval()
    for start in range(completed, len(records), outer_batch):
        batch = records.iloc[start : start + outer_batch].copy()
        sequences = [normalize_aa(value, 1024) for value in batch["sequence"].astype(str)]
        embedding = client.get_protein_repr(sequences, batch_size=inner_batch, verbose=False)
        embedding = embedding.float().detach().cpu().numpy().astype(np.float32)
        scaled = scaler.transform(embedding).astype(np.float32)
        batch["trained_prob"] = predict_proba(classifier, scaled, torch.device("cpu"), batch_size=512)
        for seed, model in random_models:
            batch[f"random_model_{seed}_prob"] = predict_proba(model, scaled, torch.device("cpu"), batch_size=512)
        batch.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False, encoding="utf-8-sig")
        print(f"[infer] {output_csv.name}: {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    return pd.read_csv(output_csv)


def base_random_probabilities(test_features: pd.DataFrame, cohort: pd.DataFrame, feature_cols, scaler, random_models) -> pd.DataFrame:
    feature_map = test_features.set_index("protein_name")
    x = feature_map.loc[cohort["protein_name"], feature_cols].to_numpy(np.float32)
    x = scaler.transform(x).astype(np.float32)
    output = cohort[["protein_name", "group", "label", "pred", "prob"]].rename(columns={"prob": "trained_base_prob"}).copy()
    for seed, model in random_models:
        output[f"random_model_{seed}_base_prob"] = predict_proba(model, x, torch.device("cpu"), batch_size=512)
    return output


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(3000, dtype=float)
    for index in range(len(means)):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_permutation_p(values: np.ndarray, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    permuted = np.empty(20000, dtype=float)
    for index in range(len(permuted)):
        permuted[index] = abs(np.mean(values * rng.choice([-1, 1], len(values))))
    return float((1 + np.sum(permuted >= observed)) / (len(permuted) + 1))


def analyse_aopc(scored: pd.DataFrame, base: pd.DataFrame) -> dict:
    data = scored.merge(base, on=["protein_name", "group", "label", "pred"], how="left", validate="many_to_one")
    trained_direction = np.where(data["pred"].to_numpy(int) == 1, 1.0, -1.0)
    data["trained_evidence_removed"] = trained_direction * (stable_logit(data["trained_base_prob"]) - stable_logit(data["trained_prob"]))
    for seed in RANDOM_MODEL_SEEDS:
        base_col = f"random_model_{seed}_base_prob"
        prob_col = f"random_model_{seed}_prob"
        direction = np.where(data[base_col].to_numpy(float) >= 0.5, 1.0, -1.0)
        data[f"random_model_{seed}_evidence_removed"] = direction * (stable_logit(data[base_col]) - stable_logit(data[prob_col]))
    data.to_csv(DATA / "aopc_cumulative_scored_with_controls.csv", index=False, encoding="utf-8-sig")

    selected = data[data["selection"] == "Attribution-ranked"].copy()
    random_part = data[data["selection"] == "Random non-overlapping"].copy()
    random_mean = random_part.groupby(["protein_name", "group", "step"], as_index=False)["trained_evidence_removed"].mean().rename(columns={"trained_evidence_removed": "random_mean"})
    peptide = selected.merge(random_mean, on=["protein_name", "group", "step"], how="inner")
    peptide["selected_minus_random"] = peptide["trained_evidence_removed"] - peptide["random_mean"]
    peptide.to_csv(DATA / "aopc_peptide_level.csv", index=False, encoding="utf-8-sig")

    stats_rows = []
    for step in sorted(peptide["step"].unique()):
        for group in GROUP_ORDER + ["All"]:
            part = peptide[peptide["step"] == step]
            if group != "All":
                part = part[part["group"] == group]
            values = part["selected_minus_random"].to_numpy(float)
            low, high = bootstrap_mean_ci(values, SEED + step * 20 + len(stats_rows))
            stats_rows.append(
                {
                    "step": int(step),
                    "group": group,
                    "n_peptides": len(values),
                    "selected_mean_evidence_removed": float(part["trained_evidence_removed"].mean()),
                    "random_mean_evidence_removed": float(part["random_mean"].mean()),
                    "selected_minus_random": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "paired_sign_permutation_p": sign_permutation_p(values, SEED + 900 + len(stats_rows)),
                }
            )
    stats = pd.DataFrame(stats_rows)
    pvalues = stats["paired_sign_permutation_p"].to_numpy(float)
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    qvalues = np.empty_like(adjusted)
    qvalues[order] = np.minimum(adjusted, 1.0)
    stats["fdr_q"] = qvalues
    stats.to_csv(TAB / "aopc_cumulative_alanine_statistics.csv", index=False, encoding="utf-8-sig")

    randomization_rows = []
    max_step = int(peptide["step"].max())
    trained_max = peptide[peptide["step"] == max_step]
    randomization_rows.extend(
        {"model": "Trained model", "seed": SEED, "protein_name": row.protein_name, "top_minus_random": row.selected_minus_random}
        for row in trained_max.itertuples(index=False)
    )
    for seed in RANDOM_MODEL_SEEDS:
        evidence_col = f"random_model_{seed}_evidence_removed"
        sel = selected[selected["step"] == max_step][["protein_name", evidence_col]]
        rnd = (
            random_part[random_part["step"] == max_step]
            .groupby("protein_name", as_index=False)[evidence_col]
            .mean()
            .rename(columns={evidence_col: "random_mean"})
        )
        merged = sel.merge(rnd, on="protein_name", how="inner")
        for row in merged.itertuples(index=False):
            randomization_rows.append(
                {"model": "Randomized model", "seed": seed, "protein_name": row.protein_name, "top_minus_random": getattr(row, evidence_col) - row.random_mean}
            )
    randomization = pd.DataFrame(randomization_rows)
    randomization.to_csv(DATA / "randomized_model_sanity_peptide_level.csv", index=False, encoding="utf-8-sig")
    random_summary = randomization.groupby(["model", "seed"], as_index=False)["top_minus_random"].agg(["mean", "median", "std", "count"]).reset_index()
    random_summary.to_csv(TAB / "randomized_model_sanity_summary.csv", index=False, encoding="utf-8-sig")

    return {"statistics": stats.to_dict("records"), "randomized_summary": random_summary.to_dict("records")}


def main() -> None:
    global SRC, MODEL_ROOT, PROTREK_CODE, CACHE, OUT, FIG, TAB, DATA
    parser = argparse.ArgumentParser(description="Complete AOPC and parameter-randomization computation for manuscript module 2")
    parser.add_argument("--source-root", type=Path,
                        help="Train directory containing feature CSV files and explain_outputs/")
    parser.add_argument("--model-root", type=Path,
                        help="Local ProTrek_650M model directory")
    parser.add_argument("--perturbation-root", type=Path, required=True,
                        help="Module 3 full-run output containing data/perturbation_selected_cohort.csv and data/perturbation_all_windows_alanine.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protrek-code-root", type=Path)
    parser.add_argument("--analysis-only", action="store_true",
                        help="Recompute statistics from already scored mutants and base probabilities")
    parser.add_argument("--scored-csv", type=Path)
    parser.add_argument("--base-csv", type=Path)
    args = parser.parse_args()
    SRC = args.source_root.resolve() if args.source_root else Path()
    MODEL_ROOT = args.model_root.resolve() if args.model_root else Path()
    PROTREK_CODE = (args.protrek_code_root or SRC / "ProTrek-main").resolve()
    CACHE = args.perturbation_root.resolve()
    OUT = args.output_dir.resolve()
    FIG, TAB, DATA = OUT / "figures", OUT / "tables", OUT / "data"
    if args.analysis_only:
        if args.scored_csv is None or args.base_csv is None:
            parser.error("--analysis-only requires --scored-csv and --base-csv")
        setup()
        scored = pd.read_csv(args.scored_csv)
        base = pd.read_csv(args.base_csv)
        summary = analyse_aopc(scored, base)
        publish_results()
        (DATA / "aopc_analysis_only_summary.json").write_text(
            json.dumps({"AOPC_and_randomization": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("AOPC statistics recomputed from saved inference records", flush=True)
        return
    if args.source_root is None:
        parser.error("--source-root is required for a full inference run")
    if args.model_root is None:
        parser.error("--model-root is required for a full inference run")
    required = [SRC / "train_protrek650m_gated_tabm.csv", SRC / "explain_outputs" / "gated_tabm_model.pt",
                CACHE / "data" / "perturbation_selected_cohort.csv", CACHE / "data" / "perturbation_all_windows_alanine.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs missing: " + ", ".join(missing))
    setup()
    checkpoint = SRC / "explain_outputs" / "gated_tabm_model.pt"
    classifier, scaler, feature_cols, config = train_or_load_gated_tabm(
        str(SRC / "train_protrek650m_gated_tabm.csv"), "label", str(checkpoint), seed=SEED, force_retrain=False
    )
    classifier = classifier.cpu().eval()
    test_features, x_test, y_test = prepare_test_data(
        str(SRC / "positive_protrek650m_test.csv"),
        str(SRC / "negative_protrek650m_test.csv"),
        feature_cols,
        scaler,
    )
    random_models = make_random_models(len(feature_cols), config)
    cohort = pd.read_csv(CACHE / "data" / "perturbation_selected_cohort.csv")
    base = base_random_probabilities(test_features, cohort, feature_cols, scaler, random_models)
    base.to_csv(DATA / "cohort_base_probabilities_trained_and_randomized.csv", index=False, encoding="utf-8-sig")

    print("[load] ProTrek_650M on CPU, then move protein encoder to GPU if available", flush=True)
    client = load_protrek_650m_local(torch.device("cpu"), str(MODEL_ROOT), str(PROTREK_CODE))
    if torch.cuda.is_available():
        try:
            client.protein_encoder = client.protein_encoder.to(torch.device("cuda"))
            print("[device] protein encoder moved to CUDA", flush=True)
        except RuntimeError as exc:
            print(f"[device] CUDA placement failed; continue on CPU: {exc}", flush=True)
            client.protein_encoder = client.protein_encoder.to(torch.device("cpu"))

    aopc_records = prepare_aopc_records()
    aopc_scored = infer_embeddings(
        aopc_records,
        DATA / "aopc_cumulative_mutants_scored.csv",
        client,
        classifier,
        random_models,
        scaler,
    )
    aopc_summary = analyse_aopc(aopc_scored, base)
    publish_results()

    summary = {"AOPC_and_randomization": aopc_summary}
    with open(DATA / "heavy_experiments_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
