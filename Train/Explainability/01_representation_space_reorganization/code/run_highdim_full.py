"""Recompute the manuscript's 1,024-D representation metrics from features and checkpoint.

This is the *complete numeric experiment* behind Fig. 6. The 5-NN model is
fitted on training embeddings and evaluated on independent-test embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from explainability_pipeline_utils import (  # noqa: E402
    extract_intermediate_outputs,
    prepare_test_data,
    set_seed,
    train_or_load_gated_tabm,
)

SEED = 42
BOOTSTRAP_REPLICATES = 2000


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def centroid_margin(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    xn = normalize_rows(x.astype(np.float64, copy=False))
    centroids = []
    for label in (0, 1):
        centroid = xn[y == label].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
    c0, c1 = centroids
    d0 = 1.0 - xn @ c0
    d1 = 1.0 - xn @ c1
    own = np.where(y == 0, d0, d1)
    other = np.where(y == 0, d1, d0)
    margin = other - own
    ratio = (1.0 - float(c0 @ c1)) / max(float(np.mean(own)), 1e-12)
    return margin, ratio


def bootstrap_mean_ci(values: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    n = len(values)
    for i in range(BOOTSTRAP_REPLICATES):
        means[i] = values[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Train directory containing feature CSV files and explain_outputs/")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = [source / "train_protrek650m_gated_tabm.csv",
                source / "positive_protrek650m_test.csv", source / "negative_protrek650m_test.csv",
                source / "explain_outputs" / "gated_tabm_model.pt"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs missing: " + ", ".join(missing))

    set_seed(SEED)
    model, scaler, features, config = train_or_load_gated_tabm(
        str(required[0]), "label", str(required[3]), seed=SEED, force_retrain=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    train = pd.read_csv(required[0], usecols=[*features, "label"])
    x_train = scaler.transform(train[features].to_numpy()).astype(np.float32)
    y_train = train.label.astype(int).to_numpy()
    test, x_test, y_test = prepare_test_data(str(required[1]), str(required[2]),
                                             features, scaler, label_col="label")
    _, adapted_train, _ = extract_intermediate_outputs(model, x_train, device=device, batch_size=2048)
    _, adapted_test, _ = extract_intermediate_outputs(model, x_test, device=device, batch_size=2048)
    assert len(x_test) == len(adapted_test) == 3072
    assert np.bincount(y_test, minlength=2).tolist() == [1536, 1536]

    rows = []
    per_sample = pd.DataFrame({"protein_name": test.protein_name.astype(str), "label": y_test})
    for representation, xtr, xte in (("Raw ProTrek", x_train, x_test),
                                      ("Gated Adapter", adapted_train, adapted_test)):
        silhouette = silhouette_samples(xte, y_test, metric="cosine")
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute",
                                   weights="uniform", n_jobs=1)
        knn.fit(xtr, y_train)
        correct = (knn.predict(xte) == y_test).astype(float)
        margin, _ = centroid_margin(xte, y_test)
        for metric, values in (("silhouette", silhouette), ("knn_accuracy", correct),
                               ("centroid_margin", margin)):
            low, high = bootstrap_mean_ci(values)
            rows.append({"representation": representation, "metric": metric,
                         "estimate": float(np.mean(values)), "ci_low": low, "ci_high": high,
                         "n_test": len(values)})
            per_sample[f"{representation}_{metric}"] = values

    differences = []
    for metric in ("silhouette", "knn_accuracy", "centroid_margin"):
        delta = (per_sample[f"Gated Adapter_{metric}"].to_numpy()
                 - per_sample[f"Raw ProTrek_{metric}"].to_numpy())
        low, high = bootstrap_mean_ci(delta, seed=SEED + 11)
        differences.append({"metric": metric, "adapted_minus_raw": float(delta.mean()),
                            "ci_low": low, "ci_high": high, "n_test": len(delta)})
    pd.DataFrame(rows).to_csv(output / "representation_metric_estimates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(differences).to_csv(output / "representation_paired_differences.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(output / "representation_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "method.json").write_text(json.dumps({"seed": SEED,
        "metric": "cosine", "n_bootstrap": BOOTSTRAP_REPLICATES,
        "knn_fit": "training set", "knn_evaluate": "independent test set",
        "model_config": config}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
