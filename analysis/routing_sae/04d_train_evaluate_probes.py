#!/usr/bin/env python3
"""Step 04D: train grouped-CV baseline, SAE, and combined logistic probes."""
from __future__ import annotations

import argparse
import json
import os
import os
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from _03_common import METADATA, N_EXPERTS, N_PRIMARY, TABLES, atomic_csv, atomic_json, require, script_record, sha256_file

ROOT = Path(os.environ.get("MOE_SAE_ANALYSIS_ROOT", Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1")).expanduser().resolve()
CACHE = ROOT / "cache"
SCRIPT = Path(__file__).resolve()
CSR_PATH = CACHE / "routing_sae_step04b_primary_csr.npz"
TOKEN_PATH = CACHE / "routing_sae_step04b_primary_tokens.parquet"
FOLD_PATH = TABLES / "routing_sae_step04c_sample_folds.csv"
SELECTION_PATH = TABLES / "routing_sae_step04c_selected_features.csv"
STEP04C_MANIFEST = METADATA / "routing_sae_step04c_manifest.json"

METRICS_PATH = TABLES / "routing_sae_step04d_expert_metrics.csv"
RECALL_PATH = TABLES / "routing_sae_step04d_recall_at4.csv"
COEFFICIENT_PATH = TABLES / "routing_sae_step04d_coefficients.csv"
PREDICTION_PATH = CACHE / "routing_sae_step04d_oof_predictions.npz"
MANIFEST_PATH = METADATA / "routing_sae_step04d_manifest.json"

MODELS = ("baseline", "sae", "combined")
N_FOLDS = 5
TOP_FEATURES = 256
C_VALUE = 1.0
MAX_ITER = 1_000
RANDOM_SEED = 42


def atomic_npz(destination: Path, **arrays: np.ndarray) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def make_model() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2", C=C_VALUE, solver="liblinear", fit_intercept=True,
        class_weight=None, max_iter=MAX_ITER, tol=1e-4,
        random_state=RANDOM_SEED, intercept_scaling=1.0,
    )


def fit_predict(model: LogisticRegression, x_train: sparse.csr_matrix, y_train: np.ndarray,
                x_test: sparse.csr_matrix) -> tuple[np.ndarray, float, int]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    prediction = model.predict_proba(x_test)[:, 1]
    return prediction, time.perf_counter() - started, convergence_warnings


def add_coefficients(rows: list[dict], fold: int, expert: int, model_name: str,
                     model: LogisticRegression, baseline_names: list[str], feature_ids: np.ndarray) -> None:
    coefficients = model.coef_[0]
    if model_name == "baseline":
        names = baseline_names
        ids = np.full(len(names), -1, dtype=np.int32)
        types = ["baseline"] * len(names)
    elif model_name == "sae":
        names = [f"sae_{feature_id}" for feature_id in feature_ids]
        ids = feature_ids
        types = ["sae"] * len(feature_ids)
    else:
        names = baseline_names + [f"sae_{feature_id}" for feature_id in feature_ids]
        ids = np.r_[np.full(len(baseline_names), -1, dtype=np.int32), feature_ids]
        types = ["baseline"] * len(baseline_names) + ["sae"] * len(feature_ids)
    require(len(coefficients) == len(names), "Coefficient/feature-name mismatch")
    for feature_type, name, feature_id, coefficient in zip(types, names, ids, coefficients):
        rows.append({
            "fold": fold, "expert": expert, "model": model_name,
            "feature_type": feature_type, "feature_name": name,
            "feature_id": int(feature_id), "coefficient": float(coefficient),
            "intercept": float(model.intercept_[0]),
        })


def metric_row(fold: int, expert: int, model_name: str, target: np.ndarray,
               prediction: np.ndarray, n_train: int, fit_seconds: float,
               n_iter: int, convergence_warnings: int) -> dict:
    return {
        "fold": fold, "expert": expert, "model": model_name,
        "n_train_tokens": n_train, "n_test_tokens": len(target),
        "test_positive_tokens": int(target.sum()), "test_prevalence": float(target.mean()),
        "auprc": float(average_precision_score(target, prediction)),
        "auroc": float(roc_auc_score(target, prediction)),
        "log_loss": float(log_loss(target, prediction, labels=[0, 1])),
        "fit_seconds": float(fit_seconds), "n_iter": n_iter,
        "convergence_warnings": convergence_warnings,
    }


def main(smoke: bool = False) -> None:
    for path in (CSR_PATH, TOKEN_PATH, FOLD_PATH, SELECTION_PATH, STEP04C_MANIFEST):
        require(path.is_file(), f"Missing step-04D input: {path}")
    prior = json.loads(STEP04C_MANIFEST.read_text(encoding="utf-8"))
    require(prior["status"] == "complete", "Step-04C is incomplete")
    for path in (FOLD_PATH, SELECTION_PATH):
        require(sha256_file(path) == prior["outputs"][str(path)]["sha256"], f"Step-04C hash mismatch: {path}")

    matrix_csr = sparse.load_npz(CSR_PATH)
    matrix_csc = matrix_csr.tocsc()
    tokens = pq.read_table(TOKEN_PATH).to_pandas()
    folds = pd.read_csv(FOLD_PATH)
    selections = pd.read_csv(SELECTION_PATH)
    require(matrix_csr.shape == (N_PRIMARY, 131_072), "Unexpected SAE matrix shape")
    require(len(tokens) == N_PRIMARY and len(folds) == 2_000, "Token/fold row count mismatch")
    require(len(selections) == N_FOLDS * N_EXPERTS * TOP_FEATURES, "Feature-selection count mismatch")

    fold_by_sample = np.full(2_000, -1, dtype=np.int8)
    fold_by_sample[folds.sample_index.to_numpy(np.int32)] = folds.fold.to_numpy(np.int8)
    token_fold = fold_by_sample[tokens.sample_index.to_numpy(np.int32)]
    experts = tokens[["expert_0", "expert_1", "expert_2", "expert_3"]].to_numpy(np.int16)
    target = np.zeros((N_PRIMARY, N_EXPERTS), dtype=bool)
    row_indices = np.arange(N_PRIMARY)[:, None]
    target[row_indices, experts] = True
    require(np.all(target.sum(axis=1) == 4), "Target matrix does not contain four experts per token")

    categorical = tokens[["domain", "token_class"]].astype(str)
    oof = {name: np.full((N_PRIMARY, N_EXPERTS), np.nan, dtype=np.float32) for name in MODELS}
    metric_rows: list[dict] = []
    recall_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    fold_range = range(1) if smoke else range(N_FOLDS)
    expert_range = range(1) if smoke else range(N_EXPERTS)

    for fold in fold_range:
        train_mask = token_fold != fold
        test_mask = token_fold == fold
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        baseline_train = encoder.fit_transform(categorical.loc[train_mask]).tocsr()
        baseline_test = encoder.transform(categorical.loc[test_mask]).tocsr()
        baseline_names = encoder.get_feature_names_out(["domain", "token_class"]).tolist()
        fold_predictions = {name: np.full((int(test_mask.sum()), N_EXPERTS), np.nan, dtype=np.float32) for name in MODELS}

        for expert in expert_range:
            chosen = selections[(selections.fold == fold) & (selections.expert == expert)].sort_values("rank")
            feature_ids = chosen.feature_id.to_numpy(np.int32)
            require(len(feature_ids) == TOP_FEATURES and len(np.unique(feature_ids)) == TOP_FEATURES, "Bad selected-feature set")
            selected_all = matrix_csc[:, feature_ids].tocsr()
            sae_train = selected_all[train_mask]
            sae_test = selected_all[test_mask]
            scaler = StandardScaler(with_mean=False, with_std=True, copy=True)
            sae_train = scaler.fit_transform(sae_train).tocsr()
            sae_test = scaler.transform(sae_test).tocsr()
            require(np.isfinite(sae_train.data).all() and np.isfinite(sae_test.data).all(), "Nonfinite scaled SAE value")
            combined_train = sparse.hstack([baseline_train, sae_train], format="csr", dtype=np.float32)
            combined_test = sparse.hstack([baseline_test, sae_test], format="csr", dtype=np.float32)
            y_train = target[train_mask, expert]
            y_test = target[test_mask, expert]

            designs = {
                "baseline": (baseline_train, baseline_test),
                "sae": (sae_train, sae_test),
                "combined": (combined_train, combined_test),
            }
            for model_name, (x_train, x_test) in designs.items():
                model = make_model()
                prediction, fit_seconds, convergence_warnings = fit_predict(model, x_train, y_train, x_test)
                prediction = prediction.astype(np.float32)
                require(np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all(), "Invalid prediction")
                fold_predictions[model_name][:, expert] = prediction
                oof[model_name][test_mask, expert] = prediction
                metric_rows.append(metric_row(
                    fold, expert, model_name, y_test, prediction, int(train_mask.sum()),
                    fit_seconds, int(model.n_iter_[0]), convergence_warnings,
                ))
                add_coefficients(coefficient_rows, fold, expert, model_name, model, baseline_names, feature_ids)

        if smoke:
            print(json.dumps({
                "status": "smoke_passed", "fold": 0, "expert": 0,
                "metrics": metric_rows, "baseline_features": len(baseline_names),
            }, ensure_ascii=False, indent=2))
            return

        true_fold = target[test_mask]
        for model_name in MODELS:
            require(np.isfinite(fold_predictions[model_name]).all(), f"Incomplete fold predictions: {model_name}")
            predicted_top4 = np.argpartition(-fold_predictions[model_name], kth=3, axis=1)[:, :4]
            overlap = np.take_along_axis(true_fold, predicted_top4, axis=1).sum(axis=1)
            recall_rows.append({
                "fold": fold, "model": model_name, "n_test_tokens": int(test_mask.sum()),
                "recall_at4": float(np.mean(overlap / 4.0)),
                "mean_correct_experts_at4": float(np.mean(overlap)),
            })

    metrics = pd.DataFrame(metric_rows)
    recall = pd.DataFrame(recall_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    require(len(metrics) == N_FOLDS * N_EXPERTS * len(MODELS), "Metric row count mismatch")
    require(len(recall) == N_FOLDS * len(MODELS), "Recall row count mismatch")
    require(metrics.isna().sum().sum() == 0 and recall.isna().sum().sum() == 0, "Missing metric value")
    require(coefficients.isna().sum().sum() == 0, "Missing coefficient value")
    require(sum(np.isnan(oof[name]).sum() for name in MODELS) == 0, "Incomplete OOF prediction matrix")
    require(int(metrics.convergence_warnings.sum()) == 0, "A probe emitted a convergence warning")
    require((metrics.n_iter < MAX_ITER).all(), "A probe reached max_iter")
    atomic_csv(metrics, METRICS_PATH)
    atomic_csv(recall, RECALL_PATH)
    atomic_csv(coefficients, COEFFICIENT_PATH)
    atomic_npz(
        PREDICTION_PATH,
        baseline=oof["baseline"], sae=oof["sae"], combined=oof["combined"],
        target=target, fold=token_fold, primary_row_index=np.arange(N_PRIMARY, dtype=np.int32),
    )

    summary = metrics.groupby("model", sort=False).agg(
        mean_auprc=("auprc", "mean"), mean_auroc=("auroc", "mean"),
        mean_log_loss=("log_loss", "mean"), total_fit_seconds=("fit_seconds", "sum"),
        max_n_iter=("n_iter", "max"),
    ).reset_index()
    recall_summary = recall.groupby("model", sort=False).recall_at4.mean().to_dict()
    outputs = [METRICS_PATH, RECALL_PATH, COEFFICIENT_PATH, PREDICTION_PATH]
    manifest = {
        "status": "complete", "step": "04D_probe_training_and_evaluation",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "models": list(MODELS), "target": "32 binary layer-11 expert-selected labels",
            "estimator": "LogisticRegression", "penalty": "L2", "C": C_VALUE,
            "solver": "liblinear", "class_weight": None, "max_iter": MAX_ITER,
            "sae_scaling": "StandardScaler(with_mean=False), fit on training fold only",
            "categorical_encoding": "OneHotEncoder(domain + token_class), fit on training fold only",
            "feature_selection": "fold/expert-specific Top-256 from step 04C",
            "primary_metric": "AUPRC", "secondary_metrics": ["AUROC", "log_loss", "Recall@4"],
        },
        "counts": {
            "fits": len(metrics), "expert_metric_rows": len(metrics), "recall_rows": len(recall),
            "coefficient_rows": len(coefficients), "oof_prediction_shape_per_model": [N_PRIMARY, N_EXPERTS],
        },
        "performance_unweighted_fold_expert_mean": summary.to_dict(orient="records"),
        "recall_at4_unweighted_fold_mean": {key: float(value) for key, value in recall_summary.items()},
        "validation": {
            "convergence_warnings": int(metrics.convergence_warnings.sum()),
            "max_observed_iterations": int(metrics.n_iter.max()), "missing_values": 0,
            "nonfinite_predictions": 0, "oof_coverage_per_model": N_PRIMARY * N_EXPERTS,
        },
        "inputs": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in [CSR_PATH, TOKEN_PATH, FOLD_PATH, SELECTION_PATH, STEP04C_MANIFEST]
        },
        "outputs": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs
        },
        "script": script_record(SCRIPT),
        "gpt_model_loaded": False, "sae_model_loaded": False,
        "gpt_or_sae_training_performed": False, "model_inference_rerun": False,
        "probe_training_performed": True, "dense_sae_matrix_created": False,
    }
    atomic_json(manifest, MANIFEST_PATH)
    print(json.dumps({
        "status": "complete", "counts": manifest["counts"],
        "performance": manifest["performance_unweighted_fold_expert_mean"],
        "recall_at4": manifest["recall_at4_unweighted_fold_mean"],
        "validation": manifest["validation"], "outputs": manifest["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Fit fold 0 / expert 0 only and write no outputs")
    args = parser.parse_args()
    main(smoke=args.smoke)
