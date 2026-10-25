#!/usr/bin/env python3
"""Step 04E-2: evaluate frozen primary-trained probes on anomalous first tokens."""
from __future__ import annotations

import json
import os
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from _03_common import METADATA, TABLES, atomic_csv, atomic_json, require, script_record, sha256_file

ROOT = Path(os.environ.get("MOE_SAE_ANALYSIS_ROOT", Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1")).expanduser().resolve()
CACHE = ROOT / "cache"
SCRIPT = Path(__file__).resolve()
TOKEN_TABLE_PATH = CACHE / "token_table.parquet"
ANNOTATION_PATH = CACHE / "token_annotations.parquet"
SAE_PATH = CACHE / "sae_layer11_sparse.parquet"
ROUTING_PATH = CACHE / "routing_top4.npz"
PRIMARY_CSR_PATH = CACHE / "routing_sae_step04b_primary_csr.npz"
PRIMARY_TOKEN_PATH = CACHE / "routing_sae_step04b_primary_tokens.parquet"
PRIMARY_PREDICTION_PATH = CACHE / "routing_sae_step04d_oof_predictions.npz"
FOLD_PATH = TABLES / "routing_sae_step04c_sample_folds.csv"
SELECTION_PATH = TABLES / "routing_sae_step04c_selected_features.csv"
COEFFICIENT_PATH = TABLES / "routing_sae_step04d_coefficients.csv"
INFERENCE_MANIFEST = METADATA / "routing_sae_step04e_inference_manifest.json"

FIRST_PREDICTION_PATH = CACHE / "routing_sae_step04e_first_token_predictions.npz"
METRICS_PATH = TABLES / "routing_sae_step04e_all_token_audit_metrics.csv"
RECALL_PATH = TABLES / "routing_sae_step04e_all_token_audit_recall.csv"
MANIFEST_PATH = METADATA / "routing_sae_step04e_all_token_manifest.json"
MODELS = ("baseline", "sae", "combined")
N_TOKENS = 382_685
N_PRIMARY = 380_685
N_FIRST = 2_000
DICT_SIZE = 131_072


def atomic_npz(destination: Path, **arrays: np.ndarray) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent)
    os.close(fd); temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_first_token_union_matrix(token_table: pd.DataFrame, first_mask: np.ndarray,
                                   first_row_by_global: np.ndarray, union_features: np.ndarray) -> sparse.csr_matrix:
    counts = token_table.active_feature_count.to_numpy(np.int64)
    row_ends = np.cumsum(counts, dtype=np.int64)
    lookup = np.full(DICT_SIZE, -1, dtype=np.int32)
    lookup[union_features] = np.arange(len(union_features), dtype=np.int32)
    row_parts: list[np.ndarray] = []; column_parts: list[np.ndarray] = []; data_parts: list[np.ndarray] = []
    offset = 0
    parquet_file = pq.ParquetFile(SAE_PATH)
    for batch in parquet_file.iter_batches(batch_size=1_000_000, columns=["feature_id", "activation"]):
        feature = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        activation = batch.column(1).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        positions = np.arange(offset, offset + len(feature), dtype=np.int64)
        token_index = np.searchsorted(row_ends, positions, side="right")
        union_column = lookup[feature]
        keep = first_mask[token_index] & (union_column >= 0)
        if keep.any():
            row_parts.append(first_row_by_global[token_index[keep]])
            column_parts.append(union_column[keep])
            data_parts.append(activation[keep])
        offset += len(feature)
    require(offset == int(counts.sum()) == 72_151_749, "SAE stream row count mismatch")
    rows = np.concatenate(row_parts); columns = np.concatenate(column_parts); data = np.concatenate(data_parts)
    matrix = sparse.coo_matrix((data, (rows, columns)), shape=(N_FIRST, len(union_features))).tocsr()
    require(matrix.nnz == len(data), "Duplicate first-token selected feature")
    require(np.isfinite(matrix.data).all() and np.count_nonzero(matrix.data == 0) == 0, "Invalid first-token activation")
    return matrix


def baseline_logit(coefficient_group: pd.DataFrame, intercept: float, domain: str, token_class: str) -> float:
    mapping = dict(zip(coefficient_group.feature_name.astype(str), coefficient_group.coefficient.astype(float)))
    return intercept + mapping.get(f"domain_{domain}", 0.0) + mapping.get(f"token_class_{token_class}", 0.0)


def recall_at4(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    top4 = np.argpartition(-prediction, kth=3, axis=1)[:, :4]
    overlap = np.take_along_axis(target, top4, axis=1).sum(axis=1)
    return float(np.mean(overlap / 4.0)), float(np.mean(overlap))


def main() -> None:
    inputs = [TOKEN_TABLE_PATH, ANNOTATION_PATH, SAE_PATH, ROUTING_PATH, PRIMARY_CSR_PATH, PRIMARY_TOKEN_PATH,
              PRIMARY_PREDICTION_PATH, FOLD_PATH, SELECTION_PATH, COEFFICIENT_PATH, INFERENCE_MANIFEST]
    for path in inputs:
        require(path.is_file(), f"Missing all-token audit input: {path}")
    inference = json.loads(INFERENCE_MANIFEST.read_text(encoding="utf-8"))
    require(inference["status"] == "complete", "04E inference incomplete")

    token_table = pq.read_table(TOKEN_TABLE_PATH, columns=["sample_id", "domain", "token_position", "active_feature_count", "is_first_token"]).to_pandas()
    annotations = pq.read_table(ANNOTATION_PATH, columns=["sample_id", "token_position", "token_class", "is_first_token"]).to_pandas()
    primary_tokens = pq.read_table(PRIMARY_TOKEN_PATH).to_pandas()
    selections = pd.read_csv(SELECTION_PATH)
    coefficients = pd.read_csv(COEFFICIENT_PATH)
    folds = pd.read_csv(FOLD_PATH)
    primary_matrix = sparse.load_npz(PRIMARY_CSR_PATH)
    first_mask = token_table.is_first_token.to_numpy(bool)
    require(len(token_table) == N_TOKENS and int(first_mask.sum()) == N_FIRST, "First-token count mismatch")
    require(np.array_equal(first_mask, annotations.is_first_token.to_numpy(bool)), "First-token annotation mismatch")

    sample_map = primary_tokens[["sample_id", "sample_index"]].drop_duplicates().set_index("sample_id").sample_index
    first_global = np.flatnonzero(first_mask)
    first_sample_index = token_table.loc[first_mask, "sample_id"].map(sample_map).to_numpy(np.int32)
    require(len(np.unique(first_sample_index)) == N_FIRST and set(first_sample_index) == set(range(N_FIRST)), "First-token/sample mapping mismatch")
    first_row_by_global = np.full(N_TOKENS, -1, dtype=np.int32)
    first_row_by_global[first_global] = first_sample_index
    union_features = np.sort(selections.feature_id.unique().astype(np.int32))
    first_union = build_first_token_union_matrix(token_table, first_mask, first_row_by_global, union_features)
    union_lookup = {int(feature): index for index, feature in enumerate(union_features)}

    fold_by_sample = np.full(N_FIRST, -1, dtype=np.int8)
    fold_by_sample[folds.sample_index.to_numpy(np.int32)] = folds.fold.to_numpy(np.int8)
    primary_sample = primary_tokens.sample_index.to_numpy(np.int32)
    primary_fold = fold_by_sample[primary_sample]
    first_domain = token_table.loc[first_mask, "domain"].astype(str).to_numpy()[np.argsort(first_sample_index)]
    first_class = annotations.loc[first_mask, "token_class"].astype(str).to_numpy()[np.argsort(first_sample_index)]

    with np.load(PRIMARY_PREDICTION_PATH, allow_pickle=False) as source:
        primary_prediction = {name: source[name].astype(np.float64) for name in MODELS}
        primary_target = source["target"].astype(bool)
    with np.load(ROUTING_PATH, allow_pickle=False) as source:
        first_experts_unsorted = source["expert_id"][first_global, 11, :].astype(np.int16)
    order = np.argsort(first_sample_index)
    first_experts = first_experts_unsorted[order]
    first_target = np.zeros((N_FIRST, 32), dtype=bool)
    first_target[np.arange(N_FIRST)[:, None], first_experts] = True
    require(np.all(first_target.sum(axis=1) == 4), "First-token target mismatch")
    unique_first_routes = np.unique(np.sort(first_experts, axis=1), axis=0)

    first_prediction = {name: np.full((N_FIRST, 32), np.nan, dtype=np.float32) for name in MODELS}
    primary_csc = primary_matrix.tocsc()
    for fold in range(5):
        train_mask = primary_fold != fold
        test_samples = np.flatnonzero(fold_by_sample == fold)
        for expert in range(32):
            chosen = selections[(selections.fold == fold) & (selections.expert == expert)].sort_values("rank")
            feature_ids = chosen.feature_id.to_numpy(np.int32)
            union_columns = np.array([union_lookup[int(feature)] for feature in feature_ids], dtype=np.int32)
            train_selected = primary_csc[train_mask][:, feature_ids].tocsr()
            scaler = StandardScaler(with_mean=False, with_std=True).fit(train_selected)
            first_selected = first_union[test_samples][:, union_columns].multiply(1.0 / scaler.scale_).tocsr()
            for model_name in MODELS:
                group = coefficients[(coefficients.fold == fold) & (coefficients.expert == expert) & coefficients.model.eq(model_name)]
                require(len(group) > 0 and group.intercept.nunique() == 1, "Coefficient group mismatch")
                intercept = float(group.intercept.iloc[0])
                logit = np.full(len(test_samples), intercept, dtype=np.float64)
                if model_name in ("baseline", "combined"):
                    baseline_group = group[group.feature_type.eq("baseline")]
                    logit += np.array([
                        baseline_logit(baseline_group, 0.0, first_domain[sample], first_class[sample])
                        for sample in test_samples
                    ])
                if model_name in ("sae", "combined"):
                    sae_group = group[group.feature_type.eq("sae")].set_index("feature_id")
                    sae_coef = sae_group.loc[feature_ids, "coefficient"].to_numpy(np.float64)
                    logit += np.asarray(first_selected @ sae_coef).ravel()
                first_prediction[model_name][test_samples, expert] = expit(logit).astype(np.float32)

    require(all(np.isfinite(first_prediction[name]).all() for name in MODELS), "Incomplete first-token predictions")
    metric_rows = []; recall_rows = []
    populations = {
        "primary_nonfirst": (primary_target, primary_prediction),
        "first_only_audit": (first_target, {name: first_prediction[name].astype(np.float64) for name in MODELS}),
        "all_tokens_audit": (
            np.vstack([primary_target, first_target]),
            {name: np.vstack([primary_prediction[name], first_prediction[name]]) for name in MODELS},
        ),
    }
    for population, (truth, prediction_set) in populations.items():
        for model_name in MODELS:
            prediction = prediction_set[model_name]
            for expert in range(32):
                y = truth[:, expert]; p = prediction[:, expert]
                positives = int(y.sum())
                auprc_defined = positives > 0
                auroc_defined = 0 < positives < len(y)
                metric_rows.append({
                    "population": population, "model": model_name, "expert": expert,
                    "n_tokens": len(y), "positive_tokens": positives, "prevalence": float(y.mean()),
                    "auprc_defined": auprc_defined,
                    "auprc": float(average_precision_score(y, p)) if auprc_defined else np.nan,
                    "auroc_defined": auroc_defined,
                    "auroc": float(roc_auc_score(y, p)) if auroc_defined else np.nan,
                    "log_loss": float(log_loss(y, p, labels=[0, 1])),
                })
            recall, correct = recall_at4(truth, prediction)
            recall_rows.append({"population": population, "model": model_name, "n_tokens": len(truth), "recall_at4": recall, "mean_correct_experts_at4": correct})
    metrics = pd.DataFrame(metric_rows); recall = pd.DataFrame(recall_rows)
    require(len(metrics) == 288 and len(recall) == 9, "All-token audit row count mismatch")
    expected_auprc_na = ~metrics.auprc_defined
    expected_auroc_na = ~metrics.auroc_defined
    require(np.array_equal(metrics.auprc.isna().to_numpy(), expected_auprc_na.to_numpy()), "Unhandled AUPRC missingness")
    require(np.array_equal(metrics.auroc.isna().to_numpy(), expected_auroc_na.to_numpy()), "Unhandled AUROC missingness")
    required_columns = [column for column in metrics.columns if column not in ("auprc", "auroc")]
    require(not metrics[required_columns].isna().any().any() and not recall.isna().any().any(), "Unhandled all-token audit missing value")
    atomic_csv(metrics, METRICS_PATH); atomic_csv(recall, RECALL_PATH)
    atomic_npz(FIRST_PREDICTION_PATH, **first_prediction, target=first_target, sample_index=np.arange(N_FIRST, dtype=np.int32), fold=fold_by_sample)

    mean_metric_records = []
    for (population, model_name), group in metrics.groupby(["population", "model"], sort=False):
        auprc_defined_count = int(group.auprc_defined.sum())
        auroc_defined_count = int(group.auroc_defined.sum())
        mean_metric_records.append({
            "population": population, "model": model_name,
            "auprc": float(group.auprc.mean()) if auprc_defined_count == 32 else None,
            "auprc_defined_experts": auprc_defined_count,
            "auroc": float(group.auroc.mean()) if auroc_defined_count == 32 else None,
            "auroc_defined_experts": auroc_defined_count,
            "log_loss": float(group.log_loss.mean()),
        })
    outputs = [FIRST_PREDICTION_PATH, METRICS_PATH, RECALL_PATH]
    manifest = {
        "status": "complete", "step": "04E_all_token_audit", "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "primary_models_refit": False, "first_tokens_used_for_training": False,
            "first_token_prediction": "frozen primary-trained fold-specific probes",
            "first_token_selected_union_features": int(len(union_features)),
            "interpretation": "sensitivity audit only because first-token SAE activations are anomalous",
        },
        "counts": {
            "first_tokens": N_FIRST, "first_union_nnz": int(first_union.nnz),
            "unique_first_token_expert_sets": int(len(unique_first_routes)),
            "first_token_expert_sets": unique_first_routes.astype(int).tolist(),
            "metric_rows": len(metrics), "recall_rows": len(recall),
        },
        "mean_metrics": mean_metric_records,
        "recall": recall.to_dict(orient="records"),
        "validation": {
            "unhandled_missing_values": 0, "nonfinite_predictions": 0,
            "handled_undefined_auprc_rows": int(expected_auprc_na.sum()),
            "handled_undefined_auroc_rows": int(expected_auroc_na.sum()),
            "first_target_labels_per_token": 4, "sample_leakage": 0,
        },
        "inputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in inputs},
        "outputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs},
        "script": script_record(SCRIPT), "model_inference_rerun": False, "probe_refit": False, "dense_sae_matrix_created": False,
    }
    atomic_json(manifest, MANIFEST_PATH)
    print(json.dumps({"status": "complete", "counts": manifest["counts"], "mean_metrics": manifest["mean_metrics"], "recall": manifest["recall"], "validation": manifest["validation"], "outputs": manifest["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
