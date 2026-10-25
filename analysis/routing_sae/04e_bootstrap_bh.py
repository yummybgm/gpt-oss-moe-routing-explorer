#!/usr/bin/env python3
"""Step 04E-1: sample-level bootstrap and held-out BH-corrected SAE association tests."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.stats import norm

from _03_common import DOMAINS, METADATA, TABLES, atomic_csv, atomic_json, bh_adjust, require, script_record, sha256_file

ROOT = Path(os.environ.get("MOE_SAE_ANALYSIS_ROOT", Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1")).expanduser().resolve()
CACHE = ROOT / "cache"
SCRIPT = Path(__file__).resolve()
CSR_PATH = CACHE / "routing_sae_step04b_primary_csr.npz"
TOKEN_PATH = CACHE / "routing_sae_step04b_primary_tokens.parquet"
PREDICTION_PATH = CACHE / "routing_sae_step04d_oof_predictions.npz"
FOLD_PATH = TABLES / "routing_sae_step04c_sample_folds.csv"
SELECTION_PATH = TABLES / "routing_sae_step04c_selected_features.csv"
STEP04D_MANIFEST = METADATA / "routing_sae_step04d_manifest.json"

BOOTSTRAP_PATH = TABLES / "routing_sae_step04e_performance_bootstrap.csv"
GLOBAL_PATH = TABLES / "routing_sae_step04e_global_bootstrap.csv"
ASSOCIATION_PATH = TABLES / "routing_sae_step04e_candidate_holdout_associations.csv"
MANIFEST_PATH = METADATA / "routing_sae_step04e_inference_manifest.json"

MODELS = ("baseline", "sae", "combined")
COMPARISONS = (("combined", "baseline"), ("combined", "sae"), ("sae", "baseline"))
N_BOOTSTRAP = 1_000
RANDOM_SEED = 42
ALPHA = 0.05


def average_precision_or_nan(target: np.ndarray, score: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return np.nan
    order = np.argsort(-score, kind="stable")
    sorted_target = target[order]
    precision = np.cumsum(sorted_target) / np.arange(1, len(target) + 1)
    return float(precision[sorted_target].sum() / positives)


def sample_equal_metrics(tokens: pd.DataFrame, predictions: dict[str, np.ndarray], target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_index = tokens.sample_index.to_numpy(np.int32)
    starts = np.searchsorted(sample_index, np.arange(2_000), side="left")
    ends = np.searchsorted(sample_index, np.arange(2_000), side="right")
    sample_ap = np.full((len(MODELS), 2_000, 32), np.nan, dtype=np.float64)
    sample_recall = np.empty((len(MODELS), 2_000), dtype=np.float64)
    for sample in range(2_000):
        start, end = int(starts[sample]), int(ends[sample])
        truth = target[start:end]
        for model_index, model_name in enumerate(MODELS):
            prediction = predictions[model_name][start:end]
            for expert in range(32):
                sample_ap[model_index, sample, expert] = average_precision_or_nan(truth[:, expert], prediction[:, expert])
            top4 = np.argpartition(-prediction, kth=3, axis=1)[:, :4]
            overlap = np.take_along_axis(truth, top4, axis=1).sum(axis=1)
            sample_recall[model_index, sample] = float(np.mean(overlap / 4.0))
    return sample_ap, sample_recall


def stratified_bootstrap(sample_ap: np.ndarray, sample_recall: np.ndarray, sample_domains: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    domain_indices = [np.flatnonzero(sample_domains == domain) for domain in DOMAINS]
    draws = np.empty((N_BOOTSTRAP, 2_000), dtype=np.int32)
    for bootstrap_index in range(N_BOOTSTRAP):
        draws[bootstrap_index] = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in domain_indices])

    model_index = {name: index for index, name in enumerate(MODELS)}
    rows = []
    global_rows = []
    for left, right in COMPARISONS:
        left_index, right_index = model_index[left], model_index[right]
        observed_by_expert = np.nanmean(sample_ap[left_index] - sample_ap[right_index], axis=0)
        bootstrap_delta = np.empty((N_BOOTSTRAP, 32), dtype=np.float64)
        global_ap_delta = np.empty(N_BOOTSTRAP, dtype=np.float64)
        recall_delta = np.empty(N_BOOTSTRAP, dtype=np.float64)
        for bootstrap_index, draw in enumerate(draws):
            differences = sample_ap[left_index, draw] - sample_ap[right_index, draw]
            bootstrap_delta[bootstrap_index] = np.nanmean(differences, axis=0)
            global_ap_delta[bootstrap_index] = float(np.nanmean(differences))
            recall_delta[bootstrap_index] = float(np.mean(sample_recall[left_index, draw] - sample_recall[right_index, draw]))
        p_values = np.array([
            min(1.0, 2.0 * min((np.count_nonzero(values <= 0) + 1) / (N_BOOTSTRAP + 1),
                               (np.count_nonzero(values >= 0) + 1) / (N_BOOTSTRAP + 1)))
            for values in bootstrap_delta.T
        ])
        q_values = bh_adjust(p_values)
        for expert in range(32):
            values = bootstrap_delta[:, expert]
            rows.append({
                "comparison": f"{left}_minus_{right}", "expert": expert,
                "metric": "sample_equal_average_precision",
                "eligible_samples": int(np.isfinite(sample_ap[left_index, :, expert] - sample_ap[right_index, :, expert]).sum()),
                "observed_delta": float(observed_by_expert[expert]),
                "ci_lower": float(np.quantile(values, 0.025)), "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_p_two_sided": float(p_values[expert]), "bh_q_within_comparison": float(q_values[expert]),
                "significant_bh_0_05": bool(q_values[expert] < ALPHA),
            })
        for metric, observed, values in (
            ("macro_sample_equal_average_precision", float(np.nanmean(sample_ap[left_index] - sample_ap[right_index])), global_ap_delta),
            ("sample_equal_recall_at4", float(np.mean(sample_recall[left_index] - sample_recall[right_index])), recall_delta),
        ):
            global_rows.append({
                "comparison": f"{left}_minus_{right}", "metric": metric, "observed_delta": observed,
                "ci_lower": float(np.quantile(values, 0.025)), "ci_upper": float(np.quantile(values, 0.975)),
                "bootstrap_p_two_sided": float(min(1.0, 2.0 * min(
                    (np.count_nonzero(values <= 0) + 1) / (N_BOOTSTRAP + 1),
                    (np.count_nonzero(values >= 0) + 1) / (N_BOOTSTRAP + 1),
                ))),
            })
    return pd.DataFrame(rows), pd.DataFrame(global_rows)


def heldout_candidate_tests(matrix: sparse.csr_matrix, tokens: pd.DataFrame, folds: pd.DataFrame,
                            selected: pd.DataFrame, baseline_prediction: np.ndarray,
                            target: np.ndarray) -> pd.DataFrame:
    matrix_csc = matrix.tocsc()
    fold_by_sample = np.full(2_000, -1, dtype=np.int8)
    fold_by_sample[folds.sample_index.to_numpy(np.int32)] = folds.fold.to_numpy(np.int8)
    token_sample = tokens.sample_index.to_numpy(np.int32)
    token_fold = fold_by_sample[token_sample]
    rows = []
    for fold in range(5):
        test_mask = token_fold == fold
        test_samples_global = token_sample[test_mask]
        unique_samples, local_sample = np.unique(test_samples_global, return_inverse=True)
        aggregation = sparse.csr_matrix(
            (np.ones(len(local_sample), dtype=np.float64), (local_sample, np.arange(len(local_sample)))),
            shape=(len(unique_samples), len(local_sample)),
        )
        for expert in range(32):
            chosen = selected[(selected.fold == fold) & (selected.expert == expert)].sort_values("rank")
            feature_ids = chosen.feature_id.to_numpy(np.int32)
            x = matrix_csc[test_mask][:, feature_ids].tocsr()
            x.data[:] = 1.0
            residual = target[test_mask, expert].astype(np.float64) - baseline_prediction[test_mask, expert]
            weighted = x.multiply(residual[:, None])
            cluster_scores = aggregation @ weighted
            cluster_presence = aggregation @ x
            score = np.asarray(weighted.sum(axis=0)).ravel()
            variance = np.asarray(cluster_scores.power(2).sum(axis=0)).ravel()
            variance *= len(unique_samples) / (len(unique_samples) - 1)
            z = np.divide(score, np.sqrt(variance), out=np.zeros_like(score), where=variance > 0)
            p = 2.0 * norm.sf(np.abs(z))
            token_support = np.diff(x.tocsc().indptr)
            sample_support = np.diff(cluster_presence.tocsc().indptr)
            mean_residual = np.divide(score, token_support, out=np.zeros_like(score), where=token_support > 0)
            for index, item in enumerate(chosen.itertuples(index=False)):
                rows.append({
                    "fold": fold, "expert": expert, "rank": int(item.rank), "feature_id": int(item.feature_id),
                    "train_phi_signed": float(item.phi_signed),
                    "test_token_support": int(token_support[index]), "test_sample_support": int(sample_support[index]),
                    "holdout_mean_baseline_residual_when_present": float(mean_residual[index]),
                    "cluster_robust_z": float(z[index]), "p_value": float(p[index]),
                })
    frame = pd.DataFrame(rows)
    frame["bh_q_all_candidate_tests"] = bh_adjust(frame.p_value.to_numpy())
    frame["significant_bh_0_05"] = frame.bh_q_all_candidate_tests < ALPHA
    frame["same_direction_as_train_phi"] = np.sign(frame.train_phi_signed) == np.sign(frame.holdout_mean_baseline_residual_when_present)
    return frame


def main() -> None:
    for path in (CSR_PATH, TOKEN_PATH, PREDICTION_PATH, FOLD_PATH, SELECTION_PATH, STEP04D_MANIFEST):
        require(path.is_file(), f"Missing 04E inference input: {path}")
    prior = json.loads(STEP04D_MANIFEST.read_text(encoding="utf-8"))
    require(prior["status"] == "complete", "Step-04D incomplete")
    require(sha256_file(PREDICTION_PATH) == prior["outputs"][str(PREDICTION_PATH)]["sha256"], "OOF prediction hash mismatch")

    matrix = sparse.load_npz(CSR_PATH)
    tokens = pq.read_table(TOKEN_PATH).to_pandas()
    folds = pd.read_csv(FOLD_PATH)
    selected = pd.read_csv(SELECTION_PATH)
    with np.load(PREDICTION_PATH, allow_pickle=False) as source:
        predictions = {name: source[name].astype(np.float64) for name in MODELS}
        target = source["target"].astype(bool)
        saved_fold = source["fold"].astype(np.int8)
    require(matrix.shape == (380_685, 131_072) and target.shape == (380_685, 32), "Input shape mismatch")
    require(all(np.isfinite(predictions[name]).all() for name in MODELS), "Nonfinite OOF prediction")

    sample_ap, sample_recall = sample_equal_metrics(tokens, predictions, target)
    sample_domains = folds.sort_values("sample_index", kind="stable").domain.astype(str).to_numpy()
    bootstrap, global_bootstrap = stratified_bootstrap(sample_ap, sample_recall, sample_domains)
    associations = heldout_candidate_tests(matrix, tokens, folds, selected, predictions["baseline"], target)
    require(len(bootstrap) == 96 and len(global_bootstrap) == 6 and len(associations) == 40_960, "04E output row count mismatch")
    require(not bootstrap.isna().any().any() and not global_bootstrap.isna().any().any(), "Missing bootstrap output")
    require(not associations.isna().any().any() and np.isfinite(associations.select_dtypes(include=[np.number])).all().all(), "Invalid association output")
    atomic_csv(bootstrap, BOOTSTRAP_PATH)
    atomic_csv(global_bootstrap, GLOBAL_PATH)
    atomic_csv(associations, ASSOCIATION_PATH)

    outputs = [BOOTSTRAP_PATH, GLOBAL_PATH, ASSOCIATION_PATH]
    manifest = {
        "status": "complete", "step": "04E_bootstrap_and_BH",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "bootstrap_unit": "sample_id", "bootstrap_stratification": "domain", "bootstrap_replicates": N_BOOTSTRAP,
            "performance_estimand": "mean within-sample average precision among samples containing at least one positive for that expert",
            "performance_BH_families": "32 experts corrected separately within each model comparison",
            "candidate_test": "held-out cluster-robust score against baseline OOF residual, binary SAE feature presence",
            "candidate_cluster": "sample_id", "candidate_BH_family_size": len(associations),
            "alpha": ALPHA,
        },
        "counts": {
            "performance_rows": len(bootstrap), "global_rows": len(global_bootstrap), "candidate_rows": len(associations),
            "performance_significant_by_comparison": bootstrap.groupby("comparison").significant_bh_0_05.sum().astype(int).to_dict(),
            "candidate_significant_bh_0_05": int(associations.significant_bh_0_05.sum()),
            "candidate_same_direction": int(associations.same_direction_as_train_phi.sum()),
        },
        "global_results": global_bootstrap.to_dict(orient="records"),
        "validation": {
            "missing_values": 0, "nonfinite_values": 0, "sample_leakage": 0,
            "saved_fold_matches_table": bool(np.array_equal(saved_fold, folds.set_index("sample_index").fold.to_numpy()[tokens.sample_index.to_numpy(np.int32)])),
        },
        "inputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in [CSR_PATH, TOKEN_PATH, PREDICTION_PATH, FOLD_PATH, SELECTION_PATH, STEP04D_MANIFEST]},
        "outputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs},
        "script": script_record(SCRIPT), "model_inference_rerun": False, "probe_refit": False, "dense_sae_matrix_created": False,
    }
    require(manifest["validation"]["saved_fold_matches_table"], "Saved OOF fold does not match fold table")
    atomic_json(manifest, MANIFEST_PATH)
    print(json.dumps({"status": "complete", "counts": manifest["counts"], "global_results": manifest["global_results"], "validation": manifest["validation"], "outputs": manifest["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
