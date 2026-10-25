#!/usr/bin/env python3
"""Step 04C: grouped, domain-balanced folds and fold-internal SAE feature screening."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from _03_common import (
    DOMAINS,
    METADATA,
    N_EXPERTS,
    N_PRIMARY,
    N_SAMPLES,
    RANDOM_SEED,
    TABLES,
    atomic_csv,
    atomic_json,
    require,
    script_record,
    sha256_file,
)

ROOT = Path(
    os.environ.get(
        "MOE_SAE_ANALYSIS_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1",
    )
).expanduser().resolve()
CACHE = ROOT / "cache"
SCRIPT = Path(__file__).resolve()
CSR_PATH = CACHE / "routing_sae_step04b_primary_csr.npz"
TOKEN_INDEX_PATH = CACHE / "routing_sae_step04b_primary_tokens.parquet"
STEP04B_MANIFEST = METADATA / "routing_sae_step04b_manifest.json"

FOLD_PATH = TABLES / "routing_sae_step04c_sample_folds.csv"
SELECTION_PATH = TABLES / "routing_sae_step04c_selected_features.csv"
AUDIT_PATH = TABLES / "routing_sae_step04c_fold_audit.csv"
MANIFEST_PATH = METADATA / "routing_sae_step04c_manifest.json"

N_FOLDS = 5
MIN_TRAIN_SAMPLE_SUPPORT = 20
TOP_FEATURES = 256
DICT_SIZE = 131_072


def build_folds(tokens: pd.DataFrame) -> pd.DataFrame:
    samples = tokens[["sample_index", "sample_id", "domain"]].drop_duplicates().sort_values("sample_index", kind="stable")
    require(len(samples) == N_SAMPLES and samples.sample_index.is_unique, "Primary-token sample map mismatch")
    require(set(samples.domain) == set(DOMAINS), "Unexpected domain set")
    rng = np.random.default_rng(RANDOM_SEED)
    folds = np.full(N_SAMPLES, -1, dtype=np.int8)
    for domain in DOMAINS:
        indices = samples.loc[samples.domain.eq(domain), "sample_index"].to_numpy(np.int32)
        require(len(indices) % N_FOLDS == 0, f"Domain {domain} cannot be evenly divided into {N_FOLDS} folds")
        shuffled = rng.permutation(indices)
        folds[shuffled] = np.arange(len(shuffled), dtype=np.int32) % N_FOLDS
    require((folds >= 0).all(), "A sample was not assigned to a fold")
    samples = samples.copy()
    samples["fold"] = folds[samples.sample_index.to_numpy(np.int32)]
    return samples.reset_index(drop=True)


def train_sample_support(matrix: sparse.csr_matrix, sample_indices: np.ndarray, train_samples: np.ndarray) -> np.ndarray:
    starts = np.searchsorted(sample_indices, np.arange(N_SAMPLES), side="left")
    ends = np.searchsorted(sample_indices, np.arange(N_SAMPLES), side="right")
    support = np.zeros(DICT_SIZE, dtype=np.int32)
    for sample_index in train_samples:
        row_start = int(starts[sample_index]); row_end = int(ends[sample_index])
        feature_start = matrix.indptr[row_start]; feature_end = matrix.indptr[row_end]
        support[np.unique(matrix.indices[feature_start:feature_end])] += 1
    return support


def main() -> None:
    for path in (CSR_PATH, TOKEN_INDEX_PATH, STEP04B_MANIFEST):
        require(path.is_file(), f"Missing step-04C input: {path}")
    prior = json.loads(STEP04B_MANIFEST.read_text(encoding="utf-8"))
    require(prior["status"] == "complete", "Step-04B is incomplete")
    require(prior["shape"] == [N_PRIMARY, DICT_SIZE] and prior["nnz"] == 28_081_931, "Step-04B shape/nnz mismatch")
    require(sha256_file(CSR_PATH) == prior["outputs"][str(CSR_PATH)]["sha256"], "Step-04B CSR hash mismatch")
    require(sha256_file(TOKEN_INDEX_PATH) == prior["outputs"][str(TOKEN_INDEX_PATH)]["sha256"], "Step-04B token-index hash mismatch")

    matrix = sparse.load_npz(CSR_PATH)
    tokens = pq.read_table(TOKEN_INDEX_PATH).to_pandas()
    require(sparse.isspmatrix_csr(matrix) and matrix.shape == (N_PRIMARY, DICT_SIZE), "Unexpected CSR matrix")
    require(len(tokens) == N_PRIMARY and tokens.isna().sum().sum() == 0, "Primary-token table mismatch")
    require(np.array_equal(tokens.primary_row_index.to_numpy(), np.arange(N_PRIMARY)), "Primary row order mismatch")
    sample_indices = tokens.sample_index.to_numpy(np.int32)
    require(np.all(sample_indices[:-1] <= sample_indices[1:]), "Tokens are not contiguous by sample")

    sample_folds = build_folds(tokens)
    atomic_csv(sample_folds, FOLD_PATH)
    fold_by_sample = np.full(N_SAMPLES, -1, dtype=np.int8)
    fold_by_sample[sample_folds.sample_index.to_numpy(np.int32)] = sample_folds.fold.to_numpy(np.int8)
    token_folds = fold_by_sample[sample_indices]

    expert_columns = [f"expert_{rank}" for rank in range(4)]
    experts = tokens[expert_columns].to_numpy(np.int16)
    selection_rows: list[dict[str, int | float]] = []
    audit_rows: list[dict[str, int | str | float]] = []

    for fold in range(N_FOLDS):
        test_sample_mask = sample_folds.fold.eq(fold).to_numpy()
        test_samples = sample_folds.loc[test_sample_mask, "sample_index"].to_numpy(np.int32)
        train_samples = sample_folds.loc[~test_sample_mask, "sample_index"].to_numpy(np.int32)
        train_token_mask = token_folds != fold
        test_token_mask = ~train_token_mask
        require(len(np.intersect1d(train_samples, test_samples)) == 0, "Train/test sample leakage")
        require(len(train_samples) == 1_600 and len(test_samples) == 400, "Unexpected fold sample counts")

        domain_counts = sample_folds.loc[test_sample_mask, "domain"].value_counts()
        require(all(int(domain_counts.get(domain, 0)) == 80 for domain in DOMAINS), "Test fold is not domain-balanced")
        train_matrix = matrix[train_token_mask]
        train_experts = experts[train_token_mask]
        n_train_tokens = train_matrix.shape[0]
        token_support = np.bincount(train_matrix.indices, minlength=DICT_SIZE).astype(np.int64)
        sample_support = train_sample_support(matrix, sample_indices, train_samples)
        eligible_common = sample_support >= MIN_TRAIN_SAMPLE_SUPPORT
        require(int(eligible_common.sum()) >= TOP_FEATURES, "Too few eligible features")

        audit_rows.append({
            "fold": fold, "record_type": "fold_total", "domain": "all", "expert": -1,
            "train_samples": len(train_samples), "test_samples": len(test_samples),
            "train_tokens": int(train_token_mask.sum()), "test_tokens": int(test_token_mask.sum()),
            "eligible_features": int(eligible_common.sum()), "positive_train_tokens": -1,
        })
        for domain in DOMAINS:
            audit_rows.append({
                "fold": fold, "record_type": "test_domain", "domain": domain, "expert": -1,
                "train_samples": -1, "test_samples": int(domain_counts[domain]),
                "train_tokens": -1, "test_tokens": int((test_token_mask & tokens.domain.eq(domain).to_numpy()).sum()),
                "eligible_features": -1, "positive_train_tokens": -1,
            })

        for expert in range(N_EXPERTS):
            selected_label = (train_experts == expert).any(axis=1)
            n_positive = int(selected_label.sum())
            require(0 < n_positive < n_train_tokens, f"Degenerate target for fold {fold}, expert {expert}")
            positive_feature_count = np.bincount(train_matrix[selected_label].indices, minlength=DICT_SIZE).astype(np.int64)
            denominator = np.sqrt(
                token_support.astype(np.float64)
                * (n_train_tokens - token_support)
                * float(n_positive)
                * float(n_train_tokens - n_positive)
            )
            numerator = positive_feature_count.astype(np.float64) * n_train_tokens - token_support * n_positive
            phi = np.divide(numerator, denominator, out=np.zeros(DICT_SIZE, dtype=np.float64), where=denominator > 0)
            eligible = eligible_common & (denominator > 0)
            eligible_ids = np.flatnonzero(eligible)
            require(len(eligible_ids) >= TOP_FEATURES, f"Too few eligible features for fold {fold}, expert {expert}")
            order = np.lexsort((
                eligible_ids,
                -token_support[eligible_ids],
                -sample_support[eligible_ids],
                -np.abs(phi[eligible_ids]),
            ))
            selected = eligible_ids[order[:TOP_FEATURES]]
            for rank, feature_id in enumerate(selected, start=1):
                selection_rows.append({
                    "fold": fold, "expert": expert, "rank": rank, "feature_id": int(feature_id),
                    "phi_signed": float(phi[feature_id]), "phi_abs": float(abs(phi[feature_id])),
                    "train_sample_support": int(sample_support[feature_id]),
                    "train_token_support": int(token_support[feature_id]),
                    "train_positive_token_support": int(positive_feature_count[feature_id]),
                    "train_positive_tokens": n_positive, "train_tokens": n_train_tokens,
                })
            audit_rows.append({
                "fold": fold, "record_type": "expert_target", "domain": "all", "expert": expert,
                "train_samples": len(train_samples), "test_samples": len(test_samples),
                "train_tokens": n_train_tokens, "test_tokens": int(test_token_mask.sum()),
                "eligible_features": int(eligible.sum()), "positive_train_tokens": n_positive,
            })

    selections = pd.DataFrame(selection_rows)
    audit = pd.DataFrame(audit_rows)
    require(len(selections) == N_FOLDS * N_EXPERTS * TOP_FEATURES, "Selection row count mismatch")
    require(selections.isna().sum().sum() == 0 and audit.isna().sum().sum() == 0, "Missing output value")
    require(np.isfinite(selections[["phi_signed", "phi_abs"]].to_numpy()).all(), "Nonfinite association score")
    require((selections.train_sample_support >= MIN_TRAIN_SAMPLE_SUPPORT).all(), "Selected feature below sample-support threshold")
    require(selections.groupby(["fold", "expert"]).feature_id.nunique().eq(TOP_FEATURES).all(), "Duplicate selected feature")
    require(selections.groupby(["fold", "expert"]).size().eq(TOP_FEATURES).all(), "Incomplete Top-feature set")
    atomic_csv(selections, SELECTION_PATH)
    atomic_csv(audit, AUDIT_PATH)

    outputs = [FOLD_PATH, SELECTION_PATH, AUDIT_PATH]
    manifest = {
        "status": "complete",
        "step": "04C_grouped_folds_and_fold_internal_feature_selection",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "n_folds": N_FOLDS, "random_seed": RANDOM_SEED,
            "group": "sample_id", "stratification": "domain-balanced exact allocation",
            "samples_per_test_fold": 400, "samples_per_domain_per_test_fold": 80,
            "feature_presence_score": "signed phi coefficient between binary feature presence and binary expert-selected label",
            "minimum_train_sample_support": MIN_TRAIN_SAMPLE_SUPPORT,
            "features_per_expert_per_fold": TOP_FEATURES,
            "tie_breaking": "phi_abs desc, train_sample_support desc, train_token_support desc, feature_id asc",
            "feature_selection_scope": "computed independently inside each training fold",
        },
        "counts": {
            "fold_assignment_rows": len(sample_folds), "selection_rows": len(selections),
            "audit_rows": len(audit), "fold_expert_models_prepared": N_FOLDS * N_EXPERTS,
            "selected_feature_sets": N_FOLDS * N_EXPERTS,
            "unique_selected_features_global": int(selections.feature_id.nunique()),
            "eligible_features_by_fold": audit.loc[audit.record_type.eq("fold_total")].sort_values("fold").eligible_features.astype(int).tolist(),
        },
        "validation": {
            "sample_leakage": 0, "missing_values": 0, "nonfinite_scores": 0,
            "test_samples_per_fold": sample_folds.groupby("fold").size().astype(int).tolist(),
            "test_samples_per_domain_per_fold": 80,
            "selected_features_per_fold_expert": TOP_FEATURES,
            "minimum_observed_selected_sample_support": int(selections.train_sample_support.min()),
        },
        "inputs": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in [CSR_PATH, TOKEN_INDEX_PATH, STEP04B_MANIFEST]
        },
        "outputs": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs
        },
        "script": script_record(SCRIPT),
        "model_training_performed": False,
        "model_loaded": False,
        "model_inference_rerun": False,
        "dense_sae_matrix_created": False,
    }
    atomic_json(manifest, MANIFEST_PATH)
    print(json.dumps({
        "status": "complete", "design": manifest["design"], "counts": manifest["counts"],
        "validation": manifest["validation"], "outputs": manifest["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
