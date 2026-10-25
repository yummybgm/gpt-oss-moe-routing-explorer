#!/usr/bin/env python3
"""Step 04B: build a cache-only sparse SAE/routing joint index for primary tokens."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from _03_common import (
    ANNOTATION_PATH,
    CACHE,
    METADATA,
    N_EXPERTS,
    N_PRIMARY,
    N_SAMPLES,
    N_TOKENS,
    ROUTING_PATH,
    SAMPLE_PATH,
    TABLES,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    ensure_dirs,
    load_inputs,
    require,
    script_record,
    sha256_file,
)

SCRIPT = Path(__file__).resolve()
TOKEN_PATH = CACHE / "token_table.parquet"
SAE_PATH = CACHE / "sae_layer11_sparse.parquet"
CONTRACT_PATH = METADATA / "routing_sae_step04a_contract.json"

CSR_PATH = CACHE / "routing_sae_step04b_primary_csr.npz"
TOKEN_INDEX_PATH = CACHE / "routing_sae_step04b_primary_tokens.parquet"
FEATURE_SUPPORT_PATH = TABLES / "routing_sae_step04b_feature_support.csv"
SUPPORT_SUMMARY_PATH = TABLES / "routing_sae_step04b_support_summary.csv"
MANIFEST_PATH = METADATA / "routing_sae_step04b_manifest.json"

DICT_SIZE = 131_072
EXPECTED_ALL_NNZ = 72_151_749
EXPECTED_PRIMARY_NNZ = 28_081_931
BATCH_SIZE = 1_000_000


def atomic_save_npz(matrix: sparse.csr_matrix, destination: Path) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        sparse.save_npz(temporary, matrix, compressed=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    ensure_dirs()
    for path in (TOKEN_PATH, SAE_PATH, CONTRACT_PATH):
        require(path.is_file(), f"Missing step-04B input: {path}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    require(contract["status"] == "input_validation_complete_method_choice_required", "Step-04A contract status mismatch")
    require(contract["fixed_inputs"]["sae_sparse_rows_all"] == EXPECTED_ALL_NNZ, "Step-04A SAE row count mismatch")
    require(contract["fixed_inputs"]["sae_sparse_rows_primary"] == EXPECTED_PRIMARY_NNZ, "Step-04A primary SAE row count mismatch")

    annotation_columns = [
        "sample_id", "token_position", "mask_primary_inference", "token_class",
        "prompt_segment", "is_first_token",
    ]
    samples, annotations, routing, _ = load_inputs(annotation_columns)
    token_table = pq.read_table(
        TOKEN_PATH,
        columns=["sample_id", "domain", "token_position", "token_id", "active_feature_count", "is_first_token"],
    ).to_pandas()
    require(len(token_table) == N_TOKENS, "token_table row count mismatch")
    require(token_table.isna().sum().sum() == 0, "Missing token-table value")
    require(np.array_equal(token_table.sample_id.astype(str), annotations.sample_id.astype(str)), "Token/annotation sample order mismatch")
    require(np.array_equal(token_table.token_position.to_numpy(), annotations.token_position.to_numpy()), "Token/annotation position mismatch")

    primary_mask = annotations.mask_primary_inference.to_numpy(bool)
    require(int(primary_mask.sum()) == N_PRIMARY, "Primary mask count mismatch")
    require(np.array_equal(~primary_mask, annotations.is_first_token.to_numpy(bool)), "Primary mask does not exclude exactly first tokens")
    active_counts = token_table.active_feature_count.to_numpy(np.int64)
    require(int(active_counts.sum()) == EXPECTED_ALL_NNZ, "All-token active-count sum mismatch")
    require(int(active_counts[primary_mask].sum()) == EXPECTED_PRIMARY_NNZ, "Primary active-count sum mismatch")
    sparse_row_ends = np.cumsum(active_counts, dtype=np.int64)

    temporary_dir = Path(tempfile.mkdtemp(prefix="step04b_", dir=CACHE))
    data_map = np.memmap(temporary_dir / "data.float32", mode="w+", dtype=np.float32, shape=(EXPECTED_PRIMARY_NNZ,))
    index_map = np.memmap(temporary_dir / "indices.int32", mode="w+", dtype=np.int32, shape=(EXPECTED_PRIMARY_NNZ,))
    primary_indptr = np.empty(N_PRIMARY + 1, dtype=np.int64)
    primary_indptr[0] = 0
    np.cumsum(active_counts[primary_mask], out=primary_indptr[1:])
    require(int(primary_indptr[-1]) == EXPECTED_PRIMARY_NNZ, "Primary indptr endpoint mismatch")

    parquet_file = pq.ParquetFile(SAE_PATH)
    require(parquet_file.metadata.num_rows == EXPECTED_ALL_NNZ, "SAE parquet row count mismatch")
    sparse_row_offset = 0
    primary_write_offset = 0
    nonfinite = 0
    zero_activations = 0
    invalid_features = 0
    token_position_mismatches = 0
    token_id_mismatches = 0
    false_thresholds = 0
    columns = ["token_position", "token_id", "feature_id", "activation", "passed_threshold"]
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        frame = batch.to_pandas()
        batch_rows = len(frame)
        sparse_positions = np.arange(sparse_row_offset, sparse_row_offset + batch_rows, dtype=np.int64)
        token_indices = np.searchsorted(sparse_row_ends, sparse_positions, side="right")
        expected_positions = token_table.token_position.to_numpy()[token_indices]
        expected_token_ids = token_table.token_id.to_numpy()[token_indices]
        token_position_mismatches += int(np.count_nonzero(frame.token_position.to_numpy() != expected_positions))
        token_id_mismatches += int(np.count_nonzero(frame.token_id.to_numpy() != expected_token_ids))

        features = frame.feature_id.to_numpy(np.int64, copy=False)
        activations = frame.activation.to_numpy(np.float32, copy=False)
        batch_primary = primary_mask[token_indices]
        invalid_features += int(np.count_nonzero((features < 0) | (features >= DICT_SIZE)))
        nonfinite += int(np.count_nonzero(~np.isfinite(activations)))
        zero_activations += int(np.count_nonzero(activations == 0))
        false_thresholds += int(np.count_nonzero(~frame.passed_threshold.to_numpy(bool)))
        selected_count = int(batch_primary.sum())
        end = primary_write_offset + selected_count
        index_map[primary_write_offset:end] = features[batch_primary].astype(np.int32, copy=False)
        data_map[primary_write_offset:end] = activations[batch_primary]
        primary_write_offset = end
        sparse_row_offset += batch_rows

    require(sparse_row_offset == EXPECTED_ALL_NNZ, "Did not stream all SAE rows")
    require(primary_write_offset == EXPECTED_PRIMARY_NNZ, "Primary sparse write count mismatch")
    require(nonfinite == 0, "Nonfinite SAE activations")
    require(zero_activations == 0, "Explicit zero SAE activation found")
    require(invalid_features == 0, "Out-of-range SAE feature ID")
    require(false_thresholds == 0, "SAE row failed threshold")
    require(token_position_mismatches == 0 and token_id_mismatches == 0, "SAE/token row order mismatch")
    data_map.flush(); index_map.flush()

    matrix = sparse.csr_matrix((data_map, index_map, primary_indptr), shape=(N_PRIMARY, DICT_SIZE), copy=False)
    require(matrix.nnz == EXPECTED_PRIMARY_NNZ, "CSR nnz mismatch")
    matrix.sort_indices()
    require(matrix.has_sorted_indices, "CSR feature indices are not sorted")
    duplicate_feature_rows = int(sum(
        np.any(np.diff(matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]) == 0)
        for row in range(matrix.shape[0])
    ))
    require(duplicate_feature_rows == 0, "Duplicate feature within a primary token")
    atomic_save_npz(matrix, CSR_PATH)
    del matrix, data_map, index_map
    shutil.rmtree(temporary_dir)

    matrix = sparse.load_npz(CSR_PATH)
    require(matrix.shape == (N_PRIMARY, DICT_SIZE) and matrix.nnz == EXPECTED_PRIMARY_NNZ, "Saved CSR reload mismatch")

    primary_global_index = np.flatnonzero(primary_mask).astype(np.int32)
    primary_sample_index = routing["token_sample_index"][primary_mask].astype(np.int32, copy=False)
    layer11 = routing["expert_id"][primary_mask, 11, :].astype(np.int16, copy=False)
    require(not (np.sort(layer11, axis=1)[:, 1:] == np.sort(layer11, axis=1)[:, :-1]).any(), "Duplicate Layer-11 expert per token")
    target_mask = np.bitwise_or.reduce(np.left_shift(np.uint32(1), layer11.astype(np.uint32)), axis=1)
    primary_tokens = pd.DataFrame({
        "primary_row_index": np.arange(N_PRIMARY, dtype=np.int32),
        "global_token_index": primary_global_index,
        "sample_index": primary_sample_index,
        "sample_id": annotations.loc[primary_mask, "sample_id"].astype(str).to_numpy(),
        "domain": token_table.loc[primary_mask, "domain"].astype(str).to_numpy(),
        "token_position": annotations.loc[primary_mask, "token_position"].to_numpy(np.int32),
        "token_class": annotations.loc[primary_mask, "token_class"].astype(str).to_numpy(),
        "prompt_segment": annotations.loc[primary_mask, "prompt_segment"].astype(str).to_numpy(),
        "expert_0": layer11[:, 0], "expert_1": layer11[:, 1],
        "expert_2": layer11[:, 2], "expert_3": layer11[:, 3],
        "expert_target_bitmask": target_mask,
    })
    require(primary_tokens.isna().sum().sum() == 0, "Missing primary-token index value")
    require(primary_tokens[["sample_id", "token_position"]].duplicated().sum() == 0, "Duplicate primary token key")
    atomic_parquet(primary_tokens, TOKEN_INDEX_PATH)

    token_support = np.bincount(matrix.indices, minlength=DICT_SIZE).astype(np.int64)
    activation_sum = np.bincount(matrix.indices, weights=matrix.data, minlength=DICT_SIZE)
    activation_sq_sum = np.bincount(matrix.indices, weights=np.square(matrix.data, dtype=np.float64), minlength=DICT_SIZE)
    activation_max = np.zeros(DICT_SIZE, dtype=np.float32)
    np.maximum.at(activation_max, matrix.indices, matrix.data)
    sample_support = np.zeros(DICT_SIZE, dtype=np.int32)
    sample_primary_counts = np.bincount(primary_sample_index, minlength=N_SAMPLES)
    sample_row_ends = np.cumsum(sample_primary_counts, dtype=np.int64)
    sample_row_starts = np.r_[0, sample_row_ends[:-1]]
    for start, end in zip(sample_row_starts, sample_row_ends):
        feature_start = matrix.indptr[int(start)]
        feature_end = matrix.indptr[int(end)]
        sample_support[np.unique(matrix.indices[feature_start:feature_end])] += 1
    nonzero = token_support > 0
    activation_mean = np.divide(activation_sum, token_support, out=np.zeros(DICT_SIZE), where=nonzero)
    activation_variance = np.divide(activation_sq_sum, token_support, out=np.zeros(DICT_SIZE), where=nonzero) - activation_mean ** 2
    activation_variance = np.maximum(activation_variance, 0.0)
    feature_support = pd.DataFrame({
        "feature_id": np.arange(DICT_SIZE, dtype=np.int32),
        "token_support": token_support,
        "sample_support": sample_support,
        "activation_mean": activation_mean,
        "activation_std": np.sqrt(activation_variance),
        "activation_max": activation_max,
    })
    require(feature_support.isna().sum().sum() == 0, "Missing feature-support statistic")
    require(np.isfinite(feature_support[["activation_mean", "activation_std", "activation_max"]].to_numpy()).all(), "Nonfinite feature-support statistic")
    require(int(feature_support.token_support.sum()) == EXPECTED_PRIMARY_NNZ, "Feature supports do not sum to primary nnz")
    atomic_csv(feature_support, FEATURE_SUPPORT_PATH)

    thresholds = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1_000]
    summary_rows = []
    for threshold in thresholds:
        eligible = token_support >= threshold
        summary_rows.append({
            "criterion": "minimum_token_support",
            "threshold": threshold,
            "eligible_features": int(eligible.sum()),
            "eligible_fraction_dictionary": float(eligible.mean()),
            "eligible_nnz": int(token_support[eligible].sum()),
            "eligible_nnz_fraction": float(token_support[eligible].sum() / EXPECTED_PRIMARY_NNZ),
        })
    for threshold in [2, 5, 10, 20, 25, 50, 100]:
        eligible = sample_support >= threshold
        summary_rows.append({
            "criterion": "minimum_sample_support",
            "threshold": threshold,
            "eligible_features": int(eligible.sum()),
            "eligible_fraction_dictionary": float(eligible.mean()),
            "eligible_nnz": int(token_support[eligible].sum()),
            "eligible_nnz_fraction": float(token_support[eligible].sum() / EXPECTED_PRIMARY_NNZ),
        })
    support_summary = pd.DataFrame(summary_rows)
    atomic_csv(support_summary, SUPPORT_SUMMARY_PATH)

    output_paths = [CSR_PATH, TOKEN_INDEX_PATH, FEATURE_SUPPORT_PATH, SUPPORT_SUMMARY_PATH]
    manifest = {
        "status": "complete",
        "step": "04B_sparse_joint_index",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "primary_unit": "token excluding exactly the first token of each sample",
            "sae_representation": "scipy CSR sparse matrix",
            "target": "32 binary expert-selected labels encoded losslessly as four expert columns plus uint32 bitmask",
            "feature_selection": "not performed in 04B; must occur inside each training fold in 04C",
            "all_token_audit": "retained in original cache; no all-token duplicate matrix was created",
        },
        "shape": [N_PRIMARY, DICT_SIZE],
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (N_PRIMARY * DICT_SIZE)),
        "active_features": int(nonzero.sum()),
        "inactive_features": int((~nonzero).sum()),
        "token_support_nonzero_quantiles": {
            str(q): float(np.quantile(token_support[nonzero], q)) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
        },
        "sample_support_nonzero_quantiles": {
            str(q): float(np.quantile(sample_support[nonzero], q)) for q in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
        },
        "validation": {
            "missing_values": 0,
            "nonfinite_values": 0,
            "explicit_zero_activations": zero_activations,
            "invalid_feature_ids": invalid_features,
            "false_threshold_rows": false_thresholds,
            "token_position_mismatches": token_position_mismatches,
            "token_id_mismatches": token_id_mismatches,
            "duplicate_feature_rows": duplicate_feature_rows,
            "target_labels_per_token": 4,
        },
        "inputs": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in [TOKEN_PATH, SAE_PATH, ANNOTATION_PATH, ROUTING_PATH, SAMPLE_PATH, CONTRACT_PATH]
        },
        "outputs": {
            str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in output_paths
        },
        "script": script_record(SCRIPT),
        "raw_source_reparsed": False,
        "model_loaded": False,
        "model_inference_rerun": False,
        "dense_sae_matrix_created": False,
    }
    atomic_json(manifest, MANIFEST_PATH)
    print(json.dumps({
        "status": "complete", "shape": manifest["shape"], "nnz": manifest["nnz"],
        "density": manifest["density"], "active_features": manifest["active_features"],
        "token_support_nonzero_quantiles": manifest["token_support_nonzero_quantiles"],
        "sample_support_nonzero_quantiles": manifest["sample_support_nonzero_quantiles"],
        "outputs": manifest["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
