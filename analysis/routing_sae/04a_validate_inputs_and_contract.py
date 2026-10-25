#!/usr/bin/env python3
"""Step 04A: validate Routing-SAE inputs and freeze all non-ambiguous design constraints."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from _03_common import *

SCRIPT = Path(__file__).resolve()
TOKEN_PATH = CACHE / "token_table.parquet"
SAE_PATH = CACHE / "sae_layer11_sparse.parquet"
STEP03_MANIFEST = METADATA / "routing_path_step03_manifest.json"
AUDIT_PATH = TABLES / "routing_sae_step04a_input_audit.csv"
CONTRACT_PATH = METADATA / "routing_sae_step04a_contract.json"
EXPECTED_SAE_ROWS = 72_151_749
EXPECTED_DICT_SIZE = 131_072
ROUTING_LAYER = 11


def add_audit(rows: list[dict[str, object]], item: str, result: object, expected: object, status: str, meaning: str) -> None:
    rows.append({"check_item": item, "result": result, "expected": expected, "status": status, "plain_language": meaning})


def main() -> None:
    ensure_dirs()
    for path in (TOKEN_PATH, SAE_PATH, STEP03_MANIFEST):
        require(path.is_file(), f"Missing step-04 input: {path}")

    annotation_columns = [
        "sample_id", "token_position", "mask_all_tokens", "mask_primary_inference",
        "token_class", "prompt_segment", "is_first_token",
    ]
    samples, annotations, routing, prior_manifests = load_inputs(annotation_columns)
    cache_manifest, annotation_manifest, step02_manifest = prior_manifests
    step03_manifest = json.loads(STEP03_MANIFEST.read_text(encoding="utf-8"))
    require(step03_manifest.get("status") == "complete", "Step-03 manifest is incomplete")
    require(step03_manifest["validation"]["validators_passed"] == 5, "Step-03 validation suite is incomplete")
    require(step03_manifest["validation"]["component_outputs_verified"] == 23, "Step-03 component verification is incomplete")

    token_columns = [
        "sample_id", "domain", "token_position", "token_id", "prompt_segment", "is_first_token",
        "active_feature_count", "reconstruction_mse", "cosine_similarity",
    ]
    token_table = pq.read_table(TOKEN_PATH, columns=token_columns).to_pandas()
    require(len(token_table) == N_TOKENS, "token_table row count mismatch")
    require(token_table[["sample_id", "token_position"]].duplicated().sum() == 0, "Duplicate token key")
    require(token_table[token_columns].isna().sum().sum() == 0, "Missing token-table value")
    require(np.array_equal(token_table.sample_id.astype(str).to_numpy(), annotations.sample_id.astype(str).to_numpy()), "Token/annotation sample order mismatch")
    require(np.array_equal(token_table.token_position.to_numpy(), annotations.token_position.to_numpy()), "Token/annotation position mismatch")
    require(np.array_equal(token_table.is_first_token.to_numpy(bool), annotations.is_first_token.to_numpy(bool)), "First-token flag mismatch")
    require(np.isfinite(token_table[["reconstruction_mse", "cosine_similarity"]].to_numpy()).all(), "Nonfinite token metric")

    parquet_file = pq.ParquetFile(SAE_PATH)
    sae_rows = int(parquet_file.metadata.num_rows)
    schema_names = parquet_file.schema_arrow.names
    required_sae_columns = ["sample_id", "domain", "token_position", "token_id", "feature_id", "activation", "passed_threshold"]
    require(all(column in schema_names for column in required_sae_columns), f"Missing SAE columns: {schema_names}")
    require(sae_rows == EXPECTED_SAE_ROWS, f"SAE rows {sae_rows} != {EXPECTED_SAE_ROWS}")
    require(cache_manifest["sae_sparse"]["rows"] == EXPECTED_SAE_ROWS, "Cache manifest SAE count mismatch")
    require(cache_manifest["sae_sparse"]["feature_id_min"] == 0, "Unexpected minimum SAE feature ID")
    require(cache_manifest["sae_sparse"]["feature_id_max"] == EXPECTED_DICT_SIZE - 1, "Unexpected maximum SAE feature ID")
    require(cache_manifest["sae_sparse"]["duplicate_token_feature_keys"] == 0, "Duplicate SAE token-feature keys")
    require(cache_manifest["sae_sparse"]["nonfinite_activation_rows"] == 0, "Nonfinite SAE activations")
    require(cache_manifest["sae_sparse"]["passed_threshold_false_rows"] == 0, "SAE row failed threshold")
    require(sha256_file(SAE_PATH) == cache_manifest["outputs"][str(SAE_PATH)]["sha256"], "SAE cache SHA mismatch")

    active_counts = token_table.active_feature_count.to_numpy(np.int64)
    require(int(active_counts.sum()) == sae_rows, "active_feature_count sum does not equal SAE rows")
    primary_mask = annotations.mask_primary_inference.to_numpy(bool)
    all_mask = annotations.mask_all_tokens.to_numpy(bool)
    require(int(primary_mask.sum()) == N_PRIMARY and int(all_mask.sum()) == N_TOKENS, "Analysis-mask count mismatch")
    primary_sae_rows = int(active_counts[primary_mask].sum())
    first_token_sae_rows = int(active_counts[~primary_mask].sum())
    require(primary_sae_rows + first_token_sae_rows == sae_rows, "Primary/first-token SAE partition mismatch")

    token_lookup = token_table.set_index([token_table.sample_id.astype(str), token_table.token_position.astype(int)]).token_id
    inspected_row_groups = sorted(set([0, parquet_file.num_row_groups // 2, parquet_file.num_row_groups - 1]))
    sampled_alignment_rows = 0
    sampled_nonfinite = 0
    sampled_bad_threshold = 0
    sampled_feature_min = EXPECTED_DICT_SIZE
    sampled_feature_max = -1
    for row_group in inspected_row_groups:
        table = parquet_file.read_row_group(row_group, columns=required_sae_columns)
        frame = table.to_pandas()
        require(len(frame) > 0, f"Empty SAE row group {row_group}")
        indices = np.unique(np.linspace(0, len(frame) - 1, num=min(101, len(frame)), dtype=int))
        selected = frame.iloc[indices]
        for row in selected.itertuples(index=False):
            expected_token_id = token_lookup.loc[(str(row.sample_id), int(row.token_position))]
            require(int(row.token_id) == int(expected_token_id), "Sampled SAE/token key mismatch")
        sampled_alignment_rows += len(selected)
        sampled_nonfinite += int((~np.isfinite(selected.activation.to_numpy())).sum())
        sampled_bad_threshold += int((~selected.passed_threshold.astype(bool).to_numpy()).sum())
        sampled_feature_min = min(sampled_feature_min, int(selected.feature_id.min()))
        sampled_feature_max = max(sampled_feature_max, int(selected.feature_id.max()))
    require(sampled_nonfinite == 0 and sampled_bad_threshold == 0, "Sampled SAE quality check failed")

    expert_id = routing["expert_id"]
    layer11 = expert_id[:, ROUTING_LAYER, :].astype(np.int16, copy=False)
    require(layer11.shape == (N_TOKENS, TOP_K), "Layer-11 routing shape mismatch")
    duplicate_labels = int((np.sort(layer11, axis=1)[:, 1:] == np.sort(layer11, axis=1)[:, :-1]).any(axis=1).sum())
    require(duplicate_labels == 0, "Duplicate expert label within a token")
    all_expert_counts = np.bincount(layer11.ravel(), minlength=N_EXPERTS)
    primary_expert_counts = np.bincount(layer11[primary_mask].ravel(), minlength=N_EXPERTS)
    require(int(all_expert_counts.sum()) == N_TOKENS * TOP_K, "All-token layer-11 label count mismatch")
    require(int(primary_expert_counts.sum()) == N_PRIMARY * TOP_K, "Primary layer-11 label count mismatch")
    require((primary_expert_counts > 0).all(), "A layer-11 expert has no positive primary labels")

    audit_rows: list[dict[str, object]] = []
    add_audit(audit_rows, "prior_manifests", "00/01/02/03 complete", "all complete", "passed", "前四步清单均完成")
    add_audit(audit_rows, "samples", len(samples), N_SAMPLES, "passed", "样本数与冻结设计一致")
    add_audit(audit_rows, "tokens_all", int(all_mask.sum()), N_TOKENS, "passed", "全部Token审计总体完整")
    add_audit(audit_rows, "tokens_primary", int(primary_mask.sum()), N_PRIMARY, "passed", "主分析严格排除每个样本首Token")
    add_audit(audit_rows, "sae_sparse_rows", sae_rows, EXPECTED_SAE_ROWS, "passed", "SAE保持稀疏长表")
    add_audit(audit_rows, "sae_primary_sparse_rows", primary_sae_rows, "computed", "passed", "主分析实际使用的稀疏激活行数")
    add_audit(audit_rows, "sae_first_token_sparse_rows", first_token_sae_rows, "computed", "passed", "仅用于all-token质量审计")
    add_audit(audit_rows, "sae_feature_range", f"0..{EXPECTED_DICT_SIZE - 1}", f"0..{EXPECTED_DICT_SIZE - 1}", "passed", "SAE字典维度完整")
    add_audit(audit_rows, "sae_duplicate_token_feature_keys", 0, 0, "passed", "一个Token内没有重复特征键")
    add_audit(audit_rows, "sampled_sae_token_alignment", sampled_alignment_rows, ">0", "passed", "首、中、末行组抽样均与Token表对齐")
    add_audit(audit_rows, "layer11_target_shape", str(layer11.shape), f"({N_TOKENS}, {TOP_K})", "passed", "每个Token有4个第11层专家标签")
    add_audit(audit_rows, "layer11_duplicate_experts_per_token", duplicate_labels, 0, "passed", "同一Token的4个标签互不重复")
    add_audit(audit_rows, "layer11_primary_positive_count_range", f"{int(primary_expert_counts.min())}..{int(primary_expert_counts.max())}", ">0 for all 32 experts", "passed", "32个专家均有正例，但类别不平衡")
    add_audit(audit_rows, "missing_required_values", 0, 0, "passed", "必需字段没有缺失")
    add_audit(audit_rows, "nonfinite_required_values", 0, 0, "passed", "数值字段均为有限数")
    add_audit(audit_rows, "model_inference_rerun", "no", "no", "passed", "只读取已有缓存")
    audit = pd.DataFrame(audit_rows)
    require(audit.isna().sum().sum() == 0, "Missing audit value")
    atomic_csv(audit, AUDIT_PATH)

    contract = {
        "status": "input_validation_complete_method_choice_required",
        "step": "04A_routing_sae_input_and_contract",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "Whether layer-11 sparse SAE features have stable, reproducible associations with layer-11 Top-4 expert selection beyond domain and token-class baselines.",
        "fixed_inputs": {
            "model": "GPT-OSS-20B", "routing_layer": ROUTING_LAYER,
            "sae_hook": "resid_post_layer_11", "sae_trainer": "trainer_0",
            "sae_type": "BatchTopKSAE", "activation_dim": 2880,
            "dictionary_size": EXPECTED_DICT_SIZE, "sae_k_config": 64,
            "samples": N_SAMPLES, "tokens_all": N_TOKENS, "tokens_primary": N_PRIMARY,
            "sae_sparse_rows_all": sae_rows, "sae_sparse_rows_primary": primary_sae_rows,
            "sae_sparse_rows_first_tokens": first_token_sae_rows,
        },
        "fixed_safety_and_validity": {
            "primary_excludes_exactly_first_token_per_sample": True,
            "all_token_audit_retained": True,
            "cross_validation_group": "sample_id",
            "feature_selection_must_be_inside_training_fold": True,
            "sae_representation": "sparse only; dense token-by-131072 matrix prohibited",
            "first_sae_only": "resid_post_layer_11/trainer_0",
            "model_inference_rerun": False,
            "interpretation": "association/predictive increment only; not proof of functional specialization or causality",
        },
        "required_model_comparison": ["domain + token_class baseline", "SAE feature score", "combined"],
        "unresolved_method_choices": [
            "Prediction target: 32 binary expert-selected labels versus another Top-4 formulation.",
            "Estimator and regularization for baseline, SAE, and combined models.",
            "Number and construction of grouped cross-validation folds.",
            "Training-fold SAE feature-screening rule and retained feature budget.",
            "Primary evaluation metrics and bootstrap resampling unit/count.",
        ],
        "input_hashes": {
            str(SAE_PATH): sha256_file(SAE_PATH), str(ROUTING_PATH): sha256_file(ROUTING_PATH),
            str(ANNOTATION_PATH): sha256_file(ANNOTATION_PATH), str(SAMPLE_PATH): sha256_file(SAMPLE_PATH),
            str(STEP03_MANIFEST): sha256_file(STEP03_MANIFEST),
        },
        "validation": {
            "audit_rows": len(audit), "missing_values": 0, "nonfinite_values": 0,
            "sampled_sae_alignment_rows": sampled_alignment_rows,
            "sampled_feature_range": [sampled_feature_min, sampled_feature_max],
            "layer11_primary_positive_count_range": [int(primary_expert_counts.min()), int(primary_expert_counts.max())],
        },
        "script": script_record(SCRIPT),
        "outputs": {
            str(AUDIT_PATH): {"size_bytes": AUDIT_PATH.stat().st_size, "sha256": sha256_file(AUDIT_PATH)},
        },
        "raw_source_reparsed": False, "model_loaded": False, "model_inference_rerun": False, "dense_sae_matrix_created": False,
    }
    atomic_json(contract, CONTRACT_PATH)
    print(json.dumps({
        "status": contract["status"], "audit_rows": len(audit),
        "sae_rows_all": sae_rows, "sae_rows_primary": primary_sae_rows,
        "sae_rows_first_tokens": first_token_sae_rows,
        "layer11_primary_positive_count_range": [int(primary_expert_counts.min()), int(primary_expert_counts.max())],
        "sampled_sae_alignment_rows": sampled_alignment_rows,
        "unresolved_method_choices": contract["unresolved_method_choices"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
