#!/usr/bin/env python3
"""Step 03E: token-filter, token-class, prompt-segment, and length sensitivity checks."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _03_common import *

SCRIPT = Path(__file__).resolve()
LENGTH_MATCH_PATH = TABLES / "routing_length_matched_samples.csv"
STEP02_SUMMARY_PATH = METADATA / "routing_domain_analysis_summary.json"
AUDIT_TABLE = TABLES / "routing_path_control_audit.csv"
ADJACENT_TABLE = TABLES / "routing_adjacent_control_sensitivity.csv"
KEY_JSD_TABLE = TABLES / "routing_key_path_control_jsd.csv"
KEY_SUMMARY_TABLE = TABLES / "routing_key_path_control_summary.csv"
MANIFEST = METADATA / "routing_path_step03e_manifest.json"
FIGURE_STEM = "routing_path_control_sensitivity"

FILTERS = [
    ("A_all", "mask_primary_A_all"),
    ("B_no_special", "mask_primary_B_no_special"),
    ("C_no_special_whitespace", "mask_primary_C_no_special_whitespace"),
    ("D_content_only", "mask_primary_D_content_only"),
    ("E_no_mojibake", "mask_primary_E_no_mojibake"),
]
TOKEN_CLASSES = ["special", "mojibake", "whitespace", "punctuation", "number", "code_symbol", "content", "other"]
PROMPT_SEGMENTS = ["question", "option_text", "scaffold"]
KEY_TARGETS = [("transition", 10), ("transition", 11), ("unordered_combination", 11), ("unordered_combination", 12)]


def truncate_mask(base_mask: np.ndarray, token_sample_index: np.ndarray, limit: int) -> np.ndarray:
    valid_rows = np.flatnonzero(base_mask)
    sample_codes = token_sample_index[valid_rows]
    starts = np.r_[0, np.flatnonzero(np.diff(sample_codes)) + 1]
    ordinals = np.arange(len(valid_rows)) - np.repeat(starts, np.diff(np.r_[starts, len(valid_rows)]))
    result = np.zeros_like(base_mask, dtype=bool)
    result[valid_rows[ordinals < limit]] = True
    return result


def boolean_column(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(bool)
    normalized = series.astype(str).str.strip().str.lower()
    require(normalized.isin(["true", "false"]).all(), f"Invalid boolean values in {series.name}")
    return normalized.eq("true").to_numpy(bool)


def transition_code_rows(expert_id: np.ndarray, from_layer: int) -> np.ndarray:
    source = expert_id[:, from_layer, :].astype(np.int32, copy=False)
    target = expert_id[:, from_layer + 1, :].astype(np.int32, copy=False)
    return (source[:, :, None] * N_EXPERTS + target[:, None, :]).reshape(len(expert_id), TOP_K * TOP_K)


def unordered_code_rows(expert_id: np.ndarray, layer: int) -> np.ndarray:
    values = np.sort(expert_id[:, layer, :].astype(np.int32, copy=False), axis=1)
    codes = (((values[:, 0] * N_EXPERTS + values[:, 1]) * N_EXPERTS + values[:, 2]) * N_EXPERTS + values[:, 3])
    return codes[:, None]


def sample_equal_centroids(
    code_rows: np.ndarray,
    mask: np.ndarray,
    token_sample_index: np.ndarray,
    sample_domain_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_codes = code_rows[mask].reshape(-1)
    width = code_rows.shape[1]
    selected_samples = token_sample_index[mask]
    token_counts = np.bincount(selected_samples, minlength=N_SAMPLES).astype(np.int64)
    valid = token_counts > 0
    active_codes = np.unique(selected_codes)
    positions = np.searchsorted(active_codes, selected_codes)
    boundaries = np.concatenate(([0], np.cumsum(token_counts * width)))
    require(boundaries[-1] == len(positions), "Control distribution boundaries mismatch")
    centroids = np.zeros((len(DOMAINS), len(active_codes)), dtype=np.float64)
    n_by_domain = np.zeros(len(DOMAINS), dtype=np.int32)
    for sample_index in np.flatnonzero(valid):
        start, end = int(boundaries[sample_index]), int(boundaries[sample_index + 1])
        local = positions[start:end]
        unique_positions, counts = np.unique(local, return_counts=True)
        domain_code = int(sample_domain_code[sample_index])
        centroids[domain_code, unique_positions] += counts / len(local)
        n_by_domain[domain_code] += 1
    for domain_code in range(len(DOMAINS)):
        if n_by_domain[domain_code] > 0:
            centroids[domain_code] /= n_by_domain[domain_code]
            require(abs(float(centroids[domain_code].sum()) - 1.0) < 1e-12, "Control centroid is not normalized")
    return centroids, n_by_domain, token_counts


def add_adjacent_rows(
    rows: list[dict[str, object]],
    control_type: str,
    control_level: str,
    control_scope: str,
    mask: np.ndarray,
    retained: np.ndarray,
    jaccard: np.ndarray,
    token_sample_index: np.ndarray,
    sample_domain_code: np.ndarray,
) -> None:
    selected_samples = token_sample_index[mask]
    token_counts = np.bincount(selected_samples, minlength=N_SAMPLES).astype(np.int64)
    valid = token_counts > 0
    token_domain_code = sample_domain_code[token_sample_index]
    for layer in range(N_LAYERS - 1):
        jac_sum = np.bincount(selected_samples, weights=jaccard[mask, layer], minlength=N_SAMPLES)
        ret_sum = np.bincount(selected_samples, weights=retained[mask, layer], minlength=N_SAMPLES)
        sample_jac = np.divide(jac_sum, token_counts, out=np.zeros(N_SAMPLES), where=valid)
        sample_ret = np.divide(ret_sum, token_counts, out=np.zeros(N_SAMPLES), where=valid)
        for domain_code, domain in enumerate(DOMAINS):
            domain_token_mask = mask & (token_domain_code == domain_code)
            domain_sample_mask = valid & (sample_domain_code == domain_code)
            require(domain_token_mask.any() and domain_sample_mask.any(), f"Missing domain for control {control_type}/{control_level}/{domain}")
            for weighting, mean_jac, mean_ret in (
                ("token_weighted", float(jaccard[domain_token_mask, layer].mean()), float(retained[domain_token_mask, layer].mean())),
                ("sample_equal", float(sample_jac[domain_sample_mask].mean()), float(sample_ret[domain_sample_mask].mean())),
            ):
                rows.append({
                    "control_type": control_type, "control_level": control_level, "control_scope": control_scope,
                    "weighting": weighting, "domain": domain, "from_layer": layer, "to_layer": layer + 1,
                    "n_tokens": int(domain_token_mask.sum()), "n_samples": int(domain_sample_mask.sum()),
                    "mean_jaccard": mean_jac, "mean_retained_count": mean_ret,
                })


def self_test() -> None:
    base = np.array([True, True, True, False, True, True, True], dtype=bool)
    sample_index = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    truncated = truncate_mask(base, sample_index, 2)
    require(truncated.tolist() == [True, True, False, False, True, True, False], "Synthetic truncation failed")
    expert = np.zeros((1, N_LAYERS, TOP_K), dtype=np.uint16)
    expert[0, 10] = [0, 1, 2, 3]; expert[0, 11] = [4, 5, 6, 7]
    require(len(np.unique(transition_code_rows(expert, 10))) == 16, "Synthetic transition code rows failed")
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    ensure_dirs()
    require(LENGTH_MATCH_PATH.is_file() and STEP02_SUMMARY_PATH.is_file(), "Missing frozen step-02 length-control inputs")
    step02_manifest = json.loads(STEP02_MANIFEST.read_text(encoding="utf-8"))
    for path in (LENGTH_MATCH_PATH, STEP02_SUMMARY_PATH):
        require(str(path) in step02_manifest["outputs"], f"Step-02 manifest does not list {path}")
        require(sha256_file(path) == step02_manifest["outputs"][str(path)]["sha256"], f"Step-02 control input hash mismatch: {path}")

    annotation_columns = [
        "sample_id", "token_position", "token_class", "prompt_segment",
        *[column for _, column in FILTERS],
    ]
    samples, annotations, routing, _ = load_inputs(annotation_columns)
    expert_id = routing["expert_id"]
    token_sample_index = routing["token_sample_index"].astype(np.int32, copy=False)
    require(np.all(token_sample_index[:-1] <= token_sample_index[1:]), "Tokens are not grouped by sample")
    domain_codes = sample_domain_codes(samples)
    primary = annotations["mask_primary_A_all"].to_numpy(bool)
    base_length = annotations["mask_primary_C_no_special_whitespace"].to_numpy(bool)

    length_match = pd.read_csv(LENGTH_MATCH_PATH)
    require(len(length_match) == N_SAMPLES and length_match.sample_id.astype(str).tolist() == samples.sample_id.astype(str).tolist(), "Length-control sample alignment mismatch")
    step02_summary = json.loads(STEP02_SUMMARY_PATH.read_text(encoding="utf-8"))
    truncation_n = int(step02_summary["length_control"]["fixed_truncation_n"])
    require(truncation_n == 50, "Frozen fixed truncation is not N=50")
    matched = boolean_column(length_match.selected_for_pooled_tercile_match)
    terciles = length_match.domain_internal_tercile.to_numpy(np.int8)
    require(int(matched.sum()) == int(step02_summary["length_control"]["matched_samples"]) == 930, "Frozen matched-sample count mismatch")

    controls: list[tuple[str, str, str, np.ndarray]] = []
    for level, column in FILTERS:
        controls.append(("token_filter", level, "robustness", annotations[column].to_numpy(bool)))
    token_class_array = annotations.token_class.astype(str).to_numpy()
    for token_class in TOKEN_CLASSES:
        controls.append(("token_class", token_class, "diagnostic", primary & (token_class_array == token_class)))
    segment_array = annotations.prompt_segment.astype(str).to_numpy()
    for segment in PROMPT_SEGMENTS:
        controls.append(("prompt_segment", segment, "robustness", primary & (segment_array == segment)))
    length_controls = [
        ("original_full_length", base_length),
        ("pooled_tercile_frequency_matched", base_length & matched[token_sample_index]),
        (f"fixed_token_truncation_n{truncation_n}", truncate_mask(base_length, token_sample_index, truncation_n)),
        ("domain_internal_tercile_low", base_length & (terciles[token_sample_index] == 0)),
        ("domain_internal_tercile_medium", base_length & (terciles[token_sample_index] == 1)),
        ("domain_internal_tercile_high", base_length & (terciles[token_sample_index] == 2)),
    ]
    controls.extend(("length", level, "robustness", mask) for level, mask in length_controls)
    require(len(controls) == 22, "Unexpected control-setting count")

    retained, jaccard = adjacent_metrics(expert_id)
    code_rows = {
        ("transition", 10): transition_code_rows(expert_id, 10),
        ("transition", 11): transition_code_rows(expert_id, 11),
        ("unordered_combination", 11): unordered_code_rows(expert_id, 11),
        ("unordered_combination", 12): unordered_code_rows(expert_id, 12),
    }
    audit_rows: list[dict[str, object]] = []
    adjacent_rows: list[dict[str, object]] = []
    jsd_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for control_type, control_level, control_scope, mask in controls:
        token_counts = np.bincount(token_sample_index[mask], minlength=N_SAMPLES)
        valid = token_counts > 0
        n_by_domain = np.array([int(np.sum(valid & (domain_codes == code))) for code in range(len(DOMAINS))])
        domains_present = int(np.sum(n_by_domain > 0))
        status = "no_tokens" if not mask.any() else ("ok_all_domains" if domains_present == len(DOMAINS) else "incomplete_domains")
        audit_rows.append({
            "control_type": control_type, "control_level": control_level, "control_scope": control_scope,
            "selected_token_count": int(mask.sum()), "valid_sample_count": int(valid.sum()),
            "domains_present": domains_present,
            "minimum_valid_samples_per_present_domain": int(n_by_domain[n_by_domain > 0].min()) if domains_present else 0,
            "maximum_valid_samples_per_present_domain": int(n_by_domain.max()) if domains_present else 0,
            "status": status,
        })
        if status != "ok_all_domains":
            continue
        add_adjacent_rows(
            adjacent_rows, control_type, control_level, control_scope, mask,
            retained, jaccard, token_sample_index, domain_codes,
        )
        for target_type, layer in KEY_TARGETS:
            centroids, target_samples_by_domain, target_token_counts = sample_equal_centroids(
                code_rows[(target_type, layer)], mask, token_sample_index, domain_codes
            )
            require((target_samples_by_domain > 0).all(), "Key target lost a domain")
            pair_values = []
            for left_code, right_code in combinations(range(len(DOMAINS)), 2):
                value = js_divergence_bits(centroids[left_code], centroids[right_code])
                pair_values.append(value)
                jsd_rows.append({
                    "control_type": control_type, "control_level": control_level, "control_scope": control_scope,
                    "target_type": target_type, "layer": layer,
                    "domain_a": DOMAINS[left_code], "domain_b": DOMAINS[right_code], "jsd_bits": value,
                    "n_samples_a": int(target_samples_by_domain[left_code]), "n_samples_b": int(target_samples_by_domain[right_code]),
                })
            summary_rows.append({
                "control_type": control_type, "control_level": control_level, "control_scope": control_scope,
                "target_type": target_type, "layer": layer,
                "selected_token_count": int(target_token_counts.sum()), "valid_sample_count": int((target_token_counts > 0).sum()),
                "minimum_valid_samples_per_domain": int(target_samples_by_domain.min()),
                "mean_pairwise_jsd_bits": float(np.mean(pair_values)),
                "max_pairwise_jsd_bits": float(np.max(pair_values)),
                "min_pairwise_jsd_bits": float(np.min(pair_values)),
            })

    audit = pd.DataFrame(audit_rows)
    adjacent = pd.DataFrame(adjacent_rows)
    key_jsd = pd.DataFrame(jsd_rows)
    key_summary = pd.DataFrame(summary_rows)
    valid_controls = int(audit.status.eq("ok_all_domains").sum())
    require(valid_controls == 21, f"Expected 21 analyzable controls, found {valid_controls}")
    require(len(adjacent) == valid_controls * 2 * len(DOMAINS) * (N_LAYERS - 1), "Unexpected adjacent-control rows")
    require(len(key_jsd) == valid_controls * len(KEY_TARGETS) * 10, "Unexpected key-control JSD rows")
    require(len(key_summary) == valid_controls * len(KEY_TARGETS), "Unexpected key-control summary rows")
    for frame, name in ((audit, "audit"), (adjacent, "adjacent"), (key_jsd, "key_jsd"), (key_summary, "key_summary")):
        require(frame.isna().sum().sum() == 0, f"Missing values in {name}")
        require(frame.select_dtypes(include=[np.number]).apply(np.isfinite).all().all(), f"Nonfinite values in {name}")
    duplicate_a_b = adjacent[
        (adjacent.control_type == "token_filter") & adjacent.control_level.isin(["A_all", "B_no_special"])
    ].pivot_table(index=["weighting", "domain", "from_layer"], columns="control_level", values="mean_jaccard")
    require(np.array_equal(duplicate_a_b.A_all.to_numpy(), duplicate_a_b.B_no_special.to_numpy()), "A/B controls should be identical because special count is zero")

    atomic_csv(audit, AUDIT_TABLE)
    atomic_csv(adjacent, ADJACENT_TABLE)
    atomic_csv(key_jsd, KEY_JSD_TABLE)
    atomic_csv(key_summary, KEY_SUMMARY_TABLE)

    plot = key_summary.copy()
    plot["label"] = plot.control_type + ": " + plot.control_level
    target_labels = ["transition L10→11", "transition L11→12", "unordered L11", "unordered L12"]
    ordered_targets = [("transition", 10), ("transition", 11), ("unordered_combination", 11), ("unordered_combination", 12)]
    ordered_controls = audit[audit.status == "ok_all_domains"].assign(label=lambda frame: frame.control_type + ": " + frame.control_level).label.tolist()
    matrix = np.zeros((len(ordered_controls), len(ordered_targets)), dtype=float)
    for row, label in enumerate(ordered_controls):
        for column, (target_type, layer) in enumerate(ordered_targets):
            selected = plot[(plot.label == label) & (plot.target_type == target_type) & (plot.layer == layer)]
            require(len(selected) == 1, "Control heatmap cell is not unique")
            matrix[row, column] = float(selected.mean_pairwise_jsd_bits.iloc[0])
    figure, axis = plt.subplots(figsize=(9.5, 10.0))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(target_labels)), target_labels, rotation=20, ha="right")
    axis.set_yticks(range(len(ordered_controls)), ordered_controls, fontsize=7.5)
    axis.set_title("Routing-path control sensitivity: mean pairwise domain JSD")
    figure.colorbar(image, ax=axis, label="Mean pairwise JSD (bits)")
    figure.tight_layout()
    png = FIG_PNG / f"{FIGURE_STEM}.png"
    svg = FIG_SVG / f"{FIGURE_STEM}.svg"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)

    outputs = [AUDIT_TABLE, ADJACENT_TABLE, KEY_JSD_TABLE, KEY_SUMMARY_TABLE, png, svg]
    robust = key_summary[key_summary.control_scope == "robustness"]
    diagnostic = key_summary[key_summary.control_scope == "diagnostic"]
    manifest = {
        "status": "complete", "step": "03E_path_controls", "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "robustness_controls": "5 token filters + 3 prompt segments + 6 frozen length controls",
            "diagnostic_controls": "8 individual token classes; special has zero tokens and is audit-only",
            "length_match_reused": True, "matched_samples": 930, "fixed_truncation_n": truncation_n,
            "sparse_class_warning": "Individual token-class results are diagnostic and retain actual valid-sample counts.",
        },
        "script": script_record(SCRIPT), "shared_script": script_record(Path(__file__).with_name("_03_common.py")),
        "inputs": {
            str(LENGTH_MATCH_PATH): {"size_bytes": LENGTH_MATCH_PATH.stat().st_size, "sha256": sha256_file(LENGTH_MATCH_PATH)},
            str(STEP02_SUMMARY_PATH): {"size_bytes": STEP02_SUMMARY_PATH.stat().st_size, "sha256": sha256_file(STEP02_SUMMARY_PATH)},
        },
        "outputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs},
        "row_counts": {str(AUDIT_TABLE): len(audit), str(ADJACENT_TABLE): len(adjacent), str(KEY_JSD_TABLE): len(key_jsd), str(KEY_SUMMARY_TABLE): len(key_summary)},
        "validation": {
            "missing_values": 0, "nonfinite_numeric": 0, "control_settings": len(audit), "analyzable_settings": valid_controls,
            "declared_empty_settings": int(audit.status.eq("no_tokens").sum()), "A_equals_B_when_special_is_empty": True,
            "robust_mean_jsd_range": [float(robust.mean_pairwise_jsd_bits.min()), float(robust.mean_pairwise_jsd_bits.max())],
            "diagnostic_mean_jsd_range": [float(diagnostic.mean_pairwise_jsd_bits.min()), float(diagnostic.mean_pairwise_jsd_bits.max())],
        },
        "raw_parquet_read": False, "model_loaded": False, "model_inference_rerun": False, "dense_sae_matrix_created": False,
    }
    atomic_json(manifest, MANIFEST)

    print("CONTROL_AUDIT")
    print(audit.to_string(index=False))
    print("ROBUST_KEY_RANGES")
    print(robust.groupby(["target_type", "layer"])["mean_pairwise_jsd_bits"].agg(["min", "mean", "max"]).to_string())
    print("FOCUS_ADJACENT_ROBUST_RANGES")
    focus_adjacent = adjacent[(adjacent.control_scope == "robustness") & (adjacent.weighting == "sample_equal") & adjacent.from_layer.isin([10, 11])]
    print(focus_adjacent.groupby(["from_layer", "domain"]).mean_jaccard.agg(["min", "max"]).to_string())


if __name__ == "__main__":
    main()
