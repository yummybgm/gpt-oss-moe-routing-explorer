#!/usr/bin/env python3
"""Independent, read-only validation of step-03E control outputs."""
from __future__ import annotations

import hashlib
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("MOE_SAE_ANALYSIS_ROOT", Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1")).expanduser().resolve()
AUDIT_PATH = ROOT / "tables/routing_path_control_audit.csv"
ADJACENT_PATH = ROOT / "tables/routing_adjacent_control_sensitivity.csv"
JSD_PATH = ROOT / "tables/routing_key_path_control_jsd.csv"
SUMMARY_PATH = ROOT / "tables/routing_key_path_control_summary.csv"
MANIFEST_PATH = ROOT / "metadata/routing_path_step03e_manifest.json"
LENGTH_MATCH_PATH = ROOT / "tables/routing_length_matched_samples.csv"
DOMAINS = ["math", "computer science", "physics", "law", "psychology"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsd_bits(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2
    value = 0.0
    for distribution in (left, right):
        positive = distribution > 0
        value += 0.5 * float(np.sum(distribution[positive] * np.log2(distribution[positive] / midpoint[positive])))
    return value


def direct_adjacent(
    expert: np.ndarray,
    token_sample_index: np.ndarray,
    sample_domains: np.ndarray,
    mask: np.ndarray,
    layer: int,
) -> dict[tuple[str, str], float]:
    left = expert[:, layer, :]
    right = expert[:, layer + 1, :]
    retained = (left[:, :, None] == right[:, None, :]).any(axis=2).sum(axis=1)
    jaccard = retained / (8.0 - retained)
    token_counts = np.bincount(token_sample_index[mask], minlength=2000)
    sums = np.bincount(token_sample_index[mask], weights=jaccard[mask], minlength=2000)
    valid = token_counts > 0
    sample_means = np.divide(sums, token_counts, out=np.zeros(2000), where=valid)
    token_domains = sample_domains[token_sample_index]
    result = {}
    for domain in DOMAINS:
        token_domain = mask & (token_domains == domain)
        sample_domain = valid & (sample_domains == domain)
        result[(domain, "token_weighted")] = float(jaccard[token_domain].mean())
        result[(domain, "sample_equal")] = float(sample_means[sample_domain].mean())
    return result


def code_rows(expert: np.ndarray, target_type: str, layer: int) -> np.ndarray:
    if target_type == "transition":
        source = expert[:, layer, :].astype(np.int32)
        target = expert[:, layer + 1, :].astype(np.int32)
        return (source[:, :, None] * 32 + target[:, None, :]).reshape(len(expert), 16)
    values = np.sort(expert[:, layer, :].astype(np.int32), axis=1)
    codes = (((values[:, 0] * 32 + values[:, 1]) * 32 + values[:, 2]) * 32 + values[:, 3])
    return codes[:, None]


def direct_centroids(
    rows: np.ndarray,
    mask: np.ndarray,
    token_sample_index: np.ndarray,
    sample_domains: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    selected_codes = rows[mask].reshape(-1)
    selected_samples = token_sample_index[mask]
    width = rows.shape[1]
    token_counts = np.bincount(selected_samples, minlength=2000)
    boundaries = np.concatenate(([0], np.cumsum(token_counts * width)))
    active = np.unique(selected_codes)
    positions = np.searchsorted(active, selected_codes)
    domain_sum = [np.zeros(len(active), dtype=float) for _ in DOMAINS]
    n_by_domain = np.zeros(len(DOMAINS), dtype=int)
    domain_to_code = {domain: code for code, domain in enumerate(DOMAINS)}
    for sample_index in np.flatnonzero(token_counts > 0):
        start, end = int(boundaries[sample_index]), int(boundaries[sample_index + 1])
        unique_positions, counts = np.unique(positions[start:end], return_counts=True)
        domain_code = domain_to_code[str(sample_domains[sample_index])]
        domain_sum[domain_code][unique_positions] += counts / (end - start)
        n_by_domain[domain_code] += 1
    return [domain_sum[code] / n_by_domain[code] for code in range(len(DOMAINS))], n_by_domain


def main() -> None:
    audit = pd.read_csv(AUDIT_PATH)
    adjacent = pd.read_csv(ADJACENT_PATH)
    key_jsd = pd.read_csv(JSD_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for raw_path, metadata in manifest["outputs"].items():
        path = Path(raw_path)
        assert path.is_file()
        assert path.stat().st_size == metadata["size_bytes"]
        assert sha256_file(path) == metadata["sha256"]
    assert (len(audit), len(adjacent), len(key_jsd), len(summary)) == (22, 4_830, 840, 84)
    for frame in (audit, adjacent, key_jsd, summary):
        assert not frame.isna().any().any()
        assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
    special = audit[(audit.control_type == "token_class") & (audit.control_level == "special")].iloc[0]
    assert special.status == "no_tokens" and int(special.selected_token_count) == 0
    assert int(audit.status.eq("ok_all_domains").sum()) == 21
    assert int(audit.control_scope.eq("robustness").sum()) == 14
    assert int(audit.control_scope.eq("diagnostic").sum()) == 8

    group = ["control_type", "control_level", "control_scope", "target_type", "layer"]
    recomputed = key_jsd.groupby(group).jsd_bits.agg(["min", "mean", "max"]).reset_index()
    merged = summary.merge(recomputed, on=group, validate="one_to_one")
    assert np.allclose(merged.mean_pairwise_jsd_bits, merged["mean"], atol=1e-12)
    assert np.allclose(merged.min_pairwise_jsd_bits, merged["min"], atol=1e-12)
    assert np.allclose(merged.max_pairwise_jsd_bits, merged["max"], atol=1e-12)
    for frame, value_column in ((adjacent, "mean_jaccard"), (key_jsd, "jsd_bits"), (summary, "mean_pairwise_jsd_bits")):
        left = frame[(frame.control_type == "token_filter") & (frame.control_level == "A_all")].reset_index(drop=True)
        right = frame[(frame.control_type == "token_filter") & (frame.control_level == "B_no_special")].reset_index(drop=True)
        assert np.array_equal(left[value_column].to_numpy(), right[value_column].to_numpy())

    samples = pq.read_table(ROOT / "cache/sample_table.parquet", columns=["domain", "sample_order", "sample_id"]).to_pandas()
    samples = samples.sort_values("sample_order", kind="stable").reset_index(drop=True)
    annotations = pq.read_table(
        ROOT / "cache/token_annotations.parquet",
        columns=["mask_primary_A_all", "mask_primary_D_content_only", "mask_primary_C_no_special_whitespace", "prompt_segment", "token_class"],
    ).to_pandas()
    with np.load(ROOT / "cache/routing_top4.npz", allow_pickle=False) as source:
        expert = source["expert_id"]
        token_sample_index = source["token_sample_index"].astype(np.int32)
    sample_domains = samples.domain.astype(str).to_numpy()
    length_match = pd.read_csv(LENGTH_MATCH_PATH)
    matched = length_match.selected_for_pooled_tercile_match.to_numpy()
    if matched.dtype != bool:
        matched = pd.Series(matched).astype(str).str.lower().eq("true").to_numpy()
    masks = {
        ("token_filter", "A_all"): annotations.mask_primary_A_all.to_numpy(bool),
        ("prompt_segment", "option_text"): annotations.mask_primary_A_all.to_numpy(bool) & annotations.prompt_segment.astype(str).eq("option_text").to_numpy(),
        ("length", "pooled_tercile_frequency_matched"): annotations.mask_primary_C_no_special_whitespace.to_numpy(bool) & matched[token_sample_index],
        ("token_class", "other"): annotations.mask_primary_A_all.to_numpy(bool) & annotations.token_class.astype(str).eq("other").to_numpy(),
    }

    adjacent_checks = []
    for (control_type, control_level), mask in masks.items():
        layer = 10 if control_level != "other" else 11
        direct = direct_adjacent(expert, token_sample_index, sample_domains, mask, layer)
        for domain in DOMAINS:
            for weighting in ("token_weighted", "sample_equal"):
                reported = float(adjacent.loc[
                    (adjacent.control_type == control_type)
                    & (adjacent.control_level == control_level)
                    & (adjacent.domain == domain)
                    & (adjacent.weighting == weighting)
                    & (adjacent.from_layer == layer), "mean_jaccard"
                ].iloc[0])
                assert abs(direct[(domain, weighting)] - reported) < 1e-12
        adjacent_checks.append({"control_type": control_type, "control_level": control_level, "from_layer": layer})

    jsd_checks = []
    target_cases = [
        ("token_filter", "D_content_only", "transition", 10, annotations.mask_primary_D_content_only.to_numpy(bool)),
        ("length", "pooled_tercile_frequency_matched", "unordered_combination", 11, masks[("length", "pooled_tercile_frequency_matched")]),
    ]
    for control_type, control_level, target_type, layer, mask in target_cases:
        centroids, n_by_domain = direct_centroids(code_rows(expert, target_type, layer), mask, token_sample_index, sample_domains)
        assert (n_by_domain > 0).all()
        for left_code, right_code in combinations(range(len(DOMAINS)), 2):
            direct_value = jsd_bits(centroids[left_code], centroids[right_code])
            reported = float(key_jsd.loc[
                (key_jsd.control_type == control_type)
                & (key_jsd.control_level == control_level)
                & (key_jsd.target_type == target_type)
                & (key_jsd.layer == layer)
                & (key_jsd.domain_a == DOMAINS[left_code])
                & (key_jsd.domain_b == DOMAINS[right_code]), "jsd_bits"
            ].iloc[0])
            assert abs(direct_value - reported) < 1e-12
        jsd_checks.append({"control_type": control_type, "control_level": control_level, "target_type": target_type, "layer": layer})

    robust = summary[summary.control_scope == "robustness"]
    diagnostic_audit = audit[(audit.control_scope == "diagnostic") & (audit.status == "ok_all_domains")]
    print(json.dumps({
        "status": "passed",
        "row_counts": {"audit": len(audit), "adjacent": len(adjacent), "key_jsd": len(key_jsd), "key_summary": len(summary)},
        "missing_values": 0,
        "manifest_output_hashes": "all_match",
        "adjacent_direct_checks": adjacent_checks,
        "jsd_direct_checks": jsd_checks,
        "robust_mean_jsd_range": [float(robust.mean_pairwise_jsd_bits.min()), float(robust.mean_pairwise_jsd_bits.max())],
        "diagnostic_min_samples_per_domain_range": [int(diagnostic_audit.minimum_valid_samples_per_present_domain.min()), int(diagnostic_audit.minimum_valid_samples_per_present_domain.max())],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
