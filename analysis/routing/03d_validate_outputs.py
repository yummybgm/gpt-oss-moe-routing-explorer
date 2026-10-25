#!/usr/bin/env python3
"""Independent, read-only validation of step-03D outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("MOE_SAE_ANALYSIS_ROOT", Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1")).expanduser().resolve()
TOP_PATH = ROOT / "tables/routing_top4_combination_top50.csv"
SUMMARY_PATH = ROOT / "tables/routing_top4_combination_summary.csv"
JSD_PATH = ROOT / "tables/routing_top4_combination_domain_jsd.csv"
MANIFEST_PATH = ROOT / "metadata/routing_path_step03d_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def encode(values: np.ndarray, combination_type: str) -> np.ndarray:
    array = values.astype(np.int32)
    if combination_type == "unordered":
        array = np.sort(array, axis=1)
    return (((array[:, 0] * 32 + array[:, 1]) * 32 + array[:, 2]) * 32 + array[:, 3])


def direct_distribution(
    expert: np.ndarray,
    token_sample_index: np.ndarray,
    primary: np.ndarray,
    domain_sample_indices: np.ndarray,
    layer: int,
    combination_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    domain_lookup = np.zeros(2000, dtype=bool)
    domain_lookup[domain_sample_indices] = True
    domain_mask = primary & domain_lookup[token_sample_index]
    domain_codes = encode(expert[domain_mask, layer, :], combination_type)
    active = np.unique(domain_codes)
    raw = np.bincount(np.searchsorted(active, domain_codes), minlength=len(active))
    sample_sum = np.zeros(len(active), dtype=np.float64)
    for sample_index in domain_sample_indices:
        sample_mask = primary & (token_sample_index == sample_index)
        sample_codes = encode(expert[sample_mask, layer, :], combination_type)
        positions = np.searchsorted(active, sample_codes)
        unique_positions, counts = np.unique(positions, return_counts=True)
        sample_sum[unique_positions] += counts / len(sample_codes)
    return active, raw / raw.sum(), sample_sum / len(domain_sample_indices)


def entropy_bits(probability: np.ndarray) -> float:
    positive = probability > 0
    return float(-(probability[positive] * np.log2(probability[positive])).sum())


def main() -> None:
    top = pd.read_csv(TOP_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    jsd = pd.read_csv(JSD_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for raw_path, metadata in manifest["outputs"].items():
        path = Path(raw_path)
        assert path.is_file()
        assert path.stat().st_size == metadata["size_bytes"]
        assert sha256_file(path) == metadata["sha256"]
    assert (len(top), len(summary), len(jsd)) == (48_960, 960, 1_920)
    for frame in (top, summary, jsd):
        assert not frame.isna().any().any()
        assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
    keys = ["population", "weighting", "domain", "layer", "combination_type"]
    probability_sums = top.groupby(keys).probability.sum()
    assert np.allclose(probability_sums.to_numpy(), 1.0, atol=1e-12)
    assert (top.probability >= -1e-12).all()
    non_other = top[top.combination != "__OTHER__"]
    monotonic = non_other.groupby(keys).probability.apply(lambda values: values.is_monotonic_decreasing)
    assert monotonic.all()
    unordered = summary[summary.combination_type == "unordered"].set_index(["population", "weighting", "domain", "layer"])
    ordered = summary[summary.combination_type == "ordered"].set_index(["population", "weighting", "domain", "layer"])
    assert (unordered.unique_combinations <= ordered.unique_combinations).all()
    unordered_rows = non_other[non_other.combination_type == "unordered"]
    assert unordered_rows[["expert_0", "expert_1", "expert_2", "expert_3"]].apply(
        lambda row: row.tolist() == sorted(row.tolist()), axis=1
    ).all()

    samples = pq.read_table(ROOT / "cache/sample_table.parquet", columns=["domain", "sample_order"]).to_pandas()
    samples = samples.sort_values("sample_order", kind="stable").reset_index(drop=True)
    annotations = pq.read_table(ROOT / "cache/token_annotations.parquet", columns=["mask_primary_inference"]).to_pandas()
    with np.load(ROOT / "cache/routing_top4.npz", allow_pickle=False) as source:
        expert = source["expert_id"]
        token_sample_index = source["token_sample_index"].astype(np.int32)
    primary = annotations.mask_primary_inference.to_numpy(bool)

    checks = []
    cases = [
        ("math", 11, "ordered"),
        ("law", 11, "unordered"),
        ("computer science", 12, "ordered"),
        ("psychology", 12, "unordered"),
    ]
    for domain, layer, combination_type in cases:
        domain_indices = np.flatnonzero(samples.domain.eq(domain).to_numpy())
        active, token_probability, sample_probability = direct_distribution(
            expert, token_sample_index, primary, domain_indices, layer, combination_type
        )
        for weighting, probability in (("token_weighted", token_probability), ("sample_equal", sample_probability)):
            order = np.argsort(-probability, kind="stable")
            expected_codes = active[order[:50]]
            expected_probability = probability[order[:50]]
            reported = top[
                (top.population == "primary_inference")
                & (top.weighting == weighting)
                & (top.domain == domain)
                & (top.layer == layer)
                & (top.combination_type == combination_type)
                & (top.combination != "__OTHER__")
            ].sort_values("rank", kind="stable")
            assert np.array_equal(reported.combination_code.to_numpy(), expected_codes)
            max_error = float(np.max(np.abs(reported.probability.to_numpy() - expected_probability)))
            assert max_error < 1e-12
            reported_summary = summary[
                (summary.population == "primary_inference")
                & (summary.weighting == weighting)
                & (summary.domain == domain)
                & (summary.layer == layer)
                & (summary.combination_type == combination_type)
            ].iloc[0]
            assert int(reported_summary.unique_combinations) == len(active)
            assert abs(float(reported_summary.entropy_bits) - entropy_bits(probability)) < 1e-12
            checks.append({
                "domain": domain, "layer": layer, "combination_type": combination_type,
                "weighting": weighting, "top50_max_abs_error": max_error,
            })

    focus = summary[(summary.population == "primary_inference") & (summary.weighting == "sample_equal") & summary.layer.isin([11, 12])]
    focus_jsd = jsd[(jsd.population == "primary_inference") & (jsd.weighting == "sample_equal") & jsd.layer.isin([11, 12])]
    print(json.dumps({
        "status": "passed",
        "row_counts": {"top50_plus_other": len(top), "summary": len(summary), "jsd": len(jsd)},
        "missing_values": 0,
        "manifest_output_hashes": "all_match",
        "probability_sum_max_error": float(np.max(np.abs(probability_sums.to_numpy() - 1))),
        "direct_checks": checks,
        "focus_unique_range": [int(focus.unique_combinations.min()), int(focus.unique_combinations.max())],
        "focus_effective_count_range": [float(focus.effective_combination_count.min()), float(focus.effective_combination_count.max())],
        "focus_top50_mass_range": [float(focus.top50_mass.min()), float(focus.top50_mass.max())],
        "focus_jsd_range_bits": [float(focus_jsd.jsd_bits.min()), float(focus_jsd.jsd_bits.max())],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
