#!/usr/bin/env python3
"""Independent, read-only validation of step-03C outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("MOE_SAE_ANALYSIS_ROOT", Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1")).expanduser().resolve()
MATRIX_PATH = ROOT / "tables/routing_focused_transition_matrices.csv"
SUMMARY_PATH = ROOT / "tables/routing_focused_transition_summary.csv"
TOP_PATH = ROOT / "tables/routing_focused_transition_top_pairs.csv"
JSD_PATH = ROOT / "tables/routing_focused_transition_domain_jsd.csv"
MANIFEST_PATH = ROOT / "metadata/routing_path_step03c_manifest.json"
DOMAINS = ["math", "computer science", "physics", "law", "psychology"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsd_bits(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2
    result = 0.0
    for distribution in (left, right):
        positive = distribution > 0
        result += 0.5 * float(np.sum(distribution[positive] * np.log2(distribution[positive] / midpoint[positive])))
    return result


def direct_distribution(
    expert: np.ndarray,
    token_sample_index: np.ndarray,
    token_mask: np.ndarray,
    domain_sample_indices: np.ndarray,
    from_layer: int,
    to_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    token_counts = []
    sample_distributions = []
    raw = np.zeros(1024, dtype=np.int64)
    domain_lookup = np.zeros(2000, dtype=bool)
    domain_lookup[domain_sample_indices] = True
    domain_token_mask = token_mask & domain_lookup[token_sample_index]
    for sample_index in domain_sample_indices:
        selected = domain_token_mask & (token_sample_index == sample_index)
        count = int(selected.sum())
        assert count > 0
        source = expert[selected, from_layer, :].astype(np.int32)
        target = expert[selected, to_layer, :].astype(np.int32)
        codes = (source[:, :, None] * 32 + target[:, None, :]).ravel()
        sample_counts = np.bincount(codes, minlength=1024)
        assert sample_counts.sum() == count * 16
        raw += sample_counts
        sample_distributions.append(sample_counts / (count * 16))
        token_counts.append(count)
    return raw / raw.sum(), np.mean(sample_distributions, axis=0)


def reported_distribution(matrix: pd.DataFrame, domain: str, from_layer: int, weighting: str) -> np.ndarray:
    selected = matrix[
        (matrix.population == "primary_inference")
        & (matrix.weighting == weighting)
        & (matrix.domain == domain)
        & (matrix.from_layer == from_layer)
    ].sort_values(["source_expert", "target_expert"], kind="stable")
    assert len(selected) == 1024
    return selected.probability.to_numpy()


def main() -> None:
    matrix = pd.read_csv(MATRIX_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    top = pd.read_csv(TOP_PATH)
    jsd = pd.read_csv(JSD_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for raw_path, metadata in manifest["outputs"].items():
        path = Path(raw_path)
        assert path.is_file()
        assert path.stat().st_size == metadata["size_bytes"]
        assert sha256_file(path) == metadata["sha256"]
    assert (len(matrix), len(summary), len(top), len(jsd)) == (40960, 40, 800, 80)
    for frame in (matrix, summary, top, jsd):
        assert not frame.isna().any().any()
        assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
    group_keys = ["population", "weighting", "domain", "from_layer"]
    probability_sums = matrix.groupby(group_keys).probability.sum()
    assert np.allclose(probability_sums.to_numpy(), 1.0, atol=1e-12)
    count_sums = matrix.groupby(group_keys).raw_pair_count.sum()
    token_totals = matrix.groupby(group_keys).n_tokens.first() * 16
    assert np.array_equal(count_sums.to_numpy(), token_totals.to_numpy())

    samples = pq.read_table(ROOT / "cache/sample_table.parquet", columns=["domain", "sample_order"]).to_pandas()
    samples = samples.sort_values("sample_order", kind="stable").reset_index(drop=True)
    annotations = pq.read_table(ROOT / "cache/token_annotations.parquet", columns=["mask_primary_inference"]).to_pandas()
    with np.load(ROOT / "cache/routing_top4.npz", allow_pickle=False) as source:
        expert = source["expert_id"]
        token_sample_index = source["token_sample_index"].astype(np.int32)
    primary = annotations.mask_primary_inference.to_numpy(bool)

    direct_checks = []
    direct_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for domain, from_layer in (("math", 10), ("law", 10), ("computer science", 11), ("psychology", 11)):
        indices = np.flatnonzero(samples.domain.eq(domain).to_numpy())
        token_weighted, sample_equal = direct_distribution(
            expert, token_sample_index, primary, indices, from_layer, from_layer + 1
        )
        direct_cache[(domain, from_layer)] = (token_weighted, sample_equal)
        reported_token = reported_distribution(matrix, domain, from_layer, "token_weighted")
        reported_sample = reported_distribution(matrix, domain, from_layer, "sample_equal")
        token_error = float(np.max(np.abs(token_weighted - reported_token)))
        sample_error = float(np.max(np.abs(sample_equal - reported_sample)))
        assert token_error < 1e-12 and sample_error < 1e-12
        direct_checks.append({
            "domain": domain, "from_layer": from_layer,
            "token_weighted_max_abs_error": token_error,
            "sample_equal_max_abs_error": sample_error,
        })

    for from_layer in (10, 11):
        math_distribution = reported_distribution(matrix, "math", from_layer, "sample_equal")
        law_distribution = reported_distribution(matrix, "law", from_layer, "sample_equal")
        direct_jsd = jsd_bits(math_distribution, law_distribution)
        reported_jsd = float(jsd.loc[
            (jsd.population == "primary_inference")
            & (jsd.weighting == "sample_equal")
            & (jsd.from_layer == from_layer)
            & (jsd.domain_a == "math")
            & (jsd.domain_b == "law"), "jsd_bits"
        ].iloc[0])
        assert abs(direct_jsd - reported_jsd) < 1e-12

    focus_summary = summary[
        (summary.population == "primary_inference") & (summary.weighting == "sample_equal")
    ]
    focus_jsd = jsd[(jsd.population == "primary_inference") & (jsd.weighting == "sample_equal")]
    print(json.dumps({
        "status": "passed",
        "row_counts": {"matrix": len(matrix), "summary": len(summary), "top": len(top), "jsd": len(jsd)},
        "missing_values": 0,
        "manifest_output_hashes": "all_match",
        "probability_sum_max_error": float(np.max(np.abs(probability_sums.to_numpy() - 1))),
        "direct_checks": direct_checks,
        "focus_entropy_range_bits": [float(focus_summary.entropy_bits.min()), float(focus_summary.entropy_bits.max())],
        "focus_top20_mass_range": [float(focus_summary.top20_mass.min()), float(focus_summary.top20_mass.max())],
        "focus_jsd_range_bits": [float(focus_jsd.jsd_bits.min()), float(focus_jsd.jsd_bits.max())],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
