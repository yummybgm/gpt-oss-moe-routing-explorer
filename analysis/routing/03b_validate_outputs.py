#!/usr/bin/env python3
"""Independent, read-only validation of step-03B outputs."""
from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(os.environ.get("MOE_SAE_ANALYSIS_ROOT", Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1")).expanduser().resolve()
TABLE = ROOT / "tables/routing_adjacent_layer_continuity.csv"
RETENTION = ROOT / "tables/routing_adjacent_retention_distribution.csv"
PER_SAMPLE = ROOT / "cache/routing_adjacent_per_sample.parquet"
MANIFEST = ROOT / "metadata/routing_path_step03b_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    table = pd.read_csv(TABLE)
    retention = pd.read_csv(RETENTION)
    per_sample = pq.read_table(PER_SAMPLE).to_pandas()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for raw_path, metadata in manifest["outputs"].items():
        output_path = Path(raw_path)
        assert output_path.is_file(), f"missing output: {output_path}"
        assert output_path.stat().st_size == metadata["size_bytes"], f"size mismatch: {output_path}"
        assert sha256_file(output_path) == metadata["sha256"], f"sha256 mismatch: {output_path}"
    assert len(table) == 460 and len(retention) == 2_300 and len(per_sample) == 92_000
    assert not table.isna().any().any() and not retention.isna().any().any() and not per_sample.isna().any().any()
    sums = retention.groupby(["population", "weighting", "domain", "from_layer"])["share"].sum()
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-12)

    annotations = pq.read_table(ROOT / "cache/token_annotations.parquet", columns=["mask_primary_inference"]).to_pandas()
    with np.load(ROOT / "cache/routing_top4.npz", allow_pickle=False) as source:
        expert = source["expert_id"]
        sample_index = source["token_sample_index"]
    samples = pq.read_table(ROOT / "cache/sample_table.parquet", columns=["domain", "sample_order"]).to_pandas()
    samples = samples.sort_values("sample_order", kind="stable").reset_index(drop=True)
    token_domains = samples["domain"].to_numpy()[sample_index]
    primary = annotations["mask_primary_inference"].to_numpy(bool)

    checks = []
    for layer in (10, 11):
        intersection = np.zeros(len(expert), dtype=np.uint8)
        for left_rank in range(4):
            intersection += np.any(
                expert[:, layer, left_rank, None] == expert[:, layer + 1, :], axis=1
            ).astype(np.uint8)
        jaccard = intersection / (8.0 - intersection)
        for domain in ("math", "computer science", "physics", "law", "psychology"):
            mask = primary & (token_domains == domain)
            direct = float(jaccard[mask].mean())
            reported = float(table.loc[
                (table.population == "primary_inference")
                & (table.weighting == "token_weighted")
                & (table.domain == domain)
                & (table.from_layer == layer), "mean_jaccard"
            ].iloc[0])
            assert abs(direct - reported) < 1e-12
            checks.append({"from_layer": layer, "domain": domain, "direct": direct, "reported": reported})

    focus = table[
        (table.population == "primary_inference")
        & (table.weighting == "sample_equal")
        & table.from_layer.isin([10, 11])
    ][["domain", "from_layer", "mean_jaccard", "mean_retained_count", "n_tokens", "n_samples"]]
    print(json.dumps({
        "status": "passed",
        "row_counts": {"main": len(table), "retention": len(retention), "per_sample": len(per_sample)},
        "missing_values": 0,
        "manifest_output_hashes": "all_match",
        "retention_share_max_error": float(np.abs(sums.to_numpy() - 1).max()),
        "independent_direct_checks": checks,
        "focus_sample_equal": focus.to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
