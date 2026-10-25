#!/usr/bin/env python3
"""Read-only preflight validation for step 03 routing-path analysis."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(
    os.environ.get(
        "MOE_SAE_ANALYSIS_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1",
    )
).expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    cache_manifest_path = ROOT / "metadata/cache_manifest.json"
    annotation_manifest_path = ROOT / "metadata/token_annotations_manifest.json"
    routing_manifest_path = ROOT / "metadata/routing_domain_analysis_manifest.json"
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in (cache_manifest_path, annotation_manifest_path, routing_manifest_path)]
    require(all(item.get("status") == "complete" for item in manifests), "A prior-step manifest is incomplete")
    require(all(item.get("model_inference_rerun") is False for item in manifests), "A prior manifest does not confirm cache-only work")

    annotations_path = ROOT / "cache/token_annotations.parquet"
    routing_path = ROOT / "cache/routing_top4.npz"
    require(sha256(annotations_path) == manifests[1]["outputs"][str(annotations_path)]["sha256"], "Annotation SHA mismatch")
    require(sha256(routing_path) == manifests[0]["outputs"][str(routing_path)]["sha256"], "Routing SHA mismatch")
    annotation_rows = pq.ParquetFile(annotations_path).metadata.num_rows
    require(annotation_rows == 382_685, f"Unexpected annotation rows: {annotation_rows}")

    with np.load(routing_path, allow_pickle=False) as routing:
        expert_id = routing["expert_id"]
        require(expert_id.shape == (382_685, 24, 4), f"Unexpected expert shape: {expert_id.shape}")
        require(expert_id.dtype == np.uint16, f"Unexpected expert dtype: {expert_id.dtype}")
        sorted_ids = np.sort(expert_id, axis=2)
        duplicate_top4 = int((sorted_ids[:, :, 1:] == sorted_ids[:, :, :-1]).any(axis=2).sum())
        require(duplicate_top4 == 0, f"Duplicate experts within Top-4: {duplicate_top4}")
        result = {
            "status": "passed",
            "prior_manifest_statuses": [item["status"] for item in manifests],
            "annotation_rows": annotation_rows,
            "expert_id_shape": list(expert_id.shape),
            "expert_id_dtype": str(expert_id.dtype),
            "expert_id_min": int(expert_id.min()),
            "expert_id_max": int(expert_id.max()),
            "duplicate_experts_within_top4": duplicate_top4,
            "primary_tokens": int(manifests[1]["token_count_primary_inference"]),
            "all_tokens": int(manifests[1]["token_count_all"]),
            "model_inference_rerun": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
