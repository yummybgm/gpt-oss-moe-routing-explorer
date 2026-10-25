"""Shared cache-only helpers for formal step-03 routing analyses."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(os.environ.get("MOE_SAE_PROJECT_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
ROOT = Path(
    os.environ.get(
        "MOE_SAE_ANALYSIS_ROOT",
        PROJECT_ROOT / "data" / "mmlu_pro_formal_2000_analysis_v1",
    )
).expanduser().resolve()
CACHE = ROOT / "cache"
TABLES = ROOT / "tables"
METADATA = ROOT / "metadata"
LOGS = ROOT / "logs"
FIG_PNG = ROOT / "figures/png"
FIG_SVG = ROOT / "figures/svg"
REPORT = ROOT / "report"

SAMPLE_PATH = CACHE / "sample_table.parquet"
ANNOTATION_PATH = CACHE / "token_annotations.parquet"
ROUTING_PATH = CACHE / "routing_top4.npz"
CACHE_MANIFEST = METADATA / "cache_manifest.json"
ANNOTATION_MANIFEST = METADATA / "token_annotations_manifest.json"
STEP02_MANIFEST = METADATA / "routing_domain_analysis_manifest.json"

N_SAMPLES = 2_000
N_TOKENS = 382_685
N_PRIMARY = 380_685
N_LAYERS = 24
N_EXPERTS = 32
TOP_K = 4
DOMAINS = ["math", "computer science", "physics", "law", "psychology"]
RANDOM_SEED = 42


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(data: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary, compression="zstd", compression_level=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_dirs() -> None:
    for path in (CACHE, TABLES, METADATA, LOGS, FIG_PNG, FIG_SVG, REPORT):
        path.mkdir(parents=True, exist_ok=True)


def load_inputs(annotation_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    paths = (SAMPLE_PATH, ANNOTATION_PATH, ROUTING_PATH, CACHE_MANIFEST, ANNOTATION_MANIFEST, STEP02_MANIFEST)
    for path in paths:
        require(path.is_file(), f"Missing step-03 input: {path}")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in (CACHE_MANIFEST, ANNOTATION_MANIFEST, STEP02_MANIFEST)]
    require(all(item.get("status") == "complete" for item in manifests), "A prior-step manifest is incomplete")
    require(all(item.get("model_inference_rerun") is False for item in manifests), "Prior work is not confirmed cache-only")
    require(sha256_file(SAMPLE_PATH) == manifests[0]["outputs"][str(SAMPLE_PATH)]["sha256"], "sample_table SHA mismatch")
    require(sha256_file(ROUTING_PATH) == manifests[0]["outputs"][str(ROUTING_PATH)]["sha256"], "routing_top4 SHA mismatch")
    require(sha256_file(ANNOTATION_PATH) == manifests[1]["outputs"][str(ANNOTATION_PATH)]["sha256"], "token_annotations SHA mismatch")

    samples = pq.read_table(SAMPLE_PATH).to_pandas().sort_values("sample_order", kind="stable").reset_index(drop=True)
    annotations = pq.read_table(ANNOTATION_PATH, columns=annotation_columns).to_pandas()
    with np.load(ROUTING_PATH, allow_pickle=False) as source:
        routing = {name: source[name] for name in source.files}
    require(len(samples) == N_SAMPLES and len(annotations) == N_TOKENS, "Input row-count mismatch")
    require(samples["sample_id"].is_unique, "Duplicate sample_id")
    require(routing["expert_id"].shape == (N_TOKENS, N_LAYERS, TOP_K), "expert_id shape mismatch")
    require(np.array_equal(routing["sample_ids"].astype(str), samples["sample_id"].astype(str).to_numpy()), "Sample order mismatch")
    token_sample_index = routing["token_sample_index"].astype(np.int32, copy=False)
    require(np.array_equal(routing["sample_ids"].astype(str)[token_sample_index], annotations["sample_id"].astype(str).to_numpy()), "Token/sample mapping mismatch")
    require(np.array_equal(routing["token_position"], annotations["token_position"].to_numpy()), "Token-position mapping mismatch")
    return samples, annotations, routing, manifests


def sample_domain_codes(samples: pd.DataFrame) -> np.ndarray:
    codes = pd.Categorical(samples["domain"], categories=DOMAINS).codes.astype(np.int8)
    require((codes >= 0).all(), "Unknown sample domain")
    return codes


def adjacent_metrics(expert_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = expert_id[:, :-1, :]
    right = expert_id[:, 1:, :]
    retained = (left[:, :, :, None] == right[:, :, None, :]).any(axis=3).sum(axis=2).astype(np.uint8)
    jaccard = retained.astype(np.float64) / (8.0 - retained)
    require(retained.shape == (len(expert_id), N_LAYERS - 1), "Adjacent retained shape mismatch")
    require(retained.min() >= 0 and retained.max() <= TOP_K, "Retained count outside 0..4")
    require(np.isfinite(jaccard).all(), "Nonfinite Jaccard")
    return retained, jaccard


def bh_adjust(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    require(p.sum() > 0 and q.sum() > 0, "JSD received empty distribution")
    p = p / p.sum(); q = q / q.sum(); midpoint = 0.5 * (p + q)
    def kl(left: np.ndarray, right: np.ndarray) -> float:
        nonzero = left > 0
        return float(np.sum(left[nonzero] * np.log2(left[nonzero] / right[nonzero])))
    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def script_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path.resolve())}
