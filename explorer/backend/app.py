#!/usr/bin/env python3
"""Read-only FastAPI service for the GPT-OSS Routing–SAE explorer."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request


API_VERSION = "routing_sae_explorer_api_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    cache_dir: Path
    layout_dir: Path
    density_dir: Path
    contract_path: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            cache_dir=_path_from_env("EXPLORER_CACHE_DIR", PROJECT_ROOT / "cache" / "v01"),
            layout_dir=_path_from_env("EXPLORER_LAYOUT_DIR", PROJECT_ROOT / "layout" / "v02"),
            density_dir=_path_from_env(
                "EXPLORER_DENSITY_DIR", PROJECT_ROOT / "layout" / "v02_top1_density"
            ),
            contract_path=_path_from_env(
                "EXPLORER_API_CONTRACT", PROJECT_ROOT / "contracts" / "v03_api_contract.json"
            ),
        )


def _native(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    return [
        {column: _native(record[column]) for column in columns}
        for record in frame[columns].to_dict(orient="records")
    ]


class ExplorerStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.contract = json.loads(settings.contract_path.read_text(encoding="utf-8"))
        self.nodes = pd.read_parquet(settings.cache_dir / "graph_nodes.parquet")
        self.tokens = pd.read_parquet(settings.cache_dir / "token_index.parquet").sort_values(
            "token_row", kind="stable"
        )
        self.tokens_by_row = self.tokens.set_index("token_row", drop=False)
        trajectory_archive = np.load(settings.cache_dir / "top1_trajectories.npz", allow_pickle=False)
        self.trajectory_nodes = trajectory_archive["node_id"]
        self.trajectory_weights = trajectory_archive["routing_weight"]
        self.trajectory_path_ids = trajectory_archive["path_id"]
        trajectory_archive.close()
        self.top4_edges = pd.read_parquet(settings.cache_dir / "top4_weighted_edges.parquet")
        self.node_stats = pd.read_parquet(settings.cache_dir / "node_stats.parquet")
        self.contexts = pd.read_parquet(settings.cache_dir / "expert_contexts.parquet")
        self.sae_index = pd.read_parquet(settings.cache_dir / "sae_query_index.parquet")
        self.layout = pd.read_parquet(settings.layout_dir / "recommended_layout.parquet")
        self.top1_edges = pd.read_parquet(settings.density_dir / "top1_transition_edges.parquet")
        self._validate_loaded_data()

    def _validate_loaded_data(self) -> None:
        expected_tokens = int(self.contract["research"]["token_count"])
        expected_nodes = int(
            self.contract["research"]["routing_layers"]
            * self.contract["research"]["experts_per_layer"]
        )
        if len(self.tokens) != expected_tokens:
            raise RuntimeError(f"Token count mismatch: {len(self.tokens)} != {expected_tokens}")
        if not np.array_equal(self.tokens["token_row"].to_numpy(), np.arange(expected_tokens)):
            raise RuntimeError("token_row is not contiguous and aligned with trajectory rows")
        if self.trajectory_nodes.shape != (expected_tokens, 24):
            raise RuntimeError(f"Unexpected trajectory shape: {self.trajectory_nodes.shape}")
        if self.trajectory_weights.shape != self.trajectory_nodes.shape:
            raise RuntimeError("Trajectory node and weight shapes differ")
        if len(self.nodes) != expected_nodes or len(self.layout) != expected_nodes:
            raise RuntimeError("Graph node or layout coverage mismatch")
        if self.layout["node_id"].duplicated().any():
            raise RuntimeError("Duplicate node_id in frozen layout")
        expected_groups = {
            (group_type, group_value)
            for group_type, values in self.contract["filter_groups"].items()
            for group_value in values
        }
        for name, frame in [
            ("top1_edges", self.top1_edges),
            ("top4_edges", self.top4_edges),
            ("node_stats", self.node_stats),
        ]:
            observed = set(zip(frame["group_type"], frame["group_value"]))
            if observed != expected_groups:
                raise RuntimeError(f"Filter group mismatch in {name}")

    def token_row(self, token_row: int) -> pd.Series:
        if token_row not in self.tokens_by_row.index:
            raise HTTPException(status_code=404, detail=f"Unknown token_row: {token_row}")
        return self.tokens_by_row.loc[token_row]

    def validate_group(self, group_type: str, group_value: str) -> None:
        allowed = self.contract["filter_groups"].get(group_type)
        if allowed is None or group_value not in allowed:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid group_type/group_value pair",
                    "group_type": group_type,
                    "group_value": group_value,
                },
            )


def get_store(request: Request) -> ExplorerStore:
    return request.app.state.store


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ExplorerStore(Settings.from_environment())
    yield


app = FastAPI(
    title="GPT-OSS Routing–SAE Explorer API",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    store = get_store(request)
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "loaded": True,
        "source_counts": {
            "tokens": len(store.tokens),
            "nodes": len(store.nodes),
            "top1_edge_rows": len(store.top1_edges),
            "top4_edge_rows": len(store.top4_edges),
            "expert_contexts": len(store.contexts),
            "sae_index_rows": len(store.sae_index),
        },
    }


@app.get("/api/meta")
def meta(request: Request) -> dict[str, Any]:
    contract = get_store(request).contract
    return {
        "api_version": API_VERSION,
        "model": contract["research"]["model"],
        "dataset": contract["research"]["dataset"],
        "filter_groups": contract["filter_groups"],
        "sae_layers": contract["sae_layers"],
        "limits": contract["limits"],
        "interpretation_warnings": contract["scientific_boundaries"],
    }


@app.get("/api/graph")
def graph(
    request: Request,
    group_type: str = "population",
    group_value: str = "all_tokens",
    edge_mode: str = Query(default="top1", pattern="^(top1|top4)$"),
    min_edge_value: float = Query(default=0.0, ge=0.0),
    max_edges: int = Query(default=25000, ge=1, le=25000),
) -> dict[str, Any]:
    store = get_store(request)
    store.validate_group(group_type, group_value)
    stats = store.node_stats[
        (store.node_stats["group_type"] == group_type)
        & (store.node_stats["group_value"] == group_value)
    ]
    nodes = store.layout.merge(
        stats.drop(columns=["layer", "expert_id"], errors="ignore"), on="node_id", how="left", validate="one_to_one"
    )
    if len(nodes) != 768 or nodes["group_token_count"].isna().any():
        raise HTTPException(status_code=500, detail="Incomplete node statistics for selected group")
    node_columns = [
        "node_id", "layer", "expert_id", "sae_available", "x", "y", "position_uncertainty",
        "group_token_count", "top1_count", "top4_selection_count", "routing_weight_mass",
        "top1_rate", "top4_slot_share", "mean_routing_weight",
    ]

    if edge_mode == "top1":
        edges = store.top1_edges[
            (store.top1_edges["group_type"] == group_type)
            & (store.top1_edges["group_value"] == group_value)
        ].copy()
        edges["edge_value"] = edges["token_count"]
        edge_columns = [
            "layer_source", "layer_target", "source_node_id", "target_node_id", "token_count", "edge_value"
        ]
    else:
        edges = store.top4_edges[
            (store.top4_edges["group_type"] == group_type)
            & (store.top4_edges["group_value"] == group_value)
        ].copy()
        edges["edge_value"] = edges["weight_mass"]
        edge_columns = [
            "layer_source", "layer_target", "source_node_id", "target_node_id",
            "token_support", "weight_mass", "edge_value",
        ]
    edges = edges[edges["edge_value"] >= min_edge_value].sort_values(
        ["edge_value", "layer_source", "source_node_id", "target_node_id"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    total = len(edges)
    edges = edges.head(max_edges)
    warning = (
        "Top-1 edges are observed dominant-expert transitions."
        if edge_mode == "top1"
        else "Top-4 edges are weighted co-occurrence and are not causal information-flow edges."
    )
    return {
        "group": {"group_type": group_type, "group_value": group_value},
        "edge_mode": edge_mode,
        "nodes": _records(nodes, node_columns),
        "edges": _records(edges, edge_columns),
        "edge_total_before_limit": total,
        "truncated": total > len(edges),
        "warnings": [
            "Coordinates are frozen layout outputs, not physical model positions.",
            warning,
        ],
    }


TOKEN_COLUMNS = [
    "token_row", "sample_id", "domain", "token_position", "token_id", "token_text",
    "prompt_segment", "is_first_token", "token_class", "mask_primary_inference",
    "top1_path_id", "top1_path_frequency",
]


@app.get("/api/tokens/search")
def token_search(
    request: Request,
    q: str | None = None,
    sample_id: str | None = None,
    domain: str | None = None,
    token_class: str | None = None,
    prompt_segment: str | None = None,
    primary_only: bool = False,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    store = get_store(request)
    frame = store.tokens
    if domain is not None and domain not in store.contract["filter_groups"]["domain"]:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")
    if token_class is not None and token_class not in store.contract["filter_groups"]["token_class"]:
        raise HTTPException(status_code=400, detail=f"Unknown token_class: {token_class}")
    if prompt_segment is not None and prompt_segment not in store.contract["filter_groups"]["prompt_segment"]:
        raise HTTPException(status_code=400, detail=f"Unknown prompt_segment: {prompt_segment}")
    mask = pd.Series(True, index=frame.index)
    if q is not None and q != "":
        mask &= frame["token_text"].astype("string").str.contains(q, case=False, regex=False, na=False)
    if sample_id is not None:
        mask &= frame["sample_id"] == sample_id
    if domain is not None:
        mask &= frame["domain"] == domain
    if token_class is not None:
        mask &= frame["token_class"] == token_class
    if prompt_segment is not None:
        mask &= frame["prompt_segment"] == prompt_segment
    if primary_only:
        mask &= frame["mask_primary_inference"]
    matches = frame.loc[mask].sort_values("token_row", kind="stable")
    items = matches.iloc[offset : offset + limit]
    return {
        "query": {
            "q": q, "sample_id": sample_id, "domain": domain, "token_class": token_class,
            "prompt_segment": prompt_segment, "primary_only": primary_only,
        },
        "total_matches": len(matches),
        "offset": offset,
        "limit": limit,
        "items": _records(items, TOKEN_COLUMNS),
    }


@app.get("/api/samples/{sample_id}/tokens")
def sample_tokens(sample_id: str, request: Request) -> dict[str, Any]:
    store = get_store(request)
    items = store.tokens[store.tokens["sample_id"] == sample_id].sort_values("token_position", kind="stable")
    if items.empty:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id}")
    maximum = int(store.contract["limits"]["sample_tokens_max"])
    if len(items) > maximum:
        raise HTTPException(status_code=500, detail=f"Sample exceeds frozen token limit: {len(items)} > {maximum}")
    return {
        "sample_id": sample_id,
        "domain": str(items.iloc[0]["domain"]),
        "token_count": len(items),
        "items": _records(items, TOKEN_COLUMNS),
    }


@app.get("/api/tokens/{token_row}")
def token_detail(token_row: int, request: Request) -> dict[str, Any]:
    store = get_store(request)
    token = store.token_row(token_row)
    node_ids = store.trajectory_nodes[token_row]
    weights = store.trajectory_weights[token_row]
    trajectory = [
        {
            "layer": layer,
            "node_id": int(node_id),
            "expert_id": int(node_id) % 32,
            "routing_weight": float(weights[layer]),
        }
        for layer, node_id in enumerate(node_ids)
    ]
    path_id = int(store.trajectory_path_ids[token_row])
    return {
        "token": {column: _native(token[column]) for column in TOKEN_COLUMNS},
        "trajectory": trajectory,
        "path": {
            "path_id": path_id,
            "frequency": int(token["top1_path_frequency"]),
            "complete_layers": len(trajectory),
        },
    }


@app.get("/api/nodes/{node_id}/contexts")
def node_contexts(
    node_id: int,
    request: Request,
    domain: str | None = None,
    limit: int = Query(default=40, ge=1, le=100),
) -> dict[str, Any]:
    store = get_store(request)
    if not 0 <= node_id < 768:
        raise HTTPException(status_code=404, detail=f"Unknown node_id: {node_id}")
    if domain is None:
        context_group_type, context_group_value = "overall", "primary_inference"
    else:
        if domain not in store.contract["filter_groups"]["domain"]:
            raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")
        context_group_type, context_group_value = "domain", domain
    frame = store.contexts[
        (store.contexts["node_id"] == node_id)
        & (store.contexts["context_group_type"] == context_group_type)
        & (store.contexts["context_group_value"] == context_group_value)
    ].sort_values("display_rank", kind="stable")
    columns = [
        "display_rank", "token_row", "sample_id", "domain", "token_position", "token_text",
        "context_text", "expert_rank", "routing_weight",
    ]
    return {
        "node": {"node_id": node_id, "layer": node_id // 32, "expert_id": node_id % 32},
        "context_group": {
            "context_group_type": context_group_type,
            "context_group_value": context_group_value,
        },
        "total_available": len(frame),
        "items": _records(frame.head(limit), columns),
        "warnings": ["High-routing-weight examples are evidence displays, not proof of expert function."],
    }


@app.get("/api/tokens/{token_row}/sae")
def token_sae(
    token_row: int,
    request: Request,
    sae_layer: int = Query(...),
    passed_threshold_only: bool = True,
    limit: int = Query(default=15, ge=1, le=100),
) -> dict[str, Any]:
    store = get_store(request)
    token = store.token_row(token_row)
    if sae_layer not in store.contract["sae_layers"]:
        raise HTTPException(status_code=400, detail=f"Unsupported sae_layer: {sae_layer}")
    index = store.sae_index[
        (store.sae_index["sample_id"] == token["sample_id"])
        & (store.sae_index["sae_layer"] == sae_layer)
    ]
    if len(index) != 1:
        raise HTTPException(status_code=404, detail="SAE sample-layer index row not found")
    index_row = index.iloc[0]
    feature_path = Path(index_row["feature_file"])
    metric_path = Path(index_row["token_metric_file"])
    if not feature_path.exists() or not metric_path.exists():
        raise HTTPException(status_code=404, detail="Indexed SAE source shard is unavailable")
    filters = [("token_position", "=", int(token["token_position"]))]
    features = pd.read_parquet(feature_path, filters=filters)
    metrics = pd.read_parquet(metric_path, filters=filters)
    if passed_threshold_only:
        features = features[features["passed_threshold"]]
    features = features.sort_values(["activation", "feature_id"], ascending=[False, True], kind="stable")
    if len(metrics) != 1:
        raise HTTPException(status_code=500, detail=f"Expected one SAE metric row, found {len(metrics)}")
    metric = metrics.iloc[0]
    feature_columns = ["feature_id", "feature_rank", "activation", "passed_threshold"]
    metric_columns = ["active_feature_count", "reconstruction_mse", "cosine_similarity"]
    warnings = [
        "SAE activations at resid_post are associations, not proof of the cause of same-layer routing.",
        "Global feature semantics require a separately validated feature-context cache.",
    ]
    if bool(token["is_first_token"]):
        warnings.insert(0, "First-token anomaly: do not use this activation as primary semantic evidence.")
    return {
        "token": {column: _native(token[column]) for column in TOKEN_COLUMNS},
        "sae_layer": sae_layer,
        "metrics": {column: _native(metric[column]) for column in metric_columns},
        "total_active_features": len(features),
        "items": _records(features.head(limit), feature_columns),
        "warnings": warnings,
    }
