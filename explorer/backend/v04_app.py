#!/usr/bin/env python3
"""V04 extension: Token-level Top-4 queries plus the static explorer frontend."""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_APP_PATH = PROJECT_ROOT / "backend" / "app.py"
spec = importlib.util.spec_from_file_location("routing_sae_v03_backend", BASE_APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import V03 backend: {BASE_APP_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class V04Store(base.ExplorerStore):
    def __init__(self, settings: base.Settings) -> None:
        super().__init__(settings)
        source = Path(
            os.environ.get(
                "EXPLORER_ROUTING_TOP4",
                PROJECT_ROOT / "data" / "routing_top4.npz",
            )
        ).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Top-4 routing source not found: {source}")
        with np.load(source, allow_pickle=False) as archive:
            self.top4_expert_id = archive["expert_id"].astype(np.uint8, copy=True)
            self.top4_routing_weight = archive["routing_weight"].astype(np.float32, copy=True)
        self.top4_source = source
        expected_shape = (len(self.tokens), 24, 4)
        if self.top4_expert_id.shape != expected_shape:
            raise RuntimeError(f"Unexpected Top-4 expert shape: {self.top4_expert_id.shape}")
        if self.top4_routing_weight.shape != expected_shape:
            raise RuntimeError(f"Unexpected Top-4 weight shape: {self.top4_routing_weight.shape}")
        expected_top1_nodes = (
            np.arange(24, dtype=np.uint16)[None, :] * 32
            + self.top4_expert_id[:, :, 0].astype(np.uint16)
        )
        if not np.array_equal(expected_top1_nodes, self.trajectory_nodes):
            raise RuntimeError("Top-4 rank-1 experts do not match the frozen Top-1 trajectories")


@asynccontextmanager
async def v04_lifespan(app):
    app.state.store = V04Store(base.Settings.from_environment())
    yield


app = base.app
app.title = "GPT-OSS Routing–SAE Explorer"
app.version = "routing_sae_explorer_api_v1_v04_extension"
app.router.lifespan_context = v04_lifespan


@app.get("/api/tokens/{token_row}/routing-top4")
def token_routing_top4(
    token_row: int,
    request: Request,
    include_all_ranks: bool = Query(default=True),
) -> dict[str, Any]:
    store: V04Store = request.app.state.store
    token = store.token_row(token_row)
    expert_ids = store.top4_expert_id[token_row]
    weights = store.top4_routing_weight[token_row]
    ranks = 4 if include_all_ranks else 1
    layers = []
    for layer in range(24):
        selections = [
            {
                "expert_rank": rank + 1,
                "expert_id": int(expert_ids[layer, rank]),
                "node_id": layer * 32 + int(expert_ids[layer, rank]),
                "routing_weight": float(weights[layer, rank]),
            }
            for rank in range(ranks)
        ]
        if not np.isclose(sum(item["routing_weight"] for item in selections), 1.0, atol=3e-7) and ranks == 4:
            raise HTTPException(status_code=500, detail=f"Top-4 weights are not normalized at layer {layer}")
        layers.append({"layer": layer, "selections": selections})
    return {
        "token_row": token_row,
        "sample_id": str(token["sample_id"]),
        "token_position": int(token["token_position"]),
        "layers": layers,
        "matrix_shape": [32, 24],
        "warnings": [
            "Cell intensity represents normalized Top-4 routing weight.",
            "The matrix reports selected experts, not causal information flow.",
        ],
    }


frontend_dir = Path(os.environ.get("EXPLORER_FRONTEND_DIR", str(PROJECT_ROOT / "frontend"))).resolve()
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
