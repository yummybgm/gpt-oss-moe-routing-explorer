"""Independent validator for the remote Neuronpedia full semantic database."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
from pathlib import Path


ROOT = Path(
    os.environ.get(
        "NEURONPEDIA_SEMANTICS_ROOT",
        Path(__file__).resolve().parents[2] / "data" / "neuronpedia_gpt_oss_20b_v1",
    )
).expanduser().resolve()
LAYERS = (3, 7, 11, 15, 19, 23)
DICT_SIZE = 131_072


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    manifest_path = ROOT / "manifests" / "final_manifest.json"
    object_manifest_path = ROOT / "manifests" / "s3_objects.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    database = Path(manifest["database"]["path"])
    require(database.is_file(), "Database missing")
    require(sha256(database) == manifest["database"]["sha256"], "Database SHA-256 mismatch")

    with object_manifest_path.open("r", encoding="utf-8") as handle:
        objects = [json.loads(line) for line in handle if line.strip()]
    require(len(objects) == 2080, f"S3 object manifest count mismatch: {len(objects)}")
    raw_bytes = 0
    for item in objects:
        path = Path(item["path"])
        require(path.is_file(), f"Raw S3 object missing: {path}")
        require(path.stat().st_size == int(item["size"]), f"Raw object size mismatch: {path}")
        require(sha256(path) == item["sha256"], f"Raw object SHA-256 mismatch: {path}")
        raw_bytes += path.stat().st_size
    require(raw_bytes == int(manifest["raw_s3_bytes"]), "Raw S3 byte total mismatch")

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        require(connection.execute("PRAGMA quick_check").fetchone()[0] == "ok", "SQLite quick_check failed")
        total = connection.execute("SELECT COUNT(*) FROM feature_semantics").fetchone()[0]
        explained = connection.execute(
            "SELECT COUNT(*) FROM feature_semantics WHERE coverage_status='np_explained'"
        ).fetchone()[0]
        missing = connection.execute(
            "SELECT COUNT(*) FROM feature_semantics WHERE coverage_status='np_missing'"
        ).fetchone()[0]
        duplicate_keys = connection.execute(
            "SELECT COUNT(*) FROM (SELECT sae_layer, feature_id, COUNT(*) n FROM feature_semantics GROUP BY 1,2 HAVING n>1)"
        ).fetchone()[0]
        api_rows = connection.execute("SELECT COUNT(*) FROM api_recheck").fetchone()[0]
        api_errors = connection.execute(
            "SELECT COUNT(*) FROM api_recheck WHERE http_status<>200 OR error<>''"
        ).fetchone()[0]
        api_raw = list((ROOT / "api_backfill" / "raw").glob("*.json.gz"))
        require(total == len(LAYERS) * DICT_SIZE, f"Feature universe mismatch: {total}")
        require(explained == 785_583 and missing == 849, "Coverage count mismatch")
        require(duplicate_keys == 0, "Duplicate feature keys")
        require(api_rows == 850 and api_errors == 0, "API audit mismatch")
        require(len(api_raw) == 850, f"API raw response count mismatch: {len(api_raw)}")
        for row in connection.execute("SELECT raw_path, raw_sha256 FROM api_recheck"):
            path = Path(row[0])
            require(path.is_file() and sha256(path) == row[1], f"API raw hash mismatch: {path}")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                envelope = json.load(handle)
            require(int(envelope["http_status"]) == 200, f"API raw status mismatch: {path}")
        layer_rows = []
        for layer in LAYERS:
            row = connection.execute(
                """
                SELECT COUNT(*),
                       SUM(coverage_status='np_explained'),
                       SUM(coverage_status='np_missing')
                FROM feature_semantics WHERE sae_layer=?
                """,
                (layer,),
            ).fetchone()
            require(row[0] == DICT_SIZE and row[1] + row[2] == DICT_SIZE, f"Layer {layer} mismatch")
            layer_rows.append({"layer": layer, "total": row[0], "explained": row[1], "missing": row[2]})
        updated = connection.execute(
            "SELECT coverage_status, primary_source_kind, primary_description FROM feature_semantics WHERE sae_layer=23 AND feature_id=92369"
        ).fetchone()
        require(updated and updated[0] == "np_explained" and updated[1] == "live_api" and updated[2], "Live API addition missing")
    finally:
        connection.close()

    result = {
        "status": "validation_passed",
        "database_sha256": manifest["database"]["sha256"],
        "s3_objects_checked": len(objects),
        "s3_bytes_checked": raw_bytes,
        "api_raw_responses_checked": 850,
        "feature_rows": total,
        "explained_features": explained,
        "missing_features": missing,
        "duplicate_feature_keys": duplicate_keys,
        "api_errors": api_errors,
        "by_layer": layer_rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
