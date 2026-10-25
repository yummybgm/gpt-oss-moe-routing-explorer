"""Download Neuronpedia bulk explanations, then API-backfill only missing features.

Designed for the remote MoE-SAE data disk. Raw S3 exports and compressed API
responses are retained for auditability; embeddings remain only in raw exports.
The normalized SQLite database stores compact semantic fields.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import os
import random
import shutil
import sqlite3
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "gpt-oss-20b"
LAYERS = (3, 7, 11, 15, 19, 23)
DICT_SIZE = 131_072
S3_BASE = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
NP_BASE = "https://www.neuronpedia.org"
USER_AGENT = "moe-sae-research-neuronpedia-bulk/1.0"
DOWNLOAD_WORKERS = 10
API_WORKERS = 4
MAX_ATTEMPTS = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def open_url(url: str, timeout: int = 90):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def s3_list(prefix: str) -> list[dict]:
    query = urllib.parse.urlencode({"list-type": "2", "prefix": prefix})
    with open_url(f"{S3_BASE}/?{query}") as response:
        root = ET.fromstring(response.read())
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if root.findtext("s3:IsTruncated", namespaces=ns) != "false":
        raise RuntimeError(f"Unexpected truncated S3 listing: {prefix}")
    result = []
    for item in root.findall("s3:Contents", ns):
        result.append(
            {
                "key": item.findtext("s3:Key", namespaces=ns) or "",
                "size": int(item.findtext("s3:Size", namespaces=ns) or 0),
                "etag": (item.findtext("s3:ETag", namespaces=ns) or "").strip('"'),
                "last_modified": item.findtext("s3:LastModified", namespaces=ns) or "",
            }
        )
    return result


def build_object_inventory() -> list[dict]:
    inventory = []
    for layer in LAYERS:
        source = f"{layer}-resid-post-aa"
        explanation_prefix = f"v1/{MODEL_ID}/{source}/explanations/"
        objects = s3_list(explanation_prefix)
        selected = [
            item
            for item in objects
            if item["key"].endswith(".jsonl.gz") or item["key"].endswith("config.json")
        ]
        for filename in ("model.jsonl", "release.jsonl", "source.jsonl", "sourceset.jsonl"):
            key = f"v1/{MODEL_ID}/{source}/{filename}"
            matches = s3_list(key)
            if len(matches) != 1 or matches[0]["key"] != key:
                raise RuntimeError(f"Missing exact metadata object: {key}")
            selected.append(matches[0])
        for item in selected:
            inventory.append({"layer": layer, "source_id": source, **item})
        batches = [item for item in selected if "/batch-" in item["key"]]
        print(
            f"inventory L{layer}: batches={len(batches)}, "
            f"bytes={sum(item['size'] for item in batches):,}",
            flush=True,
        )
    return sorted(inventory, key=lambda item: item["key"])


def local_path_for_key(root: Path, key: str) -> Path:
    prefix = f"v1/{MODEL_ID}/"
    if not key.startswith(prefix):
        raise ValueError(key)
    return root / "raw_exports" / key[len(prefix) :]


def download_object(root: Path, item: dict) -> dict:
    destination = local_path_for_key(root, item["key"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["size"])
    expected_etag = str(item["etag"])
    if destination.is_file() and destination.stat().st_size == expected_size:
        if "-" in expected_etag or md5(destination) == expected_etag:
            return {**item, "path": str(destination), "sha256": sha256(destination), "reused": True}
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        temporary = destination.with_name(f".{destination.name}.part.{os.getpid()}.{random.randrange(1_000_000)}")
        try:
            url = f"{S3_BASE}/{urllib.parse.quote(item['key'], safe='/')}"
            with open_url(url) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=4 * 1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise IOError(
                    f"size mismatch {item['key']}: {temporary.stat().st_size} != {expected_size}"
                )
            if "-" not in expected_etag and md5(temporary) != expected_etag:
                raise IOError(f"ETag/MD5 mismatch: {item['key']}")
            os.replace(temporary, destination)
            return {
                **item,
                "path": str(destination),
                "sha256": sha256(destination),
                "reused": False,
                "attempts": attempt,
            }
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1) + random.random())
    raise RuntimeError(f"Download failed: {item['key']}: {last_error!r}")


def download_all(root: Path, inventory: list[dict]) -> list[dict]:
    started = time.perf_counter()
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = [executor.submit(download_object, root, item) for item in inventory]
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            records.append(future.result())
            if number == 1 or number % 50 == 0 or number == len(futures):
                print(
                    f"download {number}/{len(futures)}; "
                    f"{sum(item['size'] for item in records):,} bytes; "
                    f"{time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    records.sort(key=lambda item: item["key"])
    manifest_path = root / "manifests" / "s3_objects.jsonl"
    atomic_text(
        manifest_path,
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
    )
    return records


DB_SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE explanation_records (
    explanation_id TEXT PRIMARY KEY,
    sae_layer INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    description TEXT NOT NULL,
    explanation_model_name TEXT,
    type_name TEXT,
    created_at TEXT,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('s3_export', 'live_api')),
    source_ref TEXT NOT NULL
);
CREATE INDEX explanation_feature_idx ON explanation_records(sae_layer, feature_id);
CREATE TABLE api_recheck (
    sae_layer INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    request_url TEXT NOT NULL,
    retrieved_at_utc TEXT NOT NULL,
    http_status INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    explanation_records INTEGER NOT NULL,
    nonempty_explanations INTEGER NOT NULL,
    activation_records INTEGER NOT NULL,
    raw_path TEXT,
    raw_sha256 TEXT,
    error TEXT NOT NULL,
    PRIMARY KEY (sae_layer, feature_id)
);
CREATE TABLE feature_semantics (
    sae_layer INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    coverage_status TEXT NOT NULL,
    primary_description TEXT,
    primary_explanation_id TEXT,
    primary_source_kind TEXT,
    explanation_count INTEGER NOT NULL,
    PRIMARY KEY (sae_layer, feature_id)
);
"""


def create_export_database(root: Path, downloaded: list[dict]) -> tuple[Path, list[tuple[int, int]]]:
    db_dir = root / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    final_db = db_dir / "neuronpedia_feature_semantics.sqlite"
    temporary_db = db_dir / f".{final_db.name}.tmp.{os.getpid()}"
    temporary_db.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary_db)
    try:
        connection.executescript(DB_SCHEMA)
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        insert_sql = """
            INSERT INTO explanation_records (
                explanation_id, sae_layer, feature_id, description,
                explanation_model_name, type_name, created_at, source_kind, source_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 's3_export', ?)
        """
        batch_files = [item for item in downloaded if "/batch-" in item["key"]]
        total_records = 0
        for number, item in enumerate(batch_files, start=1):
            layer = int(item["layer"])
            source_id = f"{layer}-resid-post-aa"
            rows = []
            with gzip.open(item["path"], "rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("modelId") != MODEL_ID or record.get("layer") != source_id:
                        raise ValueError(f"Identity mismatch: {item['key']}:{line_number}")
                    feature_id = int(record["index"])
                    if not 0 <= feature_id < DICT_SIZE:
                        raise ValueError(f"Index out of range: {item['key']}:{feature_id}")
                    description = str(record.get("description") or "").strip()
                    if not description:
                        continue
                    rows.append(
                        (
                            str(record["id"]),
                            layer,
                            feature_id,
                            description,
                            record.get("explanationModelName"),
                            record.get("typeName"),
                            record.get("createdAt"),
                            item["key"],
                        )
                    )
            connection.executemany(insert_sql, rows)
            total_records += len(rows)
            if number % 100 == 0 or number == len(batch_files):
                connection.commit()
                print(f"parse {number}/{len(batch_files)}; records={total_records:,}", flush=True)

        missing = []
        for layer in LAYERS:
            present = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT feature_id FROM explanation_records WHERE sae_layer=?",
                    (layer,),
                )
            }
            missing.extend((layer, feature_id) for feature_id in range(DICT_SIZE) if feature_id not in present)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("model_id", MODEL_ID),
                ("layers", json.dumps(LAYERS)),
                ("dict_size", str(DICT_SIZE)),
                ("created_at_utc", utc_now()),
                ("s3_explanation_records", str(total_records)),
                ("s3_missing_features", str(len(missing))),
            ],
        )
        connection.commit()
    except Exception:
        connection.close()
        temporary_db.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(temporary_db, final_db)
    return final_db, missing


def atomic_gzip_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.gz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def api_fetch_missing(root: Path, layer: int, feature_id: int) -> tuple[dict, list[dict]]:
    url = f"{NP_BASE}/api/feature/{MODEL_ID}/{layer}-resid-post-aa/{feature_id}"
    raw_path = root / "api_backfill" / "raw" / f"L{layer:02d}_F{feature_id:06d}.json.gz"
    retrieved_at = utc_now()
    if raw_path.is_file():
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            envelope = json.load(handle)
        payload = envelope["response"]
        status = int(envelope["http_status"])
        attempts = int(envelope["attempts"])
        retrieved_at = str(envelope["retrieved_at_utc"])
    else:
        last_error: Exception | None = None
        for attempts in range(1, MAX_ATTEMPTS + 1):
            try:
                with open_url(url, timeout=60) as response:
                    payload = json.load(response)
                    status = int(response.status)
                break
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in {429, 500, 502, 503, 504}:
                    raise
            except Exception as error:
                last_error = error
            if attempts < MAX_ATTEMPTS:
                time.sleep(2 ** (attempts - 1) + random.random())
        else:
            raise RuntimeError(f"API failed {url}: {last_error!r}")
        envelope = {
            "request_url": url,
            "http_status": status,
            "attempts": attempts,
            "retrieved_at_utc": retrieved_at,
            "response": payload,
        }
        atomic_gzip_json(raw_path, envelope)

    source = payload.get("source") or {}
    if source.get("id") != f"{layer}-resid-post-aa" or int(payload.get("index")) != feature_id:
        raise ValueError(f"Live API identity mismatch: L{layer}/F{feature_id}")
    explanations = payload.get("explanations") or []
    nonempty = [item for item in explanations if str(item.get("description") or "").strip()]
    audit = {
        "sae_layer": layer,
        "feature_id": feature_id,
        "request_url": url,
        "retrieved_at_utc": retrieved_at,
        "http_status": status,
        "attempts": attempts,
        "explanation_records": len(explanations),
        "nonempty_explanations": len(nonempty),
        "activation_records": len(payload.get("activations") or []),
        "raw_path": str(raw_path),
        "raw_sha256": sha256(raw_path),
        "error": "",
    }
    return audit, nonempty


def api_backfill(root: Path, db_path: Path, missing: list[tuple[int, int]]) -> None:
    started = time.perf_counter()
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=API_WORKERS) as executor:
        futures = [executor.submit(api_fetch_missing, root, layer, feature_id) for layer, feature_id in missing]
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            completed.append(future.result())
            if number == 1 or number % 25 == 0 or number == len(futures):
                explained = sum(bool(explanations) for _, explanations in completed)
                print(
                    f"api {number}/{len(futures)}; newly_explained={explained}; "
                    f"{time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    completed.sort(key=lambda item: (item[0]["sae_layer"], item[0]["feature_id"]))
    connection = sqlite3.connect(db_path)
    try:
        connection.executemany(
            """
            INSERT INTO api_recheck VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [tuple(audit[column] for column in (
                "sae_layer", "feature_id", "request_url", "retrieved_at_utc", "http_status",
                "attempts", "explanation_records", "nonempty_explanations", "activation_records",
                "raw_path", "raw_sha256", "error"
            )) for audit, _ in completed],
        )
        live_rows = []
        for audit, explanations in completed:
            for item in explanations:
                live_rows.append(
                    (
                        str(item["id"]),
                        audit["sae_layer"],
                        audit["feature_id"],
                        str(item["description"]).strip(),
                        item.get("explanationModelName"),
                        item.get("typeName"),
                        item.get("createdAt"),
                        "live_api",
                        audit["request_url"],
                    )
                )
        connection.executemany(
            "INSERT OR REPLACE INTO explanation_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            live_rows,
        )
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("api_checked_features", str(len(missing))))
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("api_newly_explained_features", str(sum(bool(explanations) for _, explanations in completed))))
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("api_completed_at_utc", utc_now()))
        connection.commit()
    finally:
        connection.close()


def finalize(root: Path, db_path: Path, downloaded: list[dict]) -> dict:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM feature_semantics")
        feature_rows = []
        primary = {}
        counts = defaultdict(int)
        query = """
            SELECT sae_layer, feature_id, explanation_id, description, source_kind
            FROM explanation_records
            ORDER BY sae_layer, feature_id,
                     CASE source_kind WHEN 'live_api' THEN 0 ELSE 1 END,
                     created_at, explanation_id
        """
        for layer, feature_id, explanation_id, description, source_kind in connection.execute(query):
            key = (int(layer), int(feature_id))
            counts[key] += 1
            primary.setdefault(key, (description, explanation_id, source_kind))
        for layer in LAYERS:
            for feature_id in range(DICT_SIZE):
                key = (layer, feature_id)
                if key in primary:
                    description, explanation_id, source_kind = primary[key]
                    feature_rows.append((layer, feature_id, "np_explained", description, explanation_id, source_kind, counts[key]))
                else:
                    feature_rows.append((layer, feature_id, "np_missing", None, None, None, 0))
                if len(feature_rows) >= 20_000:
                    connection.executemany("INSERT INTO feature_semantics VALUES (?, ?, ?, ?, ?, ?, ?)", feature_rows)
                    feature_rows.clear()
        if feature_rows:
            connection.executemany("INSERT INTO feature_semantics VALUES (?, ?, ?, ?, ?, ?, ?)", feature_rows)
        connection.commit()

        by_layer = []
        for layer in LAYERS:
            explained = connection.execute(
                "SELECT COUNT(*) FROM feature_semantics WHERE sae_layer=? AND coverage_status='np_explained'",
                (layer,),
            ).fetchone()[0]
            missing = DICT_SIZE - explained
            by_layer.append(
                {
                    "sae_layer": layer,
                    "source_id": f"{layer}-resid-post-aa",
                    "total_features": DICT_SIZE,
                    "explained_features": explained,
                    "missing_features": missing,
                    "coverage_rate": explained / DICT_SIZE,
                    "coverage_percent": explained / DICT_SIZE * 100,
                }
            )
        total_rows = connection.execute("SELECT COUNT(*) FROM feature_semantics").fetchone()[0]
        explanation_records = connection.execute("SELECT COUNT(*) FROM explanation_records").fetchone()[0]
        api_rows = connection.execute("SELECT COUNT(*) FROM api_recheck").fetchone()[0]
        api_errors = connection.execute("SELECT COUNT(*) FROM api_recheck WHERE error <> '' OR http_status <> 200").fetchone()[0]
        duplicate_keys = connection.execute(
            "SELECT COUNT(*) FROM (SELECT sae_layer, feature_id, COUNT(*) n FROM feature_semantics GROUP BY 1,2 HAVING n>1)"
        ).fetchone()[0]
        missing_rows = list(
            connection.execute(
                "SELECT sae_layer, feature_id FROM feature_semantics WHERE coverage_status='np_missing' ORDER BY 1,2"
            )
        )
        if total_rows != len(LAYERS) * DICT_SIZE or duplicate_keys != 0 or api_errors != 0:
            raise RuntimeError(
                f"Validation failed: rows={total_rows}, duplicate_keys={duplicate_keys}, api_errors={api_errors}"
            )
        connection.execute("ANALYZE")
        connection.commit()
    finally:
        connection.close()

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    coverage_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(coverage_buffer, fieldnames=list(by_layer[0]))
    writer.writeheader()
    writer.writerows(by_layer)
    coverage_path = reports / "coverage_by_layer.csv"
    atomic_text(coverage_path, coverage_buffer.getvalue())

    missing_buffer = io.StringIO(newline="")
    missing_writer = csv.writer(missing_buffer)
    missing_writer.writerow(["sae_layer", "feature_id"])
    missing_writer.writerows(missing_rows)
    missing_path = reports / "missing_feature_ids.csv"
    atomic_text(missing_path, missing_buffer.getvalue())

    explained_total = sum(item["explained_features"] for item in by_layer)
    manifest = {
        "status": "complete",
        "created_at_utc": utc_now(),
        "model_id": MODEL_ID,
        "layers": list(LAYERS),
        "features_per_layer": DICT_SIZE,
        "total_features": len(LAYERS) * DICT_SIZE,
        "explained_features": explained_total,
        "missing_features": len(missing_rows),
        "coverage_rate": explained_total / (len(LAYERS) * DICT_SIZE),
        "coverage_percent": explained_total / (len(LAYERS) * DICT_SIZE) * 100,
        "explanation_records": explanation_records,
        "api_checked_features": api_rows,
        "api_errors": api_errors,
        "raw_s3_files": len(downloaded),
        "raw_s3_bytes": sum(item["size"] for item in downloaded),
        "database": {"path": str(db_path), "bytes": db_path.stat().st_size, "sha256": sha256(db_path)},
        "coverage_csv": {"path": str(coverage_path), "sha256": sha256(coverage_path)},
        "missing_csv": {"path": str(missing_path), "sha256": sha256(missing_path)},
        "validation": {
            "feature_rows": total_rows,
            "expected_feature_rows": len(LAYERS) * DICT_SIZE,
            "duplicate_feature_keys": duplicate_keys,
            "api_errors": api_errors,
        },
        "by_layer": by_layer,
        "semantic_policy": "Neuronpedia explanations only; no project-corpus context verification",
    }
    manifest_path = root / "manifests" / "final_manifest.json"
    atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("NEURONPEDIA_SEMANTICS_ROOT", "data/neuronpedia_gpt_oss_20b_v1")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    inventory = build_object_inventory()
    downloaded = download_all(root, inventory)
    db_path, missing = create_export_database(root, downloaded)
    print(f"export parse complete; missing={len(missing)}", flush=True)
    api_backfill(root, db_path, missing)
    manifest = finalize(root, db_path, downloaded)
    print(f"wall_seconds={time.perf_counter()-started:.1f}", flush=True)
    print(f"manifest={root / 'manifests' / 'final_manifest.json'}", flush=True)
    print(f"database={manifest['database']['path']}", flush=True)


if __name__ == "__main__":
    main()
