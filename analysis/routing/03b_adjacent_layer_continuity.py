#!/usr/bin/env python3
"""Step 03B: adjacent-layer Top-4 ID overlap and retained-count analysis."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _03_common import *

SCRIPT = Path(__file__).resolve()
MAIN_TABLE = TABLES / "routing_adjacent_layer_continuity.csv"
RETENTION_TABLE = TABLES / "routing_adjacent_retention_distribution.csv"
SAMPLE_TABLE = CACHE / "routing_adjacent_per_sample.parquet"
SUMMARY_TABLE = TABLES / "routing_adjacent_summary_by_domain.csv"
MANIFEST = METADATA / "routing_path_step03b_manifest.json"
FIGURE_STEM = "routing_adjacent_layer_continuity"


def per_sample_frame(population: str, mask: np.ndarray, retained: np.ndarray, jaccard: np.ndarray,
                     token_sample_index: np.ndarray, samples: pd.DataFrame) -> pd.DataFrame:
    token_counts = np.bincount(token_sample_index[mask], minlength=N_SAMPLES).astype(np.int32)
    valid = token_counts > 0
    rows = []
    selected_samples = token_sample_index[mask]
    for layer in range(N_LAYERS - 1):
        jac_sum = np.bincount(selected_samples, weights=jaccard[mask, layer], minlength=N_SAMPLES)
        ret_sum = np.bincount(selected_samples, weights=retained[mask, layer], minlength=N_SAMPLES)
        jac_mean = np.divide(jac_sum, token_counts, out=np.zeros(N_SAMPLES), where=valid)
        ret_mean = np.divide(ret_sum, token_counts, out=np.zeros(N_SAMPLES), where=valid)
        shares = []
        for count in range(TOP_K + 1):
            count_by_sample = np.bincount(selected_samples, weights=(retained[mask, layer] == count), minlength=N_SAMPLES)
            shares.append(np.divide(count_by_sample, token_counts, out=np.zeros(N_SAMPLES), where=valid))
        for sample_index in np.flatnonzero(valid):
            rows.append({
                "population": population,
                "sample_id": str(samples.iloc[sample_index]["sample_id"]),
                "domain": str(samples.iloc[sample_index]["domain"]),
                "from_layer": layer,
                "to_layer": layer + 1,
                "valid_token_count": int(token_counts[sample_index]),
                "mean_jaccard": float(jac_mean[sample_index]),
                "mean_retained_count": float(ret_mean[sample_index]),
                **{f"retained_{count}_share": float(shares[count][sample_index]) for count in range(TOP_K + 1)},
            })
    return pd.DataFrame(rows)


def aggregate_tables(population: str, mask: np.ndarray, retained: np.ndarray, jaccard: np.ndarray,
                     token_sample_index: np.ndarray, samples: pd.DataFrame, domain_codes: np.ndarray,
                     sample_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_domains = domain_codes[token_sample_index]
    main_rows = []
    retention_rows = []
    for domain_code, domain in enumerate(DOMAINS):
        domain_mask = mask & (token_domains == domain_code)
        for layer in range(N_LAYERS - 1):
            jac_values = jaccard[domain_mask, layer]
            ret_values = retained[domain_mask, layer]
            main_rows.append({
                "population": population, "weighting": "token_weighted", "domain": domain,
                "from_layer": layer, "to_layer": layer + 1, "n_tokens": int(len(jac_values)),
                "n_samples": int(samples["domain"].eq(domain).sum()),
                "mean_jaccard": float(jac_values.mean()), "sd_jaccard": float(jac_values.std(ddof=1)),
                "mean_retained_count": float(ret_values.mean()), "sd_retained_count": float(ret_values.std(ddof=1)),
            })
            for count in range(TOP_K + 1):
                retention_rows.append({
                    "population": population, "weighting": "token_weighted", "domain": domain,
                    "from_layer": layer, "to_layer": layer + 1, "retained_count": count,
                    "share": float((ret_values == count).mean()), "n_tokens": int(len(ret_values)),
                    "n_samples": int(samples["domain"].eq(domain).sum()),
                })

            selected = sample_frame[(sample_frame["domain"] == domain) & (sample_frame["from_layer"] == layer)]
            require(len(selected) == 400, f"Unexpected sample rows for {population}/{domain}/{layer}: {len(selected)}")
            main_rows.append({
                "population": population, "weighting": "sample_equal", "domain": domain,
                "from_layer": layer, "to_layer": layer + 1,
                "n_tokens": int(selected["valid_token_count"].sum()), "n_samples": int(len(selected)),
                "mean_jaccard": float(selected["mean_jaccard"].mean()), "sd_jaccard": float(selected["mean_jaccard"].std(ddof=1)),
                "mean_retained_count": float(selected["mean_retained_count"].mean()), "sd_retained_count": float(selected["mean_retained_count"].std(ddof=1)),
            })
            for count in range(TOP_K + 1):
                retention_rows.append({
                    "population": population, "weighting": "sample_equal", "domain": domain,
                    "from_layer": layer, "to_layer": layer + 1, "retained_count": count,
                    "share": float(selected[f"retained_{count}_share"].mean()),
                    "n_tokens": int(selected["valid_token_count"].sum()), "n_samples": int(len(selected)),
                })
    return pd.DataFrame(main_rows), pd.DataFrame(retention_rows)


def self_test() -> None:
    expert = np.zeros((3, N_LAYERS, TOP_K), dtype=np.uint16)
    expert[:, :, :] = np.array([0, 1, 2, 3])
    expert[1, 1:, :] = np.array([0, 1, 2, 4])
    expert[2, 1:, :] = np.array([4, 5, 6, 7])
    retained, jaccard = adjacent_metrics(expert)
    require(retained[:, 0].tolist() == [4, 3, 0], "Synthetic retained counts failed")
    require(np.allclose(jaccard[:, 0], [1.0, 3 / 5, 0.0]), "Synthetic Jaccard failed")
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: self_test(); return
    ensure_dirs()
    columns = ["sample_id", "token_position", "mask_all_tokens", "mask_primary_inference"]
    samples, annotations, routing, manifests = load_inputs(columns)
    expert_id = routing["expert_id"].astype(np.int16, copy=False)
    token_sample_index = routing["token_sample_index"].astype(np.int32, copy=False)
    domain_codes = sample_domain_codes(samples)
    retained, jaccard = adjacent_metrics(expert_id)
    sample_parts = []; main_parts = []; retention_parts = []
    for population, mask in [
        ("primary_inference", annotations["mask_primary_inference"].to_numpy(bool)),
        ("all_tokens", annotations["mask_all_tokens"].to_numpy(bool)),
    ]:
        per_sample = per_sample_frame(population, mask, retained, jaccard, token_sample_index, samples)
        main_table, retention_table = aggregate_tables(population, mask, retained, jaccard, token_sample_index, samples, domain_codes, per_sample)
        sample_parts.append(per_sample); main_parts.append(main_table); retention_parts.append(retention_table)
    sample_output = pd.concat(sample_parts, ignore_index=True)
    main_output = pd.concat(main_parts, ignore_index=True)
    retention_output = pd.concat(retention_parts, ignore_index=True)
    require(len(sample_output) == 2 * N_SAMPLES * (N_LAYERS - 1), "Unexpected per-sample rows")
    require(len(main_output) == 2 * 2 * len(DOMAINS) * (N_LAYERS - 1), "Unexpected main rows")
    require(len(retention_output) == len(main_output) * (TOP_K + 1), "Unexpected retention rows")
    require(main_output.select_dtypes(include=[np.number]).apply(np.isfinite).all().all(), "Nonfinite main numeric values")
    require(retention_output.select_dtypes(include=[np.number]).apply(np.isfinite).all().all(), "Nonfinite retention numeric values")
    require(np.allclose(retention_output.groupby(["population", "weighting", "domain", "from_layer"])["share"].sum(), 1.0), "Retention shares do not sum to one")

    summary = main_output.groupby(["population", "weighting", "domain"], as_index=False).agg(
        mean_adjacent_jaccard=("mean_jaccard", "mean"), min_adjacent_jaccard=("mean_jaccard", "min"),
        max_adjacent_jaccard=("mean_jaccard", "max"), mean_retained_count=("mean_retained_count", "mean"))
    for layer, name in [(10, "jaccard_10_to_11"), (11, "jaccard_11_to_12")]:
        focused = main_output[main_output["from_layer"].eq(layer)][["population", "weighting", "domain", "mean_jaccard"]].rename(columns={"mean_jaccard": name})
        summary = summary.merge(focused, on=["population", "weighting", "domain"], how="left", validate="one_to_one")

    atomic_csv(main_output, MAIN_TABLE); atomic_csv(retention_output, RETENTION_TABLE)
    atomic_parquet(sample_output, SAMPLE_TABLE); atomic_csv(summary, SUMMARY_TABLE)

    plot = main_output[(main_output["population"] == "primary_inference") & (main_output["weighting"] == "sample_equal")]
    fig, axis = plt.subplots(figsize=(11.5, 4.2))
    for domain in DOMAINS:
        selected = plot[plot["domain"] == domain]
        axis.plot(selected["from_layer"], selected["mean_jaccard"], marker="o", markersize=3, linewidth=1.8, label=domain)
    axis.set(title="Adjacent-layer Top-4 ID overlap", xlabel="From layer (to next layer)", ylabel="Mean Jaccard")
    axis.set_xticks(range(N_LAYERS - 1)); axis.grid(alpha=.22); axis.legend(ncol=3, fontsize=8); fig.tight_layout()
    png = FIG_PNG / f"{FIGURE_STEM}.png"; svg = FIG_SVG / f"{FIGURE_STEM}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(svg, bbox_inches="tight"); plt.close(fig)

    outputs = [MAIN_TABLE, RETENTION_TABLE, SAMPLE_TABLE, SUMMARY_TABLE, png, svg]
    manifest = {
        "status": "complete", "step": "03B_adjacent_layer_continuity", "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {"retained_count": "intersection size of adjacent Top-4 expert-ID sets", "jaccard": "retained_count / (8 - retained_count)",
                       "cross_layer_identity_warning": "Equal numeric IDs at different layers are different parameter modules."},
        "script": script_record(SCRIPT), "shared_script": script_record(Path(__file__).with_name("_03_common.py")),
        "outputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs},
        "row_counts": {str(MAIN_TABLE): len(main_output), str(RETENTION_TABLE): len(retention_output), str(SAMPLE_TABLE): len(sample_output), str(SUMMARY_TABLE): len(summary)},
        "validation": {"missing_values": int(main_output.isna().sum().sum() + retention_output.isna().sum().sum()), "nonfinite_numeric": 0,
                       "retention_shares_sum_to_one": True, "primary_tokens": int(annotations["mask_primary_inference"].sum()), "all_tokens": int(annotations["mask_all_tokens"].sum())},
        "raw_parquet_read": False, "model_loaded": False, "model_inference_rerun": False, "dense_sae_matrix_created": False,
    }
    atomic_json(manifest, MANIFEST)
    focus = summary[(summary["population"] == "primary_inference") & (summary["weighting"] == "sample_equal")]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
