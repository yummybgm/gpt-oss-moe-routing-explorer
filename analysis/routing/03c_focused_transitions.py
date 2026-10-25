#!/usr/bin/env python3
"""Step 03C: focused 10->11 and 11->12 Top-4 cross-layer co-occurrence matrices."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _03_common import *

SCRIPT = Path(__file__).resolve()
MATRIX_TABLE = TABLES / "routing_focused_transition_matrices.csv"
SUMMARY_TABLE = TABLES / "routing_focused_transition_summary.csv"
TOP_TABLE = TABLES / "routing_focused_transition_top_pairs.csv"
JSD_TABLE = TABLES / "routing_focused_transition_domain_jsd.csv"
MANIFEST = METADATA / "routing_path_step03c_manifest.json"
FIGURE_STEM = "routing_focused_transition_heatmaps"
LAYER_PAIRS = [(10, 11), (11, 12)]
PAIR_SPACE = N_EXPERTS * N_EXPERTS


def transition_codes(expert_id: np.ndarray, from_layer: int, to_layer: int) -> np.ndarray:
    source = expert_id[:, from_layer, :].astype(np.int32, copy=False)
    target = expert_id[:, to_layer, :].astype(np.int32, copy=False)
    codes = (source[:, :, None] * N_EXPERTS + target[:, None, :]).reshape(len(expert_id), TOP_K * TOP_K)
    require(codes.min() >= 0 and codes.max() < PAIR_SPACE, "Transition code outside expected range")
    return codes.astype(np.uint16, copy=False)


def distributions_for_population(
    codes: np.ndarray,
    mask: np.ndarray,
    token_sample_index: np.ndarray,
    sample_domain_code: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    selected_codes = codes[mask].astype(np.int32, copy=False)
    selected_samples = token_sample_index[mask]
    token_counts = np.bincount(selected_samples, minlength=N_SAMPLES).astype(np.int64)
    require((token_counts > 0).all(), "At least one sample has no valid tokens")
    composite = selected_samples[:, None] * PAIR_SPACE + selected_codes
    per_sample_counts = np.bincount(composite.ravel(), minlength=N_SAMPLES * PAIR_SPACE).reshape(N_SAMPLES, PAIR_SPACE)
    require(np.array_equal(per_sample_counts.sum(axis=1), token_counts * TOP_K * TOP_K), "Per-sample transition counts mismatch")
    per_sample_probability = per_sample_counts / (token_counts[:, None] * TOP_K * TOP_K)

    token_weighted: dict[int, np.ndarray] = {}
    sample_equal: dict[int, np.ndarray] = {}
    raw_counts: dict[int, np.ndarray] = {}
    for domain_code in range(len(DOMAINS)):
        sample_mask = sample_domain_code == domain_code
        require(sample_mask.sum() == 400, f"Unexpected sample count for domain code {domain_code}")
        domain_counts = per_sample_counts[sample_mask].sum(axis=0)
        token_weighted[domain_code] = domain_counts / domain_counts.sum()
        sample_equal[domain_code] = per_sample_probability[sample_mask].mean(axis=0)
        raw_counts[domain_code] = domain_counts
    return token_weighted, sample_equal, raw_counts, token_counts


def entropy_bits(probability: np.ndarray) -> float:
    positive = probability > 0
    return float(-(probability[positive] * np.log2(probability[positive])).sum())


def self_test() -> None:
    expert = np.zeros((1, N_LAYERS, TOP_K), dtype=np.uint16)
    expert[0, 10] = [0, 1, 2, 3]
    expert[0, 11] = [4, 5, 6, 7]
    codes = transition_codes(expert, 10, 11)
    expected = sorted(source * N_EXPERTS + target for source in range(4) for target in range(4, 8))
    require(sorted(codes[0].tolist()) == expected, "Synthetic Cartesian transition codes failed")
    counts = np.bincount(codes.ravel(), minlength=PAIR_SPACE)
    require(counts.sum() == 16 and np.count_nonzero(counts) == 16, "Synthetic transition count failed")
    require(abs(entropy_bits(counts / counts.sum()) - 4.0) < 1e-12, "Synthetic entropy failed")
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    ensure_dirs()
    columns = ["sample_id", "token_position", "mask_all_tokens", "mask_primary_inference"]
    samples, annotations, routing, _ = load_inputs(columns)
    expert_id = routing["expert_id"]
    token_sample_index = routing["token_sample_index"].astype(np.int32, copy=False)
    domain_codes = sample_domain_codes(samples)
    populations = [
        ("primary_inference", annotations["mask_primary_inference"].to_numpy(bool)),
        ("all_tokens", annotations["mask_all_tokens"].to_numpy(bool)),
    ]

    matrix_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    jsd_rows: list[dict[str, object]] = []
    primary_sample_equal_for_plot: dict[tuple[int, int, str], np.ndarray] = {}

    for from_layer, to_layer in LAYER_PAIRS:
        codes = transition_codes(expert_id, from_layer, to_layer)
        for population, mask in populations:
            token_weighted, sample_equal, raw_counts, token_counts = distributions_for_population(
                codes, mask, token_sample_index, domain_codes
            )
            for weighting, distributions in (("token_weighted", token_weighted), ("sample_equal", sample_equal)):
                for domain_code, domain in enumerate(DOMAINS):
                    probability = distributions[domain_code]
                    counts = raw_counts[domain_code]
                    require(np.isfinite(probability).all() and (probability >= 0).all(), "Invalid transition probability")
                    require(abs(float(probability.sum()) - 1.0) < 1e-12, "Transition probability does not sum to one")
                    domain_samples = domain_codes == domain_code
                    n_tokens = int(token_counts[domain_samples].sum())
                    require(int(counts.sum()) == n_tokens * TOP_K * TOP_K, "Domain pair count mismatch")
                    for code in range(PAIR_SPACE):
                        matrix_rows.append({
                            "population": population,
                            "weighting": weighting,
                            "domain": domain,
                            "from_layer": from_layer,
                            "to_layer": to_layer,
                            "source_expert": code // N_EXPERTS,
                            "target_expert": code % N_EXPERTS,
                            "raw_pair_count": int(counts[code]),
                            "probability": float(probability[code]),
                            "n_tokens": n_tokens,
                            "n_samples": 400,
                        })
                    order = np.argsort(-probability, kind="stable")
                    entropy = entropy_bits(probability)
                    summary_rows.append({
                        "population": population,
                        "weighting": weighting,
                        "domain": domain,
                        "from_layer": from_layer,
                        "to_layer": to_layer,
                        "n_tokens": n_tokens,
                        "n_samples": 400,
                        "nonzero_pairs": int(np.count_nonzero(probability)),
                        "entropy_bits": entropy,
                        "effective_pair_count": float(2.0 ** entropy),
                        "top1_mass": float(probability[order[:1]].sum()),
                        "top10_mass": float(probability[order[:10]].sum()),
                        "top20_mass": float(probability[order[:20]].sum()),
                    })
                    for rank, code in enumerate(order[:20], start=1):
                        top_rows.append({
                            "population": population,
                            "weighting": weighting,
                            "domain": domain,
                            "from_layer": from_layer,
                            "to_layer": to_layer,
                            "rank": rank,
                            "source_expert": int(code // N_EXPERTS),
                            "target_expert": int(code % N_EXPERTS),
                            "raw_pair_count": int(counts[code]),
                            "probability": float(probability[code]),
                        })
                    if population == "primary_inference" and weighting == "sample_equal":
                        primary_sample_equal_for_plot[(from_layer, to_layer, domain)] = probability.reshape(N_EXPERTS, N_EXPERTS)

                for left_code, right_code in combinations(range(len(DOMAINS)), 2):
                    jsd_rows.append({
                        "population": population,
                        "weighting": weighting,
                        "from_layer": from_layer,
                        "to_layer": to_layer,
                        "domain_a": DOMAINS[left_code],
                        "domain_b": DOMAINS[right_code],
                        "jsd_bits": js_divergence_bits(distributions[left_code], distributions[right_code]),
                    })

    matrix = pd.DataFrame(matrix_rows)
    summary = pd.DataFrame(summary_rows)
    top = pd.DataFrame(top_rows)
    jsd = pd.DataFrame(jsd_rows)
    require(len(matrix) == 2 * 2 * 2 * len(DOMAINS) * PAIR_SPACE, "Unexpected matrix row count")
    require(len(summary) == 2 * 2 * 2 * len(DOMAINS), "Unexpected summary row count")
    require(len(top) == len(summary) * 20, "Unexpected top-pair row count")
    require(len(jsd) == 2 * 2 * 2 * 10, "Unexpected JSD row count")
    for frame, name in ((matrix, "matrix"), (summary, "summary"), (top, "top"), (jsd, "jsd")):
        require(frame.isna().sum().sum() == 0, f"Missing values in {name}")
        require(frame.select_dtypes(include=[np.number]).apply(np.isfinite).all().all(), f"Nonfinite values in {name}")
    probability_sums = matrix.groupby(["population", "weighting", "domain", "from_layer"])["probability"].sum()
    require(np.allclose(probability_sums.to_numpy(), 1.0, atol=1e-12), "Matrix probabilities fail normalization")
    require((top.groupby(["population", "weighting", "domain", "from_layer"])["probability"].apply(lambda values: values.is_monotonic_decreasing)).all(), "Top pairs not sorted")

    atomic_csv(matrix, MATRIX_TABLE)
    atomic_csv(summary, SUMMARY_TABLE)
    atomic_csv(top, TOP_TABLE)
    atomic_csv(jsd, JSD_TABLE)

    figure, axes = plt.subplots(2, len(DOMAINS), figsize=(16.5, 6.4), sharex=True, sharey=True)
    shared_max = max(float(values.max()) for values in primary_sample_equal_for_plot.values())
    image = None
    for row, (from_layer, to_layer) in enumerate(LAYER_PAIRS):
        for column, domain in enumerate(DOMAINS):
            axis = axes[row, column]
            values = primary_sample_equal_for_plot[(from_layer, to_layer, domain)]
            image = axis.imshow(values, origin="lower", aspect="equal", vmin=0.0, vmax=shared_max, cmap="viridis")
            top_code = int(np.argmax(values))
            axis.scatter([top_code % N_EXPERTS], [top_code // N_EXPERTS], s=22, facecolors="none", edgecolors="white", linewidths=.8)
            if row == 0:
                axis.set_title(domain, fontsize=9)
            if column == 0:
                axis.set_ylabel(f"L{from_layer} source expert")
            if row == len(LAYER_PAIRS) - 1:
                axis.set_xlabel(f"L{to_layer} target expert")
            axis.set_xticks([0, 8, 16, 24, 31]); axis.set_yticks([0, 8, 16, 24, 31])
    require(image is not None, "Figure image was not created")
    figure.suptitle("Focused cross-layer Top-4 co-occurrence matrices (primary inference)")
    figure.subplots_adjust(left=.055, right=.885, bottom=.09, top=.9, wspace=.18, hspace=.2)
    colorbar_axis = figure.add_axes([.91, .16, .012, .66])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Sample-equal co-occurrence probability", labelpad=10)
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")
    png = FIG_PNG / f"{FIGURE_STEM}.png"
    svg = FIG_SVG / f"{FIGURE_STEM}.svg"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)

    outputs = [MATRIX_TABLE, SUMMARY_TABLE, TOP_TABLE, JSD_TABLE, png, svg]
    manifest = {
        "status": "complete",
        "step": "03C_focused_transitions",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "transition": "Cartesian co-occurrence between four source-layer and four target-layer expert IDs; 16 pairs per token",
            "interpretation_limit": "Cross-layer ID co-occurrence, not causal flow and not identity of parameter modules across layers",
            "layer_pairs": [list(pair) for pair in LAYER_PAIRS],
        },
        "script": script_record(SCRIPT),
        "shared_script": script_record(Path(__file__).with_name("_03_common.py")),
        "outputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs},
        "row_counts": {
            str(MATRIX_TABLE): len(matrix), str(SUMMARY_TABLE): len(summary),
            str(TOP_TABLE): len(top), str(JSD_TABLE): len(jsd),
        },
        "validation": {
            "missing_values": 0,
            "nonfinite_numeric": 0,
            "matrix_probabilities_sum_to_one": True,
            "primary_tokens": int(annotations["mask_primary_inference"].sum()),
            "all_tokens": int(annotations["mask_all_tokens"].sum()),
        },
        "raw_parquet_read": False,
        "model_loaded": False,
        "model_inference_rerun": False,
        "dense_sae_matrix_created": False,
    }
    atomic_json(manifest, MANIFEST)

    focus_summary = summary[(summary.population == "primary_inference") & (summary.weighting == "sample_equal")]
    focus_jsd = jsd[(jsd.population == "primary_inference") & (jsd.weighting == "sample_equal")]
    print("FOCUS_SUMMARY")
    print(focus_summary.to_string(index=False))
    print("FOCUS_JSD")
    print(focus_jsd.to_string(index=False))
    print("FOCUS_TOP_PAIRS")
    print(top[(top.population == "primary_inference") & (top.weighting == "sample_equal") & (top["rank"] <= 3)].to_string(index=False))


if __name__ == "__main__":
    main()
