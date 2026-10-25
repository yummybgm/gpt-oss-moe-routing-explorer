#!/usr/bin/env python3
"""Step 03D: ordered and unordered within-layer Top-4 expert combinations."""
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
TOP_TABLE = TABLES / "routing_top4_combination_top50.csv"
SUMMARY_TABLE = TABLES / "routing_top4_combination_summary.csv"
JSD_TABLE = TABLES / "routing_top4_combination_domain_jsd.csv"
MANIFEST = METADATA / "routing_path_step03d_manifest.json"
FIGURE_STEM = "routing_top4_combination_diversity"
CODE_SPACE = N_EXPERTS ** TOP_K
TOP_N = 50


def encode_combinations(expert_ids: np.ndarray, ordered: bool) -> np.ndarray:
    values = expert_ids.astype(np.int32, copy=False)
    if not ordered:
        values = np.sort(values, axis=1)
    codes = (((values[:, 0] * N_EXPERTS + values[:, 1]) * N_EXPERTS + values[:, 2]) * N_EXPERTS + values[:, 3])
    require(codes.min() >= 0 and codes.max() < CODE_SPACE, "Combination code outside expected range")
    return codes.astype(np.int32, copy=False)


def decode_combination(code: int) -> tuple[int, int, int, int]:
    values = [0] * TOP_K
    remainder = int(code)
    for index in range(TOP_K - 1, -1, -1):
        values[index] = remainder % N_EXPERTS
        remainder //= N_EXPERTS
    return tuple(values)  # type: ignore[return-value]


def entropy_bits(probability: np.ndarray) -> float:
    positive = probability > 0
    return float(-(probability[positive] * np.log2(probability[positive])).sum())


def population_distributions(
    codes: np.ndarray,
    mask: np.ndarray,
    token_sample_index: np.ndarray,
    sample_domain_code: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    selected_codes = codes[mask]
    selected_samples = token_sample_index[mask]
    token_counts = np.bincount(selected_samples, minlength=N_SAMPLES).astype(np.int64)
    require((token_counts > 0).all(), "At least one sample has no valid tokens")
    boundaries = np.concatenate(([0], np.cumsum(token_counts)))
    require(boundaries[-1] == len(selected_codes), "Selected-token boundaries mismatch")
    active_codes = np.unique(selected_codes)
    positions = np.searchsorted(active_codes, selected_codes)
    require(np.array_equal(active_codes[positions], selected_codes), "Active-code mapping failed")

    token_weighted: list[np.ndarray] = []
    sample_equal: list[np.ndarray] = []
    raw_counts: list[np.ndarray] = []
    for domain_code in range(len(DOMAINS)):
        domain_samples = np.flatnonzero(sample_domain_code == domain_code)
        require(len(domain_samples) == 400, "Unexpected per-domain sample count")
        domain_selected = sample_domain_code[selected_samples] == domain_code
        counts = np.bincount(positions[domain_selected], minlength=len(active_codes)).astype(np.int64)
        probability_sum = np.zeros(len(active_codes), dtype=np.float64)
        for sample_index in domain_samples:
            start, end = int(boundaries[sample_index]), int(boundaries[sample_index + 1])
            local_positions = positions[start:end]
            unique_positions, local_counts = np.unique(local_positions, return_counts=True)
            probability_sum[unique_positions] += local_counts / len(local_positions)
        token_distribution = counts / counts.sum()
        sample_distribution = probability_sum / len(domain_samples)
        require(abs(float(token_distribution.sum()) - 1.0) < 1e-12, "Token-weighted combination distribution invalid")
        require(abs(float(sample_distribution.sum()) - 1.0) < 1e-12, "Sample-equal combination distribution invalid")
        token_weighted.append(token_distribution)
        sample_equal.append(sample_distribution)
        raw_counts.append(counts)
    return active_codes, token_weighted, sample_equal, raw_counts, token_counts


def self_test() -> None:
    values = np.array([[3, 1, 2, 0], [0, 1, 2, 3]], dtype=np.uint16)
    ordered = encode_combinations(values, ordered=True)
    unordered = encode_combinations(values, ordered=False)
    require(ordered[0] != ordered[1], "Ordered combinations incorrectly collapsed")
    require(unordered[0] == unordered[1], "Unordered combinations failed to collapse permutations")
    require(decode_combination(int(ordered[0])) == (3, 1, 2, 0), "Combination decode failed")
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
    require(np.all(token_sample_index[:-1] <= token_sample_index[1:]), "Tokens are not grouped by sample")
    domain_codes = sample_domain_codes(samples)
    populations = [
        ("primary_inference", annotations["mask_primary_inference"].to_numpy(bool)),
        ("all_tokens", annotations["mask_all_tokens"].to_numpy(bool)),
    ]

    top_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    jsd_rows: list[dict[str, object]] = []

    for layer in range(N_LAYERS):
        for combination_type, ordered in (("ordered", True), ("unordered", False)):
            codes = encode_combinations(expert_id[:, layer, :], ordered=ordered)
            for population, mask in populations:
                active_codes, token_weighted, sample_equal, raw_counts, token_counts = population_distributions(
                    codes, mask, token_sample_index, domain_codes
                )
                for weighting, distributions in (("token_weighted", token_weighted), ("sample_equal", sample_equal)):
                    for domain_code, domain in enumerate(DOMAINS):
                        probability = distributions[domain_code]
                        counts = raw_counts[domain_code]
                        order = np.argsort(-probability, kind="stable")
                        positive_order = order[probability[order] > 0]
                        top_indices = positive_order[:TOP_N]
                        entropy = entropy_bits(probability)
                        domain_sample_mask = domain_codes == domain_code
                        n_tokens = int(token_counts[domain_sample_mask].sum())
                        unique_count = int(np.count_nonzero(counts))
                        summary_rows.append({
                            "population": population,
                            "weighting": weighting,
                            "domain": domain,
                            "layer": layer,
                            "combination_type": combination_type,
                            "n_tokens": n_tokens,
                            "n_samples": 400,
                            "unique_combinations": unique_count,
                            "entropy_bits": entropy,
                            "effective_combination_count": float(2.0 ** entropy),
                            "top1_mass": float(probability[positive_order[:1]].sum()),
                            "top10_mass": float(probability[positive_order[:10]].sum()),
                            "top50_mass": float(probability[top_indices].sum()),
                        })
                        for rank, active_index in enumerate(top_indices, start=1):
                            combination = decode_combination(int(active_codes[active_index]))
                            top_rows.append({
                                "population": population,
                                "weighting": weighting,
                                "domain": domain,
                                "layer": layer,
                                "combination_type": combination_type,
                                "rank": rank,
                                "combination_code": int(active_codes[active_index]),
                                "combination": "-".join(map(str, combination)),
                                "expert_0": combination[0],
                                "expert_1": combination[1],
                                "expert_2": combination[2],
                                "expert_3": combination[3],
                                "raw_token_count": int(counts[active_index]),
                                "probability": float(probability[active_index]),
                                "n_tokens": n_tokens,
                                "n_samples": 400,
                            })
                        other_probability = float(1.0 - probability[top_indices].sum())
                        other_count = int(counts.sum() - counts[top_indices].sum())
                        top_rows.append({
                            "population": population,
                            "weighting": weighting,
                            "domain": domain,
                            "layer": layer,
                            "combination_type": combination_type,
                            "rank": TOP_N + 1,
                            "combination_code": -1,
                            "combination": "__OTHER__",
                            "expert_0": -1, "expert_1": -1, "expert_2": -1, "expert_3": -1,
                            "raw_token_count": other_count,
                            "probability": other_probability,
                            "n_tokens": n_tokens,
                            "n_samples": 400,
                        })

                    for left_code, right_code in combinations(range(len(DOMAINS)), 2):
                        jsd_rows.append({
                            "population": population,
                            "weighting": weighting,
                            "layer": layer,
                            "combination_type": combination_type,
                            "domain_a": DOMAINS[left_code],
                            "domain_b": DOMAINS[right_code],
                            "jsd_bits": js_divergence_bits(distributions[left_code], distributions[right_code]),
                        })

    top = pd.DataFrame(top_rows)
    summary = pd.DataFrame(summary_rows)
    jsd = pd.DataFrame(jsd_rows)
    require(len(summary) == 2 * 2 * len(DOMAINS) * N_LAYERS * 2, "Unexpected combination-summary rows")
    require(len(jsd) == 2 * 2 * N_LAYERS * 2 * 10, "Unexpected combination-JSD rows")
    require(len(top) == len(summary) * (TOP_N + 1), "Unexpected top-combination rows")
    for frame, name in ((top, "top"), (summary, "summary"), (jsd, "jsd")):
        require(frame.isna().sum().sum() == 0, f"Missing values in {name}")
        require(frame.select_dtypes(include=[np.number]).apply(np.isfinite).all().all(), f"Nonfinite values in {name}")
    top_group = ["population", "weighting", "domain", "layer", "combination_type"]
    require(np.allclose(top.groupby(top_group).probability.sum().to_numpy(), 1.0, atol=1e-12), "Top50 + OTHER probabilities do not sum to one")
    require((summary.unique_combinations >= TOP_N).all(), "A group contains fewer than Top-N combinations")

    atomic_csv(top, TOP_TABLE)
    atomic_csv(summary, SUMMARY_TABLE)
    atomic_csv(jsd, JSD_TABLE)

    focus = summary[(summary.population == "primary_inference") & (summary.weighting == "sample_equal")]
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)
    for axis, combination_type in zip(axes, ("ordered", "unordered")):
        selected_type = focus[focus.combination_type == combination_type]
        for domain in DOMAINS:
            selected = selected_type[selected_type.domain == domain]
            axis.plot(selected.layer, selected.effective_combination_count, marker="o", markersize=3, linewidth=1.5, label=domain)
        axis.set_ylabel("Effective combination count")
        axis.set_title(f"{combination_type.capitalize()} Top-4 combinations")
        axis.grid(alpha=.22)
    axes[-1].set_xlabel("Layer")
    axes[-1].set_xticks(range(N_LAYERS))
    axes[0].legend(ncol=3, fontsize=8)
    figure.suptitle("Within-layer Top-4 combination diversity (primary inference, sample equal)")
    figure.tight_layout()
    png = FIG_PNG / f"{FIGURE_STEM}.png"
    svg = FIG_SVG / f"{FIGURE_STEM}.svg"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    plt.close(figure)

    outputs = [TOP_TABLE, SUMMARY_TABLE, JSD_TABLE, png, svg]
    manifest = {
        "status": "complete",
        "step": "03D_top4_combinations",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "ordered": "Base-32 code preserving the four stored routing-slot positions",
            "unordered": "Base-32 code after sorting the four unique expert IDs",
            "reported_categories": "Top 50 combinations plus __OTHER__ per analysis group",
            "full_path_excluded": True,
        },
        "script": script_record(SCRIPT),
        "shared_script": script_record(Path(__file__).with_name("_03_common.py")),
        "outputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs},
        "row_counts": {str(TOP_TABLE): len(top), str(SUMMARY_TABLE): len(summary), str(JSD_TABLE): len(jsd)},
        "validation": {
            "missing_values": 0, "nonfinite_numeric": 0, "top50_plus_other_sum_to_one": True,
            "primary_tokens": int(annotations.mask_primary_inference.sum()),
            "all_tokens": int(annotations.mask_all_tokens.sum()),
        },
        "raw_parquet_read": False, "model_loaded": False, "model_inference_rerun": False, "dense_sae_matrix_created": False,
    }
    atomic_json(manifest, MANIFEST)

    print("FOCUS_LAYER_11_12")
    print(focus[focus.layer.isin([11, 12])].to_string(index=False))
    focus_jsd = jsd[(jsd.population == "primary_inference") & (jsd.weighting == "sample_equal")]
    print("FOCUS_JSD_RANGE")
    print(focus_jsd.groupby(["layer", "combination_type"]).jsd_bits.agg(["min", "mean", "max"]).loc[[11, 12]].to_string())


if __name__ == "__main__":
    main()
