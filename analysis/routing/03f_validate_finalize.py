#!/usr/bin/env python3
"""Step 03F: run the validation suite and finalize the cumulative step-03 deliverables."""
from __future__ import annotations

import os
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _03_common import *

SCRIPT = Path(__file__).resolve()
COMPONENT_MANIFESTS = [
    METADATA / "routing_path_step03b_manifest.json",
    METADATA / "routing_path_step03c_manifest.json",
    METADATA / "routing_path_step03d_manifest.json",
    METADATA / "routing_path_step03e_manifest.json",
]
VALIDATORS = [
    "03_validate_inputs.py",
    "03b_validate_outputs.py",
    "03c_validate_outputs.py",
    "03d_validate_outputs.py",
    "03e_validate_outputs.py",
]
SUMMARY_PATH = METADATA / "routing_path_analysis_summary.json"
INVENTORY_PATH = METADATA / "routing_path_artifact_inventory.csv"
REPORT_PATH = REPORT / "step03_routing_path_and_combinations_report_zh.md"
VALIDATION_LOG = LOGS / "03_step03_validator_suite.log"
FINAL_MANIFEST = METADATA / "routing_path_step03_manifest.json"


def atomic_text(text: str, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_validation_suite() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    log_parts = []
    for name in VALIDATORS:
        path = SCRIPT.with_name(name)
        require(path.is_file(), f"Missing validator: {path}")
        completed = subprocess.run(
            [sys.executable, str(path)], cwd=str(SCRIPT.parent),
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        log_parts.append(f"===== {name} =====\nexit_code={completed.returncode}\n{completed.stdout}\n{completed.stderr}\n")
        require(completed.returncode == 0, f"Validator failed: {name}\n{completed.stderr}")
        records.append({
            "validator": name, "status": "passed", "sha256": sha256_file(path),
            "stdout_contains_passed": ("passed" in completed.stdout.lower() or "PASS" in completed.stdout or "SELF_TEST_OK" in completed.stdout),
        })
    atomic_text("\n".join(log_parts), VALIDATION_LOG)
    return records


def component_inventory() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inventory: list[dict[str, object]] = []
    component_records: list[dict[str, object]] = []
    for manifest_path in COMPONENT_MANIFESTS:
        require(manifest_path.is_file(), f"Missing component manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("status") == "complete", f"Incomplete component manifest: {manifest_path}")
        stage = str(manifest["step"])
        for raw_path, metadata in manifest["outputs"].items():
            path = Path(raw_path)
            require(path.is_file(), f"Missing component output: {path}")
            actual_size = path.stat().st_size
            actual_hash = sha256_file(path)
            require(actual_size == metadata["size_bytes"], f"Component size mismatch: {path}")
            require(actual_hash == metadata["sha256"], f"Component hash mismatch: {path}")
            inventory.append({
                "stage": stage, "kind": "component_output", "path": str(path),
                "size_bytes": actual_size, "sha256": actual_hash, "verified": True,
            })
        component_records.append({
            "step": stage, "path": str(manifest_path), "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path), "output_count": len(manifest["outputs"]), "status": "complete",
        })
        inventory.append({
            "stage": stage, "kind": "component_manifest", "path": str(manifest_path),
            "size_bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "verified": True,
        })
    require(sum(int(record["output_count"]) for record in component_records) == 23, "Unexpected component-output count")
    return inventory, component_records


def main() -> None:
    ensure_dirs()
    validation_records = run_validation_suite()
    inventory_rows, component_records = component_inventory()

    adjacent_summary = pd.read_csv(TABLES / "routing_adjacent_summary_by_domain.csv")
    transition_summary = pd.read_csv(TABLES / "routing_focused_transition_summary.csv")
    transition_jsd = pd.read_csv(TABLES / "routing_focused_transition_domain_jsd.csv")
    combination_summary = pd.read_csv(TABLES / "routing_top4_combination_summary.csv")
    combination_jsd = pd.read_csv(TABLES / "routing_top4_combination_domain_jsd.csv")
    control_audit = pd.read_csv(TABLES / "routing_path_control_audit.csv")
    control_summary = pd.read_csv(TABLES / "routing_key_path_control_summary.csv")
    for frame, name in (
        (adjacent_summary, "adjacent_summary"), (transition_summary, "transition_summary"),
        (transition_jsd, "transition_jsd"), (combination_summary, "combination_summary"),
        (combination_jsd, "combination_jsd"), (control_audit, "control_audit"),
        (control_summary, "control_summary"),
    ):
        require(frame.isna().sum().sum() == 0, f"Missing values in final input {name}")
        require(frame.select_dtypes(include=[np.number]).apply(np.isfinite).all().all(), f"Nonfinite final input {name}")

    focus_adjacent = adjacent_summary[
        (adjacent_summary.population == "primary_inference") & (adjacent_summary.weighting == "sample_equal")
    ].copy()
    focus_transition = transition_summary[
        (transition_summary.population == "primary_inference") & (transition_summary.weighting == "sample_equal")
    ].copy()
    focus_transition_jsd = transition_jsd[
        (transition_jsd.population == "primary_inference") & (transition_jsd.weighting == "sample_equal")
    ].copy()
    focus_combination = combination_summary[
        (combination_summary.population == "primary_inference")
        & (combination_summary.weighting == "sample_equal")
        & combination_summary.layer.isin([11, 12])
    ].copy()
    focus_combination_jsd = combination_jsd[
        (combination_jsd.population == "primary_inference")
        & (combination_jsd.weighting == "sample_equal")
        & combination_jsd.layer.isin([11, 12])
    ].copy()
    robust_controls = control_summary[control_summary.control_scope == "robustness"].copy()
    diagnostic_controls = control_audit[(control_audit.control_scope == "diagnostic") & (control_audit.status == "ok_all_domains")].copy()

    summary = {
        "status": "complete",
        "step": "03_routing_path_and_expert_combinations",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Describe adjacent-layer Top-4 ID continuity, focused cross-layer co-occurrence, and within-layer ordered/unordered combinations with frozen controls.",
        "inputs": {"samples": N_SAMPLES, "tokens_all": N_TOKENS, "tokens_primary": N_PRIMARY, "layers": N_LAYERS, "experts": N_EXPERTS, "top_k": TOP_K},
        "step03b_adjacent": {
            row.domain: {"jaccard_10_to_11": float(row.jaccard_10_to_11), "jaccard_11_to_12": float(row.jaccard_11_to_12)}
            for row in focus_adjacent.itertuples()
        },
        "step03c_transition": {
            "entropy_range_bits": [float(focus_transition.entropy_bits.min()), float(focus_transition.entropy_bits.max())],
            "top20_mass_range": [float(focus_transition.top20_mass.min()), float(focus_transition.top20_mass.max())],
            "pairwise_domain_jsd_range_bits": [float(focus_transition_jsd.jsd_bits.min()), float(focus_transition_jsd.jsd_bits.max())],
        },
        "step03d_combinations_layer11_12": {
            "unique_combination_range": [int(focus_combination.unique_combinations.min()), int(focus_combination.unique_combinations.max())],
            "effective_combination_count_range": [float(focus_combination.effective_combination_count.min()), float(focus_combination.effective_combination_count.max())],
            "top50_mass_range": [float(focus_combination.top50_mass.min()), float(focus_combination.top50_mass.max())],
            "pairwise_domain_jsd_range_bits": [float(focus_combination_jsd.jsd_bits.min()), float(focus_combination_jsd.jsd_bits.max())],
        },
        "step03e_controls": {
            "settings": int(len(control_audit)), "analyzable_settings": int(control_audit.status.eq("ok_all_domains").sum()),
            "robustness_settings": int(control_audit.control_scope.eq("robustness").sum()),
            "diagnostic_settings": int(control_audit.control_scope.eq("diagnostic").sum()),
            "robust_mean_pairwise_jsd_range_bits": [float(robust_controls.mean_pairwise_jsd_bits.min()), float(robust_controls.mean_pairwise_jsd_bits.max())],
            "diagnostic_minimum_samples_per_domain_range": [int(diagnostic_controls.minimum_valid_samples_per_present_domain.min()), int(diagnostic_controls.minimum_valid_samples_per_present_domain.max())],
            "frozen_length_matched_samples": 930, "fixed_truncation_n": 50,
        },
        "interpretation": "The results support domain-related routing differences in continuity, cross-layer ID co-occurrence, and within-layer Top-4 combinations.",
        "limitations": [
            "Equal expert IDs at different layers denote different parameter modules.",
            "Cartesian cross-layer co-occurrence is not causal flow.",
            "Domain is confounded with dataset/source and text format in the current design.",
            "Routing differences alone do not establish functional specialization.",
            "Sparse individual token classes are diagnostic only.",
        ],
        "validation_suite": validation_records,
        "component_manifests": component_records,
        "model_inference_rerun": False,
        "dense_sae_matrix_created": False,
    }
    atomic_json(summary, SUMMARY_PATH)

    baseline_control = robust_controls[
        (robust_controls.control_type == "token_filter") & (robust_controls.control_level == "A_all")
    ].set_index(["target_type", "layer"])
    report_text = f"""# 步骤03阶段性报告（累计步骤00–03）

## 📍 进度与双语解读

当前位置：步骤03——Routing path 与专家组合，已完成最终复核。

工作解读：完成了 adjacent-layer continuity（相邻层Top-4编号集合重叠）、focused cross-layer co-occurrence（重点层编号共现矩阵）、ordered/unordered Top-4 combinations（保留槽位顺序或只看专家集合）以及 control sensitivity（Token筛选、Prompt位置和长度敏感性检查）。

## 步骤00–02累计基础

- 步骤00：构建统一缓存，确认2,000个样本、382,685个Token、36,737,760条Routing记录和72,151,749条稀疏SAE激活；未重跑模型推理。
- 步骤01：完成全部Token分类与12套掩码；正式推断排除每个样本首Token，保留380,685个Token。
- 步骤02：确认领域相关的路由差异；最强层为第12层（η²=0.513575），第11层η²=0.491660，平均两两JSD=0.232272 bits。

## 步骤03结果

### 03B 相邻层连续性

- 23对相邻层均完成Top-4集合Jaccard和保留专家数统计。
- 样本等权下，10→11层Jaccard为{focus_adjacent.jaccard_10_to_11.min():.6f}–{focus_adjacent.jaccard_10_to_11.max():.6f}；11→12层为{focus_adjacent.jaccard_11_to_12.min():.6f}–{focus_adjacent.jaccard_11_to_12.max():.6f}。
- 相邻层编号重叠整体较低；这不是参数模块在层间延续的证据。

### 03C 重点层编号共现

- 10→11和11→12均构建32×32矩阵；每个Token贡献16个笛卡尔共现对。
- 样本等权领域两两JSD为{focus_transition_jsd.jsd_bits.min():.6f}–{focus_transition_jsd.jsd_bits.max():.6f} bits。
- Top-20共现对质量为{focus_transition.top20_mass.min():.2%}–{focus_transition.top20_mass.max():.2%}。

### 03D 层内Top-4组合

- 24层有序和无序组合均已统计，不构造完整24层路径类别。
- 第11/12层唯一组合数为{int(focus_combination.unique_combinations.min()):,}–{int(focus_combination.unique_combinations.max()):,}；Top-50质量为{focus_combination.top50_mass.min():.2%}–{focus_combination.top50_mass.max():.2%}。
- 第11/12层组合分布的领域两两JSD为{focus_combination_jsd.jsd_bits.min():.6f}–{focus_combination_jsd.jsd_bits.max():.6f} bits。

### 03E 控制与敏感性

- 22个设置完成审计，其中21个可分析；special为0 Token，仅登记为空。
- 14个主稳健性设置全部覆盖五领域，包括930个冻结长度匹配样本和固定截断N=50。
- 基线平均两两JSD：10→11={baseline_control.loc[("transition", 10), "mean_pairwise_jsd_bits"]:.6f}，11→12={baseline_control.loc[("transition", 11), "mean_pairwise_jsd_bits"]:.6f}，第11层无序组合={baseline_control.loc[("unordered_combination", 11), "mean_pairwise_jsd_bits"]:.6f}，第12层无序组合={baseline_control.loc[("unordered_combination", 12), "mean_pairwise_jsd_bits"]:.6f} bits。
- 14个主控制的平均两两JSD总范围为{robust_controls.mean_pairwise_jsd_bits.min():.6f}–{robust_controls.mean_pairwise_jsd_bits.max():.6f} bits。

## 📊 交付物审核表

| 检查项 | 结果数据 | 通俗含义（以此判断是否通过） |
|---|---:|---|
| 输入样本 / Token | 2,000 / 382,685 | 与步骤00缓存一致 |
| 正式推断Token | 380,685 | 每个样本首Token均已排除 |
| 03B逐样本结果 | 92,000行 | 两总体×2,000样本×23对相邻层 |
| 03C矩阵明细 | 40,960行 | 两层对、双总体、双权重、五领域完整 |
| 03D组合摘要 | 960行 | 24层有序/无序组合完整 |
| 03E相邻层控制 | 4,830行 | 21个可分析设置完整 |
| 组件产物哈希 | 23/23匹配 | 文件未损坏或漂移 |
| 独立验证器 | 5/5通过 | 输入及03B–03E均有独立复算 |
| 缺失值 / 非有限数 | 0 / 0 | 可分析结果完整 |
| 模型推理重跑 | 否 | 全程只读既有缓存 |
| 稠密SAE矩阵 | 未创建 | 保持稀疏安全边界 |

## 结论与边界

步骤03支持“领域相关的路由差异”进一步体现在相邻层编号连续性、重点层跨层编号共现和同层Top-4组合结构中；这些描述性差异在14个主稳健性设置下仍可观察到。

不能据此声称专家具有功能专门化：不同层同编号专家是不同参数模块，共现不是因果流动，并且领域与数据集来源/文本格式仍然混杂。单独Token类别，尤其稀疏类别，只能作为诊断材料。

## 🛑 强制暂停

以上为第03步结果，请审核。确认无误请回复“继续”。
"""
    atomic_text(report_text, REPORT_PATH)

    final_static_paths = [SUMMARY_PATH, REPORT_PATH, VALIDATION_LOG]
    for path in final_static_paths:
        inventory_rows.append({
            "stage": "03F", "kind": "final_output", "path": str(path),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "verified": True,
        })
    script_names = ["_03_common.py", "03_validate_inputs.py", "03b_adjacent_layer_continuity.py", "03b_validate_outputs.py", "03c_focused_transitions.py", "03c_validate_outputs.py", "03d_top4_combinations.py", "03d_validate_outputs.py", "03e_path_controls.py", "03e_validate_outputs.py", "03f_validate_finalize.py"]
    for name in script_names:
        path = SCRIPT.with_name(name)
        require(path.is_file(), f"Missing step-03 script: {path}")
        inventory_rows.append({
            "stage": "03", "kind": "script", "path": str(path),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "verified": True,
        })
    inventory = pd.DataFrame(inventory_rows).drop_duplicates(subset=["path"], keep="last").sort_values(["stage", "kind", "path"], kind="stable")
    require(inventory.verified.all(), "Unverified inventory row")
    atomic_csv(inventory, INVENTORY_PATH)

    final_outputs = [SUMMARY_PATH, REPORT_PATH, INVENTORY_PATH, VALIDATION_LOG]
    final_manifest = {
        "status": "complete", "step": "03_routing_path_and_expert_combinations",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": script_record(SCRIPT), "shared_script": script_record(SCRIPT.with_name("_03_common.py")),
        "component_manifests": component_records,
        "validation_suite": validation_records,
        "outputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in final_outputs},
        "row_counts": {str(INVENTORY_PATH): len(inventory)},
        "validation": {
            "component_outputs_verified": 23, "validators_passed": len(validation_records),
            "missing_values_in_final_tables": 0, "nonfinite_values_in_final_tables": 0,
            "model_inference_rerun": False, "dense_sae_matrix_created": False,
        },
        "raw_source_reparsed": False, "model_loaded": False, "model_inference_rerun": False, "dense_sae_matrix_created": False,
    }
    atomic_json(final_manifest, FINAL_MANIFEST)
    print(json.dumps({
        "status": "complete", "validators_passed": len(validation_records),
        "component_outputs_verified": 23, "inventory_rows": len(inventory),
        "summary": str(SUMMARY_PATH), "report": str(REPORT_PATH), "manifest": str(FINAL_MANIFEST),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
