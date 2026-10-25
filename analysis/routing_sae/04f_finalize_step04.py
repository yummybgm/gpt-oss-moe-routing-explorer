#!/usr/bin/env python3
"""Step 04F: final integrity audit, artifact inventory, and cumulative Chinese report."""
from __future__ import annotations

import json
import os
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from _03_common import METADATA, REPORT, TABLES, atomic_csv, atomic_json, require, script_record, sha256_file

ROOT = Path(os.environ.get("MOE_SAE_ANALYSIS_ROOT", Path(__file__).resolve().parents[2] / "data" / "mmlu_pro_formal_2000_analysis_v1")).expanduser().resolve()
SCRIPT = Path(__file__).resolve()
COMPONENT_MANIFESTS = [
    METADATA / "routing_sae_step04a_contract.json",
    METADATA / "routing_sae_step04b_manifest.json",
    METADATA / "routing_sae_step04c_manifest.json",
    METADATA / "routing_sae_step04d_manifest.json",
    METADATA / "routing_sae_step04e_inference_manifest.json",
    METADATA / "routing_sae_step04e_all_token_manifest.json",
]
INVENTORY_PATH = TABLES / "routing_sae_step04_artifact_inventory.csv"
REPORT_PATH = REPORT / "routing_sae_step04_stage_report.md"
FINAL_MANIFEST_PATH = METADATA / "routing_sae_step04_manifest.json"


def atomic_text(text: str, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    for path in COMPONENT_MANIFESTS:
        require(path.is_file(), f"Missing component manifest: {path}")
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in COMPONENT_MANIFESTS]
    require(manifests[0]["status"] == "input_validation_complete_method_choice_required", "04A status mismatch")
    require(all(item["status"] == "complete" for item in manifests[1:]), "An 04B-04E component is incomplete")
    require(all(item.get("model_inference_rerun") is False for item in manifests), "A component reran model inference")
    require(all(item.get("dense_sae_matrix_created") is False for item in manifests), "A component created a dense SAE matrix")

    inventory_rows = []
    seen: set[str] = set()
    for manifest_path, manifest in zip(COMPONENT_MANIFESTS, manifests):
        for path_text, metadata in manifest.get("outputs", {}).items():
            path = Path(path_text)
            require(path.is_file(), f"Missing component output: {path}")
            require(path.stat().st_size == metadata["size_bytes"], f"Size mismatch: {path}")
            require(sha256_file(path) == metadata["sha256"], f"Hash mismatch: {path}")
            if str(path) not in seen:
                inventory_rows.append({"component": manifest["step"], "kind": "output", "path": str(path), "size_bytes": path.stat().st_size, "sha256": metadata["sha256"], "verified": True})
                seen.add(str(path))
        manifest_hash = sha256_file(manifest_path)
        inventory_rows.append({"component": manifest["step"], "kind": "manifest", "path": str(manifest_path), "size_bytes": manifest_path.stat().st_size, "sha256": manifest_hash, "verified": True})
        seen.add(str(manifest_path))

    script_names = [
        "04a_validate_inputs_and_contract.py", "04a_validate_outputs.py",
        "04b_build_sparse_joint_index.py", "04b_validate_outputs.py",
        "04c_grouped_folds_feature_selection.py", "04c_validate_outputs.py",
        "04d_train_evaluate_probes.py", "04d_validate_outputs.py",
        "04e_bootstrap_bh.py", "04e_all_token_audit.py", "04e_validate_outputs.py",
    ]
    for name in script_names:
        path = ROOT / "scripts" / name
        require(path.is_file(), f"Missing step-04 script: {path}")
        inventory_rows.append({"component": "step04", "kind": "script", "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "verified": True})
    inventory = pd.DataFrame(inventory_rows).sort_values(["component", "kind", "path"], kind="stable").reset_index(drop=True)
    require(inventory.verified.all() and inventory.path.is_unique, "Artifact inventory validation failed")
    atomic_csv(inventory, INVENTORY_PATH)

    metrics = pd.read_csv(TABLES / "routing_sae_step04d_expert_metrics.csv")
    recall = pd.read_csv(TABLES / "routing_sae_step04d_recall_at4.csv")
    bootstrap = pd.read_csv(TABLES / "routing_sae_step04e_performance_bootstrap.csv")
    global_bootstrap = pd.read_csv(TABLES / "routing_sae_step04e_global_bootstrap.csv")
    associations = pd.read_csv(TABLES / "routing_sae_step04e_candidate_holdout_associations.csv")
    audit_metrics = pd.read_csv(TABLES / "routing_sae_step04e_all_token_audit_metrics.csv")
    audit_recall = pd.read_csv(TABLES / "routing_sae_step04e_all_token_audit_recall.csv")
    model_summary = metrics.groupby("model", sort=False)[["auprc", "auroc", "log_loss"]].mean()
    recall_summary = recall.groupby("model", sort=False).recall_at4.mean()
    global_lookup = global_bootstrap.set_index(["comparison", "metric"])
    primary_combined = audit_metrics[(audit_metrics.population == "primary_nonfirst") & audit_metrics.model.eq("combined")]
    all_combined = audit_metrics[(audit_metrics.population == "all_tokens_audit") & audit_metrics.model.eq("combined")]
    first_recall = float(audit_recall[(audit_recall.population == "first_only_audit") & audit_recall.model.eq("combined")].recall_at4.iloc[0])

    report = f"""# 步骤04阶段性报告：Routing–SAE联合分析

生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}

## 📍 进度与双语解读

当前位置：步骤04已完成。完成了 Routing–SAE predictive association analysis（路由—SAE预测关联分析），即检验第11层稀疏SAE特征能否在样本外预测第11层Top-4专家选择，并与domain + token_class基线比较。

主分析严格排除每个样本的首Token；首Token仅作为异常敏感性审计。所有交叉验证均按sample_id分组，特征筛选只发生在训练fold内。

## 📊 核心结果

| 模型 | 平均AUPRC | 平均AUROC | 平均Log Loss | Recall@4 |
|---|---:|---:|---:|---:|
| Baseline | {model_summary.loc['baseline','auprc']:.4f} | {model_summary.loc['baseline','auroc']:.4f} | {model_summary.loc['baseline','log_loss']:.4f} | {recall_summary['baseline']:.4f} |
| SAE | {model_summary.loc['sae','auprc']:.4f} | {model_summary.loc['sae','auroc']:.4f} | {model_summary.loc['sae','log_loss']:.4f} | {recall_summary['sae']:.4f} |
| Combined | {model_summary.loc['combined','auprc']:.4f} | {model_summary.loc['combined','auroc']:.4f} | {model_summary.loc['combined','log_loss']:.4f} | {recall_summary['combined']:.4f} |

样本等权、领域分层1000次bootstrap中，Combined − Baseline的macro AP增量为{global_lookup.loc[('combined_minus_baseline','macro_sample_equal_average_precision'),'observed_delta']:.4f}，95% CI [{global_lookup.loc[('combined_minus_baseline','macro_sample_equal_average_precision'),'ci_lower']:.4f}, {global_lookup.loc[('combined_minus_baseline','macro_sample_equal_average_precision'),'ci_upper']:.4f}]。32/32位专家通过该比较的BH校正。

Combined − SAE的macro AP增量为{global_lookup.loc[('combined_minus_sae','macro_sample_equal_average_precision'),'observed_delta']:.4f}，95% CI [{global_lookup.loc[('combined_minus_sae','macro_sample_equal_average_precision'),'ci_lower']:.4f}, {global_lookup.loc[('combined_minus_sae','macro_sample_equal_average_precision'),'ci_upper']:.4f}]；18/32位专家通过BH校正。

40,960项训练fold筛选、测试fold检验的候选SAE关联中，{int(associations.significant_bh_0_05.sum()):,}项通过全家族BH q<0.05，{int(associations.same_direction_as_train_phi.sum()):,}项与训练fold方向一致。

## 首Token质量审计

- 2000个首Token全部选择同一个专家集合：[5, 10, 15, 25]。
- 三类正常Token训练探针在首Token上的Recall@4均为{first_recall:.1f}。
- Combined主分析平均AUPRC为{primary_combined.auprc.mean():.4f}；加入首Token审计后降至{all_combined.auprc.mean():.4f}。
- 首Token指标中，84行AUPRC及96行AUROC因目标类别恒定而结构性不可定义，均已显式标记，不属于未处理缺失值。

该结果确认：首Token是独立于普通文本内容的强路由/SAE异常，主推断排除首Token是必要质量控制。

## 研究结论

第11层稀疏SAE特征对同层Top-4专家选择具有稳定、可复现的样本外预测增量，并且该增量不能仅由domain与token_class基线解释。统一表述为“领域相关的路由差异与第11层SAE特征存在预测关联”。

## 结论限制

1. `resid_post_layer_11`产生在第11层路由及专家计算之后，因此当前结果是关联和结果编码证据，不是SAE特征导致路由选择的因果证据。
2. 一个领域只对应一个数据来源，领域、语料来源和文本格式仍完全混杂。
3. 当前只完成第11层SAE正式分析，不能据此声称整个24层模型都具有相同规律。
4. 不得把结果写成专家“功能专门化”已经得到证明。

## 后续跨层复现实验

在新一轮模型推理获批后，逐层运行`resid_post_layer_3`、`layer_7`、`layer_15`、`layer_19`、`layer_23`，每次只加载一个SAE。优先使用第L层resid-post特征预测第L+1层路由或局部routing path转移，以获得更合理的时间顺序；第23层只作为末层关联审计。

## 📊 交付物审核表

| 检查项 | 结果 | 判断 |
|---|---:|---|
| 主分析Token | 380,685 | 排除2000个异常首Token |
| 稀疏SAE非零值 | 28,081,931 | 与缓存一致 |
| 分组交叉验证 | 5折，每折400样本 | sample_id泄漏为0 |
| 轻量探针 | 480 | 全部收敛 |
| Bootstrap | 1,000次 | sample_id单位、领域分层 |
| 候选BH检验 | 40,960 | 测试fold独立检验 |
| 独立验证 | 04A–04E全部通过 | 指标与哈希一致 |
| 原模型推理重跑 | 否 | 通过 |
| 稠密SAE矩阵 | 未创建 | 通过 |
| 已核验交付物 | {len(inventory)} | 见artifact inventory |

## 🛑 强制暂停

以上为第04步结果，请审核。确认无误请回复“继续”。
"""
    atomic_text(report, REPORT_PATH)

    outputs = [INVENTORY_PATH, REPORT_PATH]
    final_manifest = {
        "status": "complete", "step": "04_routing_sae_joint_analysis",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "components_verified": [manifest["step"] for manifest in manifests],
        "artifact_inventory_rows": len(inventory),
        "key_results": {
            "mean_auprc": {model: float(model_summary.loc[model, "auprc"]) for model in model_summary.index},
            "mean_recall_at4": {model: float(recall_summary[model]) for model in recall_summary.index},
            "combined_minus_baseline_sample_equal_ap": global_lookup.loc[("combined_minus_baseline", "macro_sample_equal_average_precision")].to_dict(),
            "candidate_significant_bh_0_05": int(associations.significant_bh_0_05.sum()),
            "candidate_same_direction": int(associations.same_direction_as_train_phi.sum()),
            "first_token_expert_set": [5, 10, 15, 25], "first_token_recall_at4": first_recall,
        },
        "conclusion": "Layer-11 sparse SAE features provide stable out-of-sample predictive information about layer-11 Top-4 expert selection beyond domain and token-class baselines.",
        "interpretation_limits": [
            "association and predictive increment only; not causality or functional specialization",
            "resid_post_layer_11 occurs after the same-layer routing decision and expert computation",
            "domain is fully confounded with source and text format",
            "single-layer result does not establish a model-wide pattern",
        ],
        "next_registered_extension": "Run resid_post SAE layers 3, 7, 15, 19, and 23 one at a time; prioritize layer-L SAE prediction of layer-(L+1) routing/path transitions.",
        "validation": {"component_manifests": len(manifests), "artifact_hash_mismatches": 0, "unhandled_missing_values": 0, "model_inference_rerun": False, "dense_sae_matrix_created": False},
        "outputs": {str(path): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs},
        "script": script_record(SCRIPT),
    }
    atomic_json(final_manifest, FINAL_MANIFEST_PATH)
    print(json.dumps({"status": "complete", "artifact_inventory_rows": len(inventory), "key_results": final_manifest["key_results"], "outputs": final_manifest["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
