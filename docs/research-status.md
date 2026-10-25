# 研究状态

## 主线：MoE expert routing path

当前主线目标是刻画 GPT-OSS-20B 在不同输入条件下的 expert routing 规律，包括专家选择分布、相邻层连续性、跨层共现、层内 Top-4 组合和稳健性控制。

现阶段已完成 2,000 个样本的聚合分析。结果支持“领域相关 routing 差异”这一描述性结论，但不支持把这种差异直接解释为 expert 功能专门化或因果路径。

## 辅助分支：Routing–SAE

SAE 用于辅助理解 routing 与内部稀疏特征之间的统计关系。该分支已经产出预测和候选关联结果，但候选语义仍需独立证据，不能反向改变 routing 主线目标。

## 辅助分支：SAE feature semantics

官方 Neuronpedia 解释用于给 feature 添加外部语义信息。查询键是 `(sae_layer, feature_id)`；`np_missing` 仅表示没有官方解释，不代表 feature 没有语义或已经死亡。
