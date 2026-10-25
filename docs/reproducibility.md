# 复现说明

## 公开聚合结果

`results/routing/` 可直接用于复核数值、重新绘图和检查结论。公开仓库不再提供单独的简化静态 routing 页面，以免把局部层间结果误解为完整 routing explorer。

## 完整分析

完整分析需要使用者自行准备私有输入目录，并设置：

- `MOE_SAE_ANALYSIS_ROOT`：包含 `cache/`、`tables/`、`metadata/` 等目录；
- `EXPLORER_ROUTING_TOP4`：Top-4 routing NPZ；
- `NEURONPEDIA_SEMANTICS_ROOT`：官方语义数据库目录。

所有 feature 语义连接必须使用 `(sae_layer, feature_id)`，不可只用 `feature_id`。

本仓库中的统计脚本沿用冻结样本数、token 数、层数和 expert 数校验；若使用不同模型或数据，必须先修改契约并重新验证，不能只绕过检查。

完整交互工具的私有输入契约和启动方式见 [`../explorer/README.md`](../explorer/README.md)。
