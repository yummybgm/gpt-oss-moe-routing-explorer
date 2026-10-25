# 数据公开边界

## 可以公开

- 聚合后的统计表和图；
- 不含原始题目文本、逐 token 内容或个人信息的研究摘要；
- 分析、校验、可视化和数据库构建脚本；
- 数据结构、字段契约、哈希与复现说明；
- 少量专门生成的演示数据。

## 不进入 Git

- 原始 MMLU-Pro 文本和逐样本 prompt；
- 完整逐 token routing、Top-4 NPZ、稀疏 SAE activation parquet；
- Neuronpedia 完整 SQLite、批量原始下载文件和 API 原始响应；
- DeepSeek/Claude 对话记录、API ledger、请求与响应；
- API Key、SSH 配置、用户名和机器绝对路径；
- 尚未通过验证的 849 个自动生成语义标签。

本策略同时考虑版权、隐私、仓库体积、科研可审计性和避免把中间产物误当作最终结论。
