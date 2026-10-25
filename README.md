# GPT-OSS MoE Routing Explorer

这是一个面向科研过程的公开仓库，用于研究 **GPT-OSS-20B 的 MoE expert routing path 规律**，并提供可审核的聚合结果、分析脚本和完整交互式研究工具的源代码。

> 研究主线是 expert routing：专家选择、层间连续性、跨层共现、层内 Top-4 组合，以及这些模式在领域、样本和控制条件下的稳定性。SAE feature 语义解释是辅助解释分支，不是总体研究目标，也不是 routing 分析的前置条件。

## 完整交互工具

`explorer/` 保存完整 routing explorer 的前端、只读 FastAPI 后端和接口契约。使用者在合法取得并自行配置实验缓存后，可以浏览样本与 token、24 层 expert routing、Top-4 权重、expert contexts 和辅助 SAE feature 信息。

仓库**不发布**原始文本、逐 token routing 缓存、稀疏 SAE activation、模型权重或机器专用索引。因此 GitHub 文件页用于查看代码、聚合结果和研究说明；完整交互页面需要使用者在本地或 SSH 服务器上连接自己的私有数据运行。详见 [`explorer/README.md`](explorer/README.md)。

## 当前可公开研究状态

- 数据范围：2,000 个样本、382,685 个 token；正式推断使用 380,685 个非首 token。
- 路由结构：24 层、每层 32 个 experts、每个 token 记录 Top-4 routing。
- 已完成：相邻层连续性、重点层跨层共现、层内 Top-4 组合、控制与敏感性分析。
- Routing–SAE 辅助分支：已有统计关联与候选 feature 汇总；这些结果只表示关联，不构成功能或因果证明。
- SAE 语义分支：官方解释按 `(sae_layer, feature_id)` 连接；未解释项仍保持缺失状态，不在本仓库中用未经验证的自动标签替代。

## 仓库结构

```text
analysis/              路由主线与 routing–SAE 辅助分析脚本
explorer/              完整本地/SSH 浏览器的前后端代码
docs/                  数据政策、复现边界和当前研究状态
results/routing/       可公开的 routing 聚合表、图和阶段报告
results/routing_sae/   可公开的 routing–SAE 汇总结果
sae_semantics/         官方语义数据库构建/校验脚本和覆盖率摘要
configs/               路径配置示例
```

## 两种使用方式

### 1. 直接查看公开结果

无需安装依赖，直接在 GitHub 中浏览 `results/`、`docs/` 和 README。仓库中的截图、聚合表和报告用于说明当前研究状态；它们不代替底层实验数据。

### 2. 使用私有数据运行完整浏览器

复制 `.env.example` 为本机私有配置，设置数据路径后安装依赖：

```bash
python -m venv .venv
python -m pip install -r requirements.txt
cd explorer/backend
python -m uvicorn v04_app:app --host 127.0.0.1 --port 8765
```

浏览器启动后访问 `http://127.0.0.1:8765/`。完整浏览器需要未提交 Git 的缓存与索引；具体字段、环境变量和启动条件见 [`explorer/README.md`](explorer/README.md) 与 [`docs/reproducibility.md`](docs/reproducibility.md)。

## 复现边界

公开仓库提供分析逻辑、聚合结果和数据契约，但不再分发受限或体积过大的底层数据。因此它支持：

- 审核研究问题、指标定义、统计控制和结论边界；
- 对已有聚合结果重新绘图和检查；
- 在合法取得相同输入后运行完整分析。

它不声称仅凭公开仓库即可重新生成模型推理缓存。详见 [`docs/data-policy.md`](docs/data-policy.md)。

## 科学解释边界

1. 不同层的同编号 expert 是不同参数模块。
2. 跨层 expert 共现是统计关系，不等同于因果“信息流”。
3. 当前领域与数据来源、文本格式可能混杂。
4. Routing 差异本身不能证明 expert 功能专门化。
5. SAE feature 与 routing 的预测或相关关系不能直接解释为因果机制。

## 引用与许可

公开作者为 **He Zhiyun**（GitHub：[`@yummybgm`](https://github.com/yummybgm)）。如在研究或软件中使用本项目，请按 [`CITATION.cff`](CITATION.cff) 提供的信息引用。

项目原创代码采用 [MIT License](LICENSE)。第三方软件仍分别遵守其原许可证；模型、语料、外部数据库和未提交的实验数据不因本仓库的 MIT License 获得重新授权。
