# 完整 Routing Explorer

本目录保存研究过程中使用的完整交互工具代码，而不是脱离实验数据的简化静态图。

## 公开与私有边界

Git 仓库包含：

- `frontend/`：浏览器端页面；
- `backend/`：只读 FastAPI 服务；
- `contracts/`：接口字段、研究规模和校验约束。

Git 仓库不包含：原始语料、逐 token routing 数组、SAE activation 分片、模型权重、完整查询索引或机器绝对路径。没有这些私有输入时，代码可以被审核，但不能显示完整实验内容。

## 所需私有输入

后端至少需要以下已经由项目流水线生成并校验的输入：

- explorer cache：token 索引、Top-1 trajectories、Top-4 聚合边、expert contexts、SAE 查询索引等；
- frozen layout：expert 节点坐标；
- Top-1 transition density；
- Top-4 routing NPZ；
- 与上述数据一致的 API contract。

各 feature 的官方语义必须按 `(sae_layer, feature_id)` 连接，不能只按 `feature_id`。

## 启动

先创建私有环境变量或通过命令行指定路径。不要把真实绝对路径写回 Git：

```bash
python -m venv .venv
python -m pip install -r requirements.txt

export EXPLORER_CACHE_DIR=/path/to/private/cache/v01
export EXPLORER_LAYOUT_DIR=/path/to/private/layout/v02
export EXPLORER_DENSITY_DIR=/path/to/private/layout/v02_top1_density
export EXPLORER_API_CONTRACT=/path/to/private/contracts/v03_api_contract.json
export EXPLORER_ROUTING_TOP4=/path/to/private/routing_top4.npz

cd explorer/backend
python -m uvicorn v04_app:app --host 127.0.0.1 --port 8765
```

然后在运行服务的同一台机器上访问 `http://127.0.0.1:8765/`。如果服务运行在 SSH 服务器上，应使用 SSH 端口转发后在本机浏览器访问，不要把只读 API 直接暴露到公网。

## 成功标志

- `/api/health` 返回成功；
- 页面能够加载样本和 token；
- 选中 token 后可查看完整 24 层 routing；
- Top-4 expert 与权重能够显示；
- expert context 与 SAE 查询只读取已生成的索引；
- 页面和接口不会修改实验数据。
