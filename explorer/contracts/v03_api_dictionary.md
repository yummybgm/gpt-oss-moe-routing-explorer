# V03 API 字段字典

## 统一主键

| 对象 | 字段 | 含义 |
|---|---|---|
| Token | `token_row` | 0–382,684 的连续全局 Token 行号；V03 不再使用 `global_token_index` 别名 |
| 样本 | `sample_id` | MMLU-Pro 样本标识 |
| 图节点 | `node_id` | `layer × 32 + expert_id`，范围 0–767 |
| SAE 特征 | `sae_layer`, `feature_id` | 特征只能在所属 SAE 层内解释，不跨层合并 |

## 图接口

- `group_type` 与 `group_value` 必须是契约中的合法组合。
- `edge_mode=top1` 返回实际 Top-1 相邻层跳转计数 `token_count`。
- `edge_mode=top4` 返回 Top-4 外积聚合的 `weight_mass` 和 `token_support`。
- 所有节点使用 V02 已冻结的 `x`, `y`，筛选只改变节点统计和边，不重新计算布局。
- `max_edges` 只限制返回和绘制数量；响应必须同时给出限制前总数和 `truncated`，不得把截断后的边误称为完整总体。

## Token 与样本接口

- 搜索按 `token_text` 做不区分大小写的字面子串匹配，不使用正则表达式。
- `primary_only=true` 使用冻结的 `mask_primary_inference`，即排除每个样本的首 Token。
- Token 详情返回完整 24 层 Top-1 节点和权重，不允许只返回可见层或抽样层。

## 专家上下文接口

- 未指定 `domain` 时读取 `overall / primary_inference` 上下文。
- 指定领域时读取对应 `domain / <domain>` 上下文。
- 上下文按预先冻结的 `display_rank` 返回，其用途是展示证据，不替代总体频率统计。

## SAE 接口

- 仅支持层 3、7、11、15、19、23。
- 通过 `sample_id × sae_layer` 索引定位原始稀疏分片，再按 `token_position` 过滤；禁止构造稠密 Token × Feature 矩阵。
- 第 11 层历史分片没有物理 `sae_layer` 列，由已验证索引补回常量 11。
- 默认只返回 `passed_threshold=true` 的激活，并按 `activation` 降序排列。
- 若 Token 是样本首 Token，响应必须附加异常警告。
- 当前接口只回答“这个 Token 激活了哪些特征”；全局特征语义仍需单独建立并验证 feature-context 缓存。
