# SAE feature semantics（辅助分支）

本目录保留 Neuronpedia 官方语义数据库的下载、规范化和校验逻辑，以及不含完整解释文本的覆盖率摘要。

完整数据库不进入 Git。下游程序应以 `(sae_layer, feature_id)` 连接；不能只用 `feature_id`，因为不同层的编号会重复。

官方数据库中的 `np_missing` 表示没有可用的官方解释。它不是“无语义”“dead feature”或自动生成标签的同义词。

建议用法：在 routing–SAE 候选表生成后，再将候选键与只读 SQLite 连接，为结果表补充 `description` 和 `coverage_status`；不要让外部解释参与盲法阶段的本地标签生成。
