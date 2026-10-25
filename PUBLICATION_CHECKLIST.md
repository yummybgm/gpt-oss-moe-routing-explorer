# 发布前人工确认清单

以下事项必须由仓库所有者在首次推送前确认：

- [ ] GitHub 仓库名称为 `gpt-oss-moe-routing-explorer`，可见性为 Public。
- [x] README 对总体目标的表述以 MoE expert routing path 为主线。
- [x] 使用 SSH 私有缓存检查完整 `explorer/` 的健康状态、样本检索、24 层 routing、Top-4、expert contexts 和 SAE 只读查询。
- [ ] 首次公开前通过 SSH 端口转发在浏览器中目视检查页面布局和主要交互。
- [x] 发布候选目录已完成敏感信息、绝对路径、大文件和数据范围扫描，未发现完整数据、原始文本、API 请求/响应或密钥。
- [ ] 在 GitHub Desktop 的 commit 预览中再次确认上述私有内容均不在待提交文件列表内。
- [x] 公开署名已确认为 `He Zhiyun`，GitHub 用户名为 `yummybgm`，并已补充 `CITATION.cff`。
- [x] 原创代码采用 MIT License；模型、语料、外部数据库和私有实验数据不在该授权范围内。
- [ ] 把 GitHub 仓库链接发给导师/学姐前，再核对一遍 README、截图和科研边界。

本文件是人工发布门，不构成上传授权；任何远程仓库创建或 push 都需要单独确认。
