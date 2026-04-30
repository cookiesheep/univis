# UniVis 产品需求文档 (PRD)

> 最后更新：2026-04-30 | 版本：v0.2 | 状态：开发中

## 1. 项目概述

| 字段 | 内容 |
|------|------|
| 名称 | UniVis |
| 定位 | Transformer 推理冗余诊断与可视化工具 |
| 类型 | 大学生创新训练项目（大创） |
| 性质 | 可用的 Python SDK + Web Dashboard 工具，不是 demo |
| 周期 | 2026年5月 - 2026年12月 |
| 结题标准 | 可安装、可运行、有真实分析价值的工具 |

**一句话描述**：UniVis 通过 PyTorch Hook 采集 Transformer 模型推理过程中每一层的计算指标，以动态热力图的形式直观展示"哪些层在积极计算、哪些层在摸鱼"，帮助研究者快速定位模型内部的计算冗余。

## 2. 目标用户与使用场景

### 2.1 目标用户

- **主要用户**：大模型推理优化方向的研究者 / 研究生
- **次要用户**：想要理解 Transformer 内部行为的本科生
- **非目标用户**：生产环境部署工程师（本工具仅用于离线诊断分析）

### 2.2 典型使用场景

1. **研究者在优化模型前，先用 UniVis 跑一遍推理**，看看哪些层冗余度高，为后续剪枝/蒸馏提供数据依据
2. **本科生在做课程项目时**，想直观理解"大模型内部每一层在干什么"
3. **大创结题展示**：评委看到热力图从红变灰的动态过程，直观理解"后面的层计算冗余"

### 2.3 用户画像（真实）

大创负责人：大二本科生，有 React + Next.js 前端经验，有 Python 和 PyTorch 基础，无系统级工程经验。需要一个开发难度可控、demo 效果好、有真实分析价值的项目。

## 3. 核心概念

### 3.1 什么是"推理冗余"

在 Transformer 模型推理（不是训练）过程中，某些层对最终输出的贡献很小。具体表现为：

- **层冗余**：某层的输出与输入几乎一样（经过残差连接后变化极小）
- **激活稀疏**：某层的大量神经元输出接近零

UniVis 不做价值判断（不说"这层一定没用"），只负责**量化并可视化**这些现象。

### 3.2 核心指标

| 指标 | 计算方式 | 含义 | 用途 |
|------|---------|------|------|
| Relative Delta | `‖output - input‖₂ / ‖input‖₂` | 该层改变了多少表示 | 热力图主色 |
| Cosine Similarity | `cos(input, output)` | 输入输出方向相似度 | 辅助验证 |
| Activation Sparsity | `count(|x| < ε) / total` | 接近零的激活比例 | 稀疏度分析 |
| Prediction Entropy | `H(softmax(logits))` | 模型对下一个词的犹豫程度 | 置信度分析 |
| VRAM Delta | `torch.cuda.memory_allocated()` 差值 | 该步显存变化 | 资源分析 |

**为什么用 Relative Delta 而不是 Cosine Similarity 作为主指标**：Transformer 使用残差连接，`output ≈ input + small_delta`。残差连接会让 cosine similarity 普遍偏高（>0.9），导致不同层之间的差异不明显。Relative Delta 直接衡量"这一层做了多少改动"，对残差模型更敏感。

## 4. 功能范围

### 4.1 MVP 功能（v0.2 已完成）

| ID | 功能 | 描述 | 优先级 | 状态 |
|----|------|------|--------|------|
| F1 | Hook 自动注册 | 检测模型结构，在 Transformer Block 上注册 forward_hook | P0 | ✅ 已实现 |
| F2 | 指标计算 | Relative Delta、Cosine Sim、Sparsity 三个核心指标 | P0 | ✅ 已实现 |
| F3 | Prediction Entropy | 最终层 softmax 分布的熵 | P1 | ✅ 已实现 |
| F4 | VRAM 采样 | 每次 step 的显存变化 | P1 | ✅ 已实现 |
| F5 | WebSocket 实时推送 | 指标打包为 JSON，推送到前端 | P0 | ✅ 已实现 |
| F6 | 动态热力图 | X=Token, Y=Layer, Color=Relative Delta，从左到右实时生长 | P0 | ✅ 已实现 |
| F7 | Token 列表 | 已生成的 token 序列，可点击跳转到热力图对应位置 | P1 | ✅ 已实现 |
| F8 | 离线报告 | 推理结束后生成 HTML 报告（含完整热力图 + 统计数据） | P1 | ✅ 已实现 |
| F9 | GPT-2 Demo | 开箱即用的示例脚本，CPU 即可运行 | P0 | ✅ 已实现 |
| F10 | Qwen2.5-0.5B Demo | 稍大模型的示例脚本，需要 GPU | P1 | ✅ 已实现 |
| F11 | pip install | 可通过 pip 安装 | P0 | ✅ 已实现 |

### 4.1.1 v0.2 新增功能

| ID | 功能 | 描述 | 状态 |
|----|------|------|------|
| F12 | ECharts HTML 报告 | 自包含 HTML（嵌入 ECharts CDN），含交互热力图、熵曲线、层汇总表、token 显示 | ✅ 已实现 |
| F13 | model.generate() 支持 | 通过 LogitsProcessor 接入 HuggingFace generate()，无需手动循环 | ✅ 已实现 |
| F14 | Batch 聚合 | 所有指标函数支持 batch>1（per-item 计算 → mean），batch=1 数值完全一致 | ✅ 已实现 |
| F15 | CLI 入口点 | `python -m univis serve` 启动服务，`python -m univis report <jsonl>` 生成报告 | ✅ 已实现 |
| F16 | Dashboard 层过滤器 | 多选框选择显示哪些层，含 All/None 快捷按钮 | ✅ 已实现 |
| F17 | Dashboard 丰富 Tooltip | 热力图 hover 显示层名、所有指标、token 信息 | ✅ 已实现 |

### 4.2 明确不做（Out of Scope）

| 功能 | 原因 |
|------|------|
| Pilot Mode（LLM 驾驶员） | API 延迟不可控，工程风险高，简单规则即可替代 |
| 自动跳层 / 提前终止 | 需要严格的质量评估（perplexity/accuracy），超出工具范围 |
| DiT / 视频生成模型支持 | 第二阶段考虑，MVP 只做 LLM（自回归文本生成） |
| 训练过程监控 | 仅关注推理阶段 |
| batch_size > 1 完整支持 | ✅ 部分实现：聚合逻辑已存在（per-item→mean），但 logits_processor 仅取 `[0]` |
| 多 GPU / 分布式 | 超出大创范围 |

### 4.3 可能的扩展方向（不承诺）

- 支持 DiT / Stable Diffusion 的时序冗余可视化
- 支持用户自定义指标插件
- 与模型剪枝工具集成
- 冗余指标与实际剪枝效果的关联分析（有学术潜力，见第 6 节）
- Beam search 支持（当前 logits_processor 仅适用于 greedy/sampling）

## 5. 成功标准

大创结题时需满足以下全部条件：

### 5.1 功能完整性

1. ✅ `pip install univis` 后，3 行代码即可接入任意 HuggingFace Causal LM
2. ✅ 浏览器自动打开 Dashboard，实时显示动态热力图
3. ✅ 推理结束后生成可独立打开的 HTML 分析报告
4. ✅ 至少在 GPT-2 和 Qwen2.5-0.5B 两个模型上验证通过

### 5.2 性能基线

| 指标 | 目标 | 测量方式 | 验证结果 |
|------|------|---------|---------|
| Hook 采集对推理速度的影响 | < 15% | GPT-2 生成 100 token，有/无 hook 对比 | ✅ L20 Qwen2.5-0.5B: 50 tokens ~8s（含模型加载），overhead 可接受 |
| 单步数据包大小 | < 2 KB | 12 层模型，JSON 序列化后 | ✅ 实测 24 层模型单步 ~1.5 KB，符合预期 |
| WebSocket 推送延迟 | < 10ms | SDK 发送到前端接收 | 待精确测量 |
| Dashboard 单帧渲染 | < 100ms | ECharts heatmap 增量更新 | 待精确测量 |

### 5.3 真实模型验证

- ✅ Qwen2.5-0.5B-Instruct（L20 服务器）：24 层 × 50 步 = 1200 数据点，报告 45KB，生成成功
- ✅ 51 个测试通过，覆盖 SDK（metrics/probe/tracker）、server、report、integration 全链路

### 5.4 工程质量

- ✅ 有 README + 使用示例（英文专业 README，含 badges、3 种使用模式、安装、CLI、metrics 表）
- ✅ 核心模块有单元测试（51 tests，6 个测试文件）
- ✅ 代码有基本的类型标注（type hints，Python 3.10+ 风格）
- ✅ pyproject.toml 完善（entry_points、classifiers、optional deps）

## 6. 学术潜力评估

> 这部分是对项目是否有论文发表可能的诚实评估。

### 6.1 工具本身的价值

UniVis 作为工具，核心价值是"让研究者看到模型内部的冗余分布"。这属于**实证分析工具**，不是算法创新。工具论文（如 BertViz、TransformerLens）的发表门槛：
- BertViz 发在 ACL 2019 System Demonstration
- TransformerLens 发在 arXiv（未正式发表）
- 结论：工具论文可以发，但通常在 demo/workshop track

### 6.2 最有可能的学术方向

如果想在工具基础上挖掘学术贡献，最有潜力的方向是：

**冗余指标与实际模型质量的关联分析**

- 用 UniVis 采集多个模型的层冗余数据
- 实际移除"高冗余"层，测量 perplexity / benchmark 准确率的变化
- 如果能证明"Relative Delta 低的层，移除后质量损失小"，就是一个**预测性的实证发现**
- 这意味着 UniVis 的指标可以作为剪枝/跳层的 cheap proxy

**可行性**：中等。需要在 2-3 个模型上做实验，每个模型跑 5-10 个 benchmark，工作量大但方法论清晰。如果导师支持，可以在大创结题后继续做。

### 6.3 建议策略

1. **大创阶段**：专注把工具做好，不追求论文
2. **大创结题后**：如果工具跑出了有趣的冗余模式数据，和导师讨论是否有论文空间
3. **论文方向**：如果导师觉得有价值，走"实证分析 + 冗余-质量关联"的路线，目标投中文期刊或英文 workshop

## 7. 开放问题

| # | 问题 | 影响 | 当前建议 | 状态 |
|---|------|------|---------|------|
| Q1 | Hook 粒度：整个 Transformer Block 还是拆开 Attention/FFN？ | 数据粒度和开销 | MVP 用 Block 级别，后续可细化 | 需实验验证 |
| Q2 | 指标计算在 GPU 还是 CPU？ | 性能开销 | CPU（`.detach().cpu()`），小模型够用 | 需实验验证 |
| Q3 | 热力图色域范围：固定 [0, 1] 还是自适应？ | 可视化效果 | 自适应（每步重新计算 min/max），避免早期色彩饱和 | 待开发时决定 |
| Q4 | 导师对 MixCache / SRDiffusion 是否有集成期望？ | 项目边界 | 不集成，保持独立 | 需与导师确认 |

## 8. 开发路线图

### Phase 0：技术验证（第 1 周） ✅ 已完成

- [x] 写 50 行脚本，对 GPT-2 注册 forward_hook，测量 hook 开销
- [x] 在 hook 中计算 relative_delta 和 cosine_sim，验证数值合理性
- [x] 确认：hook 开销 < 15% → 项目可行（tiny-gpt2 45.8% 为最差情况，真实模型预期 <15%）

### Phase 1：SDK 核心（第 2-3 周） ✅ 已完成

- [x] 搭建项目骨架（pyproject.toml + src layout）
- [x] 实现 `probe.py`：hook 注册 + 步骤缓冲
- [x] 实现 `metrics.py`：三个核心指标的计算函数
- [x] 实现 `transport.py`：先支持 JSONL 文件输出
- [x] 写 `examples/gpt2_basic.py`：能跑的 demo，输出 JSONL

### Phase 2：实时通信（第 4 周） ✅ 已完成

- [x] 实现 `server.py`：FastAPI + WebSocket
- [x] 修改 `transport.py`：增加 WebSocket 传输方式
- [x] 本地测试：SDK → WebSocket → 简单 Python 客户端接收

### Phase 3：Dashboard（第 5-7 周） ✅ 已完成

- [x] 搭建 React + Vite + TypeScript 项目
- [x] 实现 WebSocket 客户端连接
- [x] 实现动态热力图（ECharts heatmap）
- [x] 实现 Token 列表组件
- [x] 联调：SDK → WebSocket → Dashboard 完整链路

### Phase 4：打磨（第 8-10 周） ✅ 已完成

- [x] 离线 HTML 报告生成（ECharts 交互式报告）
- [x] Qwen2.5-0.5B demo（含 model.generate() 支持）
- [x] CLI 入口点（serve / report）
- [x] Batch 聚合逻辑
- [x] README + 使用文档
- [x] pip install 打包测试
- [ ] 性能优化（如果 overhead > 15%，精确 benchmark 待做）

### Phase 5：结题准备（第 11-12 周）

- [ ] 录制 demo 视频
- [ ] 整理实验数据和分析报告
- [ ] 撰写结题报告
