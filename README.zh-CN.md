# UniVis

**面向 Transformer 推理的开源观测工具 —— 看清哪些层在计算、哪些层在冗余。**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-80-brightgreen.svg)](tests)

> English: [README.md](README.md)

UniVis 是一个轻量级推理诊断工具包，可附着在任意 PyTorch / HuggingFace Transformer 模型上，对推理过程中每一层的行为进行可视化。它通过 `forward` hook 在端侧实时计算紧凑指标，输出动态热力图与可独立打开的报告，让「计算究竟发生在哪里、又浪费在哪里」一目了然。

UniVis 是**度量与可视化**工具——它不修改、不剪枝、不加速你的模型，而是给出判断「该优化什么」的依据，并提供一条可实验的干预路径。

## 为什么需要

Transformer 推理成本高昂，但并非每一层、每一个生成步骤的贡献都相同。现有工具往往看向别处：

- **TensorBoard / W&B** 关注训练期标量（loss、学习率），在推理期把网络当作黑盒。
- **BertViz** 解释的是注意力语义，而非运行时算力开销。
- **NVIDIA Nsight / 各类 profiler** 停留在 GPU 算子级——功能强大，却难以回溯到「具体是哪一个 Transformer 层」。

UniVis 填补的是这一缺失的语义层：**推理期逐层、逐 token 的冗余**，每步仅传输几 KB。

## 特性

- **零侵入 hook** —— 不改动目标模型，挂上即用。
- **端侧计算指标** —— 张量在 hook 内部就地降维为标量，单步负载仅约 1–2 KB，对推理速度的影响可忽略。
- **5 项核心指标** —— Relative Delta、层间余弦相似度、激活稀疏度、预测熵、显存变化。
- **架构自动识别** —— GPT-2、LLaMA / Mistral / Mixtral、Qwen（1.5 / 2 / 2.5 / 3）、BERT 系列。
- **三种输出模式** —— JSONL 日志、自包含离线 HTML 报告、实时 WebSocket Dashboard。
- **`model.generate()` 集成** —— 通过 HuggingFace `LogitsProcessor` 接入，无需改动生成循环。
- **丰富的报告** —— 模型 MRI 同心环、层级脉冲、数据河流、plasma 热力图、带趋势标注的冗余排名。
- **多模型对比** —— `univis compare` 跨模型冗余对比 CLI，含雷达图与对比表。
- **Pilot 干预（实验性）** —— 基于阈值的生成提前终止，并定量评估对 perplexity 的影响。

## 工作原理

```
你的模型
    │  univis.attach(model)
    ▼
┌──────────────────────── univis SDK ───────────────────────┐
│  detection → probe (forward_hook) → metrics → transport    │
│           (自动识别)   (逐层)        (标量)    (JSONL/HTTP)  │
│                                                            │
│   指标在端侧算好 ── 只输出 KB 级 JSON                       │
└───────────────────────────┬────────────────────────────────┘
                            │ HTTP POST → WebSocket
                            ▼
                  React + ECharts Dashboard
                  （实时热力图 · 统计 · 过滤）
                            │
                            ▼  tracker.finish()
                  可独立打开的 HTML 报告
```

核心设计是**边缘计算**：高维激活在 hook 内部就地降维为标量指标，监控永远不会阻塞推理。

## 快速开始

```bash
pip install -e ".[dev]"
```

### 方式 A —— 最简（仅 SDK，离线报告）

```python
import univis

tracker = univis.attach(model, transport="file")
for token_id in generate_loop():
    tracker.on_step(token_id)
report_path = tracker.finish()   # → 自包含 HTML 报告
```

### 方式 B —— `model.generate()` 集成

```python
tracker = univis.attach(model)
lp = tracker.logits_processor(tokenizer)
output = model.generate(input_ids, logits_processor=[lp], max_new_tokens=50)
tracker.finish()
```

### 方式 C —— 实时 Dashboard

```bash
python -m univis serve              # 终端 1：WebSocket 服务
cd dashboard && npm run dev         # 终端 2：Dashboard（:5173）
python your_script.py               # 终端 3：以 transport="websocket" 运行推理
```

## 命令行

```bash
python -m univis serve --port 8765                # 启动 WebSocket 服务
python -m univis report data.jsonl -o out.html     # 由 JSONL 渲染 HTML 报告
python -m univis compare a.jsonl b.jsonl -o c.html # 跨模型对比
```

## 指标

| 指标 | 公式 | 含义 |
|---|---|---|
| Relative Delta | `‖output − input‖₂ / ‖input‖₂` | 该层对表示的改变幅度 |
| Cosine Similarity | `cos(input, output)` | 输入输出方向一致性 |
| Activation Sparsity | `count(|x| < ε) / total` | 近零激活比例 |
| Prediction Entropy | `H(softmax(logits))` | 该步模型的犹豫程度 |
| VRAM Delta | `Δ torch.cuda.memory_allocated()` | 该步引入的显存压力 |

Relative Delta 是热力图主指标。由于 Transformer 采用残差连接，余弦相似度普遍饱和在 0.9 以上，层间差异被模糊；Relative Delta 直接度量「这层改了多少」，对残差架构里的冗余更敏感。

## 支持模型

| 架构 | 变体 |
|---|---|
| GPT-2 | gpt2, gpt2-medium, gpt2-large, gpt2-xl |
| LLaMA | LLaMA 1/2/3, Mistral, Mixtral |
| Qwen | Qwen 1.5, Qwen 2, Qwen 2.5, Qwen 3 |
| BERT | BERT, RoBERTa, ALBERT, DeBERTa |

架构由模型 config 自动识别，无需手动注册。已在 Qwen2.5-0.5B / 3B / 7B 与 Qwen3-27B 级 hybrid 模型上完成验证。

## 路线图

- **Pilot 提前终止** —— 把度量变成行动：当模型足够确信时提前结束生成，并给出明确的「质量（perplexity）↔ 加速」权衡曲线。（实测简单跳层会损害质量，因此聚焦于基于置信度的提前终止。）
- **国产算力适配** —— 在国产 AI 算力（如 MXMACA 软件栈）上验证 hook 与指标采集，使冗余诊断跨 NVIDIA 与国产 GPU 通用。
- **DiT / 视频生成** —— 监控去噪步之间的时序冗余，实现 time-step 级提前终止。
- **子层粒度** —— 从 Transformer Block 下钻到 Attention 与 FFN。
- **冗余—质量关联研究** —— UniVis 指标与剪枝/跳层后质量损失的实证关联。

## 架构与测试

完整模块设计、数据模型与 API 见 [ARCHITECTURE.md](ARCHITECTURE.md)；范围与动机见 [PRD.md](PRD.md)。SDK 由 **80 个单元测试**（9 个文件：metrics / probe / report / server / tracker / pilot / integration / 三端联调）覆盖。

```
src/univis/      Python SDK —— attach() → on_step() → finish()
dashboard/       React 18 + TypeScript + ECharts 前端
examples/        可运行示例
tests/           80 个单元测试
```

## 参与贡献

欢迎贡献——新的指标插件、模型适配、Dashboard 视图与 benchmark 都有价值，参见 [CONTRIBUTING.md](CONTRIBUTING.md)。`metrics.py` 中的指标函数是纯函数、可独立测试，新增指标的风险很低。

## 许可证

[MIT](LICENSE)。

## 鸣谢

感谢中山大学 AI Systems 研究组提供的算力资源与技术讨论。本项目基于 PyTorch、FastAPI、React、ECharts 构建。
