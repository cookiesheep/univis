# UniVis 技术架构文档

> 最后更新：2026-04-30 | 版本：v0.2 | 状态：开发中
> 本文档是开发时的主要技术参考，与 PRD.md 配合使用。

## 1. 系统架构

```
用户代码
│
│  import univis
│  tracker = univis.attach(model)
│  # 方式 A：手动循环 + tracker.on_step()
│  # 方式 B：model.generate(logits_processor=[tracker.logits_processor(tokenizer)])
│  tracker.finish()  → 生成 HTML 报告
│
▼
┌───────────────────────────────────────────────────────┐
│                    univis SDK                         │
│                                                       │
│  detection.py    probe.py       metrics.py   report.py│
│  (模型结构检测)  (hook注册)     (指标计算)   (HTML报告) │
│       │            │              │              │     │
│       └────────────┘              │              │     │
│            tracker.py ◀───────────┘              │     │
│       (用户API: on_step/finish/logits_processor) │     │
│                       │                           │     │
│                 transport.py                       │     │
│            ┌────────┼────────┐                     │     │
│            │        │        │                     │     │
│         File    HttpPush  Multi               __main__.py│
│        (JSONL)  (→Server)                     (CLI入口) │
└────────────────┬─────┴────────────────────────────────┘
                 │ HTTP POST → Server → WebSocket
                 ▼
        ┌──────────────────┐
        │   Dashboard      │
        │                  │
        │  React + ECharts │
        │  动态热力图       │
        │  Token 列表       │
        │  层过滤器         │
        └──────────────────┘
```

三条独立使用路径：
1. **纯 SDK**：`attach()` → `on_step()` → `finish()` → HTML 报告（不依赖 Server/Dashboard）
2. **SDK + Server + Dashboard**：实时可视化推送（HTTP POST → WebSocket → ECharts）
3. **CLI**：`python -m univis serve` 或 `python -m univis report <jsonl>`

核心数据流：
1. **Probe（探针）**：注册在模型上，每次 forward 自动触发，采集 tensor
2. **Metrics（计算）**：将 tensor 压缩为标量指标（支持 batch>1 聚合）
3. **Transport（传输）**：将指标发送到外部（JSONL 文件 / HTTP POST）

## 2. 数据流（Token 生成的一个 Step）

```
用户代码：model(input_ids)         # 触发一次 forward pass
    │
    ▼
Layer 0 forward_hook 触发
    │ input = block_0 的输入 (来自 embedding)
    │ output = block_0 的输出
    │ 计算: relative_delta, cosine_sim, sparsity
    │ 存入 step_buffer[0]
    ▼
Layer 1 forward_hook 触发
    │ input = block_1 的输入 (block_0 的输出)
    │ output = block_1 的输出
    │ 计算: relative_delta, cosine_sim, sparsity
    │ 存入 step_buffer[1]
    ▼
... (所有 N 层完成)
    ▼
用户代码：tracker.on_step(token_idx=42)
    │
    ▼ 打包 step_buffer → JSON message
    │ 添加: entropy (从最终 logits), vram_delta
    ▼
transport.send(json_message)
    │
    ▼ WebSocket 推送
    ▼
Dashboard 追加一列到热力图
```

## 3. 数据模型（JSON Schema）

### 3.1 会话开始消息

推理开始时发送一次，描述模型信息。

```json
{
  "type": "session_start",
  "session_id": "a1b2c3d4",
  "model_name": "gpt2",
  "num_layers": 12,
  "hidden_dim": 768,
  "num_heads": 12,
  "layer_names": ["transformer.h.0", "transformer.h.1", "..."],
  "prompt_text": "Once upon a time",
  "prompt_tokens": 4
}
```

### 3.2 步骤消息

每生成一个 token 发送一次。

```json
{
  "type": "step",
  "session_id": "a1b2c3d4",
  "token_idx": 42,
  "timestamp_ms": 1718889000123,
  "generated_token": "the",
  "global": {
    "vram_total_mb": 1024.5,
    "prediction_entropy": 2.34
  },
  "layers": [
    {
      "idx": 0,
      "name": "transformer.h.0",
      "relative_delta": 0.052,
      "cosine_sim": 0.9987,
      "sparsity": 0.12
    },
    {
      "idx": 1,
      "name": "transformer.h.1",
      "relative_delta": 0.187,
      "cosine_sim": 0.9823,
      "sparsity": 0.05
    }
  ]
}
```

### 3.3 会话结束消息

推理结束时发送一次。

```json
{
  "type": "session_end",
  "session_id": "a1b2c3d4",
  "total_tokens": 50,
  "total_time_ms": 3200,
  "avg_overhead_pct": 8.5,
  "report_path": "univis_report_20260430.html"
}
```

### 3.4 数据量估算

| 模型 | 层数 | 单步 JSON 大小 | 100 token 总量 |
|------|------|---------------|---------------|
| GPT-2 (124M) | 12 | ~800 B | ~80 KB |
| Qwen2.5-0.5B | 24 | ~1.5 KB | ~150 KB |
| Qwen2.5-7B | 28 | ~1.8 KB | ~180 KB |

完全在 WebSocket 承受范围内。

## 4. SDK API 设计

### 4.1 公开接口

```python
# univis/__init__.py

def attach(
    model: torch.nn.Module,
    project: str = "default",
    metrics: list[str] | None = None,  # 默认全部指标
    hook_prefixes: list[str] | None = None,  # 自动检测或手动指定
    transport: str = "websocket",  # "websocket" | "file"
    dashboard: bool = True,
    port: int = 8765,
) -> Tracker:
    """
    将 UniVis tracker 附着到模型上。

    Returns:
        Tracker 实例，用户在推理循环中调用其方法。
    """
```

```python
class Tracker:
    def on_step(
        self,
        token_index: int,
        generated_token: str = "",
        logits: torch.Tensor | None = None,  # 用于计算 entropy
    ) -> None:
        """
        标记一个 token 生成步骤完成。
        触发：缓冲区打包 → 指标发送 → Dashboard 更新。
        """

    def finish(self, output_dir: str = ".") -> str:
        """
        结束追踪，移除 hooks，生成离线报告。

        Returns:
            报告文件路径。
        """

    def remove(self) -> None:
        """仅移除 hooks，不生成报告。用于异常退出时清理。"""
```

### 4.2 典型使用方式

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import univis

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

# 附着 UniVis
tracker = univis.attach(model, project="gpt2_demo", dashboard=True)

# 手动推理循环
input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids

for i in range(50):
    with torch.no_grad():
        outputs = model(input_ids)

    # 取下一个 token
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    token_text = tokenizer.decode(next_token[0])

    # 通知 tracker：这一步完成了
    tracker.on_step(
        token_index=i,
        generated_token=token_text,
        logits=outputs.logits[:, -1, :],  # 可选，用于 entropy
    )

    # 拼接到输入
    input_ids = torch.cat([input_ids, next_token], dim=-1)

# 结束，生成报告
report = tracker.finish()
print(f"Report: {report}")
```

## 5. 模块设计

### 5.1 probe.py — Hook 管理 ✅ 已实现

职责：在模型指定模块上注册 forward_hook，将每次 forward 的 input/output 缓存起来。

```python
class ModelProbe:
    def __init__(self, model: torch.nn.Module, hook_prefixes: list[str]):
        self.model = model
        self.hook_handles: list[RemovableHook] = []
        self.step_buffer: list[dict] = []  # 当前步骤的各层指标
        self._prev_layer_name: str | None = None
        self._register_hooks(hook_prefixes)

    def _register_hooks(self, prefixes: list[str]) -> None:
        """遍历 model.named_modules()，匹配 prefixes，注册 hook。"""

    def _hook_fn(self, layer_name: str, layer_idx: int):
        """返回一个闭包，作为 forward_hook 的回调。"""
        def hook(module, input, output):
            # input[0]: 前一层的输出 (Tensor)
            # output:   当前层的输出 (Tensor)
            inp = input[0].detach().cpu()
            out = output.detach().cpu()

            metrics = {
                "idx": layer_idx,
                "name": layer_name,
                "relative_delta": compute_relative_delta(inp, out),
                "cosine_sim": compute_cosine_sim(inp, out),
                "sparsity": compute_sparsity(out),
            }
            self.step_buffer.append(metrics)

        return hook

    def flush_step(self, token_index: int, **extra) -> dict:
        """将 step_buffer 打包为 Step Message，清空缓冲区。"""

    def remove_hooks(self) -> None:
        """移除所有已注册的 hook。"""
```

**自动检测 hook 目标的逻辑**：

```python
def detect_block_prefixes(model: torch.nn.Module) -> list[str]:
    """
    自动检测 Transformer Block 的名称前缀。
    GPT-2:     "transformer.h."
    LLaMA:     "model.layers."
    Qwen2.5:   "model.layers."
    """
    CANDIDATES = ["transformer.h.", "model.layers.", "encoder.layer.", "decoder.block."]
    names = [name for name, _ in model.named_modules()]
    for prefix in CANDIDATES:
        if any(n.startswith(prefix) and not n.replace(prefix, "").count(".") for n in names):
            return [prefix]
    # fallback: 找所有包含 "block" 或 "layer" 的顶层模块
    ...
```

### 5.2 metrics.py — 指标计算 ✅ 已实现

所有函数接收 CPU tensor，返回 float。支持 batch>1 输入（3D tensor），自动取 `[:, -1, :]` 最后 token 位置，batch 维度做 per-item 计算后取 mean。

```python
def compute_relative_delta(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> float:
    """
    ||output - input||₂ / ||input||₂

    衡量该层对表示的改变幅度。
    值越小 = 该层越"冗余"（没怎么改变表示）。
    """
    delta = output_tensor - input_tensor
    # 只取最后一个 token 位置（decode 阶段 shape=[1,1,dim]）
    if delta.dim() == 3:
        delta = delta[:, -1, :]
        input_tensor = input_tensor[:, -1, :]
    norm_delta = delta.float().norm().item()
    norm_input = input_tensor.float().norm().item()
    if norm_input < 1e-10:
        return 0.0
    return norm_delta / norm_input


def compute_cosine_sim(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> float:
    """
    cos(input, output)

    值越接近 1 = 输出和输入方向越一致（但受残差连接影响会普遍偏高）。
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor[:, -1, :]
        output_tensor = output_tensor[:, -1, :]
    a = input_tensor.float().flatten()
    b = output_tensor.float().flatten()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_sparsity(tensor: torch.Tensor, threshold: float = 1e-6) -> float:
    """
    |x| < threshold 的元素比例。

    值越高 = 该层越稀疏（大量神经元未激活）。
    """
    if tensor.dim() == 3:
        tensor = tensor[:, -1, :]
    return (tensor.abs() < threshold).float().mean().item()


def compute_entropy(logits: torch.Tensor) -> float:
    """
    H(softmax(logits))

    只对最后一个 token 位置计算。
    值越低 = 模型越确定下一个词。
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]  # [1, vocab_size]
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.item()
```

### 5.3 transport.py — 数据传输 ✅ 已实现

```python
from abc import ABC, abstractmethod

class Transport(ABC):
    @abstractmethod
    def send(self, message: dict) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class FileTransport(Transport):
    """写入 JSONL 文件。用于开发和调试。"""
    def __init__(self, path: str): ...
    def send(self, message: dict) -> None:
        self.file.write(json.dumps(message) + "\n")
    def close(self) -> None:
        self.file.close()


class HttpPushTransport(Transport):
    """通过 HTTP POST 推送到 UniVis server。"""
    def __init__(self, base_url: str, session_id: str): ...
    def send(self, message: dict) -> None:
        urllib.request.urlopen(req, timeout=5)
    def close(self) -> None: ...


class MultiTransport(Transport):
    """支持同时写入多个 transport。"""
    def __init__(self, transports: list[Transport]): ...
    def send(self, message: dict) -> None:
        for t in self._transports: t.send(message)
    def close(self) -> None:
        for t in self._transports: t.close()
```

### 5.4 server.py — WebSocket 服务 ✅ 已实现

```python
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# API endpoints (已实现):
# POST /api/push       — 接收 SDK 推送的指标数据
# WS   /ws/{id}        — WebSocket 实时转发给 Dashboard
# GET  /api/sessions   — 列出活跃 sessions
# GET  /api/health     — 健康检查
```

### 5.5 report.py — 离线 HTML 报告 ✅ 已实现

推理结束后生成自包含 HTML 文件，嵌入 ECharts CDN，无需服务器即可打开。

```python
def generate_report(
    steps: list[dict],
    meta: dict[str, Any],
    output_path: str | Path,
) -> str:
    """
    将所有 step 数据写入 HTML 模板。

    包含：
    - 交互式热力图（ECharts heatmap，hover 显示详细指标）
    - Prediction Entropy 曲线图
    - 每层冗余度统计表（avg delta/cosim/sparsity 排名）
    - 生成文本全文显示
    - XSS 防护：_safe_json() 转义 </script>
    """
```

### 5.6 __main__.py — CLI 入口 ✅ 已实现

```python
# python -m univis serve --host 0.0.0.0 --port 8765
# python -m univis report <jsonl_path> -o output.html

def main() -> None:
    """argparse 解析 serve 和 report 两个子命令。"""
    # serve: 调用 uvicorn.run('univis.server:app', ...)
    # report: 读取 JSONL → 提取 steps/meta → generate_report()
```

## 6. Dashboard 设计

### 6.1 技术栈

- React 18 + TypeScript
- Vite（构建工具）
- ECharts 5（通过 echarts-for-react）
- 原生 WebSocket API

### 6.2 组件结构

```
App ✅ 已实现
├── ConnectionBar        # WebSocket 连接状态 + session 选择
├── MainView
│   ├── HeatmapView      # 动态热力图（核心），丰富 Tooltip
│   ├── TokenList        # 已生成 token 序列
│   └── LayerFilter      # 层选择过滤器（多选框，All/None 快捷按钮）
├── StatsPanel           # 实时统计（平均冗余、总 token、VRAM）
└── ControlBar           # 开始/暂停/重置
```

### 6.3 热力图规格

```
┌─────────────────────────────────────────────┐
│ Token →  0   5   10  15  20  25  30  35  40 │
│ Layer ↓                                     │
│  0      ■   ■   ■   ■   ■   ■   ■   ■   ■  │
│  1      ■   ■   ■   ■   ■   ■   ■   ■   ■  │
│  2      ■   ■   ■   ■   ■   ■   ■   ■   ■  │
│  ...                                        │
│  11     ■   ■   ■   ■   ■   ■   ■   ■   ■  │
│                                             │
│  图例: ■ 深红 = 高活跃  ■ 白色 = 高冗余      │
│        (relative_delta 值驱动颜色)           │
└─────────────────────────────────────────────┘
```

- X 轴：Token 位置（0, 1, 2, ..., current）
- Y 轴：Layer 索引（0, 1, ..., N-1）
- 颜色：`relative_delta` 值 → visualMap 映射
  - 深红 = 高 delta = 层在积极计算
  - 白色/浅蓝 = 低 delta = 层冗余
- 动态行为：每收到一个 step 消息，追加一列
- 交互：hover 显示具体数值，点击跳转到 token 位置

### 6.4 ECharts 配置要点

```typescript
const option = {
  tooltip: { position: "top" },
  grid: { height: "70%", top: "10%" },
  xAxis: { type: "category", data: tokenLabels, name: "Token" },
  yAxis: { type: "category", data: layerLabels, name: "Layer" },
  visualMap: {
    min: 0,
    max: 0.3, // relative_delta 通常 < 0.3
    calculable: true,
    orient: "horizontal",
    inRange: { color: ["#f0f0f0", "#fee", "#fdd", "#fbb", "#f66", "#e11"] },
  },
  series: [{
    type: "heatmap",
    data: heatmapData, // [[tokenIdx, layerIdx, value], ...]
    progressive: 1000,
    animation: false, // 大数据量时关闭动画
  }],
};
```

## 7. 性能预算

### 7.1 Hook 开销分解

| 操作 | 预计耗时 (GPT-2) | 说明 |
|------|-----------------|------|
| GPU→CPU 拷贝 (hidden_dim=768) | ~0.1 ms | 12 层 × 拷贝 2 次 = ~2.4 ms |
| Relative Delta 计算 | ~0.01 ms | CPU float 运算 |
| Cosine Sim 计算 | ~0.01 ms | CPU float 运算 |
| Sparsity 计算 | ~0.01 ms | CPU float 运算 |
| JSON 序列化 + 发送 | ~0.1 ms | 数据量 < 1 KB |
| **单步总计** | **~3 ms** | |
| GPT-2 原始推理 (100 token) | ~300 ms | CPU 上 |
| 开销占比 | ~1% | |

> 以上为估算值。真实模型实测见下方。

### 7.2 真实模型实测数据

| 模型 | 硬件 | 配置 | 结果 |
|------|------|------|------|
| Qwen2.5-0.5B-Instruct | L20 GPU | 24 层 × 50 步 | 50 tokens ~8s（含模型加载），overhead 可接受 |
| sshleifer/tiny-gpt2 | CPU (WSL) | 12 层 × 100 步 | Hook 开销 ~45.8%（最差情况，模型极小导致拷贝占比高） |

### 7.3 风险点

- **大模型**：hidden_dim 增大 → GPU→CPU 拷贝变慢 → 可能需要只拷贝部分数据（采样或降精度）
- **KV Cache 场景**：decode 阶段只处理 1 个 token，tensor 形状是 [1, 1, dim]，拷贝量小
- **Prefill 阶段**：所有 prompt token 一次处理，tensor 形状是 [1, seq_len, dim]，拷贝量大 → 可以跳过 prefill 阶段的指标采集

## 8. 目录结构

```
univis/
├── pyproject.toml              # 包定义 + 依赖 + entry_points
├── PRD.md                      # 产品需求文档
├── ARCHITECTURE.md             # 本文件
├── README.md                   # 英文 README（badges、quick start、metrics 表）
│
├── src/
│   └── univis/
│       ├── __init__.py         # attach(), 公开 API
│       ├── __main__.py         # CLI 入口：serve / report ✅
│       ├── tracker.py          # Tracker 类（用户交互入口）✅
│       ├── probe.py            # ModelProbe（hook 注册 + 缓冲）✅
│       ├── metrics.py          # 指标计算函数（支持 batch 聚合）✅
│       ├── transport.py        # Transport 基类 + File/HttpPush/Multi ✅
│       ├── server.py           # FastAPI WebSocket 服务 ✅
│       ├── report.py           # ECharts HTML 报告生成 ✅
│       └── detection.py        # 自动检测模型结构 ✅
│
├── dashboard/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx              # 主框架：连接管理、session 选择、导出报告
│       ├── components/
│       │   ├── HeatmapView.tsx  # ECharts 热力图（丰富 Tooltip）
│       │   ├── TokenList.tsx    # Token 序列
│       │   ├── StatsPanel.tsx   # 统计面板
│       │   └── LayerFilter.tsx  # 层过滤器（多选框）✅ 新增
│       └── types.ts             # TypeScript 类型定义
│
├── examples/
│   ├── gpt2_basic.py            # tiny-gpt2 手动推理循环 demo
│   ├── qwen_basic.py            # Qwen2.5-0.5B benchmark（L20）
│   ├── qwen_generate.py         # Qwen2.5-0.5B + model.generate() + logits_processor
│   └── verify_features.py       # 三功能验证脚本
│
├── tests/                       # 51 tests across 6 test files
│   ├── test_metrics.py          # 18 tests：基础指标 + batch 聚合
│   ├── test_probe.py            # 6 tests：检测、hook、transport
│   ├── test_report.py           # 9 tests：数据提取、HTML 生成、XSS
│   ├── test_server.py           # 6 tests：health、push、sessions
│   ├── test_tracker.py          # 6 tests：logits_processor、报告集成
│   └── test_integration.py      # 6 tests：端到端 SDK→JSONL→report
│
└── reports/
    └── .gitkeep                 # 生成的报告存放目录
```

## 9. 依赖清单

### Python SDK

```
torch >= 2.0
numpy
fastapi >= 0.100
uvicorn
websockets
jinja2            # 报告模板
```

### Dashboard

```json
{
  "dependencies": {
    "react": "^18",
    "react-dom": "^18",
    "echarts": "^5",
    "echarts-for-react": "^3"
  },
  "devDependencies": {
    "typescript": "^5",
    "vite": "^5",
    "@vitejs/plugin-react": "^4"
  }
}
```

## 10. 开发环境

### SDK 开发

```bash
# 克隆项目
cd univis/

# 创建虚拟环境（推荐 conda）
conda create -n univis python=3.12
conda activate univis

# 可编辑模式安装
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 运行 demo
python examples/gpt2_basic.py
```

### Dashboard 开发

```bash
cd dashboard/
npm install
npm run dev        # http://localhost:5173
```

### 联调

```bash
# 终端 1：启动 WebSocket 服务
python -m univis.server

# 终端 2：启动 Dashboard
cd dashboard/ && npm run dev

# 终端 3：运行 demo
python examples/gpt2_basic.py
```

## 11. 关键技术决策记录

| 决策 | 选项 | 选择 | 原因 |
|------|------|------|------|
| 主指标 | Cosine Sim vs Relative Delta | Relative Delta | 残差连接使 cosine sim 普遍偏高，delta 更敏感 |
| Hook 粒度 | Block vs Attention/FFN | Block 级别 | 减少一半 hook 数量，先粗后细 |
| 指标计算位置 | GPU vs CPU | CPU (detach + cpu) | 小模型够用，避免 GPU 资源竞争 |
| 传输协议 | HTTP SSE vs WebSocket | WebSocket | 双向通信，Dashboard 可发控制指令 |
| 前端框架 | Streamlit vs React | React | 用户有经验，实时效果更好，可定制性高 |
| 前端图表 | D3 vs ECharts | ECharts | 热力图和矩形树图开箱即用，中文文档好 |
| Prefill 处理 | 采集 vs 跳过 | 跳过 | Prefill tensor 大，开销高，且不是分析重点 |
| 包管理 | setup.py vs pyproject.toml | pyproject.toml (src layout) | 现代标准，避免 import 混乱 |
| SDK→Server 通信 | WebSocket 直连 vs HTTP POST | HTTP POST | SDK 端无需维护长连接，更简单可靠 ✅ |
| generate() 集成 | 继承 LogitsProcessor vs 闭包 | 闭包（鸭子类型） | 不依赖 transformers 库 ✅ |
| Batch 聚合策略 | 全局 norm vs per-item→mean | per-item→mean | batch=1 数值完全一致，batch>1 语义正确 ✅ |
| 报告格式 | Jinja2 模板 vs f-string 拼接 | f-string + ECharts CDN | HTML 自包含，无外部文件依赖，Jinja2 不需要 ✅ |

## 12. 测试覆盖

### 12.1 测试概况

51 个测试，分布在 6 个测试文件中，覆盖 SDK 全链路。

| 文件 | 测试数 | 覆盖范围 |
|------|--------|---------|
| test_metrics.py | 18 | 基础指标计算（relative_delta/cosine_sim/sparsity/entropy）+ batch>1 聚合 |
| test_probe.py | 6 | 模型检测、hook 注册、step buffer flush |
| test_report.py | 9 | 数据提取、HTML 生成、XSS 防护（`</script>` 转义） |
| test_server.py | 6 | 健康检查、POST push、session 列表 |
| test_tracker.py | 6 | logits_processor 集成、报告生成集成 |
| test_integration.py | 6 | 端到端：SDK attach → on_step → JSONL → report → HTML |

### 12.2 运行测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
