# UniVis

Transformer 推理冗余诊断与可视化工具。

## 项目结构

```
src/univis/       # Python SDK（pip install -e .）
src/univis/server.py  # FastAPI WebSocket 服务
dashboard/        # React + ECharts 前端（DM Sans + JetBrains Mono 字体）
examples/         # 可运行示例
tests/            # 单元测试（67 tests）
```

## 开发命令

```bash
# SDK
pip install -e ".[dev]"         # 开发模式安装
pytest tests/                   # 跑测试（67 tests）
python examples/gpt2_basic.py   # 小模型 demo

# Server
python -m univis serve          # 启动 WebSocket 服务（:8765）

# Dashboard
cd dashboard/ && npm install    # 安装依赖
cd dashboard/ && npm run dev    # 启动开发服务器（:5173）
```

## 完整联调（三终端）

```bash
# 终端 1：启动 server
python -m univis.server

# 终端 2：启动 dashboard
cd dashboard/ && npm run dev

# 终端 3：运行推理 demo（SDK → HTTP POST → server → WS → dashboard）
python examples/gpt2_basic.py
```

## 编码规范

- Python 3.10+，使用 type hints
- 4 空格缩进，单引号字符串
- 函数和方法加简短 docstring（一行）
- 不写多余注释，代码本身应该自解释
- 模块级别 import 顺序：stdlib → third-party → local
- 所有公开 API 在 `__init__.py` 中导出

## 架构参考

- `PRD.md`：产品需求、功能范围、成功标准
- `ARCHITECTURE.md`：模块设计、数据模型、API 签名、技术决策

## 测试环境

- WSL Ubuntu（douban_crawler conda env）
- `HF_ENDPOINT=https://hf-mirror.com`（国内 HuggingFace 镜像）
- 代理：`export http_proxy=http://$(ip route show default | awk '{print $3}'):7897`
- 模型：`sshleifer/tiny-gpt2`（测试用，5MB）
- L20 服务器：Qwen2.5-0.5B/3B/7B 全部验证通过
- SSH L20：`ssh L20_public`，GPU 4-7 空闲（各 48GB）
