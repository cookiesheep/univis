# MXMACA 适配指南与环境矩阵

UniVis 的国产算力适配目标：在沐曦 MetaX 曦云 C500 + MXMACA 软件栈上验证 hook 指标采集通路与跨规模冗余基线，使冗余诊断跨 NVIDIA 与国产 GPU 通用。本页是 [README](../README.md) 中国产算力一节的展开版，随各阶段推进持续更新。

> **状态速览（2026-08）**：NVIDIA L20 与 CPU 已验证；MetaX 曦云 C500 处于**计划中**——本页描述的是目标与判据，尚未有 C500 实测数据。任何「已验证」表述只会在数据真实产生后出现。

## 为什么 UniVis 天然可移植

| 设计事实 | 对国产卡的意义 |
|---|---|
| 纯 PyTorch `forward` hook 采集，无 CUDA 专有 API | 只要有可用的 PyTorch 适配版（如 mcPyTorch）即可运行 |
| 指标端侧计算，单步 1–2 KB JSON | 采集开销不随硬件变化，不构成新瓶颈 |
| 离线路径（JSONL → HTML 报告）纯 CPU | 报告复核与 CI 无需任何 GPU |
| 不依赖 Nsight 等 GPU 厂商 profiler 工具链 | 诊断结论跨硬件可比 |

## 环境矩阵

| 环境 | 硬件 | 状态 | 已验证内容 |
|---|---|---|---|
| NVIDIA L20（48GB） | NVIDIA GPU | ✅ 已验证 | Qwen2.5-0.5B / 3B / 7B + 27B 级 hybrid 完整诊断（约 54× 参数跨度） |
| CPU（无 GPU） | — | ✅ 已验证 | 全量 80 项单元测试、离线报告渲染 |
| MetaX 曦云 C500（64GB HBM2e） | 沐曦 GPU + MXMACA | 🔄 计划中（阶段 A → B） | — |

## 阶段判据

### 阶段 A —— 通路验证（目标环境：单卡 C500 + 官方 MACA 镜像）

- 在 C500 上 `pip install -e .`（使用 MXMACA 适配版 torch，**不要**用公版 torch 覆盖）
- Qwen2.5-0.5B（FP16/BF16）完整生成一次，`transport="file"` 全程采集
- **完成判据**：产出标准 HTML 报告 + JSONL 留档 + 环境指纹（见下）

### 阶段 B —— 跨规模基线

- 0.5B / 3B / 7B 三个规模，同提示词集合
- **完成判据**：三份基线报告 + 与 NVIDIA L20 的同模型同提示词对比表（冗余画像是否一致）

### 阶段 C —— 反哺沐曦生态（远期）

- 以冗余定位结论作为 FlagGems / mcTriton 算子优化的输入示例
- **完成判据**：至少一个「诊断定位 → 算子/层优化」的完整案例

## 证据文件规范（无卡可审计）

每个阶段的验证证据统一存放在 `docs/mxmaca/` 目录，任何人无需 GPU 即可复核：

```
docs/mxmaca/
├── phase-a/
│   ├── report.html          # 自包含离线报告（直接浏览器打开）
│   ├── session.jsonl        # 原始指标数据（python -m univis report 可重渲染）
│   ├── env-fingerprint.txt  # MXMACA / mcPyTorch 精确版本、模型与 dtype、mx-smi 输出
│   └── README.md            # 复现命令（提示词、随机种子、生成参数）
└── phase-b/
    └── ...（同上结构 + l20-comparison.md 对比表）
```

环境指纹至少包含：`mx-smi` 输出、`torch.__version__`（应为 `+metax` 后缀的适配版）、MXMACA 版本、模型 ID 与 dtype、生成参数。诊断结论一律标注「仅限所测模型，不外推」。

## 如何获得 C500 环境

- **Gitee AI（模力方舟）算力市场**：按小时租用曦云 C500 实例，选择官方 MACA 镜像，适合阶段 A/B 的短期实验；
- **沐曦 × 模力方舟 AI 技能认证**：全程云端实操并提供算力，完成可获沐曦联签证书；
- **Gitee 高校版「高校人工智能助力计划」**：符合条件的高校师生可申请算力代金券。

注意事项：C500 单卡 64GB HBM2e；实例内**切勿 `pip install torch` 覆盖适配版**（公版 torch 会导致 GPU 不可用）；不支持 FP8，权重选 FP16 / BF16；GPU 状态用 `mx-smi` 查看。

## 范围声明

本页「已验证」条目仅覆盖所列模型、架构与软件栈版本。冗余结论不外推到未测模型；国产卡上的结论在阶段 B 数据产生前一律按路线图表述。
