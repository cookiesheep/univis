# Phase A 验证记录 — MetaX 曦云 C500

**状态：已完成（2026-08-19）**。UniVis 在沐曦曦云 C500（模力方舟租用实例）上完成指标采集通路验证：Qwen2.5-0.5B-Instruct 完整生成，forward hook 逐层采集正常，产出标准 HTML 报告与 JSONL 留档。全程未对 UniVis 代码做任何国产卡特化修改。

## 完成判据核对

| 判据 | 结果 |
|---|---|
| Qwen2.5-0.5B 完整生成一次 | ✅ 3 个中文提示词 × 128 token，greedy 解码，生成正常 |
| HTML 报告 | ✅ `univis_report_3d2a1dca.html`（自包含，直接打开） |
| JSONL 留档 | ✅ `univis_data_3d2a1dca.jsonl`（384 步 × 24 层） |
| 环境指纹 | ✅ `env-fingerprint.txt`（mx-smi + 精确软件版本） |

## 环境（详见 env-fingerprint.txt）

- GPU：MetaX C500（sGPU 切分实例：16GB 显存配额 / 64GB 物理，25% 算力）
- 驱动 / 运行时：Kernel 3.8.30，MACA 3.3.0.15，mx-smi 2.2.9
- 软件：Python 3.10，torch **2.8.0+metax3.3.0.2**，transformers 4.57.3（镜像预装）
- 实例镜像：官方 MACA 3.x + PyTorch 2.8.0（模力方舟算力市场）

## 复现步骤

在模力方舟租用「曦云 C500 + PyTorch 2.8.0」实例后（SSH 进入）：

```bash
git clone https://github.com/cookiesheep/univis /data/univis
export MACA_PATH=/opt/maca        # 非交互 shell 必需；交互登录时镜像自动注入
export HF_ENDPOINT=https://hf-mirror.com
PYTHONPATH=/data/univis/src /opt/conda/bin/python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import univis

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', dtype=torch.bfloat16).to('cuda')
model.eval()

tracker = univis.attach(model, transport='file', output_dir='/data/phase-a')
lp = tracker.logits_processor(tok)
text = tok.apply_chat_template([{'role': 'user', 'content': '用三句话介绍 Transformer 架构的核心思想。'}],
                               tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors='pt').to(model.device)
with torch.no_grad():
    model.generate(**enc, logits_processor=[lp], max_new_tokens=128, do_sample=False)
print(tracker.finish())
EOF
```

注意事项：实例内**不要 `pip install torch`**（会覆盖 `+metax` 适配版导致 GPU 不可用）；持久化目录仅 `/root` 与 `/data`。

## 文件清单

| 文件 | 说明 |
|---|---|
| `univis_data_3d2a1dca.jsonl` | 原始指标数据（384 步 × 24 层，session 3d2a1dca） |
| `univis_report_3d2a1dca.html` | 自包含离线报告（浏览器直接打开，无需 GPU） |
| `env-fingerprint.txt` | mx-smi 输出 + 软件栈精确版本（已脱敏，不含实例地址） |
| `run.log` | 当次运行完整日志 |

所有结论仅限本页所列模型与软件栈版本，不外推到未测模型。
