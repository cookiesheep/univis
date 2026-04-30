"""
UniVis Phase 0: Hook 开销验证
===========================
目的：验证在 GPT-2 上注册 forward_hook 并计算指标，对推理速度的影响是否 < 15%。

运行方式：
    conda activate univis          # 或你的 Python 环境
    python phase0_verify.py

预期耗时：首次运行需下载 GPT-2 权重（~500MB），之后约 30 秒完成。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像，解决 huggingface.co 不可达问题

import time
import sys

import torch

# ── 1. 加载模型 ──────────────────────────────────────────────

MODEL_NAME = "sshleifer/tiny-gpt2"  # GPT-2 架构但极小，5MB，秒下载
# 完整 GPT-2 用 "gpt2" (548MB)，网络稳定后切换
NUM_TOKENS = 30            # 每次生成 30 个 token
NUM_RUNS   = 3             # 重复跑 3 次取平均
WARMUP     = 1             # 预热次数（不计时）

print(f"Loading {MODEL_NAME}...")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print(f"Model loaded. Device: {next(model.parameters()).device}")


# ── 2. 探测模型结构 ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 1: Model Structure")
print("=" * 60)

# 列出所有 Transformer Block 模块
block_modules = []
for name, module in model.named_modules():
    # GPT-2 的 block 名为 "transformer.h.0", "transformer.h.1", ...
    # 只取恰好两层深度的（排除子模块如 attn, mlp）
    if name.startswith("transformer.h.") and name.count(".") == 2:
        block_modules.append((name, module))
        print(f"  {name}: {type(module).__name__}")

print(f"\nFound {len(block_modules)} Transformer Blocks")


# ── 3. Baseline: 无 Hook 推理 ────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Baseline (no hooks)")
print("=" * 60)

prompt = "The meaning of life is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print(f"Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")
print(f"Generating {NUM_TOKENS} tokens per run, {NUM_RUNS} runs...\n")


def generate_no_hook(model, input_ids, num_tokens):
    """手动推理循环，无 hook。"""
    ids = input_ids.clone()
    with torch.no_grad():
        for _ in range(num_tokens):
            outputs = model(ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=-1)
    return ids


baseline_times = []
for run in range(WARMUP + NUM_RUNS):
    t0 = time.perf_counter()
    output_ids = generate_no_hook(model, input_ids, NUM_TOKENS)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    if run >= WARMUP:
        baseline_times.append(elapsed)
        print(f"  Run {run - WARMUP + 1}: {elapsed * 1000:.1f} ms")

avg_baseline = sum(baseline_times) / len(baseline_times)
print(f"  Average: {avg_baseline * 1000:.1f} ms")

baseline_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"  Output: ...{baseline_text[len(prompt):]}")


# ── 4. Hooked: 有 Hook + 指标计算的推理 ───────────────────────

print("\n" + "=" * 60)
print("STEP 3: With Hooks + Metrics Computation")
print("=" * 60)


def generate_with_hooks(model, input_ids, num_tokens):
    """手动推理循环，带 hook + 指标计算。"""
    # 存储每个 step 的指标
    all_steps = []
    current_step_layers = []

    def make_hook(layer_name, layer_idx):
        def hook_fn(module, inp, output):
            # inp 是 positional args 的 tuple: (hidden_states, ...)
            # output 是 forward 返回值: (hidden_states, present_kv) 或 (hidden_states,)
            hidden_in = inp[0]                           # [batch, seq_len, dim]
            hidden_out = output[0] if isinstance(output, tuple) else output

            # 只看最后一个 token 位置（decode 阶段）
            h_in = hidden_in[:, -1, :].detach().float()  # [batch, dim]
            h_out = hidden_out[:, -1, :].detach().float()

            # --- 三个核心指标 ---
            # 1) Relative Delta: ||output - input|| / ||input||
            delta_norm = (h_out - h_in).norm().item()
            input_norm = h_in.norm().item()
            rel_delta = delta_norm / max(input_norm, 1e-10)

            # 2) Cosine Similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                h_in.flatten().unsqueeze(0),
                h_out.flatten().unsqueeze(0),
            ).item()

            # 3) Activation Sparsity
            sparsity = (h_out.abs() < 1e-6).float().mean().item()

            current_step_layers.append({
                "idx": layer_idx,
                "name": layer_name,
                "relative_delta": rel_delta,
                "cosine_sim": cos_sim,
                "sparsity": sparsity,
            })

        return hook_fn

    # 注册 hooks
    hook_handles = []
    for name, module in block_modules:
        idx = int(name.split(".")[-1])
        h = module.register_forward_hook(make_hook(name, idx))
        hook_handles.append(h)

    # 推理循环
    ids = input_ids.clone()
    with torch.no_grad():
        for step in range(num_tokens):
            current_step_layers = []  # 重置缓冲

            outputs = model(ids)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # 计算 prediction entropy
            logits = outputs.logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum().item()

            ids = torch.cat([ids, next_token], dim=-1)

            all_steps.append({
                "token_idx": step,
                "entropy": entropy,
                "layers": list(current_step_layers),  # copy
            })

    # 移除 hooks
    for h in hook_handles:
        h.remove()

    return ids, all_steps


hooked_times = []
all_steps_data = None

for run in range(WARMUP + NUM_RUNS):
    t0 = time.perf_counter()
    output_ids_hooked, all_steps_data = generate_with_hooks(model, input_ids, NUM_TOKENS)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    if run >= WARMUP:
        hooked_times.append(elapsed)
        print(f"  Run {run - WARMUP + 1}: {elapsed * 1000:.1f} ms")

avg_hooked = sum(hooked_times) / len(hooked_times)
print(f"  Average: {avg_hooked * 1000:.1f} ms")


# ── 5. Overhead 分析 ────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Overhead Analysis")
print("=" * 60)

overhead_pct = (avg_hooked - avg_baseline) / avg_baseline * 100
print(f"  Baseline:   {avg_baseline * 1000:>8.1f} ms")
print(f"  With hooks: {avg_hooked * 1000:>8.1f} ms")
print(f"  Overhead:   {overhead_pct:>+8.1f}%")

if overhead_pct < 15:
    print("\n  ✅ PASS — Hook overhead < 15%, 方案可行")
elif overhead_pct < 30:
    print("\n  ⚠️  WARNING — Hook overhead 15~30%, 需要优化后可用")
else:
    print("\n  ❌ FAIL — Hook overhead > 30%, 需要重新评估方案")


# ── 6. 指标数据采样 ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Sample Metric Values (first 3 tokens)")
print("=" * 60)

for step in all_steps_data[:3]:
    print(f"\n  Token {step['token_idx']} (entropy={step['entropy']:.3f}):")
    print(f"    {'Layer':<6} {'RelDelta':>10} {'CosSim':>10} {'Sparsity':>10}")
    for layer in step["layers"]:
        print(
            f"    h.{layer['idx']:<4} "
            f"{layer['relative_delta']:>10.4f} "
            f"{layer['cosine_sim']:>10.4f} "
            f"{layer['sparsity']:>10.4f}"
        )


# ── 7. 指标全局范围 ─────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Metric Ranges (all tokens)")
print("=" * 60)

all_deltas = [l["relative_delta"] for s in all_steps_data for l in s["layers"]]
all_cosims = [l["cosine_sim"] for s in all_steps_data for l in s["layers"]]
all_spars  = [l["sparsity"] for s in all_steps_data for l in s["layers"]]
all_ent    = [s["entropy"] for s in all_steps_data]

print(f"  Relative Delta:  min={min(all_deltas):.4f}  max={max(all_deltas):.4f}  "
      f"mean={sum(all_deltas)/len(all_deltas):.4f}")
print(f"  Cosine Sim:      min={min(all_cosims):.4f}  max={max(all_cosims):.4f}  "
      f"mean={sum(all_cosims)/len(all_cosims):.4f}")
print(f"  Sparsity:        min={min(all_spars):.4f}  max={max(all_spars):.4f}  "
      f"mean={sum(all_spars)/len(all_spars):.4f}")
print(f"  Entropy:         min={min(all_ent):.4f}  max={max(all_ent):.4f}  "
      f"mean={sum(all_ent)/len(all_ent):.4f}")

# 关键判断：relative_delta 的范围决定了热力图色域
delta_range = max(all_deltas) - min(all_deltas)
print(f"\n  Relative Delta spread: {delta_range:.4f}")
if delta_range > 0.05:
    print("  ✅ 指标有足够的区分度，热力图能看出层间差异")
else:
    print("  ⚠️  指标区分度较低，热力图可能偏平，需要调整色域或指标")


# ── 8. JSON 数据包示例 ───────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: JSON Message Sample (token 0)")
print("=" * 60)

import json

sample_step = all_steps_data[0]
sample_msg = {
    "type": "step",
    "session_id": "phase0_test",
    "token_idx": sample_step["token_idx"],
    "generated_token": tokenizer.decode(
        output_ids_hooked[0, input_ids.shape[1] + sample_step["token_idx"]]
    ),
    "global": {
        "vram_total_mb": 0,  # CPU 上无显存
        "prediction_entropy": sample_step["entropy"],
    },
    "layers": sample_step["layers"],
}

msg_json = json.dumps(sample_msg, indent=2)
print(msg_json)
print(f"\nMessage size: {len(msg_json)} bytes")


# ── 总结 ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Hook overhead:      {overhead_pct:+.1f}%  (target: < 15%)")
print(f"  Relative Delta:     [{min(all_deltas):.4f}, {max(all_deltas):.4f}]")
print(f"  Cosine Similarity:  [{min(all_cosims):.4f}, {max(all_cosims):.4f}]")
print(f"  Data per step:      ~{len(json.dumps(sample_msg))} bytes")
print(f"  Verdict:            {'PASS' if overhead_pct < 15 else 'NEEDS WORK'}")
