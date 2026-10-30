#!/bin/bash
# Stage C NVIDIA side (RTX 5090): reference (fp32, TF32 off) and tolerance baseline (bf16)
# on exactly the token ids recorded on C500.
set -u
cd /data1/$USER/stage_c || exit 1
export HF_HUB_DISABLE_XET=1 HF_HOME=/data1/$USER/hf HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=/data1/$USER/univis-venv/bin/python
strict_fp32() {
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 $PY -c "import sys, torch; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; sys.argv = ['univis'] + sys.argv[1:]; from univis.__main__ import main; main()" "$@"
}
{ date -u; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv -i ${GPU:-0}
  $PY -c "import torch, transformers, univis; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('transformers', transformers.__version__); print('univis', univis.__version__); print('device', torch.cuda.get_device_name(0)); print('allow_tf32', torch.backends.cuda.matmul.allow_tf32)"
} > nvidia-env-fingerprint.txt 2>&1
for s in 0.5B 3B 7B; do
  m=Qwen/Qwen2.5-$s-Instruct
  echo "== $s fp32 $(date -u +%T)"
  [ -f nv_fp32_$s.json ] || strict_fp32 diff record $m --dtype fp32 --device cuda --tokens-from c500_bf16_$s.json -o nv_fp32_$s.json --label "NVIDIA RTX 5090 fp32 (TF32 off)"
  echo "== $s bf16 $(date -u +%T)"
  [ -f nv_bf16_$s.json ] || $PY -m univis diff record $m --dtype bf16 --device cuda --tokens-from c500_bf16_$s.json -o nv_bf16_$s.json --label "NVIDIA RTX 5090 bf16"
  if [ -f c500_bf16long_sdpa_$s.json ] && [ ! -f nv_bf16long_$s.json ]; then
    echo "== $s long fp32 / bf16 $(date -u +%T)"
    strict_fp32 diff record $m --dtype fp32 --device cuda --tokens-from c500_bf16long_sdpa_$s.json -o nv_fp32long_$s.json --label "NVIDIA RTX 5090 fp32 (TF32 off) long"
    $PY -m univis diff record $m --dtype bf16 --device cuda --tokens-from c500_bf16long_sdpa_$s.json -o nv_bf16long_$s.json --label "NVIDIA RTX 5090 bf16 long"
  fi
done
echo "== nvidia done $(date -u +%T)"; ls -la
