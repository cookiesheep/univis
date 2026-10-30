#!/bin/bash
# Stage C on MetaX C500: record univis diff fingerprints. Never reinstalls torch.
set -u
export MACA_PATH=${MACA_PATH:-/opt/maca} HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/hf
PY=${PY:-/opt/conda/bin/python}
[ -x "$PY" ] || PY=python
OUT=/data/univis_stage_c; rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT" || exit 1

{ date -u; mx-smi
  $PY -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__); print('device', torch.cuda.get_device_name(0)); print('allow_tf32', torch.backends.cuda.matmul.allow_tf32)"
  env | grep -i -E "tf32|maca" | grep -v -i -E "token|key|pass"
  $PY -m pip list 2>/dev/null | grep -i -E "torch|maca|metax|transformers|numpy|triton|flag"
} > env-fingerprint.txt 2>&1

# the instance's pip mirror may lag behind PyPI: try it, then pypi.org, then an uploaded wheel
WHL=$(ls /data/univis-0.7.1-py3-none-any.whl 2>/dev/null)
{ [ -n "$WHL" ] && $PY -m pip install --no-deps -q "$WHL"; } \
  || $PY -m pip install --no-deps -q univis==0.7.1 \
  || $PY -m pip install --no-deps -q -i https://pypi.org/simple univis==0.7.1 \
  || $PY -m pip install --no-deps -q -i https://pypi.tuna.tsinghua.edu.cn/simple univis==0.7.1 \
  || $PY -m pip install --no-deps -q -i https://mirrors.aliyun.com/pypi/simple univis==0.7.1
$PY -c "import sys, univis; print('univis', univis.__version__); sys.exit(univis.__version__ < '0.7')" | tee -a env-fingerprint.txt
[ "${PIPESTATUS[0]}" = 0 ] || { echo "univis 0.7.x not installed, stopping"; exit 1; }

# fp32 strictly without TF32 (this container defaults to TF32 for fp32 matmuls)
strict_fp32() {
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 $PY -c "import sys, torch; torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; sys.argv = ['univis'] + sys.argv[1:]; from univis.__main__ import main; main()" "$@"
}

$PY - > probe.txt 2>&1 <<'PROBE'
import torch
e = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
a = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
e[0].record(); b = a @ a; e[1].record(); torch.cuda.synchronize()
print('cuda_event_ms', round(e[0].elapsed_time(e[1]), 3))
try:
    import flag_gems
    print('flag_gems', getattr(flag_gems, '__version__', 'present'))
except Exception as exc:
    print('flag_gems unavailable:', type(exc).__name__)
PROBE

for s in 0.5B 3B 7B; do
  m=Qwen/Qwen2.5-$s-Instruct
  echo "== $s bf16 $(date -u +%T)"
  $PY -m univis diff record $m --dtype bf16 --device cuda -o c500_bf16_$s.json --label "MetaX C500 bf16"
  if [ "$s" != "7B" ]; then
    echo "== $s fp32 (container default, TF32 on) $(date -u +%T)"
    $PY -m univis diff record $m --dtype fp32 --device cuda --tokens-from c500_bf16_$s.json \
        -o c500_fp32tf32_$s.json --label "MetaX C500 fp32 (TF32 on, container default)"
    echo "== $s fp32 strict (TF32 off) $(date -u +%T)"
    strict_fp32 diff record $m --dtype fp32 --device cuda --tokens-from c500_bf16_$s.json \
        -o c500_fp32strict_$s.json --label "MetaX C500 fp32 (TF32 off)"
  fi
done
echo "== done $(date -u +%T)"; ls -la
tar czf "$HOME/univis_stage_c.tgz" -C /data univis_stage_c && cp "$HOME/univis_stage_c.tgz" /data/ && echo "== packed: $HOME/univis_stage_c.tgz"
