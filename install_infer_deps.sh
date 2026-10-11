#!/usr/bin/env bash
# ============================================================
# 补装**推理**所需依赖（云端跑合成用）
#
# ## 为什么之前没装
#
# 这份清单最初是只为**训练**裁剪的，`s2_train_v5_full.py` 不需要这些。
# 但推理入口 `TTS_infer_pack/TTS.py` 的模块级 import 需要：
#
#     L29  AR.models.t2s_lightning_module -> pytorch_lightning + torchmetrics
#     L35  peft                           -> peft
#     L39  tools.audio_sr.AP_BWE          -> datasets1/models（tools/AP_BWE_main 本地目录，已随包）
#     L52  resampy                        -> resampy（函数内 import）
#
# ## ⚠️ numpy 陷阱
#
# 之前把 numpy 从 2.3.5 降到 1.x 的元凶是 `torchmetrics<=1.5`
# （requirements.txt 里的钉法）。**这次不钉它**，让 pip 选支持 numpy 2 的版本。
#
# 用法：
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   bash _tools/install_infer_deps.sh
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT" 2>/dev/null || true

echo "============================================================"
echo "0) 装前现状"
echo "============================================================"
python - <<'PYEOF'
for m in ("torch", "torchaudio", "numpy", "peft", "pytorch_lightning",
          "torchmetrics", "resampy", "transformers"):
    try:
        mod = __import__(m)
        print(f"  {m:20s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:20s} 缺失 ({type(e).__name__})")
PYEOF

echo
echo "============================================================"
echo "1) 先钉住 numpy（防止被依赖拖低导致 ABI 崩）"
echo "============================================================"
pip install --quiet --upgrade "numpy>=2,<2.4"
echo "  numpy 已钉"

echo
echo "============================================================"
echo "2) 装推理依赖（不钉 torchmetrics！）"
echo "============================================================"
# torchaudio 必须与 torch 同版本；若已有就跳过
python -c "import torchaudio" 2>/dev/null || pip install --quiet torchaudio

pip install --quiet pytorch-lightning        # 会带 torchmetrics，但不钉版本
pip install --quiet peft
pip install --quiet resampy

echo
echo "============================================================"
echo "3) 复核（独立进程，检查 numpy ABI 是否仍正常）"
echo "============================================================"
python - <<'PYEOF'
import subprocess, sys
code = r"""
import numpy, scipy, soundfile, librosa, pandas
print(f"  numpy      {numpy.__version__}")
print(f"  scipy      {scipy.__version__}")
import torch
print(f"  torch      {torch.__version__}  hip={getattr(torch.version,'hip',None)}")
print(f"  设备可用   {torch.cuda.is_available()}")
for m in ("torchaudio", "peft", "pytorch_lightning", "torchmetrics", "resampy"):
    try:
        mod = __import__(m)
        print(f"  {m:18s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:18s} 缺失 {type(e).__name__}")
print("  OK: ABI 正常")
"""
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("  !! 有问题：")
    print(r.stderr[-2000:])
    sys.exit(1)
PYEOF

echo
echo "============================================================"
echo "4) 冒烟：能否 import TTS_infer_pack（真正验证 BigVGAN/sv/peft 都通）"
echo "============================================================"
PYTHONPATH="$ROOT" python - <<'PYEOF'
import os, sys
root = os.environ.get("PYTHONPATH", "/mnt/workspace/zfy-v5-fullft-pkg")
for p in reversed([root, root + "/GPT_SoVITS", root + "/GPT_SoVITS/BigVGAN",
                   root + "/tools", root + "/tools/asr", root + "/tools/uvr5"]):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.chdir(root)

try:
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    print("  ✓ TTS_infer_pack 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {type(e).__name__}: {str(e)[:300]}")
    raise SystemExit(1)

# 确认 BigVGAN / sv 都能解析
import inspect
for mod, name in ((__import__("BigVGAN.bigvgan", fromlist=["BigVGAN"]), "BigVGAN"),
                  (__import__("sv", fromlist=["SV"]), "sv"),
                  (__import__("feature_extractor.cnhubert", fromlist=["CNHubert"]),
                   "feature_extractor")):
    f = getattr(mod, "__file__", "?")
    print(f"  ✓ {name:20s} {f}")
PYEOF

echo
echo "============================================================"
echo "完成。若第 4 步通过，就可以跑合成："
echo "    python _tools/cloud_synth_sweep.py"
echo "============================================================"
