# -*- coding: utf-8 -*-
"""生成**最小且安全**的依赖安装脚本（修 numpy ABI 冲突）。

## 事故复盘

我上一版 PKGS 清单有 bug：装了 `torchmetrics<=1.5` 等包，
把云上原本的 **numpy 2.3.5 降到了 1.x**，导致 scipy/soundfile 等
按 numpy2 编译的 C 扩展报：

    ValueError: numpy.dtype size changed, Expected 96 ... got 88

（96 = numpy2 的 dtype 结构，88 = numpy1 —— 方向明确）

## 本版改进

1. **显式钉住 numpy>=2**，并放在安装命令**最后**再确认一次
2. **去掉非必需包**：
   · `peft`      —— 全量微调路径**完全不用**（只有 LoRA 用）
     已验证：s2_train_v5_full.py 里 peft 那行已注释，
     module/ 下也无任何 `import peft`
   · `pytorch-lightning` / `torchmetrics` —— 只有 `s1_train.py`（GPT 训练）用，
     我们**不训 GPT**（v5 与 v2Pro 共用 s1v3.ckpt）
3. 补上真正被 import 的 `matplotlib` / `pandas` / `gradio`
   （`tools/my_utils.py` 顶层 import 了 gradio 和 pandas）
4. 分两批装：先钉 numpy，再装其余，避免解析器回退

产出：D:\\A\\zfy-v5\\install_deps_minimal.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\A\zfy-v5\install_deps_minimal.sh")

SCRIPT = r"""#!/usr/bin/env bash
# ============================================================
# GPT-SoVITS v5 全量微调 —— 最小依赖安装（修 numpy ABI 冲突）
#
# 与上一版的区别：
#   · 显式钉 numpy>=2（上一版被 torchmetrics<=1.5 拉低到 1.x）
#   · 去掉 peft（全量微调不用）与 pytorch-lightning/torchmetrics
#     （只有 s1_train 即 GPT 训练用，我们不训 GPT）
#   · 补上 tools/my_utils.py 真正 import 的 matplotlib/pandas/gradio
#
# 用法：
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   bash _tools/install_deps_minimal.sh
# ============================================================
set -e

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT"

echo "=== 0) 现状 ==="
python - <<'PY'
try:
    import numpy
    print(f"  numpy: {numpy.__version__}")
    print("  " + ("!! 是 1.x，需要升到 2.x" if numpy.__version__.startswith("1.")
                  else "OK 已是 2.x"))
except Exception as e:
    print(f"  numpy 导入失败: {e}")
PY

echo
echo "=== 1) 先把 numpy 钉回 2.x ==="
# 上界 2.4：numba/llvmlite 对 numpy 有已知上界要求
pip install --quiet --upgrade "numpy>=2,<2.4"

echo
echo "=== 2) 装最小必需依赖 ==="
# ---- 关键：与 GPT-SoVITS 版本约束一致 ----
pip install --quiet "transformers>=4.51,<5"

# ---- 音频 / 训练 ----
pip install --quiet \
    "librosa==0.10.2" \
    soundfile \
    ffmpeg-python \
    "scipy>=1.11"

# ---- 模型定义需要（f5_tts 的 DiT）----
pip install --quiet x_transformers rotary_embedding_torch einops safetensors

# ---- tools/my_utils.py 顶层 import 了 gradio 与 pandas ----
pip install --quiet gradio pandas matplotlib

# ---- 训练循环 ----
pip install --quiet tensorboard tqdm PyYAML psutil

# ---- 中文文本前端 ----
pip install --quiet cn2an pypinyin jieba jieba_fast chardet opencc

# ---- 注意：以下**不装**
#   peft                全量微调不用（已验证 s2_train_v5_full.py 无 import）
#   pytorch-lightning   只有 s1_train 用（我们不训 GPT）
#   torchmetrics        同上
#   onnxruntime-gpu     ROCm 上无效，训练也不需要

echo
echo "=== 3) 复核（**在独立进程里**验证 ABI）==="
python - <<'PY'
import subprocess, sys

code = r'''
import numpy, scipy, soundfile, librosa, pandas
print(f"  numpy     {numpy.__version__}")
print(f"  scipy     {scipy.__version__}")
print(f"  soundfile {soundfile.__version__}")
print(f"  librosa   {librosa.__version__}")
print(f"  pandas    {pandas.__version__}")
import torch
print(f"  torch     {torch.__version__}  hip={getattr(torch.version,'hip',None)}")
print(f"  设备可用  {torch.cuda.is_available()}")
print("  OK: C 扩展 ABI 一致")
'''
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("  !! 仍有 ABI 问题：")
    print(r.stderr[-2500:])
    sys.exit(1)
PY

echo
echo "=== 4) 检查版本约束 ==="
python - <<'PY'
import transformers
tv = transformers.__version__
print(f"  transformers {tv}  {'OK' if tv.startswith('4.') else '!! 需要 <5'}")
try:
    import peft
    print(f"  peft         {peft.__version__}  (未装也行，全量微调不需要)")
except ImportError:
    print("  peft         未安装  (正确，全量微调不需要)")
PY

echo
echo "=== 5) 自检：能否 import v5 模型 ==="
PYTHONPATH="$ROOT" python - <<'PY'
import os, sys, json
root = os.environ.get("PYTHONPATH", "/mnt/workspace/zfy-v5-fullft-pkg")
for p in reversed([root, f"{root}/GPT_SoVITS", f"{root}/GPT_SoVITS/BigVGAN",
                   f"{root}/tools", f"{root}/tools/asr", f"{root}/tools/uvr5"]):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.chdir(root)

import inspect
from f5_tts.model import DiT
ok = "use_step_embedding" in inspect.signature(DiT.__init__).parameters
print(f"  f5_tts.DiT 含 use_step_embedding: {ok}")

from module.models import SynthesizerTrnV3
from module.models_v5 import CFMV5
cfg = json.load(open(f"{root}/GPT_SoVITS/configs/s2.json", encoding="utf-8"))
kw = dict(cfg["model"]); kw["version"] = "v5dev"
net = SynthesizerTrnV3(cfg["data"]["filter_length"] // 2 + 1,
                       cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
                       n_speakers=cfg["data"]["n_speakers"], **kw)
n = sum(p.numel() for p in net.parameters())
print(f"  v5 模型构造成功: {n/1e6:.2f} M 参数  (CFMV5={isinstance(net.cfm, CFMV5)})")
PY

echo
echo "============================================================"
echo "完成。**请重启 Notebook 内核**，再跑后续 cell。"
echo "（当前内核里已加载的旧 numpy 不会自动更新）"
echo "============================================================"
"""

OUT.write_text(SCRIPT, encoding="utf-8", newline="\n")
print(f"已生成: {OUT}")
print(f"  体积: {OUT.stat().st_size/1024:.1f} KB")
print()
print("云上用法：")
print("  cd /mnt/workspace/zfy-v5-fullft-pkg")
print("  bash _tools/install_deps_minimal.sh")
print("  # 然后重启内核")
