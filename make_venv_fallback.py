# -*- coding: utf-8 -*-
"""生成一个**独立 venv** 的安装脚本，作为 cell 3 失败时的退路。

## 为什么需要退路

云上预装了 `transformers 5.15.1` + `peft 0.20.0`，
而 GPT-SoVITS 要求 `transformers>=4.51,<5` 且 `peft<0.18.0`。

在**主环境**里降级可能：
  · pip 解析失败（与预装包冲突）
  · 降级成功但破坏云上其它东西（我们不在乎，但环境可能被搞乱）
  · 权限问题（托管环境可能不让改主环境）

## 退路方案：venv + --system-site-packages

    python -m venv --system-site-packages /mnt/workspace/venv-gsv

这样：
  · **继承**主环境的 torch（ROCm 版，205GB 显存那个）—— 不重装、不破坏
  · venv 里装的 transformers 4.57 **优先于**主环境的 5.15.1
    （venv 的 site-packages 在 sys.path 中排前面）
  · 完全不碰主环境

用完后 `deactivate` 即可，或删掉 venv 目录。

产出：D:\\A\\zfy-v5\\setup_venv.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\A\zfy-v5\setup_venv.sh")

SCRIPT = r"""#!/usr/bin/env bash
# ============================================================
# GPT-SoVITS v5 全量微调 —— 独立 venv 环境（cell 3 失败时的退路）
#
# 特点：
#   · 用 --system-site-packages **继承**主环境的 ROCm torch
#     （不重装 torch，不会把 ROCm 版覆盖成 CPU/CUDA 版）
#   · 在 venv 内安装 transformers<5 与 peft<0.18
#     （venv 的 site-packages 优先于主环境，所以能覆盖 5.15.1）
#   · 完全不修改主环境
#
# 用法：
#   bash setup_venv.sh
#   source /mnt/workspace/venv-gsv/bin/activate
#   python -c "import transformers,peft;print(transformers.__version__,peft.__version__)"
# ============================================================
set -e

VENV=/mnt/workspace/venv-gsv
ROOT=/mnt/workspace/zfy-v5-fullft-pkg

echo "=== 1) 创建 venv（继承主环境 torch）==="
if [ -d "$VENV" ]; then
    echo "  已存在，复用: $VENV"
else
    python -m venv --system-site-packages "$VENV"
    echo "  已创建: $VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo
echo "=== 2) 确认 torch 来自主环境（ROCm 版）==="
python - <<'PY'
import torch
print(f"  torch     : {torch.__version__}")
print(f"  torch.hip : {getattr(torch.version, 'hip', None)}")
print(f"  设备可用  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  显存      : {p.total_memory/1e9:.1f} GB")
    print(f"  gcnArch   : {getattr(p, 'gcnArchName', '?')}")
PY

echo
echo "=== 3) 安装依赖（含 transformers<5 / peft<0.18）==="
# 先单独装必须降级的两个，避免解析器纠缠
pip install --quiet "transformers>=4.51,<5" "peft<0.18.0"

# 再装其余缺失依赖
pip install --quiet \
    "librosa==0.10.2" \
    "pytorch-lightning>=2.4" \
    "torchmetrics<=1.5" \
    x_transformers \
    rotary_embedding_torch \
    soundfile \
    ffmpeg-python \
    jieba \
    jieba_fast \
    pypinyin \
    cn2an \
    chardet \
    tensorboard \
    einops \
    safetensors \
    tqdm \
    PyYAML \
    psutil \
    wordsegment \
    split-lang \
    "fast_langdetect>=0.3.1" \
    opencc

echo
echo "=== 4) 复核版本 ==="
python - <<'PY'
import importlib
import transformers, peft, torch

tv = transformers.__version__
pv = peft.__version__
print(f"  transformers : {tv}   {'OK' if tv.startswith('4.') else '!! 仍需 <5'}")
maj, mi = (int(x) for x in pv.split('.')[:2])
print(f"  peft         : {pv}   {'OK' if (maj, mi) < (0, 18) else '!! 仍需 <0.18'}")
print(f"  torch        : {torch.__version__}  (继承自主环境)")

miss = []
for m in ["librosa", "pytorch_lightning", "torchmetrics", "x_transformers",
          "rotary_embedding_torch", "soundfile", "ffmpeg", "jieba",
          "pypinyin", "cn2an", "chardet", "tensorboard", "einops",
          "safetensors", "numpy", "scipy"]:
    try:
        importlib.import_module(m)
    except Exception:
        miss.append(m)
print(f"  缺失         : {miss if miss else '无'}")
PY

echo
echo "=== 5) 自检（import v5 模型）==="
cd "$ROOT"
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
cfg = json.load(open(f"{root}/GPT_SoVITS/configs/s2.json", encoding="utf-8"))
kw = dict(cfg["model"]); kw["version"] = "v5dev"
net = SynthesizerTrnV3(cfg["data"]["filter_length"] // 2 + 1,
                       cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
                       n_speakers=cfg["data"]["n_speakers"], **kw)
n = sum(p.numel() for p in net.parameters())
print(f"  v5 模型构造成功: {n/1e6:.2f} M 参数")
PY

echo
echo "============================================================"
echo "环境就绪。后续所有训练命令请先激活："
echo "    source $VENV/bin/activate"
echo "============================================================"
"""

OUT.write_text(SCRIPT, encoding="utf-8", newline="\n")
print(f"已生成: {OUT}")
print(f"  体积: {OUT.stat().st_size/1024:.1f} KB")
print()
print("用法（云上）：")
print("  cd /mnt/workspace/zfy-v5-fullft-pkg")
print("  bash _tools/setup_venv.sh")
print("  source /mnt/workspace/venv-gsv/bin/activate")
