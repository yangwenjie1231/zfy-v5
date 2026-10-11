# -*- coding: utf-8 -*-
"""Notebook 环境自检（在**下载 2.77GB 包之前**就能发现问题）。

用法（Notebook 里）：
    python notebook_preflight.py
或直接粘贴到 cell 里运行。

逐项检查：
  1. torch / ROCm / Python 版本
  2. **关键 API 是否存在**（torch.cuda.amp、cudnn、allow_tf32 …）
  3. 依赖是否齐备
  4. 显存与设备
  5. 能否 import GPT-SoVITS 的模型定义（最关键的集成测试）
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

# Windows 控制台默认 GBK，输出 ✓/✗ 之类字符会 UnicodeEncodeError 中断脚本
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

OK = "OK  "
BAD = "FAIL"
WARN = "WARN"


def hdr(t: str):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


hdr("1. 基础环境")
print(f"  Python      : {sys.version.split()[0]}")
print(f"  平台        : {platform.system()} {platform.release()}")
try:
    import torch
    print(f"  torch       : {torch.__version__}")
    print(f"  torch.hip   : {getattr(torch.version, 'hip', None)}")
    print(f"  torch.cuda  : {getattr(torch.version, 'cuda', None)}")
    print(f"  可用设备    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  设备名      : {torch.cuda.get_device_name(0)}")
        p = torch.cuda.get_device_properties(0)
        print(f"  显存        : {p.total_memory/1e9:.1f} GB")
        print(f"  gcnArchName : {getattr(p, 'gcnArchName', '?')}")
except Exception as e:  # noqa: BLE001
    print(f"  {BAD} torch 导入失败: {type(e).__name__}: {e}")
    sys.exit(1)

# ---- ffmpeg 可执行文件（音频加载的硬依赖）----
# load_audio() 用 ffmpeg **命令行程序**解码（tools/my_utils.py:24-28）。
# 只装 `ffmpeg-python` 这个 Python 包装库是不够的 —— 缺了二进制，
# 所有音频加载都失败，而 data_utils.py 的 except 会把异常吞掉并返回
# 全零张量，训练"成功"但学的是垃圾。
print()
print("  --- ffmpeg 可执行文件（音频加载硬依赖）---")
import shutil as _sh
_ff = _sh.which("ffmpeg")
if _ff:
    print(f"  {OK} {_ff}")
    try:
        import subprocess as _sp
        _v = _sp.run(["ffmpeg", "-version"], capture_output=True, text=True,
                     timeout=30)
        print(f"       {_v.stdout.splitlines()[0][:70]}")
    except Exception as _e:  # noqa: BLE001
        print(f"  {WARN} 执行 ffmpeg 失败: {type(_e).__name__}")
else:
    print(f"  {BAD} 未找到 ffmpeg 可执行文件")
    print("       -> 训练会报大量 'load audio or ssl error'，")
    print("          且训练看起来成功、实际学的是全零数据。")
    print("       解决: bash _tools/install_ffmpeg.sh")
    print("         或: apt-get install -y ffmpeg")
    print("         或: pip install imageio-ffmpeg")

hdr("2. 关键 API 是否存在（torch 2.12 可能已移除）")
checks = [
    ("torch.cuda.amp.GradScaler", "from torch.cuda.amp import GradScaler"),
    ("torch.cuda.amp.autocast", "from torch.cuda.amp import autocast"),
    ("torch.amp.GradScaler", "from torch.amp import GradScaler"),
    ("torch.amp.autocast", "from torch.amp import autocast"),
    ("torch.backends.cuda.matmul.allow_tf32",
     "x = torch.backends.cuda.matmul.allow_tf32"),
    ("torch.backends.cudnn.benchmark", "x = torch.backends.cudnn.benchmark"),
    ("torch.set_float32_matmul_precision",
     "torch.set_float32_matmul_precision('medium')"),
    ("torch.cuda.set_device", "torch.cuda.set_device(0)"),
]
api_result = {}
for name, code in checks:
    try:
        exec(code, {"torch": torch})
        print(f"  {OK} {name}")
        api_result[name] = True
    except Exception as e:  # noqa: BLE001
        print(f"  {BAD} {name}  -> {type(e).__name__}: {str(e)[:60]}")
        api_result[name] = False

hdr("3. 依赖")
DEPS = ["numpy", "scipy", "librosa", "numba", "transformers", "peft",
        "pytorch_lightning", "torchmetrics", "einops", "safetensors",
        "x_transformers", "rotary_embedding_torch", "soundfile",
        "ffmpeg", "yaml", "psutil", "jieba", "pypinyin", "cn2an",
        "chardet", "tensorboard"]
missing = []
for m in DEPS:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        print(f"  {OK} {m:24s} {v}")
    except Exception as e:  # noqa: BLE001
        print(f"  {BAD} {m:24s} {type(e).__name__}")
        missing.append(m)
print(f"\n  缺失: {missing if missing else '无'}")

hdr("4. numpy 版本（requirements 钉 <2.0，但可能过时）")
try:
    import numpy as np
    print(f"  numpy {np.__version__}")
    if np.__version__.startswith("1."):
        print(f"  {WARN} 是 1.x。本机经验：1.26.4 曾报 "
              f"OverflowError: cannot convert longdouble infinity to integer")
        print(f"       若训练报此错，升级到 numpy>=2")
    else:
        print(f"  {OK} 2.x，与本机实测一致")
except Exception as e:  # noqa: BLE001
    print(f"  {BAD} numpy 导入失败: {e}")

hdr("5. 集成测试：import GPT-SoVITS 模型定义")
print("  （这一步最关键 —— 它会同时暴露 f5_tts 解析和 torch API 两个问题）")


def find_gsv_root() -> str | None:
    """自动定位 GPT-SoVITS **检出根目录**。

    注意目录层级（我手册里曾写错，这里做成自动识别）：
        仓库根/                      <- 这就是检出根（含 config.py）
        ├─ GPT_SoVITS/              <- Python 包（module/ f5_tts/ configs/）
        ├─ config.py
        └─ tools/

    所以检出根 = 含 config.py 且含 GPT_SoVITS/ 子目录的那一层。
    """
    cands = []
    env = os.environ.get("GSV_ROOT")
    if env:
        cands.append(env)
    # 从 cwd 向上找
    here = Path(os.getcwd()).resolve()
    for up in [here, *here.parents]:
        cands.append(str(up))
    # 常见位置
    cands += [
        "/mnt/workspace/zfy-v5-fullft-pkg",
        "/mnt/workspace/zfy-v5",
        "/mnt/workspace",
    ]
    for c in cands:
        p = Path(c)
        if (p / "config.py").exists() and (p / "GPT_SoVITS").is_dir():
            return str(p)
    return None


GSV = find_gsv_root()
if GSV is None:
    print(f"  {WARN} 找不到检出根（需含 config.py + GPT_SoVITS/）")
    print(f"       当前 cwd: {os.getcwd()}")
    print(f"       解决: export GSV_ROOT=/mnt/workspace/zfy-v5-fullft-pkg")
else:
    print(f"  检出根: {GSV}")
    for p in reversed([GSV, os.path.join(GSV, "GPT_SoVITS"),
                       os.path.join(GSV, "GPT_SoVITS", "BigVGAN"),
                       os.path.join(GSV, "tools"),
                       os.path.join(GSV, "tools", "asr"),
                       os.path.join(GSV, "tools", "uvr5")]):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    os.chdir(GSV)

    try:
        import inspect
        from f5_tts.model import DiT
        sig = list(inspect.signature(DiT.__init__).parameters)
        has = "use_step_embedding" in sig
        print(f"  {'OK  ' if has else BAD} f5_tts.DiT -> {inspect.getfile(DiT)}")
        print(f"       含 use_step_embedding: {has}")
        if not has:
            print("       **解析到旧版 DiT**：检查 sys.path 顺序 / "
                  "是否有别的检出被注入")
    except Exception as e:  # noqa: BLE001
        print(f"  {BAD} f5_tts.DiT 导入失败: {type(e).__name__}: {e}")

    try:
        from module.models import SynthesizerTrnV3
        print(f"  {OK} module.models.SynthesizerTrnV3")
    except Exception as e:  # noqa: BLE001
        print(f"  {BAD} models 导入失败: {type(e).__name__}: {str(e)[:120]}")

    try:
        import torch as _t
        from module.models import SynthesizerTrnV3 as _S
        import json
        cfg = json.load(open(os.path.join(GSV, "GPT_SoVITS", "configs",
                                          "s2.json"), encoding="utf-8"))
        kw = dict(cfg["model"])
        kw["version"] = "v5dev"
        net = _S(cfg["data"]["filter_length"] // 2 + 1,
                 cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
                 n_speakers=cfg["data"]["n_speakers"], **kw)
        n = sum(p.numel() for p in net.parameters())
        print(f"  {OK} 构造 v5 SynthesizerTrnV3 成功，参数 {n/1e6:.2f} M")
        del net
    except Exception as e:  # noqa: BLE001
        print(f"  {BAD} v5 模型构造失败: {type(e).__name__}: {str(e)[:150]}")

hdr("6. 结论与建议")
blockers = []
if not api_result.get("torch.cuda.amp.autocast"):
    blockers.append("torch.cuda.amp 已被移除 -> 需给 models.py 打补丁")
if missing:
    blockers.append(f"缺依赖: {missing}")
if blockers:
    print("  发现问题:")
    for b in blockers:
        print(f"    - {b}")
else:
    print("  ✓ 未发现阻塞问题，可以开始下载包并训练")
print()
print("  若 torch.cuda.amp 缺失，运行我提供的补丁脚本:")
print("    python patch_models_for_torch212.py")
