#!/usr/bin/env bash
# ============================================================
# 一次装全三条链路的依赖（GPT 训练 / SoVITS 训练 / 推理）
#
# ## 为什么要有这个脚本
#
# 我先前把依赖清单**按用途拆成三份**，结果每次换用途就漏包：
#   · install_deps_v3.sh      只为 SoVITS 训练裁剪 -> 漏 peft/pytorch_lightning
#   · install_infer_deps.sh   只为推理补装       -> 漏 matplotlib/gradio
# 而 `matplotlib` 是 **GPT 训练的必经路径**：
#     s1_train.py -> AR/models/t2s_lightning_module.py
#                 -> AR/modules/lr_schedulers.py:6  from matplotlib import pyplot
# 一旦缺失，GPT 训练直接 ModuleNotFoundError。
#
# 所以改成**一份覆盖全部**，并且逐条链路 import 真实入口来验证。
#
# ## ⚠️ numpy 陷阱（踩过两次）
# 元凶是 `torchmetrics<=1.5`（requirements.txt 的钉法）会把 numpy 降到 1.x，
# 导致 scipy/soundfile 报 "dtype size changed, Expected 96 got 88"。
# 本脚本**先钉 numpy>=2,<2.4**，且**不钉 torchmetrics**。
#
# ## 用法
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   bash _tools/install_all_deps.sh
#   bash _tools/install_all_deps.sh --check-only    # 只体检不安装
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT" 2>/dev/null || true

CHECK_ONLY=0
[ "${1:-}" = "--check-only" ] && CHECK_ONLY=1

echo "============================================================"
echo "0) 装前现状"
echo "============================================================"
python - <<'PYEOF'
import importlib
GROUPS = {
    "基础": ["torch", "torchaudio", "numpy", "scipy"],
    "SoVITS 训练": ["librosa", "soundfile", "ffmpeg", "x_transformers",
                    "rotary_embedding_torch", "einops", "safetensors",
                    "gradio", "pandas", "matplotlib", "tensorboard"],
    "GPT 训练": ["pytorch_lightning", "torchmetrics", "matplotlib",
                 "pandas", "yaml"],
    "推理": ["peft", "resampy", "transformers", "huggingface_hub"],
    "文本前端": ["cn2an", "pypinyin", "jieba", "chardet", "opencc"],
}
for g, mods in GROUPS.items():
    print(f"  --- {g} ---")
    for m in mods:
        try:
            mod = importlib.import_module(m)
            print(f"    OK   {m:24s} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"    FAIL {m:24s} {type(e).__name__}")
PYEOF

if [ "$CHECK_ONLY" = "1" ]; then
    echo
    echo "(--check-only，不安装)"
    exit 0
fi

echo
echo "============================================================"
echo "1) 先钉住 numpy（防止被依赖拖低导致 ABI 崩）"
echo "============================================================"
pip install --quiet --upgrade "numpy>=2,<2.4"
python -c "import numpy; print('  numpy', numpy.__version__)"

echo
echo "============================================================"
echo "1b) 钉住 huggingface_hub（gradio 需要已移除的 HfFolder）"
echo "============================================================"
# 实测报错：
#   ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
# 来源：gradio/oauth.py:13  from huggingface_hub import HfFolder, whoami
# HfFolder 在 huggingface_hub 0.36 之后被移除，而本机可用版本是 0.36.2。
# 所以这里必须钉住，不能让它自由升级。
pip install --quiet "huggingface-hub==0.36.2"
python -c "import huggingface_hub as h; print('  huggingface_hub', h.__version__, ' 有HfFolder:', hasattr(h,'HfFolder'))"

echo
echo "============================================================"
echo "2) GPT 训练依赖"
echo "============================================================"
echo "  · pytorch_lightning  <- AR/data/data_module.py, t2s_lightning_module.py"
echo "  · torchmetrics       <- AR/models/t2s_model.py（随 PL 一起装，**不钉版本**）"
echo "  · matplotlib         <- AR/modules/lr_schedulers.py:6（模块级，缺了直接崩）"
echo "  · pandas / yaml      <- AR/data/dataset.py, AR/utils/io.py"
# 关键：**不要** pip install "torchmetrics<=1.5" —— 那是 numpy 被降级的元凶
pip install --quiet pytorch-lightning
pip install --quiet matplotlib
pip install --quiet pandas pyyaml

echo
echo "============================================================"
echo "3) SoVITS 训练依赖"
echo "============================================================"
echo "  · gradio<5   <- tools/my_utils.py:7（模块级；data_utils 会 import 它）"
echo "  · 其余音频/文本前端"
pip install --quiet "gradio<5"
pip install --quiet librosa soundfile ffmpeg-python
pip install --quiet x_transformers rotary_embedding_torch einops safetensors
pip install --quiet tensorboard tqdm psutil packaging
pip install --quiet cn2an pypinyin jieba chardet opencc
# 上面几个可能把 huggingface_hub 顶上去，再钉一次（gradio 需要 HfFolder）
pip install --quiet "huggingface-hub==0.36.2"

echo
echo "============================================================"
echo "4) 推理依赖"
echo "============================================================"
echo "  · peft    <- TTS_infer_pack/TTS.py:35（模块级）"
echo "  · resampy <- TTS.py:52（函数内）"
echo "  · onnxruntime <- text/g2pw/onnx_api.py:11（模块级，中文多音字消歧）"
echo "    注意：onnxruntime 1.30 是按 CUDA 13 编译的，ROCm/CPU 环境用 CPU 版即可。"
echo "    G2PW 只是**增强项**（缺了会降级到 jieba 拼音），不装也不影响合成。"
pip install --quiet peft resampy
# onnxruntime：优先 CPU 版（云上是 ROCm，CUDA EP 用不上）
pip install --quiet onnxruntime || echo "  onnxruntime 安装失败（G2PW 会自动降级，不阻塞）"
python -c "import torchaudio" 2>/dev/null || pip install --quiet torchaudio

echo
echo "============================================================"
echo "5) 复核（独立进程 + 真实 import 入口模块）"
echo "============================================================"
PYTHONPATH="$ROOT" python - <<'PYEOF'
import os, sys, subprocess
root = os.environ.get("PYTHONPATH", "/mnt/workspace/zfy-v5-fullft-pkg")

code = r'''
import os, sys, json
root = os.environ.get("PYTHONPATH", "/mnt/workspace/zfy-v5-fullft-pkg")
for p in reversed([root, root + "/GPT_SoVITS", root + "/GPT_SoVITS/BigVGAN",
                   root + "/tools", root + "/tools/asr", root + "/tools/uvr5"]):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.chdir(root)

import numpy, torch
print(f"  numpy {numpy.__version__}   torch {torch.__version__} "
      f"hip={getattr(torch.version,'hip',None)}  设备={torch.cuda.is_available()}")

fails = []

# ---- 链路 1：GPT 训练 ----
try:
    from AR.models.t2s_lightning_module import Text2SemanticLightningModule
    from AR.data.data_module import Text2SemanticDataModule
    from AR.modules.lr_schedulers import WarmupCosineLRSchedule
    print("  OK   GPT 训练链路（t2s_lightning_module + data_module + lr_schedulers）")
except Exception as e:
    print(f"  FAIL GPT 训练链路: {type(e).__name__}: {str(e)[:120]}")
    fails.append("gpt")

# ---- 链路 2：SoVITS 训练 ----
try:
    from module.data_utils import TextAudioSpeakerLoaderV4
    print("  OK   SoVITS 训练链路（TextAudioSpeakerLoaderV4）")
except Exception as e:
    print(f"  FAIL SoVITS 训练链路: {type(e).__name__}: {str(e)[:120]}")
    fails.append("sovits")

# ---- 链路 3：推理 ----
try:
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
    print("  OK   推理链路（TTS_infer_pack）")
except Exception as e:
    print(f"  FAIL 推理链路: {type(e).__name__}: {str(e)[:120]}")
    fails.append("infer")

if fails:
    raise SystemExit("缺: " + ",".join(fails))
print("  三条链路全部可 import")
'''
r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                   text=True, env={**os.environ, "PYTHONPATH": root})
print(r.stdout)
if r.returncode != 0:
    print("  !! 有问题：")
    print(r.stderr[-2500:])
    sys.exit(1)
PYEOF

echo
echo "============================================================"
echo "完成。三条链路都可 import 后即可："
echo "  SoVITS 全量微调 : python -s GPT_SoVITS/s2_train_v5_full.py --config tmp_full_e32.json"
echo "  GPT 微调        : python _tools/train_gpt.py --base both"
echo "  云端合成        : python _tools/cloud_synth_sweep.py --epochs 8,16,24,32"
echo "============================================================"
