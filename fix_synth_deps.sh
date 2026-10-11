#!/usr/bin/env bash
# ============================================================
# 修复合成链路的依赖（合成前的最后一道）
#
# ## 为什么单独写这个
# 我们已经踩了 4 次依赖坑，每次都是不同的包被 pip 顺带卸掉：
#   1. gradio/oauth.py  需要 huggingface_hub.HfFolder    （hf-hub >=0.37 删了）
#   2. transformers 5.x 需要 huggingface_hub.is_offline_mode
#   3. 修 #2 降 transformers 到 4.x -> **fast_langdetect 被卸掉**
#      -> 合成报 ModuleNotFoundError: No module named 'fast_langdetect'
#   4. 之前还有 onnxruntime（chinese2.py 硬编码 is_g2pw=True）
#
# ## 根因
# `pip install X` 会**重新求解**整个依赖图，把"新版本不再需要"的包
# 顺手卸掉/降级。而 transformers 5 -> 4 是个大回退，牵连很广。
#
# ## 本脚本
# 按**依赖链**顺序装，每步都验证 import，最后跑一次端到端探针。
#
# 用法：
#   bash _tools/fix_synth_deps.sh
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT" || { echo "!! 找不到 $ROOT"; exit 1; }

echo "############################################################"
echo "# 修复合成链路依赖"
echo "# $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"

PIP="pip install --quiet"
FAIL=0

step() { echo; echo "--- $* ---"; }

# ---------- 1) 钉住版本矩阵 ----------
step "1) 版本矩阵（这三者必须同时满足）"
echo "  transformers 4.x     <- 不需要 is_offline_mode（5.x 才要）"
echo "  huggingface-hub 0.36 <- 提供 HfFolder（gradio 4.44 需要）"
echo "  gradio 4.x           <- WebUI（训练/合成其实用不到，my_utils 已改可选）"
$PIP "transformers>=4.51,<5" 2>&1 | grep -v "^WARNING: Running pip" | tail -3
$PIP "huggingface-hub==0.36.2" 2>&1 | grep -v "^WARNING: Running pip" | tail -3
$PIP "gradio<5" 2>&1 | grep -v "^WARNING: Running pip" | tail -3

# ---------- 2) 语言检测链（被 transformers 回退牵连）----------
step "2) 语言检测链（LangSegmenter）"
echo "  fast_langdetect <- text/LangSegmenter/langsegmenter.py:10"
echo "  split_lang      <- 同上（from split_lang import LangSplitter）"
$PIP "fast_langdetect>=0.3.1" 2>&1 | grep -v "^WARNING: Running pip" | tail -3 \
    || { echo "  !! fast_langdetect 安装失败"; FAIL=1; }
$PIP split-lang 2>&1 | grep -v "^WARNING: Running pip" | tail -3 \
    || { echo "  !! split-lang 安装失败"; FAIL=1; }

# ---------- 3) 文本/G2P 链 ----------
step "3) 文本前端链"
$PIP jieba jieba_fast soundfile wordsegment 2>&1 \
    | grep -v "^WARNING: Running pip" | tail -2
$PIP cn2an pypinyin chardet opencc 2>&1 \
    | grep -v "^WARNING: Running pip" | tail -2
# g2pw.onnx 已在仓库里，onnxruntime 缺失会让 chinese2.py 崩
$PIP onnxruntime 2>&1 | grep -v "^WARNING: Running pip" | tail -2

# ---------- 4) 逐项验证 import ----------
step "4) 逐项验证"
python - <<'PYEOF'
import importlib, sys
NEED = [
    ("transformers", "文本/BERT 特征"),
    ("huggingface_hub", "HF 加载器"),
    ("gradio", "WebUI（可选，my_utils 已改桩）"),
    ("fast_langdetect", "LangSegmenter 语言检测"),
    ("split_lang", "LangSegmenter 分段"),
    ("jieba", "中文分词"),
    ("pypinyin", "拼音"),
    ("cn2an", "数字转中文"),
    ("soundfile", "音频读写"),
    ("onnxruntime", "g2pw.onnx"),
    ("librosa", "音频特征"),
    ("torch", "推理后端"),
]
bad = []
for m, why in NEED:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:18s} {getattr(mod,'__version__','?'):12s} {why}")
    except Exception as e:
        print(f"  FAIL {m:18s} {'':12s} {why}  <- {type(e).__name__}")
        bad.append(m)

# 关键兼容性断言
import huggingface_hub as h
import transformers
print()
print(f"  transformers={transformers.__version__}  "
      f"hf-hub={h.__version__}")
if transformers.__version__.startswith("5"):
    print("  !! transformers 是 5.x —— 需要 is_offline_mode，会崩")
    bad.append("transformers>=5")
if not hasattr(h, "HfFolder"):
    print("  !! hf-hub 缺 HfFolder —— gradio 会崩")
    bad.append("HfFolder")
if hasattr(h, "is_offline_mode") and transformers.__version__.startswith("4"):
    print("  ~ hf-hub 有 is_offline_mode（无妨，4.x 不用它）")

sys.exit(1 if bad else 0)
PYEOF
[ $? -ne 0 ] && FAIL=1

# ---------- 5) 端到端探针：真的构造一次 TTS ----------
step "5) 端到端探针（构造 TTS + 合成一句）"
PYTHONPATH="$ROOT" python - <<'PYEOF'
import os, sys
ROOT = os.environ.get("ZFY_ROOT", "/mnt/workspace/zfy-v5-fullft-pkg")

# ⚠️ 两种仓库布局都要认（云端是扁平、本地是嵌套）：
#     扁平: ROOT/GPT_SoVITS/utils.py           <- 云端
#     嵌套: ROOT/GPT-SoVITS/GPT_SoVITS/utils.py <- 本地
#   我第一版写死 `ROOT + "GPT-SoVITS"` -> 云端 FileNotFoundError。
def find_gsv(root: str) -> str:
    for cand in (os.path.join(root, "GPT-SoVITS"), root):
        if os.path.exists(os.path.join(cand, "GPT_SoVITS", "utils.py")):
            return cand
    return os.path.join(root, "GPT-SoVITS")

GSV = find_gsv(ROOT)
print(f"  ROOT={ROOT}")
print(f"  GSV ={GSV}")
if not os.path.exists(os.path.join(GSV, "GPT_SoVITS", "utils.py")):
    print("  !! 找不到 GPT_SoVITS/utils.py —— 检查 ROOT")
    sys.exit(1)

sys.path[:0] = [GSV, os.path.join(GSV, "GPT_SoVITS"), ROOT]
os.chdir(GSV)

import glob
# 权重目录在 ROOT 下（不在 GSV 下）
cands = glob.glob(os.path.join(
    ROOT, "SoVITS_weights_v5_full_e32_nofreeze", "*_e32_s1728_l0.pth"))
if not cands:
    cands = glob.glob(os.path.join(
        GSV, "SoVITS_weights_v5_full_e32_nofreeze", "*_e32_s1728_l0.pth"))
if not cands:
    print("  !! 找不到 e32 权重")
    for d in (ROOT, GSV):
        p = os.path.join(d, "SoVITS_weights_v5_full_e32_nofreeze")
        print(f"     试过 {p}: {os.path.isdir(p)}")
    sys.exit(1)
W = cands[0]
print(f"  权重: {os.path.basename(W)}")

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
PRE = os.path.join(GSV, "GPT_SoVITS", "pretrained_models")


def pick(*rels):
    for r in rels:
        p = os.path.join(GSV, r) if not r.startswith("/") else r
        if os.path.exists(p):
            return p
    return os.path.join(GSV, rels[0])


GPT_W = pick("ref/zfh-e4.ckpt")
REF = pick("ref/default.wav")
REF_TXT_F = pick("ref/ref.txt")
BERT = pick("GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
HUBERT = pick("GPT_SoVITS/pretrained_models/chinese-hubert-base")
for tag, p in (("GPT 权重", GPT_W), ("参考音频", REF), ("BERT", BERT),
               ("HuBERT", HUBERT)):
    print(f"  {tag:8s}: {'✓' if os.path.exists(p) else '✗'} {p}")

cfg = {"custom": {
    "device": "cuda", "is_half": False, "version": "v5dev",
    "t2s_weights_path": GPT_W,
    "vits_weights_path": W,
    "bert_base_path": BERT,
    "cnhuhbert_base_path": HUBERT,
}}
try:
    tts = TTS(TTS_Config(cfg))
    print("  ✓ TTS 构造成功")
except Exception as e:
    print(f"  !! TTS 构造失败: {type(e).__name__}: {e}")
    sys.exit(1)

rtxt = open(REF_TXT_F, encoding="utf-8").read().strip()
try:
    import numpy as np
    chunks, sr = [], None
    for item in tts.run({
        "text": "让我想想。", "text_lang": "zh",
        "ref_audio_path": REF, "prompt_text": rtxt, "prompt_lang": "zh",
        "top_k": 15, "temperature": 1.0, "repetition_penalty": 1.35,
        "text_split_method": "cut1", "batch_size": 1,
        "speed_factor": 1.0, "seed": 42, "parallel_infer": True,
    }):
        if item is None:
            continue
        sr, a = item
        chunks.append(a)
    if not chunks:
        print("  !! 无产出")
        sys.exit(1)
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    dur = len(audio) / sr
    rms = float(np.sqrt((np.asarray(audio, dtype=float) ** 2).mean()))
    print(f"  ✓ 合成成功: {dur:.2f}s @{sr}Hz  rms={rms:.5f}")
    if dur > 5.0:
        print(f"  !! 时长 {dur:.2f}s 过长（正常 1.0~1.5s）"
              f" -> ssl_proj 可能有问题")
        sys.exit(1)
    print("     ✓ 时长正常，ssl_proj/enc_p 均在位")
except Exception as e:
    print(f"  !! 合成失败: {type(e).__name__}: {e}")
    sys.exit(1)
PYEOF
[ $? -ne 0 ] && FAIL=1

echo
echo "############################################################"
if [ "$FAIL" = "0" ]; then
    echo "# ✓ 依赖全部就绪，合成探针通过"
    echo "#"
    echo "# 下一步："
    echo "#   python _tools/cloud_synth_sweep.py \\"
    echo "#       --weights-dir SoVITS_weights_v5_full_e32_nofreeze \\"
    echo "#       --epochs 8,16,24,32 --no-v2pro --out out_sweep_nofreeze"
else
    echo "# ✗ 有项目失败，见上面的 !! 行"
fi
echo "############################################################"
exit $FAIL
