#!/usr/bin/env bash
# ============================================================
# 依赖修复补丁（在 install_all_deps.sh 之后跑，或单独跑）
#
# ## 为什么需要"补丁"而不是改主脚本
#
# 云上实测：即使主脚本里 `pip install "huggingface-hub==0.36.2"`，
# 后面的包仍会把它顶回去，导致：
#     ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
# 来源：gradio/oauth.py:13  from huggingface_hub import HfFolder, whoami
# （`import gradio` 就会加载 gradio.oauth —— 实测 sys.modules 里确实有）
#
# 所以本脚本做的是：**装完 → 验证 → 不对就强制重装 → 再验证**，
# 把"钉版本"从"祈祷"变成"确认"。
#
# ## 同时补上追踪器 v2 新发现的推理依赖
# 我原先的追踪器只处理绝对导入（level==0），漏掉了相对导入链：
#     text/LangSegmenter/__init__.py  from .langsegmenter import LangSegmenter
#       -> langsegmenter.py  import fast_langdetect / split_lang
# 于是云端报 ModuleNotFoundError: No module named 'fast_langdetect'
#
# 用法：
#   cd /mnt/workspace/zfy-v5-fullft-pkg
#   bash _tools/fix_deps_patch.sh
# ============================================================
set -u

ROOT=/mnt/workspace/zfy-v5-fullft-pkg
cd "$ROOT" 2>/dev/null || true

echo "============================================================"
echo "依赖修复补丁"
echo "============================================================"

echo
echo "--- 1) 推理依赖：追踪器 v2 新发现的包 ---"
echo "  fast_langdetect  <- text/LangSegmenter/langsegmenter.py:10（相对导入链）"
echo "  split_lang       <- 同上"
echo "  jieba            <- text 前端"
echo "  soundfile        <- 音频读写"
pip install --quiet "fast_langdetect>=0.3.1" || echo "  fast_langdetect 安装失败"
pip install --quiet split-lang || echo "  split-lang 安装失败"
pip install --quiet jieba jieba_fast soundfile wordsegment || true

echo
echo "--- 1b) transformers 必须 <5（与 gradio 的硬冲突）---"
cat <<'NOTE'
  实测冲突链：
    · transformers 5.x 需要 huggingface_hub >= 1.x
        -> 报 ImportError: cannot import name 'is_offline_mode'
    · gradio 4.44 需要 huggingface_hub < 0.37（它 import HfFolder）
        -> 报 ImportError: cannot import name 'HfFolder'
    **两者互斥**，不能同时满足。

  本机验证可用的组合是：
    transformers 4.57.6 + huggingface_hub 0.36.2 + gradio 4.44.1
  所以把 transformers 钉到 <5。

  注：SoVITS 训练本身不需要 transformers，但**推理链路需要**
  （TTS_infer_pack/TTS.py:37 from transformers import AutoModelForMaskedLM）
NOTE
pip install --quiet "transformers>=4.51,<5" || echo "  transformers 安装失败"
python - <<'PYEOF'
import transformers, huggingface_hub as h, gradio
print(f"  transformers {transformers.__version__}")
print(f"  huggingface_hub {h.__version__}  HfFolder: {hasattr(h,'HfFolder')}")
print(f"  gradio {gradio.__version__}")
PYEOF

echo
echo "--- 2) huggingface_hub：装完 -> 验证 -> 不对就强制重装 ---"
HF_WANT=0.36.2

check_hf() {
    python - <<'PYEOF'
import sys
try:
    import huggingface_hub as h
except Exception as e:
    print(f"  huggingface_hub 导入失败: {e}")
    sys.exit(2)
ok = hasattr(h, "HfFolder")
print(f"  huggingface_hub {h.__version__}  HfFolder: {ok}")
sys.exit(0 if ok else 1)
PYEOF
}

pip install --quiet "huggingface-hub==${HF_WANT}" 2>/dev/null || true
if check_hf; then
    echo "  OK"
else
    echo "  !! 版本不对或缺 HfFolder，强制重装 ${HF_WANT}"
    pip install --quiet --force-reinstall --no-deps "huggingface-hub==${HF_WANT}" \
        || pip install --quiet --force-reinstall "huggingface-hub==${HF_WANT}" \
        || echo "  强制重装失败"
    if check_hf; then
        echo "  OK（强制重装后）"
    else
        echo "  !! 仍失败 —— 改用**绕过方案**：给 gradio 打补丁，见第 3 步"
    fi
fi

echo
echo "--- 3) 兜底：若 HfFolder 仍缺失，给 gradio/oauth.py 打补丁 ---"
python - <<'PYEOF'
import sys, pathlib
try:
    import huggingface_hub as h
    if hasattr(h, "HfFolder"):
        print("  HfFolder 存在，无需补丁")
        raise SystemExit(0)
    import gradio
    p = pathlib.Path(gradio.__file__).parent / "oauth.py"
    if not p.exists():
        print(f"  找不到 {p}")
        raise SystemExit(0)
    src = p.read_text(encoding="utf-8")
    if "ZFY_HFFOLDER_SHIM" in src:
        print("  已打过补丁")
        raise SystemExit(0)
    old = "from huggingface_hub import HfFolder, whoami"
    if old not in src:
        print("  未找到目标行，跳过")
        raise SystemExit(0)
    shim = (
        "# ZFY_HFFOLDER_SHIM: huggingface_hub>=0.36 移除了 HfFolder，\n"
        "# 而 gradio 仍在 import。这里补一个最小替身（仅本地 token 存取）。\n"
        "try:\n"
        "    from huggingface_hub import HfFolder, whoami\n"
        "except ImportError:\n"
        "    from huggingface_hub import whoami\n"
        "    class HfFolder:  # type: ignore\n"
        "        @staticmethod\n"
        "        def get_token():\n"
        "            import os\n"
        "            return os.environ.get('HF_TOKEN')\n"
        "        @staticmethod\n"
        "        def save_token(t):\n"
        "            import os\n"
        "            os.environ['HF_TOKEN'] = t or ''\n"
        "        @staticmethod\n"
        "        def delete_token():\n"
        "            import os\n"
        "            os.environ.pop('HF_TOKEN', None)\n"
    )
    p.write_text(src.replace(old, shim, 1), encoding="utf-8")
    print(f"  已给 {p} 打补丁")
except SystemExit:
    raise
except Exception as e:
    print(f"  补丁失败: {type(e).__name__}: {e}")
PYEOF

echo
echo "============================================================"
echo "4) 复核（真实 import 三条链路）"
echo "============================================================"
PYTHONPATH="$ROOT" python - <<'PYEOF'
import os, sys, subprocess
root = os.environ.get("PYTHONPATH", "/mnt/workspace/zfy-v5-fullft-pkg")
code = r'''
import os, sys
root = os.environ.get("PYTHONPATH", "/mnt/workspace/zfy-v5-fullft-pkg")
for p in reversed([root, root + "/GPT_SoVITS", root + "/GPT_SoVITS/BigVGAN",
                   root + "/tools", root + "/tools/asr", root + "/tools/uvr5"]):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.chdir(root)

fails = []
for label, stmt in (
    ("GPT 训练链路",
     "from AR.models.t2s_lightning_module import Text2SemanticLightningModule;"
     "from AR.data.data_module import Text2SemanticDataModule;"
     "from AR.modules.lr_schedulers import WarmupCosineLRSchedule"),
    ("SoVITS 训练链路", "from module.data_utils import TextAudioSpeakerLoaderV4"),
    ("推理链路", "from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config"),
):
    try:
        exec(stmt)
        print(f"  OK   {label}")
    except Exception as e:
        print(f"  FAIL {label}: {type(e).__name__}: {str(e)[:130]}")
        fails.append(label)

import numpy, torch
print(f"  numpy {numpy.__version__}  torch {torch.__version__}")
if fails:
    raise SystemExit("缺: " + ",".join(fails))
print("  三条链路全部可 import")
'''
r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                   text=True, env={**os.environ, "PYTHONPATH": root})
print(r.stdout)
if r.returncode != 0:
    print(r.stderr[-1500:])
    sys.exit(1)
PYEOF

echo
echo "============================================================"
echo "完成。若三条链路 OK，即可开始训练/合成。"
echo "============================================================"
