# -*- coding: utf-8 -*-
"""生成修正后的最小依赖脚本（**不再碰 transformers**）。

## 事故复盘（三次连环）

  1. 我按 requirements.txt 照搬装包 -> torchmetrics 把 numpy 降到 1.x
     -> ABI 错误（Expected 96 got 88）
  2. 修好 numpy 后装 gradio 未钉版本 -> huggingface-hub 被拉到 2.2.0
     -> transformers 4.57 报 ImportError
  3. 想降 hf-hub 到 <1.0 -> 与云上预装的 **vllm 0.28** 冲突
     （vllm 要求 transformers>=5.5.3 且 huggingface_hub>=1.27.0）

## 根本认知错误

我一直假设"训练需要 transformers"，所以反复在 transformers 版本上折腾。
但用 AST 递归追踪 `s2_train_v5_full.py` 的全部本地 import 后确认：

    transformers      不需要   （BERT 特征已在本机算好并随包上传）
    huggingface_hub   不需要
    peft              不需要   （全量微调不用）
    pytorch_lightning 不需要   （只有 s1_train 用）
    torchmetrics      不需要

    真正需要的第三方包：
      torch, numpy, scipy, librosa, soundfile, ffmpeg-python,
      x_transformers, rotary_embedding_torch, einops, safetensors,
      matplotlib, pandas, gradio, tensorboard, tqdm, PyYAML, psutil,
      cn2an, pypinyin, jieba, chardet, opencc, packaging

  而 gradio / pandas 只是被 `tools/my_utils.py` **模块级 import**，
  实际只用到 `gr.Warning` 与 `pd.read_csv`（稳定 API），
  云上现有的 gradio 6.30 / pandas 3.0.5 **可以直接用**。

## 本版策略

**完全不碰 transformers / huggingface-hub / numpy / torch**（云上现状即可用），
只补真正缺的包。这样与 vllm 的约束零冲突。

产出：D:\\A\\zfy-v5\\install_deps_v3.sh
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(r"D:\A\zfy-v5\install_deps_v3.sh")

# 用列表拼接，避免嵌套三引号（上次的 bug）
LINES = [
    "#!/usr/bin/env bash",
    "# ============================================================",
    "# GPT-SoVITS v5 全量微调 —— 最小依赖（v3：不碰 transformers）",
    "#",
    "# 关键认知：s2 训练**不需要** transformers（BERT 特征已算好）。",
    "# 所以不降 transformers / huggingface-hub / numpy / torch，",
    "# 与云上预装的 vllm 零冲突。",
    "#",
    "# 用法：",
    "#   cd /mnt/workspace/zfy-v5-fullft-pkg",
    "#   bash _tools/install_deps_v3.sh",
    "# ============================================================",
    "set -e",
    "",
    'ROOT=/mnt/workspace/zfy-v5-fullft-pkg',
    'cd "$ROOT"',
    "",
    'echo "=== 0) 现状（这些都不动）==="',
    "python - <<'PYEOF'",
    "for m in ('torch','numpy','transformers','huggingface_hub','gradio','pandas'):",
    "    try:",
    "        mod = __import__(m)",
    "        print(f'  {m:18s} {getattr(mod, \"__version__\", \"?\")}')",
    "    except Exception as e:",
    "        print(f'  {m:18s} 导入失败: {type(e).__name__}')",
    "PYEOF",
    "",
    'echo ""',
    'echo "=== 1) 补装真正缺失的依赖 ==="',
    "# 只装训练实际用到的（AST 追踪 s2_train_v5_full.py 得出）",
    "pip install --quiet \\",
    "    librosa==0.10.2 \\",
    "    soundfile \\",
    "    ffmpeg-python \\",
    "    x_transformers \\",
    "    rotary_embedding_torch \\",
    "    einops \\",
    "    safetensors \\",
    "    tensorboard \\",
    "    tqdm \\",
    "    PyYAML \\",
    "    psutil \\",
    "    packaging \\",
    "    cn2an \\",
    "    pypinyin \\",
    "    jieba \\",
    "    jieba_fast \\",
    "    chardet \\",
    "    opencc",
    "",
    "# 注意：以下**故意不装**",
    "#   transformers / huggingface_hub  训练不需要（BERT 特征已算好）",
    "#   peft                            全量微调不用",
    "#   pytorch-lightning / torchmetrics 只有 s1_train 用（不训 GPT）",
    "#   onnxruntime-gpu                 ROCm 上无效",
    "#   gradio / pandas / matplotlib    云上已有，且只被模块级 import",
    "",
    'echo ""',
    'echo "=== 2) 检查 gradio / pandas 是否可 import（模块级依赖）==="',
    "python - <<'PYEOF'",
    "import subprocess, sys",
    "code = (",
    "    'import gradio, pandas, matplotlib\\n'",
    "    'print(f\"  gradio     {gradio.__version__}\")\\n'",
    "    'print(f\"  pandas     {pandas.__version__}\")\\n'",
    "    'print(f\"  matplotlib {matplotlib.__version__}\")\\n'",
    "    'print(\"  OK: 三者均可 import\")\\n'",
    ")",
    "r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)",
    "print(r.stdout or r.stderr[-1500:])",
    "PYEOF",
    "",
    'echo ""',
    'echo "=== 3) 自检：能否 import v5 模型 ==="',
    'PYTHONPATH="$ROOT" python - <<\'PYEOF\'',
    "import os, sys, json",
    'root = os.environ.get("PYTHONPATH", "/mnt/workspace/zfy-v5-fullft-pkg")',
    "for p in reversed([root, root + '/GPT_SoVITS', root + '/GPT_SoVITS/BigVGAN',",
    "                   root + '/tools', root + '/tools/asr', root + '/tools/uvr5']):",
    "    if p in sys.path:",
    "        sys.path.remove(p)",
    "    sys.path.insert(0, p)",
    "os.chdir(root)",
    "",
    "import inspect",
    "from f5_tts.model import DiT",
    "ok = 'use_step_embedding' in inspect.signature(DiT.__init__).parameters",
    "print(f'  f5_tts.DiT 含 use_step_embedding: {ok}')",
    "",
    "from module.models import SynthesizerTrnV3",
    "from module.models_v5 import CFMV5",
    "cfg = json.load(open(root + '/GPT_SoVITS/configs/s2.json', encoding='utf-8'))",
    "kw = dict(cfg['model']); kw['version'] = 'v5dev'",
    "net = SynthesizerTrnV3(cfg['data']['filter_length'] // 2 + 1,",
    "                       cfg['train']['segment_size'] // cfg['data']['hop_length'],",
    "                       n_speakers=cfg['data']['n_speakers'], **kw)",
    "n = sum(p.numel() for p in net.parameters())",
    "print(f'  v5 模型构造成功: {n/1e6:.2f} M 参数  (CFMV5={isinstance(net.cfm, CFMV5)})')",
    "PYEOF",
    "",
    'echo ""',
    'echo "============================================================"',
    'echo "完成。若上面第 3 步通过，直接跑冒烟测试（notebook cell 8）。"',
    'echo "============================================================"',
]

OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8", newline="\n")
print(f"已生成: {OUT}")
print(f"  体积: {OUT.stat().st_size/1024:.1f} KB")
print(f"  行数: {len(LINES)}")

# 自检：脚本里不应有嵌套三引号问题
txt = OUT.read_text(encoding="utf-8")
print()
print("=== 生成脚本自检 ===")
print(f"  含 heredoc PYEOF: {txt.count('PYEOF')} 处（应为偶数）")
print(f"  奇数个 -> 会报 heredoc 错误" if txt.count("PYEOF") % 2 else "  ✓ heredoc 配对正确")
print(f"  含 'NameError' 风险字串: {'r\"\"\"' in txt}")
print()
print("云上用法：")
print("  cd /mnt/workspace/zfy-v5-fullft-pkg")
print("  bash _tools/install_deps_v3.sh")
