# -*- coding: utf-8 -*-
"""生成今晚可直接运行的最终 v5 full-FT Notebook。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\A\zfy-v5")
OUT = ROOT / "zfy_v5_fullft_final.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.strip().splitlines(True)}


cells = [
md(r"""
# 庄方宜第二代语音模型 · GPT-SoVITS v5 最终全量微调

本 Notebook 是**正式训练版**，不是实验版。

## 固定方案

- 底层：GPT-SoVITS **v5**
- 训练：SoVITS full-FT，32 epoch
- 解冻：`enc_p + ssl_proj`
- 冻结：`quantizer`（防 EMA 码本漂移）
- 保存：e8 / e16 / e24 / e32
- 导出：**完整 state_dict**，冻结的 quantizer 也会进入权重
- 上传：ModelScope `yangwenjie1231/zhuang-fangyi-voice-v2`

## 不允许出现

```text
unfrozen ['enc_p', 'quantizer', 'ssl_proj']   # 错：quantizer 被解冻
```

必须看到：

```text
unfrozen ['enc_p', 'ssl_proj']; still frozen ['quantizer']
```

按顺序运行。训练约 45–90 分钟，权重约 4×766MB。
"""),
md("## 1. 定位或克隆训练包"),
code(r'''
import os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path("/mnt/workspace/zfy-v5-fullft-pkg")
DATASET = "yangwenjie1231/zfy-v5-fullft-pkg"

# ---------- Git LFS：检测 -> 缺失则安装 -> 初始化 ----------
def ensure_git_lfs():
    probe = subprocess.run(["git", "lfs", "version"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    if probe.returncode != 0:
        print("未检测到 git-lfs，开始安装…")
        if shutil.which("apt-get"):
            # ModelScope ROCm Notebook 通常是 Ubuntu/root；不重装 git，只补 git-lfs。
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "git-lfs"], check=True)
        else:
            raise SystemExit(
                "系统缺少 git-lfs 且没有 apt-get。请先安装 git-lfs，再重跑本格。")
        probe = subprocess.run(["git", "lfs", "version"], capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
        if probe.returncode != 0:
            raise SystemExit("git-lfs 安装后仍不可用，停止 clone，避免只得到 LFS 指针")
    print("✓", probe.stdout.strip())
    subprocess.run(["git", "lfs", "install", "--skip-repo"], check=True)

ensure_git_lfs()

if not (ROOT / "GPT_SoVITS/configs/s2.json").exists():
    token_file = Path.home() / ".modelscope/credentials/git_token"
    if not token_file.exists():
        raise SystemExit("未找到 ModelScope git_token，请先登录")
    token = token_file.read_text().strip()
    url = f"https://oauth2:{token}@www.modelscope.cn/datasets/{DATASET}.git"
    subprocess.run(["git", "clone", url, str(ROOT)], check=True)

os.chdir(ROOT)
print("✓ ROOT =", ROOT)
print("  GPT_SoVITS:", (ROOT / "GPT_SoVITS").is_dir())
print("  features  :", (ROOT / "logs/zfy_v5").is_dir())
'''),
md("## 2. 拉取最终流水线脚本（兼容旧 clone）"),
code(r'''
import subprocess, sys
from pathlib import Path

os.chdir(ROOT)
# 最新训练包通常已有这些文件；旧 clone 则从 GitHub 拉取
RAW = "https://raw.githubusercontent.com/yangwenjie1231/zfy-v5/main/"
needed = [
    "final_v5_pipeline.py",
    "patch_full_export_all.py",
    "verify_full_export_all.py",
]
for name in needed:
    dst = ROOT / "_tools" / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-fL", RAW + name, "-o", str(dst)], check=True)
        print("下载", name)
    else:
        print("已有", name)
'''),
md("## 3. 安装 SoVITS 训练依赖（不重装 torch、不动 transformers）"),
code(r'''
import subprocess
os.chdir(ROOT)
subprocess.run(["bash", "_tools/install_deps_v3.sh"], check=True)
'''),
md("## 4. 体检"),
code(r'''
import subprocess, sys
os.chdir(ROOT)
subprocess.run([sys.executable, "_tools/final_v5_pipeline.py", "doctor"], check=True)
'''),
md("## 5. 应用补丁 + 端到端验证 + 生成最终配置"),
code(r'''
import subprocess, sys
os.chdir(ROOT)
# 首次运行不要 --force；只有确认要删除同名旧实验时才加
subprocess.run([sys.executable, "_tools/final_v5_pipeline.py", "prepare"], check=True)
'''),
md(r"""
### 5.1 必查输出

上格必须全部通过，并出现：

```text
推荐模式: quantizer 仍冻结
端到端验证码本未变 delta=0.00000000
ZFY_FULL_EXPORT_ALL_PATCH
```

否则不要启动训练。
"""),
md("## 6. 后台启动训练"),
code(r'''
import subprocess, sys
os.chdir(ROOT)
subprocess.run([sys.executable, "_tools/final_v5_pipeline.py", "launch"], check=True)
'''),
md("## 7. 查看进度（可反复运行）"),
code(r'''
import subprocess, sys
os.chdir(ROOT)
subprocess.run([sys.executable, "_tools/final_v5_pipeline.py", "status"])
'''),
md(r"""
## 8. 训练完成后的验收

等 status 显示 e8/e16/e24/e32 四个权重且进程退出后运行。
验收会：

- 补 v5 版本头；
- 与官方 v5 底模逐键比对；
- 确认 `enc_p` / `ssl_proj` / `quantizer` / `cfm` 全部存在；
- 写 SHA256 manifest。
"""),
code(r'''
import subprocess, sys
os.chdir(ROOT)
subprocess.run([sys.executable, "_tools/final_v5_pipeline.py", "validate"], check=True)
'''),
md("## 9. 合成 e8/e16/e24/e32 验收音频"),
code(r'''
import subprocess, sys
os.chdir(ROOT)
# 依赖修复补丁已经在 prepare 阶段应用
subprocess.run([
    sys.executable, "_tools/cloud_synth_sweep.py",
    "--weights-dir", "SoVITS_weights_v5_full_e32_final",
    "--epochs", "8,16,24,32",
    "--out", "out_sweep_final",
    "--no-v2pro",
    "--no-upload",
], check=True)
'''),
md("## 10. 自动检查时长 / 提前终止"),
code(r'''
import soundfile as sf, re
from pathlib import Path

D = ROOT / "out_sweep_final"
bad = []
rows = []
for w in sorted(D.rglob("*.wav")):
    x, sr = sf.read(w)
    dur = len(x) / sr
    # 文件名首两位对应测试文本，最低安全阈值 0.55s；长句额外人工听
    if dur < 0.55:
        bad.append((str(w.relative_to(D)), dur))
    rows.append((str(w.relative_to(D)), dur))
print("音频数:", len(rows))
print("时长范围:", min(d for _,d in rows), "~", max(d for _,d in rows))
print("过短:", bad)
assert not bad, "存在疑似提前终止音频"
print("✓ 无 <0.55s 的提前终止")
'''),
md("## 11. 上传正式模型到 ModelScope"),
code(r'''
import subprocess, sys
os.chdir(ROOT)
subprocess.run([sys.executable, "_tools/final_v5_pipeline.py", "upload"], check=True)
'''),
md(r"""
## 12. GitHub Release（回本机后）

ModelScope 上传完成后，在本机下载 e32 正式权重并执行：

```powershell
gh release create model-v2-v5-fullft `
  --repo yangwenjie1231/dsh-zhuang-fangyi-voice-v2 `
  --title "庄方宜第二代语音模型 · GPT-SoVITS v5 full-FT" `
  --notes "底层 GPT-SoVITS v5；enc_p+ssl_proj 微调；quantizer 冻结；完整自包含权重。"

gh release upload model-v2-v5-fullft `
  v5_full_e32_final_e32_s1728_l0.pth `
  final_v5_manifest.json `
  --repo yangwenjie1231/dsh-zhuang-fangyi-voice-v2
```

不要上传旧 `nofreeze` 权重作为正式模型。
"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT)
print("cells", len(cells))
